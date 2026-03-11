//! Commitment scheme trait implementation.

use crate::algebra::fields::wide::HasWide;
use crate::algebra::fields::HasUnreducedOps;
use crate::algebra::CyclotomicRing;
#[cfg(debug_assertions)]
use crate::algebra::SparseChallenge;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::primitives::serialization::Valid;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::linear::{flatten_i8_blocks, mat_vec_mul_ntt_single_i8};
use crate::protocol::commitment::utils::ntt_cache::MultiDNttBundle;
use crate::protocol::commitment::{
    AppendToTranscript, CommitmentConfig, CommitmentScheme, HachiCommitmentCore,
    HachiCommitmentLayout, HachiExpandedSetup, HachiProverSetup, HachiVerifierSetup,
    RingCommitment, RingCommitmentScheme,
};
use crate::protocol::hachi_poly_ops::{BalancedDigitPoly, HachiPolyOps};
use crate::protocol::opening_point::{BasisMode, RingOpeningPoint};
use crate::protocol::proof::{
    FlatCommitmentHint, FlatRingVec, HachiCommitmentHint, HachiLevelProof, HachiProof, PackedDigits,
};
#[cfg(any(test, debug_assertions))]
use crate::protocol::quadratic_equation::compute_m_a_reference;
use crate::protocol::quadratic_equation::QuadraticEquation;
use crate::protocol::ring_switch::eval_ring_at;
#[cfg(debug_assertions)]
use crate::protocol::ring_switch::m_row_count;
#[cfg(test)]
use crate::protocol::ring_switch::{build_alpha_evals_y, compute_m_evals_x};
use crate::protocol::ring_switch::{
    build_w_evals, commit_w, ring_switch_build_w, ring_switch_finalize, ring_switch_verifier,
    w_ring_element_count, RingSwitchOutput, WCommitmentConfig,
};
use crate::protocol::sumcheck::eq_poly::EqPolynomial;
use crate::protocol::sumcheck::hachi_sumcheck::{HachiSumcheckProver, HachiSumcheckVerifier};
#[cfg(debug_assertions)]
use crate::protocol::sumcheck::{multilinear_eval, range_check_eval};
use crate::protocol::sumcheck::{multilinear_eval_small, prove_sumcheck, verify_sumcheck};
use crate::protocol::transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use crate::protocol::transcript::Transcript;
use crate::{dispatch_ring_dim, dispatch_with_ntt};
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt};
#[cfg(debug_assertions)]
use std::iter;
use std::marker::PhantomData;
use std::time::Instant;

#[cfg(test)]
use crate::protocol::ring_switch::expand_m_a;
#[cfg(test)]
use crate::protocol::transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, DOMAIN_HACHI_PROTOCOL,
};
#[cfg(test)]
use crate::protocol::transcript::Blake2bTranscript;
#[cfg(test)]
use crate::protocol::SmallTestCommitmentConfig;

/// Minimum w vector length (in field elements) below which further folding
/// is not beneficial.  When `w.len() <= MIN_W_LEN_FOR_FOLDING`, the prover
/// sends `w` directly instead of recursing.
const MIN_W_LEN_FOR_FOLDING: usize = 4096;

/// Minimum shrink ratio (next_w / prev_w) below which further folding
/// stops being worthwhile.  If the w vector doesn't shrink by at least
/// this factor, the overhead of another fold level outweighs the saving.
const MIN_SHRINK_RATIO: f64 = 0.5;

/// Default witness length (in i8 digits) above which the prover hands off
/// to Greyhound/Labrador (D'=64) instead of sending the witness directly.
/// Individual configs can override via [`CommitmentConfig::greyhound_handoff_threshold`].
#[allow(dead_code)]
const DEFAULT_GREYHOUND_HANDOFF_THRESHOLD: usize = 65_536;

/// Ring dimension used for the Greyhound/Labrador tail.
const GREYHOUND_D: usize = 64;

#[inline]
fn relation_claim_from_rows<F: FieldCore + CanonicalField, const D: usize>(
    tau1: &[F],
    alpha: F,
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
    y_ring: &CyclotomicRing<F, D>,
) -> F {
    let eq_tau1 = EqPolynomial::evals(tau1);
    let mut acc = F::zero();
    let mut row_idx = 0usize;

    for r in v {
        if row_idx >= eq_tau1.len() {
            return acc;
        }
        acc += eq_tau1[row_idx] * eval_ring_at(r, &alpha);
        row_idx += 1;
    }
    for r in u {
        if row_idx >= eq_tau1.len() {
            return acc;
        }
        acc += eq_tau1[row_idx] * eval_ring_at(r, &alpha);
        row_idx += 1;
    }
    if row_idx < eq_tau1.len() {
        acc += eq_tau1[row_idx] * eval_ring_at(y_ring, &alpha);
    }
    acc
}

/// End-to-end PCS wrapper, generic over ring degree `D` and config `Cfg`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HachiCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

/// Output from a single prove level, needed to chain into the next level.
///
/// D-agnostic: ring elements are erased into [`HachiLevelProof`] and
/// the commitment hint is stored as [`FlatCommitmentHint`].
struct ProveLevelOutput<F: FieldCore> {
    level_proof: HachiLevelProof<F>,
    w: Vec<i8>,
    w_hint: FlatCommitmentHint,
    sumcheck_challenges: Vec<F>,
    num_u: usize,
    num_l: usize,
}

/// Prove one fold level: quad_eq -> ring_switch -> sumcheck.
///
/// Generic over the commitment config so it works for both the original
/// polynomial (using `Cfg`) and recursive w-openings (using `WCommitmentConfig`).
type CommitFn<'a, F> =
    Box<dyn FnOnce(&[i8]) -> Result<(FlatRingVec<F>, FlatCommitmentHint), HachiError> + 'a>;

#[cfg(debug_assertions)]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_level_diagnostic<F, const D: usize, Cfg>(
    expanded: &HachiExpandedSetup<F>,
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    rs: &RingSwitchOutput<F>,
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
    y_ring: &CyclotomicRing<F, D>,
    layout: HachiCommitmentLayout,
    level: usize,
) where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    let m_a =
        compute_m_a_reference::<F, D, Cfg>(expanded, opening_point, challenges, &rs.alpha, layout)
            .expect("compute_m_a diagnostic failed");

    let x_len = 1usize << rs.num_u;
    let d = D;

    let mut w_at_alpha = vec![F::zero(); x_len];
    for (x, w_at_alpha_x) in w_at_alpha.iter_mut().enumerate() {
        let mut val = F::zero();
        for y in 0..d {
            let idx = x + y * x_len;
            if idx < rs.w_evals.len() {
                val += rs.alpha_evals_y[y] * F::from_i64(rs.w_evals[idx] as i64);
            }
        }
        *w_at_alpha_x = val;
    }

    let num_rows = m_row_count::<Cfg>();
    let y_full: Vec<F> = v
        .iter()
        .chain(u.iter())
        .chain(iter::once(y_ring))
        .map(|r| eval_ring_at(r, &rs.alpha))
        .collect();

    eprintln!(
        "  [hachi prove L{level}] per-row M*w=y diagnostic (num_rows={num_rows}, x_len={x_len}, m_a_cols={}):",
        m_a.first().map_or(0, |r| r.len()),
    );
    for i in 0..num_rows {
        let mw_i: F = m_a[i]
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (x, &m_ix)| {
                acc + m_ix * w_at_alpha.get(x).copied().unwrap_or(F::zero())
            });
        let y_i = if i < y_full.len() {
            y_full[i]
        } else {
            F::zero()
        };
        let residual = mw_i - y_i;
        let row_name = match i {
            _ if i < Cfg::N_D => "D",
            _ if i < Cfg::N_D + Cfg::N_B => "B",
            _ if i == Cfg::N_D + Cfg::N_B => "bTw",
            _ if i == Cfg::N_D + Cfg::N_B + 1 => "challenge_fold",
            _ => "A",
        };
        eprintln!(
            "    row {i} ({row_name}): match={}, residual_is_zero={}, mw_is_zero={}, y_is_zero={}",
            residual.is_zero(),
            residual.is_zero(),
            mw_i.is_zero(),
            y_i.is_zero(),
        );
    }

    let verifier_claim = relation_claim_from_rows::<F, D>(&rs.tau1, rs.alpha, v, u, y_ring);
    let x_mask = x_len - 1;
    let mut prover_claim = F::zero();
    for (idx, &w) in rs.w_evals.iter().enumerate() {
        prover_claim +=
            F::from_i64(w as i64) * rs.alpha_evals_y[idx >> rs.num_u] * rs.m_evals_x[idx & x_mask];
    }
    eprintln!(
        "  [hachi prove L{level}] relation_claim cross-check: match={}, prover_is_zero={}, verifier_is_zero={}",
        verifier_claim == prover_claim,
        prover_claim.is_zero(),
        verifier_claim.is_zero(),
    );
}

#[cfg(debug_assertions)]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_level_selfcheck<F: FieldCore + FromSmallInt>(
    tau0: &[F],
    sumcheck_challenges: &[F],
    w_eval: F,
    b: usize,
    batching_coeff: F,
    alpha_evals_y: &[F],
    m_evals_x: &[F],
    num_u: usize,
    final_claim: F,
    level: usize,
) {
    let eq_val = EqPolynomial::mle(tau0, sumcheck_challenges);
    let norm_oracle = eq_val * range_check_eval(w_eval, b);
    let (x_ch, y_ch) = sumcheck_challenges.split_at(num_u);
    let alpha_val = multilinear_eval(alpha_evals_y, y_ch).unwrap();
    let m_val = multilinear_eval(m_evals_x, x_ch).unwrap();
    let relation_oracle = w_eval * alpha_val * m_val;
    let prover_expected = batching_coeff * norm_oracle + relation_oracle;
    if prover_expected != final_claim {
        eprintln!("  [hachi prove L{level}] PROVER self-check FAILED: expected != final_claim");
    } else {
        eprintln!("  [hachi prove L{level}] PROVER self-check OK");
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_one_level<F, T, const D: usize, Cfg, P>(
    expanded: &HachiExpandedSetup<F>,
    ntt_a: &NttSlotCache<D>,
    ntt_b: &NttSlotCache<D>,
    ntt_d: &NttSlotCache<D>,
    commit_w_fn: CommitFn<'_, F>,
    poly: &P,
    opening_point: &[F],
    hint: HachiCommitmentHint<F, D>,
    transcript: &mut T,
    commitment: &RingCommitment<F, D>,
    basis: BasisMode,
    level: usize,
    layout: HachiCommitmentLayout,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
    P: HachiPolyOps<F, D>,
{
    {
        let x: u8 = 0;
        eprintln!(
            "  [prove_one_level L{level}] stack ~= {:#x}",
            &x as *const u8 as usize
        );
    }
    let alpha = Cfg::D.trailing_zeros() as usize;
    if opening_point.len() < alpha {
        return Err(HachiError::InvalidPointDimension {
            expected: alpha,
            actual: opening_point.len(),
        });
    }
    let target_num_vars = layout.m_vars + layout.r_vars + alpha;
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let outer_point = &padded_point[alpha..];

    let ring_opening_point = {
        let _span = tracing::info_span!("ring_opening_point", level).entered();
        ring_opening_point_from_field::<F>(outer_point, layout.r_vars, layout.m_vars, basis)?
    };

    let t0 = Instant::now();
    let fold_scalars = &ring_opening_point.a;
    let eval_outer_scalars = &ring_opening_point.b;
    let (y_ring, w_folded) = {
        let _span = tracing::info_span!("evaluate_and_fold", level).entered();
        poly.evaluate_and_fold(eval_outer_scalars, fold_scalars, layout.block_len)
    };
    eprintln!(
        "  [hachi prove L{level}] evaluate_and_fold: {:.2}s (num_ring_elems={})",
        t0.elapsed().as_secs_f64(),
        poly.num_ring_elems()
    );

    commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
    for pt in &padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

    let t1 = Instant::now();
    let mut quad_eq = Box::new(QuadraticEquation::<F, { D }, Cfg>::new_prover(
        ntt_d,
        ring_opening_point,
        poly,
        w_folded,
        hint,
        transcript,
        commitment,
        &y_ring,
        layout,
    )?);
    eprintln!(
        "  [hachi prove L{level}] quad_eq new_prover: {:.2}s",
        t1.elapsed().as_secs_f64()
    );

    let t2 = Instant::now();
    let w =
        ring_switch_build_w::<F, { D }, Cfg>(&mut quad_eq, expanded, ntt_a, ntt_b, ntt_d, layout)?;
    eprintln!(
        "  [hachi prove L{level}] ring_switch_build_w: {:.2}s (w.len()={})",
        t2.elapsed().as_secs_f64(),
        w.len()
    );

    let t_cw = Instant::now();
    let (w_commitment_flat, w_hint_flat) = commit_w_fn(&w)?;
    eprintln!(
        "  [hachi prove L{level}] commit_w: {:.2}s (ring_dim={})",
        t_cw.elapsed().as_secs_f64(),
        w_commitment_flat.ring_dim()
    );

    let rs = ring_switch_finalize::<F, T, { D }, Cfg>(
        &quad_eq,
        expanded,
        transcript,
        w,
        w_commitment_flat,
        w_hint_flat,
        layout,
    )?;
    eprintln!(
        "  [hachi prove L{level}] ring_switch_finalize: {:.2}s (num_u={}, num_l={})",
        t2.elapsed().as_secs_f64(),
        rs.num_u,
        rs.num_l
    );

    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);

    #[cfg(debug_assertions)]
    prove_level_diagnostic::<F, D, Cfg>(
        expanded,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        &rs,
        &quad_eq.v,
        &commitment.u,
        &y_ring,
        layout,
        level,
    );

    let t3 = Instant::now();
    let relation_claim =
        relation_claim_from_rows::<F, D>(&rs.tau1, rs.alpha, &quad_eq.v, &commitment.u, &y_ring);
    let RingSwitchOutput {
        w,
        w_commitment,
        w_hint,
        w_evals,
        w_evals_field: _,
        m_evals_x,
        alpha_evals_y,
        num_u,
        num_l,
        tau0,
        tau1: _,
        b,
        alpha: _,
    } = rs;
    let w_evals_small = w_evals.clone();
    #[cfg(debug_assertions)]
    let alpha_evals_y_debug = alpha_evals_y.clone();
    #[cfg(debug_assertions)]
    let m_evals_x_debug = m_evals_x.clone();
    let mut fused_prover = HachiSumcheckProver::new(
        batching_coeff,
        w_evals,
        &tau0,
        b,
        alpha_evals_y,
        m_evals_x,
        num_u,
        num_l,
        relation_claim,
    );

    let (sumcheck_proof, sumcheck_challenges, _final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut fused_prover, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
        })?;
    eprintln!(
        "  [hachi prove L{level}] fused sumcheck: {:.2}s",
        t3.elapsed().as_secs_f64()
    );

    let w_eval = {
        let _span = tracing::info_span!("multilinear_eval", level).entered();
        multilinear_eval_small(&w_evals_small, &sumcheck_challenges)?
    };

    #[cfg(debug_assertions)]
    prove_level_selfcheck(
        &tau0,
        &sumcheck_challenges,
        w_eval,
        b,
        batching_coeff,
        &alpha_evals_y_debug,
        &m_evals_x_debug,
        num_u,
        _final_claim,
        level,
    );

    Ok(ProveLevelOutput {
        level_proof: HachiLevelProof::new::<D>(
            y_ring,
            quad_eq.v,
            sumcheck_proof,
            w_commitment,
            w_eval,
        ),
        w,
        w_hint,
        sumcheck_challenges,
        num_u,
        num_l,
    })
}

/// Whether the prover should stop folding and send `w` directly.
///
/// `prev_w_len` is the polynomial length at the previous level (or the
/// original polynomial's field-element count for level 0).
fn should_stop_folding(w_len: usize, prev_w_len: usize) -> bool {
    if w_len <= MIN_W_LEN_FOR_FOLDING {
        return true;
    }
    let ratio = w_len as f64 / prev_w_len as f64;
    ratio > MIN_SHRINK_RATIO
}

/// Derive the opening point for the next fold level from the sumcheck
/// challenges of the current level.
///
/// Sumcheck challenges are ordered `[x_0..x_{num_u-1}, y_0..y_{num_l-1}]`
/// where x selects ring elements and y selects coefficients.
/// The PCS opening point is `[inner, outer]` = `[y, x]`.
pub(crate) fn next_level_opening_point<F: FieldCore>(
    sumcheck_challenges: &[F],
    num_u: usize,
    num_l: usize,
) -> Vec<F> {
    let (x, y) = sumcheck_challenges.split_at(num_u);
    debug_assert_eq!(y.len(), num_l);
    let mut point = Vec::with_capacity(num_u + num_l);
    point.extend_from_slice(y);
    point.extend_from_slice(x);
    point
}

/// Execute the Greyhound/Labrador handoff after Hachi folding stops.
///
/// Converts the i8-digit witness to field elements, runs Greyhound evaluation
/// reduction at D'=64, computes the correct Labrador statement with u1, and
/// runs Labrador recursive proving.
///
/// # Errors
///
/// Propagates errors from Greyhound evaluation, reduce, or Labrador proving.
#[allow(dead_code)]
fn greyhound_handoff_prove<F, T>(
    current_w: &[i8],
    current_challenges: &[F],
    current_num_u: usize,
    current_num_l: usize,
    _w_eval: F,
    expanded_setup: &HachiExpandedSetup<F>,
    transcript: &mut T,
) -> Result<crate::protocol::proof::HachiProofTail<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + Valid,
    T: Transcript<F>,
{
    use crate::algebra::ring::CyclotomicRing;
    use crate::primitives::poly::multilinear_lagrange_basis;
    use crate::protocol::greyhound::{greyhound_eval, greyhound_reduce};
    use crate::protocol::labrador::comkey::derive_extendable_comkey_matrix;
    use crate::protocol::labrador::prove_with_config;
    use crate::protocol::labrador::utils::mat_vec_mul;
    use crate::protocol::proof::{
        FlatGreyhoundEvalProof, FlatLabradorProof, FlatRingVec, GreyhoundTail, HachiProofTail,
    };

    let opening_point = next_level_opening_point(current_challenges, current_num_u, current_num_l);

    let witness_coeffs: Vec<F> = current_w.iter().map(|&d| F::from_i64(d as i64)).collect();

    let comkey_seed = expanded_setup.labrador_comkey_seed();

    // Section 4.5 ring dimension switch: split opening_point into
    // coeff_point (within-ring coefficient vars) and ring_point (ring-element index vars).
    let alpha_prime = GREYHOUND_D.trailing_zeros() as usize;
    let ring_point = &opening_point[alpha_prime..];

    // Compute eval_ring = ring_mle(ring_point): evaluate the ring-element MLE.
    let ring_witness: Vec<CyclotomicRing<F, GREYHOUND_D>> = {
        let mut out = Vec::with_capacity(witness_coeffs.len().div_ceil(GREYHOUND_D));
        for chunk in witness_coeffs.chunks(GREYHOUND_D) {
            out.push(CyclotomicRing::from_coefficients(std::array::from_fn(
                |i| chunk.get(i).copied().unwrap_or_else(F::zero),
            )));
        }
        out
    };
    let n_ring = ring_witness.len().next_power_of_two();
    let k_total = n_ring.trailing_zeros() as usize;
    assert_eq!(
        ring_point.len(),
        k_total,
        "ring_point length must equal log2(n_ring)"
    );
    let mut ring_basis = vec![F::zero(); n_ring];
    multilinear_lagrange_basis(&mut ring_basis, ring_point);
    let mut eval_ring = CyclotomicRing::<F, GREYHOUND_D>::zero();
    for (elem, &basis) in ring_witness.iter().zip(ring_basis.iter()) {
        eval_ring += elem.scale(&basis);
    }

    let (gh_proof, gh_witness, _initial_statement, fold_challenges) =
        greyhound_eval::<F, T, GREYHOUND_D>(
            &witness_coeffs,
            ring_point,
            eval_ring,
            &[],
            &comkey_seed,
            transcript,
        )?;

    let t_hat = &gh_witness.rows()[2];
    let u1 = if gh_proof.config.kappa1 > 0 {
        let b_mat = derive_extendable_comkey_matrix::<F, GREYHOUND_D>(
            gh_proof.config.kappa1,
            t_hat.len(),
            &comkey_seed,
            b"labrador/comkey/B",
        );
        mat_vec_mul(&b_mat, t_hat)
    } else {
        t_hat.clone()
    };

    let beta_sq = gh_witness.norm();
    let mut statement = greyhound_reduce::<F, GREYHOUND_D>(
        &gh_proof,
        &u1,
        ring_point,
        eval_ring,
        &fold_challenges,
        &comkey_seed,
    )?;
    statement.beta_sq = beta_sq;

    let labrador_proof = prove_with_config::<F, T, GREYHOUND_D>(
        gh_witness,
        &statement,
        &gh_proof.config,
        &comkey_seed,
        transcript,
    )?;

    Ok(HachiProofTail::Greyhound(GreyhoundTail {
        greyhound_proof: FlatGreyhoundEvalProof::from_typed(&gh_proof),
        labrador_proof: FlatLabradorProof::from_typed(&labrador_proof),
        u1: FlatRingVec::from_ring_elems(&u1),
        eval_ring: FlatRingVec::from_ring_elems(&[eval_ring]),
        beta_sq,
    }))
}

/// Verify the Greyhound/Labrador tail of a Hachi proof.
///
/// Replays the Greyhound transcript operations (absorb context, claim, u2;
/// sample fold challenges), rebuilds the Labrador statement via
/// `greyhound_reduce`, and verifies the Labrador recursive proof.
///
/// # Errors
///
/// Returns an error if transcript replay, statement reduction, or Labrador
/// verification fails.
fn greyhound_handoff_verify<F, T>(
    tail: &crate::protocol::proof::GreyhoundTail<F>,
    opening_point: &[F],
    opening_value: &F,
    expanded_setup: &HachiExpandedSetup<F>,
    transcript: &mut T,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + Valid,
    T: Transcript<F>,
{
    use crate::algebra::ring::CyclotomicRing;
    use crate::primitives::poly::multilinear_lagrange_basis;
    use crate::protocol::greyhound::greyhound_reduce;
    use crate::protocol::labrador::transcript::{
        absorb_greyhound_eval_claim, absorb_greyhound_eval_context, absorb_greyhound_u2,
        sample_greyhound_fold_challenge, GreyhoundEvalTranscriptContext,
    };
    use crate::protocol::labrador::verify;

    let gh_proof = tail.greyhound_proof.to_typed::<GREYHOUND_D>();
    let u1 = tail.u1.to_vec::<GREYHOUND_D>();
    let labrador_proof = tail.labrador_proof.to_typed::<GREYHOUND_D>();

    let eval_ring_vec = tail.eval_ring.to_vec::<GREYHOUND_D>();
    if eval_ring_vec.len() != 1 {
        return Err(HachiError::InvalidInput(
            "greyhound tail: expected exactly one eval_ring element".to_string(),
        ));
    }
    let eval_ring: CyclotomicRing<F, GREYHOUND_D> = eval_ring_vec[0];

    // Section 4.5: split opening_point into coeff_point and ring_point.
    let alpha_prime = GREYHOUND_D.trailing_zeros() as usize;
    if opening_point.len() < alpha_prime {
        return Err(HachiError::InvalidPointDimension {
            expected: alpha_prime,
            actual: opening_point.len(),
        });
    }
    let coeff_point = &opening_point[..alpha_prime];
    let ring_point = &opening_point[alpha_prime..];

    eprintln!(
        "  [greyhound_handoff_verify] opening_point.len={}, alpha'={}, ring_point.len={}, coeff_point.len={}",
        opening_point.len(), alpha_prime, ring_point.len(), coeff_point.len(),
    );

    // Consistency check: w_eval == sum_{p=0}^{D'-1} eq(bits(p), coeff_point) * eval_ring.coeff[p]
    let mut coeff_basis = vec![F::zero(); GREYHOUND_D];
    multilinear_lagrange_basis(&mut coeff_basis, coeff_point);
    let mut reconstructed = F::zero();
    for (p, &basis_p) in coeff_basis.iter().enumerate() {
        reconstructed += basis_p * eval_ring.coefficients()[p];
    }
    eprintln!(
        "  [greyhound_handoff_verify] consistency check: reconstructed==opening_value? {}",
        reconstructed == *opening_value,
    );
    if reconstructed != *opening_value {
        return Err(HachiError::InvalidInput(
            "ring dimension switch: eval_ring inconsistent with w_eval".to_string(),
        ));
    }

    let comkey_seed = expanded_setup.labrador_comkey_seed();

    absorb_greyhound_eval_context(
        transcript,
        &GreyhoundEvalTranscriptContext {
            m_rows: gh_proof.m_rows,
            n_cols: gh_proof.n_cols,
            inner_vars: gh_proof.inner_vars,
            eval_point_len: ring_point.len(),
        },
    )?;
    absorb_greyhound_eval_claim(transcript, ring_point, &eval_ring);
    absorb_greyhound_u2(transcript, &gh_proof.u2);

    let fold_challenges: Vec<F> = (0..gh_proof.n_cols)
        .map(|_| sample_greyhound_fold_challenge(transcript))
        .collect();

    let mut statement = greyhound_reduce::<F, GREYHOUND_D>(
        &gh_proof,
        &u1,
        ring_point,
        eval_ring,
        &fold_challenges,
        &comkey_seed,
    )?;
    statement.beta_sq = tail.beta_sq;

    eprintln!(
        "  [greyhound_handoff_verify] statement: u1.len={}, u2.len={}, constraints.len={}, beta_sq={}",
        statement.u1.len(), statement.u2.len(), statement.constraints.len(), statement.beta_sq,
    );
    eprintln!(
        "  [greyhound_handoff_verify] labrador_proof: levels={}, final_witness_rows={}",
        labrador_proof.levels.len(),
        labrador_proof.final_opening_witness.rows().len(),
    );

    let result = verify::<F, T, GREYHOUND_D>(&statement, &labrador_proof, &comkey_seed, transcript);
    eprintln!(
        "  [greyhound_handoff_verify] labrador verify result: {}",
        if result.is_ok() { "OK" } else { "FAIL" }
    );
    result?;

    Ok(())
}

/// Dispatch a commit-w operation to the correct ring dimension.
///
/// Each match arm builds NTT caches for the target D and calls `commit_w`.
/// `#[inline(never)]` isolates the match arms in their own stack frame,
/// preventing debug-mode stack bloat from monomorphized arms.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_commit<F, Cfg>(
    commit_d: usize,
    commit_ntt_bundle: &mut MultiDNttBundle,
    expanded: &HachiExpandedSetup<F>,
    w: &[i8],
) -> Result<(FlatRingVec<F>, FlatCommitmentHint), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    dispatch_with_ntt!(
        commit_d,
        commit_ntt_bundle,
        expanded,
        |D_COMMIT, ca, cb, _cd| {
            let (wc, wh) =
                commit_w::<F, { D_COMMIT }, WCommitmentConfig<{ D_COMMIT }, Cfg>>(w, ca, cb)?;
            Ok((
                FlatRingVec::from_commitment(&wc),
                FlatCommitmentHint::from_typed(wh),
            ))
        }
    )
}

/// Dispatch a prove-level operation to the correct ring dimension.
///
/// Handles the fast-path (`level_d == D`) and the dynamic dispatch path.
/// `#[inline(never)]` isolates the monomorphized match arms in their own
/// stack frame.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_prove_level<F, T, const D: usize, Cfg>(
    level_d: usize,
    ntt_bundle: &mut MultiDNttBundle,
    expanded: &HachiExpandedSetup<F>,
    setup_ntt_a: &NttSlotCache<D>,
    setup_ntt_b: &NttSlotCache<D>,
    setup_ntt_d: &NttSlotCache<D>,
    commit_ntt_bundle: &mut MultiDNttBundle,
    commit_d: usize,
    current_w: &[i8],
    current_hint: &FlatCommitmentHint,
    current_challenges: &[F],
    current_num_u: usize,
    current_num_l: usize,
    last_w_commitment: &FlatRingVec<F>,
    last_w_eval: F,
    transcript: &mut T,
    level: usize,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    if level_d == D {
        prove_subsequent_level::<F, T, D, Cfg>(
            expanded,
            setup_ntt_a,
            setup_ntt_b,
            setup_ntt_d,
            commit_ntt_bundle,
            commit_d,
            current_w,
            current_hint,
            current_challenges,
            current_num_u,
            current_num_l,
            last_w_commitment,
            last_w_eval,
            transcript,
            level,
        )
    } else {
        dispatch_with_ntt!(
            level_d,
            ntt_bundle,
            expanded,
            |D_LEVEL, ntt_a, ntt_b, ntt_d| {
                prove_subsequent_level::<F, T, { D_LEVEL }, Cfg>(
                    expanded,
                    ntt_a,
                    ntt_b,
                    ntt_d,
                    commit_ntt_bundle,
                    commit_d,
                    current_w,
                    current_hint,
                    current_challenges,
                    current_num_u,
                    current_num_l,
                    last_w_commitment,
                    last_w_eval,
                    transcript,
                    level,
                )
            }
        )
    }
}

/// Dispatch a verify-level operation to the correct ring dimension.
///
/// Each match arm converts the D-erased commitment to a typed one,
/// derives the w-commitment layout, and calls `verify_one_level`.
/// `#[inline(never)]` isolates the monomorphized match arms.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_verify_level<F, T, Cfg>(
    level_d: usize,
    level_proof: &HachiLevelProof<F>,
    setup: &HachiVerifierSetup<F>,
    transcript: &mut T,
    opening_point: &[F],
    opening: &F,
    current_commitment: &FlatRingVec<F>,
    basis: BasisMode,
    is_last: bool,
    final_w: Option<&[F]>,
) -> Result<Vec<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    dispatch_ring_dim!(level_d, |D_LEVEL| {
        let typed_commitment: RingCommitment<F, { D_LEVEL }> =
            current_commitment.to_ring_commitment();
        let w_layout =
            <WCommitmentConfig<{ D_LEVEL }, Cfg>>::commitment_layout(opening_point.len())?;
        verify_one_level::<F, T, { D_LEVEL }, WCommitmentConfig<{ D_LEVEL }, Cfg>>(
            level_proof,
            setup,
            transcript,
            opening_point,
            opening,
            &typed_commitment,
            basis,
            is_last,
            final_w,
            w_layout,
        )
    })
}

/// Single subsequent (recursive) prove level, extracted so that the
/// dispatch match arms contain only a function call.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_subsequent_level<F, T, const D_LEVEL: usize, Cfg>(
    expanded: &HachiExpandedSetup<F>,
    ntt_a: &NttSlotCache<D_LEVEL>,
    ntt_b: &NttSlotCache<D_LEVEL>,
    ntt_d: &NttSlotCache<D_LEVEL>,
    commit_ntt_bundle: &mut MultiDNttBundle,
    commit_d: usize,
    current_w: &[i8],
    current_hint: &FlatCommitmentHint,
    current_challenges: &[F],
    current_num_u: usize,
    current_num_l: usize,
    last_w_commitment: &FlatRingVec<F>,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] last_w_eval: F,
    transcript: &mut T,
    level: usize,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let w_poly = BalancedDigitPoly::<F, { D_LEVEL }>::from_i8_digits(current_w)?;
    let opening_point = next_level_opening_point(current_challenges, current_num_u, current_num_l);

    #[cfg(debug_assertions)]
    {
        let mut field_evals: Vec<F> = current_w.iter().map(|&d| F::from_i8(d)).collect();
        field_evals.resize(w_poly.num_ring_elems() * D_LEVEL, F::zero());
        let direct_eval = multilinear_eval(&field_evals, &opening_point).unwrap();
        if last_w_eval != direct_eval {
            eprintln!("  [hachi prove L{level}] BUG: w_eval mismatch! prev_level w_eval != w_poly eval at opening_point");
            eprintln!(
                "    w_poly ring_elems={}, field_len={}, opening_point.len()={}",
                w_poly.num_ring_elems(),
                field_evals.len(),
                opening_point.len()
            );
        } else {
            eprintln!("  [hachi prove L{level}] w_eval consistency OK");
        }
    }

    let w_commitment: RingCommitment<F, { D_LEVEL }> = last_w_commitment.to_ring_commitment();
    let typed_hint: HachiCommitmentHint<F, { D_LEVEL }> = current_hint.to_typed();

    let commit_fn: CommitFn<'_, F> = Box::new(
        |w: &[i8]| -> Result<(FlatRingVec<F>, FlatCommitmentHint), HachiError> {
            if commit_d == D_LEVEL {
                let (wc, wh) = commit_w::<F, { D_LEVEL }, WCommitmentConfig<{ D_LEVEL }, Cfg>>(
                    w, ntt_a, ntt_b,
                )?;
                Ok((
                    FlatRingVec::from_commitment(&wc),
                    FlatCommitmentHint::from_typed(wh),
                ))
            } else {
                dispatch_commit::<F, Cfg>(commit_d, commit_ntt_bundle, expanded, w)
            }
        },
    );

    let w_layout = <WCommitmentConfig<{ D_LEVEL }, Cfg>>::commitment_layout(opening_point.len())?;
    prove_one_level::<F, T, { D_LEVEL }, WCommitmentConfig<{ D_LEVEL }, Cfg>, _>(
        expanded,
        ntt_a,
        ntt_b,
        ntt_d,
        commit_fn,
        &w_poly,
        &opening_point,
        typed_hint,
        transcript,
        &w_commitment,
        BasisMode::Lagrange,
        level,
        w_layout,
    )
}

impl<F, const D: usize, Cfg> CommitmentScheme<F, D> for HachiCommitmentScheme<D, Cfg>
where
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
    Cfg: CommitmentConfig,
{
    type ProverSetup = HachiProverSetup<F, D>;
    type VerifierSetup = HachiVerifierSetup<F>;
    type Commitment = RingCommitment<F, D>;
    type Proof = HachiProof<F>;
    type CommitHint = HachiCommitmentHint<F, D>;

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::setup_prover")]
    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<F, { D }, Cfg>>::setup(max_num_vars)
                .expect("commitment setup failed");
        setup
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        HachiVerifierSetup {
            expanded: setup.expanded.clone(),
        }
    }

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::commit")]
    fn commit<P: HachiPolyOps<F, D>>(
        poly: &P,
        setup: &Self::ProverSetup,
        layout: &HachiCommitmentLayout,
    ) -> Result<(Self::Commitment, Self::CommitHint), HachiError> {
        setup.assert_layout_fits(layout);
        let t_hat_all = poly.commit_inner(
            &setup.expanded.A,
            &setup.ntt_A,
            layout.block_len,
            layout.num_digits_commit,
            layout.num_digits_open,
            layout.log_basis,
        )?;
        let t_hat_flat = flatten_i8_blocks(&t_hat_all);
        let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(&setup.ntt_B, &t_hat_flat);
        let hint = HachiCommitmentHint::new(t_hat_all);
        Ok((RingCommitment { u }, hint))
    }

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::prove")]
    fn prove<T: Transcript<F>, P: HachiPolyOps<F, D>>(
        setup: &Self::ProverSetup,
        poly: &P,
        opening_point: &[F],
        hint: Self::CommitHint,
        transcript: &mut T,
        commitment: &Self::Commitment,
        basis: BasisMode,
        layout: &HachiCommitmentLayout,
    ) -> Result<Self::Proof, HachiError> {
        let t_prove_total = Instant::now();
        let mut levels = Vec::new();

        let mut ntt_bundle = MultiDNttBundle::new();
        let mut commit_ntt_bundle = MultiDNttBundle::new();

        // Level 0: original polynomial with caller-provided layout.
        // The w-commitment is produced at the NEXT level's D.
        let commit_d_0 = Cfg::d_at_level(1, 0);
        let commit_fn_0: CommitFn<'_, F> = if commit_d_0 == D {
            Box::new(
                |w: &[i8]| -> Result<(FlatRingVec<F>, FlatCommitmentHint), HachiError> {
                    let (wc, wh) = commit_w::<F, D, Cfg>(w, &setup.ntt_A, &setup.ntt_B)?;
                    Ok((
                        FlatRingVec::from_commitment(&wc),
                        FlatCommitmentHint::from_typed(wh),
                    ))
                },
            )
        } else {
            Box::new(
                |w: &[i8]| -> Result<(FlatRingVec<F>, FlatCommitmentHint), HachiError> {
                    dispatch_commit::<F, Cfg>(
                        commit_d_0,
                        &mut commit_ntt_bundle,
                        &setup.expanded,
                        w,
                    )
                },
            )
        };
        let out = prove_one_level::<F, T, D, Cfg, P>(
            &setup.expanded,
            &setup.ntt_A,
            &setup.ntt_B,
            &setup.ntt_D,
            commit_fn_0,
            poly,
            opening_point,
            hint,
            transcript,
            commitment,
            basis,
            0,
            *layout,
        )?;
        levels.push(out.level_proof);

        let mut prev_poly_len = poly.num_ring_elems() * D;
        let mut current_w = out.w;
        let mut current_hint = out.w_hint;
        let mut current_challenges = out.sumcheck_challenges;
        let mut current_num_u = out.num_u;
        let mut current_num_l = out.num_l;
        let mut level = 1usize;

        // Subsequent levels: recursive w-opening with WCommitmentConfig.
        // Each level dispatches to the ring dimension from Cfg::d_at_level.
        // The w-commitment is produced at the NEXT level's D.
        while !should_stop_folding(current_w.len(), prev_poly_len) {
            let level_d = Cfg::d_at_level(level, current_w.len());
            let commit_d = Cfg::d_at_level(level + 1, 0);

            let last_w_eval = levels.last().unwrap().w_eval;
            let last_w_commitment = &levels.last().unwrap().w_commitment;
            let out = dispatch_prove_level::<F, T, D, Cfg>(
                level_d,
                &mut ntt_bundle,
                &setup.expanded,
                &setup.ntt_A,
                &setup.ntt_B,
                &setup.ntt_D,
                &mut commit_ntt_bundle,
                commit_d,
                &current_w,
                &current_hint,
                &current_challenges,
                current_num_u,
                current_num_l,
                last_w_commitment,
                last_w_eval,
                transcript,
                level,
            )?;

            levels.push(out.level_proof);

            prev_poly_len = current_w.len();
            current_w = out.w;
            current_hint = out.w_hint;
            current_challenges = out.sumcheck_challenges;
            current_num_u = out.num_u;
            current_num_l = out.num_l;
            level += 1;
        }

        eprintln!(
            "  [hachi prove] total ({level} levels): {:.2}s",
            t_prove_total.elapsed().as_secs_f64()
        );

        let labrador_enabled = current_w.len() > Cfg::greyhound_handoff_threshold()
            && std::env::var("HACHI_NO_LABRADOR").as_deref() != Ok("1");

        let tail = if labrador_enabled {
            eprintln!("[labrador handoff started]");
            crate::protocol::labrador_handoff::labrador_handoff_prove::<F, T, GREYHOUND_D, Cfg>(
                &current_w,
                &current_challenges,
                current_num_u,
                current_num_l,
                levels.last().unwrap().w_eval,
                &setup.expanded,
                transcript,
            )?
        } else {
            let log_basis = Cfg::decomposition().log_basis;
            let final_w = PackedDigits::from_i8_digits(&current_w, log_basis);
            crate::protocol::proof::HachiProofTail::Direct(final_w)
        };

        Ok(HachiProof { levels, tail })
    }

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::verify")]
    fn verify<T: Transcript<F>>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        opening_point: &[F],
        opening: &F,
        commitment: &Self::Commitment,
        basis: BasisMode,
        layout: &HachiCommitmentLayout,
    ) -> Result<(), HachiError> {
        use crate::protocol::proof::HachiProofTail;

        if proof.levels.is_empty() {
            return Err(HachiError::InvalidProof);
        }

        let num_levels = proof.levels.len();
        let has_handoff_tail = proof.has_handoff_tail();

        let final_w_elems: Option<Vec<F>> = match &proof.tail {
            HachiProofTail::Direct(pw) => Some(pw.to_field_elems()),
            HachiProofTail::Greyhound(_) | HachiProofTail::Labrador(_) => None,
        };

        // State carried between levels.
        // Commitment is D-erased so the loop can handle varying D per level.
        let mut current_point = opening_point.to_vec();
        let mut current_opening = *opening;
        let mut current_commitment = FlatRingVec::from_commitment(commitment);
        let mut current_basis = basis;

        for (i, level_proof) in proof.levels.iter().enumerate() {
            let is_last_hachi = i == num_levels - 1;
            // With a handoff tail, the last Hachi level is NOT the
            // final level -- verification continues in Labrador.
            let is_last = is_last_hachi && !has_handoff_tail;
            let level_d = Cfg::d_at_level(i, current_point.len());
            eprintln!(
                "  [verify] level {i}, is_last={is_last}, point_len={}, D={level_d}",
                current_point.len()
            );

            let fw_ref = final_w_elems.as_deref();
            let challenges = if i == 0 {
                let typed_commitment: RingCommitment<F, D> =
                    current_commitment.to_ring_commitment();
                verify_one_level::<F, T, D, Cfg>(
                    level_proof,
                    setup,
                    transcript,
                    &current_point,
                    &current_opening,
                    &typed_commitment,
                    current_basis,
                    is_last,
                    if is_last { fw_ref } else { None },
                    *layout,
                )?
            } else {
                dispatch_verify_level::<F, T, Cfg>(
                    level_d,
                    level_proof,
                    setup,
                    transcript,
                    &current_point,
                    &current_opening,
                    &current_commitment,
                    current_basis,
                    is_last,
                    if is_last { fw_ref } else { None },
                )?
            };

            if !is_last {
                let alpha_bits = level_d.trailing_zeros() as usize;
                let num_l = alpha_bits;
                let num_u = challenges.len() - num_l;

                current_point = next_level_opening_point(&challenges, num_u, num_l);
                current_opening = level_proof.w_eval;
                current_commitment = level_proof.w_commitment.clone();
                current_basis = BasisMode::Lagrange;
            }
        }

        match &proof.tail {
            HachiProofTail::Greyhound(ref tail) => {
                greyhound_handoff_verify::<F, T>(
                    tail,
                    &current_point,
                    &current_opening,
                    &setup.expanded,
                    transcript,
                )?;
            }
            HachiProofTail::Labrador(ref tail) => {
                crate::protocol::labrador_handoff::labrador_handoff_verify::<F, T, GREYHOUND_D, Cfg>(
                    tail,
                    &current_point,
                    &current_opening,
                    &setup.expanded,
                    transcript,
                )?;
            }
            HachiProofTail::Direct(_) => {}
        }

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        unimplemented!()
    }
}

/// Verify one fold level.
///
/// At the final level, `final_w` is provided and the verifier checks w_val
/// from it directly. At intermediate levels, `level_proof.w_eval` is used.
///
/// Returns the sumcheck challenges for chaining into the next level.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn verify_one_level<F, T, const D: usize, Cfg>(
    level_proof: &HachiLevelProof<F>,
    setup: &HachiVerifierSetup<F>,
    transcript: &mut T,
    opening_point: &[F],
    opening: &F,
    commitment: &RingCommitment<F, D>,
    basis: BasisMode,
    is_last: bool,
    final_w: Option<&[F]>,
    layout: HachiCommitmentLayout,
) -> Result<Vec<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let y_ring: CyclotomicRing<F, D> = level_proof.y_ring_typed();
    let v_typed: Vec<CyclotomicRing<F, D>> = level_proof.v_typed();

    let alpha_bits = Cfg::D.trailing_zeros() as usize;
    if opening_point.len() < alpha_bits {
        return Err(HachiError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    let target_num_vars = layout.m_vars + layout.r_vars + alpha_bits;
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let inner_point = &padded_point[..alpha_bits];
    let reduced_opening_point = &padded_point[alpha_bits..];

    commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
    for pt in &padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

    let v = reduce_inner_openings_to_ring_elements::<F, { D }>(inner_point, basis)?;
    let d = F::from_u64(Cfg::D as u64);
    let trace_lhs = trace::<F, { D }>(&(y_ring * v.sigma_m1()));
    let trace_rhs = d * *opening;
    if trace_lhs != trace_rhs {
        return Err(HachiError::InvalidProof);
    }

    let ring_opening_point = ring_opening_point_from_field::<F>(
        reduced_opening_point,
        layout.r_vars,
        layout.m_vars,
        basis,
    )?;
    let quad_eq = Box::new(QuadraticEquation::<F, { D }, Cfg>::new_verifier(
        ring_opening_point,
        v_typed.clone(),
        transcript,
        commitment,
        &y_ring,
        layout,
    )?);

    let w_len = if is_last {
        final_w.map_or(0, |fw| fw.len())
    } else {
        w_ring_element_count::<F, Cfg>(layout) * D
    };
    eprintln!("  [verify] w_len={w_len}, is_last={is_last}");

    let rs = ring_switch_verifier::<F, T, { D }, Cfg>(
        &quad_eq,
        &setup.expanded,
        w_len,
        &level_proof.w_commitment,
        transcript,
        layout,
    )?;

    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);

    let fused_verifier = if is_last {
        let fw = final_w.ok_or(HachiError::InvalidProof)?;
        let (w_evals_full, _, _) = build_w_evals(fw, Cfg::D)?;
        HachiSumcheckVerifier::new(
            batching_coeff,
            w_evals_full,
            rs.tau0,
            rs.b,
            rs.alpha_evals_y,
            rs.m_evals_x,
            rs.tau1,
            v_typed,
            commitment.u.clone(),
            y_ring,
            rs.alpha,
            rs.num_u,
            rs.num_l,
        )
    } else {
        HachiSumcheckVerifier::new(
            batching_coeff,
            Vec::new(),
            rs.tau0,
            rs.b,
            rs.alpha_evals_y,
            rs.m_evals_x,
            rs.tau1,
            v_typed,
            commitment.u.clone(),
            y_ring,
            rs.alpha,
            rs.num_u,
            rs.num_l,
        )
        .with_w_val_override(level_proof.w_eval)
    };

    let challenges = verify_sumcheck::<F, _, F, _, _>(
        &level_proof.sumcheck_proof,
        &fused_verifier,
        transcript,
        |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
    )?;

    Ok(challenges)
}

/// Re-derive the ring-switch challenge `alpha` and the expanded `M_a` vector
/// by replaying the transcript from the proof data and setup, exactly as the
/// verifier does.
#[cfg(test)]
pub(crate) fn rederive_alpha_and_m_a<F, const D: usize, Cfg>(
    proof: &HachiProof<F>,
    setup: &HachiVerifierSetup<F>,
    opening_point: &[F],
    commitment: &RingCommitment<F, D>,
) -> Result<(F, Vec<F>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + 'static,
    Cfg: CommitmentConfig,
{
    let level0 = proof.levels.first().ok_or(HachiError::InvalidProof)?;
    let y_ring: CyclotomicRing<F, D> = level0.y_ring_typed();
    let v_typed: Vec<CyclotomicRing<F, D>> = level0.v_typed();

    let alpha_bits = Cfg::D.trailing_zeros() as usize;
    if opening_point.len() < alpha_bits {
        return Err(HachiError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    let layout = Cfg::commitment_layout(opening_point.len())?;
    let ring_opening_point = ring_opening_point_from_field::<F>(
        &opening_point[alpha_bits..],
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
    )?;
    let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_HACHI_PROTOCOL);

    commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
    for pt in opening_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

    let quad_eq = QuadraticEquation::<F, D, Cfg>::new_verifier(
        ring_opening_point,
        v_typed,
        &mut transcript,
        commitment,
        &y_ring,
        layout,
    )?;
    transcript.append_serde(ABSORB_SUMCHECK_W, &level0.w_commitment);
    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);
    let m_a = compute_m_a_reference::<F, D, Cfg>(
        &setup.expanded,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        &alpha,
        layout,
    )?;
    let m_a_vec = expand_m_a::<F, D>(&m_a, alpha, layout.log_basis)?;
    Ok((alpha, m_a_vec))
}

/// Re-derive the ring-switch challenge `alpha` and the fused `m_evals_x`
/// table by replaying the verifier transcript from the proof data and setup.
#[cfg(test)]
pub(crate) fn rederive_alpha_and_m_evals_x<F, const D: usize, Cfg>(
    proof: &HachiProof<F>,
    setup: &HachiVerifierSetup<F>,
    opening_point: &[F],
    commitment: &RingCommitment<F, D>,
    tau1: &[F],
) -> Result<(F, Vec<F>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + 'static,
    Cfg: CommitmentConfig,
{
    let level0 = proof.levels.first().ok_or(HachiError::InvalidProof)?;
    let y_ring: CyclotomicRing<F, D> = level0.y_ring_typed();
    let v_typed: Vec<CyclotomicRing<F, D>> = level0.v_typed();

    let alpha_bits = Cfg::D.trailing_zeros() as usize;
    if opening_point.len() < alpha_bits {
        return Err(HachiError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    let layout = Cfg::commitment_layout(opening_point.len())?;
    let ring_opening_point = ring_opening_point_from_field::<F>(
        &opening_point[alpha_bits..],
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
    )?;
    let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_HACHI_PROTOCOL);

    commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
    for pt in opening_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

    let quad_eq = QuadraticEquation::<F, D, Cfg>::new_verifier(
        ring_opening_point,
        v_typed,
        &mut transcript,
        commitment,
        &y_ring,
        layout,
    )?;
    transcript.append_serde(ABSORB_SUMCHECK_W, &level0.w_commitment);
    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);
    let alpha_evals_y = build_alpha_evals_y(alpha, D);
    let m_evals_x = compute_m_evals_x::<F, D, Cfg>(
        &setup.expanded,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        alpha,
        &alpha_evals_y,
        layout,
        tau1,
    )?;
    Ok((alpha, m_evals_x))
}

fn lagrange_weights<F: FieldCore>(point: &[F]) -> Vec<F> {
    let len = 1usize << point.len();
    let mut weights = vec![F::zero(); len];
    multilinear_lagrange_basis(&mut weights, point);
    weights
}

/// Multilinear monomial weights: `⊗ᵢ (1, xᵢ)`.
///
/// The j-th entry is `∏_{i ∈ bits(j)} point[i]`.
fn monomial_weights<F: FieldCore>(point: &[F]) -> Vec<F> {
    let len = 1usize << point.len();
    let mut weights = vec![F::zero(); len];
    weights[0] = F::one();
    for (level, &p) in point.iter().enumerate() {
        let k = 1usize << level;
        for i in (0..k).rev() {
            weights[i + k] = weights[i] * p;
        }
    }
    weights
}

fn basis_weights<F: FieldCore>(point: &[F], mode: BasisMode) -> Vec<F> {
    match mode {
        BasisMode::Lagrange => lagrange_weights(point),
        BasisMode::Monomial => monomial_weights(point),
    }
}

pub(crate) fn ring_opening_point_from_field<F: FieldCore>(
    opening_point: &[F],
    r_vars: usize,
    m_vars: usize,
    basis: BasisMode,
) -> Result<RingOpeningPoint<F>, HachiError> {
    let expected_len = r_vars
        .checked_add(m_vars)
        .ok_or_else(|| HachiError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() != expected_len {
        return Err(HachiError::InvalidPointDimension {
            expected: expected_len,
            actual: opening_point.len(),
        });
    }

    // Sequential ordering: M variables (position in block) come first,
    // R variables (block selection) come second.
    let a = basis_weights(&opening_point[..m_vars], basis);
    let b = basis_weights(&opening_point[m_vars..], basis);
    Ok(RingOpeningPoint { a, b })
}

fn reduce_inner_openings_to_ring_elements<F: FieldCore, const D: usize>(
    inner_point: &[F],
    basis: BasisMode,
) -> Result<CyclotomicRing<F, D>, HachiError> {
    let weights = basis_weights(inner_point, basis);
    if weights.len() != D {
        return Err(HachiError::InvalidInput(format!(
            "inner basis length {} does not match D={D}",
            weights.len()
        )));
    }
    Ok(CyclotomicRing::from_slice(&weights))
}

fn trace<F: FieldCore + FromSmallInt, const D: usize>(u: &CyclotomicRing<F, D>) -> F {
    let d = F::from_u64(D as u64);
    u.coefficients()[0] * d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::commitment::CommitmentConfig;
    use crate::protocol::hachi_poly_ops::DensePoly;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::test_utils::F;
    use crate::{CommitmentScheme, FromSmallInt};

    type Cfg = SmallTestCommitmentConfig;
    const D: usize = Cfg::D;
    type Scheme = HachiCommitmentScheme<D, Cfg>;

    fn make_dense_poly(num_vars: usize) -> (DensePoly<F, D>, Vec<F>) {
        let len = 1usize << num_vars;
        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
        (poly, evals)
    }

    #[test]
    fn verify_passes_for_consistent_opening() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;

        let (poly, evals) = make_dense_poly(num_vars);

        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);

        let (commitment, hint) =
            <Scheme as CommitmentScheme<F, D>>::commit(&poly, &setup, &layout).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let lw = lagrange_weights(&opening_point);
        let opening: F = evals
            .iter()
            .zip(lw.iter())
            .fold(F::zero(), |a, (&c, &w)| a + c * w);

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let proof = <Scheme as CommitmentScheme<F, D>>::prove(
            &setup,
            &poly,
            &opening_point,
            hint,
            &mut prover_transcript,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let result = <Scheme as CommitmentScheme<F, D>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn verify_rejects_wrong_opening() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;

        let (poly, evals) = make_dense_poly(num_vars);

        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);

        let (commitment, hint) =
            <Scheme as CommitmentScheme<F, D>>::commit(&poly, &setup, &layout).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let lw = lagrange_weights(&opening_point);
        let opening: F = evals
            .iter()
            .zip(lw.iter())
            .fold(F::zero(), |a, (&c, &w)| a + c * w);

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let proof = <Scheme as CommitmentScheme<F, D>>::prove(
            &setup,
            &poly,
            &opening_point,
            hint,
            &mut prover_transcript,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        )
        .unwrap();

        let wrong_opening = opening + F::one();
        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let result = <Scheme as CommitmentScheme<F, D>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &wrong_opening,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        );

        assert!(
            result.is_err(),
            "verify must reject an incorrect opening value"
        );
    }

    #[test]
    fn monomial_basis_prove_verify_round_trip() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;

        let coeffs: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &coeffs).unwrap();

        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);

        let (commitment, hint) =
            <Scheme as CommitmentScheme<F, D>>::commit(&poly, &setup, &layout).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();

        let mw = monomial_weights(&opening_point);
        let opening: F = coeffs
            .iter()
            .zip(mw.iter())
            .fold(F::zero(), |acc, (&c, &w)| acc + c * w);

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/monomial");
        let proof = <Scheme as CommitmentScheme<F, D>>::prove(
            &setup,
            &poly,
            &opening_point,
            hint,
            &mut prover_transcript,
            &commitment,
            BasisMode::Monomial,
            &layout,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/monomial");
        let result = <Scheme as CommitmentScheme<F, D>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
            BasisMode::Monomial,
            &layout,
        );

        assert!(
            result.is_ok(),
            "monomial-basis proof should verify: {result:?}"
        );
    }

    /// A config identical to `DynamicSmallTestCommitmentConfig` but with a
    /// Greyhound handoff threshold of 0 (always hand off).
    #[derive(Clone, Copy, Debug, Default)]
    struct GreyhoundTestConfig;

    impl CommitmentConfig for GreyhoundTestConfig {
        const D: usize = 16;
        const N_A: usize = 8;
        const N_B: usize = 4;
        const N_D: usize = 4;
        const CHALLENGE_WEIGHT: usize = 3;

        fn decomposition() -> crate::protocol::commitment::DecompositionParams {
            crate::protocol::commitment::DecompositionParams {
                log_basis: 3,
                log_commit_bound: 32,
                log_open_bound: Some(128),
            }
        }

        fn commitment_layout(
            max_num_vars: usize,
        ) -> Result<crate::protocol::commitment::HachiCommitmentLayout, crate::error::HachiError>
        {
            let alpha = Self::D.trailing_zeros() as usize;
            let reduced_vars = max_num_vars.checked_sub(alpha).ok_or_else(|| {
                crate::error::HachiError::InvalidSetup(
                    "max_num_vars is smaller than alpha".to_string(),
                )
            })?;
            if reduced_vars == 0 {
                return Err(crate::error::HachiError::InvalidSetup(
                    "need at least 1 reduced variable".to_string(),
                ));
            }
            let m_vars = (reduced_vars + 1) / 2;
            let r_vars = reduced_vars - m_vars;
            crate::protocol::commitment::HachiCommitmentLayout::new::<Self>(
                m_vars,
                r_vars,
                &Self::decomposition(),
            )
        }

        fn greyhound_handoff_threshold() -> usize {
            0
        }
    }

    #[test]
    fn greyhound_tail_prove_verify_round_trip() {
        type GF = crate::algebra::Fp128<0xfffffffffffffffffffffffffffffeed>;
        type GCfg = GreyhoundTestConfig;
        const GD: usize = GCfg::D;
        type GScheme = HachiCommitmentScheme<GD, GCfg>;

        let layout = GCfg::commitment_layout(16).unwrap();
        let alpha = GD.trailing_zeros() as usize;
        let num_vars = layout.m_vars + layout.r_vars + alpha;

        let len = 1usize << num_vars;
        let evals: Vec<GF> = (0..len).map(|i| GF::from_u64(i as u64)).collect();
        let poly = DensePoly::<GF, GD>::from_field_evals(num_vars, &evals).unwrap();

        let setup = <GScheme as CommitmentScheme<GF, GD>>::setup_prover(num_vars);
        let verifier_setup = <GScheme as CommitmentScheme<GF, GD>>::setup_verifier(&setup);

        let (commitment, hint) =
            <GScheme as CommitmentScheme<GF, GD>>::commit(&poly, &setup, &layout).unwrap();

        let opening_point: Vec<GF> = (0..num_vars)
            .map(|i| GF::from_u64((i + 2) as u64))
            .collect();
        let lw = lagrange_weights(&opening_point);
        let opening: GF = evals
            .iter()
            .zip(lw.iter())
            .fold(GF::zero(), |a, (&c, &w)| a + c * w);

        let mut prover_transcript = Blake2bTranscript::<GF>::new(b"test/greyhound-tail");
        let proof = <GScheme as CommitmentScheme<GF, GD>>::prove(
            &setup,
            &poly,
            &opening_point,
            hint,
            &mut prover_transcript,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        )
        .unwrap();

        assert!(
            proof.has_labrador_tail(),
            "expected Labrador tail, got {:?}",
            if proof.has_greyhound_tail() {
                "Greyhound"
            } else {
                "Direct"
            }
        );

        let mut verifier_transcript = Blake2bTranscript::<GF>::new(b"test/greyhound-tail");
        let result = <GScheme as CommitmentScheme<GF, GD>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        );

        assert!(
            result.is_ok(),
            "Greyhound-tail proof should verify: {result:?}"
        );
    }
}
