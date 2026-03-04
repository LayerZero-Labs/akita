//! Commitment scheme trait implementation.

use crate::algebra::fields::wide::HasWide;
use crate::algebra::fields::HasUnreducedOps;
use crate::algebra::{CyclotomicRing, SparseChallenge};
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::linear::{flatten_i8_blocks, mat_vec_mul_ntt_single_i8};
use crate::protocol::commitment::utils::ntt_cache::MultiDNttBundle;
use crate::protocol::commitment::{
    AppendToTranscript, CommitmentConfig, CommitmentScheme, HachiCommitmentCore,
    HachiCommitmentLayout, HachiExpandedSetup, HachiProverSetup, HachiVerifierSetup,
    RingCommitment, RingCommitmentScheme,
};
use crate::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps};
use crate::protocol::opening_point::{BasisMode, RingOpeningPoint};
use crate::protocol::proof::{
    DigitLut, FlatCommitmentHint, FlatRingVec, HachiCommitmentHint, HachiLevelProof, HachiProof,
    PackedDigits,
};
use crate::protocol::quadratic_equation::{compute_m_a_streaming, QuadraticEquation};
use crate::protocol::ring_switch::{
    build_w_evals, commit_w, eval_ring_at, m_row_count, ring_switch_build_w, ring_switch_finalize,
    ring_switch_verifier, w_ring_element_count, RingSwitchOutput, WCommitmentConfig,
};
use crate::protocol::sumcheck::eq_poly::EqPolynomial;
use crate::protocol::sumcheck::hachi_sumcheck::{HachiSumcheckProver, HachiSumcheckVerifier};
use crate::protocol::sumcheck::{
    multilinear_eval, multilinear_eval_small, prove_sumcheck, range_check_eval, verify_sumcheck,
};
use crate::protocol::transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use crate::protocol::transcript::Transcript;
use crate::{dispatch_ring_dim, dispatch_with_ntt};
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt};
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
        compute_m_a_streaming::<F, D, Cfg>(expanded, opening_point, challenges, &rs.alpha, layout)
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

    let eq_tau1 = EqPolynomial::evals(&rs.tau1);
    let mut verifier_claim = F::zero();
    for (i, eq_i) in eq_tau1.iter().enumerate() {
        let y_i = if i < y_full.len() {
            y_full[i]
        } else {
            F::zero()
        };
        verifier_claim += *eq_i * y_i;
    }
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
    let outer_weights = {
        let _span = tracing::info_span!("outer_basis_weights", level).entered();
        basis_weights(outer_point, basis)
    };
    let fold_scalars = &ring_opening_point.a;
    let (y_ring, w_folded) = {
        let _span = tracing::info_span!("evaluate_and_fold", level).entered();
        poly.evaluate_and_fold(&outer_weights, fold_scalars, layout.block_len)
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
    let num_u = rs.num_u;
    let num_l = rs.num_l;
    let w_evals_small = rs.w_evals.clone();
    let mut fused_prover = HachiSumcheckProver::new(
        batching_coeff,
        rs.w_evals,
        &rs.tau0,
        rs.b,
        &rs.alpha_evals_y,
        &rs.m_evals_x,
        num_u,
        num_l,
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

    prove_level_selfcheck(
        &rs.tau0,
        &sumcheck_challenges,
        w_eval,
        rs.b,
        batching_coeff,
        &rs.alpha_evals_y,
        &rs.m_evals_x,
        num_u,
        _final_claim,
        level,
    );

    Ok(ProveLevelOutput {
        level_proof: HachiLevelProof::new::<D>(
            y_ring,
            quad_eq.v,
            sumcheck_proof,
            rs.w_commitment,
            w_eval,
        ),
        w: rs.w,
        w_hint: rs.w_hint,
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
fn next_level_opening_point<F: FieldCore>(
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

/// Build a `DensePoly` from the flat w digit vector, converting i8 -> F
/// via lookup table and padding to the next power of two.
#[tracing::instrument(skip_all, name = "dense_poly_from_w")]
fn dense_poly_from_w<F: FieldCore + FromSmallInt, const D: usize>(
    w: &[i8],
    log_basis: u32,
) -> Result<DensePoly<F, D>, HachiError> {
    let lut = DigitLut::<F>::new(log_basis);
    let total_coeffs = w.len().next_power_of_two().max(D);
    let num_vars = total_coeffs.trailing_zeros() as usize;
    let mut padded: Vec<F> = w.iter().map(|&d| lut.get(d)).collect();
    padded.resize(total_coeffs, F::zero());
    DensePoly::from_field_evals(num_vars, &padded)
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
    last_w_eval: F,
    transcript: &mut T,
    level: usize,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let w_poly = dense_poly_from_w::<F, { D_LEVEL }>(current_w, Cfg::decomposition().log_basis)?;
    let opening_point = next_level_opening_point(current_challenges, current_num_u, current_num_l);

    {
        let field_evals: Vec<F> = w_poly
            .coeffs
            .iter()
            .flat_map(|r| r.coeffs.iter().copied())
            .collect();
        let direct_eval = multilinear_eval(&field_evals, &opening_point).unwrap();
        if last_w_eval != direct_eval {
            eprintln!("  [hachi prove L{level}] BUG: w_eval mismatch! prev_level w_eval != w_poly eval at opening_point");
            eprintln!(
                "    w_poly ring_elems={}, field_len={}, opening_point.len()={}",
                w_poly.coeffs.len(),
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
    prove_one_level::<
        F,
        T,
        { D_LEVEL },
        WCommitmentConfig<{ D_LEVEL }, Cfg>,
        DensePoly<F, { D_LEVEL }>,
    >(
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
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps,
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

        let log_basis = Cfg::decomposition().log_basis;
        let final_w = PackedDigits::from_i8_digits(&current_w, log_basis);

        Ok(HachiProof { levels, final_w })
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
        if proof.levels.is_empty() {
            return Err(HachiError::InvalidProof);
        }

        let num_levels = proof.levels.len();
        let final_w_elems: Vec<F> = proof.final_w.to_field_elems();

        // State carried between levels.
        // Commitment is D-erased so the loop can handle varying D per level.
        let mut current_point = opening_point.to_vec();
        let mut current_opening = *opening;
        let mut current_commitment = FlatRingVec::from_commitment(commitment);
        let mut current_basis = basis;

        for (i, level_proof) in proof.levels.iter().enumerate() {
            let is_last = i == num_levels - 1;
            let level_d = Cfg::d_at_level(i, current_point.len());
            eprintln!(
                "  [verify] level {i}, is_last={is_last}, point_len={}, D={level_d}",
                current_point.len()
            );

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
                    if is_last { Some(&final_w_elems) } else { None },
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
                    if is_last { Some(&final_w_elems) } else { None },
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
    let m_a = compute_m_a_streaming::<F, D, Cfg>(
        &setup.expanded,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        &alpha,
        layout,
    )?;
    let m_a_vec = expand_m_a::<F, D>(&m_a, alpha, layout.log_basis)?;
    Ok((alpha, m_a_vec))
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

fn ring_opening_point_from_field<F: FieldCore>(
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
}
