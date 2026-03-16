//! Commitment scheme trait implementation.

use crate::algebra::fields::wide::HasWide;
use crate::algebra::fields::HasUnreducedOps;
use crate::algebra::CyclotomicRing;
#[cfg(debug_assertions)]
use crate::algebra::SparseChallenge;
use crate::error::HachiError;
use crate::primitives::serialization::Valid;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::linear::{flatten_i8_blocks, mat_vec_mul_ntt_single_i8};
use crate::protocol::commitment::utils::ntt_cache::{MultiDNttBundle, MultiDNttCaches};
use crate::protocol::commitment::{
    AppendToTranscript, CommitmentConfig, CommitmentScheme, HachiCommitmentCore,
    HachiCommitmentLayout, HachiExpandedSetup, HachiProverSetup, HachiVerifierSetup,
    RingCommitment, RingCommitmentScheme,
};
use crate::protocol::hachi_poly_ops::{BalancedDigitPoly, HachiPolyOps};
use crate::protocol::labrador_handoff::{labrador_handoff_prove, labrador_handoff_verify};
#[cfg(debug_assertions)]
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, BasisMode,
};
use crate::protocol::proof::{
    FlatCommitmentHint, FlatRingVec, HachiCommitmentHint, HachiLevelProof, HachiProof,
    HachiProofTail, LabradorTail, PackedDigits,
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
use crate::protocol::sumcheck::{prove_sumcheck, verify_sumcheck};
use crate::protocol::transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use crate::protocol::transcript::Transcript;
use crate::{dispatch_ring_dim, dispatch_with_d_ntt, dispatch_with_ntt};
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
#[cfg(test)]
use crate::{HachiDeserialize, HachiSerialize};

/// Minimum w vector length (in field elements) below which further folding
/// is not beneficial.  When `w.len() <= MIN_W_LEN_FOR_FOLDING`, the prover
/// sends `w` directly instead of recursing.
const MIN_W_LEN_FOR_FOLDING: usize = 4096;

/// Minimum shrink ratio (next_w / prev_w) below which further folding
/// stops being worthwhile.  If the w vector doesn't shrink by at least
/// this factor, the overhead of another fold level outweighs the saving.
const MIN_SHRINK_RATIO: f64 = 0.5;

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

    tracing::debug!(
        level,
        num_rows,
        x_len,
        m_a_cols = m_a.first().map_or(0, |r| r.len()),
        "per-row M*w=y diagnostic"
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
        tracing::debug!(
            row = i,
            row_name,
            matches = residual.is_zero(),
            residual_is_zero = residual.is_zero(),
            mw_is_zero = mw_i.is_zero(),
            y_is_zero = y_i.is_zero(),
            "diagnostic row"
        );
    }

    let verifier_claim = relation_claim_from_rows::<F, D>(&rs.tau1, rs.alpha, v, u, y_ring);
    let x_mask = x_len - 1;
    let mut prover_claim = F::zero();
    for (idx, &w) in rs.w_evals.iter().enumerate() {
        prover_claim +=
            F::from_i64(w as i64) * rs.alpha_evals_y[idx >> rs.num_u] * rs.m_evals_x[idx & x_mask];
    }
    tracing::debug!(
        level,
        claims_match = (verifier_claim == prover_claim),
        prover_is_zero = prover_claim.is_zero(),
        verifier_is_zero = verifier_claim.is_zero(),
        "relation_claim cross-check"
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
        tracing::warn!(level, "PROVER self-check FAILED: expected != final_claim");
    } else {
        tracing::debug!(level, "PROVER self-check OK");
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
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level,
            "prove_one_level"
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

    let fold_scalars = &ring_opening_point.a;
    let eval_outer_scalars = &ring_opening_point.b;
    let (y_ring, w_folded) = {
        let _span = tracing::info_span!(
            "evaluate_and_fold",
            level,
            num_ring_elems = poly.num_ring_elems()
        )
        .entered();
        poly.evaluate_and_fold(eval_outer_scalars, fold_scalars, layout.block_len)
    };

    commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
    for pt in &padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

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

    let w =
        ring_switch_build_w::<F, { D }, Cfg>(&mut quad_eq, expanded, ntt_a, ntt_b, ntt_d, layout)?;

    let (w_commitment_flat, w_hint_flat) = {
        let _span = tracing::info_span!("commit_w_level", level).entered();
        commit_w_fn(&w)?
    };

    let rs = ring_switch_finalize::<F, T, { D }, Cfg>(
        &quad_eq,
        expanded,
        transcript,
        w,
        w_commitment_flat,
        w_hint_flat,
        layout,
    )?;

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

    let relation_claim =
        relation_claim_from_rows::<F, D>(&rs.tau1, rs.alpha, &quad_eq.v, &commitment.u, &y_ring);
    let RingSwitchOutput {
        w,
        w_commitment,
        w_hint,
        w_evals,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        num_u,
        num_l,
        tau0,
        tau1: _,
        b,
        alpha: _,
    } = rs;
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
        live_x_cols,
        num_u,
        num_l,
        relation_claim,
    );

    let (sumcheck_proof, sumcheck_challenges, _final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut fused_prover, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
        })?;

    let w_eval = {
        let _span = tracing::info_span!("multilinear_eval", level).entered();
        fused_prover.final_w_eval()
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

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_labrador_handoff_prove<F, T, const D: usize, Cfg>(
    current_w: &[i8],
    current_hint: &FlatCommitmentHint,
    current_challenges: &[F],
    current_num_u: usize,
    current_num_l: usize,
    current_commitment: &FlatRingVec<F>,
    setup: &HachiProverSetup<F, D>,
    handoff_ntt_d_cache: &mut MultiDNttCaches,
    transcript: &mut T,
) -> Result<HachiProofTail<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let handoff_d = current_commitment.ring_dim();
    if current_hint.ring_dim() != handoff_d {
        return Err(HachiError::InvalidInput(format!(
            "handoff hint/commitment D mismatch: hint={}, commitment={handoff_d}",
            current_hint.ring_dim()
        )));
    }

    if handoff_d == D {
        let typed_hint: HachiCommitmentHint<F, D> = current_hint.to_typed();
        let typed_commitment: RingCommitment<F, D> = current_commitment.to_ring_commitment();
        return labrador_handoff_prove::<F, T, D, Cfg>(
            current_w,
            &typed_hint,
            &typed_commitment,
            current_challenges,
            current_num_u,
            current_num_l,
            &setup.expanded,
            &setup.ntt_D,
            transcript,
        );
    }

    dispatch_with_d_ntt!(
        handoff_d,
        handoff_ntt_d_cache,
        &setup.expanded,
        |D_HANDOFF, ntt_d| {
            let typed_hint: HachiCommitmentHint<F, { D_HANDOFF }> = current_hint.to_typed();
            let typed_commitment: RingCommitment<F, { D_HANDOFF }> =
                current_commitment.to_ring_commitment();
            labrador_handoff_prove::<F, T, { D_HANDOFF }, Cfg>(
                current_w,
                &typed_hint,
                &typed_commitment,
                current_challenges,
                current_num_u,
                current_num_l,
                &setup.expanded,
                ntt_d,
                transcript,
            )
        }
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_labrador_handoff_verify<F, T, Cfg>(
    tail: &LabradorTail<F>,
    opening_point: &[F],
    opening: &F,
    current_commitment: &FlatRingVec<F>,
    expanded_setup: &HachiExpandedSetup<F>,
    transcript: &mut T,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let handoff_d = current_commitment.ring_dim();
    if tail.v.ring_dim() != handoff_d
        || tail.y_ring.ring_dim() != handoff_d
        || tail.labrador_proof.levels.iter().any(|level| {
            level.inner_opening_payload.ring_dim() != handoff_d
                || level.linear_garbage_payload.ring_dim() != handoff_d
                || level.jl_lift_residuals.ring_dim() != handoff_d
        })
        || tail
            .labrador_proof
            .final_opening_witness
            .rows
            .iter()
            .any(|row| row.ring_dim() != handoff_d)
    {
        return Err(HachiError::InvalidProof);
    }

    dispatch_ring_dim!(handoff_d, |D_HANDOFF| {
        let typed_commitment: RingCommitment<F, { D_HANDOFF }> =
            current_commitment.to_ring_commitment();
        labrador_handoff_verify::<F, T, { D_HANDOFF }, Cfg>(
            tail,
            opening_point,
            opening,
            &typed_commitment,
            expanded_setup,
            transcript,
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
            tracing::error!(
                level,
                ring_elems = w_poly.num_ring_elems(),
                field_len = field_evals.len(),
                point_len = opening_point.len(),
                "BUG: w_eval mismatch! prev_level w_eval != w_poly eval at opening_point"
            );
        } else {
            tracing::debug!(level, "w_eval consistency OK");
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
        let inner = poly.commit_inner_witness(
            &setup.expanded.A,
            &setup.ntt_A,
            layout.block_len,
            layout.num_digits_commit,
            layout.num_digits_open,
            layout.log_basis,
        )?;
        let inner_opening_digits_flat = flatten_i8_blocks(&inner.t_hat);
        let u: Vec<CyclotomicRing<F, D>> =
            mat_vec_mul_ntt_single_i8(&setup.ntt_B, &inner_opening_digits_flat);
        let hint = HachiCommitmentHint::with_t(inner.t_hat, inner.t);
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

        tracing::info!(
            levels = level,
            elapsed_s = t_prove_total.elapsed().as_secs_f64(),
            "hachi prove complete"
        );

        // let handoff_ring_dim = current_hint.ring_dim();
        let labrador_enabled = current_w.len() > Cfg::labrador_handoff_threshold()
            // && handoff_ring_dim <= 64
            && std::env::var("HACHI_NO_LABRADOR").as_deref() != Ok("1");
        let final_w_basis = if level > 1 {
            Cfg::w_log_basis()
        } else {
            Cfg::decomposition().log_basis
        };

        let tail = if labrador_enabled {
            tracing::info!("labrador handoff started");
            dispatch_labrador_handoff_prove::<F, T, D, Cfg>(
                &current_w,
                &current_hint,
                &current_challenges,
                current_num_u,
                current_num_l,
                &levels.last().unwrap().w_commitment,
                setup,
                &mut commit_ntt_bundle.D_mat,
                transcript,
            )?
        } else {
            let final_w = PackedDigits::from_i8_digits(&current_w, final_w_basis);
            HachiProofTail::Direct(final_w)
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
        if proof.levels.is_empty() {
            return Err(HachiError::InvalidProof);
        }

        let num_levels = proof.levels.len();
        let has_handoff_tail = proof.has_handoff_tail();

        let final_w_elems: Option<Vec<F>> = match &proof.tail {
            HachiProofTail::Direct(pw) => Some(pw.to_field_elems()),
            HachiProofTail::Labrador(_) => None,
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
            tracing::debug!(
                level = i,
                is_last,
                point_len = current_point.len(),
                D = level_d,
                "verify level"
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
            HachiProofTail::Labrador(ref tail) => {
                dispatch_labrador_handoff_verify::<F, T, Cfg>(
                    tail,
                    &current_point,
                    &current_opening,
                    &current_commitment,
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

    let v = reduce_inner_opening_to_ring_element::<F, { D }>(inner_point, basis)?;
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
    tracing::debug!(w_len, is_last, "verify ring_switch");

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

fn trace<F: FieldCore + FromSmallInt, const D: usize>(u: &CyclotomicRing<F, D>) -> F {
    let d = F::from_u64(D as u64);
    u.coefficients()[0] * d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::serialization::Compress;
    use crate::protocol::commitment::CommitmentConfig;
    use crate::protocol::hachi_poly_ops::DensePoly;
    use crate::protocol::opening_point::{lagrange_weights, monomial_weights};
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::test_utils::F;
    use crate::{CommitmentScheme, FromSmallInt};
    use std::sync::OnceLock;

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
    /// handoff threshold of 0 (always hand off to Labrador).
    #[derive(Clone, Copy, Debug, Default)]
    struct HandoffTestConfig;

    impl CommitmentConfig for HandoffTestConfig {
        const D: usize = 64;
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
            let m_vars = reduced_vars.div_ceil(2);
            let r_vars = reduced_vars - m_vars;
            crate::protocol::commitment::HachiCommitmentLayout::new::<Self>(
                m_vars,
                r_vars,
                &Self::decomposition(),
            )
        }

        fn labrador_handoff_threshold() -> usize {
            0
        }
    }

    type HandoffField = crate::algebra::Fp128<0xfffffffffffffffffffffffffffffeed>;
    type HandoffScheme = HachiCommitmentScheme<{ HandoffTestConfig::D }, HandoffTestConfig>;
    const HANDOFF_FIXTURE_LABEL: &[u8] = b"test/labrador-tail-fixture";
    const HANDOFF_SPLICE_LABEL: &[u8] = b"test/labrador-tail-splice";

    #[derive(Clone)]
    struct HandoffFixture {
        verifier_setup: HachiVerifierSetup<HandoffField>,
        layout: HachiCommitmentLayout,
        commitment: RingCommitment<HandoffField, { HandoffTestConfig::D }>,
        opening_point: Vec<HandoffField>,
        opening: HandoffField,
        proof: HachiProof<HandoffField>,
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct VariableDHandoffTestConfig;

    impl CommitmentConfig for VariableDHandoffTestConfig {
        const D: usize = 256;
        const N_A: usize = 1;
        const N_B: usize = 1;
        const N_D: usize = 1;
        const CHALLENGE_WEIGHT: usize = 23;

        fn decomposition() -> crate::protocol::commitment::DecompositionParams {
            crate::protocol::commitment::Fp128HalvingDCommitmentConfig::decomposition()
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
                    "max_num_vars must leave at least one outer variable".to_string(),
                ));
            }
            let m_vars = reduced_vars.div_ceil(2);
            let r_vars = reduced_vars - m_vars;
            crate::protocol::commitment::HachiCommitmentLayout::new::<Self>(
                m_vars,
                r_vars,
                &Self::decomposition(),
            )
        }

        fn d_at_level(level: usize, _w_num_vars: usize) -> usize {
            match level {
                0 => 256,
                _ => 128,
            }
        }

        fn n_a_at_level(level: usize) -> usize {
            match level {
                0 => 1,
                _ => 2,
            }
        }

        fn challenge_weight_for_ring_dim(d: usize) -> usize {
            match d {
                256 => 23,
                128 => 31,
                _ => panic!("VariableDHandoffTestConfig: unsupported ring dim {d}"),
            }
        }

        fn labrador_handoff_threshold() -> usize {
            0
        }
    }

    fn purge_test_setup_cache(_max_num_vars: usize) {
        #[cfg(feature = "disk-persistence")]
        {
            let cache_dir = std::env::var("LOCALAPPDATA")
                .map(std::path::PathBuf::from)
                .or_else(|_| {
                    std::env::var("HOME").map(|home| {
                        let mut p = std::path::PathBuf::from(&home);
                        if p.join("Library/Caches").exists() {
                            p.push("Library/Caches");
                        } else {
                            p.push(".cache");
                        }
                        p
                    })
                });
            if let Ok(mut path) = cache_dir {
                path.push("hachi");
                path.push(format!("hachi_{_max_num_vars}.setup"));
                let _ = std::fs::remove_file(&path);
            }
        }
    }

    fn make_handoff_fixture(eval_offset: u64, transcript_label: &[u8]) -> HandoffFixture {
        const MAX_NUM_VARS: usize = 11;
        const D: usize = HandoffTestConfig::D;

        let layout = HandoffTestConfig::commitment_layout(MAX_NUM_VARS).unwrap();
        let alpha = D.trailing_zeros() as usize;
        let num_vars = layout.m_vars + layout.r_vars + alpha;

        purge_test_setup_cache(num_vars);

        let len = 1usize << num_vars;
        let evals: Vec<HandoffField> = (0..len)
            .map(|i| HandoffField::from_u64(i as u64 + eval_offset))
            .collect();
        let poly = DensePoly::<HandoffField, D>::from_field_evals(num_vars, &evals).unwrap();

        let setup = <HandoffScheme as CommitmentScheme<HandoffField, D>>::setup_prover(num_vars);
        let verifier_setup =
            <HandoffScheme as CommitmentScheme<HandoffField, D>>::setup_verifier(&setup);

        let (commitment, hint) =
            <HandoffScheme as CommitmentScheme<HandoffField, D>>::commit(&poly, &setup, &layout)
                .unwrap();

        let opening_point: Vec<HandoffField> = (0..num_vars)
            .map(|i| HandoffField::from_u64((i + 2) as u64))
            .collect();
        let lw = lagrange_weights(&opening_point);
        let opening = evals
            .iter()
            .zip(lw.iter())
            .fold(HandoffField::zero(), |a, (&c, &w)| a + c * w);

        let mut prover_transcript = Blake2bTranscript::<HandoffField>::new(transcript_label);
        let proof = <HandoffScheme as CommitmentScheme<HandoffField, D>>::prove(
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

        HandoffFixture {
            verifier_setup,
            layout,
            commitment,
            opening_point,
            opening,
            proof,
        }
    }

    fn handoff_fixture() -> HandoffFixture {
        static FIXTURE: OnceLock<HandoffFixture> = OnceLock::new();
        FIXTURE
            .get_or_init(|| make_handoff_fixture(0, HANDOFF_FIXTURE_LABEL))
            .clone()
    }

    fn handoff_splice_fixture_a() -> HandoffFixture {
        static FIXTURE: OnceLock<HandoffFixture> = OnceLock::new();
        FIXTURE
            .get_or_init(|| make_handoff_fixture(0, HANDOFF_SPLICE_LABEL))
            .clone()
    }

    fn handoff_splice_fixture_b() -> HandoffFixture {
        static FIXTURE: OnceLock<HandoffFixture> = OnceLock::new();
        FIXTURE
            .get_or_init(|| make_handoff_fixture(17, HANDOFF_SPLICE_LABEL))
            .clone()
    }

    fn mutate_labrador_tail_fixture(
        mutator: impl FnOnce(&mut LabradorTail<HandoffField>),
    ) -> Option<HandoffFixture> {
        let mut fixture = handoff_fixture();
        let HachiProofTail::Labrador(tail) = &mut fixture.proof.tail else {
            return None;
        };
        mutator(tail);
        Some(fixture)
    }

    #[test]
    fn labrador_tail_prove_verify_round_trip() {
        let HandoffFixture {
            verifier_setup,
            layout,
            commitment,
            opening_point,
            opening,
            proof,
        } = handoff_fixture();

        let mut verifier_transcript = Blake2bTranscript::<HandoffField>::new(HANDOFF_FIXTURE_LABEL);
        let result =
            <HandoffScheme as CommitmentScheme<HandoffField, { HandoffTestConfig::D }>>::verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                &opening_point,
                &opening,
                &commitment,
                BasisMode::Lagrange,
                &layout,
            );

        assert!(result.is_ok(), "handoff proof should verify: {result:?}");
    }

    #[test]
    fn labrador_tail_serialization_round_trip() {
        let HandoffFixture {
            verifier_setup,
            layout,
            commitment,
            opening_point,
            opening,
            proof,
        } = handoff_fixture();

        let mut bytes = Vec::new();
        proof.serialize_uncompressed(&mut bytes).unwrap();
        assert_eq!(bytes.len(), proof.size());
        assert_eq!(bytes.len(), proof.serialized_size(Compress::No));

        let decoded = HachiProof::<HandoffField>::deserialize_uncompressed(&bytes[..]).unwrap();
        assert_eq!(decoded, proof);

        let mut verifier_transcript = Blake2bTranscript::<HandoffField>::new(HANDOFF_FIXTURE_LABEL);
        let result =
            <HandoffScheme as CommitmentScheme<HandoffField, { HandoffTestConfig::D }>>::verify(
                &decoded,
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
            "round-tripped proof should verify: {result:?}"
        );
    }

    #[test]
    fn labrador_tail_rejects_spliced_or_mutated_payloads() {
        let HandoffFixture {
            verifier_setup,
            layout,
            commitment,
            opening_point,
            opening,
            mut proof,
        } = handoff_splice_fixture_a();
        let proof_b = handoff_splice_fixture_b().proof;

        proof.tail = proof_b.tail.clone();
        let mut verifier_transcript = Blake2bTranscript::<HandoffField>::new(HANDOFF_SPLICE_LABEL);
        let result =
            <HandoffScheme as CommitmentScheme<HandoffField, { HandoffTestConfig::D }>>::verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                &opening_point,
                &opening,
                &commitment,
                BasisMode::Lagrange,
                &layout,
            );
        assert!(result.is_err(), "spliced handoff tail must be rejected");

        let Some(HandoffFixture {
            verifier_setup,
            layout,
            commitment,
            opening_point,
            opening,
            proof,
        }) = mutate_labrador_tail_fixture(|tail| {
            let mut y_ring = tail.y_ring.to_single::<{ HandoffTestConfig::D }>();
            y_ring.coefficients_mut()[0] += HandoffField::one();
            tail.y_ring = FlatRingVec::from_single(&y_ring);
        })
        else {
            return;
        };

        let mut verifier_transcript = Blake2bTranscript::<HandoffField>::new(HANDOFF_FIXTURE_LABEL);
        let result =
            <HandoffScheme as CommitmentScheme<HandoffField, { HandoffTestConfig::D }>>::verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                &opening_point,
                &opening,
                &commitment,
                BasisMode::Lagrange,
                &layout,
            );
        assert!(result.is_err(), "modified y_ring must be rejected");
    }

    #[test]
    fn labrador_tail_rejects_malformed_tail_metadata() {
        let Some(HandoffFixture {
            verifier_setup,
            layout,
            commitment,
            opening_point,
            opening,
            proof,
        }) = mutate_labrador_tail_fixture(|tail| {
            let last_level = tail
                .labrador_proof
                .levels
                .last_mut()
                .expect("tail proof should contain a Labrador level");
            last_level.config.tail = false;
        })
        else {
            return;
        };

        let mut verifier_transcript = Blake2bTranscript::<HandoffField>::new(HANDOFF_FIXTURE_LABEL);
        let result =
            <HandoffScheme as CommitmentScheme<HandoffField, { HandoffTestConfig::D }>>::verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                &opening_point,
                &opening,
                &commitment,
                BasisMode::Lagrange,
                &layout,
            );
        assert!(result.is_err(), "tail/config mismatch must be rejected");

        let Some(HandoffFixture {
            verifier_setup,
            layout,
            commitment,
            opening_point,
            opening,
            proof,
        }) = mutate_labrador_tail_fixture(|tail| {
            let last_level = tail
                .labrador_proof
                .levels
                .last_mut()
                .expect("tail proof should contain a Labrador level");
            last_level.jl_nonce =
                crate::protocol::labrador::guardrails::LABRADOR_MAX_JL_NONCE_RETRIES + 1;
        })
        else {
            return;
        };

        let mut verifier_transcript = Blake2bTranscript::<HandoffField>::new(HANDOFF_FIXTURE_LABEL);
        let result =
            <HandoffScheme as CommitmentScheme<HandoffField, { HandoffTestConfig::D }>>::verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                &opening_point,
                &opening,
                &commitment,
                BasisMode::Lagrange,
                &layout,
            );
        assert!(result.is_err(), "oversized JL nonce must be rejected");

        let Some(HandoffFixture {
            verifier_setup,
            layout,
            commitment,
            opening_point,
            opening,
            proof,
        }) = mutate_labrador_tail_fixture(|tail| {
            let last_level = tail
                .labrador_proof
                .levels
                .last_mut()
                .expect("tail proof should contain a Labrador level");
            last_level.virtual_row_len = 1;
        })
        else {
            return;
        };

        let mut verifier_transcript = Blake2bTranscript::<HandoffField>::new(HANDOFF_FIXTURE_LABEL);
        let result =
            <HandoffScheme as CommitmentScheme<HandoffField, { HandoffTestConfig::D }>>::verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                &opening_point,
                &opening,
                &commitment,
                BasisMode::Lagrange,
                &layout,
            );
        assert!(result.is_err(), "lossy reshape metadata must be rejected");
    }

    #[test]
    fn variable_d_handoff_uses_current_commitment_dimension() {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(|| {
                type VarScheme = HachiCommitmentScheme<
                    { VariableDHandoffTestConfig::D },
                    VariableDHandoffTestConfig,
                >;
                type GF = HandoffField;

                const MAX_NUM_VARS: usize = 10;
                let layout = VariableDHandoffTestConfig::commitment_layout(MAX_NUM_VARS).unwrap();
                let alpha = VariableDHandoffTestConfig::D.trailing_zeros() as usize;
                let num_vars = layout.m_vars + layout.r_vars + alpha;
                purge_test_setup_cache(num_vars);

                let len = 1usize << num_vars;
                let evals: Vec<GF> = (0..len).map(|i| GF::from_u64(i as u64)).collect();
                let poly = DensePoly::<GF, { VariableDHandoffTestConfig::D }>::from_field_evals(
                    num_vars, &evals,
                )
                .unwrap();

                let setup = <VarScheme as CommitmentScheme<
                    GF,
                    { VariableDHandoffTestConfig::D },
                >>::setup_prover(num_vars);
                let (commitment, hint) = <VarScheme as CommitmentScheme<
                    GF,
                    { VariableDHandoffTestConfig::D },
                >>::commit(&poly, &setup, &layout)
                .unwrap();

                let opening_point: Vec<GF> = (0..num_vars)
                    .map(|i| GF::from_u64((i + 2) as u64))
                    .collect();
                let lw = lagrange_weights(&opening_point);
                let _opening = evals
                    .iter()
                    .zip(lw.iter())
                    .fold(GF::zero(), |a, (&c, &w)| a + c * w);

                let mut prover_transcript =
                    Blake2bTranscript::<GF>::new(b"test/variable-d-labrador-tail");
                let proof =
                    <VarScheme as CommitmentScheme<GF, { VariableDHandoffTestConfig::D }>>::prove(
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

                let verifier_setup = <VarScheme as CommitmentScheme<
                    GF,
                    { VariableDHandoffTestConfig::D },
                >>::setup_verifier(&setup);
                let opening = evals
                    .iter()
                    .zip(lw.iter())
                    .fold(GF::zero(), |a, (&c, &w)| a + c * w);
                let mut verifier_transcript =
                    Blake2bTranscript::<GF>::new(b"test/variable-d-labrador-tail");
                <VarScheme as CommitmentScheme<GF, { VariableDHandoffTestConfig::D }>>::verify(
                    &proof,
                    &verifier_setup,
                    &mut verifier_transcript,
                    &opening_point,
                    &opening,
                    &commitment,
                    BasisMode::Lagrange,
                    &layout,
                )
                .unwrap();

                let carried_d = proof
                    .levels
                    .last()
                    .expect("expected at least one Hachi level")
                    .w_commit_d();
                if let HachiProofTail::Labrador(tail) = &proof.tail {
                    assert_eq!(tail.v.ring_dim(), carried_d);
                    assert_ne!(tail.v.ring_dim(), 64);
                }
            })
            .expect("failed to spawn variable-D handoff test")
            .join()
            .expect("variable-D handoff test panicked");
    }
}
