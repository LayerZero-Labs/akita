//! Commitment scheme trait implementation.

use crate::algebra::fields::wide::HasWide;
use crate::algebra::fields::HasUnreducedOps;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::serialization::Valid;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::linear::{flatten_i8_blocks, mat_vec_mul_ntt_single_i8};
use crate::protocol::commitment::utils::ntt_cache::MultiDNttCaches;
use crate::protocol::commitment::{
    hachi_recursive_level_layout_from_params, root_current_w_len, AppendToTranscript,
    CommitmentConfig, CommitmentScheme, HachiCommitmentCore, HachiCommitmentLayout,
    HachiExpandedSetup, HachiLevelParams, HachiProverSetup, HachiScheduleInputs,
    HachiVerifierSetup, RingCommitment, RingCommitmentScheme,
};
use crate::protocol::hachi_poly_ops::{HachiPolyOps, RecursiveWitnessFlat, RecursiveWitnessView};
use crate::protocol::labrador_handoff::{labrador_handoff_prove, labrador_handoff_verify};
use crate::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, BasisMode,
};
use crate::protocol::proof::{
    FlatRingVec, HachiCommitmentHint, HachiLevelProof, HachiProof, HachiProofTail, LabradorTail,
    PackedDigits, ProofRingVec,
};
use crate::protocol::quadratic_equation::{derive_stage1_challenges, QuadraticEquation};
use crate::protocol::recursive_runtime::RecursiveCommitmentHintCache;
use crate::protocol::ring_switch::{
    commit_w, ring_switch_build_w, ring_switch_finalize, ring_switch_verifier,
    w_ring_element_count, RingSwitchOutput, WCommitmentConfig,
};
use crate::protocol::sumcheck::hachi_stage1::{HachiStage1Prover, HachiStage1Verifier};
use crate::protocol::sumcheck::hachi_stage2::{
    relation_claim_from_rows, HachiStage2Prover, HachiStage2Verifier,
};
#[cfg(debug_assertions)]
use crate::protocol::sumcheck::multilinear_eval;
use crate::protocol::sumcheck::{prove_sumcheck, verify_sumcheck, SumcheckInstanceVerifier};
use crate::protocol::transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_SUMCHECK_S_CLAIM, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_ROUND,
};
use crate::protocol::transcript::Transcript;
use crate::{dispatch_ring_dim, dispatch_with_ntt};
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt};
use std::marker::PhantomData;
use std::time::Instant;

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

/// End-to-end PCS wrapper, generic over ring degree `D` and config `Cfg`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HachiCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

/// Runtime state carried between recursive prove levels.
struct RecursiveProverState<F: FieldCore> {
    w: RecursiveWitnessFlat,
    commitment: FlatRingVec<F>,
    hint: RecursiveCommitmentHintCache<F>,
    #[cfg(debug_assertions)]
    w_eval: F,
    sumcheck_challenges: Vec<F>,
    num_u: usize,
    num_l: usize,
}

/// Verifier state carried between recursive levels.
struct RecursiveVerifierState<'a, F: FieldCore> {
    opening_point: Vec<F>,
    opening: F,
    commitment: &'a ProofRingVec<F>,
    basis: BasisMode,
    w_len: usize,
}

/// Output from a single prove level, needed to extend both the proof wire and
/// the recursive prover state.
struct ProveLevelOutput<F: FieldCore> {
    level_proof: HachiLevelProof<F>,
    next_state: RecursiveProverState<F>,
}

/// Prove one fold level: quad_eq -> ring_switch -> sumcheck.
///
/// Generic over the commitment config so it works for both the original
/// polynomial (using `Cfg`) and recursive w-openings (using `WCommitmentConfig`).
type CommitFn<'a, F> = Box<
    dyn FnOnce(
            &RecursiveWitnessFlat,
            HachiScheduleInputs,
        ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), HachiError>
        + 'a,
>;

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_one_level<F, T, const D: usize, Cfg, P>(
    expanded: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    commit_w_fn: CommitFn<'_, F>,
    poly: &P,
    max_num_vars: usize,
    opening_point: &[F],
    hint: HachiCommitmentHint<F, D>,
    transcript: &mut T,
    commitment: &RingCommitment<F, D>,
    basis: BasisMode,
    level: usize,
    level_params: &HachiLevelParams,
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
    let alpha = level_params.d.trailing_zeros() as usize;
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

    let quad_eq = Box::new(QuadraticEquation::<F, { D }, Cfg>::new_prover(
        ntt_shared,
        ring_opening_point,
        poly,
        w_folded,
        level_params.clone(),
        hint,
        transcript,
        commitment,
        &y_ring,
        layout,
    )?);

    finish_prove_level::<F, T, D, Cfg>(
        expanded,
        ntt_shared,
        commit_w_fn,
        max_num_vars,
        transcript,
        &commitment.u,
        level,
        level_params,
        layout,
        quad_eq,
        y_ring,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn finish_prove_level<F, T, const D: usize, Cfg>(
    expanded: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    commit_w_fn: CommitFn<'_, F>,
    max_num_vars: usize,
    transcript: &mut T,
    commitment_u: &[CyclotomicRing<F, D>],
    level: usize,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    mut quad_eq: Box<QuadraticEquation<F, { D }, Cfg>>,
    y_ring: CyclotomicRing<F, D>,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let w = ring_switch_build_w::<F, { D }, Cfg>(
        &mut quad_eq,
        expanded,
        ntt_shared,
        level_params,
        layout,
    )?;
    let next_inputs = HachiScheduleInputs {
        max_num_vars,
        level: level + 1,
        current_w_len: w.len(),
    };

    let (w_commitment_flat, w_hint_cache) = {
        let _span = tracing::info_span!("commit_w_level", level).entered();
        commit_w_fn(&w, next_inputs)?
    };
    let w_commitment_proof = w_commitment_flat.to_proof_ring_vec();

    let rs = ring_switch_finalize::<F, T, { D }, Cfg>(
        &quad_eq,
        expanded,
        transcript,
        w,
        w_commitment_flat,
        &w_commitment_proof,
        w_hint_cache,
        level_params,
        layout,
    )?;

    let relation_claim =
        relation_claim_from_rows::<F, D>(&rs.tau1, rs.alpha, &quad_eq.v, commitment_u, &y_ring);
    let RingSwitchOutput {
        w,
        w_commitment,
        w_hint,
        w_evals_compact,
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
    let w_commitment = w_commitment.expect("prover ring switch must preserve w commitment");
    let (stage1_sumcheck, r_stage1, s_claim) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let mut stage1_prover =
            HachiStage1Prover::new(&w_evals_compact, &tau0, b, live_x_cols, num_u, num_l);
        let (stage1_sumcheck, r_stage1, stage1_final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut stage1_prover, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?;
        let s_claim = stage1_prover.final_s_claim();
        let _ = stage1_final_claim;

        (stage1_sumcheck, r_stage1, s_claim)
    };

    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &s_claim);
    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let (stage2_sumcheck, sumcheck_challenges, _stage2_final_claim, w_eval) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        let mut stage2_prover = HachiStage2Prover::new(
            batching_coeff,
            w_evals_compact,
            &r_stage1,
            s_claim,
            b,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            num_u,
            num_l,
            relation_claim,
        );
        let (stage2_sumcheck, sumcheck_challenges, _stage2_final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut stage2_prover, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?;

        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level).entered();
            stage2_prover.final_w_eval()
        };
        (
            stage2_sumcheck,
            sumcheck_challenges,
            _stage2_final_claim,
            w_eval,
        )
    };

    let (level_proof, sumcheck_challenges) = (
        HachiLevelProof::new_two_stage::<D>(
            y_ring,
            quad_eq.v,
            stage1_sumcheck,
            s_claim,
            stage2_sumcheck,
            w_commitment_proof,
            w_eval,
        ),
        sumcheck_challenges,
    );

    Ok(ProveLevelOutput {
        level_proof,
        next_state: RecursiveProverState {
            w,
            commitment: w_commitment,
            hint: w_hint.expect("prover ring switch must preserve recursive hint cache"),
            #[cfg(debug_assertions)]
            w_eval,
            sumcheck_challenges,
            num_u,
            num_l,
        },
    })
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_one_recursive_level<F, T, const D: usize, Cfg>(
    expanded: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    commit_w_fn: CommitFn<'_, F>,
    witness: &RecursiveWitnessView<'_, F, D>,
    max_num_vars: usize,
    opening_point: &[F],
    hint: HachiCommitmentHint<F, D>,
    transcript: &mut T,
    commitment: &FlatRingVec<F>,
    level: usize,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level,
            "prove_one_recursive_level"
        );
    }
    let alpha = level_params.d.trailing_zeros() as usize;
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
        ring_opening_point_from_field::<F>(
            outer_point,
            layout.r_vars,
            layout.m_vars,
            BasisMode::Lagrange,
        )?
    };

    let fold_scalars = &ring_opening_point.a;
    let eval_outer_scalars = &ring_opening_point.b;
    let (y_ring, w_folded) = {
        let _span = tracing::info_span!(
            "evaluate_and_fold",
            level,
            num_ring_elems = witness.num_ring_elems()
        )
        .entered();
        witness.evaluate_and_fold(eval_outer_scalars, fold_scalars, layout.block_len)
    };

    commitment.append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;
    for pt in &padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);
    let commitment_u = commitment.as_ring_slice::<D>()?;

    let quad_eq = Box::new(QuadraticEquation::<F, { D }, Cfg>::new_recursive_prover(
        ntt_shared,
        ring_opening_point,
        witness,
        w_folded,
        level_params.clone(),
        hint,
        transcript,
        commitment_u,
        &y_ring,
        layout,
    )?);

    finish_prove_level::<F, T, D, Cfg>(
        expanded,
        ntt_shared,
        commit_w_fn,
        max_num_vars,
        transcript,
        commitment_u,
        level,
        level_params,
        layout,
        quad_eq,
        y_ring,
    )
}

/// Whether the prover should stop folding and send `w` directly.
///
/// `prev_w_len` is the polynomial length at the previous level (or the
/// original polynomial's field-element count for level 0).
pub(crate) fn should_stop_folding(w_len: usize, prev_w_len: usize) -> bool {
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
    commit_params: HachiLevelParams,
    commit_ntt_cache: &mut MultiDNttCaches,
    expanded: &HachiExpandedSetup<F>,
    w: &RecursiveWitnessFlat,
) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    let commit_d = commit_params.d;
    dispatch_with_ntt!(
        commit_d,
        commit_ntt_cache,
        expanded,
        |D_COMMIT, ntt_shared| {
            let (wc, wh) = commit_w::<F, { D_COMMIT }, WCommitmentConfig<{ D_COMMIT }, Cfg>>(
                w,
                ntt_shared,
                &commit_params,
            )?;
            Ok((
                FlatRingVec::from_commitment(&wc),
                RecursiveCommitmentHintCache::from_typed(wh)?,
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
    ntt_cache: &mut MultiDNttCaches,
    expanded: &HachiExpandedSetup<F>,
    setup_ntt_shared: &NttSlotCache<D>,
    commit_ntt_cache: &mut MultiDNttCaches,
    max_num_vars: usize,
    current_state: &RecursiveProverState<F>,
    transcript: &mut T,
    level: usize,
    level_params: &HachiLevelParams,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    if level_d == D {
        prove_subsequent_level::<F, T, D, Cfg>(
            expanded,
            setup_ntt_shared,
            commit_ntt_cache,
            max_num_vars,
            current_state,
            transcript,
            level,
            level_params,
        )
    } else {
        dispatch_with_ntt!(level_d, ntt_cache, expanded, |D_LEVEL, ntt_shared| {
            prove_subsequent_level::<F, T, { D_LEVEL }, Cfg>(
                expanded,
                ntt_shared,
                commit_ntt_cache,
                max_num_vars,
                current_state,
                transcript,
                level,
                level_params,
            )
        })
    }
}

/// Dispatch a verify-level operation to the correct ring dimension.
///
/// Each match arm converts the D-erased commitment to a typed one,
/// derives the w-commitment layout, and calls `verify_one_level`.
/// `#[inline(never)]` isolates the monomorphized match arms.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_verify_level<F, T>(
    level_d: usize,
    level_proof: &HachiLevelProof<F>,
    setup: &HachiVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F>,
    is_last: bool,
    final_w: Option<&PackedDigits>,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> Result<Vec<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    dispatch_ring_dim!(level_d, |D_LEVEL| {
        verify_one_level::<F, T, { D_LEVEL }>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            level_params,
            layout,
        )
    })
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_labrador_handoff_prove<F, T, const D: usize, Cfg>(
    current_state: &RecursiveProverState<F>,
    setup: &HachiProverSetup<F, D>,
    handoff_ntt_cache: &mut MultiDNttCaches,
    level_params: &HachiLevelParams,
    w_layout: HachiCommitmentLayout,
    transcript: &mut T,
) -> Result<HachiProofTail<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let handoff_d = current_state.commitment.ring_dim();
    if current_state.hint.ring_dim() != handoff_d {
        return Err(HachiError::InvalidInput(format!(
            "handoff hint/commitment D mismatch: hint={}, commitment={handoff_d}",
            current_state.hint.ring_dim()
        )));
    }

    if handoff_d == D {
        let typed_hint: HachiCommitmentHint<F, D> = current_state.hint.to_typed::<D>()?;
        return labrador_handoff_prove::<F, T, D, Cfg>(
            &current_state.w,
            &typed_hint,
            &current_state.commitment,
            &current_state.sumcheck_challenges,
            current_state.num_u,
            current_state.num_l,
            &setup.expanded,
            &setup.ntt_shared,
            level_params,
            w_layout,
            transcript,
        );
    }

    dispatch_with_ntt!(
        handoff_d,
        handoff_ntt_cache,
        &setup.expanded,
        |D_HANDOFF, ntt_shared| {
            let typed_hint: HachiCommitmentHint<F, { D_HANDOFF }> =
                current_state.hint.to_typed::<{ D_HANDOFF }>()?;
            labrador_handoff_prove::<F, T, { D_HANDOFF }, Cfg>(
                &current_state.w,
                &typed_hint,
                &current_state.commitment,
                &current_state.sumcheck_challenges,
                current_state.num_u,
                current_state.num_l,
                &setup.expanded,
                ntt_shared,
                level_params,
                w_layout,
                transcript,
            )
        }
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_labrador_handoff_verify<F, T>(
    tail: &LabradorTail<F>,
    expanded_setup: &HachiExpandedSetup<F>,
    current_state: &RecursiveVerifierState<'_, F>,
    level_params: &HachiLevelParams,
    w_layout: HachiCommitmentLayout,
    transcript: &mut T,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + Valid,
    T: Transcript<F>,
{
    let handoff_d = level_params.d;
    if !current_state.commitment.can_decode_vec(handoff_d)
        || !tail.v.can_decode_vec(handoff_d)
        || !tail.y_ring.can_decode_single(handoff_d)
        || tail.labrador_proof.levels.iter().any(|level| {
            !level.inner_opening_payload.can_decode_vec(handoff_d)
                || !level.linear_garbage_payload.can_decode_vec(handoff_d)
                || !level.jl_lift_residuals.can_decode_vec(handoff_d)
        })
        || tail
            .labrador_proof
            .final_opening_witness
            .rows
            .iter()
            .any(|row| !row.can_decode_vec(handoff_d))
    {
        return Err(HachiError::InvalidProof);
    }

    dispatch_ring_dim!(handoff_d, |D_HANDOFF| {
        labrador_handoff_verify::<F, T, { D_HANDOFF }>(
            tail,
            &current_state.opening_point,
            &current_state.opening,
            current_state.commitment,
            expanded_setup,
            level_params,
            w_layout,
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
    ntt_shared: &NttSlotCache<D_LEVEL>,
    commit_ntt_cache: &mut MultiDNttCaches,
    max_num_vars: usize,
    current_state: &RecursiveProverState<F>,
    transcript: &mut T,
    level: usize,
    level_params: &HachiLevelParams,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let _setup_span = tracing::info_span!("inter_level_setup", level).entered();

    let current_w = &current_state.w;
    let w_view = current_w.view::<F, { D_LEVEL }>()?;
    let opening_point = next_level_opening_point(
        &current_state.sumcheck_challenges,
        current_state.num_u,
        current_state.num_l,
    );

    #[cfg(debug_assertions)]
    {
        let mut field_evals: Vec<F> = current_w
            .as_i8_digits()
            .iter()
            .map(|&d| F::from_i8(d))
            .collect();
        field_evals.resize(w_view.num_ring_elems() * D_LEVEL, F::zero());
        let direct_eval = multilinear_eval(&field_evals, &opening_point).unwrap();
        if current_state.w_eval != direct_eval {
            tracing::error!(
                level,
                ring_elems = w_view.num_ring_elems(),
                field_len = field_evals.len(),
                point_len = opening_point.len(),
                "BUG: w_eval mismatch! prev_level w_eval != w_poly eval at opening_point"
            );
        } else {
            tracing::debug!(level, "w_eval consistency OK");
        }
    }

    let w_layout = hachi_recursive_level_layout_from_params::<Cfg>(level_params, current_w.len())?;
    let typed_hint: HachiCommitmentHint<F, { D_LEVEL }> =
        current_state.hint.to_typed::<{ D_LEVEL }>()?;
    drop(_setup_span);

    let commit_fn: CommitFn<'_, F> = Box::new(
        |w: &RecursiveWitnessFlat,
         next_inputs: HachiScheduleInputs|
         -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), HachiError> {
            let next_params = Cfg::level_params(next_inputs);
            if next_params.d == D_LEVEL {
                let (wc, wh) = commit_w::<F, { D_LEVEL }, WCommitmentConfig<{ D_LEVEL }, Cfg>>(
                    w,
                    ntt_shared,
                    &next_params,
                )?;
                Ok((
                    FlatRingVec::from_commitment(&wc),
                    RecursiveCommitmentHintCache::from_typed(wh)?,
                ))
            } else {
                dispatch_commit::<F, Cfg>(next_params, commit_ntt_cache, expanded, w)
            }
        },
    );

    prove_one_recursive_level::<F, T, { D_LEVEL }, WCommitmentConfig<{ D_LEVEL }, Cfg>>(
        expanded,
        ntt_shared,
        commit_fn,
        &w_view,
        max_num_vars,
        &opening_point,
        typed_hint,
        transcript,
        &current_state.commitment,
        level,
        level_params,
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
        let root_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: setup.expanded.seed.max_num_vars,
            level: 0,
            current_w_len: root_current_w_len::<D>(*layout),
        });
        let mut inner = poly.commit_inner_witness(
            &setup.expanded.shared_matrix,
            &setup.ntt_shared,
            layout.block_len,
            layout.num_digits_commit,
            layout.num_digits_open,
            layout.log_basis,
        )?;
        for t_i in &mut inner.t {
            t_i.truncate(root_params.n_a);
        }
        for t_hat_i in &mut inner.t_hat {
            t_hat_i.truncate(root_params.n_a * layout.num_digits_open);
        }
        let inner_opening_digits_flat = flatten_i8_blocks(&inner.t_hat);
        let mut u: Vec<CyclotomicRing<F, D>> =
            mat_vec_mul_ntt_single_i8(&setup.ntt_shared, &inner_opening_digits_flat);
        u.truncate(root_params.n_b);
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

        let mut ntt_cache = MultiDNttCaches::new();
        let mut commit_ntt_cache = MultiDNttCaches::new();
        let max_num_vars = setup.expanded.seed.max_num_vars;
        let root_w_len = root_current_w_len::<D>(*layout);
        let root_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars,
            level: 0,
            current_w_len: root_w_len,
        });

        // Level 0: original polynomial with caller-provided layout.
        // The w-commitment is produced at the next level's params, derived from
        // public state once `w` has been built.
        let commit_fn_0: CommitFn<'_, F> = Box::new(
            |w: &RecursiveWitnessFlat,
             next_inputs: HachiScheduleInputs|
             -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), HachiError> {
                let next_params = Cfg::level_params(next_inputs);
                if next_params.d == D {
                    let (wc, wh) = commit_w::<F, D, Cfg>(w, &setup.ntt_shared, &next_params)?;
                    Ok((
                        FlatRingVec::from_commitment(&wc),
                        RecursiveCommitmentHintCache::from_typed(wh)?,
                    ))
                } else {
                    dispatch_commit::<F, Cfg>(
                        next_params,
                        &mut commit_ntt_cache,
                        &setup.expanded,
                        w,
                    )
                }
            },
        );
        let out = prove_one_level::<F, T, D, Cfg, P>(
            &setup.expanded,
            &setup.ntt_shared,
            commit_fn_0,
            poly,
            max_num_vars,
            opening_point,
            hint,
            transcript,
            commitment,
            basis,
            0,
            &root_params,
            *layout,
        )?;
        levels.push(out.level_proof);

        let mut prev_poly_len = root_w_len;
        let mut current_state = out.next_state;
        let mut level = 1usize;
        let planned_num_levels = Cfg::schedule_plan(max_num_vars)?.map(|plan| plan.levels.len());

        // Subsequent levels: recursive w-opening with WCommitmentConfig.
        // Each level dispatches to the ring dimension from Cfg::d_at_level.
        // The w-commitment is produced at the NEXT level's D.
        loop {
            let should_continue = if let Some(num_levels) = planned_num_levels {
                level < num_levels
            } else {
                !should_stop_folding(current_state.w.len(), prev_poly_len)
            };
            if !should_continue {
                break;
            }
            let level_params = Cfg::level_params(HachiScheduleInputs {
                max_num_vars,
                level,
                current_w_len: current_state.w.len(),
            });
            let level_d = level_params.d;

            let out = dispatch_prove_level::<F, T, D, Cfg>(
                level_d,
                &mut ntt_cache,
                &setup.expanded,
                &setup.ntt_shared,
                &mut commit_ntt_cache,
                max_num_vars,
                &current_state,
                transcript,
                level,
                &level_params,
            )?;

            levels.push(out.level_proof);

            prev_poly_len = current_state.w.len();
            current_state = out.next_state;
            level += 1;
        }

        tracing::info!(
            levels = level,
            elapsed_s = t_prove_total.elapsed().as_secs_f64(),
            "hachi prove complete"
        );

        let labrador_enabled = current_state.w.len() > Cfg::labrador_handoff_threshold()
            && std::env::var("HACHI_NO_LABRADOR").as_deref() != Ok("1");
        let final_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars,
            level,
            current_w_len: current_state.w.len(),
        });
        let tail = if labrador_enabled {
            tracing::info!("labrador handoff started");
            let handoff_layout = hachi_recursive_level_layout_from_params::<Cfg>(
                &final_params,
                current_state.w.len(),
            )?;
            dispatch_labrador_handoff_prove::<F, T, D, Cfg>(
                &current_state,
                setup,
                &mut commit_ntt_cache,
                &final_params,
                handoff_layout,
                transcript,
            )?
        } else {
            let final_w = PackedDigits::from_i8_digits_with_min_bits(
                current_state.w.as_i8_digits(),
                final_params.log_basis,
            );
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
        let t_verify_hachi = Instant::now();

        let num_levels = proof.levels.len();
        let has_handoff_tail = proof.has_handoff_tail();

        let final_w = proof.final_w();

        // State carried between levels.
        // Commitment is D-erased so the loop can handle varying D per level.
        let root_commitment = ProofRingVec::from_ring_elems(&commitment.u);
        let mut current_state = RecursiveVerifierState {
            opening_point: opening_point.to_vec(),
            opening: *opening,
            commitment: &root_commitment,
            basis,
            w_len: layout.num_blocks * layout.block_len * D,
        };
        let max_num_vars = setup.expanded.seed.max_num_vars;
        if let Some(plan) = Cfg::schedule_plan(max_num_vars)? {
            if num_levels != plan.levels.len() {
                return Err(HachiError::InvalidProof);
            }
        }

        for i in 0..num_levels {
            let level_proof = &proof.levels[i];
            let is_last_hachi = i == num_levels - 1;
            // With a handoff tail, the last Hachi level is NOT the
            // final level -- verification continues in Labrador.
            let is_last = is_last_hachi && !has_handoff_tail;
            let level_params = Cfg::level_params(HachiScheduleInputs {
                max_num_vars,
                level: i,
                current_w_len: current_state.w_len,
            });
            let level_d = level_params.d;
            let current_layout = if i == 0 {
                *layout
            } else {
                hachi_recursive_level_layout_from_params::<Cfg>(&level_params, current_state.w_len)?
            };
            if !current_state.commitment.can_decode_vec(level_d)
                || !level_proof.y_ring.can_decode_single(level_d)
                || !level_proof.v.can_decode_vec(level_d)
            {
                return Err(HachiError::InvalidProof);
            }
            tracing::debug!(
                level = i,
                is_last,
                point_len = current_state.opening_point.len(),
                D = level_d,
                "verify level"
            );

            let challenges = if level_d == D {
                verify_one_level::<F, T, D>(
                    level_proof,
                    setup,
                    transcript,
                    &current_state,
                    is_last,
                    if is_last { final_w } else { None },
                    &level_params,
                    current_layout,
                )?
            } else {
                dispatch_verify_level::<F, T>(
                    level_d,
                    level_proof,
                    setup,
                    transcript,
                    &current_state,
                    is_last,
                    if is_last { final_w } else { None },
                    &level_params,
                    current_layout,
                )?
            };

            if !is_last {
                let alpha_bits = level_d.trailing_zeros() as usize;
                let num_l = alpha_bits;
                let num_u = challenges.len() - num_l;
                let next_w_len = w_ring_element_count::<F>(&level_params, current_layout) * level_d;

                if i + 1 < num_levels {
                    let next_level_d = Cfg::level_params(HachiScheduleInputs {
                        max_num_vars,
                        level: i + 1,
                        current_w_len: next_w_len,
                    })
                    .d;
                    if !level_proof.next_w_commitment().can_decode_vec(next_level_d) {
                        return Err(HachiError::InvalidProof);
                    }
                }
                current_state = RecursiveVerifierState {
                    opening_point: next_level_opening_point(&challenges, num_u, num_l),
                    opening: level_proof.next_w_eval(),
                    commitment: level_proof.next_w_commitment(),
                    basis: BasisMode::Lagrange,
                    w_len: next_w_len,
                };
            }
        }

        tracing::info!(
            levels = num_levels,
            elapsed_s = t_verify_hachi.elapsed().as_secs_f64(),
            "hachi verify complete"
        );

        match &proof.tail {
            HachiProofTail::Labrador(ref tail) => {
                let handoff_params = Cfg::level_params(HachiScheduleInputs {
                    max_num_vars,
                    level: num_levels,
                    current_w_len: current_state.w_len,
                });
                let handoff_layout = hachi_recursive_level_layout_from_params::<Cfg>(
                    &handoff_params,
                    current_state.w_len,
                )?;
                if handoff_params.d == D {
                    labrador_handoff_verify::<F, T, D>(
                        tail,
                        &current_state.opening_point,
                        &current_state.opening,
                        current_state.commitment,
                        &setup.expanded,
                        &handoff_params,
                        handoff_layout,
                        transcript,
                    )?;
                } else {
                    dispatch_labrador_handoff_verify::<F, T>(
                        tail,
                        &setup.expanded,
                        &current_state,
                        &handoff_params,
                        handoff_layout,
                        transcript,
                    )?;
                }
            }
            HachiProofTail::Direct(_) => {}
        }

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Hachi"
    }
}

/// Verify one fold level.
///
/// At the final level, `final_w` is provided and the verifier checks w_val
/// from it directly. At intermediate levels, `level_proof.next_w_eval()` is used.
///
/// Returns the sumcheck challenges for chaining into the next level.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[tracing::instrument(skip_all, name = "verify_one_level")]
fn verify_one_level<F, T, const D: usize>(
    level_proof: &HachiLevelProof<F>,
    setup: &HachiVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F>,
    is_last: bool,
    final_w: Option<&PackedDigits>,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> Result<Vec<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    let y_ring = level_proof.y_ring.as_single_ring::<D>()?;
    let v_typed = level_proof.v.as_ring_slice::<D>()?;
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;

    let alpha_bits = level_params.d.trailing_zeros() as usize;
    if current_state.opening_point.len() < alpha_bits {
        return Err(HachiError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    let target_num_vars = layout.m_vars + layout.r_vars + alpha_bits;
    let mut padded_point = current_state.opening_point.clone();
    padded_point.resize(target_num_vars, F::zero());
    let inner_point = &padded_point[..alpha_bits];
    let reduced_opening_point = &padded_point[alpha_bits..];

    current_state
        .commitment
        .append_as_ring_slice::<T, D>(ABSORB_COMMITMENT, transcript)?;
    for pt in &padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);

    let v = reduce_inner_opening_to_ring_element::<F, { D }>(inner_point, current_state.basis)?;
    let d = F::from_u64(level_params.d as u64);
    let trace_lhs = trace::<F, { D }>(&(*y_ring * v.sigma_m1()));
    let trace_rhs = d * current_state.opening;
    if trace_lhs != trace_rhs {
        return Err(HachiError::InvalidProof);
    }

    let ring_opening_point = ring_opening_point_from_field::<F>(
        reduced_opening_point,
        layout.r_vars,
        layout.m_vars,
        current_state.basis,
    )?;
    let stage1_challenges =
        derive_stage1_challenges::<F, T, D>(transcript, v_typed, layout.num_blocks, level_params)?;

    let w_len = if is_last {
        final_w.map_or(0, |fw| fw.num_elems)
    } else {
        w_ring_element_count::<F>(level_params, layout) * D
    };
    tracing::debug!(w_len, is_last, "verify ring_switch");

    let rs = ring_switch_verifier::<F, T, { D }>(
        &ring_opening_point,
        &stage1_challenges,
        &setup.expanded,
        w_len,
        level_proof.next_w_commitment(),
        transcript,
        level_params,
        layout,
    )?;
    let relation_claim =
        relation_claim_from_rows(&rs.tau1, rs.alpha, v_typed, commitment_u, y_ring);
    let stage1 = &level_proof.stage1;
    let stage2 = &level_proof.stage2;
    let stage1_verifier = HachiStage1Verifier::new(rs.tau0.clone(), stage1.s_claim, rs.b);
    let r_stage1 = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        verify_sumcheck::<F, _, F, _, _>(&stage1.sumcheck, &stage1_verifier, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
        })?
    };

    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let stage2_input_claim = batching_coeff * stage1.s_claim + relation_claim;

    let stage2_verifier = if is_last {
        let fw = final_w.ok_or(HachiError::InvalidProof)?;
        HachiStage2Verifier::new_with_packed_witness(
            batching_coeff,
            stage1.s_claim,
            fw,
            r_stage1.clone(),
            rs.alpha_evals_y,
            rs.m_evals_x,
            &rs.tau1,
            v_typed,
            commitment_u,
            y_ring,
            rs.alpha,
            rs.num_u,
            rs.num_l,
        )
    } else {
        HachiStage2Verifier::new_with_claimed_w_eval(
            batching_coeff,
            stage1.s_claim,
            stage2.next_w_eval,
            r_stage1.clone(),
            rs.alpha_evals_y,
            rs.m_evals_x,
            &rs.tau1,
            v_typed,
            commitment_u,
            y_ring,
            rs.alpha,
            rs.num_u,
            rs.num_l,
        )
    };
    if stage2_input_claim != SumcheckInstanceVerifier::input_claim(&stage2_verifier) {
        return Err(HachiError::InvalidProof);
    }

    let challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        verify_sumcheck::<F, _, F, _, _>(&stage2.sumcheck, &stage2_verifier, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
        })?
    };

    Ok(challenges)
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

    fn make_verify_fixture(
        num_vars: usize,
    ) -> (
        HachiVerifierSetup<F>,
        RingCommitment<F, D>,
        HachiProof<F>,
        Vec<F>,
        F,
        HachiCommitmentLayout,
    ) {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(num_vars).unwrap();
        let full_num_vars = layout.m_vars + layout.r_vars + alpha;

        let (poly, evals) = make_dense_poly(full_num_vars);
        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(full_num_vars);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentScheme<F, D>>::commit(&poly, &setup, &layout).unwrap();

        let opening_point: Vec<F> = (0..full_num_vars)
            .map(|i| F::from_u64((i + 2) as u64))
            .collect();
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

        (
            verifier_setup,
            commitment,
            proof,
            opening_point,
            opening,
            layout,
        )
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
    fn verify_rejects_malformed_y_ring_dimension_without_panicking() {
        let (verifier_setup, commitment, mut proof, opening_point, opening, layout) =
            make_verify_fixture(16);
        let mut coeffs = proof.levels[0].y_ring.coeffs().to_vec();
        let _ = coeffs.pop().expect("expected non-empty y_ring");
        proof.levels[0].y_ring = ProofRingVec::from_coeffs(coeffs);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
            <Scheme as CommitmentScheme<F, D>>::verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                &opening_point,
                &opening,
                &commitment,
                BasisMode::Lagrange,
                &layout,
            )
        }));

        assert!(matches!(result, Ok(Err(HachiError::InvalidProof))));
    }

    #[test]
    fn verify_rejects_malformed_next_commitment_dimension_without_panicking() {
        let HandoffFixture {
            verifier_setup,
            commitment,
            mut proof,
            opening_point,
            opening,
            layout,
        } = handoff_fixture();
        let mut coeffs = proof.levels[0].stage2.next_w_commitment.coeffs().to_vec();
        let _ = coeffs.pop().expect("expected non-empty next commitment");
        proof.levels[0].stage2.next_w_commitment = ProofRingVec::from_coeffs(coeffs);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut verifier_transcript =
                Blake2bTranscript::<HandoffField>::new(HANDOFF_FIXTURE_LABEL);
            <HandoffScheme as CommitmentScheme<HandoffField, { HandoffTestConfig::D }>>::verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                &opening_point,
                &opening,
                &commitment,
                BasisMode::Lagrange,
                &layout,
            )
        }));

        assert!(matches!(result, Ok(Err(HachiError::InvalidProof))));
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

        fn decomposition() -> crate::protocol::commitment::DecompositionParams {
            crate::protocol::commitment::DecompositionParams {
                log_basis: 3,
                log_commit_bound: 32,
                log_open_bound: Some(128),
            }
        }

        fn envelope(_max_num_vars: usize) -> crate::protocol::commitment::CommitmentEnvelope {
            crate::protocol::commitment::CommitmentEnvelope {
                max_n_a: 8,
                max_n_b: 4,
                max_n_d: 4,
            }
        }

        fn stage1_challenge_config(d: usize) -> crate::algebra::SparseChallengeConfig {
            assert_eq!(d, Self::D, "unsupported ring dim {d}");
            crate::algebra::SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
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
    struct DenseHandoffTestConfig;

    impl CommitmentConfig for DenseHandoffTestConfig {
        const D: usize = crate::protocol::commitment::Fp128FullCommitmentConfig::D;

        fn decomposition() -> crate::protocol::commitment::DecompositionParams {
            crate::protocol::commitment::Fp128FullCommitmentConfig::decomposition()
        }

        fn envelope(max_num_vars: usize) -> crate::protocol::commitment::CommitmentEnvelope {
            crate::protocol::commitment::Fp128FullCommitmentConfig::envelope(max_num_vars)
        }

        fn commitment_layout(
            max_num_vars: usize,
        ) -> Result<crate::protocol::commitment::HachiCommitmentLayout, crate::error::HachiError>
        {
            crate::protocol::commitment::Fp128FullCommitmentConfig::commitment_layout(max_num_vars)
        }

        fn stage1_challenge_config(d: usize) -> crate::algebra::SparseChallengeConfig {
            crate::protocol::commitment::Fp128FullCommitmentConfig::stage1_challenge_config(d)
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
            tail.y_ring = ProofRingVec::from_single(&y_ring);
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
    fn dense_handoff_uses_current_commitment_dimension() {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(|| {
                type DenseScheme =
                    HachiCommitmentScheme<{ DenseHandoffTestConfig::D }, DenseHandoffTestConfig>;
                type GF = HandoffField;

                const MAX_NUM_VARS: usize = 10;
                let layout = DenseHandoffTestConfig::commitment_layout(MAX_NUM_VARS).unwrap();
                let alpha = DenseHandoffTestConfig::D.trailing_zeros() as usize;
                let num_vars = layout.m_vars + layout.r_vars + alpha;
                purge_test_setup_cache(num_vars);

                let len = 1usize << num_vars;
                let evals: Vec<GF> = (0..len).map(|i| GF::from_u64(i as u64)).collect();
                let poly = DensePoly::<GF, { DenseHandoffTestConfig::D }>::from_field_evals(
                    num_vars, &evals,
                )
                .unwrap();

                let setup = <DenseScheme as CommitmentScheme<
                    GF,
                    { DenseHandoffTestConfig::D },
                >>::setup_prover(num_vars);
                let (commitment, hint) = <DenseScheme as CommitmentScheme<
                    GF,
                    { DenseHandoffTestConfig::D },
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
                    Blake2bTranscript::<GF>::new(b"test/dense-labrador-tail");
                let proof =
                    <DenseScheme as CommitmentScheme<GF, { DenseHandoffTestConfig::D }>>::prove(
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

                let verifier_setup = <DenseScheme as CommitmentScheme<
                    GF,
                    { DenseHandoffTestConfig::D },
                >>::setup_verifier(&setup);
                let opening = evals
                    .iter()
                    .zip(lw.iter())
                    .fold(GF::zero(), |a, (&c, &w)| a + c * w);
                let mut verifier_transcript =
                    Blake2bTranscript::<GF>::new(b"test/dense-labrador-tail");
                <DenseScheme as CommitmentScheme<GF, { DenseHandoffTestConfig::D }>>::verify(
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

                let carried_d = DenseHandoffTestConfig::D;
                if let HachiProofTail::Labrador(tail) = &proof.tail {
                    assert_eq!(tail.v.coeff_len(), carried_d);
                    assert_eq!(tail.v.coeff_len(), DenseHandoffTestConfig::D);
                    assert_ne!(tail.v.coeff_len(), 64);
                    assert!(proof
                        .levels
                        .last()
                        .expect("expected at least one Hachi level")
                        .next_w_commitment()
                        .can_decode_vec(carried_d));
                }
            })
            .expect("failed to spawn dense handoff test")
            .join()
            .expect("dense handoff test panicked");
    }
}
