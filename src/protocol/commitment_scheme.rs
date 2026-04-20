//! Commitment scheme trait implementation.

use crate::algebra::fields::wide::HasWide;
use crate::algebra::fields::HasUnreducedOps;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::primitives::serialization::Valid;
use crate::protocol::commitment::utils::crt_ntt::{build_ntt_slot, NttSlotCache};
use crate::protocol::commitment::utils::linear::mat_vec_mul_ntt_single_i8;
use crate::protocol::commitment::utils::ntt_cache::MultiDNttCaches;
use crate::protocol::commitment::{
    exact_planned_level_execution, hachi_batched_root_layout,
    hachi_recursive_level_layout_from_params, hachi_root_runtime_plan,
    hachi_root_runtime_plan_with_batch, packed_digits_bytes,
    planned_next_log_basis_with_current_basis_and_envelope,
    planned_recursive_suffix_bytes_with_log_basis_and_envelope, AppendToTranscript,
    CommitmentConfig, CommitmentScheme, HachiBatchPlanningEnvelope, HachiRootBatchSummary,
    HachiScheduleInputs, HachiScheduleLookupKey, RingCommitment,
};
#[cfg(test)]
use crate::protocol::commitment::{root_current_w_len, scale_batched_root_layout};
use crate::protocol::hachi_poly_ops::{
    DensePoly, HachiPolyOps, RecursiveWitnessFlat, RecursiveWitnessView,
};
use crate::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, BasisMode, BlockOrder,
    RingOpeningPoint,
};
use crate::protocol::params::LevelParams;
use crate::protocol::proof::{
    DirectWitnessProof, FlatRingVec, HachiBatchedCommitmentHint, HachiBatchedProof,
    HachiBatchedRootProof, HachiCommitmentHint, HachiLevelProof, HachiProof, HachiProofStep,
    PackedDigits,
};
use crate::protocol::quadratic_equation::{derive_stage1_challenges, QuadraticEquation};
use crate::protocol::recursive_runtime::RecursiveCommitmentHintCache;
use crate::protocol::ring_switch::{
    commit_w, ring_switch_build_w, ring_switch_finalize, ring_switch_finalize_with_claim_groups,
    ring_switch_verifier, ring_switch_verifier_with_claim_groups,
    ring_switch_verifier_with_opening_points_and_claim_groups, w_ring_element_count,
    w_ring_element_count_with_claim_groups, RingSwitchOutput, WCommitmentConfig,
};
use crate::protocol::setup::{HachiExpandedSetup, HachiProverSetup, HachiVerifierSetup};
use crate::protocol::sumcheck::hachi_stage1_tree::{HachiStage1Prover, HachiStage1Verifier};
use crate::protocol::sumcheck::hachi_stage2::{
    relation_claim_from_rows, HachiStage2Prover, HachiStage2Verifier, Stage2MEvalSource,
};
use crate::protocol::sumcheck::{prove_sumcheck, verify_sumcheck, SumcheckInstanceVerifier};
use crate::protocol::transcript::labels::{
    ABSORB_BATCH_SHAPE, ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_EVAL_OPENINGS_FIELD,
    ABSORB_SUMCHECK_S_CLAIM, CHALLENGE_EVAL_BATCH, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_ROUND,
};
use crate::protocol::transcript::Transcript;
use crate::{dispatch_ring_dim, dispatch_with_ntt};
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt};
use std::marker::PhantomData;
use std::time::Instant;

#[cfg(test)]
use crate::protocol::ring_switch::w_ring_element_count_with_num_claims;
#[cfg(test)]
use crate::HachiSerialize;

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
    log_basis: u32,
    root_key: HachiScheduleLookupKey,
    planning_envelope: HachiBatchPlanningEnvelope,
    sumcheck_challenges: Vec<F>,
}

/// Verifier state carried between recursive levels.
struct RecursiveVerifierState<'a, F: FieldCore> {
    opening_point: Vec<F>,
    opening: F,
    commitment: &'a FlatRingVec<F>,
    basis: BasisMode,
    w_len: usize,
    log_basis: u32,
    root_key: HachiScheduleLookupKey,
    planning_envelope: HachiBatchPlanningEnvelope,
}

/// Output from a single prove level, needed to extend both the proof wire and
/// the recursive prover state.
struct ProveLevelOutput<F: FieldCore> {
    level_proof: HachiLevelProof<F>,
    next_state: RecursiveProverState<F>,
}

/// Output from the specialized batched root prove level.
struct BatchedProveLevelOutput<F: FieldCore> {
    root_proof: HachiBatchedRootProof<F>,
    next_state: RecursiveProverState<F>,
}

pub(crate) fn reorder_stage1_coords<F: FieldCore>(
    coords: &[F],
    col_bits: usize,
    ring_bits: usize,
) -> Vec<F> {
    assert_eq!(coords.len(), col_bits + ring_bits);
    let mut reordered = Vec::with_capacity(coords.len());
    reordered.extend_from_slice(&coords[col_bits..]);
    reordered.extend_from_slice(&coords[..col_bits]);
    reordered
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MultiPointBatchShape {
    point_group_sizes: Vec<usize>,
    claim_group_sizes: Vec<usize>,
    claim_to_point: Vec<usize>,
}

#[derive(Debug, Clone)]
struct PreparedRootOpeningPoint<F: FieldCore, const D: usize> {
    padded_point: Vec<F>,
    ring_opening_point: RingOpeningPoint<F>,
    inner_reduction: CyclotomicRing<F, D>,
}

fn flatten_batched_commitment_rows<F: FieldCore, const D: usize>(
    commitments: &[RingCommitment<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    commitments
        .iter()
        .flat_map(|commitment| commitment.u.iter().copied())
        .collect()
}

fn append_batched_commitments_to_transcript<F, T, const D: usize>(
    commitments: &[RingCommitment<F, D>],
    transcript: &mut T,
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    for commitment in commitments {
        commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
    }
}

fn validate_nonempty_group_sizes<T>(
    groups: &[&[T]],
    label: &str,
) -> Result<Vec<usize>, HachiError> {
    if groups.is_empty() {
        return Err(HachiError::InvalidInput(format!(
            "{label} requires at least one group"
        )));
    }
    let mut out = Vec::with_capacity(groups.len());
    for (group_idx, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(HachiError::InvalidInput(format!(
                "{label} group {group_idx} must be nonempty"
            )));
        }
        out.push(group.len());
    }
    Ok(out)
}

fn checked_total_claims(group_sizes: &[usize], label: &str) -> Result<usize, HachiError> {
    group_sizes.iter().try_fold(0usize, |acc, &group_size| {
        acc.checked_add(group_size)
            .ok_or_else(|| HachiError::InvalidInput(format!("{label} total claim count overflow")))
    })
}

fn flatten_poly_groups<'a, P>(poly_groups: &[&'a [P]]) -> Vec<&'a P> {
    poly_groups.iter().flat_map(|group| group.iter()).collect()
}

fn flatten_value_groups<T: Clone>(groups: &[&[T]]) -> Vec<T> {
    groups
        .iter()
        .flat_map(|group| group.iter().cloned())
        .collect()
}

fn validate_nonempty_group_sizes_by_point<T>(
    groups_by_point: &[&[&[T]]],
    label: &str,
) -> Result<MultiPointBatchShape, HachiError> {
    if groups_by_point.is_empty() {
        return Err(HachiError::InvalidInput(format!(
            "{label} requires at least one opening point"
        )));
    }
    let mut point_group_sizes = Vec::with_capacity(groups_by_point.len());
    let mut claim_group_sizes = Vec::new();
    let mut claim_to_point = Vec::new();
    for (point_idx, groups) in groups_by_point.iter().enumerate() {
        if groups.is_empty() {
            return Err(HachiError::InvalidInput(format!(
                "{label} point {point_idx} must have at least one group"
            )));
        }
        point_group_sizes.push(groups.len());
        for (group_idx, group) in groups.iter().enumerate() {
            if group.is_empty() {
                return Err(HachiError::InvalidInput(format!(
                    "{label} point {point_idx} group {group_idx} must be nonempty"
                )));
            }
            claim_group_sizes.push(group.len());
            claim_to_point.extend(std::iter::repeat_n(point_idx, group.len()));
        }
    }
    Ok(MultiPointBatchShape {
        point_group_sizes,
        claim_group_sizes,
        claim_to_point,
    })
}

fn flatten_poly_groups_by_point<'a, P>(poly_groups_by_point: &[&'a [&'a [P]]]) -> Vec<&'a P> {
    poly_groups_by_point
        .iter()
        .flat_map(|groups| groups.iter().flat_map(|group| group.iter()))
        .collect()
}

fn flatten_value_groups_by_point<T: Clone>(groups_by_point: &[&[&[T]]]) -> Vec<T> {
    groups_by_point
        .iter()
        .flat_map(|groups| groups.iter().flat_map(|group| group.iter().cloned()))
        .collect()
}

fn flatten_commitments_by_point<C: Clone>(commitments_by_point: &[&[C]]) -> Vec<C> {
    commitments_by_point
        .iter()
        .flat_map(|commitments| commitments.iter().cloned())
        .collect()
}

fn flatten_batched_hints_by_point<F: FieldCore, const D: usize>(
    hints_by_point: Vec<Vec<HachiBatchedCommitmentHint<F, D>>>,
) -> Vec<HachiBatchedCommitmentHint<F, D>> {
    hints_by_point.into_iter().flatten().collect()
}

fn single_prove_hint_from_group_hint<F: FieldCore, const D: usize>(
    hint: HachiBatchedCommitmentHint<F, D>,
) -> Result<HachiCommitmentHint<F, D>, HachiError> {
    if hint.inner_opening_digits.len() != 1
        || hint.t().is_some_and(|rows_by_poly| rows_by_poly.len() != 1)
    {
        return Err(HachiError::InvalidInput(
            "prove requires a commitment hint for exactly one polynomial".to_string(),
        ));
    }
    Ok(hint.into_flattened())
}

fn append_batch_shape_to_transcript<F, T>(
    point_group_sizes: &[usize],
    claim_group_sizes: &[usize],
    transcript: &mut T,
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_BATCH_SHAPE, &point_group_sizes.len());
    for group_count in point_group_sizes {
        transcript.append_serde(ABSORB_BATCH_SHAPE, group_count);
    }
    for claim_count in claim_group_sizes {
        transcript.append_serde(ABSORB_BATCH_SHAPE, claim_count);
    }
}

fn prepare_root_opening_point<F, const D: usize>(
    opening_point: &[F],
    basis: BasisMode,
    lp: &LevelParams,
    alpha_bits: usize,
) -> Result<PreparedRootOpeningPoint<F, D>, HachiError>
where
    F: FieldCore,
{
    let target_num_vars = lp
        .m_vars
        .checked_add(lp.r_vars)
        .and_then(|n| n.checked_add(alpha_bits))
        .ok_or_else(|| HachiError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() > target_num_vars {
        return Err(HachiError::InvalidPointDimension {
            expected: target_num_vars,
            actual: opening_point.len(),
        });
    }
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let inner_point = &padded_point[..alpha_bits];
    let outer_point = &padded_point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field::<F>(
        outer_point,
        lp.r_vars,
        lp.m_vars,
        basis,
        BlockOrder::RowMajor,
    )?;
    let inner_reduction = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis)?;
    Ok(PreparedRootOpeningPoint {
        padded_point,
        ring_opening_point,
        inner_reduction,
    })
}

fn schedule_uses_root_direct<Cfg: CommitmentConfig>(
    max_num_vars: usize,
) -> Result<bool, HachiError> {
    let key = HachiScheduleLookupKey::singleton(max_num_vars, max_num_vars, 1);
    Ok(Cfg::schedule_plan(key)?
        .as_ref()
        .is_some_and(|plan| plan.num_fold_levels() == 0))
}

fn root_direct_field_witness<F: FieldCore>(
    direct_witness: &DirectWitnessProof<F>,
) -> Result<&FlatRingVec<F>, HachiError> {
    direct_witness
        .as_field_elements()
        .ok_or(HachiError::InvalidProof)
}

fn root_direct_opening_matches<F, const D: usize, Cfg>(
    direct_witness: &DirectWitnessProof<F>,
    opening_point: &[F],
    opening: &F,
    basis: BasisMode,
) -> Result<bool, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig<Field = F>,
{
    let field_witness = root_direct_field_witness(direct_witness)?;
    let poly = DensePoly::<F, D>::from_field_evals(opening_point.len(), field_witness.coeffs())?;
    let root_lp = hachi_batched_root_layout::<Cfg, D>(opening_point.len(), 1)?;
    let alpha_bits = D.trailing_zeros() as usize;
    let prepared = prepare_root_opening_point::<F, D>(opening_point, basis, &root_lp, alpha_bits)?;
    let (y_ring, _) = poly.evaluate_and_fold(
        &prepared.ring_opening_point.b,
        &prepared.ring_opening_point.a,
        root_lp.block_len,
    );
    Ok((y_ring * prepared.inner_reduction.sigma_m1()).coefficients()[0] == *opening)
}

fn verify_root_direct_commitment<F, const D: usize, Cfg>(
    direct_witness: &DirectWitnessProof<F>,
    setup: &HachiVerifierSetup<F>,
    commitment: &RingCommitment<F, D>,
) -> Result<bool, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
    Cfg: CommitmentConfig<Field = F>,
{
    let field_witness = root_direct_field_witness(direct_witness)?;
    let poly = DensePoly::<F, D>::from_field_evals(
        field_witness.coeff_len().trailing_zeros() as usize,
        field_witness.coeffs(),
    )
    .map_err(|_| HachiError::InvalidProof)?;
    let total = setup.expanded.shared_matrix.total_ring_elements_at::<D>();
    let verifier_ntt = build_ntt_slot(setup.expanded.shared_matrix.ring_view::<D>(1, total))
        .map_err(|_| HachiError::InvalidProof)?;
    let temp_setup = HachiProverSetup {
        expanded: setup.expanded.clone(),
        ntt_shared: verifier_ntt,
    };
    let (expected_commitment, _) =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
            std::slice::from_ref(&poly),
            &temp_setup,
        )
        .map_err(|_| HachiError::InvalidProof)?;
    Ok(&expected_commitment == commitment)
}

fn prove_root_direct<F, const D: usize, P>(poly: &P) -> Result<HachiProof<F>, HachiError>
where
    F: FieldCore,
    P: HachiPolyOps<F, D>,
{
    Ok(HachiProof {
        steps: vec![HachiProofStep::Direct(poly.direct_root_witness()?)],
    })
}

fn verify_root_direct<F, T, const D: usize, Cfg>(
    proof: &HachiProof<F>,
    setup: &HachiVerifierSetup<F>,
    _transcript: &mut T,
    opening_point: &[F],
    opening: &F,
    commitment: &RingCommitment<F, D>,
    basis: BasisMode,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    if proof.steps.len() != 1 {
        return Err(HachiError::InvalidProof);
    }
    let direct_witness = proof.final_witness();
    if !root_direct_opening_matches::<F, D, Cfg>(direct_witness, opening_point, opening, basis)
        .map_err(|_| HachiError::InvalidProof)?
    {
        return Err(HachiError::InvalidProof);
    }
    if !verify_root_direct_commitment::<F, D, Cfg>(direct_witness, setup, commitment)? {
        return Err(HachiError::InvalidProof);
    }
    Ok(())
}

/// Prove one fold level: quad_eq -> ring_switch -> sumcheck.
///
type RecursiveSuffixResult<F> = (Vec<HachiLevelProof<F>>, usize, RecursiveProverState<F>);

/// Generic over the commitment config so it works for both the original
/// polynomial (using `Cfg`) and recursive w-openings (using `WCommitmentConfig`).
type CommitFn<'a, F> = Box<
    dyn FnOnce(
            &RecursiveWitnessFlat,
            LevelParams,
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
    root_key: HachiScheduleLookupKey,
    opening_point: &[F],
    hint: HachiCommitmentHint<F, D>,
    transcript: &mut T,
    commitment: &RingCommitment<F, D>,
    basis: BasisMode,
    level: usize,
    lp: &LevelParams,
    planning_envelope: HachiBatchPlanningEnvelope,
    next_level_params_override: Option<LevelParams>,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
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
    let alpha = lp.ring_dimension.trailing_zeros() as usize;
    if opening_point.len() < alpha {
        return Err(HachiError::InvalidPointDimension {
            expected: alpha,
            actual: opening_point.len(),
        });
    }
    let target_num_vars = lp.m_vars + lp.r_vars + alpha;
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let outer_point = &padded_point[alpha..];

    let ring_opening_point = {
        let _span = tracing::info_span!("ring_opening_point", level).entered();
        ring_opening_point_from_field::<F>(
            outer_point,
            lp.r_vars,
            lp.m_vars,
            basis,
            BlockOrder::RowMajor,
        )?
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
        poly.evaluate_and_fold(eval_outer_scalars, fold_scalars, lp.block_len)
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
        lp.clone(),
        hint,
        transcript,
        commitment,
        &y_ring,
        expanded.seed.max_stride,
    )?);

    finish_prove_level::<F, T, D, Cfg, Cfg>(
        expanded,
        ntt_shared,
        commit_w_fn,
        max_num_vars,
        root_key,
        transcript,
        &commitment.u,
        level,
        lp,
        planning_envelope,
        next_level_params_override,
        quad_eq,
        y_ring,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn finish_prove_level<F, T, const D: usize, LevelCfg, ScheduleCfg>(
    expanded: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    commit_w_fn: CommitFn<'_, F>,
    max_num_vars: usize,
    root_key: HachiScheduleLookupKey,
    transcript: &mut T,
    commitment_u: &[CyclotomicRing<F, D>],
    level: usize,
    lp: &LevelParams,
    planning_envelope: HachiBatchPlanningEnvelope,
    next_level_params_override: Option<LevelParams>,
    mut quad_eq: Box<QuadraticEquation<F, { D }, LevelCfg>>,
    y_ring: CyclotomicRing<F, D>,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    LevelCfg: CommitmentConfig<Field = F>,
    ScheduleCfg: CommitmentConfig<Field = F>,
{
    let w = ring_switch_build_w::<F, { D }, LevelCfg>(&mut quad_eq, expanded, ntt_shared, lp)?;
    let next_inputs = HachiScheduleInputs {
        max_num_vars,
        level: level + 1,
        current_w_len: w.len(),
    };
    let next_params = if let Some(params) = next_level_params_override {
        params
    } else {
        next_level_params_from_current_basis_and_envelope::<ScheduleCfg>(
            root_key,
            next_inputs,
            lp.log_basis,
            planning_envelope,
        )?
    };

    let (w_commitment_flat, w_hint_cache) = {
        let _span = tracing::info_span!("commit_w_level", level).entered();
        commit_w_fn(&w, next_params.clone())?
    };
    let w_commitment_proof = w_commitment_flat.clone();

    let rs = ring_switch_finalize::<F, T, { D }, LevelCfg>(
        &quad_eq,
        expanded,
        transcript,
        w,
        w_commitment_flat,
        &w_commitment_proof,
        w_hint_cache,
        lp,
    )?;

    let relation_claim = relation_claim_from_rows::<F, D>(
        &rs.tau1,
        rs.alpha,
        &quad_eq.v,
        commitment_u,
        std::slice::from_ref(&y_ring),
    );
    let RingSwitchOutput {
        w,
        w_commitment,
        w_hint,
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1: _,
        b,
        alpha: _,
    } = rs;
    let w_commitment = w_commitment.expect("prover ring switch must preserve w commitment");
    let tau0_reordered = reorder_stage1_coords(&tau0, col_bits, ring_bits);
    let (stage1_proof, r_stage1, s_claim) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let stage1_prover = HachiStage1Prover::new(
            &w_evals_compact,
            &tau0_reordered,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        )?;
        let (stage1_proof, r_stage1) = stage1_prover.prove(transcript)?;
        let s_claim = stage1_proof.s_claim;
        (stage1_proof, r_stage1, s_claim)
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
            col_bits,
            ring_bits,
            relation_claim,
        );
        let (stage2_sumcheck, sumcheck_challenges, stage2_final_claim) =
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
            stage2_final_claim,
            w_eval,
        )
    };

    let (level_proof, sumcheck_challenges) = (
        HachiLevelProof::new_two_stage::<D>(
            y_ring,
            quad_eq.v,
            stage1_proof,
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
            log_basis: next_params.log_basis,
            root_key,
            planning_envelope,
            sumcheck_challenges,
        },
    })
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_batched_root_level_with_points<F, T, const D: usize, Cfg, P>(
    expanded: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    commit_w_fn: CommitFn<'_, F>,
    polys: &[&P],
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    claim_to_point: &[usize],
    claim_group_sizes: &[usize],
    point_group_sizes: Option<&[usize]>,
    max_num_vars: usize,
    root_key: HachiScheduleLookupKey,
    hints: Vec<HachiBatchedCommitmentHint<F, D>>,
    transcript: &mut T,
    commitments: &[RingCommitment<F, D>],
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<BatchedProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
    P: HachiPolyOps<F, D>,
{
    if prepared_points.is_empty() || claim_to_point.len() != polys.len() {
        return Err(HachiError::InvalidInput(
            "invalid batched root inputs".to_string(),
        ));
    }
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= prepared_points.len())
    {
        return Err(HachiError::InvalidInput(
            "batched root claim-to-point index out of range".to_string(),
        ));
    }
    let (y_rings, w_folded_by_poly) = {
        let _span = tracing::info_span!(
            "evaluate_and_fold_batched_with_points",
            level = 0usize,
            num_polys = polys.len(),
            num_points = prepared_points.len()
        )
        .entered();
        let mut y_rings = Vec::with_capacity(polys.len());
        let mut w_folded_by_poly = Vec::with_capacity(polys.len());
        if prepared_points.len() == 1 {
            let prepared_point = &prepared_points[0];
            for poly in polys {
                let (y_ring, w_folded) = poly.evaluate_and_fold(
                    &prepared_point.ring_opening_point.b,
                    &prepared_point.ring_opening_point.a,
                    root_lp.block_len,
                );
                y_rings.push(y_ring);
                w_folded_by_poly.push(w_folded);
            }
        } else {
            for (poly, &point_idx) in polys.iter().zip(claim_to_point.iter()) {
                let prepared_point = &prepared_points[point_idx];
                let (y_ring, w_folded) = poly.evaluate_and_fold(
                    &prepared_point.ring_opening_point.b,
                    &prepared_point.ring_opening_point.a,
                    root_lp.block_len,
                );
                y_rings.push(y_ring);
                w_folded_by_poly.push(w_folded);
            }
        }
        (y_rings, w_folded_by_poly)
    };
    if let Some(point_group_sizes) = point_group_sizes {
        append_batch_shape_to_transcript::<F, T>(point_group_sizes, claim_group_sizes, transcript);
    }
    append_batched_commitments_to_transcript(commitments, transcript);
    for prepared_point in prepared_points {
        for pt in &prepared_point.padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    // Absorb field-element openings and sample γ for batched CWSS.
    let openings: Vec<F> = y_rings
        .iter()
        .zip(claim_to_point.iter())
        .map(|(y_ring, &point_idx)| {
            let v = &prepared_points[point_idx].inner_reduction;
            (*y_ring * v.sigma_m1()).coefficients()[0]
        })
        .collect();
    for opening in &openings {
        transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, opening);
    }
    let gamma: Vec<F> = (0..polys.len())
        .map(|_| transcript.challenge_scalar(CHALLENGE_EVAL_BATCH))
        .collect();

    // Compute batched y_rings: one per distinct opening point.
    let num_points = prepared_points.len();
    let batched_y_rings: Vec<CyclotomicRing<F, D>> = {
        let mut per_point = vec![CyclotomicRing::<F, D>::zero(); num_points];
        for (claim_idx, y_ring) in y_rings.iter().enumerate() {
            let point_idx = claim_to_point[claim_idx];
            per_point[point_idx] += y_ring.scale(&gamma[claim_idx]);
        }
        per_point
    };

    for y_ring in &batched_y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    let ring_opening_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_opening_point.clone())
        .collect();
    let quad_eq = Box::new(
        QuadraticEquation::<F, { D }, Cfg>::new_multipoint_batched_prover(
            ntt_shared,
            ring_opening_points,
            claim_to_point.to_vec(),
            polys,
            w_folded_by_poly,
            claim_group_sizes,
            batched_lp.clone(),
            hints,
            transcript,
            commitments,
            &batched_y_rings,
            gamma,
            num_points,
            expanded.seed.max_stride,
        )?,
    );
    finish_batched_root_level::<F, T, D, Cfg>(
        expanded,
        ntt_shared,
        commit_w_fn,
        max_num_vars,
        root_key,
        transcript,
        commitments,
        batched_lp,
        planning_envelope,
        quad_eq,
        batched_y_rings,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_batched_root_level<F, T, const D: usize, Cfg, P>(
    expanded: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    commit_w_fn: CommitFn<'_, F>,
    polys: &[&P],
    claim_group_sizes: &[usize],
    max_num_vars: usize,
    root_key: HachiScheduleLookupKey,
    opening_point: &[F],
    hints: Vec<HachiBatchedCommitmentHint<F, D>>,
    transcript: &mut T,
    commitments: &[RingCommitment<F, D>],
    basis: BasisMode,
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<BatchedProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
    P: HachiPolyOps<F, D>,
{
    let alpha_bits = root_lp.ring_dimension.trailing_zeros() as usize;
    let prepared_point =
        prepare_root_opening_point::<F, D>(opening_point, basis, root_lp, alpha_bits)?;
    let claim_to_point = vec![0usize; polys.len()];
    prove_batched_root_level_with_points::<F, T, D, Cfg, P>(
        expanded,
        ntt_shared,
        commit_w_fn,
        polys,
        std::slice::from_ref(&prepared_point),
        &claim_to_point,
        claim_group_sizes,
        None,
        max_num_vars,
        root_key,
        hints,
        transcript,
        commitments,
        root_lp,
        batched_lp,
        planning_envelope,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_multipoint_batched_root_level<F, T, const D: usize, Cfg, P>(
    expanded: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    commit_w_fn: CommitFn<'_, F>,
    polys: &[&P],
    batch_shape: &MultiPointBatchShape,
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    max_num_vars: usize,
    root_key: HachiScheduleLookupKey,
    hints: Vec<HachiBatchedCommitmentHint<F, D>>,
    transcript: &mut T,
    commitments: &[RingCommitment<F, D>],
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<BatchedProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
    P: HachiPolyOps<F, D>,
{
    prove_batched_root_level_with_points::<F, T, D, Cfg, P>(
        expanded,
        ntt_shared,
        commit_w_fn,
        polys,
        prepared_points,
        &batch_shape.claim_to_point,
        &batch_shape.claim_group_sizes,
        Some(&batch_shape.point_group_sizes),
        max_num_vars,
        root_key,
        hints,
        transcript,
        commitments,
        root_lp,
        batched_lp,
        planning_envelope,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn finish_batched_root_level<F, T, const D: usize, Cfg>(
    expanded: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    commit_w_fn: CommitFn<'_, F>,
    max_num_vars: usize,
    root_key: HachiScheduleLookupKey,
    transcript: &mut T,
    commitments: &[RingCommitment<F, D>],
    lp: &LevelParams,
    planning_envelope: HachiBatchPlanningEnvelope,
    mut quad_eq: Box<QuadraticEquation<F, { D }, Cfg>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
) -> Result<BatchedProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    let commitment_rows = flatten_batched_commitment_rows(commitments);
    let w = ring_switch_build_w::<F, { D }, Cfg>(&mut quad_eq, expanded, ntt_shared, lp)?;
    let next_inputs = HachiScheduleInputs {
        max_num_vars,
        level: 1,
        current_w_len: w.len(),
    };
    let next_params = next_level_params_from_current_basis_and_envelope::<Cfg>(
        root_key,
        next_inputs,
        lp.log_basis,
        planning_envelope,
    )?;

    let (w_commitment_flat, w_hint_cache) = {
        let _span = tracing::info_span!("commit_w_level", level = 0usize).entered();
        commit_w_fn(&w, next_params.clone())?
    };
    let w_commitment_proof = w_commitment_flat.clone();

    let rs = ring_switch_finalize_with_claim_groups::<F, T, { D }, Cfg>(
        &quad_eq,
        expanded,
        transcript,
        w,
        w_commitment_flat,
        &w_commitment_proof,
        w_hint_cache,
        lp,
    )?;

    let relation_claim = relation_claim_from_rows::<F, D>(
        &rs.tau1,
        rs.alpha,
        &quad_eq.v,
        &commitment_rows,
        &y_rings,
    );
    let RingSwitchOutput {
        w,
        w_commitment,
        w_hint,
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1: _,
        b,
        alpha: _,
    } = rs;
    let w_commitment = w_commitment.expect("prover ring switch must preserve w commitment");
    let tau0_reordered = reorder_stage1_coords(&tau0, col_bits, ring_bits);
    let (stage1_proof, r_stage1, s_claim) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let stage1_prover = HachiStage1Prover::new(
            &w_evals_compact,
            &tau0_reordered,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        )?;
        let (stage1_proof, r_stage1) = stage1_prover.prove(transcript)?;
        let s_claim = stage1_proof.s_claim;
        (stage1_proof, r_stage1, s_claim)
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
            col_bits,
            ring_bits,
            relation_claim,
        );
        let (stage2_sumcheck, sumcheck_challenges, stage2_final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut stage2_prover, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?;

        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level = 0usize).entered();
            stage2_prover.final_w_eval()
        };
        (
            stage2_sumcheck,
            sumcheck_challenges,
            stage2_final_claim,
            w_eval,
        )
    };

    let root_proof = HachiBatchedRootProof::new_two_stage::<D>(
        y_rings,
        quad_eq.v,
        stage1_proof,
        stage2_sumcheck,
        w_commitment_proof,
        w_eval,
    );

    Ok(BatchedProveLevelOutput {
        root_proof,
        next_state: RecursiveProverState {
            w,
            commitment: w_commitment,
            hint: w_hint.expect("prover ring switch must preserve recursive hint cache"),
            log_basis: next_params.log_basis,
            root_key,
            planning_envelope,
            sumcheck_challenges,
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
    root_key: HachiScheduleLookupKey,
    opening_point: &[F],
    hint: HachiCommitmentHint<F, D>,
    transcript: &mut T,
    commitment: &FlatRingVec<F>,
    level: usize,
    lp: &LevelParams,
    planning_envelope: HachiBatchPlanningEnvelope,
    next_level_params_override: Option<LevelParams>,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level,
            "prove_one_recursive_level"
        );
    }
    let alpha = lp.ring_dimension.trailing_zeros() as usize;
    if opening_point.len() < alpha {
        return Err(HachiError::InvalidPointDimension {
            expected: alpha,
            actual: opening_point.len(),
        });
    }
    let target_num_vars = lp.m_vars + lp.r_vars + alpha;
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let outer_point = &padded_point[alpha..];

    let ring_opening_point = {
        let _span = tracing::info_span!("ring_opening_point", level).entered();
        ring_opening_point_from_field::<F>(
            outer_point,
            lp.r_vars,
            lp.m_vars,
            BasisMode::Lagrange,
            BlockOrder::ColumnMajor,
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
        witness.evaluate_and_fold(
            eval_outer_scalars,
            fold_scalars,
            lp.block_len,
            lp.num_blocks,
        )
    };

    commitment.append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;
    for pt in &padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);
    let commitment_u = commitment.as_ring_slice::<D>()?;

    let quad_eq = Box::new(
        QuadraticEquation::<F, { D }, WCommitmentConfig<{ D }, Cfg>>::new_recursive_prover(
            ntt_shared,
            ring_opening_point,
            witness,
            w_folded,
            lp.clone(),
            hint,
            transcript,
            commitment_u,
            &y_ring,
            expanded.seed.max_stride,
        )?,
    );

    finish_prove_level::<F, T, D, WCommitmentConfig<{ D }, Cfg>, Cfg>(
        expanded,
        ntt_shared,
        commit_w_fn,
        max_num_vars,
        root_key,
        transcript,
        commitment_u,
        level,
        lp,
        planning_envelope,
        next_level_params_override,
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

/// Batched recursion already consults the byte planner before folding again.
///
/// The runtime safety guard here only needs to catch tiny tails and fixed
/// points, not enforce the single-proof shrink-ratio heuristic. Otherwise we
/// can stop early even when another fold would still reduce proof size.
fn should_stop_batched_folding(w_len: usize, prev_w_len: usize) -> bool {
    w_len <= MIN_W_LEN_FOR_FOLDING || w_len >= prev_w_len
}

#[cfg(test)]
fn should_continue_folding_by_bytes<Cfg: CommitmentConfig>(
    root_key: HachiScheduleLookupKey,
    level: usize,
    current_w_len: usize,
    current_log_basis: u32,
) -> Result<bool, HachiError> {
    should_continue_folding_by_bytes_and_envelope::<Cfg>(
        root_key,
        level,
        current_w_len,
        current_log_basis,
        HachiBatchPlanningEnvelope::singleton::<Cfg>(),
    )
}

fn should_continue_folding_by_bytes_and_envelope<Cfg: CommitmentConfig>(
    root_key: HachiScheduleLookupKey,
    level: usize,
    current_w_len: usize,
    current_log_basis: u32,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<bool, HachiError> {
    let direct_bytes = packed_digits_bytes(current_w_len, current_log_basis);
    Ok(
        planned_recursive_suffix_bytes_with_log_basis_and_envelope::<Cfg>(
            root_key,
            level,
            current_w_len,
            current_log_basis,
            planning_envelope,
        )? < direct_bytes,
    )
}

fn next_level_params_from_current_basis_and_envelope<Cfg: CommitmentConfig>(
    root_key: HachiScheduleLookupKey,
    next_inputs: HachiScheduleInputs,
    current_log_basis: u32,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<LevelParams, HachiError> {
    let next_log_basis = planned_next_log_basis_with_current_basis_and_envelope::<Cfg>(
        root_key,
        next_inputs,
        current_log_basis,
        planning_envelope,
    )?;
    Ok(Cfg::level_params_with_log_basis(
        next_inputs,
        next_log_basis,
    ))
}

/// Dispatch a commit-w operation to the correct ring dimension.
///
/// Each match arm builds NTT caches for the target D and calls `commit_w`.
/// `#[inline(never)]` isolates the match arms in their own stack frame,
/// preventing debug-mode stack bloat from monomorphized arms.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_commit<F, Cfg>(
    commit_params: LevelParams,
    commit_ntt_cache: &mut MultiDNttCaches,
    expanded: &HachiExpandedSetup<F>,
    w: &RecursiveWitnessFlat,
) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig<Field = F>,
{
    let commit_d = commit_params.ring_dimension;
    let stride = expanded.seed.max_stride;
    dispatch_with_ntt!(
        commit_d,
        commit_ntt_cache,
        expanded,
        |D_COMMIT, ntt_shared| {
            let (wc, wh) = commit_w::<F, { D_COMMIT }, WCommitmentConfig<{ D_COMMIT }, Cfg>>(
                w,
                ntt_shared,
                &commit_params,
                stride,
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
    level_params: &LevelParams,
    next_level_params_override: Option<LevelParams>,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
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
            next_level_params_override,
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
                next_level_params_override,
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
    final_w: Option<&DirectWitnessProof<F>>,
    lp: &LevelParams,
    block_order: BlockOrder,
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
            lp,
            block_order,
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
    level_params: &LevelParams,
    next_level_params_override: Option<LevelParams>,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    let _setup_span = tracing::info_span!("inter_level_setup", level).entered();

    let current_w = &current_state.w;
    let opening_point = current_state.sumcheck_challenges.clone();

    let w_lp = hachi_recursive_level_layout_from_params::<Cfg>(level_params, current_w.len())?;
    let w_view = current_w.view::<F, { D_LEVEL }>()?;
    let typed_hint: HachiCommitmentHint<F, { D_LEVEL }> =
        current_state.hint.to_typed::<{ D_LEVEL }>()?;
    drop(_setup_span);

    let max_stride = expanded.seed.max_stride;
    let commit_fn: CommitFn<'_, F> = Box::new(
        |w: &RecursiveWitnessFlat,
         next_params: LevelParams|
         -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), HachiError> {
            if next_params.ring_dimension == D_LEVEL {
                let (wc, wh) = commit_w::<F, { D_LEVEL }, WCommitmentConfig<{ D_LEVEL }, Cfg>>(
                    w,
                    ntt_shared,
                    &next_params,
                    max_stride,
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

    prove_one_recursive_level::<F, T, { D_LEVEL }, Cfg>(
        expanded,
        ntt_shared,
        commit_fn,
        &w_view,
        max_num_vars,
        current_state.root_key,
        &opening_point,
        typed_hint,
        transcript,
        &current_state.commitment,
        level,
        &w_lp,
        current_state.planning_envelope,
        next_level_params_override,
    )
}

fn prove_batched_recursive_suffix<F, T, const D: usize, Cfg>(
    setup: &HachiProverSetup<F, D>,
    ntt_cache: &mut MultiDNttCaches,
    commit_ntt_cache: &mut MultiDNttCaches,
    max_num_vars: usize,
    transcript: &mut T,
    next_state: RecursiveProverState<F>,
    initial_prev_w_len: usize,
) -> Result<RecursiveSuffixResult<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    let mut levels = Vec::new();
    let mut current_state = next_state;
    let mut prev_poly_len = initial_prev_w_len;
    let mut level = 1usize;

    loop {
        let current_w_len = current_state.w.len();
        if should_stop_batched_folding(current_w_len, prev_poly_len) {
            break;
        }
        let should_continue = should_continue_folding_by_bytes_and_envelope::<Cfg>(
            current_state.root_key,
            level,
            current_w_len,
            current_state.log_basis,
            current_state.planning_envelope,
        )?;
        if !should_continue {
            break;
        }
        let level_params = Cfg::level_params_with_log_basis(
            HachiScheduleInputs {
                max_num_vars,
                level,
                current_w_len,
            },
            current_state.log_basis,
        );
        let level_d = level_params.ring_dimension;

        let out = dispatch_prove_level::<F, T, D, Cfg>(
            level_d,
            ntt_cache,
            &setup.expanded,
            &setup.ntt_shared,
            commit_ntt_cache,
            max_num_vars,
            &current_state,
            transcript,
            level,
            &level_params,
            None,
        )?;

        levels.push(out.level_proof);
        prev_poly_len = current_w_len;
        current_state = out.next_state;
        level += 1;
    }

    Ok((levels, level, current_state))
}

fn finalize_batched_recursive_proof<F, const D: usize, Cfg>(
    max_num_vars: usize,
    root_proof: HachiBatchedRootProof<F>,
    levels: Vec<HachiLevelProof<F>>,
    level: usize,
    current_state: RecursiveProverState<F>,
) -> HachiBatchedProof<F>
where
    F: FieldCore,
    Cfg: CommitmentConfig<Field = F>,
{
    let final_params = Cfg::level_params_with_log_basis(
        HachiScheduleInputs {
            max_num_vars,
            level,
            current_w_len: current_state.w.len(),
        },
        current_state.log_basis,
    );
    let final_w = PackedDigits::from_i8_digits_with_min_bits(
        current_state.w.as_i8_digits(),
        final_params.log_basis,
    );
    let mut steps = levels
        .into_iter()
        .map(HachiProofStep::Fold)
        .collect::<Vec<_>>();
    steps.push(HachiProofStep::Direct(DirectWitnessProof::PackedDigits(
        final_w,
    )));

    HachiBatchedProof {
        root: root_proof,
        steps,
    }
}

fn verify_batched_recursive_suffix<'a, F, T, const D: usize, Cfg>(
    proof: &'a HachiBatchedProof<F>,
    setup: &HachiVerifierSetup<F>,
    transcript: &mut T,
    max_num_vars: usize,
    mut current_state: RecursiveVerifierState<'a, F>,
    final_w: Option<&DirectWitnessProof<F>>,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    let num_levels = proof.num_fold_levels();
    for (offset, level_proof) in proof.fold_levels().enumerate() {
        let level_index = offset + 1;
        let is_last = offset == num_levels - 1;
        let level_params = Cfg::level_params_with_log_basis(
            HachiScheduleInputs {
                max_num_vars,
                level: level_index,
                current_w_len: current_state.w_len,
            },
            current_state.log_basis,
        );
        let level_d = level_params.ring_dimension;
        let current_lp =
            hachi_recursive_level_layout_from_params::<Cfg>(&level_params, current_state.w_len)?;
        if !current_state.commitment.can_decode_vec(level_d)
            || !level_proof.y_ring.can_decode_single(level_d)
            || !level_proof.v.can_decode_vec(level_d)
        {
            return Err(HachiError::InvalidProof);
        }

        let challenges = if level_d == D {
            verify_one_level::<F, T, D>(
                level_proof,
                setup,
                transcript,
                &current_state,
                is_last,
                if is_last { final_w } else { None },
                &current_lp,
                BlockOrder::ColumnMajor,
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
                &current_lp,
                BlockOrder::ColumnMajor,
            )?
        };

        if !is_last {
            let next_w_len = w_ring_element_count::<F>(&current_lp) * level_d;
            let next_level_params = next_level_params_from_current_basis_and_envelope::<Cfg>(
                current_state.root_key,
                HachiScheduleInputs {
                    max_num_vars,
                    level: level_index + 1,
                    current_w_len: next_w_len,
                },
                current_state.log_basis,
                current_state.planning_envelope,
            )?;

            if level_index < num_levels {
                let next_level_d = next_level_params.ring_dimension;
                if !level_proof.next_w_commitment().can_decode_vec(next_level_d) {
                    return Err(HachiError::InvalidProof);
                }
            }
            current_state = RecursiveVerifierState {
                opening_point: challenges,
                opening: level_proof.next_w_eval(),
                commitment: level_proof.next_w_commitment(),
                basis: BasisMode::Lagrange,
                w_len: next_w_len,
                log_basis: next_level_params.log_basis,
                root_key: current_state.root_key,
                planning_envelope: current_state.planning_envelope,
            };
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prove_same_point_batched<F, T, const D: usize, Cfg, P>(
    setup: &HachiProverSetup<F, D>,
    poly_groups: &[&[P]],
    opening_point: &[F],
    hints: Vec<HachiBatchedCommitmentHint<F, D>>,
    transcript: &mut T,
    commitments: &[RingCommitment<F, D>],
    basis: BasisMode,
) -> Result<HachiBatchedProof<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
    P: HachiPolyOps<F, D>,
{
    let claim_group_sizes = validate_nonempty_group_sizes(poly_groups, "batched_prove")?;
    let total_claims = checked_total_claims(&claim_group_sizes, "batched_prove")?;
    if total_claims > setup.expanded.seed.max_num_batched_polys {
        return Err(HachiError::InvalidInput(format!(
            "batched_prove received {total_claims} polynomials but setup supports at most {}",
            setup.expanded.seed.max_num_batched_polys
        )));
    }
    let hint_group_sizes: Vec<usize> = hints
        .iter()
        .map(|hint| hint.inner_opening_digits.len())
        .collect();
    if commitments.len() != poly_groups.len()
        || hints.len() != poly_groups.len()
        || hint_group_sizes != claim_group_sizes
    {
        return Err(HachiError::InvalidInput(
            "batched_prove commitments, hints, and polynomial groups must have matching lengths"
                .to_string(),
        ));
    }

    let t_prove_total = Instant::now();
    let mut ntt_cache = MultiDNttCaches::new();
    let mut commit_ntt_cache = MultiDNttCaches::new();
    let max_num_vars = setup.expanded.seed.max_num_vars;
    let num_vars = opening_point.len();
    let root_plan = hachi_root_runtime_plan_with_batch::<Cfg, D>(
        max_num_vars,
        num_vars,
        setup.expanded.seed.max_num_batched_polys,
        HachiRootBatchSummary::from_claim_group_sizes(&claim_group_sizes, 1)?,
    )?;
    let root_lp = root_plan.root_lp.clone();
    let batched_lp = root_plan.level_lp.clone();

    if commitments
        .iter()
        .any(|commitment| commitment.u.len() != root_lp.b_key.row_len())
    {
        return Err(HachiError::InvalidInput(
            "batched_prove received a commitment with the wrong length".to_string(),
        ));
    }
    let flat_polys = flatten_poly_groups(poly_groups);
    let max_stride = setup.expanded.seed.max_stride;

    let commit_fn_0: CommitFn<'_, F> = Box::new(
        |w: &RecursiveWitnessFlat,
         next_params: LevelParams|
         -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), HachiError> {
            if next_params.ring_dimension == D {
                let (wc, wh) =
                    commit_w::<F, D, Cfg>(w, &setup.ntt_shared, &next_params, max_stride)?;
                Ok((
                    FlatRingVec::from_commitment(&wc),
                    RecursiveCommitmentHintCache::from_typed(wh)?,
                ))
            } else {
                dispatch_commit::<F, Cfg>(next_params, &mut commit_ntt_cache, &setup.expanded, w)
            }
        },
    );
    let BatchedProveLevelOutput {
        root_proof,
        next_state,
    } = prove_batched_root_level::<F, T, D, Cfg, P>(
        &setup.expanded,
        &setup.ntt_shared,
        commit_fn_0,
        &flat_polys,
        &claim_group_sizes,
        max_num_vars,
        root_plan.lookup_key(),
        opening_point,
        hints,
        transcript,
        commitments,
        basis,
        &root_lp,
        &batched_lp,
        root_plan.planning_envelope,
    )?;
    let (levels, level, current_state) = prove_batched_recursive_suffix::<F, T, D, Cfg>(
        setup,
        &mut ntt_cache,
        &mut commit_ntt_cache,
        max_num_vars,
        transcript,
        next_state,
        root_plan.inputs.current_w_len,
    )?;

    tracing::info!(
        levels = level,
        elapsed_s = t_prove_total.elapsed().as_secs_f64(),
        "hachi batched prove complete"
    );

    Ok(finalize_batched_recursive_proof::<F, D, Cfg>(
        max_num_vars,
        root_proof,
        levels,
        level,
        current_state,
    ))
}

#[allow(clippy::too_many_arguments)]
fn verify_same_point_batched<F, T, const D: usize, Cfg>(
    proof: &HachiBatchedProof<F>,
    setup: &HachiVerifierSetup<F>,
    transcript: &mut T,
    opening_point: &[F],
    opening_groups: &[&[F]],
    commitments: &[RingCommitment<F, D>],
    basis: BasisMode,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    let y_coeff_len = proof.root.y_rings().coeff_len();
    if !y_coeff_len.is_multiple_of(D) {
        return Err(HachiError::InvalidProof);
    }
    let claim_group_sizes = validate_nonempty_group_sizes(opening_groups, "batched_verify")
        .map_err(|_| HachiError::InvalidProof)?;
    let num_claims = checked_total_claims(&claim_group_sizes, "batched_verify")
        .map_err(|_| HachiError::InvalidProof)?;
    let openings = flatten_value_groups(opening_groups);
    if num_claims == 0 || openings.len() != num_claims || commitments.len() != opening_groups.len()
    {
        return Err(HachiError::InvalidProof);
    }
    if y_coeff_len / D != 1 {
        return Err(HachiError::InvalidProof);
    }
    if num_claims > setup.expanded.seed.max_num_batched_polys {
        return Err(HachiError::InvalidProof);
    }

    let t_verify_hachi = Instant::now();
    let max_num_vars = setup.expanded.seed.max_num_vars;
    let num_vars = opening_point.len();
    let root_plan = hachi_root_runtime_plan_with_batch::<Cfg, D>(
        max_num_vars,
        num_vars,
        setup.expanded.seed.max_num_batched_polys,
        HachiRootBatchSummary::from_claim_group_sizes(&claim_group_sizes, 1)
            .map_err(|_| HachiError::InvalidProof)?,
    )
    .map_err(|_| HachiError::InvalidProof)?;
    let root_lp = root_plan.root_lp.clone();
    let batched_lp = root_plan.level_lp.clone();

    let final_w = Some(proof.final_witness());
    let has_recursive_levels = proof.num_fold_levels() > 0;
    let root_challenges = verify_batched_root_level::<F, T, D>(
        &proof.root,
        setup,
        transcript,
        opening_point,
        &openings,
        commitments,
        &claim_group_sizes,
        basis,
        &root_lp,
        &batched_lp,
        !has_recursive_levels,
        if has_recursive_levels { None } else { final_w },
    )?;

    if has_recursive_levels {
        let root_w_len = root_plan.next_w_len();
        let first_level_d = root_plan.next_level_params.ring_dimension;
        if !proof.root.next_w_commitment().can_decode_vec(first_level_d) {
            return Err(HachiError::InvalidProof);
        }

        let current_state = RecursiveVerifierState {
            opening_point: root_challenges,
            opening: proof.root.next_w_eval(),
            commitment: proof.root.next_w_commitment(),
            basis: BasisMode::Lagrange,
            w_len: root_w_len,
            log_basis: root_plan.next_level_params.log_basis,
            root_key: root_plan.lookup_key(),
            planning_envelope: root_plan.planning_envelope,
        };
        verify_batched_recursive_suffix::<F, T, D, Cfg>(
            proof,
            setup,
            transcript,
            max_num_vars,
            current_state,
            final_w,
        )?;
    }

    tracing::info!(
        levels = proof.num_fold_levels() + 1,
        elapsed_s = t_verify_hachi.elapsed().as_secs_f64(),
        "hachi batched verify complete"
    );

    Ok(())
}

impl<F, const D: usize, Cfg> CommitmentScheme<F, D> for HachiCommitmentScheme<D, Cfg>
where
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
    Cfg: CommitmentConfig<Field = F>,
{
    type ProverSetup = HachiProverSetup<F, D>;
    type VerifierSetup = HachiVerifierSetup<F>;
    type Commitment = RingCommitment<F, D>;
    type Proof = HachiProof<F>;
    type BatchedProof = HachiBatchedProof<F>;
    type CommitHint = HachiBatchedCommitmentHint<F, D>;
    type BatchedCommitHint = Vec<HachiBatchedCommitmentHint<F, D>>;

    fn setup_prover(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Self::ProverSetup {
        HachiProverSetup::new::<Cfg>(max_num_vars, max_num_batched_polys, max_num_points)
            .expect("commitment setup failed")
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        setup.verifier_setup()
    }

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::commit")]
    fn commit<P: HachiPolyOps<F, D>>(
        polys: &[P],
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::CommitHint), HachiError> {
        if polys.is_empty() {
            return Err(HachiError::InvalidInput(
                "commit requires at least one polynomial".to_string(),
            ));
        }
        let num_vars = polys[0].num_vars();
        if polys.iter().any(|p| p.num_vars() != num_vars) {
            return Err(HachiError::InvalidInput(
                "all polynomials in a batched commit must have the same num_vars".to_string(),
            ));
        }
        if polys.len() > setup.expanded.seed.max_num_batched_polys {
            return Err(HachiError::InvalidInput(format!(
                "commit received {} polynomials but setup supports at most {}",
                polys.len(),
                setup.expanded.seed.max_num_batched_polys
            )));
        }
        if num_vars > setup.expanded.seed.max_num_vars {
            return Err(HachiError::InvalidInput(format!(
                "commit received a polynomial with {} variables but setup supports at most {}",
                num_vars, setup.expanded.seed.max_num_vars
            )));
        }
        let max_num_vars = setup.expanded.seed.max_num_vars;
        let root_plan = hachi_root_runtime_plan_with_batch::<Cfg, D>(
            max_num_vars,
            num_vars,
            setup.expanded.seed.max_num_batched_polys,
            HachiRootBatchSummary::new(polys.len(), 1, 1)?,
        )?;
        let root_lp = root_plan.root_lp.clone();

        let inner_witnesses = crate::cfg_iter!(polys)
            .map(|poly| {
                poly.commit_inner_witness(
                    &setup.expanded.shared_matrix,
                    &setup.ntt_shared,
                    root_lp.a_key.row_len(),
                    root_lp.block_len,
                    root_lp.num_digits_commit,
                    root_lp.num_digits_open,
                    root_lp.log_basis,
                    setup.expanded.seed.max_stride,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut inner_opening_digits_flat = Vec::new();
        let mut group_t_hat = Vec::with_capacity(polys.len());
        let mut group_t = Vec::with_capacity(polys.len());
        for mut inner in inner_witnesses {
            for t_i in &mut inner.t {
                t_i.truncate(root_plan.root_lp.a_key.row_len());
            }
            inner.t_hat.truncate_each_block(
                root_plan.root_lp.a_key.row_len() * root_plan.root_lp.num_digits_open,
            );
            inner_opening_digits_flat.extend_from_slice(inner.t_hat.flat_digits());
            group_t_hat.push(inner.t_hat);
            group_t.push(inner.t);
        }
        let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
            &setup.ntt_shared,
            root_lp.b_key.row_len(),
            setup.expanded.seed.max_stride,
            &inner_opening_digits_flat,
        );
        Ok((
            RingCommitment { u },
            HachiBatchedCommitmentHint::with_t(group_t_hat, group_t),
        ))
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
    ) -> Result<Self::Proof, HachiError> {
        let num_vars = opening_point.len();
        if num_vars > setup.expanded.seed.max_num_vars {
            return Err(HachiError::InvalidInput(format!(
                "prove received an opening point with {} variables but setup supports at most {}",
                num_vars, setup.expanded.seed.max_num_vars
            )));
        }
        if schedule_uses_root_direct::<Cfg>(num_vars)? {
            return prove_root_direct::<F, D, P>(poly);
        }
        let t_prove_total = Instant::now();
        let mut levels = Vec::new();
        let hint = single_prove_hint_from_group_hint(hint)?;

        let mut ntt_cache = MultiDNttCaches::new();
        let mut commit_ntt_cache = MultiDNttCaches::new();
        let max_num_vars = setup.expanded.seed.max_num_vars;
        let root_plan = hachi_root_runtime_plan::<Cfg, D>(
            max_num_vars,
            num_vars,
            setup.expanded.seed.max_num_batched_polys,
        )?;
        let exact_plan = if num_vars == max_num_vars {
            Cfg::schedule_plan(root_plan.lookup_key())?
        } else {
            None
        };
        let mut root_next_params_override = None;
        if let Some(plan) = exact_plan.as_ref() {
            let planned_root = exact_planned_level_execution::<Cfg>(
                plan,
                root_plan.inputs,
                root_plan.root_lp.log_basis,
            )?
            .ok_or_else(|| {
                HachiError::InvalidSetup(
                    "exact planned root fold did not match runtime root state".to_string(),
                )
            })?;
            if planned_root.level.lp != root_plan.root_lp {
                return Err(HachiError::InvalidSetup(
                    "exact planned root fold drifted from runtime root plan".to_string(),
                ));
            }
            root_next_params_override = Some(planned_root.next_level_params);
        }
        let root_lp = root_plan.root_lp.clone();
        let max_stride = setup.expanded.seed.max_stride;

        let commit_fn_0: CommitFn<'_, F> = Box::new(
            |w: &RecursiveWitnessFlat,
             next_params: LevelParams|
             -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), HachiError> {
                if next_params.ring_dimension == D {
                    let (wc, wh) =
                        commit_w::<F, D, Cfg>(w, &setup.ntt_shared, &next_params, max_stride)?;
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
            root_plan.lookup_key(),
            opening_point,
            hint,
            transcript,
            commitment,
            basis,
            0,
            &root_lp,
            root_plan.planning_envelope,
            root_next_params_override,
        )?;
        levels.push(out.level_proof);

        let mut prev_poly_len = root_plan.inputs.current_w_len;
        let mut current_state = out.next_state;
        let mut level = 1usize;
        let planned_num_levels = exact_plan.as_ref().map(|plan| plan.num_fold_levels());

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
            let inputs = HachiScheduleInputs {
                max_num_vars,
                level,
                current_w_len: current_state.w.len(),
            };
            let planned_level = if let Some(plan) = exact_plan.as_ref() {
                Some(
                    exact_planned_level_execution::<Cfg>(plan, inputs, current_state.log_basis)?
                        .ok_or_else(|| {
                            HachiError::InvalidSetup(
                                "exact planned recursive level did not match runtime state"
                                    .to_string(),
                            )
                        })?,
                )
            } else {
                None
            };
            let (level_params, next_level_params_override) =
                if let Some(planned_level) = planned_level {
                    (
                        planned_level.level.lp,
                        Some(planned_level.next_level_params),
                    )
                } else {
                    (
                        Cfg::level_params_with_log_basis(inputs, current_state.log_basis),
                        None,
                    )
                };
            let level_d = level_params.ring_dimension;

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
                next_level_params_override,
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

        let final_log_basis = if let Some(plan) = exact_plan.as_ref() {
            let direct_step = plan.direct_step();
            if direct_step.state.current_w_len != current_state.w.len()
                || direct_step.state.log_basis != current_state.log_basis
            {
                return Err(HachiError::InvalidSetup(
                    "exact planned direct step did not match final runtime state".to_string(),
                ));
            }
            match direct_step.witness_shape {
                crate::protocol::proof::DirectWitnessShape::PackedDigits((_, bits_per_elem)) => {
                    bits_per_elem
                }
                crate::protocol::proof::DirectWitnessShape::FieldElements(_) => {
                    return Err(HachiError::InvalidSetup(
                        "folding proof cannot terminate in field-element direct witness"
                            .to_string(),
                    ))
                }
            }
        } else {
            Cfg::level_params_with_log_basis(
                HachiScheduleInputs {
                    max_num_vars,
                    level,
                    current_w_len: current_state.w.len(),
                },
                current_state.log_basis,
            )
            .log_basis
        };
        let final_w = PackedDigits::from_i8_digits_with_min_bits(
            current_state.w.as_i8_digits(),
            final_log_basis,
        );
        let mut steps = levels
            .into_iter()
            .map(HachiProofStep::Fold)
            .collect::<Vec<_>>();
        steps.push(HachiProofStep::Direct(DirectWitnessProof::PackedDigits(
            final_w,
        )));

        Ok(HachiProof { steps })
    }

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::batched_prove")]
    fn batched_prove<T: Transcript<F>, P: HachiPolyOps<F, D>>(
        setup: &Self::ProverSetup,
        poly_groups_by_point: &[&[&[P]]],
        opening_points: &[&[F]],
        hints_by_point: Vec<Self::BatchedCommitHint>,
        transcript: &mut T,
        commitments_by_point: &[&[Self::Commitment]],
        basis: BasisMode,
    ) -> Result<Self::BatchedProof, HachiError> {
        if opening_points.is_empty() {
            return Err(HachiError::InvalidInput(
                "batched_prove requires at least one opening point".to_string(),
            ));
        }
        let num_vars = opening_points[0].len();
        if opening_points.iter().any(|pt| pt.len() != num_vars) {
            return Err(HachiError::InvalidInput(
                "batched_prove requires all opening points to have the same length".to_string(),
            ));
        }
        if num_vars > setup.expanded.seed.max_num_vars {
            return Err(HachiError::InvalidInput(format!(
                "batched_prove received opening points with {} variables but setup supports at most {}",
                num_vars, setup.expanded.seed.max_num_vars
            )));
        }
        if opening_points.len() > setup.expanded.seed.max_num_points {
            return Err(HachiError::InvalidInput(format!(
                "batched_prove received {} opening points but setup supports at most {}",
                opening_points.len(),
                setup.expanded.seed.max_num_points
            )));
        }
        let batch_shape =
            validate_nonempty_group_sizes_by_point(poly_groups_by_point, "batched_prove")?;
        if opening_points.len() != batch_shape.point_group_sizes.len()
            || commitments_by_point.len() != batch_shape.point_group_sizes.len()
            || hints_by_point.len() != batch_shape.point_group_sizes.len()
        {
            return Err(HachiError::InvalidInput(
                "batched_prove points, commitments, hints, and polynomial groups must have matching outer lengths"
                    .to_string(),
            ));
        }
        if commitments_by_point
            .iter()
            .zip(batch_shape.point_group_sizes.iter())
            .any(|(commitments, &group_count)| commitments.len() != group_count)
            || hints_by_point
                .iter()
                .zip(batch_shape.point_group_sizes.iter())
                .any(|(hints, &group_count)| hints.len() != group_count)
        {
            return Err(HachiError::InvalidInput(
                "batched_prove per-point group counts must match commitments and hints".to_string(),
            ));
        }
        if opening_points.len() == 1 {
            let mut hints_iter = hints_by_point.into_iter();
            return prove_same_point_batched::<F, T, D, Cfg, P>(
                setup,
                poly_groups_by_point[0],
                opening_points[0],
                hints_iter
                    .next()
                    .ok_or_else(|| HachiError::InvalidInput("missing batched hint".to_string()))?,
                transcript,
                commitments_by_point[0],
                basis,
            );
        }

        let total_claims = checked_total_claims(&batch_shape.claim_group_sizes, "batched_prove")?;
        if total_claims > setup.expanded.seed.max_num_batched_polys {
            return Err(HachiError::InvalidInput(format!(
                "batched_prove received {total_claims} polynomials but setup supports at most {}",
                setup.expanded.seed.max_num_batched_polys
            )));
        }

        let t_prove_total = Instant::now();
        let mut ntt_cache = MultiDNttCaches::new();
        let mut commit_ntt_cache = MultiDNttCaches::new();
        let max_num_vars = setup.expanded.seed.max_num_vars;
        let root_plan = hachi_root_runtime_plan_with_batch::<Cfg, D>(
            max_num_vars,
            num_vars,
            setup.expanded.seed.max_num_batched_polys,
            HachiRootBatchSummary::from_claim_group_sizes(
                &batch_shape.claim_group_sizes,
                opening_points.len(),
            )?,
        )?;
        let root_lp = root_plan.root_lp.clone();
        let batched_lp = root_plan.level_lp.clone();

        let alpha_bits = root_lp.ring_dimension.trailing_zeros() as usize;
        let prepared_points = opening_points
            .iter()
            .map(|opening_point| {
                prepare_root_opening_point::<F, D>(opening_point, basis, &root_lp, alpha_bits)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let flat_commitments = flatten_commitments_by_point(commitments_by_point);
        if flat_commitments
            .iter()
            .any(|commitment| commitment.u.len() != root_lp.b_key.row_len())
        {
            return Err(HachiError::InvalidInput(
                "batched_prove received a commitment with the wrong length".to_string(),
            ));
        }
        let flat_polys = flatten_poly_groups_by_point(poly_groups_by_point);
        let flat_hints = flatten_batched_hints_by_point(hints_by_point);
        let max_stride = setup.expanded.seed.max_stride;

        let commit_fn_0: CommitFn<'_, F> = Box::new(
            |w: &RecursiveWitnessFlat,
             next_params: LevelParams|
             -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), HachiError> {
                if next_params.ring_dimension == D {
                    let (wc, wh) =
                        commit_w::<F, D, Cfg>(w, &setup.ntt_shared, &next_params, max_stride)?;
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
        let BatchedProveLevelOutput {
            root_proof,
            next_state,
        } = prove_multipoint_batched_root_level::<F, T, D, Cfg, P>(
            &setup.expanded,
            &setup.ntt_shared,
            commit_fn_0,
            &flat_polys,
            &batch_shape,
            &prepared_points,
            max_num_vars,
            root_plan.lookup_key(),
            flat_hints,
            transcript,
            &flat_commitments,
            &root_lp,
            &batched_lp,
            root_plan.planning_envelope,
        )?;

        let (levels, level, current_state) = prove_batched_recursive_suffix::<F, T, D, Cfg>(
            setup,
            &mut ntt_cache,
            &mut commit_ntt_cache,
            max_num_vars,
            transcript,
            next_state,
            root_plan.inputs.current_w_len,
        )?;

        tracing::info!(
            levels = level,
            elapsed_s = t_prove_total.elapsed().as_secs_f64(),
            "hachi batched prove complete"
        );

        Ok(finalize_batched_recursive_proof::<F, D, Cfg>(
            max_num_vars,
            root_proof,
            levels,
            level,
            current_state,
        ))
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
    ) -> Result<(), HachiError> {
        let num_vars = opening_point.len();
        if num_vars > setup.expanded.seed.max_num_vars {
            return Err(HachiError::InvalidInput(format!(
                "verify received an opening point with {} variables but setup supports at most {}",
                num_vars, setup.expanded.seed.max_num_vars
            )));
        }
        if proof.num_fold_levels() == 0 {
            if !schedule_uses_root_direct::<Cfg>(num_vars)? {
                return Err(HachiError::InvalidProof);
            }
            return verify_root_direct::<F, T, D, Cfg>(
                proof,
                setup,
                transcript,
                opening_point,
                opening,
                commitment,
                basis,
            );
        }
        let max_num_vars = setup.expanded.seed.max_num_vars;
        let root_plan = hachi_root_runtime_plan::<Cfg, D>(
            max_num_vars,
            num_vars,
            setup.expanded.seed.max_num_batched_polys,
        )
        .map_err(|_| HachiError::InvalidProof)?;
        let exact_plan = if num_vars == max_num_vars {
            Cfg::schedule_plan(root_plan.lookup_key()).map_err(|_| HachiError::InvalidProof)?
        } else {
            None
        };
        let root_lp = root_plan.root_lp.clone();
        let t_verify_hachi = Instant::now();

        let num_levels = proof.num_fold_levels();

        let final_w = Some(proof.final_witness());

        let root_commitment = FlatRingVec::from_ring_elems(&commitment.u);
        let mut current_state = RecursiveVerifierState {
            opening_point: opening_point.to_vec(),
            opening: *opening,
            commitment: &root_commitment,
            basis,
            w_len: root_plan.inputs.current_w_len,
            log_basis: root_lp.log_basis,
            root_key: root_plan.lookup_key(),
            planning_envelope: root_plan.planning_envelope,
        };
        if let Some(plan) = exact_plan.as_ref() {
            let planned_root =
                exact_planned_level_execution::<Cfg>(plan, root_plan.inputs, root_lp.log_basis)
                    .map_err(|_| HachiError::InvalidProof)?
                    .ok_or(HachiError::InvalidProof)?;
            if planned_root.level.lp != root_lp {
                return Err(HachiError::InvalidProof);
            }
            if num_levels != plan.num_fold_levels() {
                return Err(HachiError::InvalidProof);
            }
        }

        for (i, level_proof) in proof.fold_levels().enumerate() {
            let is_last = i == num_levels - 1;
            let inputs = HachiScheduleInputs {
                max_num_vars,
                level: i,
                current_w_len: current_state.w_len,
            };
            let planned_level = if let Some(plan) = exact_plan.as_ref() {
                Some(
                    exact_planned_level_execution::<Cfg>(plan, inputs, current_state.log_basis)
                        .map_err(|_| HachiError::InvalidProof)?
                        .ok_or(HachiError::InvalidProof)?,
                )
            } else {
                None
            };
            let (current_lp, planned_next_level_params, planned_next_w_len) =
                if let Some(planned_level) = planned_level {
                    if i == 0 && planned_level.level.lp != root_lp {
                        return Err(HachiError::InvalidProof);
                    }
                    (
                        planned_level.level.lp,
                        Some(planned_level.next_level_params),
                        Some(planned_level.level.next_inputs.current_w_len),
                    )
                } else {
                    let level_params =
                        Cfg::level_params_with_log_basis(inputs, current_state.log_basis);
                    let current_lp = if i == 0 {
                        root_lp.clone()
                    } else {
                        hachi_recursive_level_layout_from_params::<Cfg>(
                            &level_params,
                            current_state.w_len,
                        )?
                    };
                    (current_lp, None, None)
                };
            let level_d = current_lp.ring_dimension;
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

            let block_order = if i == 0 {
                BlockOrder::RowMajor
            } else {
                BlockOrder::ColumnMajor
            };
            let challenges = if level_d == D {
                verify_one_level::<F, T, D>(
                    level_proof,
                    setup,
                    transcript,
                    &current_state,
                    is_last,
                    if is_last { final_w } else { None },
                    &current_lp,
                    block_order,
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
                    &current_lp,
                    block_order,
                )?
            };

            if !is_last {
                let next_w_len = w_ring_element_count::<F>(&current_lp) * level_d;
                if let Some(planned_next_w_len) = planned_next_w_len {
                    if next_w_len != planned_next_w_len {
                        return Err(HachiError::InvalidProof);
                    }
                }
                let next_level_params = if let Some(next_level_params) = planned_next_level_params {
                    next_level_params
                } else {
                    next_level_params_from_current_basis_and_envelope::<Cfg>(
                        current_state.root_key,
                        HachiScheduleInputs {
                            max_num_vars,
                            level: i + 1,
                            current_w_len: next_w_len,
                        },
                        current_state.log_basis,
                        current_state.planning_envelope,
                    )
                    .map_err(|_| HachiError::InvalidProof)?
                };

                if i + 1 < num_levels {
                    let next_level_d = next_level_params.ring_dimension;
                    if !level_proof.next_w_commitment().can_decode_vec(next_level_d) {
                        return Err(HachiError::InvalidProof);
                    }
                }
                current_state = RecursiveVerifierState {
                    opening_point: challenges,
                    opening: level_proof.next_w_eval(),
                    commitment: level_proof.next_w_commitment(),
                    basis: BasisMode::Lagrange,
                    w_len: next_w_len,
                    log_basis: next_level_params.log_basis,
                    root_key: current_state.root_key,
                    planning_envelope: current_state.planning_envelope,
                };
            }
        }

        tracing::info!(
            levels = num_levels,
            elapsed_s = t_verify_hachi.elapsed().as_secs_f64(),
            "hachi verify complete"
        );

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::batched_verify")]
    fn batched_verify<T: Transcript<F>>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        opening_points: &[&[F]],
        opening_groups_by_point: &[&[&[F]]],
        commitments_by_point: &[&[Self::Commitment]],
        basis: BasisMode,
    ) -> Result<(), HachiError> {
        if opening_points.is_empty() {
            return Err(HachiError::InvalidProof);
        }
        let num_vars = opening_points[0].len();
        if opening_points.iter().any(|pt| pt.len() != num_vars) {
            return Err(HachiError::InvalidProof);
        }
        if num_vars > setup.expanded.seed.max_num_vars {
            return Err(HachiError::InvalidInput(format!(
                "batched_verify received opening points with {} variables but setup supports at most {}",
                num_vars, setup.expanded.seed.max_num_vars
            )));
        }
        if opening_points.len() > setup.expanded.seed.max_num_points {
            return Err(HachiError::InvalidProof);
        }
        let y_coeff_len = proof.root.y_rings().coeff_len();
        if !y_coeff_len.is_multiple_of(D) {
            return Err(HachiError::InvalidProof);
        }
        let batch_shape =
            validate_nonempty_group_sizes_by_point(opening_groups_by_point, "batched_verify")
                .map_err(|_| HachiError::InvalidProof)?;
        let num_claims = checked_total_claims(&batch_shape.claim_group_sizes, "batched_verify")
            .map_err(|_| HachiError::InvalidProof)?;
        let openings = flatten_value_groups_by_point(opening_groups_by_point);
        if opening_points.len() != batch_shape.point_group_sizes.len()
            || commitments_by_point.len() != batch_shape.point_group_sizes.len()
            || num_claims == 0
            || openings.len() != num_claims
            || commitments_by_point
                .iter()
                .zip(batch_shape.point_group_sizes.iter())
                .any(|(commitments, &group_count)| commitments.len() != group_count)
        {
            return Err(HachiError::InvalidProof);
        }
        if y_coeff_len / D != opening_points.len() {
            return Err(HachiError::InvalidProof);
        }
        if num_claims > setup.expanded.seed.max_num_batched_polys {
            return Err(HachiError::InvalidProof);
        }
        if opening_points.len() == 1 {
            return verify_same_point_batched::<F, T, D, Cfg>(
                proof,
                setup,
                transcript,
                opening_points[0],
                opening_groups_by_point[0],
                commitments_by_point[0],
                basis,
            );
        }

        let t_verify_hachi = Instant::now();
        let max_num_vars = setup.expanded.seed.max_num_vars;
        let root_plan = hachi_root_runtime_plan_with_batch::<Cfg, D>(
            max_num_vars,
            num_vars,
            setup.expanded.seed.max_num_batched_polys,
            HachiRootBatchSummary::from_claim_group_sizes(
                &batch_shape.claim_group_sizes,
                opening_points.len(),
            )
            .map_err(|_| HachiError::InvalidProof)?,
        )
        .map_err(|_| HachiError::InvalidProof)?;
        let root_lp = root_plan.root_lp.clone();
        let batched_lp = root_plan.level_lp.clone();

        let final_w = Some(proof.final_witness());
        let alpha_bits = root_lp.ring_dimension.trailing_zeros() as usize;
        let prepared_points = opening_points
            .iter()
            .map(|opening_point| {
                prepare_root_opening_point::<F, D>(opening_point, basis, &root_lp, alpha_bits)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| HachiError::InvalidProof)?;
        let flat_commitments = flatten_commitments_by_point(commitments_by_point);

        let has_recursive_levels = proof.num_fold_levels() > 0;
        let root_challenges = verify_multipoint_batched_root_level::<F, T, D>(
            &proof.root,
            setup,
            transcript,
            &prepared_points,
            &openings,
            &flat_commitments,
            &batch_shape,
            &root_lp,
            &batched_lp,
            !has_recursive_levels,
            if has_recursive_levels { None } else { final_w },
        )?;

        if has_recursive_levels {
            let root_w_len = root_plan.next_w_len();
            let first_level_d = root_plan.next_level_params.ring_dimension;
            if !proof.root.next_w_commitment().can_decode_vec(first_level_d) {
                return Err(HachiError::InvalidProof);
            }

            let current_state = RecursiveVerifierState {
                opening_point: root_challenges,
                opening: proof.root.next_w_eval(),
                commitment: proof.root.next_w_commitment(),
                basis: BasisMode::Lagrange,
                w_len: root_w_len,
                log_basis: root_plan.next_level_params.log_basis,
                root_key: root_plan.lookup_key(),
                planning_envelope: root_plan.planning_envelope,
            };
            verify_batched_recursive_suffix::<F, T, D, Cfg>(
                proof,
                setup,
                transcript,
                max_num_vars,
                current_state,
                final_w,
            )?;
        }

        tracing::info!(
            levels = proof.num_fold_levels() + 1,
            elapsed_s = t_verify_hachi.elapsed().as_secs_f64(),
            "hachi batched verify complete"
        );

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Hachi"
    }
}

/// Verify the specialized batched root level.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn verify_batched_root_level<F, T, const D: usize>(
    root_proof: &HachiBatchedRootProof<F>,
    setup: &HachiVerifierSetup<F>,
    transcript: &mut T,
    opening_point: &[F],
    openings: &[F],
    commitments: &[RingCommitment<F, D>],
    claim_group_sizes: &[usize],
    basis: BasisMode,
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
) -> Result<Vec<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    let y_rings = root_proof.y_rings().as_ring_slice::<D>()?;
    let v_typed = root_proof.v().as_ring_slice::<D>()?;
    let num_claims = checked_total_claims(claim_group_sizes, "batched_verify")
        .map_err(|_| HachiError::InvalidProof)?;
    if y_rings.len() != 1
        || openings.len() != num_claims
        || commitments.len() != claim_group_sizes.len()
    {
        return Err(HachiError::InvalidProof);
    }
    if commitments
        .iter()
        .any(|commitment| commitment.u.len() != root_lp.b_key.row_len())
    {
        return Err(HachiError::InvalidProof);
    }
    let commitment_rows = flatten_batched_commitment_rows(commitments);

    let alpha_bits = root_lp.ring_dimension.trailing_zeros() as usize;
    if opening_point.len() < alpha_bits {
        return Err(HachiError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    let target_num_vars = root_lp.m_vars + root_lp.r_vars + alpha_bits;
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let inner_point = &padded_point[..alpha_bits];
    let reduced_opening_point = &padded_point[alpha_bits..];

    append_batched_commitments_to_transcript(commitments, transcript);
    for pt in &padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    for opening in openings {
        transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, opening);
    }
    let gamma: Vec<F> = (0..openings.len())
        .map(|_| transcript.challenge_scalar(CHALLENGE_EVAL_BATCH))
        .collect();
    for y_ring in y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    let v = reduce_inner_opening_to_ring_element::<F, { D }>(inner_point, basis)?;
    let d_field = F::from_u64(root_lp.ring_dimension as u64);
    let num_points = 1usize;
    let batched_opening: F = openings
        .iter()
        .zip(gamma.iter())
        .fold(F::zero(), |acc, (opening, &g)| acc + g * *opening);
    {
        let trace_lhs = trace::<F, { D }>(&(y_rings[0] * v.sigma_m1()));
        let trace_rhs = d_field * batched_opening;
        if trace_lhs != trace_rhs {
            return Err(HachiError::InvalidProof);
        }
    }

    let ring_opening_point = ring_opening_point_from_field::<F>(
        reduced_opening_point,
        root_lp.r_vars,
        root_lp.m_vars,
        basis,
        BlockOrder::RowMajor,
    )?;
    let total_blocks = root_lp
        .num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| HachiError::InvalidSetup("batched root block count overflow".to_string()))?;
    let stage1_challenges =
        derive_stage1_challenges::<F, T, D>(transcript, v_typed, total_blocks, batched_lp)?;

    let w_len = if is_last {
        final_w.map_or(0, DirectWitnessProof::num_elems)
    } else {
        w_ring_element_count_with_claim_groups::<F>(batched_lp, claim_group_sizes, num_points) * D
    };

    let rs = ring_switch_verifier_with_claim_groups::<F, T, { D }>(
        &ring_opening_point,
        &stage1_challenges,
        w_len,
        root_proof.next_w_commitment(),
        transcript,
        batched_lp,
        claim_group_sizes,
        &gamma,
        num_points,
    )?;
    let relation_claim =
        relation_claim_from_rows(&rs.tau1, rs.alpha, v_typed, &commitment_rows, y_rings);
    let stage1 = &root_proof.stage1;
    let stage2 = &root_proof.stage2;
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_verifier = HachiStage1Verifier::new(tau0_reordered, rs.b);
    let r_stage1 = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify(stage1, transcript)?
    };
    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let stage2_input_claim = batching_coeff * stage1.s_claim + relation_claim;
    let m_eval_source = Stage2MEvalSource::new(rs.prepared_m_eval);
    let ring_opening_points_slice = std::slice::from_ref(&ring_opening_point);
    let stage2_verifier = if is_last {
        let fw = final_w.ok_or(HachiError::InvalidProof)?;
        HachiStage2Verifier::new_with_direct_witness_batched(
            batching_coeff,
            stage1.s_claim,
            fw,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            ring_opening_points_slice,
            &rs.tau1,
            v_typed,
            &commitment_rows,
            y_rings,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    } else {
        HachiStage2Verifier::new_with_claimed_w_eval_batched(
            batching_coeff,
            stage1.s_claim,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            ring_opening_points_slice,
            &rs.tau1,
            v_typed,
            &commitment_rows,
            y_rings,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
            stage2.next_w_eval,
        )
    };
    if stage2_input_claim != SumcheckInstanceVerifier::input_claim(&stage2_verifier) {
        return Err(HachiError::InvalidProof);
    }
    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        verify_sumcheck::<F, _, F, _, _>(&stage2.sumcheck, &stage2_verifier, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
        })?
    };

    Ok(sumcheck_challenges)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn verify_multipoint_batched_root_level<F, T, const D: usize>(
    root_proof: &HachiBatchedRootProof<F>,
    setup: &HachiVerifierSetup<F>,
    transcript: &mut T,
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    openings: &[F],
    commitments: &[RingCommitment<F, D>],
    batch_shape: &MultiPointBatchShape,
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
) -> Result<Vec<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    let y_rings = root_proof.y_rings().as_ring_slice::<D>()?;
    let v_typed = root_proof.v().as_ring_slice::<D>()?;
    let num_claims =
        checked_total_claims(&batch_shape.claim_group_sizes, "multipoint_batched_verify")
            .map_err(|_| HachiError::InvalidProof)?;
    if prepared_points.is_empty()
        || y_rings.len() != prepared_points.len()
        || openings.len() != num_claims
        || commitments.len() != batch_shape.claim_group_sizes.len()
        || batch_shape.claim_to_point.len() != num_claims
    {
        return Err(HachiError::InvalidProof);
    }
    if commitments
        .iter()
        .any(|commitment| commitment.u.len() != root_lp.b_key.row_len())
    {
        return Err(HachiError::InvalidProof);
    }
    let commitment_rows = flatten_batched_commitment_rows(commitments);

    append_batch_shape_to_transcript::<F, T>(
        &batch_shape.point_group_sizes,
        &batch_shape.claim_group_sizes,
        transcript,
    );
    append_batched_commitments_to_transcript(commitments, transcript);
    for prepared_point in prepared_points {
        for pt in &prepared_point.padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    for opening in openings {
        transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, opening);
    }
    let gamma: Vec<F> = (0..openings.len())
        .map(|_| transcript.challenge_scalar(CHALLENGE_EVAL_BATCH))
        .collect();
    for y_ring in y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    let d_field = F::from_u64(root_lp.ring_dimension as u64);
    let num_points = prepared_points.len();
    let mut batched_openings = vec![F::zero(); num_points];
    for (claim_idx, (&opening, &g)) in openings.iter().zip(gamma.iter()).enumerate() {
        let point_idx = batch_shape.claim_to_point[claim_idx];
        batched_openings[point_idx] += g * opening;
    }
    for (point_idx, (y_ring, &batched_opening)) in
        y_rings.iter().zip(batched_openings.iter()).enumerate()
    {
        let v = &prepared_points[point_idx].inner_reduction;
        let trace_lhs = trace::<F, { D }>(&(*y_ring * v.sigma_m1()));
        let trace_rhs = d_field * batched_opening;
        if trace_lhs != trace_rhs {
            return Err(HachiError::InvalidProof);
        }
    }

    let total_blocks = root_lp
        .num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| HachiError::InvalidSetup("batched root block count overflow".to_string()))?;
    let stage1_challenges =
        derive_stage1_challenges::<F, T, D>(transcript, v_typed, total_blocks, batched_lp)?;

    let w_len = if is_last {
        final_w.map_or(0, DirectWitnessProof::num_elems)
    } else {
        w_ring_element_count_with_claim_groups::<F>(
            batched_lp,
            &batch_shape.claim_group_sizes,
            num_points,
        ) * D
    };

    let ring_opening_points: Vec<RingOpeningPoint<F>> = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_opening_point.clone())
        .collect();
    let rs = ring_switch_verifier_with_opening_points_and_claim_groups::<F, T, { D }>(
        &ring_opening_points,
        &batch_shape.claim_to_point,
        &stage1_challenges,
        w_len,
        root_proof.next_w_commitment(),
        transcript,
        batched_lp,
        &batch_shape.claim_group_sizes,
        &gamma,
        num_points,
    )?;
    let relation_claim =
        relation_claim_from_rows(&rs.tau1, rs.alpha, v_typed, &commitment_rows, y_rings);
    let stage1 = &root_proof.stage1;
    let stage2 = &root_proof.stage2;
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_verifier = HachiStage1Verifier::new(tau0_reordered, rs.b);
    let r_stage1 = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify(stage1, transcript)?
    };
    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let stage2_input_claim = batching_coeff * stage1.s_claim + relation_claim;
    let m_eval_source = Stage2MEvalSource::new(rs.prepared_m_eval);
    let stage2_verifier = if is_last {
        let fw = final_w.ok_or(HachiError::InvalidProof)?;
        HachiStage2Verifier::new_with_direct_witness_batched(
            batching_coeff,
            stage1.s_claim,
            fw,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &rs.tau1,
            v_typed,
            &commitment_rows,
            y_rings,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    } else {
        HachiStage2Verifier::new_with_claimed_w_eval_batched(
            batching_coeff,
            stage1.s_claim,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &rs.tau1,
            v_typed,
            &commitment_rows,
            y_rings,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
            stage2.next_w_eval,
        )
    };
    if stage2_input_claim != SumcheckInstanceVerifier::input_claim(&stage2_verifier) {
        return Err(HachiError::InvalidProof);
    }
    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        verify_sumcheck::<F, _, F, _, _>(&stage2.sumcheck, &stage2_verifier, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
        })?
    };

    Ok(sumcheck_challenges)
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
    final_w: Option<&DirectWitnessProof<F>>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    let y_ring = level_proof.y_ring.as_single_ring::<D>()?;
    let v_typed = level_proof.v.as_ring_slice::<D>()?;
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;

    let alpha_bits = lp.ring_dimension.trailing_zeros() as usize;
    if current_state.opening_point.len() < alpha_bits {
        return Err(HachiError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    let target_num_vars = lp.m_vars + lp.r_vars + alpha_bits;
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
    let d = F::from_u64(lp.ring_dimension as u64);
    let trace_lhs = trace::<F, { D }>(&(*y_ring * v.sigma_m1()));
    let trace_rhs = d * current_state.opening;
    if trace_lhs != trace_rhs {
        return Err(HachiError::InvalidProof);
    }

    let ring_opening_point = ring_opening_point_from_field::<F>(
        reduced_opening_point,
        lp.r_vars,
        lp.m_vars,
        current_state.basis,
        block_order,
    )?;
    let stage1_challenges =
        derive_stage1_challenges::<F, T, D>(transcript, v_typed, lp.num_blocks, lp)?;

    let w_len = if is_last {
        final_w.map_or(0, DirectWitnessProof::num_elems)
    } else {
        w_ring_element_count::<F>(lp) * D
    };
    tracing::debug!(w_len, is_last, "verify ring_switch");

    let rs = ring_switch_verifier::<F, T, { D }>(
        &ring_opening_point,
        &stage1_challenges,
        w_len,
        level_proof.next_w_commitment(),
        transcript,
        lp,
    )?;
    let relation_claim = relation_claim_from_rows(
        &rs.tau1,
        rs.alpha,
        v_typed,
        commitment_u,
        std::slice::from_ref(y_ring),
    );
    let stage1 = &level_proof.stage1;
    let stage2 = &level_proof.stage2;
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_verifier = HachiStage1Verifier::new(tau0_reordered, rs.b);
    let r_stage1 = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify(stage1, transcript)?
    };

    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let stage2_input_claim = batching_coeff * stage1.s_claim + relation_claim;
    let m_eval_source = Stage2MEvalSource::new(rs.prepared_m_eval);
    let ring_opening_points_slice = std::slice::from_ref(&ring_opening_point);

    let stage2_verifier = if is_last {
        let fw = final_w.ok_or(HachiError::InvalidProof)?;
        HachiStage2Verifier::new_with_direct_witness(
            batching_coeff,
            stage1.s_claim,
            fw,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            ring_opening_points_slice,
            &rs.tau1,
            v_typed,
            commitment_u,
            y_ring,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    } else {
        HachiStage2Verifier::new_with_claimed_w_eval(
            batching_coeff,
            stage1.s_claim,
            stage2.next_w_eval,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            ring_opening_points_slice,
            &rs.tau1,
            v_typed,
            commitment_u,
            y_ring,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
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
    use crate::algebra::Fp128;
    use crate::primitives::serialization::Compress;
    use crate::protocol::commitment::presets::fp128;
    use crate::protocol::commitment::schedule::{
        hachi_root_runtime_plan_with_batch, recursive_suffix_estimate_with_log_basis,
    };
    use crate::protocol::commitment::{CommitmentConfig, HachiRootBatchSummary};
    use crate::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps, OneHotPoly};
    use crate::protocol::opening_point::{
        lagrange_weights, monomial_weights, reduce_inner_opening_to_ring_element,
        ring_opening_point_from_field,
    };
    use crate::protocol::proof::{HachiBatchedProofShape, HachiProofStepShape, LevelProofShape};
    use crate::protocol::sumcheck::hachi_stage1_tree::stage1_tree_stage_shapes;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::{CommitmentScheme, FromSmallInt, HachiDeserialize};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::sync::Once;
    use tracing_subscriber::fmt::format::FmtSpan;
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::EnvFilter;
    type Cfg = fp128::D64Full;
    type F = fp128::Field;
    const D: usize = Cfg::D;
    type Scheme = HachiCommitmentScheme<D, Cfg>;
    type OneHotF = Fp128<0xfffffffffffffffffffffffffffff6cd>;
    type OneHotCfg = fp128::D64OneHot;
    const ONEHOT_D: usize = OneHotCfg::D;
    const BENCH_ONEHOT_K: usize = ONEHOT_D;
    type OneHotScheme = HachiCommitmentScheme<ONEHOT_D, OneHotCfg>;

    fn batched_shape_rounds(level_d: usize, next_w_len: usize) -> usize {
        let num_ring_elems = next_w_len / level_d;
        num_ring_elems.next_power_of_two().trailing_zeros() as usize
            + level_d.trailing_zeros() as usize
    }

    #[test]
    fn same_point_batched_root_preserves_opening_geometry() {
        for num_claims in [4usize, 6] {
            let root_plan = hachi_root_runtime_plan_with_batch::<OneHotCfg, ONEHOT_D>(
                20,
                20,
                num_claims,
                HachiRootBatchSummary::new(num_claims, 1, 1).expect("same-point batch summary"),
            )
            .expect("same-point root plan");
            assert_eq!(root_plan.root_lp.block_len, root_plan.level_lp.block_len);
            assert_eq!(root_plan.root_lp.num_blocks, root_plan.level_lp.num_blocks);
            assert_eq!(root_plan.root_lp.m_vars, root_plan.level_lp.m_vars);
            assert_eq!(root_plan.root_lp.r_vars, root_plan.level_lp.r_vars);
        }
    }

    fn expected_same_point_batched_shape(
        max_num_vars: usize,
        num_claims: usize,
        proof: &HachiBatchedProof<OneHotF>,
    ) -> HachiBatchedProofShape {
        let root_plan = hachi_root_runtime_plan_with_batch::<OneHotCfg, ONEHOT_D>(
            max_num_vars,
            max_num_vars,
            num_claims,
            HachiRootBatchSummary::new(num_claims, 1, 1).expect("same-point batch summary"),
        )
        .expect("batched root runtime plan");
        let root_w_len = root_plan.next_w_len();
        let root_shape = root_plan.level_proof_shape();
        let first_level_params = root_plan.next_level_params.clone();

        let mut step_shapes = Vec::with_capacity(proof.num_fold_levels() + 1);
        let mut current_w_len = root_w_len;
        let mut current_log_basis = first_level_params.log_basis;
        let mut current_level = 1usize;
        for _ in proof.fold_levels() {
            let inputs = HachiScheduleInputs {
                max_num_vars,
                level: current_level,
                current_w_len,
            };
            let level_params = OneHotCfg::level_params_with_log_basis(inputs, current_log_basis);
            let current_lp =
                hachi_recursive_level_layout_from_params::<OneHotCfg>(&level_params, current_w_len)
                    .expect("recursive layout");
            let next_w_len =
                w_ring_element_count::<OneHotF>(&current_lp) * current_lp.ring_dimension;
            let next_level_params = next_level_params_from_current_basis_and_envelope::<OneHotCfg>(
                root_plan.lookup_key(),
                HachiScheduleInputs {
                    max_num_vars,
                    level: current_level + 1,
                    current_w_len: next_w_len,
                },
                current_log_basis,
                root_plan.planning_envelope,
            )
            .expect("next recursive params");
            let rounds = batched_shape_rounds(current_lp.ring_dimension, next_w_len);
            step_shapes.push(HachiProofStepShape::Fold(LevelProofShape {
                y_ring_coeffs: current_lp.ring_dimension,
                v_coeffs: current_lp.d_key.row_len() * current_lp.ring_dimension,
                stage1_stages: stage1_tree_stage_shapes(rounds, 1usize << current_lp.log_basis),
                stage2_sumcheck: (rounds, 3),
                next_commit_coeffs: next_level_params.b_key.row_len()
                    * next_level_params.ring_dimension,
            }));
            current_w_len = next_w_len;
            current_log_basis = next_level_params.log_basis;
            current_level += 1;
        }
        step_shapes.push(HachiProofStepShape::Direct(
            crate::protocol::proof::DirectWitnessShape::PackedDigits((
                current_w_len,
                current_log_basis,
            )),
        ));

        HachiBatchedProofShape {
            root_shape,
            step_shapes,
        }
    }

    fn make_dense_poly(num_vars: usize) -> (DensePoly<F, D>, Vec<F>) {
        let len = 1usize << num_vars;
        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
        (poly, evals)
    }

    #[test]
    fn batched_suffix_stop_guard_does_not_preempt_profitable_fold() {
        type D64Cfg = fp128::D64OneHot;
        type D32Cfg = fp128::D32OneHot;

        // These states came from the batched onehot nv=32 profile runs that
        // regressed after adding the generic shrink-ratio guard to the batched
        // suffix. They still have profitable recursive suffixes by bytes.
        assert!(should_stop_folding(87_744, 140_672));
        assert!(!should_stop_batched_folding(87_744, 140_672));
        assert!(should_continue_folding_by_bytes::<D64Cfg>(
            HachiScheduleLookupKey::singleton(32, 32, 1),
            5,
            87_744,
            5,
        )
        .unwrap());

        assert!(should_stop_folding(129_216, 224_064));
        assert!(!should_stop_batched_folding(129_216, 224_064));
        assert!(should_continue_folding_by_bytes::<D32Cfg>(
            HachiScheduleLookupKey::singleton(32, 32, 1),
            5,
            129_216,
            4,
        )
        .unwrap());
    }

    fn assert_batched_onehot_planner_gap<const D_LOCAL: usize, CfgLocal>(
        nv: usize,
        batch_size: usize,
        slack_bytes: usize,
    ) where
        CfgLocal: CommitmentConfig<Field = OneHotF>,
    {
        type SchemeLocal<const D_INNER: usize, CfgInner> = HachiCommitmentScheme<D_INNER, CfgInner>;

        let layout =
            hachi_batched_root_layout::<CfgLocal, D_LOCAL>(nv, batch_size).expect("batched layout");
        let polys: Vec<OneHotPoly<OneHotF, D_LOCAL, u8>> = (0..batch_size)
            .map(|poly_idx| {
                debug_make_onehot_poly_generic::<D_LOCAL>(
                    &layout,
                    0x0bee_fcaf_e000_2000 + poly_idx as u64,
                )
            })
            .collect();
        let poly_refs: Vec<&OneHotPoly<OneHotF, D_LOCAL, u8>> = polys.iter().collect();
        let point = debug_random_point(nv);
        let openings: Vec<OneHotF> = polys
            .iter()
            .map(|poly| debug_opening_from_poly_generic::<D_LOCAL, _>(poly, &point, &layout))
            .collect();
        let poly_groups = [&poly_refs[..]];
        let opening_groups = [&openings[..]];

        let setup =
            <SchemeLocal<D_LOCAL, CfgLocal> as CommitmentScheme<OneHotF, D_LOCAL>>::setup_prover(
                nv, batch_size, 1,
            );
        let verifier_setup = <SchemeLocal<D_LOCAL, CfgLocal> as CommitmentScheme<
            OneHotF,
            D_LOCAL,
        >>::setup_verifier(&setup);
        let (commitment, hint) = <SchemeLocal<D_LOCAL, CfgLocal> as CommitmentScheme<
            OneHotF,
            D_LOCAL,
        >>::commit(&poly_refs, &setup)
        .expect("batched onehot commit");
        let commitments = [commitment];
        let hints = vec![hint];

        let mut prover_transcript =
            Blake2bTranscript::<OneHotF>::new(b"test/batched-onehot-planner-gap");
        let proof =
            <SchemeLocal<D_LOCAL, CfgLocal> as CommitmentScheme<OneHotF, D_LOCAL>>::batched_prove(
                &setup,
                &[&poly_groups[..]],
                &[&point[..]],
                vec![hints],
                &mut prover_transcript,
                &[&commitments[..]],
                BasisMode::Lagrange,
            )
            .expect("batched onehot prove");

        let mut verifier_transcript =
            Blake2bTranscript::<OneHotF>::new(b"test/batched-onehot-planner-gap");
        <SchemeLocal<D_LOCAL, CfgLocal> as CommitmentScheme<OneHotF, D_LOCAL>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &[&point[..]],
            &[&opening_groups[..]],
            &[&commitments[..]],
            BasisMode::Lagrange,
        )
        .expect("batched onehot verify");

        let root_plan = hachi_root_runtime_plan_with_batch::<CfgLocal, D_LOCAL>(
            nv,
            nv,
            batch_size,
            HachiRootBatchSummary::new(batch_size, 1, 1).expect("same-point batch summary"),
        )
        .expect("batched root plan");
        let estimate = recursive_suffix_estimate_with_log_basis::<CfgLocal>(
            root_plan.lookup_key(),
            root_plan.next_inputs.level,
            root_plan.next_w_len(),
            root_plan.next_level_params.log_basis,
            root_plan.planning_envelope,
        )
        .expect("recursive suffix estimate");

        let root_bytes = root_plan.level_proof_bytes::<CfgLocal>();
        let observed_total = proof.size();
        let table_total = root_bytes + estimate.table_bytes;
        let actual_total = root_bytes + estimate.actual_state_bytes;
        let actual_gap = observed_total.abs_diff(actual_total);

        assert_eq!(root_bytes, proof.root.serialized_size(Compress::No));
        assert_eq!(table_total, observed_total);
        if estimate.exact_state_match {
            assert!(
                !estimate.used_actual_state_planner,
                "exact batch-4 onehot schedule should use the keyed generated row"
            );
            assert_eq!(actual_total, observed_total);
            assert_eq!(actual_gap, 0);
        } else {
            assert!(
                estimate.used_actual_state_planner,
                "off-table batch-4 onehot proof should use the actual-state miss-path planner"
            );
            assert!(
                actual_gap <= slack_bytes,
                "actual-state suffix gap {actual_gap} exceeded slack bound {slack_bytes}"
            );
        }
    }

    fn assert_blessed_batched_onehot_exact<const D_LOCAL: usize, CfgLocal>(
        nv: usize,
        group_sizes_by_point: &[&[usize]],
    ) where
        CfgLocal: CommitmentConfig<Field = OneHotF>,
    {
        type SchemeLocal<const D_INNER: usize, CfgInner> = HachiCommitmentScheme<D_INNER, CfgInner>;

        let total_claims = group_sizes_by_point
            .iter()
            .flat_map(|sizes| sizes.iter().copied())
            .sum::<usize>();
        let claim_group_sizes: Vec<usize> = group_sizes_by_point
            .iter()
            .flat_map(|sizes| sizes.iter().copied())
            .collect();
        let batch = HachiRootBatchSummary::from_claim_group_sizes(
            &claim_group_sizes,
            group_sizes_by_point.len(),
        )
        .expect("batch summary");

        let layout = hachi_batched_root_layout::<CfgLocal, D_LOCAL>(nv, total_claims)
            .expect("batched layout");
        let polys: Vec<OneHotPoly<OneHotF, D_LOCAL, u8>> = (0..total_claims)
            .map(|poly_idx| {
                debug_make_onehot_poly_generic::<D_LOCAL>(
                    &layout,
                    0x0bee_fcaf_e100_0000 + poly_idx as u64,
                )
            })
            .collect();

        let setup =
            <SchemeLocal<D_LOCAL, CfgLocal> as CommitmentScheme<OneHotF, D_LOCAL>>::setup_prover(
                nv,
                total_claims,
                group_sizes_by_point.len(),
            );
        let verifier_setup = <SchemeLocal<D_LOCAL, CfgLocal> as CommitmentScheme<
            OneHotF,
            D_LOCAL,
        >>::setup_verifier(&setup);

        let mut poly_cursor = 0usize;
        let mut point_groups: Vec<Vec<Vec<&OneHotPoly<OneHotF, D_LOCAL, u8>>>> = Vec::new();
        let mut commitments_by_point_storage = Vec::new();
        let mut hints_by_point = Vec::new();
        let mut opening_points = Vec::new();
        let mut opening_groups_by_point_storage = Vec::new();

        for (point_idx, group_sizes) in group_sizes_by_point.iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(0xcafe_babe + point_idx as u64);
            let point: Vec<OneHotF> = (0..nv)
                .map(|_| OneHotF::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect();
            let mut groups_at_point = Vec::new();
            let mut commitments_at_point = Vec::new();
            let mut hints_at_point = Vec::new();
            let mut opening_groups_at_point = Vec::new();

            for &group_size in *group_sizes {
                let group: Vec<&OneHotPoly<OneHotF, D_LOCAL, u8>> = polys
                    [poly_cursor..poly_cursor + group_size]
                    .iter()
                    .collect();
                poly_cursor += group_size;
                let openings_for_group: Vec<OneHotF> = group
                    .iter()
                    .map(|poly| {
                        debug_opening_from_poly_generic::<D_LOCAL, _>(*poly, &point, &layout)
                    })
                    .collect();
                let (commitment, hint) = <SchemeLocal<D_LOCAL, CfgLocal> as CommitmentScheme<
                    OneHotF,
                    D_LOCAL,
                >>::commit(&group, &setup)
                .expect("batched group commit");
                groups_at_point.push(group);
                commitments_at_point.push(commitment);
                hints_at_point.push(hint);
                opening_groups_at_point.push(openings_for_group);
            }

            point_groups.push(groups_at_point);
            commitments_by_point_storage.push(commitments_at_point);
            hints_by_point.push(hints_at_point);
            opening_points.push(point);
            opening_groups_by_point_storage.push(opening_groups_at_point);
        }
        assert_eq!(poly_cursor, total_claims);

        let poly_group_slices: Vec<Vec<&[&OneHotPoly<OneHotF, D_LOCAL, u8>]>> = point_groups
            .iter()
            .map(|groups| groups.iter().map(Vec::as_slice).collect())
            .collect();
        let poly_groups_by_point: Vec<&[&[&OneHotPoly<OneHotF, D_LOCAL, u8>]]> =
            poly_group_slices.iter().map(Vec::as_slice).collect();
        let commitment_slices: Vec<&[_]> = commitments_by_point_storage
            .iter()
            .map(Vec::as_slice)
            .collect();
        let point_slices: Vec<&[OneHotF]> = opening_points.iter().map(Vec::as_slice).collect();
        let opening_group_slices: Vec<Vec<&[OneHotF]>> = opening_groups_by_point_storage
            .iter()
            .map(|groups| groups.iter().map(Vec::as_slice).collect())
            .collect();
        let opening_groups_by_point: Vec<&[&[OneHotF]]> =
            opening_group_slices.iter().map(Vec::as_slice).collect();

        let mut prover_transcript =
            Blake2bTranscript::<OneHotF>::new(b"test/blessed-batched-onehot-exact");
        let proof =
            <SchemeLocal<D_LOCAL, CfgLocal> as CommitmentScheme<OneHotF, D_LOCAL>>::batched_prove(
                &setup,
                &poly_groups_by_point,
                &point_slices,
                hints_by_point.clone(),
                &mut prover_transcript,
                &commitment_slices,
                BasisMode::Lagrange,
            )
            .expect("blessed batched onehot prove");

        let mut verifier_transcript =
            Blake2bTranscript::<OneHotF>::new(b"test/blessed-batched-onehot-exact");
        <SchemeLocal<D_LOCAL, CfgLocal> as CommitmentScheme<OneHotF, D_LOCAL>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &point_slices,
            &opening_groups_by_point,
            &commitment_slices,
            BasisMode::Lagrange,
        )
        .expect("blessed batched onehot verify");

        let root_plan =
            hachi_root_runtime_plan_with_batch::<CfgLocal, D_LOCAL>(nv, nv, total_claims, batch)
                .expect("blessed batched root plan");
        let estimate = recursive_suffix_estimate_with_log_basis::<CfgLocal>(
            root_plan.lookup_key(),
            root_plan.next_inputs.level,
            root_plan.next_w_len(),
            root_plan.next_level_params.log_basis,
            root_plan.planning_envelope,
        )
        .expect("blessed recursive suffix estimate");
        let root_bytes = root_plan.level_proof_bytes::<CfgLocal>();
        let observed_total = proof.size();

        assert_eq!(root_bytes + estimate.actual_state_bytes, observed_total);
        if estimate.exact_state_match {
            assert!(
                !estimate.used_actual_state_planner,
                "exact keyed blessed schedule should not use the miss-path planner"
            );
            assert_eq!(root_bytes + estimate.table_bytes, observed_total);
        } else {
            assert!(
                estimate.used_actual_state_planner,
                "off-table blessed schedule should use the exact miss-path planner"
            );
            assert_eq!(estimate.table_bytes, estimate.actual_state_bytes);
        }
    }

    #[test]
    fn batched_d32_onehot_planner_gap_stays_small() {
        assert_batched_onehot_planner_gap::<32, fp128::D32OneHot>(20, 4, 0);
    }

    #[test]
    fn batched_d64_onehot_planner_gap_stays_small() {
        assert_batched_onehot_planner_gap::<64, fp128::D64OneHot>(20, 4, 0);
    }

    #[test]
    fn blessed_same_point_batched_onehot_schedule_is_exact_end_to_end() {
        const GROUPS: &[usize] = &[1, 1, 4];
        assert_blessed_batched_onehot_exact::<64, fp128::D64OneHot>(20, &[GROUPS]);
    }

    #[test]
    fn blessed_multi_point_batched_onehot_schedule_is_exact_end_to_end() {
        const POINT_A_GROUPS: &[usize] = &[1, 1];
        const POINT_B_GROUPS: &[usize] = &[4];
        assert_blessed_batched_onehot_exact::<64, fp128::D64OneHot>(
            20,
            &[POINT_A_GROUPS, POINT_B_GROUPS],
        );
    }

    fn make_verify_fixture(
        num_vars: usize,
    ) -> (
        HachiVerifierSetup<F>,
        RingCommitment<F, D>,
        HachiProof<F>,
        Vec<F>,
        F,
        LevelParams,
    ) {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(num_vars).unwrap();
        let full_num_vars = layout.m_vars + layout.r_vars + alpha;

        let (poly, evals) = make_dense_poly(full_num_vars);
        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(full_num_vars, 1, 1);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentScheme<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .unwrap();

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

    fn dense_opening(evals: &[F], point: &[F]) -> F {
        let lw = lagrange_weights(point);
        evals
            .iter()
            .zip(lw.iter())
            .fold(F::zero(), |a, (&c, &w)| a + c * w)
    }

    fn init_debug_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let fmt_layer = tracing_subscriber::fmt::layer()
                .compact()
                .with_target(false)
                .with_span_events(FmtSpan::CLOSE);
            tracing_subscriber::registry()
                .with(EnvFilter::new("info"))
                .with(fmt_layer)
                .init();
        });
    }

    fn init_debug_rayon_pool() {
        #[cfg(feature = "parallel")]
        {
            static INIT: Once = Once::new();
            INIT.call_once(|| {
                rayon::ThreadPoolBuilder::new()
                    .stack_size(64 * 1024 * 1024)
                    .build_global()
                    .ok();
            });
        }
    }

    fn run_debug_on_large_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(f)
            .expect("failed to spawn debug thread")
            .join()
            .expect("debug thread panicked");
    }

    fn debug_random_point(nv: usize) -> Vec<OneHotF> {
        let mut rng = StdRng::seed_from_u64(0xcafe_babe);
        (0..nv)
            .map(|_| OneHotF::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect()
    }

    fn debug_make_onehot_poly(
        layout: &LevelParams,
        seed: u64,
    ) -> OneHotPoly<OneHotF, ONEHOT_D, u8> {
        let total_ring = layout.num_blocks * layout.block_len;
        let num_vars = layout.m_vars + layout.r_vars + ONEHOT_D.trailing_zeros() as usize;
        assert_eq!(total_ring * BENCH_ONEHOT_K, 1usize << num_vars);

        let mut rng = StdRng::seed_from_u64(seed);
        let indices: Vec<Option<u8>> = (0..total_ring)
            .map(|_| Some(rng.gen_range(0..BENCH_ONEHOT_K) as u8))
            .collect();

        OneHotPoly::<OneHotF, ONEHOT_D, u8>::new(BENCH_ONEHOT_K, indices)
            .expect("debug onehot poly")
    }

    fn debug_make_onehot_poly_generic<const D_LOCAL: usize>(
        layout: &LevelParams,
        seed: u64,
    ) -> OneHotPoly<OneHotF, D_LOCAL, u8> {
        let onehot_k = D_LOCAL;
        let total_ring = layout.num_blocks * layout.block_len;
        let num_vars = layout.m_vars + layout.r_vars + D_LOCAL.trailing_zeros() as usize;
        assert_eq!(total_ring * onehot_k, 1usize << num_vars);

        let mut rng = StdRng::seed_from_u64(seed);
        let indices: Vec<Option<u8>> = (0..total_ring)
            .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
            .collect();

        OneHotPoly::<OneHotF, D_LOCAL, u8>::new(onehot_k, indices)
            .expect("debug generic onehot poly")
    }

    fn debug_opening_from_poly<P: HachiPolyOps<OneHotF, ONEHOT_D>>(
        poly: &P,
        point: &[OneHotF],
        layout: &LevelParams,
    ) -> OneHotF {
        let alpha_bits = ONEHOT_D.trailing_zeros() as usize;
        assert_eq!(point.len(), alpha_bits + layout.m_vars + layout.r_vars);

        let inner_point = &point[..alpha_bits];
        let reduced_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            reduced_point,
            layout.r_vars,
            layout.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("debug opening point");

        let (y_ring, _) = poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            layout.block_len,
        );
        let v = reduce_inner_opening_to_ring_element::<OneHotF, ONEHOT_D>(
            inner_point,
            BasisMode::Lagrange,
        )
        .expect("debug inner opening point");
        (y_ring * v.sigma_m1()).coefficients()[0]
    }

    fn debug_opening_from_poly_generic<const D_LOCAL: usize, P: HachiPolyOps<OneHotF, D_LOCAL>>(
        poly: &P,
        point: &[OneHotF],
        layout: &LevelParams,
    ) -> OneHotF {
        let alpha_bits = D_LOCAL.trailing_zeros() as usize;
        assert_eq!(point.len(), alpha_bits + layout.m_vars + layout.r_vars);

        let inner_point = &point[..alpha_bits];
        let reduced_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            reduced_point,
            layout.r_vars,
            layout.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("debug generic opening point");

        let (y_ring, _) = poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            layout.block_len,
        );
        let v = reduce_inner_opening_to_ring_element::<OneHotF, D_LOCAL>(
            inner_point,
            BasisMode::Lagrange,
        )
        .expect("debug generic inner opening point");
        (y_ring * v.sigma_m1()).coefficients()[0]
    }

    fn debug_relation_sum_from_tables(
        w_evals_compact: &[i8],
        _live_x_cols: usize,
        alpha_evals_y: &[OneHotF],
        m_evals_x: &[OneHotF],
        start_x: usize,
        end_x: usize,
    ) -> OneHotF {
        let mut acc = OneHotF::zero();
        for x in start_x..end_x {
            let mut y_eval = OneHotF::zero();
            for (y, alpha_eval) in alpha_evals_y.iter().enumerate() {
                y_eval += *alpha_eval
                    * OneHotF::from_i64(w_evals_compact[x * alpha_evals_y.len() + y] as i64);
            }
            acc += y_eval * m_evals_x[x];
        }
        acc
    }

    #[test]
    fn commit_singleton_group_returns_single_claim_hint() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let (poly, _) = make_dense_poly(num_vars);
        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars, 1, 1);

        let (_, hint) =
            <Scheme as CommitmentScheme<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .unwrap();

        assert_eq!(hint.inner_opening_digits.len(), 1);
        assert_eq!(hint.t().unwrap().len(), 1);
    }

    #[test]
    #[ignore = "manual tracing-only relation-claim check"]
    fn debug_batched_root_relation_claim_matches_tables() {
        init_debug_tracing();
        init_debug_rayon_pool();
        run_debug_on_large_stack(|| {
            const BATCH_NUM_VARS: usize = 29;
            const BATCH_SIZE: usize = 1 << 5;

            let batch_layout =
                hachi_batched_root_layout::<OneHotCfg, ONEHOT_D>(BATCH_NUM_VARS, BATCH_SIZE)
                    .expect("batch debug layout");
            let batched_root_lp = scale_batched_root_layout::<OneHotCfg, ONEHOT_D>(
                BATCH_NUM_VARS,
                &batch_layout,
                BATCH_SIZE,
            )
            .expect("batched debug root layout");
            let batch_root_params = OneHotCfg::level_params(HachiScheduleInputs {
                max_num_vars: BATCH_NUM_VARS,
                level: 0,
                current_w_len: root_current_w_len::<ONEHOT_D>(&batch_layout),
            });

            let batch_polys: Vec<OneHotPoly<OneHotF, ONEHOT_D, u8>> = (0..BATCH_SIZE)
                .map(|idx| {
                    debug_make_onehot_poly(&batch_layout, 0x0bee_fcaf_e000_2900 + idx as u64)
                })
                .collect();
            let batch_setup = <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::setup_prover(
                BATCH_NUM_VARS,
                BATCH_SIZE,
                1,
            );
            let batch_poly_refs: Vec<&OneHotPoly<OneHotF, ONEHOT_D, u8>> =
                batch_polys.iter().collect();
            let (batch_commitment, batch_hint) =
                <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::commit(
                    &batch_poly_refs,
                    &batch_setup,
                )
                .expect("batched debug commit");
            let batch_commitments = [batch_commitment];
            let batch_hints = vec![batch_hint];
            let batch_commitment_rows = flatten_batched_commitment_rows(&batch_commitments);

            let batch_point = debug_random_point(BATCH_NUM_VARS);
            let alpha = batch_root_params.ring_dimension.trailing_zeros() as usize;
            let target_num_vars = batch_layout.m_vars + batch_layout.r_vars + alpha;
            let mut padded_point = batch_point.clone();
            padded_point.resize(target_num_vars, OneHotF::zero());
            let outer_point = &padded_point[alpha..];
            let ring_opening_point = ring_opening_point_from_field::<OneHotF>(
                outer_point,
                batch_layout.r_vars,
                batch_layout.m_vars,
                BasisMode::Lagrange,
                BlockOrder::RowMajor,
            )
            .expect("debug opening point");
            let inner_reduction = reduce_inner_opening_to_ring_element::<OneHotF, ONEHOT_D>(
                &padded_point[..alpha],
                BasisMode::Lagrange,
            )
            .expect("debug inner reduction");
            let (y_rings, w_folded_by_poly): (Vec<_>, Vec<_>) = batch_polys
                .iter()
                .map(|poly| {
                    poly.evaluate_and_fold(
                        &ring_opening_point.b,
                        &ring_opening_point.a,
                        batch_layout.block_len,
                    )
                })
                .unzip();

            let mut transcript = Blake2bTranscript::<OneHotF>::new(b"debug/relation-claim/batched");
            append_batched_commitments_to_transcript(&batch_commitments, &mut transcript);
            for pt in &padded_point {
                transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
            }
            let field_openings: Vec<OneHotF> = y_rings
                .iter()
                .map(|y_ring| (*y_ring * inner_reduction.sigma_m1()).coefficients()[0])
                .collect();
            for opening in &field_openings {
                transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, opening);
            }
            let batch_gammas: Vec<OneHotF> = (0..batch_poly_refs.len())
                .map(|_| transcript.challenge_scalar(CHALLENGE_EVAL_BATCH))
                .collect();
            let batched_y_rings: Vec<CyclotomicRing<OneHotF, ONEHOT_D>> = {
                let mut per_point = vec![CyclotomicRing::<OneHotF, ONEHOT_D>::zero(); 1];
                for (claim_idx, y_ring) in y_rings.iter().enumerate() {
                    per_point[0] += y_ring.scale(&batch_gammas[claim_idx]);
                }
                per_point
            };
            for y_ring in &batched_y_rings {
                transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
            }

            let debug_batch_hint = batch_hints[0].clone();
            let debug_w_folded_by_poly: Vec<Vec<CyclotomicRing<OneHotF, ONEHOT_D>>> =
                w_folded_by_poly.clone();
            let mut quad_eq = Box::new(
                QuadraticEquation::<OneHotF, { ONEHOT_D }, OneHotCfg>::new_batched_prover(
                    &batch_setup.ntt_shared,
                    ring_opening_point.clone(),
                    &batch_poly_refs,
                    w_folded_by_poly,
                    &[BATCH_SIZE],
                    batched_root_lp.clone(),
                    batch_hints,
                    &mut transcript,
                    &batch_commitments,
                    &batched_y_rings,
                    batch_gammas,
                    1,
                    batch_setup.expanded.seed.max_stride,
                )
                .expect("debug batched quadratic equation"),
            );
            let w = ring_switch_build_w::<OneHotF, { ONEHOT_D }, OneHotCfg>(
                &mut quad_eq,
                &batch_setup.expanded,
                &batch_setup.ntt_shared,
                &batched_root_lp,
            )
            .expect("debug batched w");
            let commit_params = OneHotCfg::level_params(HachiScheduleInputs {
                max_num_vars: BATCH_NUM_VARS,
                level: 1,
                current_w_len: w.len(),
            });
            let mut commit_ntt_cache = MultiDNttCaches::default();
            let (w_commitment_flat, w_hint_cache) = dispatch_commit::<OneHotF, OneHotCfg>(
                commit_params,
                &mut commit_ntt_cache,
                &batch_setup.expanded,
                &w,
            )
            .expect("debug batched w commit");
            let w_commitment_proof = w_commitment_flat.clone();
            let rs = ring_switch_finalize_with_claim_groups::<OneHotF, _, { ONEHOT_D }, OneHotCfg>(
                &quad_eq,
                &batch_setup.expanded,
                &mut transcript,
                w,
                w_commitment_flat,
                &w_commitment_proof,
                w_hint_cache,
                &batched_root_lp,
            )
            .expect("debug batched ring switch");

            let relation_claim = relation_claim_from_rows::<OneHotF, ONEHOT_D>(
                &rs.tau1,
                rs.alpha,
                &quad_eq.v,
                &batch_commitment_rows,
                &batched_y_rings,
            );
            let relation_sum = debug_relation_sum_from_tables(
                &rs.w_evals_compact,
                rs.live_x_cols,
                &rs.alpha_evals_y,
                &rs.m_evals_x,
                0,
                rs.live_x_cols,
            );
            let w_alpha_evals: Vec<OneHotF> = (0..rs.live_x_cols)
                .map(|x| {
                    rs.alpha_evals_y.iter().enumerate().fold(
                        OneHotF::zero(),
                        |acc, (y, alpha_eval)| {
                            acc + *alpha_eval
                                * OneHotF::from_i64(
                                    rs.w_evals_compact[x * rs.alpha_evals_y.len() + y] as i64,
                                )
                        },
                    )
                })
                .collect();
            let w_hat_len =
                batched_root_lp.num_digits_open * batched_root_lp.num_blocks * BATCH_SIZE;
            let t_hat_len = batched_root_lp.num_digits_open
                * batch_root_params.a_key.row_len()
                * batched_root_lp.num_blocks
                * BATCH_SIZE;
            let z_pre_len = batched_root_lp.inner_width() * batched_root_lp.num_digits_fold;
            let num_eval_rows = 1usize;
            let num_commitment_groups = 1usize;
            let m_rows = batch_root_params.m_row_count_with_commitments_and_public_outputs(
                num_commitment_groups,
                num_eval_rows,
            );
            let r_tail_len = m_rows
                * crate::protocol::ring_switch::r_decomp_levels::<OneHotF>(
                    batched_root_lp.log_basis,
                );
            let w_hat_relation_sum = debug_relation_sum_from_tables(
                &rs.w_evals_compact,
                rs.live_x_cols,
                &rs.alpha_evals_y,
                &rs.m_evals_x,
                0,
                w_hat_len,
            );
            let t_hat_relation_sum = debug_relation_sum_from_tables(
                &rs.w_evals_compact,
                rs.live_x_cols,
                &rs.alpha_evals_y,
                &rs.m_evals_x,
                w_hat_len,
                w_hat_len + t_hat_len,
            );
            let z_pre_relation_sum = debug_relation_sum_from_tables(
                &rs.w_evals_compact,
                rs.live_x_cols,
                &rs.alpha_evals_y,
                &rs.m_evals_x,
                w_hat_len + t_hat_len,
                w_hat_len + t_hat_len + z_pre_len,
            );
            let r_tail_relation_sum = debug_relation_sum_from_tables(
                &rs.w_evals_compact,
                rs.live_x_cols,
                &rs.alpha_evals_y,
                &rs.m_evals_x,
                w_hat_len + t_hat_len + z_pre_len,
                w_hat_len + t_hat_len + z_pre_len + r_tail_len,
            );
            let eq_tau1 = crate::algebra::eq_poly::EqPolynomial::evals(&rs.tau1);
            // Row layout: consistency (1) | public (num_eval_rows) | D (n_d) |
            //             B (n_b * num_commitment_groups) | A (n_a)
            let consistency_weight = eq_tau1[0];
            let public_weights = &eq_tau1[1..(1 + num_eval_rows)];
            let d_start = 1 + num_eval_rows;
            let b_start = d_start + batch_root_params.d_key.row_len();
            let a_start = b_start + batch_root_params.b_key.row_len() * num_commitment_groups;
            let a_weights = &eq_tau1[a_start..m_rows];
            let alpha_pows = &rs.alpha_evals_y;
            let eval_sparse_alpha = |challenge: &crate::algebra::SparseChallenge| -> OneHotF {
                challenge
                    .positions
                    .iter()
                    .zip(challenge.coeffs.iter())
                    .fold(OneHotF::zero(), |acc, (&pos, &coeff)| {
                        acc + OneHotF::from_i64(coeff as i64) * alpha_pows[pos as usize]
                    })
            };
            let eval_ring_at_pows_local =
                |ring: &CyclotomicRing<OneHotF, ONEHOT_D>, pows: &[OneHotF]| -> OneHotF {
                    ring.coefficients()
                        .iter()
                        .zip(pows.iter())
                        .fold(OneHotF::zero(), |acc, (coeff, alpha_pow)| {
                            acc + *coeff * *alpha_pow
                        })
                };
            let c_alphas: Vec<OneHotF> = quad_eq.challenges.iter().map(eval_sparse_alpha).collect();
            let gadget_scalars = |levels: usize| -> Vec<OneHotF> {
                let base = OneHotF::from_canonical_u128_reduced(1u128 << batched_root_lp.log_basis);
                let mut out = Vec::with_capacity(levels);
                let mut power = OneHotF::one();
                for _ in 0..levels {
                    out.push(power);
                    power *= base;
                }
                out
            };
            let g1_open = gadget_scalars(batched_root_lp.num_digits_open);
            let g1_commit = gadget_scalars(batched_root_lp.num_digits_commit);
            let fold_gadget = gadget_scalars(batched_root_lp.num_digits_fold);
            let r_gadget = gadget_scalars(
                crate::protocol::ring_switch::r_decomp_levels::<OneHotF>(batched_root_lp.log_basis),
            );
            let debug_stride = batch_setup.expanded.seed.max_stride;
            let d_view = batch_setup
                .expanded
                .shared_matrix
                .ring_view::<ONEHOT_D>(batch_root_params.d_key.row_len(), debug_stride);
            let b_view = batch_setup
                .expanded
                .shared_matrix
                .ring_view::<ONEHOT_D>(batch_root_params.b_key.row_len(), debug_stride);
            let a_view = batch_setup
                .expanded
                .shared_matrix
                .ring_view::<ONEHOT_D>(batch_root_params.a_key.row_len(), debug_stride);
            let denom = alpha_pows[ONEHOT_D - 1] * rs.alpha + OneHotF::one();
            let expected_d_sum = quad_eq
                .v
                .iter()
                .enumerate()
                .take(batch_root_params.d_key.row_len())
                .fold(OneHotF::zero(), |acc, (di, row)| {
                    acc + eq_tau1[d_start + di]
                        * crate::protocol::ring_switch::eval_ring_at(row, &rs.alpha)
                });
            let expected_b_sum =
                batch_commitment_rows
                    .iter()
                    .enumerate()
                    .fold(OneHotF::zero(), |acc, (bi, row)| {
                        acc + eq_tau1[b_start + bi]
                            * crate::protocol::ring_switch::eval_ring_at(row, &rs.alpha)
                    });
            let expected_public_sum = batched_y_rings.iter().enumerate().fold(
                OneHotF::zero(),
                |acc, (point_idx, y_ring)| {
                    acc + public_weights[point_idx]
                        * crate::protocol::ring_switch::eval_ring_at(y_ring, &rs.alpha)
                },
            );
            let stored_t_by_poly = debug_batch_hint
                .t()
                .expect("debug batched stored t rows")
                .to_vec();
            let mut debug_hint_flat = debug_batch_hint.into_flattened();
            debug_hint_flat
                .ensure_t_recomposed(batched_root_lp.num_digits_open, batched_root_lp.log_basis)
                .expect("debug batched t recomposition");
            let debug_t_hat = debug_hint_flat.inner_opening_digits.clone();
            let _debug_t_hat_flat = debug_t_hat.flat_digits().to_vec();
            let debug_t = debug_hint_flat.t().expect("debug batched t rows");
            let debug_w_folded_flat: Vec<_> = debug_w_folded_by_poly
                .clone()
                .into_iter()
                .flatten()
                .collect();
            let debug_w_hat: Vec<Vec<[i8; ONEHOT_D]>> = debug_w_folded_by_poly
                .iter()
                .flat_map(|folded_rows| {
                    folded_rows.iter().map(|w_i| {
                        w_i.balanced_decompose_pow2_i8(
                            batched_root_lp.num_digits_open,
                            batched_root_lp.log_basis,
                        )
                    })
                })
                .collect();
            let debug_w_hat_flat: Vec<_> = debug_w_hat
                .iter()
                .flat_map(|block| block.iter().copied())
                .collect();
            let mut debug_z_witnesses = batch_polys
                .iter()
                .zip(quad_eq.challenges.chunks(batched_root_lp.num_blocks))
                .map(|(poly, poly_challenges)| {
                    poly.decompose_fold(
                        poly_challenges,
                        batched_root_lp.block_len,
                        batched_root_lp.num_digits_commit,
                        batched_root_lp.log_basis,
                    )
                });
            let mut debug_z = debug_z_witnesses.next().expect("debug batched z witness");
            for witness in debug_z_witnesses {
                for (dst, src) in debug_z.z_pre.iter_mut().zip(witness.z_pre.iter()) {
                    *dst += *src;
                }
                for (dst, src) in debug_z
                    .centered_coeffs
                    .iter_mut()
                    .zip(witness.centered_coeffs.iter())
                {
                    for k in 0..ONEHOT_D {
                        dst[k] += src[k];
                    }
                }
            }
            debug_z.centered_inf_norm = debug_z
                .centered_coeffs
                .iter()
                .flat_map(|coeffs| coeffs.iter())
                .map(|coeff| coeff.unsigned_abs())
                .max()
                .unwrap_or(0);
            let (first_block_t_matches, sampled_first_poly_z_matches) = match batch_polys[0]
                .cached_blocks()
                .expect("batch poly must have its block cache built before the debug check")
            {
                crate::protocol::hachi_poly_ops::OneHotBlocks::Regular(regular_blocks) => {
                    let first_block_ref_t = regular_blocks[0].iter().fold(
                        CyclotomicRing::<OneHotF, ONEHOT_D>::zero(),
                        |mut acc, entry| {
                            a_view.row(0)[entry.pos_in_block()]
                                .shift_accumulate_into(&mut acc, entry.coeff_idx());
                            acc
                        },
                    );
                    let first_poly_challenges = &quad_eq.challenges[..batched_root_lp.num_blocks];
                    let first_poly_z = batch_polys[0].decompose_fold(
                        first_poly_challenges,
                        batched_root_lp.block_len,
                        batched_root_lp.num_digits_commit,
                        batched_root_lp.log_basis,
                    );
                    let sample_positions = [
                        0usize,
                        1,
                        2,
                        17,
                        123,
                        1024,
                        batched_root_lp.block_len / 2,
                        batched_root_lp.block_len - 1,
                    ];
                    let sampled_z_matches = sample_positions.into_iter().all(|pos| {
                        let ref_z = regular_blocks
                            .iter()
                            .zip(first_poly_challenges.iter())
                            .fold(
                                CyclotomicRing::<OneHotF, ONEHOT_D>::zero(),
                                |mut acc, (block_entries, challenge)| {
                                    let entry = block_entries[pos];
                                    debug_assert_eq!(entry.pos_in_block(), pos);
                                    let mut mono = CyclotomicRing::<OneHotF, ONEHOT_D>::zero();
                                    mono.coefficients_mut()[entry.coeff_idx()] = OneHotF::one();
                                    mono.mul_by_sparse_into(challenge, &mut acc);
                                    acc
                                },
                            );
                        first_poly_z.z_pre[pos] == ref_z
                    });
                    (
                        stored_t_by_poly[0][0][0] == first_block_ref_t,
                        sampled_z_matches,
                    )
                }
                crate::protocol::hachi_poly_ops::OneHotBlocks::General(_) => (false, false),
            };
            let debug_y = crate::protocol::quadratic_equation::generate_y::<OneHotF, ONEHOT_D>(
                &quad_eq.v,
                &batch_commitment_rows,
                &batched_y_rings,
                batch_root_params.d_key.row_len(),
                batch_root_params.b_key.row_len(),
                batch_root_params.a_key.row_len(),
            )
            .expect("debug batched y");
            let debug_r =
                crate::protocol::quadratic_equation::compute_r_split_eq::<OneHotF, ONEHOT_D>(
                    &batched_root_lp,
                    &batch_setup.expanded,
                    &quad_eq.challenges,
                    &debug_w_hat_flat,
                    &debug_t_hat,
                    debug_t,
                    &debug_w_folded_flat,
                    &debug_z.centered_coeffs,
                    debug_z.centered_inf_norm,
                    &debug_y,
                    &[BATCH_SIZE],
                    1,
                    batched_root_lp.num_blocks,
                    batched_root_lp.inner_width(),
                    batch_setup.expanded.seed.max_stride,
                    &batch_setup.ntt_shared,
                )
                .expect("debug batched r");
            let stored_t_flat: Vec<_> = stored_t_by_poly.iter().flatten().cloned().collect();
            let stored_a_t = quad_eq.challenges.iter().zip(stored_t_flat.iter()).fold(
                CyclotomicRing::<OneHotF, ONEHOT_D>::zero(),
                |mut acc, (challenge, block_rows)| {
                    block_rows[0].mul_by_sparse_into(challenge, &mut acc);
                    acc
                },
            );
            let reduced_a_t = quad_eq.challenges.iter().zip(debug_t.iter()).fold(
                CyclotomicRing::<OneHotF, ONEHOT_D>::zero(),
                |mut acc, (challenge, block_rows)| {
                    block_rows[0].mul_by_sparse_into(challenge, &mut acc);
                    acc
                },
            );
            let reduced_a_z = debug_z.z_pre.iter().enumerate().fold(
                CyclotomicRing::<OneHotF, ONEHOT_D>::zero(),
                |mut acc, (k, z_ring)| {
                    a_view.row(0)[k].mul_accumulate_into(z_ring, &mut acc);
                    acc
                },
            );
            let reduced_a_diff = reduced_a_t - reduced_a_z;
            let direct_raw_a_t = c_alphas.iter().zip(debug_t.iter()).fold(
                OneHotF::zero(),
                |acc, (c_alpha, block_rows)| {
                    acc + *c_alpha
                        * crate::protocol::ring_switch::eval_ring_at(&block_rows[0], &rs.alpha)
                },
            );
            let direct_raw_a_z =
                debug_z
                    .z_pre
                    .iter()
                    .enumerate()
                    .fold(OneHotF::zero(), |acc, (k, z_ring)| {
                        acc - eval_ring_at_pows_local(&a_view.row(0)[k], alpha_pows)
                            * crate::protocol::ring_switch::eval_ring_at(z_ring, &rs.alpha)
                    });
            let direct_raw_a_r =
                -(denom * crate::protocol::ring_switch::eval_ring_at(&debug_r[a_start], &rs.alpha));
            let direct_raw_a_total = direct_raw_a_t + direct_raw_a_z + direct_raw_a_r;
            let d_matrix_width = batched_root_lp.d_matrix_width();
            let d_group_w = (0..w_hat_len).fold(OneHotF::zero(), |acc, x| {
                let coeff =
                    (0..batch_root_params.d_key.row_len()).fold(OneHotF::zero(), |inner, di| {
                        inner
                            + eq_tau1[d_start + di]
                                * eval_ring_at_pows_local(
                                    &d_view.row(di)[x % d_matrix_width],
                                    alpha_pows,
                                )
                    });
                acc + w_alpha_evals[x] * coeff
            });
            let d_group_r =
                (0..batch_root_params.d_key.row_len()).fold(OneHotF::zero(), |acc, di| {
                    let row_idx = d_start + di;
                    let row_start = w_hat_len + t_hat_len + z_pre_len + row_idx * r_gadget.len();
                    acc + (0..r_gadget.len()).fold(OneHotF::zero(), |inner, level_idx| {
                        inner
                            + w_alpha_evals[row_start + level_idx]
                                * (-(eq_tau1[row_idx] * denom * r_gadget[level_idx]))
                    })
                });
            let outer_width = batched_root_lp.outer_width();
            let b_group_t = (0..t_hat_len).fold(OneHotF::zero(), |acc, x| {
                let coeff =
                    (0..batch_root_params.b_key.row_len()).fold(OneHotF::zero(), |inner, bi| {
                        inner
                            + eq_tau1[b_start + bi]
                                * eval_ring_at_pows_local(
                                    &b_view.row(bi)[x % outer_width],
                                    alpha_pows,
                                )
                    });
                acc + w_alpha_evals[w_hat_len + x] * coeff
            });
            let b_group_r =
                (0..batch_root_params.b_key.row_len()).fold(OneHotF::zero(), |acc, bi| {
                    let row_idx = b_start + bi;
                    let row_start = w_hat_len + t_hat_len + z_pre_len + row_idx * r_gadget.len();
                    acc + (0..r_gadget.len()).fold(OneHotF::zero(), |inner, level_idx| {
                        inner
                            + w_alpha_evals[row_start + level_idx]
                                * (-(eq_tau1[row_idx] * denom * r_gadget[level_idx]))
                    })
                });
            let public_group_w = (0..w_hat_len).fold(OneHotF::zero(), |acc, x| {
                let blocks_per_claim = batched_root_lp.num_blocks * batched_root_lp.num_digits_open;
                let claim_idx = x / blocks_per_claim;
                let claim_offset = x % blocks_per_claim;
                let block_idx = claim_offset / batched_root_lp.num_digits_open;
                let digit_idx = claim_offset % batched_root_lp.num_digits_open;
                acc + w_alpha_evals[x]
                    * public_weights[0]
                    * quad_eq.gamma()[claim_idx]
                    * ring_opening_point.b[block_idx]
                    * g1_open[digit_idx]
            });
            let public_group_r = (0..num_eval_rows).fold(OneHotF::zero(), |acc, point_idx| {
                let row_idx = 1 + point_idx;
                let row_start = w_hat_len + t_hat_len + z_pre_len + row_idx * r_gadget.len();
                acc + (0..r_gadget.len()).fold(OneHotF::zero(), |inner, level_idx| {
                    inner
                        + w_alpha_evals[row_start + level_idx]
                            * (-(eq_tau1[row_idx] * denom * r_gadget[level_idx]))
                })
            });
            let row4_group_w = (0..w_hat_len).fold(OneHotF::zero(), |acc, x| {
                let blocks_per_claim = batched_root_lp.num_blocks * batched_root_lp.num_digits_open;
                let claim_idx = x / blocks_per_claim;
                let claim_offset = x % blocks_per_claim;
                let block_idx = claim_offset / batched_root_lp.num_digits_open;
                let digit_idx = claim_offset % batched_root_lp.num_digits_open;
                let global_block_idx = claim_idx * batched_root_lp.num_blocks + block_idx;
                acc + w_alpha_evals[x]
                    * consistency_weight
                    * c_alphas[global_block_idx]
                    * g1_open[digit_idx]
            });
            let row4_group_z = (0..z_pre_len).fold(OneHotF::zero(), |acc, idx| {
                let k = idx / batched_root_lp.num_digits_fold;
                let fold_idx = idx % batched_root_lp.num_digits_fold;
                let block_idx = k / batched_root_lp.num_digits_commit;
                let digit_idx = k % batched_root_lp.num_digits_commit;
                acc + w_alpha_evals[w_hat_len + t_hat_len + idx]
                    * (-(consistency_weight
                        * ring_opening_point.a[block_idx]
                        * g1_commit[digit_idx]
                        * fold_gadget[fold_idx]))
            });
            let row4_group_r = {
                let row_start = w_hat_len + t_hat_len + z_pre_len;
                (0..r_gadget.len()).fold(OneHotF::zero(), |acc, level_idx| {
                    acc + w_alpha_evals[row_start + level_idx]
                        * (-(consistency_weight * denom * r_gadget[level_idx]))
                })
            };
            let a_group_t = (0..t_hat_len).fold(OneHotF::zero(), |acc, x| {
                let blocks_per_claim = batch_root_params.a_key.row_len()
                    * batched_root_lp.num_digits_open
                    * batched_root_lp.num_blocks;
                let claim_idx = x / blocks_per_claim;
                let claim_offset = x % blocks_per_claim;
                let block_idx = claim_offset
                    / (batch_root_params.a_key.row_len() * batched_root_lp.num_digits_open);
                let rem = claim_offset
                    % (batch_root_params.a_key.row_len() * batched_root_lp.num_digits_open);
                let a_idx = rem / batched_root_lp.num_digits_open;
                let digit_idx = rem % batched_root_lp.num_digits_open;
                let global_block_idx = claim_idx * batched_root_lp.num_blocks + block_idx;
                acc + w_alpha_evals[w_hat_len + x]
                    * a_weights[a_idx]
                    * c_alphas[global_block_idx]
                    * g1_open[digit_idx]
            });
            let a_group_z = (0..z_pre_len).fold(OneHotF::zero(), |acc, idx| {
                let k = idx / batched_root_lp.num_digits_fold;
                let fold_idx = idx % batched_root_lp.num_digits_fold;
                let block_idx = k / batched_root_lp.num_digits_commit;
                let coeff =
                    a_weights
                        .iter()
                        .enumerate()
                        .fold(OneHotF::zero(), |inner, (a_idx, eq_i)| {
                            inner
                                + *eq_i * eval_ring_at_pows_local(&a_view.row(a_idx)[k], alpha_pows)
                        });
                let _ = block_idx;
                acc + w_alpha_evals[w_hat_len + t_hat_len + idx]
                    * (-(coeff * fold_gadget[fold_idx]))
            });
            let a_group_r =
                a_weights
                    .iter()
                    .enumerate()
                    .fold(OneHotF::zero(), |acc, (row_offset, eq_i)| {
                        let row_idx = a_start + row_offset;
                        let row_start =
                            w_hat_len + t_hat_len + z_pre_len + row_idx * r_gadget.len();
                        acc + (0..r_gadget.len()).fold(OneHotF::zero(), |inner, level_idx| {
                            inner
                                + w_alpha_evals[row_start + level_idx]
                                    * (-(*eq_i * denom * r_gadget[level_idx]))
                        })
                    });

            tracing::info!(
                relation_claim_u128 = relation_claim.to_canonical_u128(),
                relation_sum_u128 = relation_sum.to_canonical_u128(),
                w_hat_relation_sum_u128 = w_hat_relation_sum.to_canonical_u128(),
                t_hat_relation_sum_u128 = t_hat_relation_sum.to_canonical_u128(),
                z_pre_relation_sum_u128 = z_pre_relation_sum.to_canonical_u128(),
                r_tail_relation_sum_u128 = r_tail_relation_sum.to_canonical_u128(),
                d_group_u128 = (d_group_w + d_group_r).to_canonical_u128(),
                expected_d_u128 = expected_d_sum.to_canonical_u128(),
                b_group_u128 = (b_group_t + b_group_r).to_canonical_u128(),
                expected_b_u128 = expected_b_sum.to_canonical_u128(),
                public_group_u128 = (public_group_w + public_group_r).to_canonical_u128(),
                expected_public_u128 = expected_public_sum.to_canonical_u128(),
                row4_group_u128 = (row4_group_w + row4_group_z + row4_group_r).to_canonical_u128(),
                a_group_t_u128 = a_group_t.to_canonical_u128(),
                a_group_z_u128 = a_group_z.to_canonical_u128(),
                a_group_r_u128 = a_group_r.to_canonical_u128(),
                a_group_u128 = (a_group_t + a_group_z + a_group_r).to_canonical_u128(),
                first_block_t_matches,
                sampled_first_poly_z_matches,
                stored_a_ring_matches = stored_a_t == reduced_a_z,
                stored_vs_recomposed_t = stored_t_flat == debug_t,
                reduced_a_ring_matches = reduced_a_t == reduced_a_z,
                reduced_a_diff_alpha_u128 =
                    crate::protocol::ring_switch::eval_ring_at(&reduced_a_diff, &rs.alpha)
                        .to_canonical_u128(),
                direct_raw_a_t_u128 = direct_raw_a_t.to_canonical_u128(),
                direct_raw_a_z_u128 = direct_raw_a_z.to_canonical_u128(),
                direct_raw_a_r_u128 = direct_raw_a_r.to_canonical_u128(),
                direct_raw_a_total_u128 = direct_raw_a_total.to_canonical_u128(),
                live_x_cols = rs.live_x_cols,
                col_bits = rs.col_bits,
                ring_bits = rs.ring_bits,
                "batched relation claim consistency"
            );
            tracing::info!(
                matches = relation_sum == relation_claim,
                "batched relation claim comparison complete"
            );
        });
    }

    #[test]
    #[ignore = "manual tracing-only benchmark breakdown"]
    fn debug_onehot_batched_profile_compare() {
        init_debug_tracing();
        init_debug_rayon_pool();
        run_debug_on_large_stack(|| {
            const SINGLE_NUM_VARS: usize = 34;
            const BATCH_NUM_VARS: usize = 29;
            const BATCH_SIZE: usize = 1 << 5;

            let single_layout =
                OneHotCfg::commitment_layout(SINGLE_NUM_VARS).expect("single debug layout");
            let batch_layout =
                hachi_batched_root_layout::<OneHotCfg, ONEHOT_D>(BATCH_NUM_VARS, BATCH_SIZE)
                    .expect("batch debug layout");
            let batched_root_lp = scale_batched_root_layout::<OneHotCfg, ONEHOT_D>(
                BATCH_NUM_VARS,
                &batch_layout,
                BATCH_SIZE,
            )
            .expect("batched debug root layout");

            let single_root_params = OneHotCfg::level_params(HachiScheduleInputs {
                max_num_vars: SINGLE_NUM_VARS,
                level: 0,
                current_w_len: root_current_w_len::<ONEHOT_D>(&single_layout),
            });
            let _batch_root_params = OneHotCfg::level_params(HachiScheduleInputs {
                max_num_vars: BATCH_NUM_VARS,
                level: 0,
                current_w_len: root_current_w_len::<ONEHOT_D>(&batch_layout),
            });

            let single_root_w_ring = w_ring_element_count::<OneHotF>(&single_root_params);
            let batched_root_w_ring =
                w_ring_element_count_with_num_claims::<OneHotF>(&batched_root_lp, BATCH_SIZE);

            tracing::info!(
                ?single_layout,
                ?batch_layout,
                ?batched_root_lp,
                single_root_w_ring,
                batched_root_w_ring,
                single_root_w_coeffs = single_root_w_ring * ONEHOT_D,
                batched_root_w_coeffs = batched_root_w_ring * ONEHOT_D,
                total_field_single = 1usize << SINGLE_NUM_VARS,
                total_field_batched = BATCH_SIZE * (1usize << BATCH_NUM_VARS),
                "onehot root comparison"
            );

            let single_poly = debug_make_onehot_poly(&single_layout, 0x0bee_fcaf_e000_0034);
            let batch_polys: Vec<OneHotPoly<OneHotF, ONEHOT_D, u8>> = (0..BATCH_SIZE)
                .map(|idx| {
                    debug_make_onehot_poly(&batch_layout, 0x0bee_fcaf_e000_2900 + idx as u64)
                })
                .collect();

            let single_point = debug_random_point(SINGLE_NUM_VARS);
            let batch_point = debug_random_point(BATCH_NUM_VARS);
            let single_opening =
                debug_opening_from_poly(&single_poly, &single_point, &single_layout);
            let batch_openings: Vec<OneHotF> = batch_polys
                .iter()
                .map(|poly| debug_opening_from_poly(poly, &batch_point, &batch_layout))
                .collect();

            let single_setup = <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::setup_prover(
                SINGLE_NUM_VARS,
                1,
                1,
            );
            let single_verifier_setup =
                <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::setup_verifier(
                    &single_setup,
                );
            let (single_commitment, single_hint) =
                <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::commit(
                    std::slice::from_ref(&single_poly),
                    &single_setup,
                )
                .expect("single debug commit");

            let _single_prove_span = tracing::info_span!("debug_single_prove").entered();
            let mut single_prover_transcript =
                Blake2bTranscript::<OneHotF>::new(b"debug/onehot/single");
            let single_proof = <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::prove(
                &single_setup,
                &single_poly,
                &single_point,
                single_hint,
                &mut single_prover_transcript,
                &single_commitment,
                BasisMode::Lagrange,
            )
            .expect("single debug prove");
            drop(_single_prove_span);

            let _single_verify_span = tracing::info_span!("debug_single_verify").entered();
            <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::verify(
                &single_proof,
                &single_verifier_setup,
                &mut Blake2bTranscript::<OneHotF>::new(b"debug/onehot/single"),
                &single_point,
                &single_opening,
                &single_commitment,
                BasisMode::Lagrange,
            )
            .expect("single debug verify");
            drop(_single_verify_span);

            let batch_setup = <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::setup_prover(
                BATCH_NUM_VARS,
                BATCH_SIZE,
                1,
            );
            let batch_verifier_setup =
                <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::setup_verifier(&batch_setup);
            let batch_poly_groups = [&batch_polys[..]];
            let (batch_commitment, batch_hint) = <OneHotScheme as CommitmentScheme<
                OneHotF,
                ONEHOT_D,
            >>::commit(&batch_polys, &batch_setup)
            .expect("batched debug commit");
            let batch_commitments = [batch_commitment];
            let batch_hints = vec![batch_hint];

            let _batched_prove_span = tracing::info_span!("debug_batched_prove").entered();
            let mut batch_prover_transcript =
                Blake2bTranscript::<OneHotF>::new(b"debug/onehot/batched");
            let batch_proof = <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::batched_prove(
                &batch_setup,
                &[&batch_poly_groups[..]],
                &[&batch_point[..]],
                vec![batch_hints],
                &mut batch_prover_transcript,
                &[&batch_commitments[..]],
                BasisMode::Lagrange,
            )
            .expect("batched debug prove");
            drop(_batched_prove_span);

            let _batched_verify_span = tracing::info_span!("debug_batched_verify").entered();
            let batch_opening_groups = [&batch_openings[..]];
            <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::batched_verify(
                &batch_proof,
                &batch_verifier_setup,
                &mut Blake2bTranscript::<OneHotF>::new(b"debug/onehot/batched"),
                &[&batch_point[..]],
                &[&batch_opening_groups[..]],
                &[&batch_commitments[..]],
                BasisMode::Lagrange,
            )
            .expect("batched debug verify");
            drop(_batched_verify_span);
        });
    }

    #[test]
    fn batched_commit_matches_individual_commits() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;
        let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 1) as u64)).collect();
        let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 3 + 7) as u64)).collect();
        let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
        let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars, 2, 1);
        let poly_groups = [std::slice::from_ref(&poly_a), std::slice::from_ref(&poly_b)];

        let (batched_commitments, batched_hints): (Vec<_>, Vec<_>) = poly_groups
            .iter()
            .map(|group| <Scheme as CommitmentScheme<F, D>>::commit(group, &setup))
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .unzip();
        let (commitment_a, hint_a) =
            <Scheme as CommitmentScheme<F, D>>::commit(std::slice::from_ref(&poly_a), &setup)
                .unwrap();
        let (commitment_b, hint_b) =
            <Scheme as CommitmentScheme<F, D>>::commit(std::slice::from_ref(&poly_b), &setup)
                .unwrap();

        assert_eq!(batched_commitments, vec![commitment_a, commitment_b]);
        assert_eq!(batched_hints, vec![hint_a, hint_b]);
    }

    #[test]
    fn batched_verify_passes_for_consistent_openings() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;
        let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 5) as u64)).collect();
        let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 7 + 3) as u64)).collect();
        let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
        let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars, 2, 1);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);
        let poly_group = [&poly_a, &poly_b];
        let poly_groups = [&poly_group[..]];
        let (commitment, hint) =
            <Scheme as CommitmentScheme<F, D>>::commit(&poly_group, &setup).unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 9) as u64)).collect();
        let openings = [
            dense_opening(&evals_a, &opening_point),
            dense_opening(&evals_b, &opening_point),
        ];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/batched-prove");
        let proof = <Scheme as CommitmentScheme<F, D>>::batched_prove(
            &setup,
            &[&poly_groups[..]],
            &[&opening_point[..]],
            vec![hints],
            &mut prover_transcript,
            &[&commitments[..]],
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut bytes = Vec::new();
        let shape = proof.shape();
        proof.serialize_uncompressed(&mut bytes).unwrap();
        let proof = HachiBatchedProof::<F>::deserialize_uncompressed(&*bytes, &shape).unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/batched-prove");
        let opening_groups = [&openings[..]];
        let result = <Scheme as CommitmentScheme<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &[&opening_point[..]],
            &[&opening_groups[..]],
            &[&commitments[..]],
            BasisMode::Lagrange,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn batched_onehot_roundtrip_matches_public_shape_context() {
        const NV: usize = 15;
        const BATCH_SIZE: usize = 2;

        let layout =
            hachi_batched_root_layout::<OneHotCfg, ONEHOT_D>(NV, BATCH_SIZE).expect("layout");
        let total_field = (layout.num_blocks * layout.block_len)
            .checked_mul(ONEHOT_D)
            .expect("total field size overflow");
        let total_chunks = total_field / BENCH_ONEHOT_K;
        assert_eq!(total_chunks * BENCH_ONEHOT_K, total_field);

        let polys: Vec<OneHotPoly<OneHotF, ONEHOT_D, u8>> = (0..BATCH_SIZE)
            .map(|poly_idx| {
                debug_make_onehot_poly(&layout, 0x0bee_fcaf_e000_1500 + poly_idx as u64)
            })
            .collect();
        let poly_refs: Vec<&OneHotPoly<OneHotF, ONEHOT_D, u8>> = polys.iter().collect();
        let point = debug_random_point(NV);
        let openings: Vec<OneHotF> = polys
            .iter()
            .map(|poly| debug_opening_from_poly(poly, &point, &layout))
            .collect();

        let setup =
            <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::setup_prover(NV, BATCH_SIZE, 1);
        let verifier_setup =
            <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::setup_verifier(&setup);
        let poly_groups = [&poly_refs[..]];
        let (commitment, hint) =
            <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::commit(&poly_refs, &setup)
                .expect("batched onehot commit");
        let commitments = [commitment];
        let hints = vec![hint];

        let mut prover_transcript = Blake2bTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
        let proof = <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::batched_prove(
            &setup,
            &[&poly_groups[..]],
            &[&point[..]],
            vec![hints],
            &mut prover_transcript,
            &[&commitments[..]],
            BasisMode::Lagrange,
        )
        .expect("batched onehot prove");

        let expected_shape = expected_same_point_batched_shape(NV, BATCH_SIZE, &proof);
        let actual_shape = proof.shape();
        assert_eq!(
            expected_shape.root_shape.y_ring_coeffs,
            actual_shape.root_shape.y_ring_coeffs
        );
        assert_eq!(
            expected_shape.root_shape.v_coeffs,
            actual_shape.root_shape.v_coeffs
        );
        assert_eq!(
            expected_shape.root_shape.stage1_stages,
            actual_shape.root_shape.stage1_stages
        );
        assert_eq!(
            expected_shape.root_shape.stage2_sumcheck,
            actual_shape.root_shape.stage2_sumcheck
        );
        assert_eq!(
            expected_shape.root_shape.next_commit_coeffs,
            actual_shape.root_shape.next_commit_coeffs
        );
        assert_eq!(expected_shape.step_shapes, actual_shape.step_shapes);
        let mut bytes = Vec::new();
        proof.serialize_uncompressed(&mut bytes).unwrap();
        let decoded =
            HachiBatchedProof::<OneHotF>::deserialize_uncompressed(&*bytes, &expected_shape)
                .expect("deserialize batched proof with derived shape");
        assert_eq!(decoded, proof);

        let opening_groups = [&openings[..]];
        let mut verifier_transcript =
            Blake2bTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
        <OneHotScheme as CommitmentScheme<OneHotF, ONEHOT_D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            &[&point[..]],
            &[&opening_groups[..]],
            &[&commitments[..]],
            BasisMode::Lagrange,
        )
        .expect("batched onehot verify");
    }

    #[test]
    fn batched_verify_rejects_wrong_opening() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;
        let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 11) as u64)).collect();
        let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 5 + 13) as u64)).collect();
        let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
        let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars, 2, 1);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);
        let poly_group = [&poly_a, &poly_b];
        let poly_groups = [&poly_group[..]];
        let (commitment, hint) =
            <Scheme as CommitmentScheme<F, D>>::commit(&poly_group, &setup).unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 4) as u64)).collect();
        let mut openings = [
            dense_opening(&evals_a, &opening_point),
            dense_opening(&evals_b, &opening_point),
        ];
        openings[1] += F::one();

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/batched-prove/bad");
        let proof = <Scheme as CommitmentScheme<F, D>>::batched_prove(
            &setup,
            &[&poly_groups[..]],
            &[&opening_point[..]],
            vec![hints],
            &mut prover_transcript,
            &[&commitments[..]],
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/batched-prove/bad");
        let opening_groups = [&openings[..]];
        let result = <Scheme as CommitmentScheme<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &[&opening_point[..]],
            &[&opening_groups[..]],
            &[&commitments[..]],
            BasisMode::Lagrange,
        );

        assert!(matches!(result, Err(HachiError::InvalidProof)));
    }

    #[test]
    fn batched_verify_rejects_batch_count_beyond_setup_capacity() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;
        let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 17) as u64)).collect();
        let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 3 + 19) as u64)).collect();
        let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
        let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars, 2, 1);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);
        let poly_group = [&poly_a, &poly_b];
        let poly_groups = [&poly_group[..]];
        let (commitment, hint) =
            <Scheme as CommitmentScheme<F, D>>::commit(&poly_group, &setup).unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 6) as u64)).collect();
        let openings = vec![
            dense_opening(&evals_a, &opening_point),
            dense_opening(&evals_b, &opening_point),
        ];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/batched-prove/oversized");
        let proof = <Scheme as CommitmentScheme<F, D>>::batched_prove(
            &setup,
            &[&poly_groups[..]],
            &[&opening_point[..]],
            vec![hints],
            &mut prover_transcript,
            &[&commitments[..]],
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut oversized_proof = proof.clone();
        let mut oversized_y_coeffs = oversized_proof.root.y_rings().coeffs().to_vec();
        oversized_y_coeffs.extend(vec![F::zero(); D]);
        oversized_proof.root.y_rings = FlatRingVec::from_coeffs(oversized_y_coeffs);

        let mut oversized_openings = openings;
        oversized_openings.push(F::zero());
        let oversized_opening_groups = [&oversized_openings[..]];

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/batched-prove/oversized");
        let result = <Scheme as CommitmentScheme<F, D>>::batched_verify(
            &oversized_proof,
            &verifier_setup,
            &mut verifier_transcript,
            &[&opening_point[..]],
            &[&oversized_opening_groups[..]],
            &[&commitments[..]],
            BasisMode::Lagrange,
        );

        assert!(matches!(result, Err(HachiError::InvalidProof)));
    }

    #[test]
    fn verify_passes_for_consistent_opening() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;

        let (poly, evals) = make_dense_poly(num_vars);

        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars, 1, 1);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);

        let (commitment, hint) =
            <Scheme as CommitmentScheme<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .unwrap();

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
        );

        assert!(result.is_ok());
    }

    #[test]
    fn verify_rejects_wrong_opening() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;

        let (poly, evals) = make_dense_poly(num_vars);

        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars, 1, 1);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);

        let (commitment, hint) =
            <Scheme as CommitmentScheme<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .unwrap();

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
        );

        assert!(
            result.is_err(),
            "verify must reject an incorrect opening value"
        );
    }

    #[test]
    fn verify_rejects_malformed_y_ring_dimension_without_panicking() {
        let (verifier_setup, commitment, mut proof, opening_point, opening, _layout) =
            make_verify_fixture(16);
        let first_level = proof
            .fold_levels_mut()
            .next()
            .expect("expected at least one fold level");
        let mut coeffs = first_level.y_ring.coeffs().to_vec();
        let _ = coeffs.pop().expect("expected non-empty y_ring");
        first_level.y_ring = FlatRingVec::from_coeffs(coeffs);

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

        let setup = <Scheme as CommitmentScheme<F, D>>::setup_prover(num_vars, 1, 1);
        let verifier_setup = <Scheme as CommitmentScheme<F, D>>::setup_verifier(&setup);

        let (commitment, hint) =
            <Scheme as CommitmentScheme<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .unwrap();

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
        );

        assert!(
            result.is_ok(),
            "monomial-basis proof should verify: {result:?}"
        );
    }

    #[test]
    fn tiny_d32_root_direct_helpers_accept_valid_proof() {
        type DirectCfg = fp128::D32Full;
        type DirectF = fp128::Field;
        const DIRECT_D: usize = DirectCfg::D;
        type DirectScheme = HachiCommitmentScheme<DIRECT_D, DirectCfg>;

        let num_vars = 4usize;
        let evals: Vec<DirectF> = (0..(1usize << num_vars))
            .map(|i| DirectF::from_u64((i + 1) as u64))
            .collect();
        let poly = DensePoly::<DirectF, DIRECT_D>::from_field_evals(num_vars, &evals).unwrap();
        let opening_point = vec![DirectF::zero(); num_vars];
        let opening = evals[0];

        let setup =
            <DirectScheme as CommitmentScheme<DirectF, DIRECT_D>>::setup_prover(num_vars, 1, 1);
        let verifier_setup =
            <DirectScheme as CommitmentScheme<DirectF, DIRECT_D>>::setup_verifier(&setup);
        let (commitment, hint) = <DirectScheme as CommitmentScheme<DirectF, DIRECT_D>>::commit(
            std::slice::from_ref(&poly),
            &setup,
        )
        .unwrap();
        let mut prover_transcript = Blake2bTranscript::<DirectF>::new(b"test/tiny-direct");
        let proof = <DirectScheme as CommitmentScheme<DirectF, DIRECT_D>>::prove(
            &setup,
            &poly,
            &opening_point,
            hint,
            &mut prover_transcript,
            &commitment,
            BasisMode::Lagrange,
        )
        .unwrap();

        assert_eq!(proof.num_fold_levels(), 0);
        assert!(root_direct_opening_matches::<DirectF, DIRECT_D, DirectCfg>(
            proof.final_witness(),
            &opening_point,
            &opening,
            BasisMode::Lagrange,
        )
        .unwrap());
        assert!(
            verify_root_direct_commitment::<DirectF, DIRECT_D, DirectCfg>(
                proof.final_witness(),
                &verifier_setup,
                &commitment,
            )
            .unwrap()
        );
        let mut verifier_transcript = Blake2bTranscript::<DirectF>::new(b"test/tiny-direct");
        verify_root_direct::<DirectF, _, DIRECT_D, DirectCfg>(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
            BasisMode::Lagrange,
        )
        .unwrap();
    }
}
