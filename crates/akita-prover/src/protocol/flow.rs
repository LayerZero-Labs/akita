//! Prover flow state shared by root orchestration during crate extraction.

use crate::kernels::crt_ntt::NttSlotCache;
use crate::protocol::ring_switch::{
    ring_switch_build_w, ring_switch_finalize, ring_switch_finalize_with_claim_groups,
    RingSwitchOutput,
};
use crate::protocol::setup_claim_reduction::prove_setup_claim_reduction;
use crate::protocol::sumcheck::{AkitaStage1Prover, AkitaStage2Prover};
use crate::{
    AkitaPolyOps, MultiDNttCaches, ProverClaims, QuadraticEquation, RecursiveCommitmentHintCache,
    RecursiveWitnessFlat, RecursiveWitnessView,
};
use akita_algebra::CyclotomicRing;
use akita_field::fields::wide::HasWide;
use akita_field::fields::HasUnreducedOps;
use akita_field::{AkitaError, CanonicalField, FieldCore, HalvingField, RandomSampling};
use akita_sumcheck::{prove_sumcheck, SumcheckProof};
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_EVAL_OPENINGS_FIELD,
    ABSORB_SUMCHECK_S_CLAIM, CHALLENGE_EVAL_BATCH, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::Transcript;
use akita_types::{
    append_batch_shape_to_transcript, append_batched_commitments_to_transcript,
    checked_total_claims, flatten_batched_commitment_rows, prepare_root_opening_point,
    reduce_inner_opening_to_ring_element, relation_claim_from_rows, reorder_stage1_coords,
    ring_opening_point_from_field, schedule_is_root_direct, schedule_num_fold_levels,
    validate_batched_inputs, AkitaBatchedProof, AkitaBatchedRootProof, AkitaCommitmentHint,
    AkitaExpandedSetup, AkitaLevelProof, AkitaProofStep, AkitaRootBatchSummary,
    AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaStage1Proof, BasisMode, BlockOrder,
    DirectWitnessProof, FlatRingVec, LevelParams, MultiPointBatchShape, PackedDigits,
    PreparedRootOpeningPoint, RingCommitment, Schedule, SetupClaimReductionPayload, Step,
};
use akita_verifier::prepare_m_eval;

/// Prover-side handle for one polynomial whose recursive opening the
/// next fold level must serve.
///
/// Mirrors the verifier's `RecursiveOpeningClaim`: `w`/`commitment`/
/// `hint` together materialize the next-level proof of the opening at
/// `opening_point`. `opening_point` is the stage-2 sumcheck challenge
/// vector produced at the level that emitted this handle.
pub struct RecursivePolyHandle<F: FieldCore> {
    /// Recursive witness whose opening will be proved at the next level.
    pub w: RecursiveWitnessFlat,
    /// Commitment to the recursive witness.
    pub commitment: FlatRingVec<F>,
    /// D-erased recursive commitment hint cache.
    pub hint: RecursiveCommitmentHintCache<F>,
    /// Digit basis for `w`, as `log2(b)`.
    pub log_basis: u32,
    /// Opening point at which the next level evaluates this commitment.
    pub opening_point: Vec<F>,
}

/// Runtime state carried between recursive prove levels.
///
/// Each entry of `handles` is one polynomial whose opening must be
/// proved at the next fold level. The single-poly recursive path uses
/// `handles.len() == 1`; Phase D-full slice F adds an additional handle
/// for the shared setup polynomial `S` so the next level discharges
/// the deferred `S(r_setup) = y_setup` claim alongside the folded
/// witness via multi-claim batched Hachi.
pub struct RecursiveProverState<F: FieldCore> {
    /// Per-polynomial handles to discharge at the next fold level.
    pub handles: Vec<RecursivePolyHandle<F>>,
}

/// Output from a single prove level, used to extend proof wire data and state.
pub struct ProveLevelOutput<F: FieldCore> {
    /// Fold proof produced at this level.
    pub level_proof: AkitaLevelProof<F>,
    /// Recursive prover state for the next level.
    pub next_state: RecursiveProverState<F>,
}

/// Raw pieces produced by the unified root-level prover.
///
/// Callers assemble either a singleton or batched root proof from these
/// components while sharing the same inner prover flow.
pub struct RootLevelRawOutput<F: FieldCore, const D: usize> {
    /// Gamma-combined public y-rings, one per opening point.
    pub y_rings: Vec<CyclotomicRing<F, D>>,
    /// Public v rows for the root relation.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// Stage-1 sumcheck proof.
    pub stage1: AkitaStage1Proof<F>,
    /// Stage-2 sumcheck proof.
    pub stage2_sumcheck: SumcheckProof<F>,
    /// Optional setup-side claim-reduction payload appended after stage 2.
    pub stage2_setup_claim_reduction: Option<SetupClaimReductionPayload<F>>,
    /// Recursive witness commitment carried in the proof.
    pub w_commitment_proof: FlatRingVec<F>,
    /// Claimed terminal evaluation of the recursive witness at this level.
    pub w_eval: F,
    /// Recursive prover state for the first suffix level.
    pub next_state: RecursiveProverState<F>,
}

/// Outcome of the recursive fold suffix after the root level.
pub struct RecursiveSuffixOutcome<F: FieldCore> {
    /// Per-level fold proofs, in order. Does not include the root proof.
    pub levels: Vec<AkitaLevelProof<F>>,
    /// Total fold-level count reached, including the root level.
    pub num_levels: usize,
    /// Prover state at the terminal direct step.
    pub final_state: RecursiveProverState<F>,
    /// `log_basis` for the terminal packed-digit witness.
    pub final_log_basis: u32,
}

/// Config-free flattened view of batched prover claims.
pub struct PreparedBatchedProveInputs<'a, F: FieldCore, P, const D: usize> {
    /// Distinct opening points in caller order.
    pub opening_points: Vec<&'a [F]>,
    /// Commitments flattened in point/group order.
    pub commitments_by_point: Vec<RingCommitment<F, D>>,
    /// Multipoint batch shape derived from the claims.
    pub batch_shape: MultiPointBatchShape,
    /// Total claim count used by schedule/layout lookup.
    pub layout_num_claims: usize,
    /// Number of variables in every opened polynomial.
    pub num_vars: usize,
    /// Polynomials flattened in claim order.
    pub flat_polys: Vec<&'a P>,
    /// Commitment hints flattened in claim-group order.
    pub flat_hints: Vec<AkitaCommitmentHint<F, D>>,
}

/// Pick the `log_basis` for the terminal packed-digit witness.
///
/// The planner's final direct step is authoritative and must match the
/// runtime recursive state.
///
/// # Errors
///
/// Returns an error if the schedule does not terminate in a direct step or if
/// the terminal direct step does not match the runtime witness length/basis.
pub fn resolve_final_log_basis<F>(
    schedule: &Schedule,
    current_state: &RecursiveProverState<F>,
) -> Result<u32, AkitaError>
where
    F: FieldCore,
{
    let Some(Step::Direct(direct_step)) = schedule.steps.last() else {
        return Err(AkitaError::InvalidSetup(
            "schedule must terminate in a direct step".to_string(),
        ));
    };
    let handle = &current_state.handles[0];
    if direct_step.current_w_len != handle.w.len() || direct_step.bits_per_elem != handle.log_basis
    {
        return Err(AkitaError::InvalidSetup(
            "scheduled direct step did not match final runtime state".to_string(),
        ));
    }
    Ok(direct_step.bits_per_elem)
}

/// Assemble fold-level proofs followed by the terminal packed-digit witness.
pub fn build_final_proof_steps<F>(
    levels: Vec<AkitaLevelProof<F>>,
    final_state: &RecursiveProverState<F>,
    final_log_basis: u32,
) -> Vec<AkitaProofStep<F>>
where
    F: FieldCore,
{
    let final_handle = &final_state.handles[0];
    let final_w =
        PackedDigits::from_i8_digits_with_min_bits(final_handle.w.as_i8_digits(), final_log_basis);
    let mut steps = levels
        .into_iter()
        .map(AkitaProofStep::Fold)
        .collect::<Vec<_>>();
    steps.push(AkitaProofStep::Direct(DirectWitnessProof::PackedDigits(
        final_w,
    )));
    steps
}

/// Validate and flatten batched prover claims into the root proof shape.
///
/// # Errors
///
/// Returns an error if the claim shape exceeds setup capacity, mixes
/// incompatible dimensions, or has malformed batch counts.
pub fn prepare_batched_prove_inputs<'a, F, P, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    claims: ProverClaims<'a, F, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
) -> Result<PreparedBatchedProveInputs<'a, F, P, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    validate_batched_inputs(expanded, &claims, |group| group.polynomials.len(), true)?;

    let opening_points: Vec<&'a [F]> = claims.iter().map(|(point, _)| *point).collect();
    let commitments_by_point: Vec<RingCommitment<F, D>> = claims
        .iter()
        .flat_map(|(_, groups)| groups.iter().map(|group| group.commitment.clone()))
        .collect();
    let num_vars = opening_points[0].len();
    let batch_shape = MultiPointBatchShape {
        point_group_sizes: claims.iter().map(|(_, groups)| groups.len()).collect(),
        claim_group_sizes: claims
            .iter()
            .flat_map(|(_, groups)| groups.iter().map(|group| group.polynomials.len()))
            .collect(),
        claim_to_point: claims
            .iter()
            .enumerate()
            .flat_map(|(point_idx, (_, groups))| {
                groups
                    .iter()
                    .flat_map(move |group| std::iter::repeat_n(point_idx, group.polynomials.len()))
            })
            .collect(),
    };
    let layout_num_claims = checked_total_claims(&batch_shape.claim_group_sizes, "batched_prove")?;

    let flat_polys = claims
        .iter()
        .flat_map(|(_, groups)| groups.iter().flat_map(|group| group.polynomials.iter()))
        .collect();
    let flat_hints = claims
        .into_iter()
        .flat_map(|(_, groups)| groups.into_iter().map(|group| group.hint))
        .collect();

    Ok(PreparedBatchedProveInputs {
        opening_points,
        commitments_by_point,
        batch_shape,
        layout_num_claims,
        num_vars,
        flat_polys,
        flat_hints,
    })
}

/// Build a root-direct batched proof from already-validated prover claims.
///
/// Root schedule policy decides when the direct shortcut applies. This helper
/// owns only the config-free proof payload assembly from polynomial direct
/// witnesses.
///
/// # Errors
///
/// Returns an error if any polynomial cannot produce a direct root witness.
pub fn prove_root_direct_from_claims<F, const D: usize, P, C, H>(
    claims: &ProverClaims<'_, F, P, C, H>,
) -> Result<AkitaBatchedProof<F>, AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    let flat_polys = claims
        .iter()
        .flat_map(|(_, groups)| groups.iter().flat_map(|group| group.polynomials.iter()))
        .collect::<Vec<_>>();
    prove_root_direct_from_polys::<F, D, P>(&flat_polys)
}

/// Build a root-direct batched proof from flattened polynomial references.
///
/// # Errors
///
/// Returns an error if any polynomial cannot produce a direct root witness.
pub fn prove_root_direct_from_polys<F, const D: usize, P>(
    polys: &[&P],
) -> Result<AkitaBatchedProof<F>, AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    let witnesses = polys
        .iter()
        .map(|poly| poly.direct_root_witness())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AkitaBatchedProof {
        root: AkitaBatchedRootProof::new_direct(witnesses),
        steps: Vec::new(),
    })
}

/// Drive batched proving up to the config-selected folded-root policy.
///
/// This owns the config-free top-level prover work: validate/flatten public
/// prover claims, derive the schedule lookup key, select the schedule through
/// the supplied policy callback, apply the root-direct shortcut when the
/// selected schedule says no fold is needed, and derive the first recursive
/// schedule inputs for folded roots. Folded-root proving still runs in the
/// caller-supplied closure while config-selected recursive commitment layouts
/// remain outside this crate.
///
/// # Errors
///
/// Returns an error if claim preparation, schedule selection, root-direct
/// witness construction, root-next parameter selection, or folded-root proving
/// fails.
#[allow(clippy::too_many_arguments)]
pub fn prove_batched_with_policy<
    'a,
    F,
    T,
    P,
    const D: usize,
    SelectSchedule,
    SelectRootNext,
    ProveFolded,
>(
    expanded: &AkitaExpandedSetup<F>,
    claims: ProverClaims<'a, F, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
    transcript: &mut T,
    basis: BasisMode,
    select_schedule: SelectSchedule,
    select_root_next_params: SelectRootNext,
    prove_folded: ProveFolded,
) -> Result<AkitaBatchedProof<F>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    SelectSchedule:
        FnOnce(usize, usize, usize, AkitaRootBatchSummary) -> Result<Schedule, AkitaError>,
    SelectRootNext: FnOnce(&Schedule, AkitaScheduleInputs) -> Result<LevelParams, AkitaError>,
    ProveFolded: FnOnce(
        PreparedBatchedProveInputs<'a, F, P, D>,
        Schedule,
        LevelParams,
        &mut T,
        BasisMode,
    ) -> Result<AkitaBatchedProof<F>, AkitaError>,
{
    let prepared_claims = prepare_batched_prove_inputs::<F, P, D>(expanded, claims)?;
    let batch_summary = AkitaRootBatchSummary::from_claim_group_sizes(
        &prepared_claims.batch_shape.claim_group_sizes,
        prepared_claims.opening_points.len(),
    )?;
    let max_num_vars = expanded.seed.max_num_vars;
    let root_key = AkitaScheduleLookupKey::with_batch(
        max_num_vars,
        prepared_claims.num_vars,
        prepared_claims.layout_num_claims,
        batch_summary,
    );
    let schedule = select_schedule(
        max_num_vars,
        prepared_claims.num_vars,
        prepared_claims.layout_num_claims,
        batch_summary,
    )?;

    if schedule_is_root_direct(&schedule) {
        return prove_root_direct_from_polys::<F, D, P>(&prepared_claims.flat_polys);
    }

    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        return Err(AkitaError::InvalidSetup(
            "root schedule does not start with a fold".to_string(),
        ));
    };
    let next_inputs = AkitaScheduleInputs {
        max_num_vars: root_key.max_num_vars,
        level: 1,
        current_w_len: root_step.next_w_len,
    };
    let root_next_params = select_root_next_params(&schedule, next_inputs)?;

    prove_folded(
        prepared_claims,
        schedule,
        root_next_params,
        transcript,
        basis,
    )
}

/// Build the recursive suffix from a root handoff, then assemble the final
/// folded batched proof.
///
/// The caller owns suffix schedule/config policy inside `build_suffix`; this
/// helper owns the config-free handoff from root raw output into suffix
/// construction and final proof assembly.
///
/// # Errors
///
/// Returns an error if suffix construction fails.
pub fn build_folded_batched_proof_with_suffix<F, const D: usize, BuildSuffix>(
    raw: RootLevelRawOutput<F, D>,
    build_suffix: BuildSuffix,
) -> Result<(AkitaBatchedProof<F>, usize), AkitaError>
where
    F: FieldCore,
    BuildSuffix: FnOnce(RecursiveProverState<F>) -> Result<RecursiveSuffixOutcome<F>, AkitaError>,
{
    let RootLevelRawOutput {
        y_rings,
        v,
        stage1,
        stage2_sumcheck,
        stage2_setup_claim_reduction,
        w_commitment_proof,
        w_eval,
        next_state,
    } = raw;
    let suffix = build_suffix(next_state)?;
    let RecursiveSuffixOutcome {
        levels,
        num_levels,
        final_state,
        final_log_basis,
    } = suffix;
    let root = AkitaBatchedRootProof::new_two_stage_with_setup_claim_reduction::<D>(
        y_rings,
        v,
        stage1,
        stage2_sumcheck,
        stage2_setup_claim_reduction,
        w_commitment_proof,
        w_eval,
    );
    let steps = build_final_proof_steps(levels, &final_state, final_log_basis);
    Ok((AkitaBatchedProof { root, steps }, num_levels))
}

/// Prove a folded batched root and assemble the recursive suffix.
///
/// The prover crate owns config-free folded-root preparation: root schedule
/// shape checks, opening-point reduction, commitment row shape validation,
/// root fold proving, recursive suffix handoff, and final proof assembly. The
/// caller supplies the already-selected first recursive commitment params plus
/// policy callbacks for committing root's next `w` and proving the suffix.
///
/// # Errors
///
/// Returns an error if the schedule is not folded, root inputs are malformed,
/// root proving fails, or suffix construction fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_folded_batched_with_policy<'a, F, T, P, const D: usize, CommitRootNext, BuildSuffix>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    prepared_claims: PreparedBatchedProveInputs<'a, F, P, D>,
    schedule: &Schedule,
    basis: BasisMode,
    root_next_params: &LevelParams,
    commit_root_next: CommitRootNext,
    build_suffix: BuildSuffix,
) -> Result<(AkitaBatchedProof<F>, usize), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
    CommitRootNext: FnOnce(
        &mut MultiDNttCaches,
        &RecursiveWitnessFlat,
    )
        -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
    BuildSuffix: FnOnce(
        &mut MultiDNttCaches,
        &mut MultiDNttCaches,
        RecursiveProverState<F>,
        &Schedule,
        &mut T,
    ) -> Result<RecursiveSuffixOutcome<F>, AkitaError>,
{
    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        return Err(AkitaError::InvalidSetup(
            "root schedule does not start with a fold".to_string(),
        ));
    };

    let mut ntt_cache = MultiDNttCaches::new();
    let mut commit_ntt_cache = MultiDNttCaches::new();
    let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
    let prepared_points = prepared_claims
        .opening_points
        .iter()
        .map(|opening_point| {
            prepare_root_opening_point::<F, D>(opening_point, basis, &root_step.params, alpha_bits)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if prepared_claims
        .commitments_by_point
        .iter()
        .any(|commitment| commitment.u.len() != root_step.params.b_key.row_len())
    {
        return Err(AkitaError::InvalidInput(
            "batched_prove received a commitment with the wrong length".to_string(),
        ));
    }

    let raw = prove_root_fold_with_params::<F, T, D, P, _>(
        expanded,
        ntt_shared,
        transcript,
        &prepared_claims.flat_polys,
        &prepared_claims.batch_shape,
        &prepared_points,
        &prepared_claims.commitments_by_point,
        prepared_claims.flat_hints,
        &root_step.params,
        root_next_params.log_basis,
        |w| commit_root_next(&mut commit_ntt_cache, w),
    )?;

    build_folded_batched_proof_with_suffix::<F, D, _>(raw, |next_state| {
        build_suffix(
            &mut ntt_cache,
            &mut commit_ntt_cache,
            next_state,
            schedule,
            transcript,
        )
    })
}

/// Drive recursive fold suffix levels using caller-supplied schedule and
/// ring-dimension policies.
///
/// Root config policy selects the current/next level parameters through
/// `select_fold_execution`, and dynamic ring dispatch lives inside
/// `prove_level`. This helper owns the config-free suffix loop, state
/// threading, and terminal direct-basis resolution.
///
/// # Errors
///
/// Returns an error if schedule selection, level proving, or terminal direct
/// basis resolution fails.
pub fn prove_recursive_suffix_with_policy<F, SelectFold, ProveLevel>(
    max_num_vars: usize,
    initial_state: RecursiveProverState<F>,
    schedule: &Schedule,
    mut select_fold_execution: SelectFold,
    mut prove_level: ProveLevel,
) -> Result<RecursiveSuffixOutcome<F>, AkitaError>
where
    F: FieldCore,
    SelectFold:
        FnMut(usize, AkitaScheduleInputs, u32) -> Result<(LevelParams, LevelParams), AkitaError>,
    ProveLevel: FnMut(
        usize,
        &RecursiveProverState<F>,
        &LevelParams,
        LevelParams,
    ) -> Result<ProveLevelOutput<F>, AkitaError>,
{
    let mut levels = Vec::new();
    let mut current_state = initial_state;
    let mut level = 1usize;
    let planned_num_levels = schedule_num_fold_levels(schedule);

    loop {
        let handle = &current_state.handles[0];
        let current_w_len = handle.w.len();
        if level >= planned_num_levels {
            break;
        }

        let inputs = AkitaScheduleInputs {
            max_num_vars,
            level,
            current_w_len,
        };
        let (level_params, next_params) = select_fold_execution(level, inputs, handle.log_basis)?;
        let out = prove_level(level, &current_state, &level_params, next_params)?;

        levels.push(out.level_proof);
        current_state = out.next_state;
        level += 1;
    }

    let final_log_basis = resolve_final_log_basis(schedule, &current_state)?;

    Ok(RecursiveSuffixOutcome {
        levels,
        num_levels: level,
        final_state: current_state,
        final_log_basis,
    })
}

/// Prove one recursive fold level after the caller has built its quadratic
/// equation and selected the commitment policy for the next `w`.
///
/// The caller owns config/schedule decisions through `commit_w_for_next`; this
/// function owns the config-free prover mechanics: build `w`, commit it using
/// that closure, finish ring switching, run stage-1/stage-2 sumchecks, and
/// produce the next recursive state.
///
/// Phase D-full v2 hooks the deferred setup-claim-reduction
/// `(r_setup, s_opening_value)` into the next level's recursive open
/// here; see `specs/phase-d-full-handoff.md` slice F. The single-handle
/// path remains in place; the per-level `mle` check in
/// `verify_setup_claim_reduction` anchors soundness until slice F lands.
///
/// # Errors
///
/// Returns an error if ring switching, recursive commitment, or either
/// sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_fold_level_from_quadratic<F, T, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    commitment_rows: &[CyclotomicRing<F, D>],
    level: usize,
    lp: &LevelParams,
    next_log_basis: u32,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
{
    let w = ring_switch_build_w::<F, { D }>(&mut quad_eq, expanded, ntt_shared, lp)?;
    let (w_commitment_flat, w_hint_cache) = {
        let _span = tracing::info_span!("commit_w_level", level).entered();
        commit_w_for_next(&w)?
    };
    let w_commitment_proof = w_commitment_flat.clone();

    let rs = ring_switch_finalize::<F, T, { D }>(
        &quad_eq,
        expanded,
        transcript,
        w,
        w_commitment_flat,
        &w_commitment_proof,
        w_hint_cache,
        lp,
    )?;

    let relation_claim =
        relation_claim_from_rows::<F, D>(&rs.tau1, rs.alpha, &quad_eq.v, commitment_rows, &y_rings);
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
        tau1,
        b,
        alpha,
    } = rs;
    let w_commitment = w_commitment.ok_or_else(|| {
        AkitaError::InvalidSetup("prover ring switch dropped w commitment".to_string())
    })?;
    let tau0_reordered = reorder_stage1_coords(&tau0, col_bits, ring_bits);
    let (stage1_proof, r_stage1, s_claim) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let stage1_prover = AkitaStage1Prover::new(
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
    let claim_to_point = quad_eq.claim_to_point().to_vec();
    let claim_group_sizes = quad_eq.claim_group_sizes().to_vec();
    let gamma_for_prepare = quad_eq.gamma().to_vec();
    let num_eval_rows_for_prepare = quad_eq.num_eval_rows();
    let opening_points_len = quad_eq.opening_points().len();
    let stage1_challenges_for_prepare = quad_eq.challenges.clone();
    let (stage2_sumcheck, sumcheck_challenges, _stage2_final_claim, w_eval) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
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

    let (setup_claim_reduction, r_setup) = if lp.use_setup_claim_reduction {
        let _span = tracing::info_span!("setup_claim_reduction", level).entered();
        let prepared = prepare_m_eval::<F, D>(
            &stage1_challenges_for_prepare,
            alpha,
            lp,
            &tau1,
            &claim_group_sizes,
            &gamma_for_prepare,
            num_eval_rows_for_prepare,
            opening_points_len,
            &claim_to_point,
        )?;
        let x_challenges = &sumcheck_challenges[ring_bits..];
        let out = prove_setup_claim_reduction::<F, _, D>(
            &prepared,
            expanded,
            x_challenges,
            alpha,
            transcript,
        )?;
        let r_setup = out.challenges.clone();
        let payload = SetupClaimReductionPayload {
            m_setup_eval: out.input_claim,
            s_opening_value: out.s_opening_value,
            sumcheck: out.proof,
        };
        (Some(payload), Some(r_setup))
    } else {
        (None, None)
    };

    let (level_proof, sumcheck_challenges) = (
        AkitaLevelProof::new_two_stage_with_setup_claim_reduction::<D>(
            y_rings,
            quad_eq.v,
            stage1_proof,
            stage2_sumcheck,
            setup_claim_reduction,
            w_commitment_proof,
            w_eval,
        ),
        sumcheck_challenges,
    );

    // Phase D-full v2 slice F will route the deferred setup-claim
    // `(r_setup, s_opening_value)` into the next level's recursive
    // open as a second handle; see `specs/phase-d-full-handoff.md`.
    let _ = r_setup;

    let handles = vec![RecursivePolyHandle {
        w,
        commitment: w_commitment,
        hint: w_hint.ok_or_else(|| {
            AkitaError::InvalidSetup("prover ring switch dropped recursive hint cache".to_string())
        })?,
        log_basis: next_log_basis,
        opening_point: sumcheck_challenges,
    }];

    Ok(ProveLevelOutput {
        level_proof,
        next_state: RecursiveProverState { handles },
    })
}

/// Prove one recursive fold level using already-selected current and next
/// level parameters.
///
/// Thin single-claim wrapper over [`prove_recursive_multi_fold_with_params`].
/// Construction sites with one polynomial pass through this helper to
/// preserve the legacy single-claim recursive wire shape.
///
/// # Errors
///
/// Returns whatever [`prove_recursive_multi_fold_with_params`] returns.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_recursive_fold_with_params<F, T, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    witness: &RecursiveWitnessView<'_, F, D>,
    opening_point: &[F],
    hint: AkitaCommitmentHint<F, D>,
    commitment: &FlatRingVec<F>,
    level: usize,
    level_params: &LevelParams,
    next_log_basis: u32,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
{
    prove_recursive_multi_fold_with_params::<F, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        &[witness],
        &[opening_point],
        vec![hint],
        &[commitment],
        level,
        level_params,
        next_log_basis,
        commit_w_for_next,
    )
}

/// Prove one recursive fold level with N polynomial claims jointly.
///
/// All `witnesses`, `opening_points`, `hints`, and `commitments` slices
/// must have the same length `N`. Each claim's opening point may have a
/// different length (each is padded to the level's
/// `m_vars + r_vars + alpha_bits` independently); the level's
/// [`LevelParams`] is shared across all claims.
///
/// The wire shape for `N == 1` exactly matches today's legacy
/// single-claim recursive wire: one commitment + one padded point + one
/// y-ring, no openings absorbed, no `gamma` sampled. For `N > 1` the
/// transcript layout mirrors [`verify_one_level`]'s multi-claim path:
/// commitments × N, padded points × N, openings × N, sample `gamma` × N,
/// y-rings × N. (For now this assumes a 1-claim-per-point layout, so
/// `num_eval_rows == N` and each y-ring carries a single claim's
/// contribution.)
///
/// Phase D-full slice F discharges the deferred setup-claim
/// `(r_setup, s_opening_value)` here as `claims[1]` (the `S` opening),
/// alongside the folded witness as `claims[0]`. Slice E first extends
/// this primitive to admit per-claim `LevelParams` and mixed witness
/// types; today it requires homogeneous i8-digit witnesses sharing
/// one `LevelParams`.
///
/// # Errors
///
/// Returns an error if slice lengths disagree, any opening-point
/// length underflows the level's alpha, witness folding fails, the
/// recursive quadratic equation rejects, or the folded prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_recursive_multi_fold_with_params<F, T, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    witnesses: &[&RecursiveWitnessView<'_, F, D>],
    opening_points: &[&[F]],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    commitments: &[&FlatRingVec<F>],
    level: usize,
    level_params: &LevelParams,
    next_log_basis: u32,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
{
    let num_claims = witnesses.len();
    if num_claims == 0
        || opening_points.len() != num_claims
        || hints.len() != num_claims
        || commitments.len() != num_claims
    {
        return Err(AkitaError::InvalidInput(
            "prove_recursive_multi_fold_with_params: slice length mismatch".to_string(),
        ));
    }

    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level,
            num_claims,
            "prove_recursive_multi_fold_with_params"
        );
    }

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    let target_num_vars = level_params.m_vars + level_params.r_vars + alpha;

    // Per-claim padded points, ring opening points, inner reductions,
    // evaluate_and_fold outputs.
    let mut padded_points: Vec<Vec<F>> = Vec::with_capacity(num_claims);
    let mut ring_opening_points: Vec<akita_types::RingOpeningPoint<F>> =
        Vec::with_capacity(num_claims);
    let mut inner_reductions: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(num_claims);
    let mut per_claim_y_rings: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(num_claims);
    let mut per_claim_w_folded: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(num_claims);
    for (claim_idx, (witness, opening_point)) in
        witnesses.iter().zip(opening_points.iter()).enumerate()
    {
        if opening_point.len() < alpha {
            return Err(AkitaError::InvalidPointDimension {
                expected: alpha,
                actual: opening_point.len(),
            });
        }
        let mut padded_point = opening_point.to_vec();
        padded_point.resize(target_num_vars, F::zero());
        let inner_point = &padded_point[..alpha];
        let outer_point = &padded_point[alpha..];

        let inner_reduction =
            reduce_inner_opening_to_ring_element::<F, { D }>(inner_point, BasisMode::Lagrange)?;
        let ring_opening_point = {
            let _span =
                tracing::info_span!("ring_opening_point", level, claim_idx = claim_idx).entered();
            ring_opening_point_from_field::<F>(
                outer_point,
                level_params.r_vars,
                level_params.m_vars,
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
                claim_idx = claim_idx,
                num_ring_elems = witness.num_ring_elems()
            )
            .entered();
            witness.evaluate_and_fold(
                eval_outer_scalars,
                fold_scalars,
                level_params.block_len,
                level_params.num_blocks,
            )
        };

        padded_points.push(padded_point);
        ring_opening_points.push(ring_opening_point);
        inner_reductions.push(inner_reduction);
        per_claim_y_rings.push(y_ring);
        per_claim_w_folded.push(w_folded);
    }

    // Multi-claim transcript layout mirroring `verify_one_level`:
    //   commitments × N, padded points × N, [openings × N, sample γ × N if N>1],
    //   y-rings × N_points.
    for commitment in commitments {
        commitment.append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;
    }
    for padded_point in &padded_points {
        for pt in padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    let gamma: Vec<F> = if num_claims > 1 {
        // Each claim's opening is the first coefficient of
        // `y_ring * σ_{-1}(v)`. This matches the verifier's per-point
        // trace check: `trace(y_ring * σ_{-1}(v)) = d · γ · opening`.
        let openings: Vec<F> = inner_reductions
            .iter()
            .zip(per_claim_y_rings.iter())
            .map(|(inner_reduction, y_ring)| {
                (*y_ring * inner_reduction.sigma_m1()).coefficients()[0]
            })
            .collect();
        for opening in &openings {
            transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, opening);
        }
        (0..num_claims)
            .map(|_| transcript.challenge_scalar(CHALLENGE_EVAL_BATCH))
            .collect()
    } else {
        vec![F::one()]
    };
    // With the 1-claim-per-point inference rule, each claim drives its
    // own y-ring slot. The verifier's trace check re-derives the
    // per-point combined opening from `gamma[i] * opening[i]`.
    for y_ring in &per_claim_y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    // Build commitment-row references for the recursive QE.
    let commitment_us: Vec<&[CyclotomicRing<F, D>]> = commitments
        .iter()
        .map(|c| c.as_ring_slice::<D>())
        .collect::<Result<Vec<_>, _>>()?;
    let claim_to_point: Vec<usize> = (0..num_claims).collect();
    let claim_group_sizes: Vec<usize> = vec![1usize; num_claims];
    let num_eval_rows = num_claims;

    let quad_eq = Box::new(QuadraticEquation::<F, { D }>::new_recursive_prover(
        ntt_shared,
        ring_opening_points,
        claim_to_point,
        witnesses,
        per_claim_w_folded,
        &claim_group_sizes,
        level_params.clone(),
        hints,
        transcript,
        &commitment_us,
        &per_claim_y_rings,
        gamma,
        num_eval_rows,
        expanded.seed.max_stride,
    )?);

    // Commitment-rows slice for `prove_fold_level_from_quadratic`. For
    // N == 1 this is just the single commitment's u; for N > 1 we
    // concatenate all commitment u-rows.
    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if num_claims == 1 {
        None
    } else {
        let mut rows = Vec::with_capacity(num_claims * level_params.b_key.row_len());
        for commitment_u in &commitment_us {
            rows.extend_from_slice(commitment_u);
        }
        Some(rows)
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(rows) => rows.as_slice(),
        None => commitment_us[0],
    };

    prove_fold_level_from_quadratic::<F, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        commitment_rows,
        level,
        level_params,
        next_log_basis,
        quad_eq,
        per_claim_y_rings,
        commit_w_for_next,
    )
}

/// Prove one recursive fold level from D-erased recursive state using
/// caller-supplied config policy.
///
/// The prover crate owns the state unpacking, typed recursive witness view,
/// typed hint conversion, opening-point handoff, and fold proof mechanics.
/// The caller supplies only the current-witness layout policy and the
/// next-level recursive commitment policy.
///
/// # Errors
///
/// Returns an error if the current witness cannot be viewed at `D`, the hint
/// cannot be typed at `D`, layout selection fails, or recursive proving fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_recursive_level_with_policy<F, T, const D: usize, CurrentLayout, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    current_state: &RecursiveProverState<F>,
    level: usize,
    level_params: &LevelParams,
    next_log_basis: u32,
    current_layout: CurrentLayout,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    CurrentLayout: FnOnce(&LevelParams, usize) -> Result<LevelParams, AkitaError>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
{
    let _setup_span = tracing::info_span!("inter_level_setup", level).entered();

    if current_state.handles.is_empty() {
        return Err(AkitaError::InvalidInput(
            "prove_recursive_level_with_policy: empty recursive state".to_string(),
        ));
    }
    let current_w_len = current_state.handles[0].w.len();
    let w_lp = current_layout(level_params, current_w_len)?;
    let views: Vec<RecursiveWitnessView<'_, F, D>> = current_state
        .handles
        .iter()
        .map(|h| h.w.view::<F, D>())
        .collect::<Result<_, _>>()?;
    let view_refs: Vec<&RecursiveWitnessView<'_, F, D>> = views.iter().collect();
    let opening_points: Vec<Vec<F>> = current_state
        .handles
        .iter()
        .map(|h| h.opening_point.clone())
        .collect();
    let opening_point_refs: Vec<&[F]> = opening_points.iter().map(Vec::as_slice).collect();
    let typed_hints: Vec<AkitaCommitmentHint<F, D>> = current_state
        .handles
        .iter()
        .map(|h| h.hint.to_typed::<D>())
        .collect::<Result<_, _>>()?;
    let commitment_refs: Vec<&FlatRingVec<F>> = current_state
        .handles
        .iter()
        .map(|h| &h.commitment)
        .collect();
    drop(_setup_span);

    prove_recursive_multi_fold_with_params::<F, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        &view_refs,
        &opening_point_refs,
        typed_hints,
        &commitment_refs,
        level,
        &w_lp,
        next_log_basis,
        commit_w_for_next,
    )
}

/// Prove the folded root level using already-selected root and next-level
/// parameters.
///
/// The caller owns schedule/config selection and passes the expected next
/// recursive witness length, next digit basis, and commitment policy for that
/// witness. This function owns root polynomial folding, public root transcript
/// setup, root quadratic-equation construction, and the folded-root prover
/// mechanics.
///
/// # Errors
///
/// Returns an error if root inputs are malformed, polynomial folding or
/// quadratic-equation construction fails, or the folded-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_root_fold_with_params<F, T, const D: usize, P, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    polys: &[&P],
    batch_shape: &MultiPointBatchShape,
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    next_log_basis: u32,
    commit_w_for_next: CommitW,
) -> Result<RootLevelRawOutput<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
{
    let claim_to_point = &batch_shape.claim_to_point;
    let claim_group_sizes = &batch_shape.claim_group_sizes;
    let point_group_sizes = &batch_shape.point_group_sizes;

    if prepared_points.is_empty() || claim_to_point.len() != polys.len() {
        return Err(AkitaError::InvalidInput(
            "invalid root-level inputs".to_string(),
        ));
    }
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= prepared_points.len())
    {
        return Err(AkitaError::InvalidInput(
            "root-level claim-to-point index out of range".to_string(),
        ));
    }

    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level = 0usize,
            num_claims = claim_to_point.len(),
            num_points = prepared_points.len(),
            "prove_root_fold_with_params"
        );
    }

    let (per_claim_y_rings, w_folded_by_poly) = {
        let _span = tracing::info_span!(
            "evaluate_and_fold",
            level = 0usize,
            num_polys = polys.len(),
            num_points = prepared_points.len()
        )
        .entered();
        let mut per_claim_y_rings = Vec::with_capacity(polys.len());
        let mut w_folded_by_poly = Vec::with_capacity(polys.len());
        for (poly, &point_idx) in polys.iter().zip(claim_to_point.iter()) {
            let prepared_point = &prepared_points[point_idx];
            let (y_ring, w_folded) = poly.evaluate_and_fold(
                &prepared_point.ring_opening_point.b,
                &prepared_point.ring_opening_point.a,
                root_params.block_len,
            );
            per_claim_y_rings.push(y_ring);
            w_folded_by_poly.push(w_folded);
        }
        (per_claim_y_rings, w_folded_by_poly)
    };

    append_batch_shape_to_transcript::<F, T>(point_group_sizes, claim_group_sizes, transcript);
    append_batched_commitments_to_transcript(commitments, transcript);
    for prepared_point in prepared_points {
        for pt in &prepared_point.padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
    }

    let openings: Vec<F> = per_claim_y_rings
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

    let num_points = prepared_points.len();
    let mut y_rings = vec![CyclotomicRing::<F, D>::zero(); num_points];
    for (claim_idx, y_ring) in per_claim_y_rings.iter().enumerate() {
        let point_idx = claim_to_point[claim_idx];
        y_rings[point_idx] += y_ring.scale(&gamma[claim_idx]);
    }
    for y_ring in &y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    let ring_opening_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_opening_point.clone())
        .collect();
    let quad_eq = Box::new(QuadraticEquation::<F, { D }>::new_prover(
        ntt_shared,
        ring_opening_points,
        claim_to_point.clone(),
        polys,
        w_folded_by_poly,
        claim_group_sizes,
        root_params.clone(),
        hints,
        transcript,
        commitments,
        &y_rings,
        gamma,
        expanded.seed.max_stride,
    )?);

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    prove_root_fold_from_quadratic::<F, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        commitment_rows,
        root_params,
        next_log_basis,
        quad_eq,
        y_rings,
        commit_w_for_next,
    )
}

/// Prove the folded root level after root orchestration has built its
/// quadratic equation and selected the next recursive commitment policy.
///
/// The root caller owns transcript setup for public openings and gamma
/// batching, schedule selection, and the commitment-row view used by the root
/// relation. This function owns the config-free prover mechanics from `w`
/// construction through the stage proofs and next recursive state.
///
/// # Errors
///
/// Returns an error if ring switching, recursive commitment, or either
/// sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_root_fold_from_quadratic<F, T, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    commitment_rows: &[CyclotomicRing<F, D>],
    lp: &akita_types::LevelParams,
    next_log_basis: u32,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    commit_w_for_next: CommitW,
) -> Result<RootLevelRawOutput<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
{
    let w = ring_switch_build_w::<F, { D }>(&mut quad_eq, expanded, ntt_shared, lp)?;
    let (w_commitment_flat, w_hint_cache) = {
        let _span = tracing::info_span!("commit_w_level", level = 0usize).entered();
        commit_w_for_next(&w)?
    };
    let w_commitment_proof = w_commitment_flat.clone();

    let rs = ring_switch_finalize_with_claim_groups::<F, T, { D }>(
        &quad_eq,
        expanded,
        transcript,
        w,
        w_commitment_flat,
        &w_commitment_proof,
        w_hint_cache,
        lp,
    )?;

    let relation_claim =
        relation_claim_from_rows::<F, D>(&rs.tau1, rs.alpha, &quad_eq.v, commitment_rows, &y_rings);

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
        tau1,
        b,
        alpha,
    } = rs;
    let w_commitment = w_commitment.ok_or_else(|| {
        AkitaError::InvalidSetup("prover ring switch dropped w commitment".to_string())
    })?;
    let tau0_reordered = reorder_stage1_coords(&tau0, col_bits, ring_bits);
    let (stage1_proof, r_stage1, s_claim) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let stage1_prover = AkitaStage1Prover::new(
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
    let claim_to_point = quad_eq.claim_to_point().to_vec();
    let claim_group_sizes = quad_eq.claim_group_sizes().to_vec();
    let gamma_for_prepare = quad_eq.gamma().to_vec();
    let num_eval_rows_for_prepare = quad_eq.num_eval_rows();
    let opening_points_len = quad_eq.opening_points().len();
    let stage1_challenges_for_prepare = quad_eq.challenges.clone();
    let (stage2_sumcheck, sumcheck_challenges, _stage2_final_claim, w_eval) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
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

    let (stage2_setup_claim_reduction, r_setup) = if lp.use_setup_claim_reduction {
        let _span = tracing::info_span!("setup_claim_reduction", level = 0usize).entered();
        let prepared = prepare_m_eval::<F, D>(
            &stage1_challenges_for_prepare,
            alpha,
            lp,
            &tau1,
            &claim_group_sizes,
            &gamma_for_prepare,
            num_eval_rows_for_prepare,
            opening_points_len,
            &claim_to_point,
        )?;
        let x_challenges = &sumcheck_challenges[ring_bits..];
        let out = prove_setup_claim_reduction::<F, _, D>(
            &prepared,
            expanded,
            x_challenges,
            alpha,
            transcript,
        )?;
        let r_setup = out.challenges.clone();
        let payload = SetupClaimReductionPayload {
            m_setup_eval: out.input_claim,
            s_opening_value: out.s_opening_value,
            sumcheck: out.proof,
        };
        (Some(payload), Some(r_setup))
    } else {
        (None, None)
    };

    // Phase D-full v2 slice F will route the deferred setup-claim
    // `(r_setup, s_opening_value)` into the next level's recursive
    // open as a second handle; see `specs/phase-d-full-handoff.md`.
    let _ = r_setup;

    let handles = vec![RecursivePolyHandle {
        w,
        commitment: w_commitment,
        hint: w_hint.ok_or_else(|| {
            AkitaError::InvalidSetup("prover ring switch dropped recursive hint cache".to_string())
        })?,
        log_basis: next_log_basis,
        opening_point: sumcheck_challenges,
    }];

    Ok(RootLevelRawOutput {
        y_rings,
        v: quad_eq.v,
        stage1: stage1_proof,
        stage2_sumcheck,
        stage2_setup_claim_reduction,
        w_commitment_proof,
        w_eval,
        next_state: RecursiveProverState { handles },
    })
}
