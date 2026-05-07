//! Prover flow state shared by root orchestration during crate extraction.

use crate::kernels::crt_ntt::NttSlotCache;
use crate::protocol::ring_switch::{
    ring_switch_build_w, ring_switch_finalize, ring_switch_finalize_with_claim_groups,
    RingSwitchOutput,
};
use crate::protocol::sumcheck::{AkitaStage1Prover, AkitaStage2Prover};
use crate::{
    AkitaPolyOps, MultiDNttCaches, ProverClaims, ProverIncidenceGroup, QuadraticEquation,
    RecursiveCommitmentHintCache, RecursiveWitnessFlat, RecursiveWitnessView,
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
    flatten_batched_commitment_rows, prepare_root_opening_point, relation_claim_from_rows,
    reorder_stage1_coords, ring_opening_point_from_field, schedule_is_root_direct,
    schedule_num_fold_levels, validate_batched_inputs, AkitaBatchedProof, AkitaBatchedRootProof,
    AkitaCommitmentHint, AkitaExpandedSetup, AkitaLevelProof, AkitaProofStep,
    AkitaRootBatchSummary, AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaStage1Proof,
    BasisMode, BlockOrder, ClaimIncidence, ClaimIncidenceLimits, ClaimIncidenceSummary,
    DirectWitnessProof, FlatRingVec, IncidenceClaim, LevelParams, MultiPointBatchShape,
    PackedDigits, PreparedRootOpeningPoint, RingCommitment, Schedule, Step,
};

/// Runtime state carried between recursive prove levels.
pub struct RecursiveProverState<F: FieldCore> {
    /// Current recursive witness.
    pub w: RecursiveWitnessFlat,
    /// Current recursive witness commitment.
    pub commitment: FlatRingVec<F>,
    /// D-erased recursive commitment hint cache.
    pub hint: RecursiveCommitmentHintCache<F>,
    /// Current digit basis, as `log2(b)`.
    pub log_basis: u32,
    /// Sumcheck challenges that become the next recursive opening point.
    pub sumcheck_challenges: Vec<F>,
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
pub struct PreparedBatchedProveInputs<'a, F: FieldCore, E: FieldCore, P, const D: usize> {
    /// Distinct opening points in caller order.
    pub opening_points: Vec<&'a [E]>,
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
    if direct_step.current_w_len != current_state.w.len()
        || direct_step.bits_per_elem != current_state.log_basis
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
    let final_w =
        PackedDigits::from_i8_digits_with_min_bits(final_state.w.as_i8_digits(), final_log_basis);
    let mut steps = levels
        .into_iter()
        .map(AkitaProofStep::Fold)
        .collect::<Vec<_>>();
    steps.push(AkitaProofStep::Direct(DirectWitnessProof::PackedDigits(
        final_w,
    )));
    steps
}

struct ProverPreparedIncidence<'a, F: FieldCore, E: FieldCore, P, const D: usize> {
    points: Vec<&'a [E]>,
    groups: Vec<ProverIncidenceGroup<'a, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>>,
    summary: ClaimIncidenceSummary,
}

fn prover_claims_to_incidence<'a, F, E, P, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    claims: ProverClaims<'a, E, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
) -> Result<ProverPreparedIncidence<'a, F, E, P, D>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
{
    let points: Vec<&'a [E]> = claims.iter().map(|(point, _)| *point).collect();
    let mut groups = Vec::new();
    let mut incidence_claims = Vec::new();

    for (point_idx, (_, groups_at_point)) in claims.into_iter().enumerate() {
        for group in groups_at_point {
            let group_idx = groups.len();
            let prover_group = ProverIncidenceGroup::from(group);
            incidence_claims.extend((0..prover_group.poly_count()).map(|poly_idx| {
                IncidenceClaim {
                    point_idx,
                    group_idx,
                    poly_idx,
                    // Prover inputs do not contain claimed evaluations. The
                    // shared incidence validator ignores this field, so zero is
                    // only a structural placeholder.
                    claimed_eval: E::zero(),
                }
            }));
            groups.push(prover_group);
        }
    }

    let verifier_groups = groups
        .iter()
        .map(ProverIncidenceGroup::incidence_group)
        .collect();
    let incidence = ClaimIncidence {
        points: points.clone(),
        groups: verifier_groups,
        claims: incidence_claims,
    };
    let summary = incidence.validate(ClaimIncidenceLimits {
        max_num_vars: expanded.seed.max_num_vars,
        max_num_points: expanded.seed.max_num_points,
        max_num_claims: expanded.seed.max_num_batched_polys,
    })?;

    Ok(ProverPreparedIncidence {
        points,
        groups,
        summary,
    })
}

/// Validate and flatten batched prover claims into the root proof shape.
///
/// # Errors
///
/// Returns an error if the claim shape exceeds setup capacity, mixes
/// incompatible dimensions, or has malformed batch counts.
pub fn prepare_batched_prove_inputs<'a, F, E, P, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    claims: ProverClaims<'a, E, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
) -> Result<PreparedBatchedProveInputs<'a, F, E, P, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore,
{
    validate_batched_inputs(expanded, &claims, |group| group.polynomials.len(), true)?;

    let prepared_incidence = prover_claims_to_incidence(expanded, claims)?;
    let opening_points = prepared_incidence.points;
    let commitments_by_point = prepared_incidence
        .groups
        .iter()
        .map(|group| group.commitment.clone())
        .collect();
    let num_vars = prepared_incidence.summary.num_vars;
    let layout_num_claims = prepared_incidence.summary.num_claims;
    let batch_shape = prepared_incidence.summary.multi_point_batch_shape();
    let flat_polys = prepared_incidence
        .summary
        .claim_to_group
        .iter()
        .zip(prepared_incidence.summary.claim_poly_indices.iter())
        .map(|(&group_idx, &poly_idx)| &prepared_incidence.groups[group_idx].polynomials[poly_idx])
        .collect();
    let flat_hints = prepared_incidence
        .groups
        .into_iter()
        .map(|group| group.hint)
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
        PreparedBatchedProveInputs<'a, F, F, P, D>,
        Schedule,
        LevelParams,
        &mut T,
        BasisMode,
    ) -> Result<AkitaBatchedProof<F>, AkitaError>,
{
    let prepared_claims = prepare_batched_prove_inputs::<F, F, P, D>(expanded, claims)?;
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
    let root = AkitaBatchedRootProof::new_two_stage::<D>(
        y_rings,
        v,
        stage1,
        stage2_sumcheck,
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
    prepared_claims: PreparedBatchedProveInputs<'a, F, F, P, D>,
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
        root_step.next_w_len,
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
        let current_w_len = current_state.w.len();
        if level >= planned_num_levels {
            break;
        }

        let inputs = AkitaScheduleInputs {
            max_num_vars,
            level,
            current_w_len,
        };
        let (level_params, next_params) =
            select_fold_execution(level, inputs, current_state.log_basis)?;
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
    commitment_u: &[CyclotomicRing<F, D>],
    level: usize,
    lp: &LevelParams,
    next_log_basis: u32,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    y_ring: CyclotomicRing<F, D>,
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

    let rs = ring_switch_finalize::<F, F, T, { D }>(
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

    let (level_proof, sumcheck_challenges) = (
        AkitaLevelProof::new_two_stage::<D>(
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
            hint: w_hint.ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "prover ring switch dropped recursive hint cache".to_string(),
                )
            })?,
            log_basis: next_log_basis,
            sumcheck_challenges,
        },
    })
}

/// Prove one recursive fold level using already-selected current and next
/// level parameters.
///
/// The caller owns schedule/config selection and passes the next-level
/// commitment policy as a closure. This function owns recursive opening-point
/// reduction, witness folding, public recursive transcript absorbs, recursive
/// quadratic-equation construction, and the folded-level prover mechanics.
///
/// # Errors
///
/// Returns an error if the recursive opening point has the wrong dimension,
/// witness folding or quadratic-equation construction fails, or the folded
/// prover fails.
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
    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level,
            "prove_recursive_fold_with_params"
        );
    }

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    if opening_point.len() < alpha {
        return Err(AkitaError::InvalidPointDimension {
            expected: alpha,
            actual: opening_point.len(),
        });
    }
    let target_num_vars = level_params.m_vars + level_params.r_vars + alpha;
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let outer_point = &padded_point[alpha..];

    let ring_opening_point = {
        let _span = tracing::info_span!("ring_opening_point", level).entered();
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

    commitment.append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;
    for pt in &padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);
    let commitment_u = commitment.as_ring_slice::<D>()?;

    let quad_eq = Box::new(QuadraticEquation::<F, { D }>::new_recursive_prover(
        ntt_shared,
        ring_opening_point,
        witness,
        w_folded,
        level_params.clone(),
        hint,
        transcript,
        commitment_u,
        &y_ring,
        expanded.seed.max_stride,
    )?);

    prove_fold_level_from_quadratic::<F, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        commitment_u,
        level,
        level_params,
        next_log_basis,
        quad_eq,
        y_ring,
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

    let current_w = &current_state.w;
    let opening_point = current_state.sumcheck_challenges.clone();
    let w_lp = current_layout(level_params, current_w.len())?;
    let w_view = current_w.view::<F, D>()?;
    let typed_hint: AkitaCommitmentHint<F, D> = current_state.hint.to_typed::<D>()?;
    drop(_setup_span);

    prove_recursive_fold_with_params::<F, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        &w_view,
        &opening_point,
        typed_hint,
        &current_state.commitment,
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
    expected_w_len: usize,
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
        expected_w_len,
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
    expected_w_len: usize,
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
    if w.len() != expected_w_len {
        return Err(AkitaError::InvalidSetup(
            "scheduled root next-w length did not match runtime witness".to_string(),
        ));
    }
    let (w_commitment_flat, w_hint_cache) = {
        let _span = tracing::info_span!("commit_w_level", level = 0usize).entered();
        commit_w_for_next(&w)?
    };
    let w_commitment_proof = w_commitment_flat.clone();

    let rs = ring_switch_finalize_with_claim_groups::<F, F, T, { D }>(
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
        tau1: _,
        b,
        alpha: _,
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

    Ok(RootLevelRawOutput {
        y_rings,
        v: quad_eq.v,
        stage1: stage1_proof,
        stage2_sumcheck,
        w_commitment_proof,
        w_eval,
        next_state: RecursiveProverState {
            w,
            commitment: w_commitment,
            hint: w_hint.ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "prover ring switch dropped recursive hint cache".to_string(),
                )
            })?,
            log_basis: next_log_basis,
            sumcheck_challenges,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp2, Fp32, NegOneNr};
    use akita_types::{AkitaSetupSeed, FlatMatrix};

    type F = Fp32<251>;
    type E = Fp2<F, NegOneNr>;

    fn setup() -> AkitaExpandedSetup<F> {
        AkitaExpandedSetup {
            seed: AkitaSetupSeed {
                max_num_vars: 3,
                max_num_batched_polys: 4,
                max_num_points: 2,
                max_stride: 1,
                public_matrix_seed: [0u8; 32],
            },
            shared_matrix: FlatMatrix::from_flat_data(vec![F::zero()], 1),
        }
    }

    #[test]
    fn prover_claim_preparation_accepts_extension_points() {
        let point = [
            E::new(F::from_u64(1), F::from_u64(2)),
            E::new(F::from_u64(3), F::from_u64(4)),
        ];
        let polys = [10usize, 11usize];
        let commitment = RingCommitment::<F, 2>::default();
        let claims = vec![(
            &point[..],
            vec![crate::CommittedPolynomials {
                polynomials: &polys[..],
                commitment: &commitment,
                hint: AkitaCommitmentHint::new(Vec::new()),
            }],
        )];

        let prepared = prepare_batched_prove_inputs::<F, E, usize, 2>(&setup(), claims)
            .expect("extension-valued prover points should validate by shape");

        assert_eq!(prepared.opening_points, vec![&point[..]]);
        assert_eq!(prepared.batch_shape.point_group_sizes, vec![1]);
        assert_eq!(prepared.batch_shape.claim_group_sizes, vec![2]);
        assert_eq!(prepared.batch_shape.claim_to_point, vec![0, 0]);
        assert_eq!(prepared.layout_num_claims, 2);
        assert_eq!(prepared.flat_polys, vec![&polys[0], &polys[1]]);
    }
}
