//! Top-level batched verifier orchestration once a schedule is selected.

use crate::{verify_fold_batched_proof, verify_root_direct_openings};
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_transcript::Transcript;
use akita_types::{
    checked_total_claims, schedule_is_root_direct, AkitaBatchedProof, AkitaBatchedRootProof,
    AkitaRootBatchSummary, AkitaScheduleInputs, AkitaVerifierSetup, BasisMode, DirectWitnessProof,
    LevelParams, OpeningStatement, RingCommitment, Schedule, Step,
};

/// Config-derived layouts needed by the folded-root verifier branch.
pub struct FoldVerifierLayouts {
    /// Root verifier layout derived for the selected folded-root schedule.
    pub root_lp: LevelParams,
    /// First recursive-level params reached by the root fold.
    pub next_level_params: LevelParams,
}

/// Schedule context selected by the root scheme/config layer.
pub enum BatchedVerifierScheduleContext {
    /// The selected schedule uses the root-direct fast path.
    RootDirect,
    /// The selected schedule starts with a folded root.
    Fold(Box<FoldVerifierLayouts>),
}

/// Build the verifier schedule context for an already-selected proof schedule.
///
/// Root config policy supplies the two layout callbacks; this helper owns only
/// the public schedule shape interpretation needed by verifier replay.
///
/// # Errors
///
/// Returns an error if the schedule is empty or either supplied layout callback
/// rejects the selected folded-root schedule.
pub fn prepare_batched_verifier_schedule_context<RootLayout, NextParams>(
    max_num_vars: usize,
    schedule: &Schedule,
    mut root_layout: RootLayout,
    mut next_params: NextParams,
) -> Result<BatchedVerifierScheduleContext, AkitaError>
where
    RootLayout: FnMut(AkitaScheduleInputs, &LevelParams) -> Result<LevelParams, AkitaError>,
    NextParams: FnMut(AkitaScheduleInputs) -> Result<LevelParams, AkitaError>,
{
    match schedule.steps.first() {
        Some(Step::Direct(_)) => Ok(BatchedVerifierScheduleContext::RootDirect),
        Some(Step::Fold(root_step)) => {
            let root_inputs = AkitaScheduleInputs {
                max_num_vars,
                level: 0,
                current_w_len: root_step.current_w_len,
            };
            let root_lp = root_layout(root_inputs, &root_step.params)?;
            let next_inputs = AkitaScheduleInputs {
                max_num_vars,
                level: 1,
                current_w_len: root_step.next_w_len,
            };
            let next_level_params = next_params(next_inputs)?;
            Ok(BatchedVerifierScheduleContext::Fold(Box::new(
                FoldVerifierLayouts {
                    root_lp,
                    next_level_params,
                },
            )))
        }
        None => Err(AkitaError::InvalidProof),
    }
}

/// Verify a batched proof after root schedule selection.
///
/// This owns the root-proof variant dispatch, direct witness/opening checks,
/// folded-root replay, and recursive suffix replay. The caller supplies only
/// the config-derived schedule context and a callback for root-direct
/// commitment recomputation.
///
/// # Errors
///
/// Returns an error if the proof shape disagrees with the schedule context,
/// direct openings fail, direct commitment recomputation fails, or folded-root
/// verification rejects.
#[allow(clippy::too_many_arguments)]
pub fn verify_batched_proof_with_schedule<'a, F, T, const D: usize, DirectCommitmentCheck>(
    proof: &AkitaBatchedProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    opening_statement: OpeningStatement<'a, F, RingCommitment<F, D>>,
    basis: BasisMode,
    schedule: &Schedule,
    schedule_context: BatchedVerifierScheduleContext,
    verify_direct_commitments: DirectCommitmentCheck,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
    DirectCommitmentCheck: FnOnce(
        &[DirectWitnessProof<F>],
        &[RingCommitment<F, D>],
        &[usize],
        &[usize],
    ) -> Result<(), AkitaError>,
{
    let opening_points = opening_statement.opening_points();
    let commitments = opening_statement.commitments();
    let claims = opening_statement.claims();
    let point_group_sizes = opening_statement.point_group_sizes();
    let claim_group_sizes = opening_statement.claim_group_sizes();
    let claim_to_point = opening_statement.claim_to_point();

    match &proof.root {
        AkitaBatchedRootProof::Direct { witnesses } => {
            if !proof.steps.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            if !schedule_is_root_direct(schedule)
                || !matches!(schedule_context, BatchedVerifierScheduleContext::RootDirect)
            {
                return Err(AkitaError::InvalidProof);
            }
            verify_root_direct_openings(
                witnesses,
                opening_points,
                claims,
                &claim_group_sizes,
                &claim_to_point,
                basis,
            )?;
            verify_direct_commitments(
                witnesses,
                commitments,
                &point_group_sizes,
                &claim_group_sizes,
            )?;
        }
        AkitaBatchedRootProof::Fold(_) => {
            let BatchedVerifierScheduleContext::Fold(layouts) = schedule_context else {
                return Err(AkitaError::InvalidProof);
            };
            verify_fold_batched_proof::<F, T, D>(
                proof,
                setup,
                transcript,
                opening_points,
                claims,
                commitments,
                &point_group_sizes,
                &claim_group_sizes,
                &claim_to_point,
                basis,
                schedule,
                &layouts.root_lp,
                &layouts.next_level_params,
            )?;
        }
    }

    Ok(())
}

/// Verify a batched proof using caller-supplied config/policy callbacks.
///
/// This is the verifier crate's top-level orchestration entrypoint for the
/// current crate split. It owns public claim normalization, schedule-context
/// construction, root-direct and folded-root dispatch, and recursive verifier
/// replay. The root aggregate crate supplies only config-backed schedule/layout
/// selection and the root-direct commitment recomputation callback.
///
/// # Errors
///
/// Returns an error if public claims are malformed, schedule/layout policy
/// rejects the proof shape, root-direct commitment recomputation rejects, or
/// proof replay fails.
#[allow(clippy::too_many_arguments)]
pub fn verify_batched_with_policy<
    'a,
    F,
    T,
    const D: usize,
    SelectSchedule,
    RootLayout,
    NextParams,
    DirectParams,
    DirectCommitmentCheck,
>(
    proof: &AkitaBatchedProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    opening_statement: OpeningStatement<'a, F, RingCommitment<F, D>>,
    basis: BasisMode,
    select_schedule: SelectSchedule,
    root_layout: RootLayout,
    next_params: NextParams,
    direct_params: DirectParams,
    verify_direct_commitments: DirectCommitmentCheck,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
    SelectSchedule:
        FnOnce(usize, usize, usize, AkitaRootBatchSummary) -> Result<Schedule, AkitaError>,
    RootLayout: FnMut(AkitaScheduleInputs, &LevelParams) -> Result<LevelParams, AkitaError>,
    NextParams: FnMut(&Schedule, AkitaScheduleInputs) -> Result<LevelParams, AkitaError>,
    DirectParams: FnOnce(usize, usize) -> Result<LevelParams, AkitaError>,
    DirectCommitmentCheck: FnOnce(
        &[DirectWitnessProof<F>],
        &AkitaVerifierSetup<F>,
        &[RingCommitment<F, D>],
        &[usize],
        &[usize],
        &LevelParams,
    ) -> Result<(), AkitaError>,
{
    opening_statement.matches_setup(
        setup.expanded.seed.max_num_vars,
        setup.expanded.seed.max_num_batched_polys,
        setup.expanded.seed.max_num_points,
    )?;
    let num_vars = opening_statement.num_vars();
    let layout_num_claims = opening_statement.num_claims();
    let batch_summary = opening_statement
        .batch_summary()
        .map_err(|_| AkitaError::InvalidProof)?;

    let schedule = select_schedule(
        setup.expanded.seed.max_num_vars,
        num_vars,
        layout_num_claims,
        batch_summary,
    )
    .map_err(|_| AkitaError::InvalidProof)?;

    let mut next_params = next_params;
    let schedule_context = prepare_batched_verifier_schedule_context(
        setup.expanded.seed.max_num_vars,
        &schedule,
        root_layout,
        |next_inputs| next_params(&schedule, next_inputs),
    )
    .map_err(|_| AkitaError::InvalidProof)?;

    verify_batched_proof_with_schedule::<F, T, D, _>(
        proof,
        setup,
        transcript,
        opening_statement,
        basis,
        &schedule,
        schedule_context,
        |witnesses, commitments, point_group_sizes, claim_group_sizes| {
            let total_claims = checked_total_claims(claim_group_sizes, "root_direct_verify")
                .map_err(|_| AkitaError::InvalidProof)?;
            let params =
                direct_params(num_vars, total_claims).map_err(|_| AkitaError::InvalidProof)?;
            verify_direct_commitments(
                witnesses,
                setup,
                commitments,
                point_group_sizes,
                claim_group_sizes,
                &params,
            )
        },
    )
}
