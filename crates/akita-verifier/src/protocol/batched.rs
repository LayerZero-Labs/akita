//! Top-level batched verifier orchestration once a schedule is selected.

use crate::proof::claims::{prepare_verifier_claims, PreparedVerifierClaims};
use crate::proof::direct::verify_root_direct_openings_with_incidence;
use crate::protocol::levels::verify_fold_batched_proof;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, RandomSampling};
use akita_transcript::Transcript;
use akita_types::{
    schedule_is_root_direct, AkitaBatchedProof, AkitaBatchedRootProof, AkitaRootBatchSummary,
    AkitaScheduleInputs, AkitaVerifierSetup, BasisMode, ClaimIncidenceSummary, DirectWitnessProof,
    LevelParams, RingCommitment, Schedule, Step, VerifierClaims,
};

/// Config-derived layouts needed by the folded-root verifier branch.
pub(crate) struct FoldVerifierLayouts {
    /// Root verifier layout derived for the selected folded-root schedule.
    pub(crate) root_lp: LevelParams,
    /// First recursive-level params reached by the root fold.
    pub(crate) next_level_params: LevelParams,
}

/// Schedule context selected by the root scheme/config layer.
pub(crate) enum BatchedVerifierScheduleContext {
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
pub(crate) fn prepare_batched_verifier_schedule_context<RootLayout, NextParams>(
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
pub(crate) fn verify_batched_proof_with_schedule<
    'a,
    F,
    E,
    C,
    T,
    const D: usize,
    DirectCommitmentCheck,
>(
    proof: &AkitaBatchedProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    prepared_claims: PreparedVerifierClaims<'a, E, RingCommitment<F, D>>,
    basis: BasisMode,
    schedule: &Schedule,
    schedule_context: BatchedVerifierScheduleContext,
    verify_direct_commitments: DirectCommitmentCheck,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: ExtField<F>,
    C: ExtField<F>,
    T: Transcript<F>,
    DirectCommitmentCheck: FnOnce(
        &[DirectWitnessProof<F>],
        &[RingCommitment<F, D>],
        &ClaimIncidenceSummary,
    ) -> Result<(), AkitaError>,
{
    let PreparedVerifierClaims {
        opening_points,
        commitments,
        openings,
        incidence_summary,
        num_vars: _,
        layout_num_claims: _,
        batch_summary: _,
    } = prepared_claims;

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
            verify_root_direct_openings_with_incidence(
                witnesses,
                &opening_points,
                &openings,
                &incidence_summary,
                basis,
            )?;
            verify_direct_commitments(witnesses, &commitments, &incidence_summary)?;
        }
        AkitaBatchedRootProof::Fold(_) => {
            let BatchedVerifierScheduleContext::Fold(layouts) = schedule_context else {
                return Err(AkitaError::InvalidProof);
            };
            verify_fold_batched_proof::<F, E, C, T, D>(
                proof,
                setup,
                transcript,
                &opening_points,
                &openings,
                &commitments,
                &incidence_summary,
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
    E,
    C,
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
    claims: VerifierClaims<'a, E, RingCommitment<F, D>>,
    basis: BasisMode,
    select_schedule: SelectSchedule,
    root_layout: RootLayout,
    next_params: NextParams,
    direct_params: DirectParams,
    verify_direct_commitments: DirectCommitmentCheck,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: ExtField<F>,
    C: ExtField<F>,
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
        &ClaimIncidenceSummary,
        &LevelParams,
    ) -> Result<(), AkitaError>,
{
    let prepared_claims = prepare_verifier_claims(&setup.expanded, &claims)?;
    let num_vars = prepared_claims.num_vars;
    let layout_num_claims = prepared_claims.layout_num_claims;
    let batch_summary = prepared_claims.batch_summary;

    let max_num_vars = setup.expanded.seed.max_num_vars;
    let schedule = select_schedule(max_num_vars, num_vars, layout_num_claims, batch_summary)
        .map_err(|_| AkitaError::InvalidProof)?;

    let mut next_params = next_params;
    let schedule_context = prepare_batched_verifier_schedule_context(
        max_num_vars,
        &schedule,
        root_layout,
        |next_inputs| next_params(&schedule, next_inputs),
    )
    .map_err(|_| AkitaError::InvalidProof)?;

    verify_batched_proof_with_schedule::<F, E, C, T, D, _>(
        proof,
        setup,
        transcript,
        prepared_claims,
        basis,
        &schedule,
        schedule_context,
        |witnesses, commitments, incidence_summary| {
            let total_claims = incidence_summary.num_claims;
            let params =
                direct_params(num_vars, total_claims).map_err(|_| AkitaError::InvalidProof)?;
            verify_direct_commitments(witnesses, setup, commitments, incidence_summary, &params)
        },
    )
}
