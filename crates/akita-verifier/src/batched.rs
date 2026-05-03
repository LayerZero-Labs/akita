//! Top-level batched verifier orchestration once a schedule is selected.

use crate::{verify_fold_batched_proof, verify_root_direct_openings, PreparedVerifierClaims};
use akita_field::{CanonicalField, FieldCore, FieldSampling, HachiError};
use akita_transcript::Transcript;
use akita_types::{
    schedule_is_root_direct, BasisMode, DirectWitnessProof, HachiBatchedProof,
    HachiBatchedRootProof, HachiVerifierSetup, LevelParams, MultiPointBatchShape, RingCommitment,
    Schedule,
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
    proof: &HachiBatchedProof<F>,
    setup: &HachiVerifierSetup<F>,
    transcript: &mut T,
    prepared_claims: PreparedVerifierClaims<'a, F, RingCommitment<F, D>>,
    basis: BasisMode,
    schedule: &Schedule,
    schedule_context: BatchedVerifierScheduleContext,
    verify_direct_commitments: DirectCommitmentCheck,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    DirectCommitmentCheck: FnOnce(
        &[DirectWitnessProof<F>],
        &[RingCommitment<F, D>],
        &MultiPointBatchShape,
    ) -> Result<(), HachiError>,
{
    let PreparedVerifierClaims {
        opening_points,
        commitments,
        openings,
        batch_shape,
        num_vars: _,
        layout_num_claims: _,
        batch_summary: _,
    } = prepared_claims;

    match &proof.root {
        HachiBatchedRootProof::Direct { witnesses } => {
            if !proof.steps.is_empty() {
                return Err(HachiError::InvalidProof);
            }
            if !schedule_is_root_direct(schedule)
                || !matches!(schedule_context, BatchedVerifierScheduleContext::RootDirect)
            {
                return Err(HachiError::InvalidProof);
            }
            verify_root_direct_openings(
                witnesses,
                &opening_points,
                &openings,
                &batch_shape,
                basis,
            )?;
            verify_direct_commitments(witnesses, &commitments, &batch_shape)?;
        }
        HachiBatchedRootProof::Fold(_) => {
            let BatchedVerifierScheduleContext::Fold(layouts) = schedule_context else {
                return Err(HachiError::InvalidProof);
            };
            verify_fold_batched_proof::<F, T, D>(
                proof,
                setup,
                transcript,
                &opening_points,
                &openings,
                &commitments,
                &batch_shape,
                basis,
                schedule,
                &layouts.root_lp,
                &layouts.next_level_params,
            )?;
        }
    }

    Ok(())
}
