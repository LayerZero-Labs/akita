use akita_error::AkitaError;

use super::{candidate_schedule_descriptor_bytes, ScheduleCandidate};
use crate::{PlannerPolicy, SelectionPolicyId};

/// Complete-schedule ordering for one catalog-bound selection policy.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CompleteScheduleScore {
    Direct {
        proof_bytes: usize,
        setup_field_elements: usize,
        descriptor: Vec<u8>,
    },
    MixedDimension {
        setup_field_elements: usize,
        proof_bytes: usize,
        descriptor: Vec<u8>,
    },
    RecursiveSetup {
        first_direct_setup_field_len: usize,
        proof_bytes: usize,
        setup_field_elements: usize,
        descriptor: Vec<u8>,
    },
}

pub(crate) fn complete_schedule_score(
    policy: &PlannerPolicy,
    candidate: &ScheduleCandidate,
) -> Result<CompleteScheduleScore, AkitaError> {
    let descriptor = candidate_schedule_descriptor_bytes(candidate)?;
    match policy.selection_policy {
        SelectionPolicyId::MinEstimatedProofPayload => Ok(CompleteScheduleScore::Direct {
            proof_bytes: candidate.total_bytes,
            setup_field_elements: candidate.setup_field_elements,
            descriptor,
        }),
        SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload => {
            Ok(CompleteScheduleScore::MixedDimension {
                setup_field_elements: candidate.setup_field_elements,
                proof_bytes: candidate.total_bytes,
                descriptor,
            })
        }
        SelectionPolicyId::MinFirstDirectSetupThenPayload => {
            let first_direct_setup_field_len =
                candidate.first_direct_setup_field_len.ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "recursive setup candidate is missing its first direct setup size".into(),
                    )
                })?;
            Ok(CompleteScheduleScore::RecursiveSetup {
                first_direct_setup_field_len,
                proof_bytes: candidate.total_bytes,
                setup_field_elements: candidate.setup_field_elements,
                descriptor,
            })
        }
    }
}

pub(crate) fn select_complete_candidate<'a>(
    policy: &PlannerPolicy,
    candidates: impl IntoIterator<Item = &'a ScheduleCandidate>,
) -> Result<Option<&'a ScheduleCandidate>, AkitaError> {
    let mut best = None;
    for candidate in candidates {
        let score = complete_schedule_score(policy, candidate)?;
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score < *best_score)
        {
            best = Some((score, candidate));
        }
    }
    Ok(best.map(|(_, candidate)| candidate))
}

#[cfg(test)]
#[path = "../test/objective.rs"]
mod tests;
