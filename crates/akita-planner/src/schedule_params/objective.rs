use akita_field::AkitaError;

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
mod tests {
    use super::CompleteScheduleScore;

    #[test]
    fn direct_score_prefers_setup_only_after_proof_ties() {
        let smaller_proof = CompleteScheduleScore::Direct {
            proof_bytes: 99,
            setup_field_elements: 1_000,
            descriptor: vec![2],
        };
        let smaller_setup = CompleteScheduleScore::Direct {
            proof_bytes: 100,
            setup_field_elements: 1,
            descriptor: vec![1],
        };
        assert!(smaller_proof < smaller_setup);

        let same_proof_smaller_setup = CompleteScheduleScore::Direct {
            proof_bytes: 99,
            setup_field_elements: 999,
            descriptor: vec![3],
        };
        assert!(same_proof_smaller_setup < smaller_proof);

        let complete_tie_smaller_descriptor = CompleteScheduleScore::Direct {
            proof_bytes: 99,
            setup_field_elements: 999,
            descriptor: vec![1],
        };
        assert!(complete_tie_smaller_descriptor < same_proof_smaller_setup);
    }

    #[test]
    fn mixed_dimension_score_prefers_proof_only_after_setup_ties() {
        let smaller_setup = CompleteScheduleScore::MixedDimension {
            setup_field_elements: 99,
            proof_bytes: 1_000,
            descriptor: vec![2],
        };
        let smaller_proof = CompleteScheduleScore::MixedDimension {
            setup_field_elements: 100,
            proof_bytes: 1,
            descriptor: vec![1],
        };
        assert!(smaller_setup < smaller_proof);

        let same_setup_smaller_proof = CompleteScheduleScore::MixedDimension {
            setup_field_elements: 99,
            proof_bytes: 999,
            descriptor: vec![3],
        };
        assert!(same_setup_smaller_proof < smaller_setup);
    }

    #[test]
    fn recursive_score_uses_total_setup_only_after_primary_coordinates() {
        let smaller_proof = CompleteScheduleScore::RecursiveSetup {
            first_direct_setup_field_len: 10,
            proof_bytes: 99,
            setup_field_elements: 1_000,
            descriptor: vec![2],
        };
        let smaller_total_setup = CompleteScheduleScore::RecursiveSetup {
            first_direct_setup_field_len: 10,
            proof_bytes: 100,
            setup_field_elements: 1,
            descriptor: vec![1],
        };
        assert!(smaller_proof < smaller_total_setup);

        let same_proof_smaller_total_setup = CompleteScheduleScore::RecursiveSetup {
            first_direct_setup_field_len: 10,
            proof_bytes: 99,
            setup_field_elements: 999,
            descriptor: vec![3],
        };
        assert!(same_proof_smaller_total_setup < smaller_proof);
    }
}
