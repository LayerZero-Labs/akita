use akita_error::AkitaError;

use super::{candidate_schedule_descriptor_bytes, CandidateMetrics, ScheduleCandidate};
use crate::{PlannerPolicy, SelectionPolicyId};

/// Complete-schedule ordering: numeric policy coordinates, an optional legacy
/// root output-witness tie-break, then the canonical descriptor.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct CompleteScheduleScore {
    objective: CompleteObjectiveBound,
    legacy_root_output_witness_len: Option<usize>,
    descriptor: Vec<u8>,
}

/// Numeric prefix of a complete-schedule objective. These coordinates omit
/// complete-schedule tie-breaks, so a bound may prune only when it is strictly
/// worse than an already completed candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CompleteObjectiveBound {
    Direct {
        proof_bytes: usize,
        setup_field_elements: usize,
    },
    SetupFirst {
        first_direct_setup_capacity: usize,
        proof_bytes: usize,
        setup_field_elements: usize,
    },
    PaddedSetupEnvelopeFirst {
        setup_envelope_capacity: usize,
        first_direct_setup_capacity: usize,
        proof_bytes: usize,
        first_direct_output_witness_len: usize,
    },
}

impl CompleteObjectiveBound {
    pub(crate) fn for_direct_edge(
        policy: &PlannerPolicy,
        first_direct_setup_capacity: usize,
        first_direct_output_witness_len: usize,
        proof_bytes: usize,
        setup_field_elements: usize,
    ) -> Self {
        match policy.selection_policy {
            SelectionPolicyId::MinEstimatedProofPayloadV2 => Self::Direct {
                proof_bytes,
                setup_field_elements,
            },
            SelectionPolicyId::MinFirstDirectSetupThenPayloadV2 => Self::SetupFirst {
                first_direct_setup_capacity,
                proof_bytes,
                setup_field_elements,
            },
            SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3 => {
                Self::PaddedSetupEnvelopeFirst {
                    setup_envelope_capacity: akita_types::padded_setup_prefix_len(
                        setup_field_elements,
                    ),
                    first_direct_setup_capacity,
                    proof_bytes,
                    first_direct_output_witness_len,
                }
            }
        }
    }

    fn for_candidate(policy: &PlannerPolicy, metrics: CandidateMetrics) -> Self {
        Self::for_direct_edge(
            policy,
            metrics.first_direct_setup_capacity.field_elements(),
            metrics.first_direct_output_witness_len,
            metrics.proof_bytes(),
            metrics.setup_field_elements,
        )
    }

    pub(crate) fn is_strictly_worse_than(self, incumbent: CandidateMetrics) -> bool {
        match self {
            Self::Direct {
                proof_bytes,
                setup_field_elements,
            } => {
                (proof_bytes, setup_field_elements)
                    > (incumbent.proof_bytes(), incumbent.setup_field_elements)
            }
            Self::SetupFirst {
                first_direct_setup_capacity,
                proof_bytes,
                setup_field_elements,
            } => {
                (
                    first_direct_setup_capacity,
                    proof_bytes,
                    setup_field_elements,
                ) > (
                    incumbent.first_direct_setup_capacity.field_elements(),
                    incumbent.proof_bytes(),
                    incumbent.setup_field_elements,
                )
            }
            Self::PaddedSetupEnvelopeFirst {
                setup_envelope_capacity,
                first_direct_setup_capacity,
                proof_bytes,
                first_direct_output_witness_len,
            } => {
                (
                    setup_envelope_capacity,
                    first_direct_setup_capacity,
                    proof_bytes,
                    first_direct_output_witness_len,
                ) > (
                    akita_types::padded_setup_prefix_len(incumbent.setup_field_elements),
                    incumbent.first_direct_setup_capacity.field_elements(),
                    incumbent.proof_bytes(),
                    incumbent.first_direct_output_witness_len,
                )
            }
        }
    }

    pub(crate) fn setup_envelope_is_strictly_worse_than(self, incumbent: CandidateMetrics) -> bool {
        match self {
            Self::PaddedSetupEnvelopeFirst {
                setup_envelope_capacity,
                ..
            } => {
                setup_envelope_capacity
                    > akita_types::padded_setup_prefix_len(incumbent.setup_field_elements)
            }
            Self::Direct { .. } | Self::SetupFirst { .. } => false,
        }
    }
}

pub(crate) fn complete_schedule_score(
    policy: &PlannerPolicy,
    candidate: &ScheduleCandidate,
    diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
) -> Result<CompleteScheduleScore, AkitaError> {
    let root_output_witness_len = candidate
        .folds
        .first()
        .ok_or_else(|| {
            AkitaError::InvalidSetup("complete schedule is missing its root fold".into())
        })?
        .output_witness_len;
    let descriptor = candidate_schedule_descriptor_bytes(
        None,
        &candidate.folds,
        &candidate.terminal.params,
        diagnostics,
    )?;
    let metrics = candidate.metrics();
    if matches!(
        policy.selection_policy,
        SelectionPolicyId::MinFirstDirectSetupThenPayloadV2
            | SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
    ) && candidate.first_direct_setup_field_len.is_none()
    {
        return Err(AkitaError::InvalidSetup(
            "setup-first candidate is missing its first direct setup size".into(),
        ));
    }
    Ok(CompleteScheduleScore {
        objective: CompleteObjectiveBound::for_candidate(policy, metrics),
        legacy_root_output_witness_len: (policy.selection_policy
            != SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3)
            .then_some(root_output_witness_len),
        descriptor,
    })
}

pub(crate) fn select_complete_candidate<'a>(
    policy: &PlannerPolicy,
    candidates: impl IntoIterator<Item = &'a ScheduleCandidate>,
    diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
) -> Result<Option<&'a ScheduleCandidate>, AkitaError> {
    let mut best = None;
    for candidate in candidates {
        let score = complete_schedule_score(policy, candidate, diagnostics)?;
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
