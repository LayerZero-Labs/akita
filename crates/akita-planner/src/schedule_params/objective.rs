use akita_field::AkitaError;

use super::{candidate_schedule_descriptor_bytes, CandidateMetrics, ScheduleCandidate};
use crate::{PlannerPolicy, SelectionPolicyId};

/// Complete-schedule ordering: numeric policy coordinates followed by the
/// canonical descriptor tie-break.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct CompleteScheduleScore {
    objective: CompleteObjectiveBound,
    descriptor: Vec<u8>,
}

/// Numeric prefix of a complete-schedule objective. These coordinates omit
/// the canonical descriptor, so a bound may prune only when it is strictly
/// worse than an already completed candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CompleteObjectiveBound {
    Direct {
        proof_bytes: usize,
        setup_field_elements: usize,
    },
    MixedDimension {
        setup_field_elements: usize,
        proof_bytes: usize,
    },
    RecursiveSetup {
        first_direct_setup_capacity: usize,
        proof_bytes: usize,
        setup_field_elements: usize,
    },
}

impl CompleteObjectiveBound {
    pub(crate) fn for_direct_edge(
        policy: &PlannerPolicy,
        first_direct_setup_capacity: usize,
        proof_bytes: usize,
        setup_field_elements: usize,
    ) -> Self {
        match policy.selection_policy {
            SelectionPolicyId::MinEstimatedProofPayload => Self::Direct {
                proof_bytes,
                setup_field_elements,
            },
            SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload => {
                Self::MixedDimension {
                    setup_field_elements,
                    proof_bytes,
                }
            }
            SelectionPolicyId::MinFirstDirectSetupThenPayload => Self::RecursiveSetup {
                first_direct_setup_capacity,
                proof_bytes,
                setup_field_elements,
            },
        }
    }

    fn for_candidate(policy: &PlannerPolicy, metrics: CandidateMetrics) -> Self {
        Self::for_direct_edge(
            policy,
            metrics.first_direct_setup_capacity.field_elements(),
            metrics.proof_bytes,
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
                    > (incumbent.proof_bytes, incumbent.setup_field_elements)
            }
            Self::MixedDimension {
                setup_field_elements,
                proof_bytes,
            } => {
                (setup_field_elements, proof_bytes)
                    > (incumbent.setup_field_elements, incumbent.proof_bytes)
            }
            Self::RecursiveSetup {
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
                    incumbent.proof_bytes,
                    incumbent.setup_field_elements,
                )
            }
        }
    }

    /// Compare the coordinates that remain ordered after a parent can mask
    /// the child's setup envelope. The total-setup coordinate is deliberately
    /// excluded; if capacity and proof tie, the parent may make setup tie too,
    /// leaving the canonical descriptor decisive.
    pub(crate) fn is_strictly_worse_for_recursive_parent(
        self,
        incumbent: CandidateMetrics,
    ) -> bool {
        match self {
            Self::RecursiveSetup {
                first_direct_setup_capacity,
                proof_bytes,
                ..
            } => {
                (first_direct_setup_capacity, proof_bytes)
                    > (
                        incumbent.first_direct_setup_capacity.field_elements(),
                        incumbent.proof_bytes,
                    )
            }
            Self::Direct { .. } | Self::MixedDimension { .. } => false,
        }
    }
}

pub(crate) fn complete_schedule_score(
    policy: &PlannerPolicy,
    candidate: &ScheduleCandidate,
    diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
) -> Result<CompleteScheduleScore, AkitaError> {
    let descriptor = candidate_schedule_descriptor_bytes(candidate, diagnostics)?;
    let metrics = candidate.metrics();
    if policy.selection_policy == SelectionPolicyId::MinFirstDirectSetupThenPayload
        && candidate.first_direct_setup_field_len.is_none()
    {
        return Err(AkitaError::InvalidSetup(
            "recursive setup candidate is missing its first direct setup size".into(),
        ));
    }
    Ok(CompleteScheduleScore {
        objective: CompleteObjectiveBound::for_candidate(policy, metrics),
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
