use std::collections::BTreeMap;

use akita_field::AkitaError;

use crate::{schedule_params::stage3_payload_bytes_for_successor, PlannerPolicy};

use super::{child_choice, ParentPayloadKey, PendingScheduleCandidate, ScheduleCandidate};

#[derive(Clone, Copy)]
pub(super) enum FrontierProjection {
    Both,
    FirstDirectSetup,
    Payload,
}

impl FrontierProjection {
    const fn includes_first_direct_setup(self) -> bool {
        matches!(self, Self::Both | Self::FirstDirectSetup)
    }

    const fn includes_payload(self) -> bool {
        matches!(self, Self::Both | Self::Payload)
    }
}

pub(super) fn consider_child_suffixes<'a>(
    edge: &super::ChildEdge<'_>,
    child_candidates: impl IntoIterator<Item = &'a ScheduleCandidate>,
    incoming_setup_prefix: Option<usize>,
    projection: FrontierProjection,
    frontier: &mut ProjectedFrontier,
) -> Result<(), AkitaError> {
    for suffix in child_candidates {
        let Some(candidate) = child_choice(edge, suffix)? else {
            continue;
        };
        if incoming_setup_prefix.is_some_and(|natural_len| {
            candidate.suffix_folds.len() + 1 < 2
                || candidate.metrics().first_direct_setup_capacity
                    >= crate::schedule_params::SetupPrefixCapacity::for_natural_len(natural_len)
        }) {
            continue;
        }
        frontier.consider_pending(edge.policy, candidate, projection)?;
    }
    Ok(())
}

fn parent_visible_cost(
    policy: &PlannerPolicy,
    first: Option<&akita_types::CommittedGroupParams>,
) -> Result<ParentPayloadKey, AkitaError> {
    let Some(first) = first else {
        return Ok(ParentPayloadKey {
            outer_payload_bytes: 0,
            stage3_payload_bytes: 0,
        });
    };
    let outer_payload_bytes = first
        .outer_payload_geometry()?
        .transmitted_coefficients()
        .checked_mul(akita_types::layout::field_bytes(
            policy.decomposition.field_bits(),
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("first-fold payload size overflow".into()))?;
    Ok(ParentPayloadKey {
        outer_payload_bytes,
        stage3_payload_bytes: stage3_payload_bytes_for_successor(policy, Some(first))?,
    })
}

fn first_parent_visible_cost(
    policy: &PlannerPolicy,
    candidate: &ScheduleCandidate,
) -> Result<ParentPayloadKey, AkitaError> {
    parent_visible_cost(policy, candidate.first_fold_params())
}

#[derive(Clone, Default)]
pub(crate) struct ObjectiveChoices {
    pub(crate) setup: Option<ScheduleCandidate>,
    pub(crate) payload: Option<ScheduleCandidate>,
}

#[derive(Default)]
pub(super) struct ProjectedFrontier {
    pub(super) by_parent_cost: BTreeMap<ParentPayloadKey, ObjectiveChoices>,
}

impl ProjectedFrontier {
    pub(super) fn could_improve(
        &self,
        policy: &PlannerPolicy,
        parent_cost: ParentPayloadKey,
        metrics: super::super::CandidateMetrics,
        projection: FrontierProjection,
    ) -> bool {
        let choices = self.by_parent_cost.get(&parent_cost);
        let setup = policy.recursive_setup_planning
            && projection.includes_first_direct_setup()
            && choices
                .and_then(|choices| choices.setup.as_ref())
                .is_none_or(|best| {
                    let best = best.metrics();
                    (
                        metrics.first_direct_setup_capacity,
                        metrics.proof_bytes,
                        metrics.setup_field_elements,
                    ) <= (
                        best.first_direct_setup_capacity,
                        best.proof_bytes,
                        best.setup_field_elements,
                    )
                });
        let payload = projection.includes_payload()
            && choices
                .and_then(|choices| choices.payload.as_ref())
                .is_none_or(|best| {
                    let best = best.metrics();
                    (metrics.proof_bytes, metrics.setup_field_elements)
                        <= (best.proof_bytes, best.setup_field_elements)
                });
        setup || payload
    }

    fn consider(
        &mut self,
        policy: &PlannerPolicy,
        parent_cost: ParentPayloadKey,
        candidate: ScheduleCandidate,
        projection: FrontierProjection,
    ) -> Result<(), AkitaError> {
        let metrics = candidate.metrics();
        let choices = self.by_parent_cost.entry(parent_cost).or_default();
        if policy.recursive_setup_planning && projection.includes_first_direct_setup() {
            let score = (
                metrics.first_direct_setup_capacity,
                metrics.proof_bytes,
                metrics.setup_field_elements,
            );
            let improves = if let Some(best) = choices.setup.as_ref() {
                let best_metrics = best.metrics();
                let best_score = (
                    best_metrics.first_direct_setup_capacity,
                    best_metrics.proof_bytes,
                    best_metrics.setup_field_elements,
                );
                score < best_score
                    || (score == best_score
                        && super::super::candidate_suffix_descriptor_bytes(&candidate)
                            < super::super::candidate_suffix_descriptor_bytes(best))
            } else {
                true
            };
            if improves {
                choices.setup = Some(candidate.clone());
            }
        }
        if projection.includes_payload() {
            let score = (metrics.proof_bytes, metrics.setup_field_elements);
            let improves = if let Some(best) = choices.payload.as_ref() {
                let best_metrics = best.metrics();
                let best_score = (best_metrics.proof_bytes, best_metrics.setup_field_elements);
                score < best_score
                    || (score == best_score
                        && super::super::candidate_suffix_descriptor_bytes(&candidate)
                            < super::super::candidate_suffix_descriptor_bytes(best))
            } else {
                true
            };
            if improves {
                choices.payload = Some(candidate);
            }
        }
        Ok(())
    }

    pub(super) fn consider_candidate(
        &mut self,
        policy: &PlannerPolicy,
        candidate: ScheduleCandidate,
        projection: FrontierProjection,
    ) -> Result<(), AkitaError> {
        let parent_cost = first_parent_visible_cost(policy, &candidate)?;
        self.consider(policy, parent_cost, candidate, projection)
    }

    fn consider_pending(
        &mut self,
        policy: &PlannerPolicy,
        pending: PendingScheduleCandidate,
        projection: FrontierProjection,
    ) -> Result<(), AkitaError> {
        let outer_payload_bytes = pending
            .first_fold
            .params
            .outer_payload_geometry()?
            .transmitted_coefficients()
            .checked_mul(akita_types::layout::field_bytes(
                policy.decomposition.field_bits(),
            ))
            .ok_or_else(|| AkitaError::InvalidSetup("first-fold payload size overflow".into()))?;
        let parent_cost = ParentPayloadKey {
            outer_payload_bytes,
            stage3_payload_bytes: stage3_payload_bytes_for_successor(
                policy,
                Some(pending.first_fold.params.as_ref()),
            )?,
        };
        if self.could_improve(policy, parent_cost, pending.metrics(), projection) {
            self.consider(policy, parent_cost, pending.into_candidate(), projection)?;
        }
        Ok(())
    }
}
