use std::collections::BTreeMap;

use akita_field::AkitaError;

use crate::PlannerPolicy;

use super::{child_choice, FirstFoldKey, PendingScheduleCandidate, ScheduleCandidate};

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
            candidate.suffix_folds.is_empty()
                || candidate.metrics().first_direct_setup_capacity
                    >= crate::schedule_params::SetupPrefixCapacity::for_natural_len(natural_len)
        }) {
            continue;
        }
        frontier.consider_pending(edge.policy, candidate, projection)?;
    }
    Ok(())
}

fn parent_visible_cost(first: Option<&akita_types::CommittedGroupParams>) -> FirstFoldKey {
    FirstFoldKey {
        descriptor: first.map(akita_types::CommittedGroupParams::canonical_descriptor_bytes),
    }
}

fn first_parent_visible_cost(candidate: &ScheduleCandidate) -> FirstFoldKey {
    parent_visible_cost(candidate.first_fold_params())
}

fn setup_score(
    metrics: super::super::CandidateMetrics,
) -> (crate::schedule_params::SetupPrefixCapacity, usize, usize) {
    (
        metrics.first_direct_setup_capacity,
        metrics.proof_bytes,
        metrics.setup_field_elements,
    )
}

fn payload_score(metrics: super::super::CandidateMetrics) -> (usize, usize) {
    (metrics.proof_bytes, metrics.setup_field_elements)
}

#[derive(Clone, Default)]
pub(crate) struct ObjectiveChoices {
    pub(crate) setup: Option<ScheduleCandidate>,
    pub(crate) payload: Option<ScheduleCandidate>,
}

#[derive(Default)]
pub(super) struct ProjectedFrontier {
    pub(super) by_parent_cost: BTreeMap<FirstFoldKey, ObjectiveChoices>,
}

impl ProjectedFrontier {
    pub(super) fn could_improve(
        &self,
        policy: &PlannerPolicy,
        parent_cost: &FirstFoldKey,
        metrics: super::super::CandidateMetrics,
        projection: FrontierProjection,
    ) -> bool {
        let choices = self.by_parent_cost.get(parent_cost);
        let setup = policy.recursive_setup_planning
            && projection.includes_first_direct_setup()
            && choices
                .and_then(|choices| choices.setup.as_ref())
                .is_none_or(|best| {
                    let best = best.metrics();
                    setup_score(metrics) <= setup_score(best)
                });
        let payload = projection.includes_payload()
            && choices
                .and_then(|choices| choices.payload.as_ref())
                .is_none_or(|best| {
                    let best = best.metrics();
                    payload_score(metrics) <= payload_score(best)
                });
        setup || payload
    }

    fn consider(
        &mut self,
        policy: &PlannerPolicy,
        parent_cost: FirstFoldKey,
        candidate: ScheduleCandidate,
        projection: FrontierProjection,
    ) -> Result<(), AkitaError> {
        let metrics = candidate.metrics();
        let choices = self.by_parent_cost.entry(parent_cost).or_default();
        if policy.recursive_setup_planning && projection.includes_first_direct_setup() {
            let score = setup_score(metrics);
            let improves = if let Some(best) = choices.setup.as_ref() {
                let best_metrics = best.metrics();
                let best_score = setup_score(best_metrics);
                score < best_score
                    || (score == best_score
                        && super::super::candidate_schedule_descriptor_bytes(&candidate)?
                            < super::super::candidate_schedule_descriptor_bytes(best)?)
            } else {
                true
            };
            if improves {
                choices.setup = Some(candidate.clone());
            }
        }
        if projection.includes_payload() {
            let score = payload_score(metrics);
            let improves = if let Some(best) = choices.payload.as_ref() {
                let best_metrics = best.metrics();
                let best_score = payload_score(best_metrics);
                score < best_score
                    || (score == best_score
                        && super::super::candidate_schedule_descriptor_bytes(&candidate)?
                            < super::super::candidate_schedule_descriptor_bytes(best)?)
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
        let parent_cost = first_parent_visible_cost(&candidate);
        self.consider(policy, parent_cost, candidate, projection)
    }

    fn consider_pending(
        &mut self,
        policy: &PlannerPolicy,
        pending: PendingScheduleCandidate,
        projection: FrontierProjection,
    ) -> Result<(), AkitaError> {
        let parent_cost = FirstFoldKey {
            descriptor: Some(pending.first_fold.params.canonical_descriptor_bytes()),
        };
        if self.could_improve(policy, &parent_cost, pending.metrics(), projection) {
            self.consider(policy, parent_cost, pending.into_candidate(), projection)?;
        }
        Ok(())
    }
}
