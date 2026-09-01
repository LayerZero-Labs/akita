use std::{collections::BTreeMap, sync::Arc};

use akita_error::AkitaError;

use crate::{schedule_params::CompleteObjectiveBound, PlannerPolicy};

use super::{
    child_choice, child_edge_price, ParentObservableKey, PendingScheduleCandidate,
    ScheduleCandidate,
};

#[derive(Clone, Copy)]
pub(super) enum Projection {
    FirstDirectSetup,
    Payload,
}

impl Projection {
    const ALL: [Self; 2] = [Self::FirstDirectSetup, Self::Payload];
}

pub(super) fn consider_child_suffixes<'a>(
    edge: &super::ChildEdge<'_>,
    successor_class: &ParentObservableKey,
    child_candidates: impl IntoIterator<Item = &'a ScheduleCandidate>,
    incoming_setup_prefix: Option<usize>,
    projections: &[Projection],
    frontier: &mut ProjectedFrontier,
) -> Result<(), AkitaError> {
    let mut child_candidates = child_candidates.into_iter();
    let Some(first) = child_candidates.next() else {
        return Ok(());
    };
    if first_parent_visible_cost(edge.policy, first)? != *successor_class {
        return Err(AkitaError::InvalidSetup(
            "suffix frontier candidate disagrees with its parent-observable class".into(),
        ));
    }
    // `SuffixResult` partitions candidates by every successor coordinate a
    // parent can observe. Price the edge and grinding plan once for that class;
    // rebuilding them for descriptor-distinct members is redundant.
    let edge_price = child_edge_price(edge, first)?;
    let edge_nonce_bits = edge.grinding_nonce_bits(first)?;
    let parent_cost = ParentObservableKey::new(edge.policy, Some(&edge.candidate_params), None)?;
    for suffix in std::iter::once(first).chain(child_candidates) {
        let Some(candidate) = child_choice(edge, edge_price, edge_nonce_bits, suffix)? else {
            continue;
        };
        if incoming_setup_prefix.is_some_and(|natural_len| {
            candidate.suffix_folds.is_empty()
                || candidate.metrics().first_direct_setup_capacity
                    >= crate::schedule_params::SetupPrefixCapacity::for_natural_len(natural_len)
        }) {
            continue;
        }
        frontier.consider_pending(
            edge.policy,
            edge.diagnostics,
            &parent_cost,
            candidate,
            projections,
        )?;
    }
    Ok(())
}

fn parent_visible_cost(
    policy: &PlannerPolicy,
    first: Option<&akita_types::CommittedGroupParams>,
    terminal: Option<&akita_types::TerminalFoldParams>,
) -> Result<ParentObservableKey, AkitaError> {
    ParentObservableKey::new(policy, first, terminal)
}

fn first_parent_visible_cost(
    policy: &PlannerPolicy,
    candidate: &ScheduleCandidate,
) -> Result<ParentObservableKey, AkitaError> {
    parent_visible_cost(
        policy,
        candidate.first_fold_params(),
        candidate
            .folds
            .is_empty()
            .then_some(&candidate.terminal.params),
    )
}

#[derive(Clone, Copy)]
struct SetupScore {
    first_direct_setup_capacity: crate::schedule_params::SetupPrefixCapacity,
    cost: crate::schedule_params::PackedProofCost,
    setup_field_elements: usize,
}

#[derive(Clone, Copy)]
struct PayloadScore {
    cost: crate::schedule_params::PackedProofCost,
    setup_field_elements: usize,
}

fn setup_score(metrics: super::super::CandidateMetrics) -> SetupScore {
    SetupScore {
        first_direct_setup_capacity: metrics.first_direct_setup_capacity,
        cost: metrics.cost,
        setup_field_elements: metrics.setup_field_elements,
    }
}

fn payload_score(metrics: super::super::CandidateMetrics) -> PayloadScore {
    PayloadScore {
        cost: metrics.cost,
        setup_field_elements: metrics.setup_field_elements,
    }
}

#[derive(Clone)]
struct ProjectedCandidate {
    descriptor: Arc<[u8]>,
    descriptor_context: DescriptorOrderContext,
    admission: ParentAdmissionClass,
    schedule: ScheduleCandidate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DescriptorOrderContext {
    fold_count: usize,
    first_fold_descriptor: Option<Arc<[u8]>>,
}

impl DescriptorOrderContext {
    fn for_candidate(candidate: &ScheduleCandidate) -> Self {
        Self {
            fold_count: candidate.folds.len(),
            first_fold_descriptor: candidate
                .first_fold_params()
                .map(akita_types::CommittedGroupParams::canonical_descriptor_bytes)
                .map(Arc::from),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ParentAdmissionClass {
    fold_depth: u8,
    first_direct_setup_capacity: crate::schedule_params::SetupPrefixCapacity,
}

impl ParentAdmissionClass {
    pub(super) fn for_candidate(candidate: &ScheduleCandidate) -> Self {
        Self {
            fold_depth: candidate.folds.len().min(2) as u8,
            first_direct_setup_capacity: candidate.metrics().first_direct_setup_capacity,
        }
    }

    fn admits_every_parent_of(self, other: Self) -> bool {
        self.fold_depth >= other.fold_depth
            && self.first_direct_setup_capacity <= other.first_direct_setup_capacity
    }

    pub(super) fn is_admitted_by(
        self,
        require_child_fold: bool,
        offloaded: bool,
        natural_setup_field_len: usize,
    ) -> bool {
        (!require_child_fold || self.fold_depth >= 1)
            && (!offloaded
                || (self.fold_depth >= 2
                    && self.first_direct_setup_capacity
                        < crate::schedule_params::SetupPrefixCapacity::for_natural_len(
                            natural_setup_field_len,
                        )))
    }
}

#[derive(Clone, Default)]
pub(crate) struct ProjectedObjectiveChoices {
    setup: Vec<ProjectedCandidate>,
    payload: Vec<ProjectedCandidate>,
}

/// Completed frontier choices stored in the exact suffix memo.
///
/// Parents consume only the schedules. Dropping the cached descriptors and
/// descriptor contexts here keeps the exact suffix memo compact.
pub(super) struct ObjectiveChoices {
    setup: Vec<ScheduleCandidate>,
    payload: Vec<ScheduleCandidate>,
}

impl ObjectiveChoices {
    pub(super) fn setup_candidates(&self) -> impl Iterator<Item = &ScheduleCandidate> {
        self.setup.iter()
    }

    pub(super) fn payload_candidates(&self) -> impl Iterator<Item = &ScheduleCandidate> {
        self.payload.iter()
    }
}

impl ProjectedObjectiveChoices {
    pub(super) fn candidate_count(&self) -> usize {
        self.setup.len().saturating_add(self.payload.len())
    }

    pub(super) fn setup_candidates(&self) -> impl Iterator<Item = &ScheduleCandidate> {
        self.setup.iter().map(|candidate| &candidate.schedule)
    }

    pub(super) fn payload_candidates(&self) -> impl Iterator<Item = &ScheduleCandidate> {
        self.payload.iter().map(|candidate| &candidate.schedule)
    }

    pub(super) fn into_payload_candidates(self) -> Vec<ScheduleCandidate> {
        self.payload
            .into_iter()
            .map(|candidate| candidate.schedule)
            .collect()
    }

    pub(super) fn into_objective_choices(self) -> ObjectiveChoices {
        ObjectiveChoices {
            setup: self
                .setup
                .into_iter()
                .map(|candidate| candidate.schedule)
                .collect(),
            payload: self
                .payload
                .into_iter()
                .map(|candidate| candidate.schedule)
                .collect(),
        }
    }

    fn projected(&self, projection: Projection) -> &[ProjectedCandidate] {
        match projection {
            Projection::FirstDirectSetup => &self.setup,
            Projection::Payload => &self.payload,
        }
    }

    fn projected_mut(&mut self, projection: Projection) -> &mut Vec<ProjectedCandidate> {
        match projection {
            Projection::FirstDirectSetup => &mut self.setup,
            Projection::Payload => &mut self.payload,
        }
    }
}

#[derive(Default)]
pub(super) struct ProjectedFrontier {
    pub(super) by_parent_cost: BTreeMap<ParentObservableKey, ProjectedObjectiveChoices>,
}

impl ProjectedFrontier {
    pub(super) fn candidate_count(&self) -> usize {
        self.by_parent_cost
            .values()
            .map(ProjectedObjectiveChoices::candidate_count)
            .sum()
    }

    pub(super) fn recursive_direct_bound_is_strictly_worse(
        &self,
        parent_cost: &ParentObservableKey,
        first_direct_setup_capacity: crate::schedule_params::SetupPrefixCapacity,
        lower_bound: CompleteObjectiveBound,
    ) -> bool {
        let candidate_admission = ParentAdmissionClass {
            fold_depth: 2,
            first_direct_setup_capacity,
        };
        let Some(choices) = self.by_parent_cost.get(parent_cost) else {
            return false;
        };
        recursive_direct_bound_is_dominated(candidate_admission, lower_bound, choices)
    }

    fn consider(
        &mut self,
        policy: &PlannerPolicy,
        diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
        parent_cost: ParentObservableKey,
        candidate: ScheduleCandidate,
        projections: &[Projection],
    ) -> Result<(), AkitaError> {
        let admission = ParentAdmissionClass::for_candidate(&candidate);
        let metrics = candidate.metrics();
        let choices = self.by_parent_cost.get(&parent_cost);
        let keep = |projection| match projection {
            Projection::FirstDirectSetup => {
                policy.selection_policy == crate::SelectionPolicyId::MinFirstDirectSetupThenPayload
                    && !choices.is_some_and(|choices| {
                        choices.projected(projection).iter().any(|existing| {
                            setup_primary_strictly_dominates(
                                setup_score(existing.schedule.metrics()),
                                existing.admission,
                                setup_score(metrics),
                                admission,
                            )
                        })
                    })
            }
            Projection::Payload => !choices.is_some_and(|choices| {
                choices.projected(projection).iter().any(|existing| {
                    payload_primary_strictly_dominates(
                        payload_score(existing.schedule.metrics()),
                        existing.admission,
                        payload_score(metrics),
                        admission,
                    )
                })
            }),
        };
        let retained_projections = projections
            .iter()
            .copied()
            .filter(|&projection| keep(projection))
            .collect::<Vec<_>>();
        if retained_projections.is_empty() {
            return Ok(());
        }
        let projected = ProjectedCandidate {
            descriptor: super::super::candidate_schedule_descriptor_bytes(&candidate, diagnostics)?
                .into(),
            descriptor_context: DescriptorOrderContext::for_candidate(&candidate),
            admission,
            schedule: candidate,
        };
        let choices = self.by_parent_cost.entry(parent_cost).or_default();
        for projection in retained_projections {
            let dominates = match projection {
                Projection::FirstDirectSetup => setup_dominates,
                Projection::Payload => payload_dominates,
            };
            insert_projected(
                choices.projected_mut(projection),
                projected.clone(),
                dominates,
            );
        }
        Ok(())
    }

    pub(super) fn consider_candidate(
        &mut self,
        policy: &PlannerPolicy,
        diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
        candidate: ScheduleCandidate,
        projections: &[Projection],
    ) -> Result<(), AkitaError> {
        let parent_cost = first_parent_visible_cost(policy, &candidate)?;
        self.consider(policy, diagnostics, parent_cost, candidate, projections)
    }

    fn consider_pending(
        &mut self,
        policy: &PlannerPolicy,
        diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
        parent_cost: &ParentObservableKey,
        pending: PendingScheduleCandidate,
        projections: &[Projection],
    ) -> Result<(), AkitaError> {
        self.consider(
            policy,
            diagnostics,
            parent_cost.clone(),
            pending.into_candidate(),
            projections,
        )
    }
}

fn recursive_direct_bound_is_dominated(
    candidate_admission: ParentAdmissionClass,
    lower_bound: CompleteObjectiveBound,
    incumbents: &ProjectedObjectiveChoices,
) -> bool {
    Projection::ALL.into_iter().all(|projection| {
        projection_bound_is_dominated(
            projection,
            candidate_admission,
            lower_bound,
            incumbents
                .projected(projection)
                .iter()
                .map(|candidate| (candidate.admission, candidate.schedule.metrics())),
        )
    })
}

fn projection_bound_is_dominated(
    projection: Projection,
    candidate_admission: ParentAdmissionClass,
    lower_bound: CompleteObjectiveBound,
    incumbents: impl IntoIterator<Item = (ParentAdmissionClass, super::super::CandidateMetrics)>,
) -> bool {
    incumbents.into_iter().any(|(admission, metrics)| {
        admission.admits_every_parent_of(candidate_admission)
            && match projection {
                Projection::FirstDirectSetup => {
                    lower_bound.is_strictly_worse_for_recursive_parent(metrics)
                }
                Projection::Payload => lower_bound.is_strictly_worse_for_recursive_payload(metrics),
            }
    })
}

fn setup_dominates(left: &ProjectedCandidate, right: &ProjectedCandidate) -> bool {
    setup_projection_dominates(
        ProjectionOrder {
            score: setup_score(left.schedule.metrics()),
            descriptor: left.descriptor.as_ref(),
            context: &left.descriptor_context,
            admission: left.admission,
        },
        ProjectionOrder {
            score: setup_score(right.schedule.metrics()),
            descriptor: right.descriptor.as_ref(),
            context: &right.descriptor_context,
            admission: right.admission,
        },
    )
}

fn setup_primary_strictly_dominates(
    left_score: SetupScore,
    left_admission: ParentAdmissionClass,
    right_score: SetupScore,
    right_admission: ParentAdmissionClass,
) -> bool {
    left_admission.admits_every_parent_of(right_admission)
        && (left_score.first_direct_setup_capacity < right_score.first_direct_setup_capacity
            || (left_score.first_direct_setup_capacity == right_score.first_direct_setup_capacity
                && left_score
                    .cost
                    .strictly_better_for_every_parent(right_score.cost)))
}

#[derive(Clone, Copy)]
struct ProjectionOrder<'a, Score> {
    score: Score,
    descriptor: &'a [u8],
    context: &'a DescriptorOrderContext,
    admission: ParentAdmissionClass,
}

fn setup_projection_dominates(
    left: ProjectionOrder<'_, SetupScore>,
    right: ProjectionOrder<'_, SetupScore>,
) -> bool {
    left.admission.admits_every_parent_of(right.admission)
        && (left.score.first_direct_setup_capacity < right.score.first_direct_setup_capacity
            || (left.score.first_direct_setup_capacity == right.score.first_direct_setup_capacity
                && (left
                    .score
                    .cost
                    .strictly_better_for_every_parent(right.score.cost)
                    || (left
                        .score
                        .cost
                        .never_worse_for_every_parent(right.score.cost)
                        && left.score.setup_field_elements <= right.score.setup_field_elements
                        && left.context == right.context
                        && left.descriptor <= right.descriptor))))
}

fn payload_dominates(left: &ProjectedCandidate, right: &ProjectedCandidate) -> bool {
    payload_projection_dominates(
        ProjectionOrder {
            score: payload_score(left.schedule.metrics()),
            descriptor: left.descriptor.as_ref(),
            context: &left.descriptor_context,
            admission: left.admission,
        },
        ProjectionOrder {
            score: payload_score(right.schedule.metrics()),
            descriptor: right.descriptor.as_ref(),
            context: &right.descriptor_context,
            admission: right.admission,
        },
    )
}

fn payload_primary_strictly_dominates(
    left_score: PayloadScore,
    left_admission: ParentAdmissionClass,
    right_score: PayloadScore,
    right_admission: ParentAdmissionClass,
) -> bool {
    left_admission.admits_every_parent_of(right_admission)
        && left_score
            .cost
            .strictly_better_for_every_parent(right_score.cost)
}

fn payload_projection_dominates(
    left: ProjectionOrder<'_, PayloadScore>,
    right: ProjectionOrder<'_, PayloadScore>,
) -> bool {
    left.admission.admits_every_parent_of(right.admission)
        && (left
            .score
            .cost
            .strictly_better_for_every_parent(right.score.cost)
            || (left
                .score
                .cost
                .never_worse_for_every_parent(right.score.cost)
                && left.score.setup_field_elements <= right.score.setup_field_elements
                && left.context == right.context
                && left.descriptor <= right.descriptor))
}

fn insert_projected(
    frontier: &mut Vec<ProjectedCandidate>,
    candidate: ProjectedCandidate,
    dominates: fn(&ProjectedCandidate, &ProjectedCandidate) -> bool,
) {
    if frontier
        .iter()
        .any(|existing| dominates(existing, &candidate))
    {
        return;
    }
    frontier.retain(|existing| !dominates(&candidate, existing));
    frontier.push(candidate);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        payload_primary_strictly_dominates, payload_projection_dominates,
        projection_bound_is_dominated, setup_primary_strictly_dominates,
        setup_projection_dominates, DescriptorOrderContext, ParentAdmissionClass, PayloadScore,
        Projection, ProjectionOrder, SetupScore,
    };
    use crate::schedule_params::{
        CandidateMetrics, CompleteObjectiveBound, PackedProofCost, SetupPrefixCapacity,
    };

    fn context(fold_count: usize, first_fold: u8) -> DescriptorOrderContext {
        DescriptorOrderContext {
            fold_count,
            first_fold_descriptor: (fold_count != 0).then(|| Arc::from([first_fold])),
        }
    }

    fn admission(fold_depth: u8, natural_len: usize) -> ParentAdmissionClass {
        ParentAdmissionClass {
            fold_depth,
            first_direct_setup_capacity: SetupPrefixCapacity::for_natural_len(natural_len),
        }
    }

    fn order<'a, Score>(
        score: Score,
        descriptor: &'a [u8],
        context: &'a DescriptorOrderContext,
        admission: ParentAdmissionClass,
    ) -> ProjectionOrder<'a, Score> {
        ProjectionOrder {
            score,
            descriptor,
            context,
            admission,
        }
    }

    fn setup_score(
        capacity: SetupPrefixCapacity,
        payload_bytes: usize,
        nonce_bits: usize,
        setup_field_elements: usize,
    ) -> SetupScore {
        SetupScore {
            first_direct_setup_capacity: capacity,
            cost: PackedProofCost::new(payload_bytes, nonce_bits).unwrap(),
            setup_field_elements,
        }
    }

    fn payload_score(
        payload_bytes: usize,
        nonce_bits: usize,
        setup_field_elements: usize,
    ) -> PayloadScore {
        PayloadScore {
            cost: PackedProofCost::new(payload_bytes, nonce_bits).unwrap(),
            setup_field_elements,
        }
    }

    #[test]
    fn setup_projection_keeps_setup_descriptor_tradeoffs_that_a_parent_can_mask() {
        let smaller_setup = setup_score(SetupPrefixCapacity::for_natural_len(8), 100, 0, 64);
        let smaller_descriptor = setup_score(SetupPrefixCapacity::for_natural_len(8), 100, 0, 128);
        assert!(!setup_projection_dominates(
            order(smaller_setup, &[2], &context(2, 7), admission(2, 8)),
            order(smaller_descriptor, &[1], &context(2, 7), admission(2, 8),),
        ));
        assert!(!setup_projection_dominates(
            order(smaller_descriptor, &[1], &context(2, 7), admission(2, 8),),
            order(smaller_setup, &[2], &context(2, 7), admission(2, 8)),
        ));

        assert!(setup_projection_dominates(
            order(
                setup_score(SetupPrefixCapacity::for_natural_len(4), 100, 0, 256),
                &[9],
                &context(2, 8),
                admission(2, 4),
            ),
            order(smaller_setup, &[2], &context(3, 7), admission(2, 8)),
        ));
        assert!(setup_projection_dominates(
            order(
                setup_score(SetupPrefixCapacity::for_natural_len(8), 99, 0, 256),
                &[9],
                &context(3, 8),
                admission(2, 8),
            ),
            order(smaller_setup, &[2], &context(2, 7), admission(2, 8)),
        ));
    }

    #[test]
    fn payload_projection_keeps_setup_descriptor_tradeoffs_that_a_parent_can_mask() {
        assert!(!payload_projection_dominates(
            order(
                payload_score(100, 0, 64),
                &[2],
                &context(2, 7),
                admission(2, 8)
            ),
            order(
                payload_score(100, 0, 128),
                &[1],
                &context(2, 7),
                admission(2, 8)
            ),
        ));
        assert!(!payload_projection_dominates(
            order(
                payload_score(100, 0, 128),
                &[1],
                &context(2, 7),
                admission(2, 8)
            ),
            order(
                payload_score(100, 0, 64),
                &[2],
                &context(2, 7),
                admission(2, 8)
            ),
        ));
        assert!(payload_projection_dominates(
            order(
                payload_score(99, 0, 256),
                &[9],
                &context(3, 8),
                admission(2, 4)
            ),
            order(
                payload_score(100, 0, 64),
                &[1],
                &context(2, 7),
                admission(2, 8)
            ),
        ));
        assert!(payload_projection_dominates(
            order(
                payload_score(100, 0, 64),
                &[1],
                &context(2, 7),
                admission(2, 8)
            ),
            order(
                payload_score(100, 0, 128),
                &[2],
                &context(2, 7),
                admission(2, 8)
            ),
        ));
    }

    #[test]
    fn payload_projection_prices_every_nonce_alignment() {
        let admission = admission(2, 8);
        let context = context(2, 7);
        let smaller_payload = order(payload_score(100, 8, 64), &[1], &context, admission);
        let smaller_nonce = order(payload_score(101, 0, 64), &[2], &context, admission);

        assert!(payload_projection_dominates(smaller_payload, smaller_nonce));
        assert!(!payload_projection_dominates(
            smaller_nonce,
            smaller_payload,
        ));
    }

    #[test]
    fn projection_dominance_preserves_parent_admission_and_descriptor_order() {
        let score = payload_score(100, 0, 64);
        let two_fold = admission(2, 8);

        assert!(!payload_projection_dominates(
            order(
                payload_score(99, 0, 32),
                &[1],
                &context(1, 7),
                admission(1, 8)
            ),
            order(score, &[2], &context(2, 7), two_fold),
        ));
        assert!(!payload_projection_dominates(
            order(
                payload_score(99, 0, 32),
                &[1],
                &context(2, 7),
                admission(2, 16)
            ),
            order(score, &[2], &context(2, 7), two_fold),
        ));
        assert!(!payload_projection_dominates(
            order(score, &[1], &context(2, 8), two_fold),
            order(score, &[2], &context(2, 7), two_fold),
        ));
        assert!(!payload_projection_dominates(
            order(score, &[1], &context(3, 7), two_fold),
            order(score, &[2], &context(2, 7), two_fold),
        ));

        assert!(!admission(0, 8).is_admitted_by(true, false, 16));
        assert!(admission(1, 8).is_admitted_by(true, false, 16));
        assert!(!admission(1, 8).is_admitted_by(false, true, 16));
        assert!(admission(2, 8).is_admitted_by(false, true, 16));
        assert!(!admission(2, 16).is_admitted_by(false, true, 16));
    }

    #[test]
    fn strict_primary_dominance_does_not_consider_maskable_setup_or_ties() {
        let capacity = SetupPrefixCapacity::for_natural_len(8);
        let compatible = admission(2, 8);
        assert!(setup_primary_strictly_dominates(
            setup_score(capacity, 99, 0, 256),
            compatible,
            setup_score(capacity, 100, 0, 64),
            compatible,
        ));
        assert!(!setup_primary_strictly_dominates(
            setup_score(capacity, 100, 0, 64),
            compatible,
            setup_score(capacity, 100, 0, 128),
            compatible,
        ));
        assert!(payload_primary_strictly_dominates(
            payload_score(99, 0, 256),
            compatible,
            payload_score(100, 0, 64),
            compatible,
        ));
        assert!(!payload_primary_strictly_dominates(
            payload_score(100, 0, 64),
            compatible,
            payload_score(100, 0, 128),
            compatible,
        ));
        assert!(!payload_primary_strictly_dominates(
            payload_score(99, 0, 32),
            admission(1, 8),
            payload_score(100, 0, 64),
            compatible,
        ));
    }

    fn metrics(natural_len: usize, proof_bytes: usize) -> CandidateMetrics {
        CandidateMetrics {
            first_direct_setup_capacity: SetupPrefixCapacity::for_natural_len(natural_len),
            cost: PackedProofCost::new(proof_bytes, 0).unwrap(),
            setup_field_elements: 0,
        }
    }

    #[test]
    fn recursive_bound_requires_dominance_in_both_parent_projections() {
        let candidate_admission = admission(2, 16);
        let lower_bound = CompleteObjectiveBound::SetupFirst {
            first_direct_setup_capacity: 16,
            proof_bytes: 10,
            setup_field_elements: 0,
        };
        let setup_winner = (admission(2, 8), metrics(8, 100));

        assert!(!projection_bound_is_dominated(
            Projection::Payload,
            candidate_admission,
            lower_bound,
            [setup_winner],
        ));

        assert!(projection_bound_is_dominated(
            Projection::FirstDirectSetup,
            candidate_admission,
            lower_bound,
            [setup_winner],
        ));
        assert!(projection_bound_is_dominated(
            Projection::Payload,
            candidate_admission,
            lower_bound,
            [(admission(2, 8), metrics(8, 9))],
        ));
        assert!(!projection_bound_is_dominated(
            Projection::Payload,
            candidate_admission,
            lower_bound,
            [(admission(2, 8), metrics(8, 10))],
        ));
    }
}
