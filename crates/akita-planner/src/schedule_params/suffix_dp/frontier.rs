use std::{collections::BTreeMap, sync::Arc};

use akita_error::AkitaError;

use crate::PlannerPolicy;

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

    const fn bit(self) -> u8 {
        match self {
            Self::FirstDirectSetup => 1,
            Self::Payload => 2,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct ProjectionMask(u8);

impl ProjectionMask {
    fn insert(&mut self, projection: Projection) {
        self.0 |= projection.bit();
    }

    fn remove(&mut self, projection: Projection) {
        self.0 &= !projection.bit();
    }

    const fn contains(self, projection: Projection) -> bool {
        self.0 & projection.bit() != 0
    }

    const fn is_empty(self) -> bool {
        self.0 == 0
    }

    fn iter(self) -> impl Iterator<Item = Projection> {
        Projection::ALL
            .into_iter()
            .filter(move |&projection| self.contains(projection))
    }
}

#[derive(Clone, Copy)]
pub(super) struct PricedChildEdge {
    edge_price: super::ChildEdgePrice,
    edge_grinding_cost: akita_types::TranscriptGrindingCost,
}

pub(super) fn price_child_edge(
    edge: &super::ChildEdge<'_>,
    successor_class: &ParentObservableKey,
    representative: &ScheduleCandidate,
) -> Result<PricedChildEdge, AkitaError> {
    if first_parent_visible_cost(edge.policy, representative)? != *successor_class {
        return Err(AkitaError::InvalidSetup(
            "suffix frontier candidate disagrees with its parent-observable class".into(),
        ));
    }
    let edge_price = child_edge_price(edge, representative)?;
    let edge_grinding_cost = edge.grinding_cost(representative, edge_price.relation_geometry)?;
    Ok(PricedChildEdge {
        edge_price,
        edge_grinding_cost,
    })
}

pub(super) fn consider_child_suffixes<'a>(
    edge: &super::ChildEdge<'_>,
    child_candidates: impl IntoIterator<Item = &'a ScheduleCandidate>,
    priced_edge: PricedChildEdge,
    parent_cost: &ParentObservableKey,
    incoming_setup_prefix: Option<usize>,
    projections: &[Projection],
    frontier: &mut ProjectedFrontier,
) -> Result<(), AkitaError> {
    // `SuffixResult` partitions candidates by every successor coordinate a
    // parent can observe. Price the edge and grinding plan once for that class;
    // rebuilding them for descriptor-distinct members is redundant.
    for suffix in child_candidates {
        let Some(candidate) = child_choice(
            edge,
            priced_edge.edge_price,
            priced_edge.edge_grinding_cost,
            suffix,
        )?
        else {
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
            parent_cost,
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
    first_direct_output_witness_len: usize,
    cost: crate::schedule_params::PackedProofCost,
    setup_field_elements: usize,
}

#[derive(Clone, Copy)]
struct PayloadScore {
    cost: crate::schedule_params::PackedProofCost,
    setup_field_elements: usize,
}

fn setup_envelope_score(
    selection_policy: crate::SelectionPolicyId,
    setup_field_elements: usize,
) -> usize {
    if selection_policy
        == crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
    {
        akita_types::padded_setup_prefix_len(setup_field_elements)
    } else {
        setup_field_elements
    }
}

fn setup_score(
    selection_policy: crate::SelectionPolicyId,
    metrics: super::super::CandidateMetrics,
) -> SetupScore {
    SetupScore {
        first_direct_setup_capacity: metrics.first_direct_setup_capacity,
        first_direct_output_witness_len: metrics.first_direct_output_witness_len,
        cost: metrics.cost,
        setup_field_elements: setup_envelope_score(selection_policy, metrics.setup_field_elements),
    }
}

fn payload_score(
    selection_policy: crate::SelectionPolicyId,
    metrics: super::super::CandidateMetrics,
) -> PayloadScore {
    PayloadScore {
        cost: metrics.cost,
        setup_field_elements: setup_envelope_score(selection_policy, metrics.setup_field_elements),
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

    fn for_pending(candidate: &PendingScheduleCandidate) -> Self {
        Self {
            fold_count: candidate.suffix_folds.len() + 1,
            first_fold_descriptor: Some(
                candidate
                    .first_fold
                    .params
                    .canonical_descriptor_bytes()
                    .into(),
            ),
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

    fn for_pending(candidate: &PendingScheduleCandidate) -> Self {
        Self {
            fold_depth: (candidate.suffix_folds.len() + 1).min(2) as u8,
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
        let retained_projections = self.retained_primary_projections(
            policy,
            &parent_cost,
            metrics,
            admission,
            projections,
        );
        if retained_projections.is_empty() {
            return Ok(());
        }
        let projected = ProjectedCandidate {
            descriptor: super::super::candidate_schedule_descriptor_bytes(
                None,
                &candidate.folds,
                &candidate.terminal.params,
                diagnostics,
            )?
            .into(),
            descriptor_context: DescriptorOrderContext::for_candidate(&candidate),
            admission,
            schedule: candidate,
        };
        let choices = self.by_parent_cost.entry(parent_cost).or_default();
        for projection in retained_projections.iter() {
            insert_projected(
                choices.projected_mut(projection),
                projected.clone(),
                |left, right| match projection {
                    Projection::FirstDirectSetup => {
                        setup_dominates_for_policy(policy.selection_policy, left, right)
                    }
                    Projection::Payload => {
                        payload_dominates_for_policy(policy.selection_policy, left, right)
                    }
                },
            );
        }
        Ok(())
    }

    fn retained_primary_projections(
        &self,
        policy: &PlannerPolicy,
        parent_cost: &ParentObservableKey,
        metrics: super::super::CandidateMetrics,
        admission: ParentAdmissionClass,
        projections: &[Projection],
    ) -> ProjectionMask {
        let choices = self.by_parent_cost.get(parent_cost);
        let keep = |projection| match projection {
            Projection::FirstDirectSetup => {
                matches!(
                policy.selection_policy,
                crate::SelectionPolicyId::MinFirstDirectSetupThenPayloadV2
                    | crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
            ) && !choices.is_some_and(|choices| {
                    choices.projected(projection).iter().any(|existing| {
                        setup_primary_strictly_dominates(
                            policy.selection_policy,
                            setup_score(policy.selection_policy, existing.schedule.metrics()),
                            existing.admission,
                            setup_score(policy.selection_policy, metrics),
                            admission,
                        )
                    })
                })
            }
            Projection::Payload => !choices.is_some_and(|choices| {
                choices.projected(projection).iter().any(|existing| {
                    payload_primary_strictly_dominates(
                        policy.selection_policy,
                        payload_score(policy.selection_policy, existing.schedule.metrics()),
                        existing.admission,
                        payload_score(policy.selection_policy, metrics),
                        admission,
                    )
                })
            }),
        };
        let mut retained = ProjectionMask::default();
        for &projection in projections {
            if keep(projection) {
                retained.insert(projection);
            }
        }
        retained
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
        let admission = ParentAdmissionClass::for_pending(&pending);
        let metrics = pending.metrics();
        let mut retained_projections =
            self.retained_primary_projections(policy, parent_cost, metrics, admission, projections);
        if retained_projections.is_empty() {
            return Ok(());
        }

        // A candidate needs its canonical descriptor to resolve equal numeric
        // frontiers, but it does not need a newly allocated fold-chain node
        // until at least one projection actually retains it.
        let descriptor: Arc<[u8]> = pending.descriptor_bytes(diagnostics)?.into();
        let descriptor_context = DescriptorOrderContext::for_pending(&pending);
        if let Some(choices) = self.by_parent_cost.get(parent_cost) {
            for projection in retained_projections.iter() {
                if choices
                    .projected(projection)
                    .iter()
                    .any(|existing| match projection {
                        Projection::FirstDirectSetup => setup_projection_dominates(
                            policy.selection_policy,
                            projected_setup_order(policy.selection_policy, existing),
                            ProjectionOrder {
                                score: setup_score(policy.selection_policy, metrics),
                                descriptor: descriptor.as_ref(),
                                context: &descriptor_context,
                                admission,
                            },
                        ),
                        Projection::Payload => payload_projection_dominates(
                            policy.selection_policy,
                            projected_payload_order(policy.selection_policy, existing),
                            ProjectionOrder {
                                score: payload_score(policy.selection_policy, metrics),
                                descriptor: descriptor.as_ref(),
                                context: &descriptor_context,
                                admission,
                            },
                        ),
                    })
                {
                    retained_projections.remove(projection);
                }
            }
        }
        if retained_projections.is_empty() {
            return Ok(());
        }

        let projected = ProjectedCandidate {
            descriptor,
            descriptor_context,
            admission,
            schedule: pending.into_candidate(),
        };
        let choices = self.by_parent_cost.entry(parent_cost.clone()).or_default();
        for projection in retained_projections.iter() {
            let frontier = choices.projected_mut(projection);
            frontier.retain(|existing| match projection {
                Projection::FirstDirectSetup => !setup_projection_dominates(
                    policy.selection_policy,
                    projected_setup_order(policy.selection_policy, &projected),
                    projected_setup_order(policy.selection_policy, existing),
                ),
                Projection::Payload => !payload_projection_dominates(
                    policy.selection_policy,
                    projected_payload_order(policy.selection_policy, &projected),
                    projected_payload_order(policy.selection_policy, existing),
                ),
            });
            frontier.push(projected.clone());
        }
        Ok(())
    }
}

fn setup_dominates_for_policy(
    selection_policy: crate::SelectionPolicyId,
    left: &ProjectedCandidate,
    right: &ProjectedCandidate,
) -> bool {
    setup_projection_dominates(
        selection_policy,
        projected_setup_order(selection_policy, left),
        projected_setup_order(selection_policy, right),
    )
}

fn projected_setup_order(
    selection_policy: crate::SelectionPolicyId,
    candidate: &ProjectedCandidate,
) -> ProjectionOrder<'_, SetupScore> {
    ProjectionOrder {
        score: setup_score(selection_policy, candidate.schedule.metrics()),
        descriptor: candidate.descriptor.as_ref(),
        context: &candidate.descriptor_context,
        admission: candidate.admission,
    }
}

fn setup_primary_strictly_dominates(
    selection_policy: crate::SelectionPolicyId,
    left_score: SetupScore,
    left_admission: ParentAdmissionClass,
    right_score: SetupScore,
    right_admission: ParentAdmissionClass,
) -> bool {
    if !left_admission.admits_every_parent_of(right_admission)
        || !left_score.cost.queries_never_worse(right_score.cost)
    {
        return false;
    }
    if matches!(
        selection_policy,
        crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
    ) {
        return left_score.setup_field_elements <= right_score.setup_field_elements
            && (left_score.first_direct_setup_capacity < right_score.first_direct_setup_capacity
                || (left_score.first_direct_setup_capacity
                    == right_score.first_direct_setup_capacity
                    && (left_score
                        .cost
                        .strictly_better_for_every_parent(right_score.cost)
                        || (left_score
                            .cost
                            .never_worse_for_every_parent(right_score.cost)
                            && left_score.first_direct_output_witness_len
                                < right_score.first_direct_output_witness_len))));
    }
    left_score.first_direct_setup_capacity < right_score.first_direct_setup_capacity
        || (left_score.first_direct_setup_capacity == right_score.first_direct_setup_capacity
            && left_score
                .cost
                .strictly_better_for_every_parent(right_score.cost))
}

#[derive(Clone, Copy)]
struct ProjectionOrder<'a, Score> {
    score: Score,
    descriptor: &'a [u8],
    context: &'a DescriptorOrderContext,
    admission: ParentAdmissionClass,
}

fn setup_projection_dominates(
    selection_policy: crate::SelectionPolicyId,
    left: ProjectionOrder<'_, SetupScore>,
    right: ProjectionOrder<'_, SetupScore>,
) -> bool {
    if !left.admission.admits_every_parent_of(right.admission)
        || !left.score.cost.queries_never_worse(right.score.cost)
    {
        return false;
    }
    let cost_never_worse = left
        .score
        .cost
        .never_worse_for_every_parent(right.score.cost);
    let equal_output_is_canonical = cost_never_worse
        && left.score.first_direct_output_witness_len
            == right.score.first_direct_output_witness_len
        && left.context == right.context
        && left.descriptor <= right.descriptor;
    let equal_later_coordinates_are_canonical =
        cost_never_worse && left.context == right.context && left.descriptor <= right.descriptor;
    if matches!(
        selection_policy,
        crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
    ) {
        return left.score.setup_field_elements <= right.score.setup_field_elements
            && (left.score.first_direct_setup_capacity < right.score.first_direct_setup_capacity
                || (left.score.first_direct_setup_capacity
                    == right.score.first_direct_setup_capacity
                    && (left
                        .score
                        .cost
                        .strictly_better_for_every_parent(right.score.cost)
                        || (cost_never_worse
                            && left.score.first_direct_output_witness_len
                                < right.score.first_direct_output_witness_len)
                        || equal_output_is_canonical)));
    }
    left.score.first_direct_setup_capacity < right.score.first_direct_setup_capacity
        || (left.score.first_direct_setup_capacity == right.score.first_direct_setup_capacity
            && (left
                .score
                .cost
                .strictly_better_for_every_parent(right.score.cost)
                || (equal_later_coordinates_are_canonical
                    && left.score.setup_field_elements <= right.score.setup_field_elements)))
}

fn payload_dominates_for_policy(
    selection_policy: crate::SelectionPolicyId,
    left: &ProjectedCandidate,
    right: &ProjectedCandidate,
) -> bool {
    payload_projection_dominates(
        selection_policy,
        projected_payload_order(selection_policy, left),
        projected_payload_order(selection_policy, right),
    )
}

fn projected_payload_order(
    selection_policy: crate::SelectionPolicyId,
    candidate: &ProjectedCandidate,
) -> ProjectionOrder<'_, PayloadScore> {
    ProjectionOrder {
        score: payload_score(selection_policy, candidate.schedule.metrics()),
        descriptor: candidate.descriptor.as_ref(),
        context: &candidate.descriptor_context,
        admission: candidate.admission,
    }
}

fn payload_primary_strictly_dominates(
    selection_policy: crate::SelectionPolicyId,
    left_score: PayloadScore,
    left_admission: ParentAdmissionClass,
    right_score: PayloadScore,
    right_admission: ParentAdmissionClass,
) -> bool {
    left_admission.admits_every_parent_of(right_admission)
        && left_score
            .cost
            .strictly_better_for_every_parent(right_score.cost)
        && (selection_policy
            != crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
            || left_score.setup_field_elements <= right_score.setup_field_elements)
}

fn payload_projection_dominates(
    selection_policy: crate::SelectionPolicyId,
    left: ProjectionOrder<'_, PayloadScore>,
    right: ProjectionOrder<'_, PayloadScore>,
) -> bool {
    left.admission.admits_every_parent_of(right.admission)
        && (selection_policy
            != crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
            || left.score.setup_field_elements <= right.score.setup_field_elements)
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
    dominates: impl Fn(&ProjectedCandidate, &ProjectedCandidate) -> bool,
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
mod tests;
