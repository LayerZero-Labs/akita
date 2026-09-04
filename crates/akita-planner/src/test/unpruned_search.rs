use super::*;

#[path = "unpruned_search/candidate.rs"]
mod candidate;
#[path = "unpruned_search/frontier.rs"]
mod frontier;
#[path = "unpruned_search/relation.rs"]
mod relation;
#[path = "unpruned_search/score.rs"]
mod score;
#[path = "unpruned_search/suffix.rs"]
mod suffix;

use candidate::{prepend_fold, prepend_root, terminal};
use frontier::{retain as retain_frontier_candidate, OracleFrontier};
use relation::OracleRelationState;
use score::{schedule_descriptor_bytes, score, OracleScore};
use suffix::visit_suffixes;

struct UnprunedCtx<'a> {
    policy: &'a PlannerPolicy,
    ring_challenge_config: &'a dyn Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct UnprunedState {
    level: usize,
    input_witness_len: usize,
    current_log_basis: u32,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
    dimension_ceiling: CommitmentRingDims,
    payload_phase: akita_types::CommitmentPayloadPhase,
    relation_state: OracleRelationState,
}

impl std::hash::Hash for UnprunedState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.level.hash(state);
        self.input_witness_len.hash(state);
        self.current_log_basis.hash(state);
        self.source_moment.hash(state);
        self.dimension_ceiling.d_a().hash(state);
        self.dimension_ceiling.d_b().hash(state);
        self.dimension_ceiling.d_d().hash(state);
        self.payload_phase.hash(state);
        self.relation_state.hash(state);
    }
}

pub(super) const MAX_ORACLE_SUFFIX_STATES: usize = 2_000_000;
pub(super) const MAX_ORACLE_COMPLETE_SCHEDULES: usize = 1_000_000;
pub(super) const MAX_ORACLE_RECURSION_DEPTH: usize = 4;

#[derive(Default)]
struct OracleWork {
    suffix_states: usize,
    reduced_fold_candidates: usize,
    linf_candidates: usize,
    l2_candidates: usize,
}

type OracleMemo = std::collections::HashMap<UnprunedState, Arc<Vec<ScheduleCandidate>>>;

impl OracleWork {
    fn visit_suffix_state(&mut self) -> Result<(), AkitaError> {
        self.suffix_states = self.suffix_states.checked_add(1).ok_or_else(|| {
            AkitaError::InvalidSetup("unpruned suffix work counter overflow".into())
        })?;
        if self.suffix_states > MAX_ORACLE_SUFFIX_STATES {
            return Err(AkitaError::InvalidSetup(
                "unpruned fixture exceeded its suffix-state work bound".into(),
            ));
        }
        Ok(())
    }

    fn record_reduced_fold_candidates(&mut self, count: usize) -> Result<(), AkitaError> {
        self.reduced_fold_candidates =
            self.reduced_fold_candidates
                .checked_add(count)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("unpruned reduced-candidate counter overflow".into())
                })?;
        Ok(())
    }

    fn record_candidate_route(&mut self, params: &CommittedGroupParams) -> Result<(), AkitaError> {
        let counter = match params.inner().matrix.security_route() {
            akita_types::InnerCommitSecurityRoute::Linf(_) => &mut self.linf_candidates,
            akita_types::InnerCommitSecurityRoute::L2 { .. } => &mut self.l2_candidates,
        };
        *counter = counter.checked_add(1).ok_or_else(|| {
            AkitaError::InvalidSetup("unpruned candidate-route counter overflow".into())
        })?;
        Ok(())
    }
}

#[derive(Debug)]
pub(super) struct OracleSearchResult {
    pub(super) planned: PlannedFoldSchedule,
    pub(super) suffix_states: usize,
    pub(super) complete_schedules: usize,
    pub(super) reduced_fold_candidates: usize,
    pub(super) linf_candidates: usize,
    pub(super) l2_candidates: usize,
}

struct RootCandidate<'a> {
    params: &'a CommittedGroupParams,
    input_witness_len: usize,
    output_witness_len: usize,
}

fn consider_complete_schedule(
    policy: &PlannerPolicy,
    schedule_key: &akita_types::AkitaScheduleLookupKey,
    root: RootCandidate<'_>,
    suffix: &ScheduleCandidate,
    complete_schedules: &std::cell::Cell<usize>,
    selected: &mut Option<(OracleScore, ScheduleCandidate)>,
) -> Result<(), AkitaError> {
    let visited = complete_schedules.get().checked_add(1).ok_or_else(|| {
        AkitaError::InvalidSetup("unpruned complete-schedule counter overflow".into())
    })?;
    if visited > MAX_ORACLE_COMPLETE_SCHEDULES {
        return Err(AkitaError::InvalidSetup(
            "unpruned fixture exceeded its complete-schedule work bound".into(),
        ));
    }
    complete_schedules.set(visited);
    let candidate = prepend_root(
        policy,
        schedule_key,
        root.input_witness_len,
        root.params,
        root.output_witness_len,
        suffix,
    )?;
    if !policy.admits_setup_field_elements(candidate.setup_field_elements) {
        return Ok(());
    }
    let candidate_score = score(policy, &candidate)?;
    if selected
        .as_ref()
        .is_none_or(|(best_score, _)| candidate_score < *best_score)
    {
        *selected = Some((candidate_score, candidate));
    }
    Ok(())
}

pub(super) fn find_schedule(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    honest_fold_policy: HonestFoldPolicySpec,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
) -> Result<OracleSearchResult, AkitaError> {
    key.validate()?;
    akita_schedules::planner_support::validate_policy(policy)?;

    let field_bits = policy.decomposition.field_bits();
    let input_witness_len = 1usize.checked_shl(key.num_vars() as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("unpruned traversal root witness too large".into())
    })?;
    let (min_log_basis, max_log_basis) = crate::policy::log_basis_search_range_at_level(policy, 0);
    let mut selected: Option<(OracleScore, ScheduleCandidate)> = None;
    let mut work = OracleWork::default();
    let mut memo = OracleMemo::new();
    let complete_schedules = std::cell::Cell::new(0usize);
    let schedule_key = akita_types::AkitaScheduleLookupKey::single(key);
    let ctx = UnprunedCtx {
        policy,
        ring_challenge_config: &ring_challenge_config,
    };
    let inner_source =
        root_inner_basis_source(honest_fold_policy, policy.decomposition.log_commit_bound);
    let (min_inner_basis, max_inner_basis) = inner_source.search_range(policy)?;
    let relation_state = OracleRelationState::QuotientPrefix;
    for log_basis in min_log_basis..=max_log_basis {
        for inner_basis in min_inner_basis..=max_inner_basis {
            for root_dimensions in
                dimension_candidates(policy, 0, initial_dimension_ceiling(policy)?)?
            {
                let alpha = root_dimensions.d_a().trailing_zeros() as usize;
                let reduced_vars = key.num_vars().saturating_sub(alpha);
                if reduced_vars == 0 {
                    continue;
                }
                let root_openings =
                    crate::schedule_params::PlannerOpeningCandidate::coefficient_packing_domain(
                        0,
                        policy.claim_ext_degree,
                        root_dimensions,
                    )?;
                for root_opening in root_openings {
                    for (root_params, output_witness_len) in
                        crate::planner::exhaustive_root_candidates_for_reference(
                            &schedule_key,
                            honest_fold_policy,
                            policy,
                            root_dimensions,
                            root_opening,
                            inner_basis,
                            log_basis,
                        )?
                    {
                        let next_source_moment = if policy.selective_l2_response_model_enabled() {
                            let opening_layout = schedule_key.opening_layout()?;
                            let source_groups = crate::response_model::root_group_source_moments(
                                &root_params,
                                &opening_layout,
                                honest_fold_policy,
                                &[],
                                policy.decomposition,
                            )?;
                            Some(crate::response_model::next_source_moment(
                                &root_params,
                                &opening_layout,
                                &source_groups,
                                field_bits,
                                policy.claim_ext_degree,
                            )?)
                        } else {
                            None
                        };
                        visit_suffixes(
                            &ctx,
                            UnprunedState {
                                level: 1,
                                input_witness_len: output_witness_len,
                                current_log_basis: log_basis,
                                source_moment: next_source_moment,
                                dimension_ceiling: root_dimensions,
                                payload_phase:
                                    akita_types::CommitmentPayloadPhase::CompressedPrefix,
                                relation_state,
                            },
                            &mut memo,
                            &mut work,
                            &mut |suffix| {
                                consider_complete_schedule(
                                    policy,
                                    &schedule_key,
                                    RootCandidate {
                                        params: &root_params,
                                        input_witness_len,
                                        output_witness_len,
                                    },
                                    &suffix,
                                    &complete_schedules,
                                    &mut selected,
                                )
                            },
                        )?;
                    }
                }
            }
        }
    }

    let Some((_, selected)) = selected else {
        return Err(AkitaError::UnsupportedSchedule(
            "unpruned traversal found no complete schedule".into(),
        ));
    };
    let cached_first_direct_setup_field_len = if matches!(
        policy.selection_policy,
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayloadV2
            | crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
    ) {
        selected.first_direct_setup_field_len.map(NonZeroUsize::get)
    } else {
        None
    };
    let selected_descriptor = schedule_descriptor_bytes(&selected)?;
    let planned = materialize_candidate_schedule(
        selected.cost.proof_bytes(),
        selected.setup_field_elements,
        cached_first_direct_setup_field_len,
        policy,
        &schedule_key.opening_layout()?,
        selected.folds.to_vec(),
        selected.terminal.as_ref().clone(),
    )?;
    if selected_descriptor != planned.schedule.canonical_descriptor_bytes() {
        return Err(AkitaError::InvalidSetup(
            "oracle candidate descriptor disagrees with its materialized schedule".into(),
        ));
    }
    Ok(OracleSearchResult {
        planned,
        suffix_states: work.suffix_states,
        complete_schedules: complete_schedules.get(),
        reduced_fold_candidates: work.reduced_fold_candidates,
        linf_candidates: work.linf_candidates,
        l2_candidates: work.l2_candidates,
    })
}
