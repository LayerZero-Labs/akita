//! Root schedule planning.

use std::time::Instant;

mod root_layouts;
mod root_openings;
pub(crate) use root_layouts::PreparedRootFinalGroup;
#[cfg(all(test, feature = "catalog-gen"))]
use root_layouts::{
    root_final_group_level_params_candidate, MultiGroupRootCandidateCtx,
    RootFinalGroupCandidateInput,
};
pub(crate) use root_openings::smallest_witness_root_candidates;

use akita_error::{checked, AkitaError};
use akita_params::sis::{
    decomposed_s_block_ring_count, num_digits_open, rounded_up_collision_inf_norm,
    rounded_up_role_a_inf_norm, CommittedSourceContract, HonestFoldPolicy, HonestFoldSizingQuery,
    OpenCommitMatrixParams, SisMatrixRole,
};
use akita_params::ScheduleLookupKey;
use akita_params::{
    CommitmentRingDims, CommittedGroupParams, DecompositionParams, GroupCommitPhaseParams,
    GroupOpenPhaseParams, OpeningClaimsLayout, PlannedFoldSchedule, PolynomialGroupLayout,
    PrecommittedGroupAdmissionPolicy,
};

use akita_schedules::planner_support::projected_collision_role_price;
use akita_schedules::ResolvedScheduleRow;

use crate::schedule_params::{
    derive_ab_commitment_candidate, derive_selected_suffix_schedule,
    materialize_candidate_schedule, recursive_split_search_domain, select_complete_candidate,
    AbCommitmentCandidateRequest, CandidateMaterializationCost, PlannerOpeningCandidate,
    RingChallengeConfigFn, ScheduleMemo, SetupFieldBudget, SuffixCtx, SuffixState,
};
use crate::PlannerPolicy;

#[cfg(test)]
#[path = "test/adapted_schedule.rs"]
mod adapted_schedule_tests;
#[cfg(all(test, feature = "catalog-gen"))]
#[path = "test/root_candidates.rs"]
mod root_candidates;
#[cfg(test)]
#[path = "test/root_setup_accumulation.rs"]
mod root_setup_accumulation_tests;
#[cfg(all(test, feature = "catalog-gen"))]
pub(crate) use root_candidates::{
    exhaustive_root_candidates_for_reference, root_level_candidates_with_fresh_preparation,
};

type PrecommittedGroupSeed = (GroupCommitPhaseParams, CommittedSourceContract);

#[derive(Clone, Copy)]
pub(crate) struct ScheduleSearchOptions<'a> {
    pub(crate) relation_traversal_order: super::schedule_params::RelationTraversalOrder,
    pub(crate) relation_mode_filter: super::schedule_params::RelationModeFilter,
    pub(crate) root_main_constraint: Option<&'a CommittedGroupParams>,
    pub(crate) adaptation_guide: Option<&'a akita_params::FoldSchedule>,
    #[cfg(test)]
    pub(crate) query_prefix_count: u64,
}

impl ScheduleSearchOptions<'_> {
    const fn canonical() -> Self {
        Self {
            relation_traversal_order: super::schedule_params::RelationTraversalOrder::Canonical,
            relation_mode_filter: super::schedule_params::RelationModeFilter::All,
            root_main_constraint: None,
            adaptation_guide: None,
            #[cfg(test)]
            query_prefix_count: 0,
        }
    }
}

/// Partition precommitted groups into interchangeable classes: groups with
/// equal commit-phase profiles and equal source contracts.
///
/// Root planning assigns one opening to each class, and every group of the
/// class opens with it. Groups of one class therefore materialize to identical
/// parameters, and the opening search grows with the number of classes rather
/// than with the number of groups.
pub(crate) fn precommitted_group_equivalence_classes(
    profiles: &[GroupCommitPhaseParams],
    source_contracts: &[CommittedSourceContract],
) -> Result<Vec<Vec<usize>>, AkitaError> {
    if profiles.len() != source_contracts.len() {
        return Err(AkitaError::InvalidSetup(
            "group-batch planning requires one source contract per precommitted profile"
                .to_string(),
        ));
    }
    let mut classes: Vec<Vec<usize>> = Vec::new();
    for index in 0..profiles.len() {
        if let Some(indices) = classes.iter_mut().find(|indices| {
            let representative = indices[0];
            profiles[representative] == profiles[index]
                && source_contracts[representative] == source_contracts[index]
        }) {
            indices.push(index);
        } else {
            classes.push(vec![index]);
        }
    }
    Ok(classes)
}

fn materialize_precommitted_group_for_open_basis(
    (layout, source_contract): &PrecommittedGroupSeed,
    policy: &PlannerPolicy,
    opening: PlannerOpeningCandidate,
    shared_opening_ring_dimension: usize,
    log_basis_open: u32,
) -> Result<Option<GroupOpenPhaseParams>, AkitaError> {
    let honest_fold_policy = source_contract
        .class()
        .honest_fold_policy(source_contract.decomposition().field_bits());
    let ring_dimension = layout.inner.matrix.ring_dimension();
    opening.validate_for(
        policy.claim_ext_degree,
        CommitmentRingDims {
            inner: ring_dimension,
            outer: layout.outer.matrix.ring_dimension(),
            opening: shared_opening_ring_dimension,
        },
    )?;
    let num_chunks = policy.chunks_at_level(0);
    let num_fold_coeffs = layout
        .inner
        .matrix
        .input_width()
        .checked_mul(ring_dimension)
        .and_then(|count| count.checked_mul(num_chunks))
        .ok_or_else(|| AkitaError::InvalidSetup("precommitted fold width overflow".into()))?;
    let group_claims = layout.group.num_polynomials();
    let num_digits_fold = honest_fold_policy.num_digits_fold(HonestFoldSizingQuery {
        ring_dimension,
        challenge_dimension: opening.challenge_dimension(ring_dimension),
        num_claims: group_claims,

        num_live_ring_elements_per_claim: layout.blocks.live_ring_elements_per_claim,
        num_positions_per_block: layout.blocks.positions_per_block,
        num_live_blocks: layout.blocks.live_blocks,

        num_chunks,
        num_fold_coeffs,
        witness_norms: source_contract
            .source_norms(
                layout.inner.digits.log_basis,
                layout.inner.digits.num_digits,
                ring_dimension,
                akita_error::checked::pow2(layout.group.num_vars())
                    .and_then(|len| len.checked_mul(group_claims))
                    .ok_or_else(|| AkitaError::InvalidSetup("source length overflow".into()))?,
            )?
            .fold_witness,
        log_basis_response: log_basis_open,
        challenge_config: &opening.challenge_config(),
    })?;
    let Some(required_a_bound) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        ring_dimension,
        log_basis_open,
        &opening.challenge_config(),
        num_digits_fold,
        num_chunks,
    ) else {
        return Ok(None);
    };
    let declared_a_bound =
        layout.inner.matrix.coeff_linf_bound().ok_or_else(|| {
            AkitaError::InvalidSetup("precommitted A cannot use an L2 route".into())
        })?;
    let Some(required_b_bound) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        SisMatrixRole::Outer,
        layout.outer.matrix.ring_dimension(),
        log_basis_open,
    ) else {
        return Ok(None);
    };
    if required_a_bound > declared_a_bound
        || required_b_bound > layout.outer.matrix.coeff_linf_bound()
    {
        return Ok(None);
    }
    GroupOpenPhaseParams::admit(
        *layout,
        num_digits_fold,
        PrecommittedGroupAdmissionPolicy {
            decomposition: policy.decomposition,
            sis_security_policy: policy.sis_security_policy,
            sis_table_digest: policy.sis_table_digest,
            sis_modulus_profile: policy.sis_modulus_profile,
            num_response_chunks: num_chunks,
        },
        opening.method(),
        opening.challenge_config(),
        log_basis_open,
    )
    .map(Some)
}

pub(crate) fn root_batch_next_w_len(
    field_bits: u32,
    extension_degree: usize,
    params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
) -> Result<Option<usize>, AkitaError> {
    if !params.compression_sources_supported()? {
        return Ok(None);
    }
    params
        .output_witness_len_for_field_bits(field_bits, extension_degree, opening_batch)
        .map(Some)
}

pub(crate) struct PreparedRootProducers {
    groups: Vec<GroupOpenPhaseParams>,
    d_width: usize,
}

impl PreparedRootProducers {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        key: &ScheduleLookupKey,
        precommitted_source_contracts: &[CommittedSourceContract],
        policy: &PlannerPolicy,
        dimensions: CommitmentRingDims,
        opening: PlannerOpeningCandidate,
        precommitted_class_openings: &[PlannerOpeningCandidate],
        candidate_log_basis_open: u32,
    ) -> Result<Option<Self>, AkitaError> {
        dimensions.validate_role_projection()?;
        opening.validate_for(policy.claim_ext_degree, dimensions)?;
        let alpha = dimensions.d_a().trailing_zeros() as usize;
        if key.final_group.num_vars().saturating_sub(alpha) == 0 {
            return Ok(None);
        }
        let equivalence_classes = precommitted_group_equivalence_classes(
            &key.precommitteds,
            precommitted_source_contracts,
        )?;
        if precommitted_class_openings
            .iter()
            .any(|candidate| candidate.is_coefficient_packing() != opening.is_coefficient_packing())
        {
            return Ok(None);
        }
        let shared_opening_ring_dimension = dimensions.d_d();
        if !crate::schedule_params::precommitted_groups_support_opening_dimension(
            key.precommitteds.iter(),
            shared_opening_ring_dimension,
        ) {
            return Ok(None);
        }
        if precommitted_class_openings.len() != equivalence_classes.len() {
            return Err(AkitaError::InvalidSetup(
                "root precommit opening candidate count mismatch".into(),
            ));
        }
        let mut admitted = Vec::new();
        for (indices, &opening) in equivalence_classes.iter().zip(precommitted_class_openings) {
            let index = indices[0];
            let Some(group) = materialize_precommitted_group_for_open_basis(
                &(
                    key.precommitteds[index],
                    precommitted_source_contracts[index],
                ),
                policy,
                opening,
                shared_opening_ring_dimension,
                candidate_log_basis_open,
            )?
            else {
                return Ok(None);
            };
            admitted.push((
                group,
                group.d_segment_width(policy.claim_ext_degree, shared_opening_ring_dimension)?,
            ));
        }
        Self::from_class_openings(
            &equivalence_classes,
            admitted.into_iter(),
            key.precommitteds.len(),
        )
        .map(Some)
    }

    fn from_class_openings(
        classes: &[Vec<usize>],
        openings: impl Iterator<Item = (GroupOpenPhaseParams, usize)>,
        num_groups: usize,
    ) -> Result<Self, AkitaError> {
        let mut groups = vec![None; num_groups];
        let mut d_width = 0usize;
        for (indices, (group, width)) in classes.iter().zip(openings) {
            d_width = checked::mul_add(width, indices.len(), d_width)
                .ok_or_else(|| AkitaError::InvalidSetup("root batch D width overflow".into()))?;
            for &index in indices {
                groups[index] = Some(group);
            }
        }
        let groups = groups
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "precommitted equivalence classes must cover every group".into(),
                )
            })?;
        Ok(Self { groups, d_width })
    }

    fn combine_with(
        &self,
        main: &CommittedGroupParams,
    ) -> Result<Option<CommittedGroupParams>, AkitaError> {
        if self.groups.is_empty() {
            return Ok(Some(main.clone()));
        }
        let width = main
            .open_matrix
            .input_width()
            .checked_add(self.d_width)
            .ok_or_else(|| AkitaError::InvalidSetup("root batch D width overflow".into()))?;
        let Ok(matrix) =
            OpenCommitMatrixParams::try_new_with_min_rank(main.open_matrix.sis_table_key(), width)
        else {
            return Ok(None);
        };
        let groups = self
            .groups
            .iter()
            .copied()
            .chain(std::iter::once(*main.own_group()))
            .collect();
        let mut params = CommittedGroupParams::try_new(
            groups,
            matrix,
            main.payload_mode,
            main.ring_relation_mode,
            main.source_encoding,
            akita_params::ChunkedWitnessCfg::default(),
        )?;
        params.witness_chunk = main.witness_chunk;
        Ok(Some(params))
    }
}

/// Build the fold schedule selected by a full schedule lookup key.
///
/// `precommitted_source_contracts` must contain the actual producer declaration
/// for each frozen precommit, in lookup-key order. Frozen commit geometry does
/// not authenticate source class or bound by itself: callers must obtain these
/// contracts from the configurations that produced the commitments. The
/// planner uses them only to size the offline honest-response model; the
/// verifier continues to enforce the response caps frozen into the schedule.
pub fn find_schedule(
    key: &ScheduleLookupKey,
    final_source_contract: CommittedSourceContract,
    precommitted_source_contracts: &[CommittedSourceContract],
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
) -> Result<PlannedFoldSchedule, AkitaError> {
    find_schedule_in_relation_order(
        key,
        final_source_contract,
        precommitted_source_contracts,
        policy,
        ring_challenge_config,
        ScheduleSearchOptions::canonical(),
    )
}

/// Adapt a validated scalar main row to an exact grouped-root key.
///
/// The main row's root A/B geometry and opening plan are frozen. Its recursive
/// depth, per-level dimensions, block split, slicing, bases, opening family,
/// payload/relation phases, and direct-vs-offloaded edges form a guide. The
/// grouped root's shared D matrix and every length-, setup-, rank-, response-,
/// and relation-derived suffix value are rebuilt under `policy`. This guided
/// search is conditional on that skeleton; [`find_schedule`] searches the full
/// policy domain, with exact next-witness ranking for root opening products above 256.
///
/// When that frozen skeleton is no longer feasible, the guided search reports
/// [`AkitaError::UnsupportedSchedule`] and adaptation falls back to
/// [`find_schedule`] for the same key. Adaptation therefore fails only when the
/// full search fails too, or when the request itself is invalid. Invalid
/// requests (no precommitted group, a mismatched final group, or a main row
/// that does not pass audit under `policy`) are rejected without a fallback.
/// A fallback row is the full search's selected result and need not retain
/// any part of the main row's skeleton. When both searches fail, the error
/// carries both reasons.
///
/// The supplied row is expected to come from a caller-approved
/// [`akita_schedules::ValidatedScheduleCatalog`]. The resulting expanded row must
/// still be admitted into the caller's final trusted catalog before runtime
/// proving or verification.
pub fn find_adapted_schedule(
    main_row: &ResolvedScheduleRow,
    request: &crate::emit::GroupedGenerationRequest,
    final_source_contract: CommittedSourceContract,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
) -> Result<PlannedFoldSchedule, AkitaError> {
    let key = request.key();
    let precommitted_source_contracts = request.source_contracts();
    match find_adapted_schedule_for_key(
        main_row,
        &key,
        final_source_contract,
        &precommitted_source_contracts,
        policy,
        &ring_challenge_config,
    ) {
        Err(AkitaError::UnsupportedSchedule(guided)) => find_schedule(
            &key,
            final_source_contract,
            &precommitted_source_contracts,
            policy,
            ring_challenge_config,
        )
        .map_err(|error| match error {
            AkitaError::UnsupportedSchedule(full) => AkitaError::UnsupportedSchedule(format!(
                "{full}; the guided search was also infeasible: {guided}"
            )),
            error => error,
        }),
        adapted => adapted,
    }
}

fn find_adapted_schedule_for_key(
    main_row: &ResolvedScheduleRow,
    key: &ScheduleLookupKey,
    final_source_contract: CommittedSourceContract,
    precommitted_source_contracts: &[CommittedSourceContract],
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
) -> Result<PlannedFoldSchedule, AkitaError> {
    akita_schedules::planner_support::validate_policy(policy)?;
    key.validate(policy.decomposition.field_bits())?;
    if key.precommitteds.is_empty() {
        return Err(AkitaError::InvalidInput(
            "adapted planning requires at least one precommitted group".into(),
        ));
    }
    if key.precommitteds.len() != precommitted_source_contracts.len() {
        return Err(AkitaError::InvalidInput(
            "adapted planning requires one source contract per precommitted group".into(),
        ));
    }
    if !main_row.profiles().precommitteds.is_empty()
        || !main_row
            .schedule()
            .root
            .params
            .precommitted_groups()
            .is_empty()
    {
        return Err(AkitaError::InvalidInput(
            "adapted planning requires a scalar main-group-only row".into(),
        ));
    }
    if key.final_group != main_row.profiles().final_group.group {
        return Err(AkitaError::InvalidInput(
            "adapted final-group layout does not match the main row".into(),
        ));
    }
    // Re-audit the baseline under the caller's current policy. Adaptation never
    // treats a row produced for another policy epoch as a geometry hint.
    ResolvedScheduleRow::try_new(
        main_row.profiles().clone(),
        main_row.schedule().clone(),
        policy,
    )?;

    find_schedule_in_relation_order(
        key,
        final_source_contract,
        precommitted_source_contracts,
        policy,
        ring_challenge_config,
        ScheduleSearchOptions {
            root_main_constraint: Some(&main_row.schedule().root.params),
            adaptation_guide: Some(main_row.schedule()),
            ..ScheduleSearchOptions::canonical()
        },
    )
}

/// Build a schedule under a test-only relation-mode restriction.
#[cfg(feature = "test-support")]
pub fn find_schedule_for_test_relation_mode(
    key: &ScheduleLookupKey,
    final_source_contract: CommittedSourceContract,
    precommitted_source_contracts: &[CommittedSourceContract],
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    relation_mode_filter: super::schedule_params::TestRelationModeFilter,
) -> Result<PlannedFoldSchedule, AkitaError> {
    find_schedule_in_relation_order(
        key,
        final_source_contract,
        precommitted_source_contracts,
        policy,
        ring_challenge_config,
        ScheduleSearchOptions {
            relation_mode_filter: relation_mode_filter.into(),
            ..ScheduleSearchOptions::canonical()
        },
    )
}

/// Canonical schedule search with an internal traversal-order seam used to
/// prove that candidate enumeration does not affect selection.
pub(crate) fn find_schedule_in_relation_order(
    key: &ScheduleLookupKey,
    final_source_contract: CommittedSourceContract,
    precommitted_source_contracts: &[CommittedSourceContract],
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    options: ScheduleSearchOptions<'_>,
) -> Result<PlannedFoldSchedule, AkitaError> {
    let diagnostics = crate::diagnostics::active();
    let diagnostics = diagnostics.as_deref();
    akita_schedules::planner_support::validate_policy(policy)?;
    key.validate(policy.decomposition.field_bits())?;
    if final_source_contract.decomposition().field_bits() != policy.decomposition.field_bits()
        || final_source_contract.decomposition().log_commit_bound
            != policy.decomposition.log_commit_bound
    {
        return Err(AkitaError::InvalidInput(
            "final source contract field and bound must match the planner policy".into(),
        ));
    }
    if key.precommitteds.len() != precommitted_source_contracts.len() {
        return Err(AkitaError::InvalidInput(
            "schedule planning requires one source contract per precommitted group".into(),
        ));
    }
    if precommitted_source_contracts
        .iter()
        .any(|contract| contract.decomposition().field_bits() != policy.decomposition.field_bits())
    {
        return Err(AkitaError::InvalidInput(
            "precommitted source contract field must match the consuming planner field".into(),
        ));
    }
    let ring_challenge_config: RingChallengeConfigFn<'_> = &ring_challenge_config;
    let root_input_witness_len = checked::pow2(key.final_group.num_vars()).ok_or_else(|| {
        AkitaError::InvalidSetup("multi-group root-fold witness length overflow".to_string())
    })?;
    let dimension_ceiling = super::schedule_params::initial_dimension_ceiling(policy)?;
    let initial_state = SuffixState {
        input_chunks: None,
        level: 0,
        current_witness_len: root_input_witness_len,
        current_lb: 0,
        source_moment: None,
        dimension_ceiling,
        topology: super::schedule_params::SuffixTopology::Direct {
            payload_phase: akita_params::CommitmentPayloadPhase::CompressedPrefix,
            relation_phase: super::schedule_params::RingRelationPhase::QuotientPrefix,
        },
    };
    #[cfg(test)]
    let query_prefix_count = options.query_prefix_count;
    #[cfg(not(test))]
    let query_prefix_count = 0;
    // Recursive setup planning searches under widening power-of-two budgets. Exactness:
    // - Admission under a power-of-two budget `B` is padded setup `<= B`, and
    //   every dominance, bound and level prune keeps a side whose padded
    //   setup is no larger. A rejected candidate can therefore displace only
    //   candidates that are also over `B`. Frontiers are unbounded dominance
    //   sets, split domains are fixed by geometry, and each pass gets a fresh
    //   memo, so the in-budget part of every frontier is the unbudgeted one.
    // - Setup is a maximum over levels, terminal and offloaded children, so
    //   rejecting a partial schedule never rejects a completion within `B`.
    // - A pass that selects `W` returns the unbudgeted winner, which fits
    //   because its padded setup is at most `W`'s. A pass that rejects
    //   nothing is the unbudgeted search. Otherwise the next budget is the
    //   smallest rejected setup, padded; if that overflows, the last pass runs
    //   without a budget. The starting budget changes only the pass count.
    let mut padded_setup_limit = (policy.setup_field_budget.is_none()
        && policy.selection_policy
            == crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenExactProofAndWork)
        .then_some(1);
    let best = loop {
        let pass_budget = padded_setup_limit.or(policy.setup_field_budget);
        let setup_field_budget = pass_budget.map(SetupFieldBudget::new);
        let suffix_ctx = SuffixCtx {
            policy,
            challenge_order: policy.transcript_grinding_order()?,
            diagnostics,
            ring_challenge_config,
            key: PolynomialGroupLayout::singleton(key.final_group.num_vars()),
            setup_field_budget: setup_field_budget.as_ref(),
            root_lookup_key: Some(key),
            root_main_constraint: options.root_main_constraint,
            adaptation_guide: options.adaptation_guide,
            root_source_contract: Some(final_source_contract),
            precommitted_source_contracts,
            level_zero_is_root: true,
            relation_traversal_order: options.relation_traversal_order,
            relation_mode_filter: options.relation_mode_filter,
        };
        let mut memo = ScheduleMemo::new();
        let suffix_started = diagnostics.map(|_| Instant::now());
        let pass_start = diagnostics.map(|diagnostics| diagnostics.search_work());
        let suffix = derive_selected_suffix_schedule(
            &suffix_ctx,
            &mut memo,
            initial_state,
            0,
            super::schedule_params::QuerySearch::Root(query_prefix_count),
        );
        if let (Some(diagnostics), Some(started)) = (diagnostics, suffix_started) {
            diagnostics.add_suffix_dp_time(started.elapsed());
            let (hits, misses) = memo.setup_prefix_cache_diagnostics();
            diagnostics.record_setup_prefix_cache(hits, misses);
        }
        let suffix = suffix?;
        let best = select_complete_candidate(policy, suffix.setup_candidates(), diagnostics)?;
        if let (Some(diagnostics), Some(start)) = (diagnostics, pass_start) {
            diagnostics.record_setup_budget_pass(pass_budget, start);
        }
        // `Some(None)` asks for a final pass without a budget.
        let widened = match (best, padded_setup_limit, setup_field_budget.as_ref()) {
            (None, Some(limit), Some(budget)) => budget.min_rejected().map(|rejected| {
                // This pass's budget is `limit`, so every recorded rejection exceeds it.
                debug_assert!(rejected > limit);
                rejected.checked_next_power_of_two()
            }),
            _ => None,
        };
        match widened {
            Some(next) => padded_setup_limit = next,
            None => break best.cloned(),
        }
    };
    let Some(best) = best else {
        if key.precommitteds.is_empty() {
            return Err(AkitaError::UnsupportedSchedule(format!(
                "no mixed-D schedule in the audited fold domain for num_vars={}, num_polynomials={}",
                key.final_group.num_vars(),
                key.final_group.num_polynomials()
            )));
        }
        let class_count = precommitted_group_equivalence_classes(
            &key.precommitteds,
            precommitted_source_contracts,
        )?
        .len();
        let bounded_note = if u32::try_from(class_count)
            .ok()
            .and_then(|classes| {
                akita_challenges::PRODUCTION_FOLD_CHALLENGE_RING_DIMS
                    .len()
                    .checked_pow(classes)
            })
            .is_none_or(|products| {
                products > crate::schedule_params::MAX_PRECOMMIT_OPENING_PRODUCTS
            }) {
            format!(
                "; {class_count} distinct precommitted classes can exceed the {}-assignment \
                 root opening bound; larger domains retain the smallest next-witness \
                 assignments, so some root openings may not have been searched",
                crate::schedule_params::MAX_PRECOMMIT_OPENING_PRODUCTS
            )
        } else {
            String::new()
        };
        return Err(AkitaError::UnsupportedSchedule(format!(
            "no multi-group schedule in the audited fold domain for num_vars={}{bounded_note}",
            key.final_group.num_vars()
        )));
    };
    let first_direct_setup_field_len = Some(
        best.first_direct_setup_field_len
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "setup-first schedule is missing its first direct setup size".into(),
                )
            })?
            .get(),
    );
    if let Some(diagnostics) = diagnostics {
        let metrics = best.metrics();
        let folds = best.folds.to_vec();
        let root_output_witness_len = folds
            .first()
            .ok_or_else(|| {
                AkitaError::InvalidSetup("selected schedule is missing its root fold".into())
            })?
            .output_witness_len;
        diagnostics.record_selected(
            policy.selection_policy,
            metrics,
            root_output_witness_len,
            folds
                .iter()
                .map(|fold| crate::diagnostics::SelectedFoldDiagnostics {
                    dimensions: fold.params.role_dims(),
                    relation_mode: fold.params.ring_relation_mode,
                })
                .collect(),
        );
    }
    let materialization_started = diagnostics.map(|_| Instant::now());
    let root_layout = key.opening_layout()?;
    let planned = materialize_candidate_schedule(
        CandidateMaterializationCost {
            proof_bytes: best.cost.proof_bytes(),
            grinding: best.cost.grinding_cost(),
            num_setup_field_elements: best.setup_field_elements,
            first_direct_setup_field_len,
        },
        policy,
        &root_layout,
        key.root_accumulated_setup_prefix_len,
        best.folds.to_vec(),
        best.terminal.as_ref().clone(),
    );
    if let (Some(diagnostics), Some(started)) = (diagnostics, materialization_started) {
        diagnostics.add_final_materialization_time(started.elapsed());
    }
    planned
}

#[cfg(all(test, feature = "catalog-gen"))]
#[path = "test/host_scale_coverage.rs"]
mod host_scale_coverage_tests;
#[cfg(test)]
#[path = "test/setup_prefix_challenge_hook.rs"]
mod setup_prefix_challenge_hook_tests;
#[cfg(all(test, feature = "catalog-gen"))]
#[path = "test/planner_totality.rs"]
mod totality_tests;
