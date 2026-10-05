use super::*;
use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

#[cfg(feature = "catalog-gen")]
#[test]
fn incoming_width_guard_preserves_the_unpruned_complete_winner() {
    use super::super::objective::complete_schedule_score;
    use super::super::unpruned_search::{prepend_fold, prepend_root};
    use super::super::{
        derive_terminal_candidates, derive_unpruned_fold_candidates_for_oracle,
        select_complete_candidate, CandidateFoldChain, PlannerOpeningCandidate,
    };

    // A small root with many live blocks makes eight large, equal witness
    // bodies. The two consumers emit equal-length witnesses, but a wider
    // consumer adds enough producer padding to overturn its local dominance.
    let mut policy = crate::policy::direct_only_policy(policy_of::<OneHot>());
    policy.claim_ext_degree = 1;
    policy.opening_basis_range = (2, 2);
    policy.witness_chunk = akita_params::ChunkedWitnessCfg {
        num_chunks: 8,
        num_activated_levels: 1,
    };
    policy.selective_l2_response_model = crate::SelectiveL2ResponseModelId::Disabled;
    let dimensions = CommitmentRingDims::uniform(64);
    policy.ring_dimension_schedule_mode =
        crate::RingDimensionScheduleMode::UniformDimension { ring_dimension: 64 };
    policy.selection_policy =
        crate::SelectionPolicyId::for_policy(false, policy.ring_dimension_schedule_mode);
    akita_schedules::planner_support::validate_policy(&policy).unwrap();
    let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(16));
    let opening_layout = key.opening_layout().unwrap();
    let opening = PlannerOpeningCandidate::coefficient_packing(0, 1, dimensions, 64)
        .unwrap()
        .unwrap();
    let (producer, witness_len) = crate::planner::exhaustive_root_candidates_for_reference(
        &key,
        OneHot::committed_source_contract().unwrap(),
        &policy,
        dimensions,
        opening,
        policy.inner_basis_range.0,
        2,
    )
    .unwrap()
    .into_iter()
    .find(|(params, _)| {
        params.blocks().positions_per_block == 2
            && params.outer_slice_count() == akita_params::CommitmentSliceCount::ONE
    })
    .unwrap();
    let shape = Arc::new(
        akita_params::WitnessLayout::new(
            &producer,
            &opening_layout,
            &akita_params::RelationWitnessGeometry::for_level(&producer, &opening_layout, 1)
                .unwrap(),
            8,
            akita_params::RelationQuotientPlan::for_field_bits(&producer, 128).unwrap(),
        )
        .unwrap()
        .chunk_shape()
        .unwrap(),
    );
    let request = RecursiveCandidateRequest {
        policy: &policy,
        input_chunks: Some(*shape),
        payload_mode: akita_params::CommitmentPayloadMode::Compressed,
        opening,
        dimensions,
        current_witness_len: witness_len,
        source: crate::InnerBasisSource::BalancedDigits { log_basis: 2 },
        log_basis_inner: 2,
        log_basis_open: 2,
        fold_level: 1,
        source_moment: None,
        relation_traversal_order: RelationTraversalOrder::Canonical,
        guide: None,
    };
    let mut raw =
        derive_unpruned_fold_candidates_for_oracle(request, RelationSearchDomain::QuotientOnly)
            .unwrap();
    raw.retain(|(params, _)| {
        [1024, 4096].contains(&params.blocks().positions_per_block)
            && params.outer_slice_count() == akita_params::CommitmentSliceCount::ONE
    });
    raw.sort_by_key(|(params, _)| params.blocks().positions_per_block);
    assert_eq!(raw.len(), 2);
    assert_eq!(
        raw[0].1, raw[1].1,
        "outgoing lengths must not mask the guard"
    );
    assert!(raw
        .iter()
        .all(|(params, _)| params.witness_chunk.num_chunks == 1));
    let aligned_lengths = raw
        .iter()
        .map(|(params, _)| {
            shape
                .align(
                    akita_params::FoldSuccessor::Recursive(params)
                        .source_block_len()
                        .unwrap(),
                    1,
                )
                .unwrap()
                .0
        })
        .collect::<Vec<_>>();
    assert!(aligned_lengths[0] < aligned_lengths[1]);

    // Hold the continuation fixed, so only the incoming width can change the
    // producer cost. Both consumers have the same basis and outgoing length.
    let next_len = raw[0].1;
    let terminal_params = derive_terminal_candidates(RecursiveCandidateRequest {
        input_chunks: None,
        opening: PlannerOpeningCandidate::evaluation_trace(
            OneHot::ring_challenge_config(64).unwrap(),
        ),
        current_witness_len: next_len,
        fold_level: 2,
        ..request
    })
    .unwrap()
    .into_iter()
    .find(|params| params.blocks().positions_per_block == 2048)
    .unwrap();
    let (terminal, terminal_bytes) = try_terminal_direct_suffix_cost(
        &policy,
        next_len,
        &terminal_params,
        128,
        key.final_group,
        2,
        None,
        None,
        None,
    )
    .unwrap()
    .unwrap();
    let child = ScheduleCandidate {
        first_direct_setup_field_len: NonZeroUsize::new(
            active_setup_field_len(
                &terminal_params,
                &suffix_opening_layout(next_len, None).unwrap(),
            )
            .unwrap(),
        ),
        first_direct_output_witness_len: 0,
        cost: ProofCost::new(terminal_bytes, 0, 0, 0).unwrap(),
        setup_field_elements: terminal_setup_field_elements(&terminal.params).unwrap(),
        folds: CandidateFoldChain::default(),
        terminal: Arc::new(terminal),
    };
    let complete = |params: &CommittedGroupParams| {
        let input_len = shape
            .align(
                akita_params::FoldSuccessor::Recursive(params)
                    .source_block_len()
                    .unwrap(),
                1,
            )
            .unwrap()
            .0;
        let suffix = prepend_fold(&policy, 1, input_len, params, &child)
            .unwrap()
            .unwrap();
        let candidate = prepend_root(&policy, &key, 1 << 16, &producer, &suffix)
            .unwrap()
            .unwrap();
        // The counterexample must be an admitted schedule, not an artificial
        // pair whose costs only look plausible to the local pruner.
        let materialized = super::super::materialize_candidate_schedule(
            super::super::CandidateMaterializationCost {
                proof_bytes: candidate.cost.proof_bytes(),
                grinding: candidate.cost.grinding_cost(),
                num_setup_field_elements: candidate.setup_field_elements,
                first_direct_setup_field_len: None,
            },
            &policy,
            &opening_layout,
            candidate.folds.to_vec(),
            candidate.terminal.as_ref().clone(),
        )
        .unwrap();
        materialized
            .schedule
            .validate_nonterminal_opening_execution(policy.claim_ext_degree)
            .unwrap();
        candidate
    };
    let unpruned = raw
        .iter()
        .map(|(params, _)| complete(params))
        .collect::<Vec<_>>();
    let oracle = select_complete_candidate(&policy, &unpruned, None)
        .unwrap()
        .unwrap();
    let oracle_score = complete_schedule_score(&policy, oracle, None).unwrap();
    assert_eq!(
        oracle.folds.to_vec()[1].params.blocks().positions_per_block,
        1024
    );
    assert_eq!(
        unpruned[0].cost.proof_bytes(),
        unpruned[1].cost.proof_bytes()
    );
    assert!(unpruned[0].cost.exact_score() < unpruned[1].cost.exact_score());

    let ctx = SuffixCtx {
        policy: &policy,
        challenge_order: policy.transcript_grinding_order().unwrap(),
        diagnostics: None,
        ring_challenge_config: &OneHot::ring_challenge_config,
        key: key.final_group,
        setup_field_budget: None,
        root_lookup_key: Some(&key),
        root_main_constraint: None,
        adaptation_guide: None,
        root_source_contract: Some(OneHot::committed_source_contract().unwrap()),
        precommitted_source_contracts: &[],
        level_zero_is_root: true,
        relation_traversal_order: RelationTraversalOrder::Canonical,
        relation_mode_filter: RelationModeFilter::All,
    };
    let state = SuffixState {
        level: 1,
        current_witness_len: witness_len,
        input_chunks: Some(&shape),
        current_lb: 2,
        source_moment: None,
        dimension_ceiling: dimensions,
        topology: SuffixTopology::Direct {
            payload_phase: akita_params::CommitmentPayloadPhase::CompressedPrefix,
            relation_phase: RingRelationPhase::QuotientPrefix,
        },
    };
    let consumer_layout = suffix_opening_layout(witness_len, None).unwrap();
    for reverse in [false, true] {
        let generate = || {
            let mut candidates = raw
                .iter()
                .map(|(params, next_witness_len)| candidates::RawFoldCandidate {
                    params: params.clone(),
                    next_witness_len: *next_witness_len,
                })
                .collect::<Vec<_>>();
            if reverse {
                candidates.reverse();
            }
            attach_source_moments(&ctx, state, false, candidates).unwrap()
        };
        let guarded = prune::level_candidates(&consumer_layout, true, generate()).unwrap();
        let guarded_schedules = guarded
            .iter()
            .map(|candidate| complete(&candidate.params))
            .collect::<Vec<_>>();
        let selected = select_complete_candidate(&policy, &guarded_schedules, None)
            .unwrap()
            .unwrap();
        assert_eq!(selected.cost, oracle.cost);
        assert_eq!(
            complete_schedule_score(&policy, selected, None).unwrap(),
            oracle_score
        );

        // For these single-chunk consumers, false disables only the incoming
        // width conjunct. Requiring one survivor proves every other dominance
        // condition holds, instead of merely exercising different widths.
        let unguarded = prune::level_candidates(&consumer_layout, false, generate()).unwrap();
        assert_eq!(unguarded.len(), 1);
        assert_eq!(unguarded[0].params.blocks().positions_per_block, 4096);
        let wrong = complete(&unguarded[0].params);
        assert!(complete_schedule_score(&policy, &wrong, None).unwrap() > oracle_score);
    }
}

#[test]
fn memo_ownership_is_shared_but_compared_by_value() {
    let shape = akita_params::WitnessChunkShape {
        body_lengths: [64; akita_params::MAX_WITNESS_CHUNKS],
        num_chunks: 2,
        tail_prefix_len: 0,
        tail_alignment: 1,
        tail_len: 5,
        tail_suffix_alignment: 1,
    };
    let policy = policy_of::<OneHot>();
    let shared = Arc::new(shape);
    let state = SuffixState {
        level: 1,
        current_witness_len: 133,
        input_chunks: Some(&shared),
        current_lb: 3,
        source_moment: None,
        dimension_ceiling: CommitmentRingDims::uniform(64),
        topology: SuffixTopology::Direct {
            payload_phase: akita_params::CommitmentPayloadPhase::CompressedPrefix,
            relation_phase: RingRelationPhase::QuotientPrefix,
        },
    };
    let key = state.memo_key(&policy);
    let clone = key.clone();
    assert!(Arc::ptr_eq(
        key.input_chunks.as_ref().unwrap(),
        clone.input_chunks.as_ref().unwrap(),
    ));
    let equal_shape = Arc::new(shape);
    let equal_key = SuffixState {
        input_chunks: Some(&equal_shape),
        ..state
    }
    .memo_key(&policy);
    let mut different_shape = shape;
    different_shape.body_lengths[0] += 1;
    different_shape.body_lengths[1] -= 1;
    let different_shape = Arc::new(different_shape);
    let different_key = SuffixState {
        input_chunks: Some(&different_shape),
        ..state
    }
    .memo_key(&policy);
    let mut memo = ScheduleMemo::new();
    memo.insert(key, empty_suffix_result(), None);
    assert!(memo.get(&equal_key).is_some());
    assert!(memo.get(&different_key).is_none());
    memo.insert(equal_key, empty_suffix_result(), None);
    assert_eq!(memo.len(), 1);
    memo.insert(different_key, empty_suffix_result(), None);
    assert_eq!(memo.len(), 2);
}

#[test]
fn memo_key_has_a_bounded_inline_size() {
    let size = std::mem::size_of::<ScheduleMemoKey>();
    eprintln!(
        "ScheduleMemoKey={size} WitnessChunkShape={}",
        std::mem::size_of::<akita_params::WitnessChunkShape>()
    );
    assert!(
        size <= std::mem::size_of::<Option<crate::response_model::SourceMomentEstimate>>()
            + 16 * std::mem::size_of::<usize>(),
        "memo keys must not embed the ownership body array: {size}"
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn contracting_chunk_search_matches_unpruned_complete_objective() {
    for num_chunks in [2, 8] {
        let mut policy = policy_of::<OneHot>();
        policy.witness_chunk = akita_params::ChunkedWitnessCfg {
            num_chunks,
            num_activated_levels: 1,
        };
        policy.ring_dimension_schedule_mode = crate::RingDimensionScheduleMode::UniformDimension {
            ring_dimension: 256,
        };
        policy.selection_policy = crate::SelectionPolicyId::for_policy(
            policy.recursive_setup_planning,
            policy.ring_dimension_schedule_mode,
        );
        policy.selective_l2_response_model = crate::SelectiveL2ResponseModelId::Disabled;
        policy.inner_basis_range.1 = policy.inner_basis_range.0;
        let key = PolynomialGroupLayout::singleton(20);
        let lookup_key = ScheduleLookupKey::single(key);
        let selected = crate::planner::find_schedule(
            &lookup_key,
            OneHot::committed_source_contract().unwrap(),
            &[],
            &policy,
            OneHot::ring_challenge_config,
        )
        .unwrap();
        assert_eq!(
            selected.schedule.root.params.witness_chunk.num_chunks,
            num_chunks
        );
        assert!(!selected.schedule.recursive_folds.is_empty());
        assert_eq!(
            selected.schedule.recursive_folds[0]
                .params
                .witness_chunk
                .num_chunks,
            1
        );
        assert!(
            selected.schedule.recursive_folds.len()
                < super::super::unpruned_search::MAX_ORACLE_RECURSION_DEPTH
        );
        let unpruned = super::super::unpruned_search::find_schedule(
            key,
            &policy,
            OneHot::committed_source_contract().unwrap(),
            OneHot::ring_challenge_config,
        )
        .unwrap()
        .planned;
        assert_eq!(selected.estimate, unpruned.estimate);
        assert_eq!(
            selected.schedule.canonical_descriptor_bytes(),
            unpruned.schedule.canonical_descriptor_bytes()
        );
    }
}

#[test]
fn contracting_chunk_pruning_preserves_consumer_widths_in_both_orders() {
    for num_chunks in [2, 8] {
        let mut policy = policy_of::<OneHot>();
        policy.witness_chunk = akita_params::ChunkedWitnessCfg {
            num_chunks,
            num_activated_levels: 1,
        };
        policy.ring_dimension_schedule_mode = crate::RingDimensionScheduleMode::UniformDimension {
            ring_dimension: 256,
        };
        policy.selection_policy = crate::SelectionPolicyId::for_policy(
            policy.recursive_setup_planning,
            policy.ring_dimension_schedule_mode,
        );
        policy.selective_l2_response_model = crate::SelectiveL2ResponseModelId::Disabled;
        policy.inner_basis_range.1 = policy.inner_basis_range.0;
        let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(20));
        let planned = crate::planner::find_schedule(
            &key,
            OneHot::committed_source_contract().unwrap(),
            &[],
            &policy,
            OneHot::ring_challenge_config,
        )
        .unwrap();
        let producer = &planned.schedule.root.params;
        let layout = key.opening_layout().unwrap();
        let shape = Arc::new(
            akita_params::WitnessLayout::new(
                producer,
                &layout,
                &akita_params::RelationWitnessGeometry::for_level(
                    producer,
                    &layout,
                    policy.claim_ext_degree,
                )
                .unwrap(),
                num_chunks,
                akita_params::RelationQuotientPlan::for_field_bits(
                    producer,
                    policy.decomposition.field_bits(),
                )
                .unwrap(),
            )
            .unwrap()
            .chunk_shape()
            .unwrap(),
        );
        let ctx = SuffixCtx {
            policy: &policy,
            challenge_order: policy.transcript_grinding_order().unwrap(),
            diagnostics: None,
            ring_challenge_config: &OneHot::ring_challenge_config,
            key: key.final_group,
            setup_field_budget: None,
            root_lookup_key: Some(&key),
            root_main_constraint: None,
            adaptation_guide: None,
            root_source_contract: Some(OneHot::committed_source_contract().unwrap()),
            precommitted_source_contracts: &[],
            level_zero_is_root: true,
            relation_traversal_order: RelationTraversalOrder::Canonical,
            relation_mode_filter: RelationModeFilter::All,
        };
        let state = SuffixState {
            level: 1,
            // Search states carry the producer before successor alignment;
            // the materialized root already contains the selected width's padding.
            current_witness_len: shape.align(1, 1).unwrap().0,
            input_chunks: Some(&shape),
            current_lb: producer.open().digits.log_basis,
            source_moment: None,
            dimension_ceiling: producer.role_dims(),
            topology: SuffixTopology::Direct {
                payload_phase: akita_params::CommitmentPayloadPhase::CompressedPrefix,
                relation_phase: RingRelationPhase::QuotientPrefix,
            },
        };
        let domain = candidates::CandidateDomain::prepare(&ctx, state).unwrap();
        let generate = || {
            let mut memo = ScheduleMemo::new();
            let mut raw = Vec::new();
            for basis in
                policy.opening_basis_range.0.max(state.current_lb)..=policy.opening_basis_range.1
            {
                raw.extend(
                    domain
                        .generate_recursive_for_opening_basis(
                            &ctx,
                            state,
                            basis,
                            &mut memo.setup_prefixes,
                        )
                        .unwrap()
                        .folds,
                );
            }
            attach_source_moments(&ctx, state, false, raw).unwrap()
        };
        let widths = |candidates: &[PlannedFoldCandidate]| {
            candidates
                .iter()
                .map(|candidate| {
                    assert_eq!(candidate.params.witness_chunk.num_chunks, 1);
                    akita_params::FoldSuccessor::Recursive(&candidate.params)
                        .source_block_len()
                        .unwrap()
                })
                .collect::<std::collections::BTreeSet<_>>()
        };
        let candidates = generate();
        let expected_widths = widths(&candidates);
        assert!(
            expected_widths.len() > 1,
            "fixture must exercise distinct consumer widths"
        );
        let aligned_lengths = expected_widths
            .iter()
            .map(|&width| shape.align(width, 1).unwrap().0)
            .collect::<std::collections::BTreeSet<_>>();
        let domains = aligned_lengths
            .iter()
            .map(|len| len.next_power_of_two())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(domains.len() > 1, "fixture must cross a power-of-two boundary: chunks={num_chunks}, widths={expected_widths:?}, lengths={aligned_lengths:?}");
        assert!(
            aligned_lengths.len() > 1,
            "producer padding must depend on consumer width"
        );
        let pruned = prune::level_candidates(&domain.opening_layout, true, candidates).unwrap();
        assert_eq!(widths(&pruned), expected_widths);
        let mut reversed = generate();
        reversed.reverse();
        let reversed = prune::level_candidates(&domain.opening_layout, true, reversed).unwrap();
        let descriptors = |candidates: &[PlannedFoldCandidate]| {
            candidates
                .iter()
                .map(|candidate| candidate.params.canonical_descriptor_bytes())
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_eq!(descriptors(&pruned), descriptors(&reversed));
    }
}
