use super::*;
use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

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
