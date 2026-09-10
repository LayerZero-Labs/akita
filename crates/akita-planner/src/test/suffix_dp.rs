use super::offloaded_witness_contracts;
use std::collections::VecDeque;

fn memo_key(level: usize, incoming_setup_prefix: Option<usize>) -> super::ScheduleMemoKey {
    let topology = incoming_setup_prefix.map_or(
        super::SuffixTopology::Direct {
            payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
            relation_phase: super::RingRelationPhase::QuotientPrefix,
        },
        |natural_len| super::SuffixTopology::SetupPrefixed { natural_len },
    );
    super::ScheduleMemoKey {
        level,
        current_witness_len: 1024,
        current_lb: 3,
        source_moment: None,
        d_a: 64,
        d_b: 64,
        d_d: 64,
        topology,
    }
}

#[test]
fn suffix_memo_retains_every_completed_state_and_replaces_in_place() {
    let direct = memo_key(1, None);
    let prefixed = memo_key(2, Some(1));
    let mut memo = super::ScheduleMemo::new();
    for key in [direct, prefixed] {
        memo.insert(key, super::empty_suffix_result(), None);
    }
    assert!(memo.contains(&direct));
    assert_eq!(memo.len(), 2);
    assert!(memo.contains(&prefixed));

    memo.insert(direct, super::empty_suffix_result(), None);
    assert_eq!(memo.len(), 2);
    assert!(memo.contains(&direct));
    assert!(memo.contains(&prefixed));
}

#[test]
fn relation_transition_authority_is_monotone_and_part_of_the_memo_identity() {
    use akita_types::RingRelationMode::{QuotientLift, ReducedEvaluation};

    let prefix = super::RingRelationPhase::QuotientPrefix;
    let reduced = super::RingRelationPhase::ReducedEvaluationSuffix;
    let direct_trace = super::RelationCandidateTopology::DirectEvaluationTrace;
    assert_eq!(prefix.candidate_modes(1, direct_trace), &[QuotientLift]);
    assert_eq!(
        prefix.candidate_modes(2, direct_trace),
        &[QuotientLift, ReducedEvaluation]
    );
    assert_eq!(
        reduced.candidate_modes(2, direct_trace),
        &[ReducedEvaluation]
    );
    assert_eq!(prefix.after(ReducedEvaluation), reduced);
    assert!(reduced
        .candidate_modes(
            2,
            super::RelationCandidateTopology::SetupPrefixedEvaluationTrace
        )
        .is_empty());

    let quotient_key = memo_key(2, None);
    let mut reduced_key = quotient_key;
    reduced_key.topology = super::SuffixTopology::Direct {
        payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
        relation_phase: reduced,
    };
    assert_ne!(quotient_key, reduced_key);
}

#[test]
fn suffix_cache_gives_referenced_entry_a_second_chance() {
    let hot = memo_key(1, None);
    let cold = memo_key(2, None);
    let mut entries = std::collections::HashMap::from([
        (
            hot,
            super::MemoEntry {
                result: super::empty_suffix_result(),
                referenced: true,
            },
        ),
        (
            cold,
            super::MemoEntry {
                result: super::empty_suffix_result(),
                referenced: false,
            },
        ),
    ]);
    let mut insertion_order = VecDeque::from([hot, cold]);

    super::evict_suffix_entry(&mut entries, &mut insertion_order);

    assert!(entries.contains_key(&hot));
    assert!(!entries.contains_key(&cold));
    assert_eq!(insertion_order, VecDeque::from([hot]));
}

#[test]
fn parent_observable_key_tracks_grinding_successor_geometry() {
    let policy = akita_config::policy_of::<akita_config::proof_optimized::fp128::Dense>();
    let challenge = akita_challenges::SparseChallengeConfig::production_for_ring_dim(64)
        .expect("D64 challenge");
    let mut shell = akita_types::CommittedGroupParams::params_only(
        akita_types::SisModulusProfileId::Q128OffsetA7F7,
        256,
        2,
        2,
        2,
        2,
        challenge,
    );
    shell.payload_mode = akita_types::CommitmentPayloadMode::Raw;
    let evaluation_trace = shell.with_decomp(8, 64, 2, 2, 2).unwrap();
    let mut wider_opening = shell.with_decomp(8, 128, 2, 2, 2).unwrap();
    assert_ne!(
        evaluation_trace.canonical_descriptor_bytes(),
        wider_opening.canonical_descriptor_bytes()
    );
    assert_ne!(
        evaluation_trace.recursive_opening_num_vars().unwrap(),
        wider_opening.recursive_opening_num_vars().unwrap()
    );
    assert_ne!(
        super::ParentObservableKey::new(&policy, Some(&evaluation_trace), None).unwrap(),
        super::ParentObservableKey::new(&policy, Some(&wider_opening), None).unwrap(),
        "a parent grinding edge prices the successor opening width"
    );
    let opening_layout = super::suffix_opening_layout(1024, None).unwrap();
    assert_eq!(
        akita_schedules::planner_support::nonterminal_level_payload_bytes(
            &policy,
            &evaluation_trace,
            &opening_layout,
            akita_types::FoldSuccessor::Recursive(&evaluation_trace),
            512,
        )
        .unwrap(),
        akita_schedules::planner_support::nonterminal_level_payload_bytes(
            &policy,
            &evaluation_trace,
            &opening_layout,
            akita_types::FoldSuccessor::Recursive(&wider_opening),
            512,
        )
        .unwrap(),
        "successors in one parent-observable bucket must price identically"
    );

    let mut reduced_successor = evaluation_trace.clone();
    reduced_successor.ring_relation_mode = akita_types::RingRelationMode::ReducedEvaluation;
    assert_ne!(
        evaluation_trace.canonical_descriptor_bytes(),
        reduced_successor.canonical_descriptor_bytes()
    );
    assert_eq!(
        evaluation_trace.recursive_opening_num_vars().unwrap(),
        reduced_successor.recursive_opening_num_vars().unwrap()
    );
    assert_eq!(
        super::ParentObservableKey::new(&policy, Some(&evaluation_trace), None).unwrap(),
        super::ParentObservableKey::new(&policy, Some(&reduced_successor), None).unwrap(),
        "relation details invisible to the parent must share one successor class"
    );

    let mut descriptor_distinct = evaluation_trace.clone();
    let inner = descriptor_distinct.inner().matrix;
    descriptor_distinct.own_group_mut().profile.inner.matrix =
        akita_types::InnerCommitMatrixParams::new_unchecked(
            inner.security_policy(),
            inner
                .sis_table_key()
                .expect("test inner matrix has a SIS table key")
                .table_digest,
            inner.sis_modulus_profile(),
            inner.output_rank() * 2,
            inner.input_width(),
            inner
                .coeff_linf_bound()
                .expect("test inner matrix has a coefficient bound"),
            inner.ring_dimension(),
        );
    assert_ne!(
        evaluation_trace.canonical_descriptor_bytes(),
        descriptor_distinct.canonical_descriptor_bytes(),
        "the test requires descriptor-distinct successors"
    );
    assert_eq!(
        super::ParentObservableKey::new(&policy, Some(&evaluation_trace), None).unwrap(),
        super::ParentObservableKey::new(&policy, Some(&descriptor_distinct), None).unwrap(),
        "successor details invisible to the parent must share one class"
    );
    let layout = akita_types::OpeningClaimsLayout::new(10, 1).unwrap();
    let grinding_cost = |successor| {
        let successor = akita_types::FoldSuccessor::Recursive(successor);
        let relation_geometry = evaluation_trace
            .relation_address_geometry(
                &layout,
                policy.claim_ext_degree,
                successor.ring_dimension(),
                512,
            )
            .unwrap();
        akita_types::transcript_grinding_cost_for_planner_edge(
            &evaluation_trace,
            relation_geometry,
            &layout,
            successor,
            policy.decomposition.field_bits(),
            policy.claim_ext_degree,
            1,
        )
        .unwrap()
    };
    assert_eq!(
        grinding_cost(&evaluation_trace),
        grinding_cost(&descriptor_distinct),
        "one parent-observable successor class must have one grinding price"
    );
    assert_eq!(
        grinding_cost(&evaluation_trace),
        grinding_cost(&reduced_successor),
        "relation details invisible to the parent must not change grinding price"
    );

    let outer = wider_opening.outer().matrix;
    wider_opening.own_group_mut().profile.outer.matrix =
        akita_types::OuterCommitMatrixParams::new_unchecked(
            outer.security_policy(),
            outer.sis_table_key().table_digest,
            outer.sis_modulus_profile(),
            outer.output_rank() * 2,
            outer.input_width(),
            outer.coeff_linf_bound(),
            outer.ring_dimension(),
        );
    assert_ne!(
        super::ParentObservableKey::new(&policy, Some(&evaluation_trace), None).unwrap(),
        super::ParentObservableKey::new(&policy, Some(&wider_opening), None).unwrap(),
        "changing the transmitted successor payload must change the parent key"
    );
}

#[test]
fn terminal_seed_requires_a_scalar_state_without_setup_prefix() {
    assert!(super::state_allows_terminal_seed(false, false));
    assert!(!super::state_allows_terminal_seed(true, false));
    assert!(!super::state_allows_terminal_seed(false, true));
    assert!(!super::state_allows_terminal_seed(true, true));
}

#[test]
fn guided_early_pruning_is_limited_to_complete_roots() {
    assert!(matches!(
        super::GuideScope::for_state(true),
        Some(super::GuideScope::CompleteRoot)
    ));
    assert!(super::GuideScope::for_state(false).is_none());
}

#[test]
fn memo_key_discards_dimension_history_after_adaptive_cutoff() {
    let mut policy = akita_config::policy_of::<akita_config::proof_optimized::fp128::OneHot>();
    let crate::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels, ..
    } = policy.ring_dimension_schedule_mode
    else {
        panic!("test preset must be adaptive");
    };
    let state = |level, dimension_ceiling| super::SuffixState {
        level,
        current_witness_len: 1024,
        current_lb: 3,
        source_moment: None,
        dimension_ceiling,
        topology: super::SuffixTopology::Direct {
            payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
            relation_phase: super::RingRelationPhase::QuotientPrefix,
        },
    };
    let d64 = akita_types::CommitmentRingDims::uniform(64);
    let d256 = akita_types::CommitmentRingDims::uniform(256);

    assert_ne!(
        state(num_search_levels - 1, d64).memo_key(&policy),
        state(num_search_levels - 1, d256).memo_key(&policy),
        "dimension ceilings remain semantically active during adaptive search"
    );
    assert_eq!(
        state(num_search_levels, d64).memo_key(&policy),
        state(num_search_levels, d256).memo_key(&policy),
        "uniform suffix states must not retain dead dimension history"
    );

    policy.ring_dimension_schedule_mode =
        crate::RingDimensionScheduleMode::UniformDimension { ring_dimension: 64 };
    assert_ne!(
        state(num_search_levels, d64).memo_key(&policy),
        state(num_search_levels, d256).memo_key(&policy),
        "uniform-mode keys retain the explicit caller ceiling"
    );
}

#[test]
fn fp32_suffix_memo_key_retains_only_the_effective_transition_ceiling() {
    let policy = akita_config::policy_of::<akita_config::proof_optimized::fp32::OneHot>();
    let crate::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels, ..
    } = policy.ring_dimension_schedule_mode
    else {
        panic!("test preset must be adaptive");
    };
    let state = |dimension_ceiling| super::SuffixState {
        level: num_search_levels,
        current_witness_len: 1024,
        current_lb: 3,
        source_moment: None,
        dimension_ceiling,
        topology: super::SuffixTopology::Direct {
            payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
            relation_phase: super::RingRelationPhase::QuotientPrefix,
        },
    };

    assert_eq!(
        state(akita_types::CommitmentRingDims::uniform(128)).memo_key(&policy),
        state(akita_types::CommitmentRingDims::uniform(256)).memo_key(&policy),
        "D128 and larger ceilings admit the same fp32 suffix domain"
    );
    assert_ne!(
        state(akita_types::CommitmentRingDims::uniform(64)).memo_key(&policy),
        state(akita_types::CommitmentRingDims::uniform(128)).memo_key(&policy),
        "a D64 transition must prevent suffix states from rising back to D128"
    );
}

#[test]
fn offloaded_contraction_accepts_exact_threefold_boundary() {
    assert!(offloaded_witness_contracts(300, 2, 0, 128, 100, 2, 3).unwrap());
    assert!(!offloaded_witness_contracts(299, 2, 0, 128, 100, 2, 3).unwrap());
    assert!(!offloaded_witness_contracts(300, 2, 0, 128, 100, 2, 4).unwrap());
}

#[test]
fn offloaded_contraction_prices_changed_digit_basis() {
    assert!(offloaded_witness_contracts(900, 2, 0, 128, 100, 6, 3).unwrap());
    assert!(!offloaded_witness_contracts(899, 2, 0, 128, 100, 6, 3).unwrap());
}

#[test]
fn offloaded_contraction_includes_full_field_setup_prefix() {
    assert!(offloaded_witness_contracts(100, 2, 100, 128, 1000, 4, 3).unwrap());
    assert!(!offloaded_witness_contracts(100, 2, 90, 128, 1000, 4, 3).unwrap());
}
