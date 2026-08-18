use super::offloaded_witness_contracts;

fn memo_key(level: usize, incoming_setup_prefix: Option<usize>) -> super::ScheduleMemoKey {
    super::ScheduleMemoKey {
        level,
        current_witness_len: 1024,
        current_lb: 3,
        source_moment: None,
        incoming_setup_prefix,
        d_a: 64,
        d_b: 64,
        d_d: 64,
        payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
    }
}

#[test]
fn suffix_memo_retains_every_completed_state_and_replaces_in_place() {
    let direct = memo_key(1, None);
    let prefixed = memo_key(2, Some(1));
    let mut memo = super::ScheduleMemo::new();
    for key in [direct, prefixed] {
        memo.insert(key, super::empty_suffix_result());
    }
    assert!(memo.contains(&direct));
    assert_eq!(memo.len(), 2);
    assert!(memo.contains(&prefixed));

    memo.insert(direct, super::empty_suffix_result());
    assert_eq!(memo.len(), 2);
    assert!(memo.contains(&direct));
    assert!(memo.contains(&prefixed));
}

#[test]
fn parent_observable_key_ignores_unpriced_successor_opening_details() {
    let policy = akita_config::policy_of::<akita_config::proof_optimized::fp128::Dense>();
    let challenge = akita_challenges::SparseChallengeConfig::production_for_ring_dim(64)
        .expect("D64 challenge");
    let mut evaluation_trace = akita_types::CommittedGroupParams::params_only(
        akita_types::SisModulusProfileId::Q128OffsetA7F7,
        256,
        2,
        2,
        2,
        2,
        challenge,
    );
    evaluation_trace.payload_mode = akita_types::CommitmentPayloadMode::Raw;
    let mut packing = evaluation_trace.clone();
    packing.opening_method = akita_types::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    packing.log_basis_open = 4;
    packing.num_digits_open = 32;
    assert_ne!(
        evaluation_trace.canonical_descriptor_bytes(),
        packing.canonical_descriptor_bytes()
    );
    assert_eq!(
        super::ParentObservableKey::new(&policy, Some(&evaluation_trace)).unwrap(),
        super::ParentObservableKey::new(&policy, Some(&packing)).unwrap(),
        "a parent prices only the successor outer payload and setup-prefix payload"
    );
    assert_eq!(
        akita_schedules::planner_support::nonterminal_level_payload_bytes(
            &policy,
            &evaluation_trace,
            Some(&evaluation_trace),
            1024,
            512,
        )
        .unwrap(),
        akita_schedules::planner_support::nonterminal_level_payload_bytes(
            &policy,
            &evaluation_trace,
            Some(&packing),
            1024,
            512,
        )
        .unwrap(),
        "successors in one parent-observable bucket must price identically"
    );

    let outer = packing.outer_commit_matrix;
    packing.outer_commit_matrix = akita_types::OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank() * 2,
        outer.input_width(),
        outer.coeff_linf_bound(),
        outer.ring_dimension(),
    );
    assert_ne!(
        super::ParentObservableKey::new(&policy, Some(&evaluation_trace)).unwrap(),
        super::ParentObservableKey::new(&policy, Some(&packing)).unwrap(),
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
        incoming_setup_prefix: None,
        dimension_ceiling,
        payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
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
        incoming_setup_prefix: None,
        dimension_ceiling,
        payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
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
