use super::{dominates_mixed_score, offloaded_witness_contracts};
use crate::schedule_params::MixedScore;
use std::collections::VecDeque;

#[test]
fn suffix_cache_gives_referenced_entry_a_second_chance() {
    let key = |level| super::ScheduleMemoKey {
        level,
        current_witness_len: 1024,
        current_lb: 3,
        source_moment: None,
        incoming_setup_prefix: None,
        d_a: 64,
        d_b: 64,
        d_d: 64,
        payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
    };
    let hot = key(1);
    let cold = key(2);
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
fn full_suffix_cache_evicts_within_the_inserted_state_class() {
    assert!(super::eviction_uses_direct_queue(true, true, true));
    assert!(!super::eviction_uses_direct_queue(false, true, true));
    assert!(!super::eviction_uses_direct_queue(true, false, true));
    assert!(super::eviction_uses_direct_queue(false, true, false));
}

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
fn suffix_cache_preserves_capacity_and_class_local_second_chance() {
    let direct_hot = memo_key(1, None);
    let direct_cold = memo_key(2, None);
    let prefixed_cold = memo_key(3, Some(1));
    let prefixed_other = memo_key(4, Some(1));
    let mut memo = super::ScheduleMemo::with_capacity(4);
    for key in [direct_hot, direct_cold, prefixed_cold, prefixed_other] {
        memo.insert(key, super::empty_suffix_result());
        assert!(memo.internal_invariants_hold());
    }

    memo.get(&direct_hot).expect("mark direct entry as hot");
    let new_direct = memo_key(5, None);
    memo.insert(new_direct, super::empty_suffix_result());
    assert!(memo.contains(&direct_hot));
    assert!(!memo.contains(&direct_cold));
    assert!(memo.contains(&prefixed_cold));
    assert!(memo.contains(&prefixed_other));
    assert_eq!(memo.queue_lengths(), (2, 2));
    assert_eq!(memo.len(), 4);
    assert!(memo.internal_invariants_hold());

    let new_prefixed = memo_key(6, Some(2));
    memo.insert(new_prefixed, super::empty_suffix_result());
    assert!(memo.contains(&direct_hot));
    assert!(memo.contains(&new_direct));
    assert!(!memo.contains(&prefixed_cold));
    assert!(memo.contains(&prefixed_other));
    assert!(memo.contains(&new_prefixed));
    assert_eq!(memo.queue_lengths(), (2, 2));
    assert_eq!(memo.len(), 4);
    assert!(memo.internal_invariants_hold());

    memo.insert(new_prefixed, super::empty_suffix_result());
    assert_eq!(memo.queue_lengths(), (2, 2));
    assert_eq!(memo.len(), 4);
    assert!(memo.internal_invariants_hold());
}

#[test]
fn suffix_cache_falls_back_only_when_inserted_class_is_empty() {
    let prefixed_one = memo_key(1, Some(1));
    let prefixed_two = memo_key(2, Some(1));
    let direct = memo_key(3, None);
    let mut memo = super::ScheduleMemo::with_capacity(2);
    memo.insert(prefixed_one, super::empty_suffix_result());
    memo.insert(prefixed_two, super::empty_suffix_result());
    assert!(memo.internal_invariants_hold());

    memo.insert(direct, super::empty_suffix_result());
    assert!(memo.contains(&direct));
    assert!(!memo.contains(&prefixed_one));
    assert!(memo.contains(&prefixed_two));
    assert_eq!(memo.queue_lengths(), (1, 1));
    assert_eq!(memo.len(), 2);
    assert!(memo.internal_invariants_hold());

    let direct_one = memo_key(4, None);
    let direct_two = memo_key(5, None);
    let prefixed = memo_key(6, Some(2));
    let mut memo = super::ScheduleMemo::with_capacity(2);
    memo.insert(direct_one, super::empty_suffix_result());
    memo.insert(direct_two, super::empty_suffix_result());
    memo.insert(prefixed, super::empty_suffix_result());
    assert!(memo.contains(&prefixed));
    assert!(!memo.contains(&direct_one));
    assert!(memo.contains(&direct_two));
    assert_eq!(memo.queue_lengths(), (1, 1));
    assert_eq!(memo.len(), 2);
    assert!(memo.internal_invariants_hold());
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
fn mixed_frontier_keeps_lower_payload_child_until_parent_masks_setup() {
    let lower_setup = MixedScore {
        setup_field_elements: 10,
        proof_bytes: 20,
    };
    let lower_payload = MixedScore {
        setup_field_elements: 15,
        proof_bytes: 10,
    };
    assert!(!dominates_mixed_score(lower_setup, lower_payload));
    assert!(!dominates_mixed_score(lower_payload, lower_setup));

    let parent_setup = 20;
    let lower_setup_complete = MixedScore {
        setup_field_elements: parent_setup.max(lower_setup.setup_field_elements),
        proof_bytes: lower_setup.proof_bytes,
    };
    let lower_payload_complete = MixedScore {
        setup_field_elements: parent_setup.max(lower_payload.setup_field_elements),
        proof_bytes: lower_payload.proof_bytes,
    };
    assert!(lower_payload_complete < lower_setup_complete);
}

#[test]
fn mixed_frontier_keeps_equal_payload_alternatives_for_descriptor_ties() {
    let lower_setup = MixedScore {
        setup_field_elements: 10,
        proof_bytes: 20,
    };
    let higher_setup = MixedScore {
        setup_field_elements: 15,
        proof_bytes: 20,
    };

    assert!(!dominates_mixed_score(lower_setup, higher_setup));
    assert!(!dominates_mixed_score(higher_setup, lower_setup));

    let parent_setup = 20;
    assert_eq!(
        parent_setup.max(lower_setup.setup_field_elements),
        parent_setup.max(higher_setup.setup_field_elements)
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
