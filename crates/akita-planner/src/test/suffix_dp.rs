use super::{
    offloaded_witness_contracts, SuffixResult, SuffixSearchCache, SuffixState,
    MAX_SUFFIX_SEARCH_CACHE_ENTRIES,
};
use std::{collections::BTreeMap, sync::Arc};

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

#[test]
fn suffix_cache_evicts_without_conflating_exact_states() {
    let result = Arc::new(SuffixResult {
        best_by_first_direct_setup_per_lb: BTreeMap::new(),
        best_by_payload_per_lb: BTreeMap::new(),
    });
    let mut memo = SuffixSearchCache::new();
    for witness_len in 0..=MAX_SUFFIX_SEARCH_CACHE_ENTRIES {
        memo.insert(
            SuffixState {
                level: 1,
                current_witness_len: witness_len,
                current_lb: 3,
                incoming_setup_prefix: None,
                payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
            },
            &result,
        );
    }

    assert_eq!(memo.entries.len(), MAX_SUFFIX_SEARCH_CACHE_ENTRIES);
    assert!(memo
        .get(&SuffixState {
            level: 1,
            current_witness_len: 0,
            current_lb: 3,
            incoming_setup_prefix: None,
            payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
        })
        .is_none());
    assert!(memo
        .get(&SuffixState {
            level: 1,
            current_witness_len: MAX_SUFFIX_SEARCH_CACHE_ENTRIES,
            current_lb: 3,
            incoming_setup_prefix: None,
            payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
        })
        .is_some());

    let exact_none = SuffixState {
        level: 2,
        current_witness_len: 7,
        current_lb: 4,
        incoming_setup_prefix: None,
        payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
    };
    let exact_zero = SuffixState {
        incoming_setup_prefix: Some(0),
        ..exact_none
    };
    memo.insert(exact_none, &result);
    memo.insert(exact_zero, &result);
    assert!(memo.get(&exact_none).is_some());
    assert!(memo.get(&exact_zero).is_some());
}
