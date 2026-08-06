use super::{offloaded_witness_contracts, ScheduleMemo, SuffixResult, MAX_SCHEDULE_MEMO_ENTRIES};
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
fn schedule_memo_evicts_oldest_entry_at_capacity() {
    let result = Arc::new(SuffixResult {
        best_by_first_direct_setup_per_lb: BTreeMap::new(),
        best_by_payload_per_lb: BTreeMap::new(),
    });
    let mut memo = ScheduleMemo::new();
    for witness_len in 0..=MAX_SCHEDULE_MEMO_ENTRIES {
        memo.insert(
            (
                1,
                witness_len,
                3,
                0,
                akita_types::CommitmentPayloadPhase::CompressedPrefix,
            ),
            &result,
        );
    }

    assert_eq!(memo.entries.len(), MAX_SCHEDULE_MEMO_ENTRIES);
    assert!(memo
        .get(&(
            1,
            0,
            3,
            0,
            akita_types::CommitmentPayloadPhase::CompressedPrefix,
        ))
        .is_none());
    assert!(memo
        .get(&(
            1,
            MAX_SCHEDULE_MEMO_ENTRIES,
            3,
            0,
            akita_types::CommitmentPayloadPhase::CompressedPrefix,
        ))
        .is_some());
}
