use super::*;
use crate::logging::{
    clear_thread_events, finish_proof_ranges, record_context, record_context_repeated,
    record_proof_boundary, thread_events, thread_proof_ranges, PENDING_PROOF_RANGE,
};
use crate::ProtocolMessageKind;

#[test]
fn snapshots_stream_all_run_shapes_and_survive_clear() {
    for groups in [
        vec![],
        vec![7],
        vec![7; 129],
        vec![0, 1, 2, 4, 5, 5, 5, 6, 7, 3, 3, u32::MAX, 0],
    ] {
        clear_thread_events();
        let mut expected = Vec::new();
        for group in groups {
            let record = crate::logging::tests::challenge(group);
            record_context(record);
            expected.push(TranscriptEvent::Context(record));
        }
        let log = thread_event_log().unwrap();
        assert_eq!(log.len(), expected.len());
        assert_eq!(log.is_empty(), expected.is_empty());
        assert_eq!(log.events().collect::<Vec<_>>(), expected);
        assert_eq!(log.events().collect::<Vec<_>>(), thread_events());
        assert_eq!(
            log.runs()
                .flat_map(|(record, count)| std::iter::repeat_n(
                    TranscriptEvent::Context(record),
                    count
                ))
                .collect::<Vec<_>>(),
            expected
        );
        let mut events = log.events();
        for (index, expected) in expected.iter().enumerate() {
            assert_eq!(events.len(), log.len() - index);
            assert_eq!(events.next(), Some(*expected));
        }
        assert_eq!(events.len(), 0);
        assert_eq!(events.next(), None);
        assert_eq!(events.next(), None);
        clear_thread_events();
        assert_eq!(log.events().collect::<Vec<_>>(), expected);
    }
}

#[test]
fn snapshot_preserves_boundaries_and_every_context_field() {
    clear_thread_events();
    let first = crate::logging::tests::challenge(0);
    record_context_repeated(first, 0);
    let mut expected = Vec::new();
    for field in 0..14 {
        let mut record = first;
        match field {
            0 => record.domain[0] ^= 1,
            1 => record.version += 1,
            2 => record.kind = ProtocolMessageKind::ProofAtoms as u32,
            3 => record.atom_count += 1,
            4 => record.encoded_bytes += 1,
            5 => record.challenge_bytes += 1,
            coordinate => record.site_id[(coordinate - 6) * 4] ^= 1,
        }
        record_context_repeated(record, 3);
        expected.extend(std::iter::repeat_n(TranscriptEvent::Context(record), 3));
    }
    let mut proof = first;
    proof.kind = ProtocolMessageKind::ProofAtoms as u32;
    record_context(proof);
    record_proof_boundary(proof, 0);
    expected.push(TranscriptEvent::Context(proof));
    record_context_repeated(first, 65);
    record_proof_boundary(first, 8);
    expected.extend(std::iter::repeat_n(TranscriptEvent::Context(first), 65));
    record_context(proof);
    record_proof_boundary(proof, 8);
    expected.push(TranscriptEvent::Context(proof));
    let ranges = thread_proof_ranges();
    let pending = PENDING_PROOF_RANGE.with(|pending| *pending.borrow());
    let log = thread_event_log().unwrap();
    assert_eq!(thread_proof_ranges(), ranges);
    assert_eq!(
        PENDING_PROOF_RANGE.with(|pending| *pending.borrow()),
        pending
    );
    finish_proof_ranges(24);
    let ranges = thread_proof_ranges();
    assert_eq!(ranges.len(), 2);
    assert_eq!((ranges[0].start, ranges[0].len), (0, 8));
    assert_eq!((ranges[1].start, ranges[1].len), (8, 16));
    assert_eq!(log.events().collect::<Vec<_>>(), expected);
    assert_eq!(log.events().collect::<Vec<_>>(), thread_events());
    assert_eq!(log.len(), expected.len());
    clear_thread_events();
}

#[test]
fn snapshot_counts_large_runs_without_expansion_and_rejects_overflow() {
    clear_thread_events();
    let record = crate::logging::tests::challenge(0);
    record_context_repeated(record, usize::MAX);
    let log = thread_event_log().unwrap();
    assert_eq!(log.len(), usize::MAX);
    assert_eq!(log.runs().collect::<Vec<_>>(), vec![(record, usize::MAX)]);
    record_context(record);
    assert!(thread_event_log().is_none());
    clear_thread_events();
    THREAD_EVENTS.with(|events| {
        events.borrow_mut().push(ContextRun {
            first: record,
            last_group: u32::MAX,
            repetitions: 1,
        });
    });
    assert_eq!(
        thread_event_log().map(|log| log.len()),
        usize::try_from(1u64 << 32).ok()
    );
    clear_thread_events();
}

#[test]
fn snapshot_equality_uses_logical_events_not_run_partition() {
    let first = crate::logging::tests::challenge(0);
    let joined = TranscriptEventLog {
        runs: vec![ContextRun {
            first,
            last_group: 0,
            repetitions: 2,
        }],
        count: 2,
    };
    let split = TranscriptEventLog {
        runs: vec![ContextRun::new(first), ContextRun::new(first)],
        count: 2,
    };
    assert_eq!(joined, split);
}
