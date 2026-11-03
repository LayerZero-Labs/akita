//! Feature-gated diagnostics for Spongefish protocol contexts.

use crate::{ProtocolContextRecord, ProtocolMessageKind, ProtocolSiteId};
use std::cell::RefCell;

thread_local! {
    static THREAD_EVENTS: RefCell<Vec<ContextRun>> = const { RefCell::new(Vec::new()) };
    static THREAD_PROOF_RANGES: RefCell<Vec<ProofMessageRange>> = const { RefCell::new(Vec::new()) };
    static PENDING_PROOF_RANGE: RefCell<Option<(ProtocolContextRecord, usize)>> = const { RefCell::new(None) };
}

/// One proof channel event recorded for structural diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptEvent {
    /// Fixed-format diagnostic metadata for a message group or challenge.
    Context(ProtocolContextRecord),
}

// Padding checks can draw millions of challenges in one opening, and the
// caller's diagnostic log can span many openings. Keep consecutive challenge
// groups as lossless runs; expand only when a caller requests the event list.
struct ContextRun {
    first: ProtocolContextRecord,
    last_group: u32,
}

impl ContextRun {
    fn new(first: ProtocolContextRecord) -> Self {
        Self {
            last_group: ProtocolSiteId::from_bytes(first.site_id).group,
            first,
        }
    }

    fn extend(&mut self, record: ProtocolContextRecord) -> bool {
        if self.first.kind != ProtocolMessageKind::Challenge as u32 {
            return false;
        }
        let Some(group) = self.last_group.checked_add(1) else {
            return false;
        };
        let mut site = ProtocolSiteId::from_bytes(self.first.site_id);
        site.group = group;
        let expected = ProtocolContextRecord {
            site_id: site.to_bytes(),
            ..self.first
        };
        if record != expected {
            return false;
        }
        self.last_group = group;
        true
    }

    fn events(&self) -> impl Iterator<Item = TranscriptEvent> + '_ {
        let mut site = ProtocolSiteId::from_bytes(self.first.site_id);
        (site.group..=self.last_group).map(move |group| {
            site.group = group;
            TranscriptEvent::Context(ProtocolContextRecord {
                site_id: site.to_bytes(),
                ..self.first
            })
        })
    }
}

/// Prover-observed location of one fixed-shape proof-message group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofMessageRange {
    /// Diagnostic identity of the proof group.
    pub context: ProtocolContextRecord,
    /// Byte offset in the Spongefish argument before the group is emitted.
    pub start: usize,
    /// Actual number of argument bytes emitted for the group.
    pub len: usize,
}

pub(crate) fn record_context(record: ProtocolContextRecord) {
    THREAD_EVENTS.with(|events| {
        let mut events = events.borrow_mut();
        if !events.last_mut().is_some_and(|run| run.extend(record)) {
            events.push(ContextRun::new(record));
        }
    });
}

fn finish_pending(end: usize) {
    let pending = PENDING_PROOF_RANGE.with(|pending| pending.borrow_mut().take());
    let Some((context, start)) = pending else {
        return;
    };
    let Some(len) = end.checked_sub(start) else {
        return;
    };
    THREAD_PROOF_RANGES.with(|ranges| {
        ranges.borrow_mut().push(ProofMessageRange {
            context,
            start,
            len,
        });
    });
}

pub(crate) fn record_proof_boundary(record: ProtocolContextRecord, start: usize) {
    finish_pending(start);
    if matches!(
        record.kind,
        kind if kind == crate::ProtocolMessageKind::ProofLength as u32
            || kind == crate::ProtocolMessageKind::ProofAtoms as u32
            || kind == crate::ProtocolMessageKind::GrindingNonce as u32
            || kind == crate::ProtocolMessageKind::FoldResponseNonce as u32
    ) {
        PENDING_PROOF_RANGE.with(|pending| *pending.borrow_mut() = Some((record, start)));
    }
}

pub(crate) fn finish_proof_ranges(end: usize) {
    finish_pending(end);
}

/// Clear proof channel events recorded by the current thread.
pub fn clear_thread_events() {
    THREAD_EVENTS.with(|events| events.borrow_mut().clear());
    THREAD_PROOF_RANGES.with(|ranges| ranges.borrow_mut().clear());
    PENDING_PROOF_RANGE.with(|pending| *pending.borrow_mut() = None);
}

/// Clone proof channel events recorded by the current thread.
#[must_use]
pub fn thread_events() -> Vec<TranscriptEvent> {
    THREAD_EVENTS.with(|events| {
        events
            .borrow()
            .iter()
            .flat_map(ContextRun::events)
            .collect()
    })
}

/// Clone fixed-shape prover proof-message ranges recorded by the current thread.
#[must_use]
pub fn thread_proof_ranges() -> Vec<ProofMessageRange> {
    THREAD_PROOF_RANGES.with(|ranges| ranges.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{new_prover_channel, new_verifier_channel, prover_context, verifier_context};

    fn challenge(group: u32) -> ProtocolContextRecord {
        ProtocolContextRecord::new(
            ProtocolSiteId {
                family: 3,
                group,
                ..ProtocolSiteId::default()
            }
            .to_bytes(),
            ProtocolMessageKind::Challenge as u32,
            0,
            0,
            16,
        )
    }

    #[test]
    fn million_consecutive_challenges_retain_one_run() {
        clear_thread_events();
        for group in 0..1_000_000 {
            record_context(challenge(group));
        }
        THREAD_EVENTS.with(|events| {
            let events = events.borrow();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].first, challenge(0));
            assert_eq!(events[0].last_group, 999_999);
            assert!(events.capacity() < 16);
        });
        clear_thread_events();
    }

    #[test]
    fn expanded_events_preserve_order_and_broken_runs() {
        clear_thread_events();
        let mut other_kind = challenge(7);
        other_kind.kind = ProtocolMessageKind::PublicValue as u32;
        let records = [
            challenge(3),
            challenge(4),
            challenge(6),
            other_kind,
            challenge(8),
            challenge(9),
            challenge(2),
            challenge(3),
            challenge(u32::MAX - 1),
            challenge(u32::MAX),
            challenge(0),
            challenge(1),
        ];
        for record in records {
            record_context(record);
        }
        assert_eq!(
            thread_events(),
            records.map(TranscriptEvent::Context).to_vec()
        );
        THREAD_EVENTS.with(|events| assert_eq!(events.borrow().len(), 7));
        clear_thread_events();
    }

    #[test]
    fn runs_compare_every_context_field_and_site_coordinate() {
        for field in 0..14 {
            clear_thread_events();
            let first = challenge(10);
            let mut second = challenge(11);
            match field {
                0 => second.domain[0] ^= 1,
                1 => second.version += 1,
                2 => second.kind = ProtocolMessageKind::ProofAtoms as u32,
                3 => second.atom_count += 1,
                4 => second.encoded_bytes += 1,
                5 => second.challenge_bytes += 1,
                site_coordinate => {
                    // Include group itself: a skipped group must start a run.
                    let offset = (site_coordinate - 6) * 4;
                    second.site_id[offset] ^= 1;
                }
            }
            record_context(first);
            record_context(second);
            assert_eq!(
                thread_events(),
                vec![
                    TranscriptEvent::Context(first),
                    TranscriptEvent::Context(second)
                ]
            );
            THREAD_EVENTS.with(|events| assert_eq!(events.borrow().len(), 2));
        }
        clear_thread_events();
    }

    #[test]
    fn nonchallenge_groups_stay_separate_and_clear_resets_all_diagnostics() {
        clear_thread_events();
        for group in 0..3 {
            let mut record = challenge(group);
            record.kind = ProtocolMessageKind::ProofAtoms as u32;
            record_context(record);
            record_proof_boundary(record, group as usize * 16);
        }
        finish_proof_ranges(48);
        THREAD_EVENTS.with(|events| assert_eq!(events.borrow().len(), 3));
        let ranges = thread_proof_ranges();
        assert_eq!(ranges.len(), 3);
        for (group, range) in ranges.iter().enumerate() {
            assert_eq!(range.start, group * 16);
            assert_eq!(range.len, 16);
        }
        let mut pending = challenge(9);
        pending.kind = ProtocolMessageKind::ProofAtoms as u32;
        record_proof_boundary(pending, 48);
        PENDING_PROOF_RANGE.with(|pending| assert!(pending.borrow().is_some()));
        clear_thread_events();
        assert!(thread_events().is_empty());
        assert!(thread_proof_ranges().is_empty());
        PENDING_PROOF_RANGE.with(|pending| assert!(pending.borrow().is_none()));
        record_context(challenge(12));
        assert_eq!(
            thread_events(),
            vec![TranscriptEvent::Context(challenge(12))]
        );
        clear_thread_events();
    }

    #[test]
    fn prover_and_verifier_record_identical_proof_stream_contexts() {
        let record = ProtocolContextRecord::new([7; 32], 3, 2, 64, 32);

        clear_thread_events();
        let mut prover = new_prover_channel(b"logging");
        prover_context(&mut prover, record);
        let prover_events = thread_events();

        clear_thread_events();
        let mut verifier = new_verifier_channel(b"logging", &[]);
        verifier_context(&mut verifier, record);
        assert_eq!(thread_events(), prover_events);
        assert_eq!(prover_events, vec![TranscriptEvent::Context(record)]);
    }
}
