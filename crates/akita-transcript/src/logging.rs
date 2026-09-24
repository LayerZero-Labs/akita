//! Feature-gated diagnostics for native Spongefish protocol contexts.

use crate::ProtocolContextRecord;
use std::cell::RefCell;

thread_local! {
    static THREAD_EVENTS: RefCell<Vec<TranscriptEvent>> = const { RefCell::new(Vec::new()) };
    static THREAD_PROOF_RANGES: RefCell<Vec<ProofMessageRange>> = const { RefCell::new(Vec::new()) };
    static PENDING_PROOF_RANGE: RefCell<Option<(ProtocolContextRecord, usize)>> = const { RefCell::new(None) };
}

/// One native transcript event recorded for structural diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptEvent {
    /// Fixed-format diagnostic metadata for a message group or challenge.
    Context(ProtocolContextRecord),
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
    THREAD_EVENTS.with(|events| events.borrow_mut().push(TranscriptEvent::Context(record)));
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

/// Clear native transcript events recorded by the current thread.
pub fn clear_thread_events() {
    THREAD_EVENTS.with(|events| events.borrow_mut().clear());
    THREAD_PROOF_RANGES.with(|ranges| ranges.borrow_mut().clear());
    PENDING_PROOF_RANGE.with(|pending| *pending.borrow_mut() = None);
}

/// Clone native transcript events recorded by the current thread.
#[must_use]
pub fn thread_events() -> Vec<TranscriptEvent> {
    THREAD_EVENTS.with(|events| events.borrow().clone())
}

/// Clone fixed-shape prover proof-message ranges recorded by the current thread.
#[must_use]
pub fn thread_proof_ranges() -> Vec<ProofMessageRange> {
    THREAD_PROOF_RANGES.with(|ranges| ranges.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{new_native_prover, new_native_verifier, prover_context, verifier_context};

    #[test]
    fn prover_and_verifier_record_identical_native_contexts() {
        let record = ProtocolContextRecord::new([7; 32], 3, 2, 64, 32);

        clear_thread_events();
        let mut prover = new_native_prover(b"logging", b"instance").unwrap();
        prover_context(&mut prover, record);
        let prover_events = thread_events();

        clear_thread_events();
        let mut verifier = new_native_verifier(b"logging", b"instance", &[]).unwrap();
        verifier_context(&mut verifier, record);
        assert_eq!(thread_events(), prover_events);
        assert_eq!(prover_events, vec![TranscriptEvent::Context(record)]);
    }
}
