//! Feature-gated diagnostics for native Spongefish protocol contexts.

use crate::ProtocolContextRecord;
use std::cell::RefCell;

thread_local! {
    static THREAD_EVENTS: RefCell<Vec<TranscriptEvent>> = const { RefCell::new(Vec::new()) };
    static THREAD_PROOF_RANGES: RefCell<Vec<ProofMessageRange>> = const { RefCell::new(Vec::new()) };
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
    /// Public fixed byte length recorded for the group.
    pub len: usize,
}

pub(crate) fn record_context(record: ProtocolContextRecord) {
    THREAD_EVENTS.with(|events| events.borrow_mut().push(TranscriptEvent::Context(record)));
}

pub(crate) fn record_proof_range(record: ProtocolContextRecord, start: usize) {
    if record.kind != crate::ProtocolMessageKind::ProofAtoms as u32 {
        return;
    }
    let Ok(len) = usize::try_from(record.encoded_bytes) else {
        return;
    };
    THREAD_PROOF_RANGES.with(|ranges| {
        ranges.borrow_mut().push(ProofMessageRange {
            context: record,
            start,
            len,
        });
    });
}

/// Clear native transcript events recorded by the current thread.
pub fn clear_thread_events() {
    THREAD_EVENTS.with(|events| events.borrow_mut().clear());
    THREAD_PROOF_RANGES.with(|ranges| ranges.borrow_mut().clear());
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
