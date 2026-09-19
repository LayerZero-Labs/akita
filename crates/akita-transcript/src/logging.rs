//! Feature-gated diagnostics for native Spongefish protocol contexts.

use crate::ProtocolContextRecord;
use std::cell::RefCell;

thread_local! {
    static THREAD_EVENTS: RefCell<Vec<TranscriptEvent>> = const { RefCell::new(Vec::new()) };
}

/// One native transcript event recorded for structural diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptEvent {
    /// A fixed-format public context record absorbed before a message group or challenge.
    Context(ProtocolContextRecord),
}

pub(crate) fn record_context(record: ProtocolContextRecord) {
    THREAD_EVENTS.with(|events| events.borrow_mut().push(TranscriptEvent::Context(record)));
}

/// Clear native transcript events recorded by the current thread.
pub fn clear_thread_events() {
    THREAD_EVENTS.with(|events| events.borrow_mut().clear());
}

/// Clone native transcript events recorded by the current thread.
#[must_use]
pub fn thread_events() -> Vec<TranscriptEvent> {
    THREAD_EVENTS.with(|events| events.borrow().clone())
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
