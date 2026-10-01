//! Test-only switch for prover fault injection (feature `fault-injection`).
//!
//! A fault-injecting prover emits sumcheck messages for a statement that does
//! not hold. The prover's round-consistency self-check would refuse, so the
//! caller can skip it on this thread. Verification is unaffected.

use std::cell::Cell;

thread_local! {
    static SKIP_ROUND_CLAIM_CHECK: Cell<bool> = const { Cell::new(false) };
}

/// Run `f` with the prover-side round-claim self-check skipped on this thread.
pub fn with_round_claim_check_skipped<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SKIP_ROUND_CLAIM_CHECK.set(self.0);
        }
    }
    let _restore = Restore(SKIP_ROUND_CLAIM_CHECK.replace(true));
    f()
}

pub(crate) fn round_claim_check_skipped() -> bool {
    SKIP_ROUND_CLAIM_CHECK.get()
}
