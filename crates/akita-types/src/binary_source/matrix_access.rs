//! Test instrumentation for the online verifier's no-A-read contract.

use akita_error::AkitaError;
use std::cell::Cell;

thread_local! {
    static FORBID_A: Cell<bool> = const { Cell::new(false) };
}

/// Reject typed A materialization on this test thread until the guard is dropped.
pub struct BinaryAReadGuard(bool);

impl BinaryAReadGuard {
    pub fn forbid() -> Self {
        Self(FORBID_A.with(|flag| flag.replace(true)))
    }
}
impl Drop for BinaryAReadGuard {
    fn drop(&mut self) {
        FORBID_A.with(|flag| flag.set(self.0));
    }
}

pub(super) fn check() -> Result<(), AkitaError> {
    if FORBID_A.with(Cell::get) {
        return Err(AkitaError::Internal(
            "online verifier attempted to read typed A".into(),
        ));
    }
    Ok(())
}
