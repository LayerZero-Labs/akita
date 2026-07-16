//! Runtime-only caches for recursive Akita prove levels.
//!
//! These structures sit between the recursive `w` witness and the verifier-
//! facing proof wire. They preserve the commitment-side prover caches that the
//! next recursive level needs, without forcing the prover to round-trip through
//! the proof-oriented flat adapters each time.
//!
//! # S5 (runtime-ring cutover): D-free cache
//!
//! The cache now holds the D-free [`AkitaCommitmentHint`] (the decomposed digit
//! stream) directly. The former D-typed `recomposed_inner_rows` are not cached;
//! callers recompute them on demand from the digit stream (see
//! [`crate::compute::recompose_hint_inner_rows`]). The cache is therefore a thin
//! D-free wrapper that exists only to give the recursive next-level commitment a
//! named place to carry its hint.

use akita_types::AkitaCommitmentHint;
use jolt_field::FieldCore;

/// D-erased prover cache for a recursive commitment hint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecursiveCommitmentHintCache<F: FieldCore> {
    hint: AkitaCommitmentHint<F>,
}

impl<F: FieldCore> RecursiveCommitmentHintCache<F> {
    /// Wrap a D-free commitment hint for carry across a recursive level.
    pub fn from_hint(hint: AkitaCommitmentHint<F>) -> Self {
        Self { hint }
    }

    /// Borrow the cached D-free commitment hint.
    pub fn hint(&self) -> &AkitaCommitmentHint<F> {
        &self.hint
    }

    /// Consume the cache and return the D-free commitment hint.
    pub fn into_hint(self) -> AkitaCommitmentHint<F> {
        self.hint
    }
}
