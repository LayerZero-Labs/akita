//! Shared verifier input shipped from the artifact generator to a Jolt guest.
//!
//! [`statement`] owns the verifier-facing input and claim model. [`wire`] owns
//! its canonical, allocation-bounded byte representation and setup decoders.

#![allow(clippy::missing_errors_doc)]

mod case;
mod statement;
mod wire;

pub use case::AkitaJoltCase;
pub use statement::{AkitaJoltInputs, AkitaJoltOpeningGroup};
pub use wire::{
    frame_with_schedule_catalog, read_blob_case, split_schedule_catalog, BLOB_COMPRESS,
    BLOB_VALIDATE, MAX_JOLT_BLOB_BYTES,
};

// `akita-algebra` is pulled in only so downstream consumers can rely on
// `CommittedGroup<F>` having all of its trait bounds satisfied.
#[doc(hidden)]
pub use akita_algebra as _akita_algebra_dep;
