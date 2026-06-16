//! The self-contained commit subsystem.
//!
//! At its core is one Ajtai commit primitive,
//! `CommitBackend::ajtai_commit(commitment_key, spec, opening)`, that every
//! matrix multiply in the scheme (`A`, `B`, `B'`, `F`) flows through. The
//! pipeline functions (`commit`, `batched_commit`, `commit_w`,
//! `commit_inner_one`, `outer_commit`) live in the same module. The rest of
//! `akita-prover` only sees the re-exported entry points plus the
//! `CommitBackend` / `AjtaiOpeningView` traits.

mod ajtai;
mod decompose;
mod entry;
mod inner;
mod opening_view;
mod outer;
mod pipeline;
mod recursive;

// The narrow public surface.
pub use ajtai::backend::CommitBackend;
pub use ajtai::opening::{AjtaiOpeningType, ZeroScan};
pub use ajtai::spec::{MatrixRole, MatrixSpec, RingDomain};
pub use entry::{batched_commit, commit};
pub use opening_view::AjtaiOpeningView;
pub use pipeline::{
    batched_commit_with_params, commit_with_params, prepare_batched_commit_inputs,
    prepare_commit_inputs,
};
pub use recursive::commit_w;

// Internal helpers shared with the setup-prefix and zk-hiding commit paths,
// but not part of the external surface.
#[cfg(feature = "zk")]
pub(crate) use inner::commit_inner_one;
pub(crate) use pipeline::{
    commit_inner_block_digit_count, commit_inner_flat_digit_count,
    validate_commit_outer_input_nonempty, validate_onehot_chunk_size_for_params,
};
