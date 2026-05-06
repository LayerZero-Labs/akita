//! Protocol-level Fiat-Shamir challenge samplers.
//!
//! Public surface:
//!
//! - [`SparseChallenge`] — the dependency-light data type representing one
//!   sampled sparse polynomial in `F[X]/(X^D + 1)`. Most workspace consumers
//!   only ever import this type.
//! - [`SparseChallengeConfig`] — the policy enum that selects which sampling
//!   family is used (`Uniform`, `ExactShell`, `BoundedL1Norm`) and exposes
//!   policy questions like `l1_norm()` / `infinity_norm()` / `validate()` to
//!   `akita-config`, `akita-types`, and `akita-planner`.
//! - [`sample_sparse_challenges`] — the transcript-driven sampler that turns
//!   a config plus a Fiat-Shamir transcript into challenges.
//!
//! Each [`SparseChallengeConfig`] variant has a dedicated implementation in a
//! private `sampler` submodule (`uniform`, `exact_shell`, `bounded_l1`). The
//! SHAKE256-backed XOF cursor and the bounded-`L1` suffix-count table types
//! are crate-internal and not part of the public API.

mod challenge;
mod config;
mod sampler;
mod stage1;

pub use challenge::{IntegerChallenge, SparseChallenge};
pub use config::SparseChallengeConfig;
pub use sampler::sample_sparse_challenges;
pub use stage1::{
    sample_stage1_challenges, tensor_stage1_left_digest, tensor_stage1_split, Stage1ChallengeShape,
    Stage1Challenges, TensorStage1Challenges,
};
