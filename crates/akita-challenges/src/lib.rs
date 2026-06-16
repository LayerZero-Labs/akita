//! Protocol-level Fiat-Shamir challenge samplers.
//!
//! Public surface:
//!
//! - [`SparseChallenge`] — the dependency-light data type representing one
//!   sampled sparse polynomial in `F[X]/(X^D + 1)`.
//! - [`SparseChallengeConfig`] — the policy enum that selects which sampling
//!   family is used (`Uniform`, `ExactShell`, `BoundedL1Norm`) and exposes
//!   policy questions like `l1_norm()` / `infinity_norm()` / `validate()` to
//!   `akita-config`, `akita-types`, and `akita-planner`.
//! - [`sample_sparse_challenges`] — the transcript-driven sampler that turns
//!   a config plus a Fiat-Shamir transcript into challenges.
//! - [`ChallengeShape`] / [`Challenges`] — tensor-aware folding
//!   challenge selection and sampled challenge containers.
//! - [`TensorChallenges`] — the tensor-only factored representation used when
//!   a folding round samples left/right challenge vectors instead of one flat
//!   vector.
//!
//! Each [`SparseChallengeConfig`] variant has a dedicated implementation in a
//! private `sampler` submodule (`uniform`, `exact_shell`, `bounded_l1`). The
//! SHAKE256-backed XOF cursor and the bounded-`L1` suffix-count table types
//! are crate-internal and not part of the public API.

mod challenge;
mod config;
pub mod jl;
mod sampler;
mod tensor;

pub use challenge::{IntegerChallenge, SparseChallenge};
pub use config::{SparseChallengeConfig, MIN_FOLD_CHALLENGE_ENTROPY_BITS};
pub use jl::mle::{build_jl_row_weights_reference, eval_jl_mle_at_reference};
pub use jl::{
    build_jl_row_weights, center_coefficients, eval_jl_mle_at, eval_mle_from_weights, JlImage,
    JlProjectionMatrix, DEFAULT_JL_ROWS, MAX_JL_DIGIT,
};
pub use sampler::sample_sparse_challenges;
pub use tensor::{
    sample_folding_challenges, stage1_fold_challenge_labels, tensor_left_digest, tensor_split,
    ChallengeLabels, ChallengeShape, ChallengeShape as TensorChallengeShape, Challenges,
    TensorChallenges,
};
