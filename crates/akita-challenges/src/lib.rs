//! Protocol-level Fiat-Shamir challenge samplers.
//!
//! Public surface:
//!
//! - [`SparseChallenge`] — the dependency-light data type representing one
//!   sampled sparse polynomial in `F[X]/(X^D + 1)`.
//! - [`SparseChallengeConfig`] — fixed-weight sparse family `(count_pm1, count_pm2)`
//!   exposing policy questions like `l1_norm()` / `infinity_norm()` / `validate()`
//!   to `akita-config`, `akita-types`, and `akita-planner`.
//! - [`sample_sparse_challenges`] — the transcript-driven sampler that turns
//!   a config plus a Fiat-Shamir transcript into sparse challenges.
//! - [`FoldDraw`] / [`LiveFoldDraw`] / [`PreviewFoldDraw`] — tensor-aware
//!   fold-challenge drawing over live or preview transcript state.
//! - [`ChallengeShape`] / [`Challenges`] — tensor-aware folding
//!   challenge selection and sampled challenge containers.
//! - [`TensorChallenges`] — the tensor-only factored representation used when
//!   a folding round samples left/right challenge vectors instead of one flat
//!   vector.
//!
//! Sampling uses the signed-sparse path in a private `sampler` submodule. The
//! SHAKE256-backed XOF cursor is crate-internal and not part of the public API.

mod challenge;
mod config;
mod fold_draw;
mod grind_probe;
pub mod jl;
mod sampler;
mod tensor;

pub use akita_transcript::FoldChallengeSeedPreview;
pub use challenge::SparseChallenge;
pub use config::{
    SparseChallengeConfig, D64_PRODUCTION_PM1_COUNT, D64_PRODUCTION_PM2_COUNT,
    MIN_FOLD_CHALLENGE_ENTROPY_BITS, PRODUCTION_FOLD_CHALLENGE_RING_DIMS,
};
pub use fold_draw::{FoldDraw, LiveFoldDraw, PreviewFoldDraw};
pub use grind_probe::grind_probe_permutation;
pub use jl::mle::{
    build_jl_row_weights, build_jl_row_weights_from_row_eq, build_jl_row_weights_reference,
    eval_jl_mle_at, eval_jl_mle_at_from_eq_tables, eval_jl_mle_at_reference, eval_jl_mle_at_scalar,
    eval_jl_mle_at_scalar_from_eq_tables, eval_mle_from_weights,
};
pub use jl::{
    center_coefficients, project_digits_reference, project_digits_scalar, JlImage,
    JlProjectionMatrix, DEFAULT_JL_ROWS, MAX_JL_DIGIT,
};

/// Bench-only surface for criterion JL benches (not a stable API).
#[doc(hidden)]
pub mod jl_bench {
    pub use crate::jl::mle::{
        build_jl_row_weights_from_row_eq, eval_jl_mle_at_from_eq_tables, eval_jl_mle_at_scalar,
        eval_jl_mle_at_scalar_from_eq_tables,
    };
    pub use crate::jl::{project_digits_reference, project_digits_scalar};
}
pub use sampler::sample_sparse_challenges;
pub use tensor::{
    fold_sparse_challenge_sample_count, tensor_left_digest, tensor_split,
    witness_fold_challenge_labels, ChallengeLabels, ChallengeShape,
    ChallengeShape as TensorChallengeShape, Challenges, TensorChallenges,
};
