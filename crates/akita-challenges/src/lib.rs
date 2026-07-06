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
//!   a config plus a Fiat-Shamir transcript into challenges.
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
mod sampler;
mod tensor;

pub use akita_transcript::FoldChallengeSeedPreview;
pub use challenge::SparseChallenge;
pub use config::{
    SparseChallengeConfig, D64_PRODUCTION_PM1_COUNT, D64_PRODUCTION_PM2_COUNT,
    MIN_FOLD_CHALLENGE_ENTROPY_BITS, PRODUCTION_FOLD_CHALLENGE_RING_DIMS,
};
pub use fold_draw::{preview_folding_challenges, sample_folding_challenges};
pub use grind_probe::grind_probe_permutation;
pub use sampler::{
    sample_sparse_challenges, sparse_challenge_absorb_buf, sparse_challenges_from_seed,
};
pub use tensor::{
    fold_sparse_challenge_sample_count, tensor_left_digest, tensor_split,
    witness_fold_challenge_labels, ChallengeLabels, ChallengeShape,
    ChallengeShape as TensorChallengeShape, Challenges, TensorChallenges,
};
