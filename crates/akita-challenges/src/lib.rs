//! Protocol-level Fiat-Shamir challenge samplers.
//!
//! Public surface:
//!
//! - [`SparseChallenge`] — the dependency-light data type representing one
//!   sampled sparse polynomial in `F[X]/(X^D + 1)`.
//! - [`SparseChallengeConfig`] — fixed-weight sparse family `(count_pm1, count_pm2)`
//!   exposing policy questions like `l1_norm()` / `infinity_norm()` / `validate()`
//!   to `akita-config`, `akita-types`, and `akita-planner`.
//! - [`FoldDraw`] and its Spongefish adapters — fold-challenge drawing
//!   over live or preview state.
//! - [`Challenges`] — sampled folding challenges in claim-major block order.
//!
//! - [`SparseChallengeStream`] — the fixed-length message layout and decoder
//!   for one group's challenges.

mod challenge;
mod challenges;
mod config;
mod fold_draw;
mod sampler;

pub use challenge::{
    SparseChallenge, SparseChallengeCoefficients, SparseChallengePositions, INLINE_SPARSE_WEIGHT,
};
pub use challenges::Challenges;
pub use config::{
    selective_l2_challenge_config, selective_l2_operator_norm_rejection, OperatorNormRejection,
    SparseChallengeConfig, D128_L2_OP_NORM_PM1_COUNT, D128_L2_OP_NORM_PM2_COUNT,
    D128_SELECTIVE_L2_CHALLENGE_CONFIG, D64_L2_OP_NORM_PM1_COUNT, D64_L2_OP_NORM_PM2_COUNT,
    D64_PRODUCTION_PM1_COUNT, D64_PRODUCTION_PM2_COUNT, D64_SELECTIVE_L2_CHALLENGE_CONFIG,
    MIN_FOLD_CHALLENGE_ENTROPY_BITS, PRODUCTION_FOLD_CHALLENGE_RING_DIMS,
};
#[cfg(feature = "prover")]
pub use fold_draw::PreviewFoldDraw;
pub use fold_draw::{
    FoldChallengeDrawDomain, FoldChallengeFamily, FoldDraw, ProverFoldDraw, VerifierFoldDraw,
};
pub use sampler::SparseChallengeStream;
