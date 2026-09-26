//! Protocol-level Fiat-Shamir challenge samplers.
//!
//! Public surface:
//!
//! - [`SparseChallenge`] — the dependency-light data type representing one
//!   sampled sparse polynomial in `F[X]/(X^D + 1)`.
//! - [`SparseChallengeConfig`] — fixed-weight sparse family `(count_pm1, count_pm2)`
//!   exposing policy questions like `l1_norm()` / `infinity_norm()` / `validate()`
//!   to `akita-config`, `akita-types`, and `akita-planner`.
//! - `BinaryChallengeProfile` / `BinaryChallengeSampler` — exact
//!   parity-injective fixed- and bounded-weight support families for the
//!   degree-162 and degree-486 LaBinius scalar rings (with `labinius-challenges`).
//! - [`FoldDraw`] and its native Spongefish adapters — fold-challenge drawing
//!   over live or preview native state.
//! - [`Challenges`] — sampled folding challenges in claim-major block order.
//!
//! Sampling uses the signed-sparse path in a private `sampler` submodule. The
//! SHAKE256-backed XOF cursor is crate-internal and not part of the public API.

#[cfg(feature = "labinius-challenges")]
mod binary;
mod challenge;
mod challenges;
mod config;
mod fold_draw;
mod sampler;

#[cfg(feature = "labinius-challenges")]
pub use binary::{
    BinaryChallenge, BinaryChallengeFamily, BinaryChallengeProfile, BinaryChallengeSampler,
    BinaryChallengeTerm, BinaryScalarRing, BinarySignRule, INLINE_BINARY_WEIGHT,
};
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
pub use fold_draw::{
    fold_challenge_sample_label, FoldChallengeDrawDomain, FoldDraw, NativePreviewFoldDraw,
    NativeProverFoldDraw, NativeVerifierFoldDraw,
};
