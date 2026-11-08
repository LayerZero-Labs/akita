//! Protocol-level Fiat-Shamir challenge samplers.
//!
//! Sparse ring challenges, their families and operator-norm rejection, and
//! the per-group message stream live in `jolt_crypto::challenge`; this crate
//! re-exports them under their former paths. It owns fold-challenge drawing:
//! [`FoldDraw`] and its Spongefish adapters, over live or preview state, and
//! [`FoldChallengeDistribution`], the fold-challenge family a schedule names.
//! `BinaryChallengeProfile` / `BinaryChallengeSampler` are the exact
//! parity-injective fixed- and bounded-weight support families for the
//! degree-162 and degree-486 LaBinius scalar rings (with `labinius-challenges`).

#[cfg(feature = "labinius-challenges")]
mod binary;
mod family;
mod fold_draw;

#[cfg(feature = "labinius-challenges")]
pub use binary::{
    BinaryChallenge, BinaryChallengeFamily, BinaryChallengeProfile, BinaryChallengeSampler,
    BinaryChallengeTerm, BinaryScalarRing, BinarySignRule, ColumnChallengeAddress,
    MultigroupColumnSampler, INLINE_BINARY_WEIGHT, MULTIGROUP_COLUMN_DOMAIN,
    MULTIGROUP_COLUMN_FAMILY_ID,
};
pub use family::FoldChallengeDistribution;
#[cfg(feature = "prover")]
pub use fold_draw::PreviewFoldDraw;
pub use fold_draw::{
    FoldChallengeDrawDomain, FoldChallengeFamily, FoldDraw, ProverFoldDraw, VerifierFoldDraw,
};
pub use jolt_crypto::challenge::{
    selective_l2_challenge_config, selective_l2_operator_norm_rejection, Challenges,
    OperatorNormRejection, SparseChallenge, SparseChallengeCoefficients, SparseChallengeConfig,
    SparseChallengePositions, SparseChallengeStream, D128_L2_OP_NORM_PM1_COUNT,
    D128_L2_OP_NORM_PM2_COUNT, D128_SELECTIVE_L2_CHALLENGE_CONFIG, D64_L2_OP_NORM_PM1_COUNT,
    D64_L2_OP_NORM_PM2_COUNT, D64_PRODUCTION_PM1_COUNT, D64_PRODUCTION_PM2_COUNT,
    D64_SELECTIVE_L2_CHALLENGE_CONFIG, INLINE_SPARSE_WEIGHT, MIN_FOLD_CHALLENGE_ENTROPY_BITS,
    PRODUCTION_FOLD_CHALLENGE_RING_DIMS,
};
