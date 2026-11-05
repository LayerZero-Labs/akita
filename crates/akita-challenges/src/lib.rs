//! LaBinius binary challenge families and the fold-challenge draw adapters,
//! until `jolt-crypto` owns them. Sparse challenges live in
//! `jolt_crypto::challenge`; the fold family in `akita-params`.
//!
//! - `BinaryChallengeProfile` / `BinaryChallengeSampler` — exact
//!   parity-injective fixed- and bounded-weight support families for the
//!   degree-162 and degree-486 LaBinius scalar rings (with
//!   `labinius-challenges`), and the binary root fold family.
//! - [`FoldDraw`] and its Spongefish adapters — fold-challenge drawing
//!   over live or preview state.

#[cfg(feature = "labinius-challenges")]
mod binary;
mod fold_draw;

#[cfg(feature = "labinius-challenges")]
pub use binary::{
    root_family, BinaryChallenge, BinaryChallengeFamily, BinaryChallengeProfile,
    BinaryChallengeSampler, BinaryChallengeTerm, BinaryScalarRing, BinarySignRule,
    BINARY_CHALLENGE_SEED_LEN, INLINE_BINARY_WEIGHT,
};
#[cfg(feature = "prover")]
pub use fold_draw::PreviewFoldDraw;
pub use fold_draw::{FoldDraw, ProverFoldDraw, VerifierFoldDraw};
