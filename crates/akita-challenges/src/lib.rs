//! The LaBinius binary challenge families, until `jolt-crypto` owns them.
//!
//! With `labinius-challenges`: `BinaryChallengeProfile` /
//! `BinaryChallengeSampler` — exact parity-injective fixed- and bounded-weight
//! support families for the degree-162 and degree-486 LaBinius scalar rings.
//! Sparse challenges live in `jolt_crypto::challenge`.

#[cfg(feature = "labinius-challenges")]
mod binary;

#[cfg(feature = "labinius-challenges")]
pub use binary::{
    BinaryChallenge, BinaryChallengeDraw, BinaryChallengeFamily, BinaryChallengeProfile,
    BinaryChallengeSampler, BinaryChallengeTerm, BinaryScalarRing, BinarySignRule,
    INLINE_BINARY_WEIGHT,
};
