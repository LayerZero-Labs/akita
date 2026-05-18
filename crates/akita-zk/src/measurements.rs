//! Measurement-only protocol variants.
//!
//! The items in this module are useful for proof-size and rejection-policy
//! experiments, but they must not be treated as production zero-knowledge
//! protocols.

pub use crate::protocols::opening::{
    prove_compact_public_sign_gaertner_ajtai_opening, prove_public_sign_gaertner_ajtai_opening,
    verify_compact_public_sign_gaertner_ajtai_opening, verify_public_sign_gaertner_ajtai_opening,
    CompactPublicSignGaertnerAjtaiOpeningProof, PublicSignGaertnerAjtaiOpeningProof,
};
