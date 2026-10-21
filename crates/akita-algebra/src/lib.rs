//! Concrete algebra backends and arithmetic building blocks.
//!
//! This module includes multilinear polynomial, eq-polynomial, and offset-eq
//! utilities.
//!
//! Concrete fields, field packing, the cyclotomic ring, NTT/CRT arithmetic,
//! and the smooth FFT live in `jolt-algebra`. Sparse Fiat–Shamir challenge
//! representations and samplers live in `jolt-crypto`.
//!
//! The LaBinius extensions keep their arithmetic here until `jolt-algebra`
//! carries it: binary tower fields (`labinius-binary`) and the trinomial ring
//! with its small-prime limb transform (`labinius-trinomial`).

#![warn(missing_docs)]
#![warn(unreachable_pub)]

#[cfg(feature = "labinius-binary")]
pub mod binary;
mod cyclotomic_inverse;
pub mod eq_poly;
pub mod offset_eq;
pub mod poly;
pub mod split_eq;
#[cfg(feature = "labinius-trinomial")]
pub mod trinomial;

// Flat re-exports for convenience.
pub use cyclotomic_inverse::CyclotomicInverse;
pub use eq_poly::{EqPolynomial, SplitEqEvals};
pub use split_eq::GruenSplitEq;
#[cfg(feature = "labinius-trinomial")]
pub use trinomial::{
    embed_scalar, pack_scalar_components, shifted_remainder_evaluations, unpack_scalar_components,
    MinusTrinomial, PlusTrinomial, Prime64Offset23703, Prime64Offset23703Ext2,
    Prime64Offset23703Nr5, TrinomialError, TrinomialField, TrinomialI8Lut,
    TrinomialLimbAccumulator, TrinomialLimbDomain, TrinomialLimbSlots, TrinomialModulus,
    TrinomialNtt, TrinomialNttDomain, TrinomialNttWorkspace, TrinomialRing,
    PRIME64_OFFSET_23703_MODULUS,
};
