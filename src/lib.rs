#![cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    feature(stdarch_x86_avx512)
)]
//! # hachi
//!
//! A high performance and modular implementation of the Hachi polynomial commitment scheme.
//!
//! Hachi is a lattice-based polynomial commitment scheme with transparent setup and
//! post-quantum security guarantees.
//!
//! ## Key Features
//!
//! - **Post-quantum secure**: Based on lattice hardness assumptions
//! - **Transparent setup**: No trusted setup required
//! - **Modular design**: Flexible trait-based architecture
//! - **Performance optimizations**: Optional parallelization support
//!
//! ## Structure
//!
//! ### Core Modules
//! - [`primitives`] - Core traits and abstractions
//!   - [`primitives::arithmetic`] - Field and module traits for lattice arithmetic
//!   - [`primitives::poly`] - Multilinear polynomial utility functions
//!   - [`primitives::transcript`] - Fiat-Shamir transcript trait
//!   - [`primitives::serialization`] - Serialization abstractions
//! - [`error`] - Error types
//!
//! ## Feature Flags
//!
//! - `parallel` - Enable Rayon parallelization for improved performance

#![warn(missing_docs)]
#![warn(unreachable_pub)]

/// Error types for Hachi PCS operations
pub mod error;

/// Primitive traits and operations
pub mod primitives;

/// Concrete algebra backends (prime fields, extensions, rings)
pub mod algebra;

/// Conditional parallelism utilities (`cfg_iter!`, `cfg_into_iter!`, etc.)
#[macro_use]
#[doc(hidden)]
pub mod parallel;

/// Protocol-layer transcript and commitment abstractions
pub mod protocol;

/// Shared test configuration and helpers.
#[doc(hidden)]
pub mod test_utils;

pub use error::HachiError;
pub use primitives::arithmetic::{
    AdditiveGroup, CanonicalField, FieldCore, FieldSampling, FromSmallInt, Invertible, Module,
    PseudoMersenneField,
};
pub use primitives::serialization::{HachiDeserialize, HachiSerialize};
pub use protocol::{
    BasisMode, BlockOrder, CommitmentScheme, DenseMultilinear, DensePoly, DynamicCommitHint,
    DynamicCommitmentScheme, DynamicHachiCommitmentScheme, DynamicHachiProverSetup,
    DynamicHachiVerifierSetup, DynamicRingCommitment, DynamicRootConfigFamily, HachiPolyOps,
    HachiRootBatchSummary, MultilinearPolynomial, OneHotIndex, OneHotMultilinear, OneHotPoly,
    Transcript,
};
