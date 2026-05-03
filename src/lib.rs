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
//! - `akita-field` - Field and module traits plus core error types
//! - `akita-serialization` - Serialization abstractions
//! - [`primitives`] - Remaining in-crate primitive helpers
//!   - [`primitives::poly`] - Multilinear polynomial utility functions
//!   - [`primitives::transcript`] - Fiat-Shamir transcript trait
//!
//! ## Feature Flags
//!
//! - `parallel` - Enable Rayon parallelization for improved performance

#![warn(missing_docs)]
#![warn(unreachable_pub)]

/// Primitive traits and operations
pub mod primitives;

/// Concrete algebra backends (prime fields, extensions, rings)
pub mod algebra;

/// Offline planner modules and validation/codegen helpers.
#[doc(hidden)]
pub mod planner;

/// Protocol-layer transcript and commitment abstractions
pub mod protocol;

pub use akita_field::HachiError;
pub use akita_field::{
    cfg_chunks, cfg_chunks_mut, cfg_fold_reduce, cfg_into_iter, cfg_iter, cfg_iter_mut, cfg_join,
};
pub use akita_field::{
    AdditiveGroup, CanonicalField, FieldCore, FieldSampling, FromSmallInt, Invertible, Module,
    PseudoMersenneField, SmoothFftField,
};
pub use akita_serialization::{HachiDeserialize, HachiSerialize};
pub use protocol::{
    BasisMode, BlockOrder, CommitmentProver, CommitmentVerifier, CommittedOpenings,
    CommittedPolynomials, DensePoly, HachiPolyOps, OneHotIndex, OneHotPoly, OpeningPoints,
    ProverClaims, Transcript, VerifierClaims,
};
