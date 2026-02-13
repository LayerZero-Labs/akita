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
//!   - [`primitives::poly`] - Multilinear polynomial traits and operations
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

/// Protocol-layer transcript and commitment abstractions
pub mod protocol;

pub use error::HachiError;
pub use primitives::arithmetic::{
    CanonicalField, CtInvertible, FieldCore, FieldSampling, HachiRoutines, Module,
    PseudoMersenneField,
};
pub use primitives::poly::{MultilinearLagrange, Polynomial};
pub use primitives::serialization::{HachiDeserialize, HachiSerialize};
pub use protocol::{CommitmentScheme, StreamingCommitmentScheme, Transcript};
