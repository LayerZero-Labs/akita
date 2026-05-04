#![cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    feature(stdarch_x86_avx512)
)]
//! # Akita PCS
//!
//! A high performance and modular implementation of the Akita polynomial commitment scheme.
//!
//! Akita is a lattice-based polynomial commitment scheme with transparent setup and
//! post-quantum security guarantees. It descends from Hachi while carrying the current
//! Akita crate decomposition work.
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
//! - `akita-algebra` - Concrete fields, rings, NTTs, and polynomial helpers
//! - `akita-transcript` - Fiat-Shamir transcript implementations and labels
//! - `akita-challenges` - Fiat-Shamir challenge sampling helpers
//! - `akita-sumcheck` - Generic sumcheck proof types, traits, and drivers
//! - `akita-verifier` - Verifier replay without prover-only polynomial backends
//! - `akita-prover` - Commitment and proving kernels
//! - `akita-scheme` - End-to-end [`AkitaCommitmentScheme`] orchestration
//!
//! Verifier-only consumers should depend directly on `akita-verifier`,
//! `akita-types`, and `akita-config`. This umbrella crate is convenient for
//! examples and end-to-end use, but it intentionally re-exports prover-facing
//! APIs as well.
//!
//! ## Feature Flags
//!
//! - `parallel` - Enable Rayon parallelization for improved performance

#![warn(missing_docs)]
#![warn(unreachable_pub)]

pub use akita_field::HachiError;
pub use akita_field::{
    cfg_chunks, cfg_chunks_mut, cfg_fold_reduce, cfg_into_iter, cfg_iter, cfg_iter_mut, cfg_join,
};
pub use akita_field::{
    AdditiveGroup, CanonicalField, FieldCore, FieldSampling, FromSmallInt, Invertible, Module,
    PseudoMersenneField, SmoothFftField,
};
pub use akita_prover::{CommitmentProver, CommittedPolynomials, HachiPolyOps, ProverClaims};
pub use akita_scheme::AkitaCommitmentScheme;
pub use akita_serialization::{HachiDeserialize, HachiSerialize};
pub use akita_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
pub use akita_types::{BasisMode, BlockOrder};
