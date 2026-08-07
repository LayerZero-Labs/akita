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
//! - `jolt-field` - Shared field traits, concrete fields, packing, and FFT helpers
//! - `akita-error` - Akita protocol errors
//! - `akita-serialization` - Serialization abstractions
//! - `akita-algebra` - Modules, rings, NTTs, and polynomial helpers
//! - `akita-transcript` - Fiat-Shamir transcript implementations and labels
//! - `akita-challenges` - Fiat-Shamir challenge sampling helpers
//! - `akita-sumcheck` - Generic sumcheck proof types, traits, and drivers
//! - `akita-verifier` - Verifier replay without prover-only polynomial backends
//! - `akita-prover` - Commitment and proving kernels
//! - `akita-pcs` - End-to-end [`AkitaCommitmentScheme`] orchestration plus public re-exports
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

mod scheme;
#[cfg(all(test, any(feature = "schedules-default", feature = "profile-ci")))]
#[path = "../tests/support/mod.rs"]
mod test_support;

pub use akita_algebra::Module;
// Specialized field surfaces mirror jolt-field's curated facades.
pub use akita_algebra::fft;
pub use akita_algebra::fft::SmoothFftField;
pub use akita_prover::{
    CommitmentComputeBackend, ComputeBackendSetup, CpuBackend, CpuPreparedSetup,
    CyclicRowsComputeBackend, DecomposeFoldWitness, DenseCommitInput, DenseCommitRowsPlan,
    DigitRowsComputeBackend, FlatBlockTable, LevelProveStacks, MultiChunkEntry, OneHotCommitBlocks,
    OneHotCommitRowsPlan, OpeningProveBackendFor, OperationCtx, PreparedGroupProveOps,
    PreparedProverGroup, ProveBackendFor, ProverOpeningData, RecursiveProveBackend,
    RecursiveWitnessCommitRowsPlan, RingSwitchComputeBackend, RingSwitchQuotientRowsPlan,
    RingSwitchRelationRows, RingSwitchRelationRowsPlan, RootCommitBackend, RootCommitSource,
    RootOpeningSource, RootPolyShape, RootProveBackend, RootProvePoly, RootTensorSource,
    SelectedProverOpeningData, SingleChunkEntry, SparseRingBlockEntry, SparseRingCommitRowsPlan,
    TensorBackendFor, TieredProveStacks, UniformProverStack, RECURSIVE_SUFFIX_RING_DIMENSIONS,
};
pub use akita_serialization::{AkitaDeserialize, AkitaSerialize};
pub use akita_transcript::{AkitaTranscript, Transcript};
pub use akita_types::{BasisMode, OpeningClaims, OpeningClaimsLayout, PolynomialGroupClaims};
pub use scheme::AkitaCommitmentScheme;
