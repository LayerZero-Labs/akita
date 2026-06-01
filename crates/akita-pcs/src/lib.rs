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
//! - `akita-field` - Field traits, concrete fields, packing, and core error types
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

pub use akita_algebra::Module;
pub use akita_field::AkitaError;
pub use akita_field::{
    cfg_chunks, cfg_chunks_mut, cfg_fold_reduce, cfg_into_iter, cfg_iter, cfg_iter_mut, cfg_join,
};
pub use akita_field::{
    is_registered_prime_offset, pseudo_mersenne_modulus, registered_prime_offset_spec,
    AdditiveGroup, BalancedDigitLookup, CanonicalField, ExtField, FieldCore, Fp128, Fp128Packing,
    Fp32, Fp32Packing, Fp64, Fp64Packing, FpExt2, FpExt2Config, FromPrimitiveInt, HalvingField,
    HasPacking, Invertible, LiftBase, NoPacking, PackedField, PackedValue, PowerBasisFpExt4,
    PowerBasisFpExt4Config, Prime128Offset159, Prime128Offset2355, Prime128Offset275,
    Prime128OffsetA7F7, Prime24Offset3, Prime30Offset35, Prime31Offset19, Prime32Offset99,
    Prime40Offset195, Prime48Offset59, Prime56Offset27, Prime64Offset59, PrimeOffsetSpec,
    PseudoMersenneField, RandomSampling, SmoothFftField, TowerBasisFpExt4, TowerBasisFpExt4Config,
    PRIME_OFFSET_IMPLEMENTED_MAX_BITS, PRIME_OFFSET_MAX, PRIME_OFFSET_SPECS,
};
pub use akita_prover::{
    AkitaPolyOps, CommitmentComputeBackend, CommitmentProver, CommittedPolynomials,
    ComputeBackendSetup, CpuBackend, CpuPreparedSetup, CyclicRowsComputeBackend,
    DecomposeFoldWitness, DenseCommitInput, DenseCommitRowsPlan, DigitRowsComputeBackend,
    FlatBlockTable, MultiChunkEntry, OneHotCommitBlocks, OneHotCommitRowsPlan, ProverClaims,
    ProverComputeBackend, RecursiveWitnessCommitRowsPlan, RingSwitchComputeBackend,
    RingSwitchQuotientRowsPlan, RingSwitchRelationRows, RingSwitchRelationRowsPlan,
    SingleChunkEntry, SparseRingBlockEntry, SparseRingCommitRowsPlan,
};
pub use akita_serialization::{AkitaDeserialize, AkitaSerialize};
pub use akita_transcript::{AkitaTranscript, Transcript};
pub use akita_types::{BasisMode, BlockOrder};
pub use scheme::AkitaCommitmentScheme;
