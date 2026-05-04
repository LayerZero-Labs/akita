#![cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    feature(stdarch_x86_avx512)
)]
//! # Akita PCS
//!
//! A high performance and modular implementation of the Akita polynomial commitment scheme.
//!
//! Akita is a lattice-based polynomial commitment scheme with transparent setup and
//! post-quantum security guarantees. It descends from Akita while carrying the current
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

pub use akita_algebra::Module;
pub use akita_field::AkitaError;
pub use akita_field::{
    cfg_chunks, cfg_chunks_mut, cfg_fold_reduce, cfg_into_iter, cfg_iter, cfg_iter_mut, cfg_join,
};
pub use akita_field::{
    is_pow2_offset, pow2_offset, pseudo_mersenne_modulus, AdditiveGroup, CanonicalField, ExtField,
    FieldCore, FieldSampling, Fp128, Fp128Packing, Fp2, Fp2Config, Fp32, Fp32Packing, Fp4,
    Fp4Config, Fp64, Fp64Packing, FromSmallInt, HasPacking, Invertible, LiftBase, NoPacking,
    PackedField, PackedValue, Pow2Offset128Field, Pow2Offset24Field, Pow2Offset30Field,
    Pow2Offset31Field, Pow2Offset32Field, Pow2Offset40Field, Pow2Offset48Field, Pow2Offset56Field,
    Pow2Offset64Field, Pow2OffsetPrimeSpec, Prime128Offset159, Prime128Offset2355,
    Prime128Offset275, Prime128OffsetA7F7, PseudoMersenneField, SmoothFftField,
    POW2_OFFSET_IMPLEMENTED_MAX_BITS, POW2_OFFSET_MAX, POW2_OFFSET_PRIMES, POW2_OFFSET_TABLE,
};
pub use akita_prover::{AkitaPolyOps, CommitmentProver, CommittedPolynomials, ProverClaims};
pub use akita_scheme::AkitaCommitmentScheme;
pub use akita_serialization::{AkitaDeserialize, AkitaSerialize};
pub use akita_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
pub use akita_types::{BasisMode, BlockOrder};
