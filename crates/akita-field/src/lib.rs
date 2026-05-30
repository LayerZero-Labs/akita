//! Field traits, concrete fields, and core error types for Akita.

#![warn(missing_docs)]
#![warn(unreachable_pub)]

/// Field arithmetic traits.
pub mod arithmetic;
/// Error types shared by Akita crates.
pub mod error;
/// Concrete prime fields, extension fields, packing, and field FFT helpers.
pub mod fields;
/// Implementations of Jolt's slim field hierarchy for Akita types.
mod jolt_traits;
/// Conditional parallelism utilities.
pub mod parallel;

pub use arithmetic::{
    AdditiveAccumulator, AdditiveGroup, BalancedDigitLookup, CanonicalBitLength, CanonicalBytes,
    CanonicalField, CanonicalU64, FieldCore, FixedByteSize, FixedBytes, FromPrimitiveInt,
    HalvingField, Invertible, MulPow2, MulPrimitiveInt, NaiveAccumulator, One, PseudoMersenneField,
    RandomSampling, ReducingBytes, RingAccumulator, RingCore, SmoothFftField, TranscriptChallenge,
    WithAccumulator, Zero,
};
pub use error::AkitaError;
pub use fields::{
    canonical_frobenius_thetas, is_registered_prime_offset, pseudo_mersenne_modulus,
    registered_prime_offset_spec, solve_frobenius_moore, validate_canonical_frobenius_thetas,
    AccumPair, Ext2, ExtField, FoldMatrixFp16, FoldMatrixFp32, Fp128, Fp128MulU64Accum,
    Fp128Packing, Fp128ProductAccum, Fp128x8i32, Fp16, Fp16Packing, Fp2, Fp2Config, Fp32,
    Fp32Packing, Fp32ProductAccum, Fp32x2i32, Fp64, Fp64Packing, Fp64ProductAccum, Fp64x4i32,
    FrobeniusExtField, HasOptimizedFold, HasPacking, HasUnreducedOps, HasWide, LiftBase, MulBase,
    NegOneNr, NoPacking, PackedField, PackedValue, PowerBasisFp4, PowerBasisFp4Config,
    PowerBasisFp4MulBackend, Prime128Offset159, Prime128Offset2355, Prime128Offset275,
    Prime128OffsetA7F7, Prime16Offset99, Prime24Offset3, Prime30Offset35, Prime31Offset19,
    Prime32Offset99, Prime40Offset195, Prime48Offset59, Prime56Offset27, Prime64Offset59,
    PrimeOffsetSpec, ReduceTo, RingSubfieldFp4, RingSubfieldFp4MulBackend, RingSubfieldFp8,
    RingSubfieldFp8MulBackend, TowerBasisFp4, TowerBasisFp4Config, TwoNr, UnitNr,
    PRIME_OFFSET_IMPLEMENTED_MAX_BITS, PRIME_OFFSET_MAX, PRIME_OFFSET_SPECS,
};
