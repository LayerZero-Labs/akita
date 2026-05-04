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
    AdditiveGroup, BalancedDigitLookup, CanonicalField, FieldCore, FromPrimitiveInt, HalvingField,
    Invertible, One, PseudoMersenneField, RandomSampling, SmoothFftField, Zero,
};
pub use error::AkitaError;
pub use fields::{
    is_pow2_offset, pow2_offset, pseudo_mersenne_modulus, AccumPair, Ext2, Ext4, ExtField, Fp128,
    Fp128MulU64Accum, Fp128Packing, Fp128ProductAccum, Fp128x8i32, Fp2, Fp2Config, Fp32,
    Fp32Packing, Fp32x2i32, Fp4, Fp4Config, Fp64, Fp64Packing, Fp64ProductAccum, Fp64x4i32,
    HasPacking, HasUnreducedOps, HasWide, LiftBase, NegOneNr, NoPacking, PackedField, PackedValue,
    Pow2Offset128Field, Pow2Offset24Field, Pow2Offset30Field, Pow2Offset31Field, Pow2Offset32Field,
    Pow2Offset40Field, Pow2Offset48Field, Pow2Offset56Field, Pow2Offset64Field,
    Pow2OffsetPrimeSpec, Prime128Offset159, Prime128Offset2355, Prime128Offset275,
    Prime128OffsetA7F7, ReduceTo, TwoNr, UnitNr, POW2_OFFSET_IMPLEMENTED_MAX_BITS, POW2_OFFSET_MAX,
    POW2_OFFSET_PRIMES, POW2_OFFSET_TABLE,
};
