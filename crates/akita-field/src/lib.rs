//! Field traits, concrete fields, and core error types for Akita.

#![warn(missing_docs)]
#![warn(unreachable_pub)]

/// Compatibility adapters for external trait hierarchies (the single Jolt seam).
mod compat;
/// Error types shared by Akita crates.
pub mod error;
/// Concrete prime fields, extension fields, packing, and field FFT helpers.
///
/// Internal: the public surface is the curated root re-exports plus the
/// [`packed`], [`unreduced`], and [`fft`] facades — not these module paths.
pub(crate) mod fields;
/// Conditional parallelism utilities.
pub mod parallel;
/// Native field trait hierarchy (algebra + capability traits).
pub mod traits;

pub use error::AkitaError;
pub use fields::{
    canonical_frobenius_thetas, is_registered_prime_offset, pseudo_mersenne_modulus,
    registered_prime_offset_spec, solve_frobenius_moore, validate_canonical_frobenius_thetas, Ext2,
    ExtField, Fp128, Fp32, Fp64, FpExt2, FpExt2Config, FrobeniusExtField, LiftBase, MulBase,
    MulBaseUnreduced, NegOneNr, PowerBasisFpExt4, PowerBasisFpExt4Config,
    PowerBasisFpExt4MulBackend, Prime128Offset159, Prime128Offset2355, Prime128Offset275,
    Prime128OffsetA7F7, Prime24Offset3, Prime30Offset35, Prime31Offset19, Prime32Offset99,
    Prime40Offset195, Prime48Offset59, Prime56Offset27, Prime64Offset59, PrimeOffsetSpec,
    RingSubfieldFpExt4, RingSubfieldFpExt4MulBackend, RingSubfieldFpExt8,
    RingSubfieldFpExt8MulBackend, TowerBasisFpExt4, TowerBasisFpExt4Config, TwoNr, UnitNr,
    PRIME_OFFSET_IMPLEMENTED_MAX_BITS, PRIME_OFFSET_MAX, PRIME_OFFSET_SPECS,
};
pub use traits::{
    AdditiveAccumulator, AdditiveGroup, BalancedDigitLookup, CanonicalBitLength, CanonicalBytes,
    CanonicalField, CanonicalU64, FieldCore, FixedByteSize, FixedBytes, FromPrimitiveInt,
    HalvingField, Invertible, MulPow2, MulPrimitiveInt, NaiveAccumulator, One, PseudoMersenneField,
    RandomSampling, ReducingBytes, RingAccumulator, RingCore, SmoothFftField, TranscriptChallenge,
    WithAccumulator, Zero,
};

/// SIMD packing surface: packed prime and extension representations plus the
/// [`packed::HasPacking`] bridge from a scalar field to its packed form.
pub mod packed {
    pub use crate::fields::packed::ext::{
        PackedFpExt2, PackedPowerBasisFpExt4, PackedRingSubfieldFpExt4, PackedRingSubfieldFpExt8,
        PackedTowerBasisFpExt4,
    };
    pub use crate::fields::packed::{
        Fp128Packing, Fp32Packing, Fp64Packing, HasPacking, NoPacking, PackedField, PackedValue,
    };
}

/// Unreduced / wide-accumulator arithmetic: deferred-reduction limb accumulators
/// and the [`unreduced::HasWide`] / [`unreduced::ReduceTo`] reduction bridges.
pub mod unreduced {
    pub use crate::fields::unreduced::{
        AccumPair, FoldMatrixFp32, FoldMatrixFp64, Fp128MulU64Accum, Fp128ProductAccum, Fp128x8i32,
        Fp32ProductAccum, Fp32x2i32, Fp64ProductAccum, Fp64x4i32, FpExt2Fp64ProductAccum,
        HasOptimizedFold, HasUnreducedOps, HasWide, ReduceTo, RingSubfieldFpExt4Fp32ProductAccum,
    };
}

/// Smooth-domain FFT helpers: the [`fft::SmoothDomain`] evaluation domain plus
/// root-of-unity and Reed–Solomon extension utilities.
pub mod fft {
    pub use crate::fields::fft::{field_pow, primitive_nth_root, rs_extend_fft, SmoothDomain};
}
