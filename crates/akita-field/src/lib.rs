//! Field traits, concrete fields, and core error types for Akita.

#![warn(missing_docs)]
#![warn(unreachable_pub)]

/// Compatibility adapters for external trait hierarchies (the single Jolt seam).
mod compat;
/// Error types shared by Akita crates.
pub mod error;
mod ext;
/// Smooth-domain FFT helpers.
pub mod fft;
/// SIMD packing surface.
pub mod packed;
/// Conditional parallelism utilities.
pub mod parallel;
mod prime;
/// Native field trait hierarchy (algebra + capability traits).
pub mod traits;
/// Unreduced / wide-accumulator arithmetic.
pub mod unreduced;

pub use error::AkitaError;
pub use ext::lift::{
    canonical_frobenius_thetas, solve_frobenius_moore, validate_canonical_frobenius_thetas,
    ExtField, FrobeniusExtField, LiftBase, MulBase, MulBaseUnreduced,
};
pub use ext::{
    Ext2, FpExt2, FpExt2Config, FpExt4, FpExt4MulBackend, FpExt8, FpExt8MulBackend, NegOneNr, TwoNr,
};
pub use prime::{
    is_registered_prime_offset, pseudo_mersenne_modulus, registered_prime_offset_spec, Fp128, Fp32,
    Fp64, Prime128Offset159, Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7,
    Prime24Offset3, Prime30Offset35, Prime31Offset19, Prime32Offset99, Prime40Offset195,
    Prime48Offset59, Prime56Offset27, Prime64Offset59, PrimeOffsetSpec,
    PRIME_OFFSET_IMPLEMENTED_MAX_BITS, PRIME_OFFSET_MAX, PRIME_OFFSET_SPECS,
};
pub use traits::{
    AdditiveAccumulator, AdditiveGroup, BalancedDigitLookup, CanonicalBitLength, CanonicalBytes,
    CanonicalField, CanonicalU64, FieldCore, FixedByteSize, FixedBytes, FromPrimitiveInt,
    HalvingField, Invertible, MulPow2, MulPrimitiveInt, NaiveAccumulator, One, PseudoMersenneField,
    RandomSampling, ReducingBytes, RingAccumulator, RingCore, SmoothFftField, TranscriptChallenge,
    WithAccumulator, Zero,
};
