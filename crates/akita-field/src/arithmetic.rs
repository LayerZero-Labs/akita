#![allow(missing_docs)]

use akita_serialization::{AkitaDeserialize, AkitaSerialize};

pub use jolt_field::{
    AdditiveAccumulator, AdditiveGroup, CanonicalBitLength, CanonicalBytes, CanonicalU64,
    FieldCore, FixedByteSize, FixedBytes, FromPrimitiveInt, Invertible, MulPow2, MulPrimitiveInt,
    NaiveAccumulator, RandomSampling, ReducingBytes, RingAccumulator, RingCore,
    TranscriptChallenge, WithAccumulator,
};
pub use num_traits::{One, Zero};

/// Canonical integer representation for prime (base) field elements.
///
/// Provides a bijection between field elements and integers in `[0, p)`.
/// Only meaningful for base prime fields where elements ARE residues mod p.
/// Extension fields should NOT implement this trait.
pub trait CanonicalField:
    FieldCore + FromPrimitiveInt + AkitaSerialize + AkitaDeserialize<Context = ()>
{
    /// Return canonical integer representation as `u128`.
    fn to_canonical_u128(self) -> u128;

    /// Bit-width of the field modulus.
    fn modulus_bits() -> u32;

    /// Construct from canonical value if it is in range.
    fn from_canonical_u128_checked(val: u128) -> Option<Self>;

    /// Construct from canonical value reduced modulo the field modulus.
    fn from_canonical_u128_reduced(val: u128) -> Self;
}

/// Field types with a cheap halving operation.
///
/// This is intentionally narrower than core field algebra: only protocol paths
/// that divide by two should require it.
pub trait HalvingField: FieldCore {
    /// Divide this element by two.
    fn half(self) -> Self;

    /// Multiplicative inverse of 2.
    #[inline]
    fn two_inv() -> Self {
        Self::one().half()
    }
}

/// Balanced signed-digit lookup support for small power-of-two bases.
pub trait BalancedDigitLookup: FromPrimitiveInt + Zero + Copy {
    /// Lookup table mapping balanced digit index to field element.
    ///
    /// For `log_basis` in `1..=6`, returns a 64-entry table where
    /// `table[i]` = `from_i64(i - b/2)` for `i < b = 2^log_basis`,
    /// and zero for `i >= b`.
    fn digit_lut(log_basis: u32) -> [Self; 64] {
        debug_assert!(log_basis > 0 && log_basis <= 6);
        let b = 1usize << log_basis;
        let half_b = (b >> 1) as i64;
        std::array::from_fn(|i| {
            if i < b {
                Self::from_i64(i as i64 - half_b)
            } else {
                Self::zero()
            }
        })
    }
}

/// Metadata for pseudo-Mersenne style moduli (`2^k - c`).
pub trait PseudoMersenneField: CanonicalField {
    /// Exponent `k` in `2^k - c`.
    const MODULUS_BITS: u32;

    /// Offset `c` in `2^k - c`.
    const MODULUS_OFFSET: u128;
}

/// Field carrying a precomputed primitive root of its largest smooth
/// multiplicative subgroup, suitable for NTT-based FFT.
///
/// A *smooth* subgroup is one whose order factors into small primes; the
/// FFT in `akita-algebra` requires a primitive `n`-th root
/// of unity for each transform size `n`. Rather than hunting for one at
/// runtime, the field exposes a single primitive root of its full smooth
/// subgroup; any primitive `n`-th root for `n | SMOOTH_SUBGROUP_ORDER`
/// is obtained by `SMOOTH_OMEGA ^ (SMOOTH_SUBGROUP_ORDER / n)`.
///
/// Implementors must guarantee that:
/// 1. `SMOOTH_SUBGROUP_ORDER` divides `p − 1`.
/// 2. `SMOOTH_OMEGA` (interpreted as a canonical field element) has
///    exact multiplicative order `SMOOTH_SUBGROUP_ORDER`.
///
/// Both invariants are checked by the `omega_has_declared_order` test
/// in `src/algebra/fields/fft.rs` and re-derived against
/// `find_primitive_nth_root` so the constants cannot drift.
pub trait SmoothFftField: CanonicalField + PseudoMersenneField {
    /// Order of the largest smooth multiplicative subgroup we support
    /// for FFT. Must divide `p − 1`.
    const SMOOTH_SUBGROUP_ORDER: usize;

    /// Canonical `u128` representation of a primitive
    /// `SMOOTH_SUBGROUP_ORDER`-th root of unity in the field.
    const SMOOTH_OMEGA: u128;
}
