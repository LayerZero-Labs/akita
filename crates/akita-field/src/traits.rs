#![allow(missing_docs)]

use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use rand_core::RngCore;

pub use num_traits::{One, Zero};

/// Minimal additive-group operations shared by fields, rings, and accumulators.
///
/// Native Akita definition (mirrors the slim Jolt field-trait shape). The
/// `jolt-compat` layer forwards Jolt's `AdditiveGroup` to this for interop.
pub trait AdditiveGroup:
    Sized
    + Clone
    + Copy
    + Send
    + Sync
    + Zero
    + Add<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + AddAssign<Self>
    + Sub<Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + SubAssign<Self>
    + Neg<Output = Self>
{
}

/// Core ring arithmetic: additive group plus multiplication and `one`.
pub trait RingCore:
    AdditiveGroup
    + One
    + PartialEq
    + Eq
    + Default
    + Debug
    + Display
    + Hash
    + Mul<Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + MulAssign<Self>
    + Sum<Self>
    + for<'a> Sum<&'a Self>
    + Product<Self>
    + for<'a> Product<&'a Self>
{
    /// Returns `self * self`.
    #[inline]
    fn square(&self) -> Self {
        *self * *self
    }
}

/// Ring-level inversion capability with explicit zero handling.
pub trait Invertible: RingCore {
    /// Multiplicative inverse, or `None` for the zero element.
    fn inverse(&self) -> Option<Self>;

    /// Multiplicative inverse with zero mapped to zero.
    #[inline]
    fn inv_or_zero(self) -> Self {
        self.inverse().unwrap_or_else(Self::zero)
    }
}

/// Core field capability: a ring that is also invertible.
pub trait FieldCore: RingCore + Invertible {}

/// Embed primitive integer values into a scalar object.
///
/// Native Akita definition (mirrors Jolt's `FromPrimitiveInt`); `jolt-compat`
/// forwards the Jolt trait to this. Implementors provide the 64/128-bit
/// constructors; the narrower widths default through them.
pub trait FromPrimitiveInt: Sized {
    /// Maps `true`/`false` to `1`/`0`.
    #[inline]
    fn from_bool(v: bool) -> Self {
        if v {
            Self::from_u64(1)
        } else {
            Self::from_u64(0)
        }
    }

    /// Embeds a `u8`.
    #[inline]
    fn from_u8(v: u8) -> Self {
        Self::from_u64(v as u64)
    }

    /// Embeds an `i8`.
    #[inline]
    fn from_i8(v: i8) -> Self {
        Self::from_i64(v as i64)
    }

    /// Embeds a `u16`.
    #[inline]
    fn from_u16(v: u16) -> Self {
        Self::from_u64(v as u64)
    }

    /// Embeds an `i16`.
    #[inline]
    fn from_i16(v: i16) -> Self {
        Self::from_i64(v as i64)
    }

    /// Embeds a `u32`.
    #[inline]
    fn from_u32(v: u32) -> Self {
        Self::from_u64(v as u64)
    }

    /// Embeds an `i32`.
    #[inline]
    fn from_i32(v: i32) -> Self {
        Self::from_i64(v as i64)
    }

    /// Embeds a `u64`.
    fn from_u64(v: u64) -> Self;
    /// Embeds an `i64`.
    fn from_i64(v: i64) -> Self;
    /// Embeds a `u128`.
    fn from_u128(v: u128) -> Self;
    /// Embeds an `i128`.
    fn from_i128(v: i128) -> Self;
}

/// Multiplication by powers of two.
pub trait MulPow2: RingCore + FromPrimitiveInt {
    /// Multiplies this ring element by the integer `2^pow`.
    #[inline]
    fn mul_pow_2(&self, pow: usize) -> Self {
        assert!(pow <= 255, "pow > 255");
        let mut res = *self;
        let mut p = pow;
        while p >= 64 {
            res *= Self::from_u64(1 << 63);
            p -= 63;
        }
        res * Self::from_u64(1 << p)
    }
}

/// Multiplication by primitive integer scalars.
pub trait MulPrimitiveInt: RingCore + FromPrimitiveInt {
    /// Multiplies by a `u64`.
    #[inline(always)]
    fn mul_u64(&self, n: u64) -> Self {
        *self * Self::from_u64(n)
    }

    /// Multiplies by an `i64`.
    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        *self * Self::from_i64(n)
    }

    /// Multiplies by a `u128`.
    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        *self * Self::from_u128(n)
    }

    /// Multiplies by an `i128`.
    #[inline(always)]
    fn mul_i128(&self, n: i128) -> Self {
        *self * Self::from_i128(n)
    }
}

/// Fixed byte-size metadata for canonical encodings.
pub trait FixedByteSize {
    /// Byte length of the fixed-size encoding.
    const NUM_BYTES: usize;
}

/// Canonical little-endian byte encoding.
pub trait CanonicalBytes: Sized + FixedByteSize {
    /// Writes the canonical little-endian encoding into `out`.
    ///
    /// `out` must be exactly [`FixedByteSize::NUM_BYTES`] long; implementations
    /// may panic otherwise. This is a fixed, caller-sized buffer contract — not a
    /// proof-controlled length — so verifier-reachable callers (e.g. the transcript
    /// sponge) must allocate `out` to `NUM_BYTES`; prefer [`Self::to_bytes_le_vec`]
    /// or [`FixedBytes::to_bytes_array`], which size the buffer for you.
    fn to_bytes_le(&self, out: &mut [u8]);

    /// Returns the canonical little-endian encoding as a vector.
    #[inline]
    fn to_bytes_le_vec(&self) -> Vec<u8> {
        let mut out = vec![0u8; Self::NUM_BYTES];
        self.to_bytes_le(&mut out);
        out
    }
}

/// Reducing little-endian byte constructor.
pub trait ReducingBytes: Sized {
    /// Deserializes little-endian bytes by reducing into this type.
    fn from_le_bytes_mod_order(bytes: &[u8]) -> Self;
}

/// Fixed-array convenience API for canonical field/value encodings.
pub trait FixedBytes<const N: usize>: CanonicalBytes + ReducingBytes + FixedByteSize {
    /// Returns the canonical fixed-size byte encoding.
    #[inline]
    fn to_bytes_array(&self) -> [u8; N] {
        debug_assert_eq!(Self::NUM_BYTES, N);
        let mut out = [0u8; N];
        self.to_bytes_le(&mut out);
        out
    }

    /// Reducing constructor from a fixed-size byte array.
    #[inline]
    fn from_bytes_array(bytes: &[u8; N]) -> Self {
        Self::from_le_bytes_mod_order(bytes)
    }
}

/// Significant-bit introspection for canonical representatives.
pub trait CanonicalBitLength {
    /// Number of significant bits in this element's canonical representative
    /// (zero has zero significant bits).
    fn num_bits(&self) -> u32;
}

/// Checked extraction of canonical representatives that fit in `u64`.
pub trait CanonicalU64 {
    /// Returns the canonical representative as `u64` if it fits.
    fn to_canonical_u64_checked(&self) -> Option<u64>;
}

/// RNG-backed sampling for tests and witnesses.
pub trait RandomSampling {
    /// Samples a random element.
    fn random<R: RngCore>(rng: &mut R) -> Self;
}

/// Fiat-Shamir challenge decoding from squeezed transcript bytes.
pub trait TranscriptChallenge:
    Sized + Copy + Default + PartialEq + Eq + Debug + Hash + Sync + Send + 'static
{
    /// Constructs a challenge from transcript bytes.
    fn from_challenge_bytes(bytes: &[u8]) -> Self;
}

/// Accumulates additive values with potentially deferred reduction.
pub trait AdditiveAccumulator: Default + Copy + Send + Sync {
    /// The element type this accumulator reduces to.
    type Element: AdditiveGroup;

    /// Adds one element into the accumulator.
    fn add(&mut self, value: Self::Element);

    /// Merges another accumulator's partial sum into this one.
    fn merge(&mut self, other: Self);

    /// Finalizes: reduces the accumulated value to an element.
    fn reduce(self) -> Self::Element;
}

/// Accumulates products with potentially deferred modular reduction.
///
/// `fmadd` must equal `result += a * b` in the field, `merge` must equal adding
/// another accumulator's partial result, and `reduce` must return the field
/// element equal to the accumulated sum of products.
pub trait RingAccumulator: AdditiveAccumulator
where
    Self::Element: RingCore + FromPrimitiveInt,
{
    /// Fused multiply-add: `self += a * b` without intermediate reduction.
    fn fmadd(&mut self, a: Self::Element, b: Self::Element);

    /// Fused multiply-add with a `u8` scalar: `self += a * F::from(b)`.
    #[inline]
    fn fmadd_u8(&mut self, a: Self::Element, b: u8) {
        self.fmadd(a, Self::Element::from_u8(b));
    }

    /// Fused multiply-add with a `u64` scalar: `self += a * F::from(b)`.
    #[inline]
    fn fmadd_u64(&mut self, a: Self::Element, b: u64) {
        self.fmadd(a, Self::Element::from_u64(b));
    }

    /// Fused multiply-add with an `i64` scalar: `self += a * F::from(b)`.
    #[inline]
    fn fmadd_i64(&mut self, a: Self::Element, b: i64) {
        self.fmadd(a, Self::Element::from_i64(b));
    }

    /// Fused multiply-add with a `bool` scalar: `self += a` when `b` is true.
    #[inline]
    fn fmadd_bool(&mut self, a: Self::Element, b: bool) {
        if b {
            self.fmadd(a, <Self::Element as One>::one());
        }
    }
}

/// Associates an additive redundant accumulator with an element type.
pub trait WithAccumulator: AdditiveGroup {
    /// Accumulator type whose element is `Self`.
    type Accumulator: AdditiveAccumulator<Element = Self>;
}

/// Naive accumulator using standard field arithmetic.
///
/// Every [`fmadd`](RingAccumulator::fmadd) performs a full modular multiply and
/// add; the fallback for fields without a wide-integer optimization.
#[derive(Clone, Copy)]
pub struct NaiveAccumulator<R: RingCore + FromPrimitiveInt>(R);

impl<R: RingCore + FromPrimitiveInt> Default for NaiveAccumulator<R> {
    #[inline]
    fn default() -> Self {
        Self(R::zero())
    }
}

impl<R: RingCore + FromPrimitiveInt> AdditiveAccumulator for NaiveAccumulator<R> {
    type Element = R;

    #[inline]
    fn add(&mut self, value: R) {
        self.0 += value;
    }

    #[inline]
    fn merge(&mut self, other: Self) {
        self.0 += other.0;
    }

    #[inline]
    fn reduce(self) -> R {
        self.0
    }
}

impl<R: RingCore + FromPrimitiveInt> RingAccumulator for NaiveAccumulator<R> {
    #[inline]
    fn fmadd(&mut self, a: R, b: R) {
        self.0 += a * b;
    }
}

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
/// in `src/fft.rs` and re-derived against
/// `find_primitive_nth_root` so the constants cannot drift.
pub trait SmoothFftField: CanonicalField + PseudoMersenneField {
    /// Order of the largest smooth multiplicative subgroup we support
    /// for FFT. Must divide `p − 1`.
    const SMOOTH_SUBGROUP_ORDER: usize;

    /// Canonical `u128` representation of a primitive
    /// `SMOOTH_SUBGROUP_ORDER`-th root of unity in the field.
    const SMOOTH_OMEGA: u128;
}
