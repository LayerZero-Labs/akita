//! Solinas-style reduction for 128-bit primes `p = 2^128 - c` with sparse `c`.
//!
//! This is an opt-in field backend. The generic [`crate::algebra::fields::Fp128`] remains
//! unchanged and can act as a correctness oracle.
//!
//! ## Naming convention for built-in primes
//!
//! The built-in type names encode the **signed terms as they appear in the modulus `p`**
//! (excluding the leading `+2^128` term). For example, `Prime128M13M4P0` denotes:
//!
//! `p = 2^128 - 2^13 - 2^4 + 2^0`.
//!
//! Internally, Solinas reduction uses the offset `c = 2^128 - p`, so the signed
//! decomposition of `c` is the sign-flipped version of the one encoded in `p`.

use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::{CanonicalField, FieldCore, FieldSampling, Invertible, PseudoMersenneField};
use rand_core::RngCore;
use std::io::{Read, Write};
use std::marker::PhantomData;

/// Parameters for a Solinas-reducible 128-bit prime field `p = 2^128 - c`.
///
/// Contract (enforced by the `solinas_prime!` macro for the built-in primes):
/// - `P` is odd and nonzero
/// - `C = 2^128 - P` (computed as `0u128.wrapping_sub(P)`)
/// - `C < 2^64`, sufficient for the two-fold Solinas reduction in [`SolinasFp128`]
pub trait SolinasParams: 'static + Copy + Send + Sync {
    /// Modulus `p`.
    const P: u128;
    /// Offset `c = 2^128 - p`.
    const C: u128;
}

/// Pack two u64 limbs into a `[u64; 2]` (lo, hi).
#[inline(always)]
const fn pack(lo: u64, hi: u64) -> [u64; 2] {
    [lo, hi]
}

/// Convert u128 to `[u64; 2]` limb representation.
#[inline(always)]
const fn from_u128(x: u128) -> [u64; 2] {
    [x as u64, (x >> 64) as u64]
}

/// Convert `[u64; 2]` limb representation to u128.
#[inline(always)]
const fn to_u128(x: [u64; 2]) -> u128 {
    x[0] as u128 | (x[1] as u128) << 64
}

/// `a + b·c + carry` widening to 128 bits; returns `(lo64, hi64)`.
#[inline(always)]
fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let ret = a as u128 + (b as u128) * (c as u128) + carry as u128;
    (ret as u64, (ret >> 64) as u64)
}

/// 128-bit prime field with Solinas-folding reduction for `p = 2^128 - c`.
///
/// Internally stored as `[u64; 2]` (lo, hi) for 8-byte alignment and
/// direct access to individual limbs without shifting.
#[derive(Debug, Clone, Copy, Default)]
pub struct SolinasFp128<M: SolinasParams>(pub(crate) [u64; 2], PhantomData<M>);

impl<M: SolinasParams> PartialEq for SolinasFp128<M> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<M: SolinasParams> Eq for SolinasFp128<M> {}

impl<M: SolinasParams> SolinasFp128<M> {
    const C_LO: u64 = M::C as u64;

    /// Create an element from a canonical representative in `[0, p)`.
    #[inline]
    pub fn from_canonical_u128(x: u128) -> Self {
        debug_assert!(x < M::P);
        Self(from_u128(x), PhantomData)
    }

    /// Return the canonical representative in `[0, p)`.
    #[inline]
    pub fn to_canonical_u128(self) -> u128 {
        to_u128(self.0)
    }

    #[inline(always)]
    fn add_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let (s, carry) = to_u128(a).overflowing_add(to_u128(b));
        let (reduced, borrow) = s.overflowing_sub(M::P);
        from_u128(if carry | !borrow { reduced } else { reduced.wrapping_add(M::P) })
    }

    #[inline(always)]
    fn sub_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let (diff, borrow) = to_u128(a).overflowing_sub(to_u128(b));
        from_u128(if borrow { diff.wrapping_add(M::P) } else { diff })
    }

    /// Fold 2 + canonicalize: reduce `[t0, t1] + t2·2^128` into `[0, p)`.
    #[inline(always)]
    fn fold2_canonicalize(t0: u64, t1: u64, t2: u64) -> [u64; 2] {
        let c = Self::C_LO;
        let ct2 = (c as u128) * (t2 as u128);
        let base = (t1 as u128) << 64 | t0 as u128;
        let (s, overflow) = base.overflowing_add(ct2);
        // Overflow → true value is s + 2^128 ≡ s + C (mod p).
        let s = s.wrapping_add((overflow as u128).wrapping_neg() & M::C);

        // Canonicalize: since P = 2^128 − C, subtracting P is adding C.
        let (reduced, carry) = s.overflowing_add(M::C);
        from_u128(if carry { reduced } else { s })
    }

    #[inline(always)]
    fn mul_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let (a0, a1) = (a[0], a[1]);
        let (b0, b1) = (b[0], b[1]);
        let c = Self::C_LO;

        // Schoolbook 2×2 → 4 u64 limbs.
        let (r0, carry) = mac(0, a0, b0, 0);
        let (r1, r2) = mac(0, a0, b1, carry);
        let (r1, carry) = mac(r1, a1, b0, 0);
        let (r2, r3) = mac(r2, a1, b1, carry);

        // Solinas fold 1: [t0,t1,t2] = [r0,r1] + c·[r2,r3].
        let (t0, carry) = mac(r0, c, r2, 0);
        let (t1, t2) = mac(r1, c, r3, carry);

        Self::fold2_canonicalize(t0, t1, t2)
    }

    #[inline(always)]
    fn sqr_raw(a: [u64; 2]) -> [u64; 2] {
        Self::mul_raw(a, a)
    }

    /// Squaring, equivalent to `self * self`.
    #[inline(always)]
    pub fn square(self) -> Self {
        Self(Self::sqr_raw(self.0), PhantomData)
    }

    fn pow_u128(self, mut exp: u128) -> Self {
        let mut base = self;
        let mut acc = Self::one();
        while exp > 0 {
            if (exp & 1) == 1 {
                acc = acc * base;
            }
            base = Self(Self::sqr_raw(base.0), PhantomData);
            exp >>= 1;
        }
        acc
    }
}

impl<M: SolinasParams> std::ops::Add for SolinasFp128<M> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(Self::add_raw(self.0, rhs.0), PhantomData)
    }
}

impl<M: SolinasParams> std::ops::Sub for SolinasFp128<M> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(Self::sub_raw(self.0, rhs.0), PhantomData)
    }
}

impl<M: SolinasParams> std::ops::Mul for SolinasFp128<M> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(Self::mul_raw(self.0, rhs.0), PhantomData)
    }
}

impl<M: SolinasParams> std::ops::Neg for SolinasFp128<M> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        let zero = pack(0, 0);
        Self(Self::sub_raw(zero, self.0), PhantomData)
    }
}

impl<'a, M: SolinasParams> std::ops::Add<&'a Self> for SolinasFp128<M> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, M: SolinasParams> std::ops::Sub<&'a Self> for SolinasFp128<M> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, M: SolinasParams> std::ops::Mul<&'a Self> for SolinasFp128<M> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<M: SolinasParams> Valid for SolinasFp128<M> {
    fn check(&self) -> Result<(), SerializationError> {
        if to_u128(self.0) < M::P {
            Ok(())
        } else {
            Err(SerializationError::InvalidData(
                "SolinasFp128 out of range".into(),
            ))
        }
    }
}

impl<M: SolinasParams> HachiSerialize for SolinasFp128<M> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        to_u128(self.0).serialize_with_mode(&mut writer, Compress::No)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        16
    }
}

impl<M: SolinasParams> HachiDeserialize for SolinasFp128<M> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let x = u128::deserialize_with_mode(&mut reader, Compress::No, validate)?;
        if matches!(validate, Validate::Yes) && x >= M::P {
            return Err(SerializationError::InvalidData(
                "SolinasFp128 out of range".to_string(),
            ));
        }

        // If validation is disabled, reduce without division.
        // For moduli `p = 2^128 - c` with `c < 2^64`, we have `p > 2^127`,
        // hence any `u128` input is in `[0, 2p)` and one conditional subtract suffices.
        let out = if matches!(validate, Validate::Yes) {
            x
        } else {
            let (sub, borrow) = x.overflowing_sub(M::P);
            if borrow { x } else { sub }
        };
        Ok(Self(from_u128(out), PhantomData))
    }
}

impl<M: SolinasParams> FieldCore for SolinasFp128<M> {
    fn zero() -> Self {
        Self(pack(0, 0), PhantomData)
    }

    fn one() -> Self {
        Self(pack(1, 0), PhantomData)
    }

    fn is_zero(&self) -> bool {
        self.0 == [0, 0]
    }

    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }

    fn sub(&self, rhs: &Self) -> Self {
        *self - *rhs
    }

    fn mul(&self, rhs: &Self) -> Self {
        *self * *rhs
    }

    fn inv(self) -> Option<Self> {
        let inv = self.inv_or_zero();
        if self.is_zero() {
            None
        } else {
            Some(inv)
        }
    }
}

impl<M: SolinasParams> Invertible for SolinasFp128<M> {
    fn inv_or_zero(self) -> Self {
        let candidate = self.pow_u128(M::P.wrapping_sub(2));
        let v = to_u128(self.0);
        let nz = ((v | v.wrapping_neg()) >> 127) & 1;
        let mask = 0u128.wrapping_sub(nz);
        let masked = to_u128(candidate.0) & mask;
        Self(from_u128(masked), PhantomData)
    }
}

impl<M: SolinasParams> FieldSampling for SolinasFp128<M> {
    fn sample<R: RngCore>(rng: &mut R) -> Self {
        // Rejection sampling without division. Acceptance probability is ~1 - C/2^128.
        loop {
            let lo = rng.next_u64();
            let hi = rng.next_u64();
            let x = lo as u128 | (hi as u128) << 64;
            if x < M::P {
                return Self(pack(lo, hi), PhantomData);
            }
        }
    }
}

impl<M: SolinasParams> CanonicalField for SolinasFp128<M> {
    fn from_u64(val: u64) -> Self {
        Self::from_canonical_u128_reduced(val as u128)
    }

    fn from_i64(val: i64) -> Self {
        if val >= 0 {
            Self::from_u64(val as u64)
        } else {
            -Self::from_u64((-val) as u64)
        }
    }

    fn to_canonical_u128(self) -> u128 {
        to_u128(self.0)
    }

    fn from_canonical_u128_checked(val: u128) -> Option<Self> {
        if val < M::P {
            Some(Self(from_u128(val), PhantomData))
        } else {
            None
        }
    }

    fn from_canonical_u128_reduced(val: u128) -> Self {
        let (sub, borrow) = val.overflowing_sub(M::P);
        Self(from_u128(if borrow { val } else { sub }), PhantomData)
    }
}

impl<M: SolinasParams> PseudoMersenneField for SolinasFp128<M> {
    const MODULUS_BITS: u32 = 128;
    const MODULUS_OFFSET: u128 = M::C;
}

/// Generate a `SolinasParams` implementation from a signed power-of-two decomposition of `C`.
///
/// Example (if `c = 2^13 + 2^4 - 2^0`, then `p = 2^128 - 2^13 - 2^4 + 2^0`):
///
/// `solinas_prime!(Prime128M13M4P0Params, P, [(13, +1), (4, +1), (0, -1)]);`
macro_rules! solinas_prime {
    ($name:ident, $p:expr, [$(($shift:expr, $sign:tt 1)),+ $(,)?]) => {
        #[derive(Debug, Clone, Copy)]
        #[doc = "Auto-generated Solinas prime parameter set."]
        pub struct $name;

        impl SolinasParams for $name {
            const P: u128 = $p;
            const C: u128 = 0u128.wrapping_sub(Self::P);
        }

        const _: () = {
            const P: u128 = <$name as SolinasParams>::P;
            const C: u128 = <$name as SolinasParams>::C;
            assert!(P.wrapping_add(C) == 0);
            assert!(P != 0);
            assert!((P & 1) == 1);
            assert!(C < (1u128 << 64));

            let mut c_terms: u128 = 0;
            $(
                assert!($shift < 128);
                let t: u128 = 1u128 << $shift;
                c_terms = solinas_prime!(@cterm c_terms, t, $sign);
            )+
            assert!(c_terms == C);
        };
    };

    (@cterm $acc:expr, $t:expr, +) => {{
        $acc.wrapping_add($t)
    }};
    (@cterm $acc:expr, $t:expr, -) => {{
        $acc.wrapping_sub($t)
    }};
}

// ---- Built-in sparse primes (descriptive names encode the decomposition of C) ----
//
// Note: Names encode the signed terms in `p`; the `(shift, ±1)` lists encode the signed terms in `c`.

solinas_prime!(
    Prime128M13M4P0Params,
    0xffffffffffffffffffffffffffffdff1u128,
    [(13, +1), (4, +1), (0, -1)]
);
solinas_prime!(
    Prime128M37P3P0Params,
    0xffffffffffffffffffffffe000000009u128,
    [(37, +1), (3, -1), (0, -1)]
);
solinas_prime!(
    Prime128M52M3P0Params,
    0xffffffffffffffffffeffffffffffff9u128,
    [(52, +1), (3, +1), (0, -1)]
);
solinas_prime!(
    Prime128M54P4P0Params,
    0xffffffffffffffffffc0000000000011u128,
    [(54, +1), (4, -1), (0, -1)]
);
solinas_prime!(
    Prime128M8M4M1M0Params,
    0xfffffffffffffffffffffffffffffeedu128,
    [(8, +1), (4, +1), (1, +1), (0, +1)]
);

/// Field element modulo `p = 2^128 - 2^13 - 2^4 + 2^0`.
pub type Prime128M13M4P0 = SolinasFp128<Prime128M13M4P0Params>;
/// Field element modulo `p = 2^128 - 2^37 + 2^3 + 2^0`.
pub type Prime128M37P3P0 = SolinasFp128<Prime128M37P3P0Params>;
/// Field element modulo `p = 2^128 - 2^52 - 2^3 + 2^0`.
pub type Prime128M52M3P0 = SolinasFp128<Prime128M52M3P0Params>;
/// Field element modulo `p = 2^128 - 2^54 + 2^4 + 2^0`.
pub type Prime128M54P4P0 = SolinasFp128<Prime128M54P4P0Params>;
/// Field element modulo `p = 2^128 - 2^8 - 2^4 - 2^1 - 2^0`.
pub type Prime128M8M4M1M0 = SolinasFp128<Prime128M8M4M1M0Params>;
