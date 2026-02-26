//! 128-bit prime field for primes of the form `p = 2^128 − c` with `c < 2^64`.
//!
//! Uses Solinas-style two-fold reduction: no Montgomery form, ~23 cycles/mul
//! on both AArch64 and x86-64.  The offset `c` is computed at compile time
//! from the const-generic modulus `P`.
//!
//! ## Naming convention for built-in primes
//!
//! The built-in type names encode the **signed terms as they appear in the
//! modulus `p`** (excluding the leading `+2^128` term).  For example,
//! `Prime128M13M4P0` denotes `p = 2^128 − 2^13 − 2^4 + 2^0`.

use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::{CanonicalField, FieldCore, FieldSampling, Invertible, PseudoMersenneField};
use rand_core::RngCore;
use std::io::{Read, Write};

// ---------------------------------------------------------------------------
// Limb helpers
// ---------------------------------------------------------------------------

/// Pack two u64 limbs into `[lo, hi]`.
#[inline(always)]
const fn pack(lo: u64, hi: u64) -> [u64; 2] {
    [lo, hi]
}

/// Convert `u128` → `[u64; 2]`.
#[inline(always)]
const fn from_u128(x: u128) -> [u64; 2] {
    [x as u64, (x >> 64) as u64]
}

/// Convert `[u64; 2]` → `u128`.
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

// ---------------------------------------------------------------------------
// Fp128
// ---------------------------------------------------------------------------

/// 128-bit prime field element for primes `p = 2^128 − c` with `c < 2^64`.
///
/// Stored as `[u64; 2]` (lo, hi) for 8-byte alignment and direct limb access.
///
/// The offset `c = 2^128 − p` and all derived constants are computed at
/// compile time from the const-generic `P`.  Instantiating `Fp128` with a
/// modulus that is not of this form is a compile-time error.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fp128<const P: u128>(pub(crate) [u64; 2]);

impl<const P: u128> PartialEq for Fp128<P> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const P: u128> Eq for Fp128<P> {}

impl<const P: u128> Fp128<P> {
    /// Offset `c = 2^128 − p`.  Validated at compile time.
    const C: u128 = {
        let c = 0u128.wrapping_sub(P);
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        assert!(c < (1u128 << 64), "P must be 2^128 - c with c < 2^64");
        // Fused overflow+canonicalize requires C(C+1) < P.
        assert!(
            c * (c + 1) < P,
            "C(C+1) < P required for fused canonicalize"
        );
        c
    };
    const C_LO: u64 = Self::C as u64;

    /// Create from a canonical representative in `[0, p)`.
    #[inline]
    pub fn from_canonical_u128(x: u128) -> Self {
        debug_assert!(x < P);
        Self(from_u128(x))
    }

    /// Return the canonical representative in `[0, p)`.
    #[inline]
    pub fn to_canonical_u128(self) -> u128 {
        to_u128(self.0)
    }

    #[inline(always)]
    fn add_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let (s, carry) = to_u128(a).overflowing_add(to_u128(b));
        let (reduced, borrow) = s.overflowing_sub(P);
        from_u128(if carry | !borrow {
            reduced
        } else {
            reduced.wrapping_add(P)
        })
    }

    #[inline(always)]
    fn sub_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let (diff, borrow) = to_u128(a).overflowing_sub(to_u128(b));
        from_u128(if borrow { diff.wrapping_add(P) } else { diff })
    }

    /// Fold 2 + canonicalize: reduce `[t0, t1] + t2·2^128` into `[0, p)`.
    ///
    /// Correctness argument for the fused overflow+canonicalize:
    ///
    /// Let `v = base + C·t2` (mathematical, not mod 2^128).
    /// From the fold-1 mac chain, `t2 ≤ C`, so `C·t2 ≤ C²`.
    ///
    /// - **No overflow** (`v < 2^128`): `s = v`, and the standard
    ///   canonicalize applies — `s + C` carries iff `s ≥ P`.
    /// - **Overflow** (`v ≥ 2^128`): `s = v − 2^128`, so `s < C·t2 ≤ C²`.
    ///   The correct reduced value is `s + C` (since `2^128 ≡ C mod P`).
    ///   Because `s + C < C² + C = C(C+1)` and `C(C+1) < P` for all
    ///   `C < 2^64`, the value `s + C` is already in `[0, P)` — no
    ///   further canonicalization is needed, and `s + C < 2^128` so the
    ///   add does NOT carry.
    ///
    /// Therefore `if (overflow | carry) { s + C } else { s }` is correct
    /// in both cases, fusing the overflow correction with canonicalization.
    #[inline(always)]
    fn fold2_canonicalize(t0: u64, t1: u64, t2: u64) -> [u64; 2] {
        let ct2 = (Self::C_LO as u128) * (t2 as u128);
        let base = (t1 as u128) << 64 | t0 as u128;
        let (s, overflow) = base.overflowing_add(ct2);
        let (reduced, carry) = s.overflowing_add(Self::C);
        from_u128(if overflow | carry { reduced } else { s })
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
        Self(Self::sqr_raw(self.0))
    }

    fn pow_u128(self, mut exp: u128) -> Self {
        let mut base = self;
        let mut acc = Self::one();
        while exp > 0 {
            if (exp & 1) == 1 {
                acc = acc * base;
            }
            base = Self(Self::sqr_raw(base.0));
            exp >>= 1;
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// Operator impls
// ---------------------------------------------------------------------------

impl<const P: u128> std::ops::Add for Fp128<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(Self::add_raw(self.0, rhs.0))
    }
}

impl<const P: u128> std::ops::Sub for Fp128<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(Self::sub_raw(self.0, rhs.0))
    }
}

impl<const P: u128> std::ops::Mul for Fp128<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(Self::mul_raw(self.0, rhs.0))
    }
}

impl<const P: u128> std::ops::Neg for Fp128<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(Self::sub_raw(pack(0, 0), self.0))
    }
}

impl<'a, const P: u128> std::ops::Add<&'a Self> for Fp128<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, const P: u128> std::ops::Sub<&'a Self> for Fp128<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, const P: u128> std::ops::Mul<&'a Self> for Fp128<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

impl<const P: u128> Valid for Fp128<P> {
    fn check(&self) -> Result<(), SerializationError> {
        if to_u128(self.0) < P {
            Ok(())
        } else {
            Err(SerializationError::InvalidData("Fp128 out of range".into()))
        }
    }
}

impl<const P: u128> HachiSerialize for Fp128<P> {
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

impl<const P: u128> HachiDeserialize for Fp128<P> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let x = u128::deserialize_with_mode(&mut reader, Compress::No, validate)?;
        if matches!(validate, Validate::Yes) && x >= P {
            return Err(SerializationError::InvalidData(
                "Fp128 out of range".to_string(),
            ));
        }

        // Without validation, reduce without division.
        // For `p = 2^128 − c` with `c < 2^64` we have `p > 2^127`,
        // so any `u128` is in `[0, 2p)` and one conditional subtract suffices.
        let out = if matches!(validate, Validate::Yes) {
            x
        } else {
            let (sub, borrow) = x.overflowing_sub(P);
            if borrow {
                x
            } else {
                sub
            }
        };
        Ok(Self(from_u128(out)))
    }
}

// ---------------------------------------------------------------------------
// Core field traits
// ---------------------------------------------------------------------------

impl<const P: u128> FieldCore for Fp128<P> {
    fn zero() -> Self {
        Self(pack(0, 0))
    }

    fn one() -> Self {
        Self(pack(1, 0))
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

impl<const P: u128> Invertible for Fp128<P> {
    fn inv_or_zero(self) -> Self {
        let candidate = self.pow_u128(P.wrapping_sub(2));
        let v = to_u128(self.0);
        let nz = ((v | v.wrapping_neg()) >> 127) & 1;
        let mask = 0u128.wrapping_sub(nz);
        let masked = to_u128(candidate.0) & mask;
        Self(from_u128(masked))
    }
}

impl<const P: u128> FieldSampling for Fp128<P> {
    fn sample<R: RngCore>(rng: &mut R) -> Self {
        loop {
            let lo = rng.next_u64();
            let hi = rng.next_u64();
            let x = lo as u128 | (hi as u128) << 64;
            if x < P {
                return Self(pack(lo, hi));
            }
        }
    }
}

impl<const P: u128> CanonicalField for Fp128<P> {
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
        if val < P {
            Some(Self(from_u128(val)))
        } else {
            None
        }
    }

    fn from_canonical_u128_reduced(val: u128) -> Self {
        let (sub, borrow) = val.overflowing_sub(P);
        Self(from_u128(if borrow { val } else { sub }))
    }
}

impl<const P: u128> PseudoMersenneField for Fp128<P> {
    const MODULUS_BITS: u32 = 128;
    const MODULUS_OFFSET: u128 = Self::C;
}

// ---------------------------------------------------------------------------
// Built-in prime type aliases
// ---------------------------------------------------------------------------

/// `p = 2^128 − 2^13 − 2^4 + 2^0`  (C = 8207).
pub type Prime128M13M4P0 = Fp128<0xffffffffffffffffffffffffffffdff1>;
/// `p = 2^128 − 2^37 + 2^3 + 2^0`  (C = 137438953463).
pub type Prime128M37P3P0 = Fp128<0xffffffffffffffffffffffe000000009>;
/// `p = 2^128 − 2^52 − 2^3 + 2^0`  (C = 4503599627370487).
pub type Prime128M52M3P0 = Fp128<0xffffffffffffffffffeffffffffffff9>;
/// `p = 2^128 − 2^54 + 2^4 + 2^0`  (C = 18014398509481967).
pub type Prime128M54P4P0 = Fp128<0xffffffffffffffffffc0000000000011>;
/// `p = 2^128 − 2^8 − 2^4 − 2^1 − 2^0`  (C = 275).
pub type Prime128M8M4M1M0 = Fp128<0xfffffffffffffffffffffffffffffeed>;
