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

use std::io::{Read, Write};
use std::ops::{Add, Mul, Neg, Sub};

use rand_core::RngCore;

use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::{CanonicalField, FieldCore, FieldSampling, Invertible, PseudoMersenneField};

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

/// `a * b` widening to 128 bits; returns `(lo64, hi64)`.
#[inline(always)]
fn mul64_wide(a: u64, b: u64) -> (u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        // Safety: this block is compiled only when `bmi2` is enabled.
        return unsafe { mul64_wide_bmi2(a, b) };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let prod = (a as u128) * (b as u128);
        (prod as u64, (prod >> 64) as u64)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn mul64_wide_bmi2(a: u64, b: u64) -> (u64, u64) {
    let mut hi = 0;
    // Safety: caller guarantees CPU support via cfg-gated compilation.
    let lo = unsafe { std::arch::x86_64::_mulx_u64(a, b, &mut hi) };
    (lo, hi)
}

#[inline(always)]
const fn is_pow2_u64(x: u64) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

#[inline(always)]
const fn log2_pow2_u64(mut x: u64) -> u32 {
    let mut k = 0u32;
    while x > 1 {
        x >>= 1;
        k += 1;
    }
    k
}

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
    /// +1 means `C = 2^a + 1`, -1 means `C = 2^a - 1`, 0 means generic.
    const C_SHIFT_KIND: i8 = {
        let c = Self::C_LO;
        if c > 1 && is_pow2_u64(c - 1) {
            1
        } else if c == u64::MAX || is_pow2_u64(c + 1) {
            -1
        } else {
            0
        }
    };
    const C_SHIFT: u32 = {
        let c = Self::C_LO;
        if Self::C_SHIFT_KIND == 1 {
            log2_pow2_u64(c - 1)
        } else if Self::C_SHIFT_KIND == -1 {
            if c == u64::MAX {
                64
            } else {
                log2_pow2_u64(c + 1)
            }
        } else {
            0
        }
    };

    /// Multiply by `C = 2^128 - P`. For `C = 2^a ± 1`, this is shift/add or
    /// shift/sub only; otherwise it falls back to generic widening multiply.
    #[inline(always)]
    fn mul_c_wide(x: u64) -> (u64, u64) {
        if Self::C_SHIFT_KIND == 1 {
            let v = ((x as u128) << Self::C_SHIFT) + x as u128;
            (v as u64, (v >> 64) as u64)
        } else if Self::C_SHIFT_KIND == -1 {
            let v = ((x as u128) << Self::C_SHIFT) - x as u128;
            (v as u64, (v >> 64) as u64)
        } else {
            mul64_wide(Self::C_LO, x)
        }
    }

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
        let (ct2_lo, ct2_hi) = Self::mul_c_wide(t2);

        let (s0, carry0) = t0.overflowing_add(ct2_lo);
        let (s1a, carry1a) = t1.overflowing_add(ct2_hi);
        let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
        let overflow = carry1a | carry1b;

        let (r0, carry2) = s0.overflowing_add(Self::C_LO);
        let (r1, carry3) = s1.overflowing_add(carry2 as u64);

        pack(
            if overflow | carry3 { r0 } else { s0 },
            if overflow | carry3 { r1 } else { s1 },
        )
    }

    /// Solinas fold for exactly 4 limbs: `[r0,r1] + C·[r2,r3]` → 3 limbs,
    /// then `fold2_canonicalize`.
    #[inline(always)]
    fn reduce_4(r0: u64, r1: u64, r2: u64, r3: u64) -> [u64; 2] {
        let (cr2_lo, cr2_hi) = Self::mul_c_wide(r2);
        let (cr3_lo, cr3_hi) = Self::mul_c_wide(r3);

        let t0_sum = r0 as u128 + cr2_lo as u128;
        let t0 = t0_sum as u64;
        let carryf = (t0_sum >> 64) as u64;

        let t1_sum = r1 as u128 + cr2_hi as u128 + cr3_lo as u128 + carryf as u128;
        let t1 = t1_sum as u64;

        let t2_sum = cr3_hi as u128 + (t1_sum >> 64);
        let t2 = t2_sum as u64;
        debug_assert_eq!(t2_sum >> 64, 0);

        Self::fold2_canonicalize(t0, t1, t2)
    }

    #[inline(always)]
    fn mul_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let [r0, r1, r2, r3] = Self(a).mul_wide(Self(b));
        Self::reduce_4(r0, r1, r2, r3)
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

    /// Extract the canonical `[lo, hi]` limb representation.
    #[inline(always)]
    pub fn to_limbs(self) -> [u64; 2] {
        self.0
    }

    /// 128×64 → 192-bit widening multiply, **no reduction**.
    ///
    /// Returns `[lo, mid, hi]` representing `self · other` as a 192-bit
    /// integer.  Cost: 2 widening `mul64`.
    #[inline(always)]
    pub fn mul_wide_u64(self, other: u64) -> [u64; 3] {
        let (a0, a1) = (self.0[0], self.0[1]);
        let (p0_lo, p0_hi) = mul64_wide(a0, other);
        let (p1_lo, p1_hi) = mul64_wide(a1, other);
        let mid = p0_hi as u128 + p1_lo as u128;
        let hi = p1_hi + (mid >> 64) as u64;
        [p0_lo, mid as u64, hi]
    }

    /// 128×128 → 256-bit widening multiply, **no reduction**.
    ///
    /// Returns `[r0, r1, r2, r3]` representing `self · other` as a 256-bit
    /// integer.  This is the schoolbook 2×2 portion of the Solinas multiply,
    /// without the reduction fold.  Cost: 4 widening `mul64`.
    #[inline(always)]
    pub fn mul_wide(self, other: Self) -> [u64; 4] {
        let (a0, a1) = (self.0[0], self.0[1]);
        let (b0, b1) = (other.0[0], other.0[1]);
        let (p00_lo, p00_hi) = mul64_wide(a0, b0);
        let (p01_lo, p01_hi) = mul64_wide(a0, b1);
        let (p10_lo, p10_hi) = mul64_wide(a1, b0);
        let (p11_lo, p11_hi) = mul64_wide(a1, b1);

        let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
        let r0 = p00_lo;
        let r1 = row1 as u64;
        let carry1 = (row1 >> 64) as u64;

        let row2 = p01_hi as u128 + p10_hi as u128 + p11_lo as u128 + carry1 as u128;
        let r2 = row2 as u64;
        let carry2 = (row2 >> 64) as u64;

        let row3 = p11_hi as u128 + carry2 as u128;
        let r3 = row3 as u64;
        debug_assert_eq!(row3 >> 64, 0);

        [r0, r1, r2, r3]
    }

    /// 128×128 → 256-bit widening multiply with a raw `u128` operand,
    /// **no reduction**.
    #[inline(always)]
    pub fn mul_wide_u128(self, other: u128) -> [u64; 4] {
        self.mul_wide(Self(from_u128(other)))
    }

    /// Reduce an arbitrary-width little-endian limb array to a canonical
    /// field element via iterated Solinas folding.
    ///
    /// Each fold splits at the 128-bit boundary and replaces
    /// `hi · 2^128` with `hi · C`, reducing width by one limb per
    /// iteration.  Supports 0–10 input limbs (up to 640 bits).
    ///
    /// # Panics
    ///
    /// Panics if `limbs.len() > 10`.
    #[inline(always)]
    pub fn solinas_reduce(limbs: &[u64]) -> Self {
        match limbs.len() {
            0 => Self::zero(),
            1 => Self(pack(limbs[0], 0)),
            2 => Self::from_canonical_u128_reduced(to_u128([limbs[0], limbs[1]])),
            3 => Self(Self::fold2_canonicalize(limbs[0], limbs[1], limbs[2])),
            4 => Self(Self::reduce_4(limbs[0], limbs[1], limbs[2], limbs[3])),
            5 => {
                let (l0, l1, l2, l3, l4) = (limbs[0], limbs[1], limbs[2], limbs[3], limbs[4]);
                let (c2_lo, c2_hi) = Self::mul_c_wide(l2);
                let (c3_lo, c3_hi) = Self::mul_c_wide(l3);
                let (c4_lo, c4_hi) = Self::mul_c_wide(l4);

                let s0 = l0 as u128 + c2_lo as u128;
                let s1 = l1 as u128 + c2_hi as u128 + c3_lo as u128 + (s0 >> 64);
                let s2 = c3_hi as u128 + c4_lo as u128 + (s1 >> 64);
                let s3 = c4_hi as u128 + (s2 >> 64);
                debug_assert_eq!(s3 >> 64, 0);

                Self(Self::reduce_4(s0 as u64, s1 as u64, s2 as u64, s3 as u64))
            }
            n => {
                assert!(n <= 10, "solinas_reduce supports at most 10 limbs");
                let mut buf = [0u64; 11];
                buf[..n].copy_from_slice(limbs);
                let mut len = n;
                let c = Self::C_LO;

                while len > 5 {
                    let high_len = len - 2;
                    let mut next = [0u64; 11];

                    let mut carry: u64 = 0;
                    for i in 0..high_len {
                        let wide = c as u128 * buf[i + 2] as u128 + carry as u128;
                        next[i] = wide as u64;
                        carry = (wide >> 64) as u64;
                    }
                    next[high_len] = carry;

                    let s0 = next[0] as u128 + buf[0] as u128;
                    next[0] = s0 as u64;
                    let s1 = next[1] as u128 + buf[1] as u128 + (s0 >> 64);
                    next[1] = s1 as u64;
                    let mut c_out = (s1 >> 64) as u64;
                    for limb in &mut next[2..=high_len] {
                        if c_out == 0 {
                            break;
                        }
                        let s = *limb as u128 + c_out as u128;
                        *limb = s as u64;
                        c_out = (s >> 64) as u64;
                    }
                    debug_assert_eq!(c_out, 0);

                    buf = next;
                    len -= 1;
                    while len > 5 && buf[len - 1] == 0 {
                        len -= 1;
                    }
                }

                Self::solinas_reduce(&buf[..len])
            }
        }
    }
}

impl<const P: u128> Add for Fp128<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(Self::add_raw(self.0, rhs.0))
    }
}

impl<const P: u128> Sub for Fp128<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(Self::sub_raw(self.0, rhs.0))
    }
}

impl<const P: u128> Mul for Fp128<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(Self::mul_raw(self.0, rhs.0))
    }
}

impl<const P: u128> Neg for Fp128<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(Self::sub_raw(pack(0, 0), self.0))
    }
}

impl<'a, const P: u128> Add<&'a Self> for Fp128<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, const P: u128> Sub<&'a Self> for Fp128<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, const P: u128> Mul<&'a Self> for Fp128<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

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
/// `p = 2^128 − 2^18 − 2^0`  (C = 2^18 + 1).
pub type Prime128M18M0 = Fp128<0xfffffffffffffffffffffffffffbffff>;
/// `p = 2^128 − 2^54 + 2^0`  (C = 2^54 − 1).
pub type Prime128M54P0 = Fp128<0xffffffffffffffffffc0000000000001>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FieldSampling, PseudoMersenneField};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_core::RngCore;

    type F = Prime128M8M4M1M0;

    #[test]
    fn to_limbs_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0xdead_beef_cafe_1234);
        for _ in 0..1000 {
            let a: F = FieldSampling::sample(&mut rng);
            assert_eq!(Fp128(a.to_limbs()), a);
        }
    }

    #[test]
    fn mul_wide_u64_matches_full_mul() {
        let mut rng = StdRng::seed_from_u64(0x1122_3344_5566_7788);
        for _ in 0..1000 {
            let a: F = FieldSampling::sample(&mut rng);
            let b = rng.next_u64();
            let expected = a * F::from_u64(b);
            let reduced = F::solinas_reduce(&a.mul_wide_u64(b));
            assert_eq!(reduced, expected);
        }
    }

    #[test]
    fn mul_wide_matches_full_mul() {
        let mut rng = StdRng::seed_from_u64(0xaabb_ccdd_eeff_0011);
        for _ in 0..1000 {
            let a: F = FieldSampling::sample(&mut rng);
            let b: F = FieldSampling::sample(&mut rng);
            let expected = a * b;
            let reduced = F::solinas_reduce(&a.mul_wide(b));
            assert_eq!(reduced, expected);
        }
    }

    #[test]
    fn mul_wide_u128_matches_full_mul() {
        let mut rng = StdRng::seed_from_u64(0x9988_7766_5544_3322);
        for _ in 0..1000 {
            let a: F = FieldSampling::sample(&mut rng);
            let b = rng.next_u64() as u128 | ((rng.next_u64() as u128) << 64);
            let expected = a * F::from_canonical_u128_reduced(b);
            let reduced = F::solinas_reduce(&a.mul_wide_u128(b));
            assert_eq!(reduced, expected);
        }
    }

    #[test]
    fn solinas_reduce_small_inputs() {
        assert_eq!(F::solinas_reduce(&[]), F::zero());
        assert_eq!(F::solinas_reduce(&[42]), F::from_u64(42));
        let one_shifted = F::from_canonical_u128_reduced(1u128 << 64);
        assert_eq!(F::solinas_reduce(&[0, 1]), one_shifted);
    }

    #[test]
    fn solinas_reduce_4_limbs_max() {
        // 2^256 - 1 ≡ C² - 1 (mod P), since 2^128 ≡ C
        let c = F::from_canonical_u128_reduced(<F as PseudoMersenneField>::MODULUS_OFFSET);
        let expected = c * c - F::one();
        assert_eq!(F::solinas_reduce(&[u64::MAX; 4]), expected);
    }

    #[test]
    fn solinas_reduce_9_limbs() {
        // 1 + 2^512 = 1 + (2^128)^4 ≡ 1 + C^4
        let c = F::from_canonical_u128_reduced(<F as PseudoMersenneField>::MODULUS_OFFSET);
        let expected = F::one() + c * c * c * c;
        assert_eq!(F::solinas_reduce(&[1, 0, 0, 0, 0, 0, 0, 0, 1]), expected);
    }

    #[test]
    fn solinas_reduce_accumulated_products() {
        let mut rng = StdRng::seed_from_u64(0xfeed_face_0bad_c0de);
        let mut acc = [0u64; 5];
        let mut expected = F::zero();

        for _ in 0..200 {
            let a: F = FieldSampling::sample(&mut rng);
            let b = rng.next_u64();
            let wide = a.mul_wide_u64(b);

            let mut carry: u64 = 0;
            for j in 0..5 {
                let addend = if j < 3 { wide[j] } else { 0 };
                let sum = acc[j] as u128 + addend as u128 + carry as u128;
                acc[j] = sum as u64;
                carry = (sum >> 64) as u64;
            }
            assert_eq!(carry, 0);
            expected = expected + a * F::from_u64(b);
        }

        assert_eq!(F::solinas_reduce(&acc), expected);
    }

    #[test]
    fn solinas_reduce_cross_prime() {
        // Verify with Prime128M18M0 (C = 2^18 + 1, shift+add path)
        type G = Prime128M18M0;
        let c = G::from_canonical_u128_reduced(<G as PseudoMersenneField>::MODULUS_OFFSET);
        let expected = c * c - G::one();
        assert_eq!(G::solinas_reduce(&[u64::MAX; 4]), expected);

        // Verify with Prime128M54P0 (C = 2^54 - 1, shift-sub path)
        type H = Prime128M54P0;
        let c = H::from_canonical_u128_reduced(<H as PseudoMersenneField>::MODULUS_OFFSET);
        let expected = c * c - H::one();
        assert_eq!(H::solinas_reduce(&[u64::MAX; 4]), expected);
    }
}
