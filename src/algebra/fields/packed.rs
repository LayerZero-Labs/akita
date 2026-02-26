//! Packed field abstractions and Fp128 field backends.
//!
//! This module is intentionally field-scoped for now (no ring/protocol wiring yet).

use crate::algebra::fields::Fp128;
use crate::FieldCore;
use core::ops::{Add, Mul, Sub};

/// Array-like packed values over a scalar type.
pub trait PackedValue: 'static + Copy + Send + Sync {
    /// Scalar value type carried by each lane.
    type Value: 'static + Copy + Send + Sync;

    /// Number of scalar lanes.
    const WIDTH: usize;

    /// Build from a lane generator.
    fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value;

    /// Extract one lane.
    fn extract(&self, lane: usize) -> Self::Value;

    /// Pack a scalar slice into packed values.
    ///
    /// # Panics
    ///
    /// Panics if the length is not divisible by `WIDTH`.
    #[inline]
    fn pack_slice(buf: &[Self::Value]) -> Vec<Self> {
        assert!(
            buf.len() % Self::WIDTH == 0,
            "slice length {} must be divisible by WIDTH {}",
            buf.len(),
            Self::WIDTH
        );
        buf.chunks_exact(Self::WIDTH)
            .map(|chunk| Self::from_fn(|i| chunk[i]))
            .collect()
    }

    /// Packed prefix + scalar suffix split.
    #[inline]
    fn pack_slice_with_suffix(buf: &[Self::Value]) -> (Vec<Self>, &[Self::Value]) {
        let split = buf.len() - (buf.len() % Self::WIDTH);
        let (packed, suffix) = buf.split_at(split);
        (Self::pack_slice(packed), suffix)
    }

    /// Unpack packed values into a flat scalar vector.
    #[inline]
    fn unpack_slice(buf: &[Self]) -> Vec<Self::Value> {
        let mut out = Vec::with_capacity(buf.len() * Self::WIDTH);
        for packed in buf {
            for lane in 0..Self::WIDTH {
                out.push(packed.extract(lane));
            }
        }
        out
    }
}

/// Packed arithmetic over a scalar field.
pub trait PackedField:
    PackedValue<Value = Self::Scalar> + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    /// Scalar field type.
    type Scalar: FieldCore;

    /// Broadcast one scalar across all lanes.
    fn broadcast(value: Self::Scalar) -> Self;
}

/// Scalar fallback packed type with one lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct NoPacking<T>(pub [T; 1]);

impl<T> PackedValue for NoPacking<T>
where
    T: 'static + Copy + Send + Sync,
{
    type Value = T;
    const WIDTH: usize = 1;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        Self([f(0)])
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert_eq!(lane, 0);
        self.0[0]
    }
}

impl<T: FieldCore> Add for NoPacking<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0]])
    }
}

impl<T: FieldCore> Sub for NoPacking<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0]])
    }
}

impl<T: FieldCore> Mul for NoPacking<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self([self.0[0] * rhs.0[0]])
    }
}

impl<T: FieldCore + 'static> PackedField for NoPacking<T> {
    type Scalar = T;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self([value])
    }
}

/// AArch64 first packed `Fp128` backend.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod aarch64_neon {
    use super::{PackedField, PackedValue};
    use crate::algebra::fields::Fp128;
    use crate::FieldCore;
    use core::arch::aarch64::{
        uint64x2_t, vaddq_u64, vandq_u64, vbslq_u64, vcgtq_u64, vcltq_u64, vdupq_n_u64, veorq_u64,
        vorrq_u64, vsubq_u64,
    };
    use core::fmt;
    use core::mem::transmute;
    use core::ops::{Add, Mul, Sub};

    /// Number of packed `Fp128` lanes in this backend.
    pub const WIDTH: usize = 2;

    /// True SoA layout for two packed `Fp128` lanes.
    ///
    /// `lo = [lane0.lo, lane1.lo]`
    /// `hi = [lane0.hi, lane1.hi]`
    #[derive(Clone, Copy)]
    pub struct PackedFp128Neon<const P: u128> {
        lo: [u64; 2],
        hi: [u64; 2],
    }

    #[inline(always)]
    fn to_vec(x: [u64; 2]) -> uint64x2_t {
        // SAFETY: `uint64x2_t` and `[u64; 2]` have identical lane layout.
        unsafe { transmute::<[u64; 2], uint64x2_t>(x) }
    }

    #[inline(always)]
    fn from_vec(v: uint64x2_t) -> [u64; 2] {
        // SAFETY: `uint64x2_t` and `[u64; 2]` have identical lane layout.
        unsafe { transmute::<uint64x2_t, [u64; 2]>(v) }
    }

    #[inline(always)]
    fn mask_to_bit(mask: uint64x2_t) -> uint64x2_t {
        // SAFETY: NEON intrinsics are available under this cfg.
        unsafe { vandq_u64(mask, vdupq_n_u64(1)) }
    }

    #[inline(always)]
    const fn modulus_lo<const P: u128>() -> u64 {
        P as u64
    }

    #[inline(always)]
    const fn modulus_hi<const P: u128>() -> u64 {
        (P >> 64) as u64
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

    impl<const P: u128> PackedFp128Neon<P> {
        const C: u128 = {
            let c = 0u128.wrapping_sub(P);
            assert!(P != 0, "modulus must be nonzero");
            assert!(P & 1 == 1, "modulus must be odd");
            assert!(c < (1u128 << 64), "P must be 2^128 - c with c < 2^64");
            assert!(
                c * (c + 1) < P,
                "C(C+1) < P required for fused canonicalize"
            );
            c
        };
        const C_LO: u64 = Self::C as u64;
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

        #[inline(always)]
        fn mul_wide_u64(a: u64, b: u64) -> (u64, u64) {
            let prod = (a as u128) * (b as u128);
            (prod as u64, (prod >> 64) as u64)
        }

        #[inline(always)]
        fn mul_c_wide(x: u64) -> (u64, u64) {
            if Self::C_SHIFT_KIND == 1 {
                let v = ((x as u128) << Self::C_SHIFT) + x as u128;
                (v as u64, (v >> 64) as u64)
            } else if Self::C_SHIFT_KIND == -1 {
                let v = ((x as u128) << Self::C_SHIFT) - x as u128;
                (v as u64, (v >> 64) as u64)
            } else {
                Self::mul_wide_u64(Self::C_LO, x)
            }
        }

        #[inline(always)]
        fn fold2_canonicalize(t0: u64, t1: u64, t2: u64) -> (u64, u64) {
            let (ct2_lo, ct2_hi) = Self::mul_c_wide(t2);

            let (s0, carry0) = t0.overflowing_add(ct2_lo);
            let (s1a, carry1a) = t1.overflowing_add(ct2_hi);
            let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
            let overflow = carry1a | carry1b;

            let (r0, carry2) = s0.overflowing_add(Self::C_LO);
            let (r1, carry3) = s1.overflowing_add(carry2 as u64);

            if overflow | carry3 {
                (r0, r1)
            } else {
                (s0, s1)
            }
        }

        #[inline(always)]
        fn mul_raw_lane(a0: u64, a1: u64, b0: u64, b1: u64) -> (u64, u64) {
            let (p00_lo, p00_hi) = Self::mul_wide_u64(a0, b0);
            let (p01_lo, p01_hi) = Self::mul_wide_u64(a0, b1);
            let (p10_lo, p10_hi) = Self::mul_wide_u64(a1, b0);
            let (p11_lo, p11_hi) = Self::mul_wide_u64(a1, b1);

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
    }

    impl<const P: u128> Default for PackedFp128Neon<P> {
        #[inline]
        fn default() -> Self {
            Self::broadcast(Fp128::zero())
        }
    }

    impl<const P: u128> fmt::Debug for PackedFp128Neon<P> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_tuple("PackedFp128Neon")
                .field(&[self.extract(0), self.extract(1)])
                .finish()
        }
    }

    impl<const P: u128> PartialEq for PackedFp128Neon<P> {
        #[inline]
        fn eq(&self, other: &Self) -> bool {
            self.extract(0) == other.extract(0) && self.extract(1) == other.extract(1)
        }
    }

    impl<const P: u128> Eq for PackedFp128Neon<P> {}

    impl<const P: u128> PackedValue for PackedFp128Neon<P> {
        type Value = Fp128<P>;
        const WIDTH: usize = WIDTH;

        #[inline]
        fn from_fn<F>(mut f: F) -> Self
        where
            F: FnMut(usize) -> Self::Value,
        {
            let x0 = f(0);
            let x1 = f(1);
            Self {
                lo: [x0.0[0], x1.0[0]],
                hi: [x0.0[1], x1.0[1]],
            }
        }

        #[inline]
        fn extract(&self, lane: usize) -> Self::Value {
            debug_assert!(lane < WIDTH);
            Fp128([self.lo[lane], self.hi[lane]])
        }
    }

    impl<const P: u128> Add for PackedFp128Neon<P> {
        type Output = Self;
        #[inline]
        fn add(self, rhs: Self) -> Self {
            let lo_a = to_vec(self.lo);
            let hi_a = to_vec(self.hi);
            let lo_b = to_vec(rhs.lo);
            let hi_b = to_vec(rhs.hi);

            // SAFETY: NEON intrinsics are available under this cfg.
            let (out_lo, out_hi) = unsafe {
                let p_lo = vdupq_n_u64(modulus_lo::<P>());
                let p_hi = vdupq_n_u64(modulus_hi::<P>());

                // 128-bit sum with carry tracking.
                let sum_lo = vaddq_u64(lo_a, lo_b);
                let carry_lo = mask_to_bit(vcltq_u64(sum_lo, lo_a));

                let hi_tmp = vaddq_u64(hi_a, hi_b);
                let carry_hi1 = vcltq_u64(hi_tmp, hi_a);
                let sum_hi = vaddq_u64(hi_tmp, carry_lo);
                let carry_hi2 = vcltq_u64(sum_hi, hi_tmp);
                let carry_128 = vorrq_u64(carry_hi1, carry_hi2);

                // Reduced candidate: sum - P.
                let red_lo = vsubq_u64(sum_lo, p_lo);
                let borrow_lo = mask_to_bit(vcgtq_u64(p_lo, sum_lo));

                let red_hi_tmp = vsubq_u64(sum_hi, p_hi);
                let borrow_hi1 = vcgtq_u64(p_hi, sum_hi);
                let red_hi = vsubq_u64(red_hi_tmp, borrow_lo);
                let borrow_hi2 = vcltq_u64(red_hi_tmp, borrow_lo);
                let borrow = vorrq_u64(borrow_hi1, borrow_hi2);

                // Use reduced when overflowed or when sum >= P.
                let not_borrow = veorq_u64(borrow, vdupq_n_u64(u64::MAX));
                let use_reduced = vorrq_u64(carry_128, not_borrow);
                let out_lo = vbslq_u64(use_reduced, red_lo, sum_lo);
                let out_hi = vbslq_u64(use_reduced, red_hi, sum_hi);
                (out_lo, out_hi)
            };

            Self {
                lo: from_vec(out_lo),
                hi: from_vec(out_hi),
            }
        }
    }

    impl<const P: u128> Sub for PackedFp128Neon<P> {
        type Output = Self;
        #[inline]
        fn sub(self, rhs: Self) -> Self {
            let lo_a = to_vec(self.lo);
            let hi_a = to_vec(self.hi);
            let lo_b = to_vec(rhs.lo);
            let hi_b = to_vec(rhs.hi);

            // SAFETY: NEON intrinsics are available under this cfg.
            let (out_lo, out_hi) = unsafe {
                let p_lo = vdupq_n_u64(modulus_lo::<P>());
                let p_hi = vdupq_n_u64(modulus_hi::<P>());

                // 128-bit diff with borrow tracking.
                let diff_lo = vsubq_u64(lo_a, lo_b);
                let borrow_lo = mask_to_bit(vcltq_u64(lo_a, lo_b));

                let diff_hi_tmp = vsubq_u64(hi_a, hi_b);
                let borrow_hi1 = vcltq_u64(hi_a, hi_b);
                let diff_hi = vsubq_u64(diff_hi_tmp, borrow_lo);
                let borrow_hi2 = vcltq_u64(diff_hi_tmp, borrow_lo);
                let borrow_128 = vorrq_u64(borrow_hi1, borrow_hi2);

                // Correct by +P when diff underflowed.
                let corr_lo = vaddq_u64(diff_lo, p_lo);
                let carry_lo = mask_to_bit(vcltq_u64(corr_lo, diff_lo));

                let corr_hi_tmp = vaddq_u64(diff_hi, p_hi);
                let corr_hi = vaddq_u64(corr_hi_tmp, carry_lo);

                let out_lo = vbslq_u64(borrow_128, corr_lo, diff_lo);
                let out_hi = vbslq_u64(borrow_128, corr_hi, diff_hi);
                (out_lo, out_hi)
            };

            Self {
                lo: from_vec(out_lo),
                hi: from_vec(out_hi),
            }
        }
    }

    impl<const P: u128> Mul for PackedFp128Neon<P> {
        type Output = Self;
        #[inline]
        fn mul(self, rhs: Self) -> Self {
            let (o0_lo, o0_hi) = Self::mul_raw_lane(self.lo[0], self.hi[0], rhs.lo[0], rhs.hi[0]);
            let (o1_lo, o1_hi) = Self::mul_raw_lane(self.lo[1], self.hi[1], rhs.lo[1], rhs.hi[1]);

            Self {
                lo: [o0_lo, o1_lo],
                hi: [o0_hi, o1_hi],
            }
        }
    }

    impl<const P: u128> PackedField for PackedFp128Neon<P> {
        type Scalar = Fp128<P>;

        #[inline]
        fn broadcast(value: Self::Scalar) -> Self {
            Self::from_fn(|_| value)
        }
    }
}

/// Scalar field -> packed field association.
pub trait HasPacking: FieldCore {
    /// Packed representation for this scalar field.
    type Packing: PackedField<Scalar = Self>;
}

/// Selected packed backend for `Fp128`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub type Fp128Packing<const P: u128> = aarch64_neon::PackedFp128Neon<P>;

/// Scalar fallback packed backend for non-AArch64/NEON targets.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
pub type Fp128Packing<const P: u128> = NoPacking<Fp128<P>>;

impl<const P: u128> HasPacking for Fp128<P> {
    type Packing = Fp128Packing<P>;
}

#[cfg(test)]
mod tests {
    use super::{HasPacking, PackedField, PackedValue};
    use crate::algebra::fields::Prime128M13M4P0;
    use crate::CanonicalField;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    fn rand_u128<R: RngCore>(rng: &mut R) -> u128 {
        let lo = rng.next_u64() as u128;
        let hi = rng.next_u64() as u128;
        lo | (hi << 64)
    }

    #[test]
    fn packed_add_sub_mul_match_scalar() {
        type F = Prime128M13M4P0;
        type PF = <F as HasPacking>::Packing;

        let mut rng = StdRng::seed_from_u64(0x55aa_4422_1177_0033);
        let len = PF::WIDTH * 17 + 3;
        let lhs: Vec<F> = (0..len)
            .map(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
            .collect();
        let rhs: Vec<F> = (0..len)
            .map(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
            .collect();

        let (lhs_p, lhs_s) = PF::pack_slice_with_suffix(&lhs);
        let (rhs_p, rhs_s) = PF::pack_slice_with_suffix(&rhs);

        let add_p: Vec<PF> = lhs_p
            .iter()
            .zip(rhs_p.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let sub_p: Vec<PF> = lhs_p
            .iter()
            .zip(rhs_p.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let mul_p: Vec<PF> = lhs_p
            .iter()
            .zip(rhs_p.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        let mut add_out = PF::unpack_slice(&add_p);
        let mut sub_out = PF::unpack_slice(&sub_p);
        let mut mul_out = PF::unpack_slice(&mul_p);

        for (&a, &b) in lhs_s.iter().zip(rhs_s.iter()) {
            add_out.push(a + b);
            sub_out.push(a - b);
            mul_out.push(a * b);
        }

        for i in 0..len {
            assert_eq!(
                add_out[i],
                lhs[i] + rhs[i],
                "packed add mismatch at lane {i}"
            );
            assert_eq!(
                sub_out[i],
                lhs[i] - rhs[i],
                "packed sub mismatch at lane {i}"
            );
            assert_eq!(
                mul_out[i],
                lhs[i] * rhs[i],
                "packed mul mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn broadcast_and_extract_roundtrip() {
        type F = Prime128M13M4P0;
        type PF = <F as HasPacking>::Packing;

        let x = F::from_u64(42);
        let p = PF::broadcast(x);
        for lane in 0..PF::WIDTH {
            assert_eq!(p.extract(lane), x);
        }
    }
}
