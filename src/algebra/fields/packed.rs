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
        lo: uint64x2_t,
        hi: uint64x2_t,
    }

    #[inline(always)]
    fn vec_from_u64x2(x0: u64, x1: u64) -> uint64x2_t {
        // SAFETY: `uint64x2_t` and `[u64; 2]` have identical lane layout.
        unsafe { transmute::<[u64; 2], uint64x2_t>([x0, x1]) }
    }

    #[inline(always)]
    fn u64x2_from_vec(v: uint64x2_t) -> [u64; 2] {
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
                lo: vec_from_u64x2(x0.0[0], x1.0[0]),
                hi: vec_from_u64x2(x0.0[1], x1.0[1]),
            }
        }

        #[inline]
        fn extract(&self, lane: usize) -> Self::Value {
            debug_assert!(lane < WIDTH);
            let lo = u64x2_from_vec(self.lo);
            let hi = u64x2_from_vec(self.hi);
            Fp128([lo[lane], hi[lane]])
        }
    }

    impl<const P: u128> Add for PackedFp128Neon<P> {
        type Output = Self;
        #[inline]
        fn add(self, rhs: Self) -> Self {
            // SAFETY: NEON intrinsics are available under this cfg.
            let (out_lo, out_hi) = unsafe {
                let p_lo = vdupq_n_u64(modulus_lo::<P>());
                let p_hi = vdupq_n_u64(modulus_hi::<P>());

                // 128-bit sum with carry tracking.
                let sum_lo = vaddq_u64(self.lo, rhs.lo);
                let carry_lo = mask_to_bit(vcltq_u64(sum_lo, self.lo));

                let hi_tmp = vaddq_u64(self.hi, rhs.hi);
                let carry_hi1 = vcltq_u64(hi_tmp, self.hi);
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
                lo: out_lo,
                hi: out_hi,
            }
        }
    }

    impl<const P: u128> Sub for PackedFp128Neon<P> {
        type Output = Self;
        #[inline]
        fn sub(self, rhs: Self) -> Self {
            // SAFETY: NEON intrinsics are available under this cfg.
            let (out_lo, out_hi) = unsafe {
                let p_lo = vdupq_n_u64(modulus_lo::<P>());
                let p_hi = vdupq_n_u64(modulus_hi::<P>());

                // 128-bit diff with borrow tracking.
                let diff_lo = vsubq_u64(self.lo, rhs.lo);
                let borrow_lo = mask_to_bit(vcltq_u64(self.lo, rhs.lo));

                let diff_hi_tmp = vsubq_u64(self.hi, rhs.hi);
                let borrow_hi1 = vcltq_u64(self.hi, rhs.hi);
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
                lo: out_lo,
                hi: out_hi,
            }
        }
    }

    impl<const P: u128> Mul for PackedFp128Neon<P> {
        type Output = Self;
        #[inline]
        fn mul(self, rhs: Self) -> Self {
            // Keep SoA layout through mul/reduce to avoid extract/repack overhead.
            let a_lo = u64x2_from_vec(self.lo);
            let a_hi = u64x2_from_vec(self.hi);
            let b_lo = u64x2_from_vec(rhs.lo);
            let b_hi = u64x2_from_vec(rhs.hi);

            let o0 = Fp128::<P>([a_lo[0], a_hi[0]]) * Fp128::<P>([b_lo[0], b_hi[0]]);
            let o1 = Fp128::<P>([a_lo[1], a_hi[1]]) * Fp128::<P>([b_lo[1], b_hi[1]]);

            Self {
                lo: vec_from_u64x2(o0.0[0], o1.0[0]),
                hi: vec_from_u64x2(o0.0[1], o1.0[1]),
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
