use super::*;

/// Number of packed `Fp64` lanes.
pub(crate) const FP64_WIDTH: usize = 2;

/// NEON packed `Fp64` backend: 2 lanes in `uint64x2_t`.
#[derive(Clone, Copy)]
pub struct PackedFp64Neon<const P: u64> {
    vals: [u64; 2],
}

impl<const P: u64> PackedFp64Neon<P> {
    const BITS: u32 = 64 - P.leading_zeros();

    const C_LO: u64 = {
        let c = if Self::BITS == 64 {
            0u64.wrapping_sub(P)
        } else {
            (1u64 << Self::BITS) - P
        };
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        c
    };

    const MASK64: u64 = if Self::BITS < 64 {
        (1u64 << Self::BITS) - 1
    } else {
        u64::MAX
    };

    const MASK_U128: u128 = if Self::BITS == 64 {
        u64::MAX as u128
    } else {
        (1u128 << Self::BITS) - 1
    };

    const FOLD_IN_U64: bool =
        Self::BITS < 64 && (Self::C_LO as u128) < (1u128 << (64 - Self::BITS));

    /// Whether a coefficient containing three raw products is reduced to a
    /// canonical value by the same two-fold reducer. Sub-word storage ensures
    /// the three-product sum itself fits in `u128`; this bound ensures one
    /// final conditional subtraction is sufficient after the folds.
    const FUSE_EXT2_TWO_NR: bool =
        Self::BITS < 64 && 3 * (Self::C_LO as u128) * (Self::C_LO as u128 + 1) < P as u128;

    #[inline(always)]
    fn mul_c_narrow(x: u64) -> u64 {
        Self::C_LO.wrapping_mul(x)
    }

    #[inline(always)]
    fn reduce_product(lo: u64, hi: u64) -> u64 {
        if Self::FOLD_IN_U64 {
            let high = (lo >> Self::BITS) | (hi << (64 - Self::BITS));
            let f1 = (lo & Self::MASK64).wrapping_add(Self::mul_c_narrow(high));
            let f2 = (f1 & Self::MASK64).wrapping_add(Self::mul_c_narrow(f1 >> Self::BITS));
            let reduced = f2.wrapping_sub(P);
            let borrow = reduced >> 63;
            reduced.wrapping_add(borrow.wrapping_neg() & P)
        } else if Self::BITS < 64 {
            let high = (lo >> Self::BITS) | (hi << (64 - Self::BITS));
            let high_overflow = hi >> Self::BITS;
            let c_high = (Self::C_LO as u128) * (high as u128)
                + (((Self::C_LO * high_overflow) as u128) << 64);
            let (fold1_lo, carry) = (lo & Self::MASK64).overflowing_add(c_high as u64);
            let fold1_hi = ((c_high >> 64) as u64) + u64::from(carry);
            let fold1_high = (fold1_lo >> Self::BITS) | (fold1_hi << (64 - Self::BITS));
            let fold2 = (fold1_lo & Self::MASK64) + Self::mul_c_narrow(fold1_high);
            let reduced = fold2.wrapping_sub(P);
            let borrow = u64::from(fold2 < P);
            reduced.wrapping_add(borrow.wrapping_neg() & P)
        } else {
            let x = lo as u128 | ((hi as u128) << 64);
            let f1 =
                (x & Self::MASK_U128) + (Self::C_LO as u128) * ((x >> Self::BITS) as u64 as u128);
            let f2 =
                (f1 & Self::MASK_U128) + (Self::C_LO as u128) * ((f1 >> Self::BITS) as u64 as u128);
            let reduced = f2.wrapping_sub(P as u128);
            let borrow = reduced >> 127;
            reduced.wrapping_add(borrow.wrapping_neg() & (P as u128)) as u64
        }
    }
}

impl<const P: u64> Default for PackedFp64Neon<P> {
    #[inline]
    fn default() -> Self {
        Self { vals: [0; 2] }
    }
}

impl<const P: u64> fmt::Debug for PackedFp64Neon<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp64Neon").field(&self.vals).finish()
    }
}

impl<const P: u64> PartialEq for PackedFp64Neon<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.vals == other.vals
    }
}

impl<const P: u64> Eq for PackedFp64Neon<P> {}

impl<const P: u64> Add for PackedFp64Neon<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let a = to_vec(self.vals);
        let b = to_vec(rhs.vals);
        let result = unsafe {
            let p = vdupq_n_u64(P);
            if Self::BITS == 64 {
                let s = vaddq_u64(a, b);
                let overflow = vcltq_u64(s, a);
                let folded = vaddq_u64(s, vandq_u64(overflow, vdupq_n_u64(Self::C_LO)));
                let reduced = vsubq_u64(folded, p);
                let borrow = vcltq_u64(folded, p);
                vbslq_u64(borrow, folded, reduced)
            } else {
                // For every sub-word modulus, 2P < 2^64.
                let s = vaddq_u64(a, b);
                let reduced = vsubq_u64(s, p);
                let borrow = vcltq_s64(vreinterpretq_s64_u64(reduced), vdupq_n_s64(0));
                vbslq_u64(borrow, s, reduced)
            }
        };
        Self {
            vals: from_vec(result),
        }
    }
}

impl<const P: u64> Sub for PackedFp64Neon<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let a = to_vec(self.vals);
        let b = to_vec(rhs.vals);
        let result = unsafe {
            let d = vsubq_u64(a, b);
            let p = vdupq_n_u64(P);
            if Self::BITS < 64 {
                let underflow = vcltq_s64(vreinterpretq_s64_u64(d), vdupq_n_s64(0));
                vbslq_u64(underflow, vaddq_u64(d, p), d)
            } else {
                let underflow = vcltq_u64(a, b);
                vaddq_u64(d, vandq_u64(underflow, p))
            }
        };
        Self {
            vals: from_vec(result),
        }
    }
}

impl<const P: u64> Mul for PackedFp64Neon<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let x0 = (self.vals[0] as u128) * (rhs.vals[0] as u128);
        let x1 = (self.vals[1] as u128) * (rhs.vals[1] as u128);
        let r0 = Self::reduce_product(x0 as u64, (x0 >> 64) as u64);
        let r1 = Self::reduce_product(x1 as u64, (x1 >> 64) as u64);
        Self { vals: [r0, r1] }
    }
}

impl<const P: u64> PackedValue for PackedFp64Neon<P> {
    type Value = Fp64<P>;
    const WIDTH: usize = FP64_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        Self {
            vals: [f(0).0, f(1).0],
        }
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < FP64_WIDTH);
        Fp64(self.vals[lane])
    }
}

impl<const P: u64> AddAssign for PackedFp64Neon<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64> SubAssign for PackedFp64Neon<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64> MulAssign for PackedFp64Neon<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u64> PackedField for PackedFp64Neon<P> {
    type Scalar = Fp64<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self { vals: [value.0; 2] }
    }

    #[inline(always)]
    fn fp_ext2_mul<C>(a0: Self, a1: Self, b0: Self, b1: Self) -> (Self, Self)
    where
        C: FpExt2Config<Self::Scalar>,
    {
        if C::IS_TWO && Self::FUSE_EXT2_TWO_NR {
            let mut c0 = [0; 2];
            let mut c1 = [0; 2];
            for lane in 0..2 {
                let p00 = (a0.vals[lane] as u128) * (b0.vals[lane] as u128);
                let p11 = (a1.vals[lane] as u128) * (b1.vals[lane] as u128);
                let p01 = (a0.vals[lane] as u128) * (b1.vals[lane] as u128);
                let p10 = (a1.vals[lane] as u128) * (b0.vals[lane] as u128);
                let z0 = p00 + p11 + p11;
                let z1 = p01 + p10;
                c0[lane] = Self::reduce_product(z0 as u64, (z0 >> 64) as u64);
                c1[lane] = Self::reduce_product(z1 as u64, (z1 >> 64) as u64);
            }
            return (Self { vals: c0 }, Self { vals: c1 });
        }

        let v0 = a0 * b0;
        let v1 = a1 * b1;
        let cross = (a0 + a1) * (b0 + b1);
        (
            v0 + C::mul_non_residue(v1, Self::broadcast),
            cross - v0 - v1,
        )
    }
}
