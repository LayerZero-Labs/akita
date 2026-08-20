use super::*;

/// Number of packed `Fp64` lanes.
pub(crate) const FP64_WIDTH: usize = 2;

/// NEON packed `Fp64` backend: 2 lanes in `uint64x2_t`.
#[derive(Clone, Copy)]
pub struct PackedFp64Neon<const P: u64> {
    vals: [u64; 2],
}

impl<const P: u64> Default for PackedFp64Neon<P> {
    #[inline]
    fn default() -> Self {
        Self { vals: [0; 2] }
    }
}

impl<const P: u64> PackedFp64Neon<P> {
    /// Multiply and reduce one lane for a 63-bit pseudo-Mersenne modulus.
    ///
    /// Keeping both Solinas folds in general-purpose registers avoids the
    /// scalar-to-NEON packing generated for the portable `u128` reducer.
    #[inline(always)]
    fn mul_reduce_63(lhs: u64, rhs: u64) -> u64 {
        debug_assert_eq!(Fp64::<P>::BITS, 63);
        let result: u64;
        let c = Fp64::<P>::C;
        let reduction_bias = (1u64 << 63) + c;

        // For BITS = 63, P = 2^63 - C. The field invariant C(C + 1) < P
        // guarantees that two folds leave a value below 2P:
        //
        //   x = r + q*2^63  ->  f1 = r + q*C
        //   f1 = s + t*2^63 ->  f2 = s + t*C < 2P.
        //
        // `umulh` plus `mul` preserves the first fold's carry, which the
        // narrow u64 reducer cannot do for a 63-bit modulus with C > 1.
        // Adding 2^64 - P = 2^63 + C sets carry exactly when f2 >= P, so the
        // final `csel` performs one canonical subtraction without a branch.
        // SAFETY: the assembly has no memory or stack effects, all temporaries
        // are declared outputs, and the function is called only for BITS = 63.
        unsafe {
            core::arch::asm!(
                "umulh {high}, {lhs}, {rhs}",
                "mul {result}, {lhs}, {rhs}",
                "extr {quotient}, {high}, {result}, #63",
                "and {result}, {result}, #0x7fffffffffffffff",
                "umulh {product_hi}, {quotient}, {c}",
                "mul {product_lo}, {quotient}, {c}",
                "adds {result}, {result}, {product_lo}",
                "adc {high}, {product_hi}, xzr",
                "extr {quotient}, {high}, {result}, #63",
                "and {result}, {result}, #0x7fffffffffffffff",
                "madd {result}, {quotient}, {c}, {result}",
                "adds {high}, {result}, {reduction_bias}",
                "csel {result}, {high}, {result}, hs",
                lhs = in(reg) lhs,
                rhs = in(reg) rhs,
                c = in(reg) c,
                reduction_bias = in(reg) reduction_bias,
                result = out(reg) result,
                high = out(reg) _,
                quotient = out(reg) _,
                product_lo = out(reg) _,
                product_hi = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
        result
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
            if Fp64::<P>::BITS == 64 {
                let s = vaddq_u64(a, b);
                let overflow = vcltq_u64(s, a);
                let folded = vaddq_u64(s, vandq_u64(overflow, vdupq_n_u64(Fp64::<P>::C)));
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
            let neg_p = vdupq_n_u64(P.wrapping_neg());
            let underflow = vcltq_u64(a, b);
            // `d - (-P)` is `d + P` modulo 2^64. Using -P keeps the
            // full-word correction equal to the small offset C.
            vsubq_u64(d, vandq_u64(underflow, neg_p))
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
        if Fp64::<P>::BITS == 63 {
            return Self {
                vals: [
                    Self::mul_reduce_63(self.vals[0], rhs.vals[0]),
                    Self::mul_reduce_63(self.vals[1], rhs.vals[1]),
                ],
            };
        }
        let x0 = (self.vals[0] as u128) * (rhs.vals[0] as u128);
        let x1 = (self.vals[1] as u128) * (rhs.vals[1] as u128);
        let r0 = Fp64::<P>::reduce_product_wide(x0 as u64, (x0 >> 64) as u64);
        let r1 = Fp64::<P>::reduce_product_wide(x1 as u64, (x1 >> 64) as u64);
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
        if C::NON_RESIDUE_KIND == FpExt2NonResidueKind::Two && Fp64::<P>::EXT2_TWO_FUSION_SAFE {
            let mut c0 = [0; 2];
            let mut c1 = [0; 2];
            for lane in 0..2 {
                let p00 = (a0.vals[lane] as u128) * (b0.vals[lane] as u128);
                let p11 = (a1.vals[lane] as u128) * (b1.vals[lane] as u128);
                let p01 = (a0.vals[lane] as u128) * (b1.vals[lane] as u128);
                let p10 = (a1.vals[lane] as u128) * (b0.vals[lane] as u128);
                let z0 = p00 + p11 + p11;
                let z1 = p01 + p10;
                c0[lane] = Fp64::<P>::reduce_three_product_sum(z0 as u64, (z0 >> 64) as u64);
                c1[lane] = Fp64::<P>::reduce_three_product_sum(z1 as u64, (z1 >> 64) as u64);
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
