//! Packed field abstractions and architecture-specific SIMD backends.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512dq"))
))]
pub(crate) mod avx2;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq"
))]
pub(crate) mod avx512;
pub(crate) mod ext;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) mod neon;

use crate::fields::ext::{
    power_basis_fp_ext4_mul_coeffs, ring_subfield_fp_ext8_mul_schedule,
    ring_subfield_fp_ext8_square_schedule, FpExt2Config, PowerBasisFpExt4Config,
    TowerBasisFpExt4Config,
};
use crate::fields::{Fp128, Fp32, Fp64};
use crate::{FieldCore, Invertible};
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use num_traits::{One, Zero};

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

    /// Square one packed value.
    #[inline(always)]
    fn square(self) -> Self {
        self * self
    }

    /// Invert one packed value lane-wise.
    #[inline]
    fn inverse(self) -> Option<Self>
    where
        Self::Scalar: Invertible,
    {
        let mut inverses = Vec::with_capacity(Self::WIDTH);
        for lane in 0..Self::WIDTH {
            inverses.push(self.extract(lane).inverse()?);
        }
        Some(Self::from_fn(|i| inverses[i]))
    }

    /// Backend hook for multiplying two packed `FpExt2` values in coefficient form.
    #[inline(always)]
    fn fp_ext2_mul<C>(a0: Self, a1: Self, b0: Self, b1: Self) -> (Self, Self)
    where
        C: FpExt2Config<Self::Scalar>,
    {
        let v0 = a0 * b0;
        let v1 = a1 * b1;
        let cross = (a0 + a1) * (b0 + b1);
        (
            v0 + C::mul_non_residue(v1, Self::broadcast),
            cross - v0 - v1,
        )
    }

    /// Backend hook for multiplying packed power-basis quartics.
    #[inline(always)]
    fn power_basis_fp_ext4_mul<C>(a: [Self; 4], b: [Self; 4]) -> [Self; 4]
    where
        C: PowerBasisFpExt4Config<Self::Scalar>,
    {
        power_basis_fp_ext4_mul_coeffs::<Self::Scalar, C, Self, _>(a, b, Self::broadcast)
    }

    /// Backend hook for multiplying packed tower-basis quartics.
    #[inline(always)]
    fn tower_basis_fp_ext4_mul<C2, C4>(a: [Self; 4], b: [Self; 4]) -> [Self; 4]
    where
        C2: FpExt2Config<Self::Scalar>,
        C4: TowerBasisFpExt4Config<Self::Scalar, C2>,
    {
        let [a0, a1, a2, a3] = a;
        let [b0, b1, b2, b3] = b;
        let (v0_0, v0_1) = Self::fp_ext2_mul::<C2>(a0, a2, b0, b2);
        let (v1_0, v1_1) = Self::fp_ext2_mul::<C2>(a1, a3, b1, b3);
        let nr = C4::non_residue();
        let (nr_v1_0, nr_v1_1) = if nr.coeffs[0].is_zero() && nr.coeffs[1] == Self::Scalar::one() {
            (C2::mul_non_residue(v1_1, Self::broadcast), v1_0)
        } else {
            Self::fp_ext2_mul::<C2>(
                Self::broadcast(nr.coeffs[0]),
                Self::broadcast(nr.coeffs[1]),
                v1_0,
                v1_1,
            )
        };
        let (cross_0, cross_1) = Self::fp_ext2_mul::<C2>(a0 + a1, a2 + a3, b0 + b1, b2 + b3);
        [
            v0_0 + nr_v1_0,
            cross_0 - v0_0 - v1_0,
            v0_1 + nr_v1_1,
            cross_1 - v0_1 - v1_1,
        ]
    }

    /// Backend hook for multiplying packed ring-subfield quartics.
    #[inline(always)]
    fn ring_subfield_fp_ext4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        let [a0, a1, a2, a3] = a;
        let [b0, b1, b2, b3] = b;
        let tail0 = a1 * b1 + a2 * b2 + a3 * b3;
        [
            a0 * b0 + tail0 + tail0,
            a0 * b1 + a1 * b0 + a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2,
            a0 * b2 + a2 * b0 + a1 * b1 + a1 * b3 + a3 * b1 - a3 * b3,
            a0 * b3 + a3 * b0 + a1 * b2 + a2 * b1 - a2 * b3 - a3 * b2,
        ]
    }

    /// Backend hook for squaring packed ring-subfield quartics.
    #[inline(always)]
    fn ring_subfield_fp_ext4_square(a: [Self; 4]) -> [Self; 4] {
        let [a0, a1, a2, a3] = a;
        let x0 = a0;
        let x1 = a2;
        let y0 = a1 - a3;
        let y1 = a3;

        let x0x1 = x0 * x1;
        let y0y1 = y0 * y1;
        let x1_square = x1 * x1;
        let y1_square = y1 * y1;
        let aa = (x0 * x0 + x1_square + x1_square, x0x1 + x0x1);
        let bb = (y0 * y0 + y1_square + y1_square, y0y1 + y0y1);

        let v0 = x0 * y0;
        let v1 = x1 * y1;
        let ab = (v0 + v1 + v1, (x0 + x1) * (y0 + y1) - v0 - v1);
        let constant = (bb.0 + bb.0 + bb.1 + bb.1, bb.0 + bb.1 + bb.1);
        let coeff_e1 = (ab.0 + ab.0, ab.1 + ab.1);

        [
            aa.0 + constant.0,
            coeff_e1.0 + coeff_e1.1,
            aa.1 + constant.1,
            coeff_e1.1,
        ]
    }

    /// Backend hook for inverting packed ring-subfield quartics.
    #[inline(always)]
    fn ring_subfield_fp_ext4_inverse(a: [Self; 4]) -> Option<[Self; 4]>
    where
        Self::Scalar: Invertible,
    {
        let zero = Self::broadcast(Self::Scalar::zero());
        let [a0, a1, a2, a3] = a;
        let x0 = a0;
        let x1 = a2;
        let y0 = a1 - a3;
        let y1 = a3;

        let x0x1 = x0 * x1;
        let y0y1 = y0 * y1;
        let x1_square = x1 * x1;
        let y1_square = y1 * y1;
        let aa = (x0 * x0 + x1_square + x1_square, x0x1 + x0x1);
        let bb = (y0 * y0 + y1_square + y1_square, y0y1 + y0y1);
        let nr_bb = (bb.0 + bb.0 + bb.1 + bb.1, bb.0 + bb.1 + bb.1);
        let norm = (aa.0 - nr_bb.0, aa.1 - nr_bb.1);
        let inv_norm_base = (norm.0 * norm.0 - (norm.1 * norm.1 + norm.1 * norm.1)).inverse()?;
        let inv_norm = (norm.0 * inv_norm_base, (zero - norm.1) * inv_norm_base);

        let v0 = x0 * inv_norm.0;
        let v1 = x1 * inv_norm.1;
        let constant = (
            v0 + v1 + v1,
            (x0 + x1) * (inv_norm.0 + inv_norm.1) - v0 - v1,
        );
        let neg_y0 = zero - y0;
        let neg_y1 = zero - y1;
        let w0 = neg_y0 * inv_norm.0;
        let w1 = neg_y1 * inv_norm.1;
        let e1_coeff = (
            w0 + w1 + w1,
            (neg_y0 + neg_y1) * (inv_norm.0 + inv_norm.1) - w0 - w1,
        );

        Some([constant.0, e1_coeff.0 + e1_coeff.1, constant.1, e1_coeff.1])
    }

    /// Backend hook for multiplying packed ring-subfield degree-8 elements.
    #[inline(always)]
    fn ring_subfield_fp_ext8_mul(a: [Self; 8], b: [Self; 8]) -> [Self; 8] {
        ring_subfield_fp_ext8_mul_schedule(
            a,
            b,
            Self::broadcast(Self::Scalar::zero()),
            |x, y| x + y,
            |x, y| x - y,
            |x, y| x * y,
        )
    }

    /// Backend hook for squaring packed ring-subfield degree-8 elements.
    #[inline(always)]
    fn ring_subfield_fp_ext8_square(a: [Self; 8]) -> [Self; 8] {
        ring_subfield_fp_ext8_square_schedule(
            a,
            Self::broadcast(Self::Scalar::zero()),
            |x, y| x + y,
            |x, y| x - y,
            |x, y| x * y,
        )
    }
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

impl<T: FieldCore> AddAssign for NoPacking<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: FieldCore> SubAssign for NoPacking<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: FieldCore> MulAssign for NoPacking<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: FieldCore + 'static> PackedField for NoPacking<T> {
    type Scalar = T;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self([value])
    }
}

/// Scalar field -> packed field association.
pub trait HasPacking: FieldCore {
    /// Packed representation for this scalar field.
    type Packing: PackedField<Scalar = Self>;
}

/// Selected packed backend for `Fp128`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub type Fp128Packing<const P: u128> = neon::PackedFp128Neon<P>;

/// Selected packed backend for `Fp128`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq"
))]
pub type Fp128Packing<const P: u128> = avx512::PackedFp128Avx512<P>;

/// Selected packed backend for `Fp128`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512dq"))
))]
pub type Fp128Packing<const P: u128> = avx2::PackedFp128Avx2<P>;

/// Selected packed backend for `Fp128`.
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2")
)))]
pub type Fp128Packing<const P: u128> = NoPacking<Fp128<P>>;

impl<const P: u128> HasPacking for Fp128<P> {
    type Packing = Fp128Packing<P>;
}

/// Selected packed backend for `Fp32`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub type Fp32Packing<const P: u32> = neon::PackedFp32Neon<P>;

/// Selected packed backend for `Fp32`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq"
))]
pub type Fp32Packing<const P: u32> = avx512::PackedFp32Avx512<P>;

/// Selected packed backend for `Fp32`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512dq"))
))]
pub type Fp32Packing<const P: u32> = avx2::PackedFp32Avx2<P>;

/// Selected packed backend for `Fp32`.
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2")
)))]
pub type Fp32Packing<const P: u32> = NoPacking<Fp32<P>>;

impl<const P: u32> HasPacking for Fp32<P> {
    type Packing = Fp32Packing<P>;
}

/// Selected packed backend for `Fp64`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub type Fp64Packing<const P: u64> = neon::PackedFp64Neon<P>;

/// Selected packed backend for `Fp64`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq"
))]
pub type Fp64Packing<const P: u64> = avx512::PackedFp64Avx512<P>;

/// Selected packed backend for `Fp64`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512dq"))
))]
pub type Fp64Packing<const P: u64> = avx2::PackedFp64Avx2<P>;

/// Selected packed backend for `Fp64`.
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2")
)))]
pub type Fp64Packing<const P: u64> = NoPacking<Fp64<P>>;

impl<const P: u64> HasPacking for Fp64<P> {
    type Packing = Fp64Packing<P>;
}

#[cfg(test)]
mod tests {
    use super::{HasPacking, PackedField, PackedValue};
    use crate::fields::{
        Fp32, Prime128Offset275, Prime24Offset3, Prime31Offset19, Prime32Offset99,
        Prime40Offset195, Prime64Offset59,
    };
    use crate::{CanonicalField, FieldCore, RandomSampling};
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    fn rand_u128<R: RngCore>(rng: &mut R) -> u128 {
        let lo = rng.next_u64() as u128;
        let hi = rng.next_u64() as u128;
        lo | (hi << 64)
    }

    fn check_packed_add_sub_mul<F, PF>(seed: u64)
    where
        F: FieldCore + RandomSampling + PartialEq + std::fmt::Debug,
        PF: PackedField<Scalar = F> + PackedValue<Value = F>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let len = PF::WIDTH * 17 + 3;
        let lhs: Vec<F> = (0..len).map(|_| RandomSampling::random(&mut rng)).collect();
        let rhs: Vec<F> = (0..len).map(|_| RandomSampling::random(&mut rng)).collect();

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

    fn check_broadcast_roundtrip<F, PF>(val: F)
    where
        F: FieldCore + PartialEq + std::fmt::Debug,
        PF: PackedField<Scalar = F> + PackedValue<Value = F>,
    {
        let p = PF::broadcast(val);
        for lane in 0..PF::WIDTH {
            assert_eq!(p.extract(lane), val);
        }
    }

    fn check_packed_fp32_edge_lanes<const P: u32, PF>()
    where
        PF: PackedField<Scalar = Fp32<P>> + PackedValue<Value = Fp32<P>>,
    {
        let p_minus_one = Fp32::<P>::from_canonical_u32(P - 1);
        let p_minus_two = Fp32::<P>::from_canonical_u32(P - 2);
        let values = [
            Fp32::<P>::zero(),
            Fp32::<P>::one(),
            p_minus_two,
            p_minus_one,
        ];
        let a = PF::from_fn(|i| values[i % values.len()]);
        let b = PF::from_fn(|i| values[(i + 1) % values.len()]);

        let add = a + b;
        let sub = a - b;
        let mul = a * b;

        for lane in 0..PF::WIDTH {
            let lhs = values[lane % values.len()];
            let rhs = values[(lane + 1) % values.len()];
            assert_eq!(add.extract(lane), lhs + rhs, "packed add edge lane {lane}");
            assert_eq!(sub.extract(lane), lhs - rhs, "packed sub edge lane {lane}");
            assert_eq!(mul.extract(lane), lhs * rhs, "packed mul edge lane {lane}");
        }
    }

    #[test]
    fn packed_fp128_add_sub_mul_match_scalar() {
        type F = Prime128Offset275;
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
    fn fp128_broadcast_and_extract_roundtrip() {
        type F = Prime128Offset275;
        type PF = <F as HasPacking>::Packing;
        check_broadcast_roundtrip::<F, PF>(F::from_u64(42));
    }

    #[test]
    fn packed_fp32_24b_add_sub_mul() {
        type F = Prime24Offset3;
        type PF = <F as HasPacking>::Packing;
        check_packed_add_sub_mul::<F, PF>(0xaa24_bb24_cc24_dd24);
    }

    #[test]
    fn packed_fp32_31b_add_sub_mul() {
        type F = Prime31Offset19;
        type PF = <F as HasPacking>::Packing;
        check_packed_add_sub_mul::<F, PF>(0xaa31_bb31_cc31_dd31);
    }

    #[test]
    fn packed_fp32_31b_edge_lanes() {
        type F = Prime31Offset19;
        type PF = <F as HasPacking>::Packing;
        check_packed_fp32_edge_lanes::<
            { crate::fields::prime::pseudo_mersenne::PRIME31_OFFSET19_MODULUS },
            PF,
        >();
    }

    #[test]
    fn packed_mersenne31_edge_lanes() {
        type F = Fp32<{ (1u32 << 31) - 1 }>;
        type PF = <F as HasPacking>::Packing;
        check_packed_fp32_edge_lanes::<{ (1u32 << 31) - 1 }, PF>();
    }

    /// Stress the 31-bit pseudo-Mersenne (`C > 1`) packed multiply against the
    /// scalar reference across boundary values and a large random sweep. This
    /// confirms (does not justify) the exact correctness proof on
    /// `mul_pmersenne31_vec`: the tightest cases are `z = (P-1)^2` and inputs
    /// that drive the second fold's `t'` toward `2P`.
    #[test]
    fn packed_fp32_31b_mul_matches_scalar_stress() {
        type F = Prime31Offset19;
        type PF = <F as HasPacking>::Packing;
        const P: u32 = crate::fields::prime::pseudo_mersenne::PRIME31_OFFSET19_MODULUS;

        let boundary = [
            0u32,
            1,
            2,
            3,
            19,
            1 << 15,
            1 << 30,
            (1 << 30) + 1,
            (P - 1) / 2,
            P - 3,
            P - 2,
            P - 1,
        ];

        let mut inputs: Vec<F> = boundary.iter().map(|&v| F::from_canonical_u32(v)).collect();
        let mut rng = StdRng::seed_from_u64(0x31be_19ca_fe00_1357);
        for _ in 0..(1 << 16) {
            inputs.push(F::from_canonical_u32(rng.next_u32() % P));
        }

        let lhs: Vec<F> = inputs.clone();
        let rhs: Vec<F> = {
            let mut r = inputs.clone();
            r.rotate_left(1);
            r
        };

        let (lhs_p, lhs_s) = PF::pack_slice_with_suffix(&lhs);
        let (rhs_p, rhs_s) = PF::pack_slice_with_suffix(&rhs);
        let mul_p: Vec<PF> = lhs_p
            .iter()
            .zip(rhs_p.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        let mut mul_out = PF::unpack_slice(&mul_p);
        for (&a, &b) in lhs_s.iter().zip(rhs_s.iter()) {
            mul_out.push(a * b);
        }
        for i in 0..lhs.len() {
            assert_eq!(mul_out[i], lhs[i] * rhs[i], "packed mul mismatch at {i}");
        }

        // Full boundary x boundary cross product (every tight combination).
        for &x in &boundary {
            for &y in &boundary {
                let a = PF::broadcast(F::from_canonical_u32(x));
                let b = PF::broadcast(F::from_canonical_u32(y));
                let got = (a * b).extract(0);
                let want = F::from_canonical_u32(x) * F::from_canonical_u32(y);
                assert_eq!(got, want, "boundary mul {x}*{y}");
            }
        }
    }

    #[test]
    fn packed_fp32_32b_add_sub_mul() {
        type F = Prime32Offset99;
        type PF = <F as HasPacking>::Packing;
        check_packed_add_sub_mul::<F, PF>(0xaa32_bb32_cc32_dd32);
    }

    /// Regression guard for the 32-bit (`BITS == 32`) packed base multiply.
    ///
    /// For these primes the two-fold Solinas residue can land in `[2^32, 2*P)`
    /// (up to `2^32 + C^2`). The packed `Mul` recombine must subtract `P` on the
    /// full 64-bit lanes before packing; a 32-bit recombine drops bit 32 and
    /// returns a result that is `C` too small. The probability of hitting this
    /// window with uniform random inputs is `~C/2^32 ≈ 2e-6`, so the random
    /// parity sweep misses it; these vectors hit it deterministically. They were
    /// found by exhaustively comparing the truncating recombine to the true
    /// modular product (all land in the overflow window on `Prime32Offset99`).
    #[test]
    fn packed_fp32_32b_mul_two_fold_overflow_window() {
        type F = Prime32Offset99;
        type PF = <F as HasPacking>::Packing;
        const VECTORS: [(u32, u32); 7] = [
            (3136721438, 3536064673),
            (2498152412, 1827148629),
            (2062525777, 3207684599),
            (4027016701, 3739597742),
            (2476582663, 3902052967),
            (4161561975, 3109742861),
            (1924659530, 1057556213),
        ];
        for (x, y) in VECTORS {
            let a = F::from_canonical_u32(x);
            let b = F::from_canonical_u32(y);
            let got = (PF::broadcast(a) * PF::broadcast(b)).extract(0);
            assert_eq!(got, a * b, "packed 32b mul mismatch for {x} * {y}");
        }
    }

    #[test]
    fn fp32_broadcast_and_extract_roundtrip() {
        type F = Prime24Offset3;
        type PF = <F as HasPacking>::Packing;
        check_broadcast_roundtrip::<F, PF>(F::from_u64(42));
    }

    #[test]
    fn packed_fp64_40b_add_sub_mul() {
        type F = Prime40Offset195;
        type PF = <F as HasPacking>::Packing;
        check_packed_add_sub_mul::<F, PF>(0xaa40_bb40_cc40_dd40);
    }

    #[test]
    fn packed_fp64_64b_add_sub_mul() {
        type F = Prime64Offset59;
        type PF = <F as HasPacking>::Packing;
        check_packed_add_sub_mul::<F, PF>(0xaa64_bb64_cc64_dd64);
    }

    #[test]
    fn fp64_broadcast_and_extract_roundtrip() {
        type F = Prime40Offset195;
        type PF = <F as HasPacking>::Packing;
        check_broadcast_roundtrip::<F, PF>(F::from_u64(42));
    }
}
