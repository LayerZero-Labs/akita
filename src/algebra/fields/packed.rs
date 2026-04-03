//! Packed field abstractions and architecture-specific SIMD backends.

use crate::algebra::fields::{Fp128, Fp32, Fp64};
use crate::FieldCore;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

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
pub type Fp128Packing<const P: u128> = super::packed_neon::PackedFp128Neon<P>;

/// Selected packed backend for `Fp128`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq"
))]
pub type Fp128Packing<const P: u128> = super::packed_avx512::PackedFp128Avx512<P>;

/// Selected packed backend for `Fp128`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512dq"))
))]
pub type Fp128Packing<const P: u128> = super::packed_avx2::PackedFp128Avx2<P>;

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
pub type Fp32Packing<const P: u32> = super::packed_neon::PackedFp32Neon<P>;

/// Selected packed backend for `Fp32`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq"
))]
pub type Fp32Packing<const P: u32> = super::packed_avx512::PackedFp32Avx512<P>;

/// Selected packed backend for `Fp32`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512dq"))
))]
pub type Fp32Packing<const P: u32> = super::packed_avx2::PackedFp32Avx2<P>;

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
pub type Fp64Packing<const P: u64> = super::packed_neon::PackedFp64Neon<P>;

/// Selected packed backend for `Fp64`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq"
))]
pub type Fp64Packing<const P: u64> = super::packed_avx512::PackedFp64Avx512<P>;

/// Selected packed backend for `Fp64`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512dq"))
))]
pub type Fp64Packing<const P: u64> = super::packed_avx2::PackedFp64Avx2<P>;

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
    use crate::algebra::fields::{
        Pow2Offset24Field, Pow2Offset31Field, Pow2Offset32Field, Pow2Offset40Field,
        Pow2Offset64Field, Prime128Offset275,
    };
    use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt};
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    fn rand_u128<R: RngCore>(rng: &mut R) -> u128 {
        let lo = rng.next_u64() as u128;
        let hi = rng.next_u64() as u128;
        lo | (hi << 64)
    }

    fn check_packed_add_sub_mul<F, PF>(seed: u64)
    where
        F: FieldCore + FieldSampling + PartialEq + std::fmt::Debug,
        PF: PackedField<Scalar = F> + PackedValue<Value = F>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let len = PF::WIDTH * 17 + 3;
        let lhs: Vec<F> = (0..len).map(|_| FieldSampling::sample(&mut rng)).collect();
        let rhs: Vec<F> = (0..len).map(|_| FieldSampling::sample(&mut rng)).collect();

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
        type F = Pow2Offset24Field;
        type PF = <F as HasPacking>::Packing;
        check_packed_add_sub_mul::<F, PF>(0xaa24_bb24_cc24_dd24);
    }

    #[test]
    fn packed_fp32_31b_add_sub_mul() {
        type F = Pow2Offset31Field;
        type PF = <F as HasPacking>::Packing;
        check_packed_add_sub_mul::<F, PF>(0xaa31_bb31_cc31_dd31);
    }

    #[test]
    fn packed_fp32_32b_add_sub_mul() {
        type F = Pow2Offset32Field;
        type PF = <F as HasPacking>::Packing;
        check_packed_add_sub_mul::<F, PF>(0xaa32_bb32_cc32_dd32);
    }

    #[test]
    fn fp32_broadcast_and_extract_roundtrip() {
        type F = Pow2Offset24Field;
        type PF = <F as HasPacking>::Packing;
        check_broadcast_roundtrip::<F, PF>(F::from_u64(42));
    }

    #[test]
    fn packed_fp64_40b_add_sub_mul() {
        type F = Pow2Offset40Field;
        type PF = <F as HasPacking>::Packing;
        check_packed_add_sub_mul::<F, PF>(0xaa40_bb40_cc40_dd40);
    }

    #[test]
    fn packed_fp64_64b_add_sub_mul() {
        type F = Pow2Offset64Field;
        type PF = <F as HasPacking>::Packing;
        check_packed_add_sub_mul::<F, PF>(0xaa64_bb64_cc64_dd64);
    }

    #[test]
    fn fp64_broadcast_and_extract_roundtrip() {
        type F = Pow2Offset40Field;
        type PF = <F as HasPacking>::Packing;
        check_broadcast_roundtrip::<F, PF>(F::from_u64(42));
    }
}
