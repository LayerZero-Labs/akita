//! Packed additive-only backends used by one-hot kernels.

use super::packed::PackedValue;
use super::wide::{Fp32x2i32, Fp64x4i32, HasAdditiveWide};
use super::{Fp128, Fp32, Fp64};
use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

/// Scalar type that supports lane-wise additive packing.
pub trait AdditiveScalar:
    'static
    + Copy
    + Default
    + Send
    + Sync
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Neg<Output = Self>
{
}

impl<T> AdditiveScalar for T where
    T: 'static
        + Copy
        + Default
        + Send
        + Sync
        + Add<Output = T>
        + AddAssign
        + Sub<Output = T>
        + SubAssign
        + Neg<Output = T>
{
}

/// Packed additive execution over a scalar additive type.
pub trait PackedAdditive:
    PackedValue<Value = Self::Scalar>
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Neg<Output = Self>
{
    /// Scalar value stored in each lane.
    type Scalar: AdditiveScalar;

    /// Broadcast one scalar across all lanes.
    fn broadcast(value: Self::Scalar) -> Self;

    /// Packed zero.
    #[inline]
    fn zero() -> Self {
        Self::broadcast(Self::Scalar::default())
    }
}

/// Scalar fallback additive packing with one lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct NoAdditivePacking<T>(pub [T; 1]);

impl<T> PackedValue for NoAdditivePacking<T>
where
    T: AdditiveScalar,
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

impl<T> Add for NoAdditivePacking<T>
where
    T: AdditiveScalar,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0]])
    }
}

impl<T> AddAssign for NoAdditivePacking<T>
where
    T: AdditiveScalar,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
    }
}

impl<T> Sub for NoAdditivePacking<T>
where
    T: AdditiveScalar,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0]])
    }
}

impl<T> SubAssign for NoAdditivePacking<T>
where
    T: AdditiveScalar,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.0[0];
    }
}

impl<T> Neg for NoAdditivePacking<T>
where
    T: AdditiveScalar,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0]])
    }
}

impl<T> PackedAdditive for NoAdditivePacking<T>
where
    T: AdditiveScalar,
{
    type Scalar = T;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self([value])
    }
}

/// Scalar type -> additive packed backend association.
pub trait HasAdditivePacking: HasAdditiveWide {
    /// Packed additive backend over [`HasAdditiveWide::AdditiveWide`].
    type AdditivePacking: PackedAdditive<Scalar = Self::AdditiveWide>;
}

/// Selected packed additive backend for signed `i32` accumulators.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub type SignedPacking = super::packed_neon::PackedI32Neon;

/// Selected packed additive backend for signed `i32` accumulators.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
pub type SignedPacking = NoAdditivePacking<i32>;

/// Selected packed additive backend for `Fp128` additive-wide accumulators.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub type Fp128AdditivePacking = super::packed_neon::PackedFp128x6i32Neon;

/// Selected packed additive backend for `Fp128` additive-wide accumulators.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
pub type Fp128AdditivePacking = NoAdditivePacking<Fp128x6i32>;

impl<const P: u32> HasAdditivePacking for Fp32<P> {
    type AdditivePacking = NoAdditivePacking<Fp32x2i32>;
}

impl<const P: u64> HasAdditivePacking for Fp64<P> {
    type AdditivePacking = NoAdditivePacking<Fp64x4i32>;
}

impl<const P: u128> HasAdditivePacking for Fp128<P> {
    type AdditivePacking = Fp128AdditivePacking;
}

#[cfg(test)]
mod tests {
    use super::{HasAdditivePacking, PackedAdditive, PackedValue, SignedPacking};
    use crate::algebra::fields::{Fp128x6i32, Prime128M8M4M1M0};
    use crate::{FieldSampling, FromSmallInt};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn signed_packing_roundtrip_and_add() {
        type P = SignedPacking;
        let values = [1i32, -3, 7, 11, -5, 9, 2];
        let (packed, suffix) = P::pack_slice_with_suffix(&values);
        let mut out = P::unpack_slice(&packed);
        out.extend_from_slice(suffix);
        assert_eq!(out, values);

        let one = P::broadcast(1);
        let added: Vec<P> = packed.iter().copied().map(|x| x + one).collect();
        let mut add_out = P::unpack_slice(&added);
        add_out.extend(suffix.iter().map(|&x| x + 1));
        let expected: Vec<i32> = values.iter().map(|&x| x + 1).collect();
        assert_eq!(add_out, expected);
    }

    #[test]
    fn fp128_additive_backend_matches_scalar_lanes() {
        type F = Prime128M8M4M1M0;
        type P = <F as HasAdditivePacking>::AdditivePacking;

        let mut rng = StdRng::seed_from_u64(7);
        let len = P::WIDTH * 5 + 1;
        let scalars: Vec<Fp128x6i32> = (0..len)
            .map(|_| Fp128x6i32::from(F::sample(&mut rng)))
            .collect();

        let (packed, suffix) = P::pack_slice_with_suffix(&scalars);
        let mut roundtrip = P::unpack_slice(&packed);
        roundtrip.extend_from_slice(suffix);
        assert_eq!(roundtrip, scalars);

        let one = Fp128x6i32::from(F::from_i32(1));
        let added: Vec<P> = packed
            .iter()
            .copied()
            .map(|x| x + P::broadcast(one))
            .collect();
        let mut add_out = P::unpack_slice(&added);
        add_out.extend(suffix.iter().map(|&x| x + one));
        let expected: Vec<Fp128x6i32> = scalars.iter().copied().map(|x| x + one).collect();
        assert_eq!(add_out, expected);
    }
}
