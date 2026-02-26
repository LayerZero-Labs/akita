//! Packed field abstractions and Fp128 field backends.
//!
//! This module is intentionally field-scoped for now (no ring/protocol wiring yet).

use crate::algebra::fields::Fp128;
use crate::FieldCore;
use core::mem::align_of;
use core::ops::{Add, Mul, Sub};
use core::slice;

/// Array-like packed values over a scalar type.
pub trait PackedValue: 'static + Copy + Send + Sync {
    /// Scalar value type carried by each lane.
    type Value: 'static + Copy + Send + Sync;

    /// Number of scalar lanes.
    const WIDTH: usize;

    /// Reinterpret a lane slice as one packed value.
    ///
    /// # Panics
    ///
    /// Panics if `slice.len() != Self::WIDTH`.
    fn from_slice(slice: &[Self::Value]) -> &Self;

    /// Reinterpret a mutable lane slice as one packed value.
    ///
    /// # Panics
    ///
    /// Panics if `slice.len() != Self::WIDTH`.
    fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self;

    /// Build from a lane generator.
    fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value;

    /// Borrow as scalar lanes.
    fn as_slice(&self) -> &[Self::Value];

    /// Mutably borrow as scalar lanes.
    fn as_slice_mut(&mut self) -> &mut [Self::Value];

    /// Reinterpret a scalar slice as packed values.
    ///
    /// # Panics
    ///
    /// Panics if the length is not divisible by `WIDTH`.
    #[inline]
    fn pack_slice(buf: &[Self::Value]) -> &[Self] {
        assert!(
            buf.len() % Self::WIDTH == 0,
            "slice length {} must be divisible by WIDTH {}",
            buf.len(),
            Self::WIDTH
        );
        // This holds for all repr(transparent) wrappers used here.
        debug_assert!(align_of::<Self>() <= align_of::<Self::Value>());
        let ptr = buf.as_ptr().cast::<Self>();
        let n = buf.len() / Self::WIDTH;
        // SAFETY: length divisibility and alignment checked above.
        unsafe { slice::from_raw_parts(ptr, n) }
    }

    /// Reinterpret a mutable scalar slice as packed values.
    ///
    /// # Panics
    ///
    /// Panics if the length is not divisible by `WIDTH`.
    #[inline]
    fn pack_slice_mut(buf: &mut [Self::Value]) -> &mut [Self] {
        assert!(
            buf.len() % Self::WIDTH == 0,
            "slice length {} must be divisible by WIDTH {}",
            buf.len(),
            Self::WIDTH
        );
        debug_assert!(align_of::<Self>() <= align_of::<Self::Value>());
        let ptr = buf.as_mut_ptr().cast::<Self>();
        let n = buf.len() / Self::WIDTH;
        // SAFETY: length divisibility and alignment checked above.
        unsafe { slice::from_raw_parts_mut(ptr, n) }
    }

    /// Packed prefix + scalar suffix split.
    #[inline]
    fn pack_slice_with_suffix(buf: &[Self::Value]) -> (&[Self], &[Self::Value]) {
        let split = buf.len() - (buf.len() % Self::WIDTH);
        let (packed, suffix) = buf.split_at(split);
        (Self::pack_slice(packed), suffix)
    }

    /// Mutable packed prefix + scalar suffix split.
    #[inline]
    fn pack_slice_with_suffix_mut(buf: &mut [Self::Value]) -> (&mut [Self], &mut [Self::Value]) {
        let split = buf.len() - (buf.len() % Self::WIDTH);
        let (packed, suffix) = buf.split_at_mut(split);
        (Self::pack_slice_mut(packed), suffix)
    }

    /// Extract one lane.
    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        self.as_slice()[lane]
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
    fn from_slice(slice: &[Self::Value]) -> &Self {
        assert_eq!(slice.len(), 1);
        // SAFETY: `NoPacking<T>` is repr(transparent) over `[T;1]`.
        unsafe { &*slice.as_ptr().cast::<Self>() }
    }

    #[inline]
    fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
        assert_eq!(slice.len(), 1);
        // SAFETY: `NoPacking<T>` is repr(transparent) over `[T;1]`.
        unsafe { &mut *slice.as_mut_ptr().cast::<Self>() }
    }

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        Self([f(0)])
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Value] {
        &self.0
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [Self::Value] {
        &mut self.0
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
    use core::ops::{Add, Mul, Sub};

    /// Number of packed `Fp128` lanes in this backend.
    pub const WIDTH: usize = 4;

    /// Packed `Fp128` lane container for AArch64-first backend work.
    ///
    /// Arithmetic is lane-wise and intentionally keeps scalar `Fp128` semantics.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct PackedFp128Neon<const P: u128>(pub [Fp128<P>; WIDTH]);

    impl<const P: u128> PackedValue for PackedFp128Neon<P> {
        type Value = Fp128<P>;
        const WIDTH: usize = WIDTH;

        #[inline]
        fn from_slice(slice: &[Self::Value]) -> &Self {
            assert_eq!(slice.len(), WIDTH);
            // SAFETY: repr(transparent) over `[Fp128<P>; WIDTH]`.
            unsafe { &*slice.as_ptr().cast::<Self>() }
        }

        #[inline]
        fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
            assert_eq!(slice.len(), WIDTH);
            // SAFETY: repr(transparent) over `[Fp128<P>; WIDTH]`.
            unsafe { &mut *slice.as_mut_ptr().cast::<Self>() }
        }

        #[inline]
        fn from_fn<F>(f: F) -> Self
        where
            F: FnMut(usize) -> Self::Value,
        {
            Self(core::array::from_fn(f))
        }

        #[inline]
        fn as_slice(&self) -> &[Self::Value] {
            &self.0
        }

        #[inline]
        fn as_slice_mut(&mut self) -> &mut [Self::Value] {
            &mut self.0
        }
    }

    impl<const P: u128> Add for PackedFp128Neon<P> {
        type Output = Self;
        #[inline]
        fn add(self, rhs: Self) -> Self {
            let [a0, a1, a2, a3] = self.0;
            let [b0, b1, b2, b3] = rhs.0;
            Self([a0 + b0, a1 + b1, a2 + b2, a3 + b3])
        }
    }

    impl<const P: u128> Sub for PackedFp128Neon<P> {
        type Output = Self;
        #[inline]
        fn sub(self, rhs: Self) -> Self {
            let [a0, a1, a2, a3] = self.0;
            let [b0, b1, b2, b3] = rhs.0;
            Self([a0 - b0, a1 - b1, a2 - b2, a3 - b3])
        }
    }

    impl<const P: u128> Mul for PackedFp128Neon<P> {
        type Output = Self;
        #[inline]
        fn mul(self, rhs: Self) -> Self {
            let [a0, a1, a2, a3] = self.0;
            let [b0, b1, b2, b3] = rhs.0;
            Self([a0 * b0, a1 * b1, a2 * b2, a3 * b3])
        }
    }

    impl<const P: u128> PackedField for PackedFp128Neon<P> {
        type Scalar = Fp128<P>;

        #[inline]
        fn broadcast(value: Self::Scalar) -> Self {
            Self([value; WIDTH])
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
    use crate::{CanonicalField, FieldCore};
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

        let mut add_out = vec![F::zero(); len];
        let mut sub_out = vec![F::zero(); len];
        let mut mul_out = vec![F::zero(); len];

        let (lhs_p, lhs_s) = PF::pack_slice_with_suffix(&lhs);
        let (rhs_p, rhs_s) = PF::pack_slice_with_suffix(&rhs);
        let (add_p, add_s) = PF::pack_slice_with_suffix_mut(&mut add_out);
        let (sub_p, sub_s) = PF::pack_slice_with_suffix_mut(&mut sub_out);
        let (mul_p, mul_s) = PF::pack_slice_with_suffix_mut(&mut mul_out);

        for ((dst, &a), &b) in add_p.iter_mut().zip(lhs_p.iter()).zip(rhs_p.iter()) {
            *dst = a + b;
        }
        for ((dst, &a), &b) in sub_p.iter_mut().zip(lhs_p.iter()).zip(rhs_p.iter()) {
            *dst = a - b;
        }
        for ((dst, &a), &b) in mul_p.iter_mut().zip(lhs_p.iter()).zip(rhs_p.iter()) {
            *dst = a * b;
        }

        for ((dst, &a), &b) in add_s.iter_mut().zip(lhs_s.iter()).zip(rhs_s.iter()) {
            *dst = a + b;
        }
        for ((dst, &a), &b) in sub_s.iter_mut().zip(lhs_s.iter()).zip(rhs_s.iter()) {
            *dst = a - b;
        }
        for ((dst, &a), &b) in mul_s.iter_mut().zip(lhs_s.iter()).zip(rhs_s.iter()) {
            *dst = a * b;
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
