//! Packed extension field types using transpose-based packing.
//!
//! A `PackedFp2` stores `[PF; 2]` where `PF` is the packed base field.
//! Each `PF` lane contains the corresponding coefficient of an `Fp2` element.
//! This enables WIDTH-fold parallel arithmetic over `Fp2` using existing SIMD
//! base-field operations.

use crate::algebra::fields::ext::{Fp2, Fp2Config, Fp4, Fp4Config};
use crate::algebra::fields::packed::{HasPacking, PackedField, PackedValue};
use crate::primitives::serialization::Valid;
use crate::FieldCore;
use core::ops::{Add, Mul, Sub};

/// Packed `Fp2` elements stored in transpose layout: `[PF; 2]`.
///
/// If `PF` has width `W`, this represents `W` parallel `Fp2` values.
pub struct PackedFp2<F: FieldCore, C: Fp2Config<F>, PF: PackedField<Scalar = F>> {
    pub c0: PF,
    pub c1: PF,
    _marker: std::marker::PhantomData<fn() -> (F, C)>,
}

impl<F: FieldCore, C: Fp2Config<F>, PF: PackedField<Scalar = F>> Clone for PackedFp2<F, C, PF> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldCore, C: Fp2Config<F>, PF: PackedField<Scalar = F>> Copy for PackedFp2<F, C, PF> {}

impl<F: FieldCore, C: Fp2Config<F>, PF: PackedField<Scalar = F>> std::fmt::Debug
    for PackedFp2<F, C, PF>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedFp2").finish_non_exhaustive()
    }
}

impl<F: FieldCore, C: Fp2Config<F>, PF: PackedField<Scalar = F>> PackedFp2<F, C, PF> {
    #[inline]
    pub fn new(c0: PF, c1: PF) -> Self {
        Self {
            c0,
            c1,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline]
    fn mul_nr(x: PF) -> PF {
        if C::IS_NEG_ONE {
            let zero = PF::broadcast(F::zero());
            zero - x
        } else {
            PF::broadcast(C::non_residue()) * x
        }
    }
}

impl<F, C, PF> PackedValue for PackedFp2<F, C, PF>
where
    F: FieldCore + Valid + 'static,
    C: Fp2Config<F> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Value = Fp2<F, C>;
    const WIDTH: usize = PF::WIDTH;

    fn from_fn<G>(mut f: G) -> Self
    where
        G: FnMut(usize) -> Self::Value,
    {
        let mut c0s = Vec::with_capacity(PF::WIDTH);
        let mut c1s = Vec::with_capacity(PF::WIDTH);
        for i in 0..PF::WIDTH {
            let val = f(i);
            c0s.push(val.c0);
            c1s.push(val.c1);
        }
        Self::new(PF::from_fn(|i| c0s[i]), PF::from_fn(|i| c1s[i]))
    }

    fn extract(&self, lane: usize) -> Self::Value {
        Fp2::new(self.c0.extract(lane), self.c1.extract(lane))
    }
}

impl<F, C, PF> Add for PackedFp2<F, C, PF>
where
    F: FieldCore,
    C: Fp2Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.c0 + rhs.c0, self.c1 + rhs.c1)
    }
}

impl<F, C, PF> Sub for PackedFp2<F, C, PF>
where
    F: FieldCore,
    C: Fp2Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.c0 - rhs.c0, self.c1 - rhs.c1)
    }
}

impl<F, C, PF> Mul for PackedFp2<F, C, PF>
where
    F: FieldCore,
    C: Fp2Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let v0 = self.c0 * rhs.c0;
        let v1 = self.c1 * rhs.c1;
        Self::new(
            v0 + Self::mul_nr(v1),
            (self.c0 + self.c1) * (rhs.c0 + rhs.c1) - v0 - v1,
        )
    }
}

impl<F, C, PF> PackedField for PackedFp2<F, C, PF>
where
    F: FieldCore + Valid + 'static,
    C: Fp2Config<F> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Scalar = Fp2<F, C>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(PF::broadcast(value.c0), PF::broadcast(value.c1))
    }
}

impl<F, C> HasPacking for Fp2<F, C>
where
    F: FieldCore + Valid + HasPacking + 'static,
    C: Fp2Config<F> + 'static,
{
    type Packing = PackedFp2<F, C, F::Packing>;
}

/// Packed `Fp4` elements stored in transpose layout: `[PackedFp2; 2]`.
pub struct PackedFp4<
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: Fp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
> {
    pub c0: PackedFp2<F, C2, PF>,
    pub c1: PackedFp2<F, C2, PF>,
    _marker: std::marker::PhantomData<fn() -> C4>,
}

impl<F, C2, C4, PF> Clone for PackedFp4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: Fp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, C2, C4, PF> Copy for PackedFp4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: Fp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
}

impl<F, C2, C4, PF> std::fmt::Debug for PackedFp4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: Fp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedFp4").finish_non_exhaustive()
    }
}

impl<F, C2, C4, PF> PackedFp4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: Fp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    pub fn new(c0: PackedFp2<F, C2, PF>, c1: PackedFp2<F, C2, PF>) -> Self {
        Self {
            c0,
            c1,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F, C2, C4, PF> PackedValue for PackedFp4<F, C2, C4, PF>
where
    F: FieldCore + Valid + 'static,
    C2: Fp2Config<F> + 'static,
    C4: Fp4Config<F, C2> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Value = Fp4<F, C2, C4>;
    const WIDTH: usize = PF::WIDTH;

    fn from_fn<G>(mut f: G) -> Self
    where
        G: FnMut(usize) -> Self::Value,
    {
        let mut c0s: Vec<Fp2<F, C2>> = Vec::with_capacity(PF::WIDTH);
        let mut c1s: Vec<Fp2<F, C2>> = Vec::with_capacity(PF::WIDTH);
        for i in 0..PF::WIDTH {
            let val = f(i);
            c0s.push(val.c0);
            c1s.push(val.c1);
        }
        Self::new(
            PackedFp2::from_fn(|i| c0s[i]),
            PackedFp2::from_fn(|i| c1s[i]),
        )
    }

    fn extract(&self, lane: usize) -> Self::Value {
        Fp4::new(self.c0.extract(lane), self.c1.extract(lane))
    }
}

impl<F, C2, C4, PF> Add for PackedFp4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: Fp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.c0 + rhs.c0, self.c1 + rhs.c1)
    }
}

impl<F, C2, C4, PF> Sub for PackedFp4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: Fp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.c0 - rhs.c0, self.c1 - rhs.c1)
    }
}

impl<F, C2, C4, PF> Mul for PackedFp4<F, C2, C4, PF>
where
    F: FieldCore + Valid + 'static,
    C2: Fp2Config<F> + 'static,
    C4: Fp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let nr2 = PackedFp2::broadcast(C4::non_residue());
        let v0 = self.c0 * rhs.c0;
        let v1 = self.c1 * rhs.c1;
        Self::new(
            v0 + nr2 * v1,
            (self.c0 + self.c1) * (rhs.c0 + rhs.c1) - v0 - v1,
        )
    }
}

impl<F, C2, C4, PF> PackedField for PackedFp4<F, C2, C4, PF>
where
    F: FieldCore + Valid + 'static,
    C2: Fp2Config<F> + 'static,
    C4: Fp4Config<F, C2> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Scalar = Fp4<F, C2, C4>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(
            PackedFp2::broadcast(value.c0),
            PackedFp2::broadcast(value.c1),
        )
    }
}

impl<F, C2, C4> HasPacking for Fp4<F, C2, C4>
where
    F: FieldCore + Valid + HasPacking + 'static,
    C2: Fp2Config<F> + 'static,
    C4: Fp4Config<F, C2> + 'static,
{
    type Packing = PackedFp4<F, C2, C4, F::Packing>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::ext::{Ext2, Ext4, TwoNr, UnitNr};
    use crate::algebra::Fp64;
    use crate::{FieldSampling, FromSmallInt};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Fp64<4294967197>;
    type E2 = Ext2<F>;
    type E4 = Ext4<F>;
    type PE2 = PackedFp2<F, TwoNr, <F as HasPacking>::Packing>;
    type PE4 = PackedFp4<F, TwoNr, UnitNr, <F as HasPacking>::Packing>;

    #[test]
    fn packed_fp2_add() {
        let mut rng = StdRng::seed_from_u64(100);
        let width = <PE2 as PackedValue>::WIDTH;
        let a_elems: Vec<E2> = (0..width).map(|_| E2::sample(&mut rng)).collect();
        let b_elems: Vec<E2> = (0..width).map(|_| E2::sample(&mut rng)).collect();

        let pa = PE2::from_fn(|i| a_elems[i]);
        let pb = PE2::from_fn(|i| b_elems[i]);
        let pc = pa + pb;

        for i in 0..width {
            assert_eq!(pc.extract(i), a_elems[i] + b_elems[i]);
        }
    }

    #[test]
    fn packed_fp2_mul() {
        let mut rng = StdRng::seed_from_u64(200);
        let width = <PE2 as PackedValue>::WIDTH;
        let a_elems: Vec<E2> = (0..width).map(|_| E2::sample(&mut rng)).collect();
        let b_elems: Vec<E2> = (0..width).map(|_| E2::sample(&mut rng)).collect();

        let pa = PE2::from_fn(|i| a_elems[i]);
        let pb = PE2::from_fn(|i| b_elems[i]);
        let pc = pa * pb;

        for i in 0..width {
            assert_eq!(
                pc.extract(i),
                a_elems[i] * b_elems[i],
                "packed Fp2 mul mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn packed_fp2_broadcast() {
        let val = E2::new(F::from_u64(7), F::from_u64(11));
        let packed = PE2::broadcast(val);
        let width = <PE2 as PackedValue>::WIDTH;
        for i in 0..width {
            assert_eq!(packed.extract(i), val);
        }
    }

    #[test]
    fn packed_fp4_mul() {
        let mut rng = StdRng::seed_from_u64(300);
        let width = <PE4 as PackedValue>::WIDTH;
        let a_elems: Vec<E4> = (0..width).map(|_| E4::sample(&mut rng)).collect();
        let b_elems: Vec<E4> = (0..width).map(|_| E4::sample(&mut rng)).collect();

        let pa = PE4::from_fn(|i| a_elems[i]);
        let pb = PE4::from_fn(|i| b_elems[i]);
        let pc = pa * pb;

        for i in 0..width {
            assert_eq!(
                pc.extract(i),
                a_elems[i] * b_elems[i],
                "packed Fp4 mul mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn pack_unpack_roundtrip_fp2() {
        let mut rng = StdRng::seed_from_u64(400);
        let width = <PE2 as PackedValue>::WIDTH;
        let elems: Vec<E2> = (0..width * 3).map(|_| E2::sample(&mut rng)).collect();

        let packed = PE2::pack_slice(&elems);
        let unpacked = PE2::unpack_slice(&packed);

        assert_eq!(elems, unpacked);
    }
}
