//! Packed extension field types using transpose-based packing.
//!
//! A `PackedFp2` stores `[PF; 2]` where `PF` is the packed base field.
//! Each `PF` lane contains the corresponding coefficient of an `Fp2` element.
//! This enables WIDTH-fold parallel arithmetic over `Fp2` using existing SIMD
//! base-field operations.

use crate::fields::ext::{
    Fp2, Fp2Config, PowerBasisFp4, PowerBasisFp4Config, PowerBasisFp4MulBackend, TowerBasisFp4,
    TowerBasisFp4Config,
};
use crate::fields::packed::{HasPacking, PackedField, PackedValue};
use crate::FieldCore;
use akita_serialization::Valid;
use core::ops::{Add, Mul, Sub};

/// Packed `Fp2` elements stored in transpose layout: `[PF; 2]`.
///
/// If `PF` has width `W`, this represents `W` parallel `Fp2` values.
pub struct PackedFp2<F: FieldCore, C: Fp2Config<F>, PF: PackedField<Scalar = F>> {
    /// Degree-0 coefficient (packed across SIMD lanes).
    pub c0: PF,
    /// Degree-1 coefficient (packed across SIMD lanes).
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
    /// Create a `PackedFp2` from its two packed coefficients.
    #[inline]
    pub fn new(c0: PF, c1: PF) -> Self {
        Self {
            c0,
            c1,
            _marker: std::marker::PhantomData,
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
            c0s.push(val.coeffs[0]);
            c1s.push(val.coeffs[1]);
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let (c0, c1) = PF::fp2_mul::<C>(self.c0, self.c1, rhs.c0, rhs.c1);
        Self::new(c0, c1)
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
        Self::new(
            PF::broadcast(value.coeffs[0]),
            PF::broadcast(value.coeffs[1]),
        )
    }
}

impl<F, C> HasPacking for Fp2<F, C>
where
    F: FieldCore + Valid + HasPacking + 'static,
    C: Fp2Config<F> + 'static,
{
    type Packing = PackedFp2<F, C, F::Packing>;
}

/// Packed `TowerBasisFp4` elements stored in transpose layout: `[PackedFp2; 2]`.
pub struct PackedTowerBasisFp4<
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
> {
    /// Packed tower-basis coefficients `[b0, b1]`.
    pub coeffs: [PackedFp2<F, C2, PF>; 2],
    _marker: std::marker::PhantomData<fn() -> C4>,
}

impl<F, C2, C4, PF> Clone for PackedTowerBasisFp4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, C2, C4, PF> Copy for PackedTowerBasisFp4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
}

impl<F, C2, C4, PF> std::fmt::Debug for PackedTowerBasisFp4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedTowerBasisFp4")
            .finish_non_exhaustive()
    }
}

impl<F, C2, C4, PF> PackedTowerBasisFp4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    /// Create a `PackedTowerBasisFp4` from its two `PackedFp2` halves.
    #[inline]
    pub fn new(c0: PackedFp2<F, C2, PF>, c1: PackedFp2<F, C2, PF>) -> Self {
        Self {
            coeffs: [c0, c1],
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F, C2, C4, PF> PackedValue for PackedTowerBasisFp4<F, C2, C4, PF>
where
    F: FieldCore + Valid + 'static,
    C2: Fp2Config<F> + 'static,
    C4: TowerBasisFp4Config<F, C2> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Value = TowerBasisFp4<F, C2, C4>;
    const WIDTH: usize = PF::WIDTH;

    fn from_fn<G>(mut f: G) -> Self
    where
        G: FnMut(usize) -> Self::Value,
    {
        let mut c0s: Vec<Fp2<F, C2>> = Vec::with_capacity(PF::WIDTH);
        let mut c1s: Vec<Fp2<F, C2>> = Vec::with_capacity(PF::WIDTH);
        for i in 0..PF::WIDTH {
            let val = f(i);
            c0s.push(val.coeffs[0]);
            c1s.push(val.coeffs[1]);
        }
        Self::new(
            PackedFp2::from_fn(|i| c0s[i]),
            PackedFp2::from_fn(|i| c1s[i]),
        )
    }

    fn extract(&self, lane: usize) -> Self::Value {
        TowerBasisFp4::new(self.coeffs[0].extract(lane), self.coeffs[1].extract(lane))
    }
}

impl<F, C2, C4, PF> Add for PackedTowerBasisFp4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(
            self.coeffs[0] + rhs.coeffs[0],
            self.coeffs[1] + rhs.coeffs[1],
        )
    }
}

impl<F, C2, C4, PF> Sub for PackedTowerBasisFp4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(
            self.coeffs[0] - rhs.coeffs[0],
            self.coeffs[1] - rhs.coeffs[1],
        )
    }
}

impl<F, C2, C4, PF> Mul for PackedTowerBasisFp4<F, C2, C4, PF>
where
    F: FieldCore + Valid + 'static,
    C2: Fp2Config<F> + 'static,
    C4: TowerBasisFp4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let [c0, c1, c2, c3] = PF::tower_basis_fp4_mul::<C2, C4>(
            [
                self.coeffs[0].c0,
                self.coeffs[1].c0,
                self.coeffs[0].c1,
                self.coeffs[1].c1,
            ],
            [
                rhs.coeffs[0].c0,
                rhs.coeffs[1].c0,
                rhs.coeffs[0].c1,
                rhs.coeffs[1].c1,
            ],
        );
        Self::new(PackedFp2::new(c0, c2), PackedFp2::new(c1, c3))
    }
}

impl<F, C2, C4, PF> PackedField for PackedTowerBasisFp4<F, C2, C4, PF>
where
    F: FieldCore + Valid + PowerBasisFp4MulBackend<C2> + 'static,
    C2: Fp2Config<F> + 'static,
    C4: TowerBasisFp4Config<F, C2> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Scalar = TowerBasisFp4<F, C2, C4>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(
            PackedFp2::broadcast(value.coeffs[0]),
            PackedFp2::broadcast(value.coeffs[1]),
        )
    }
}

impl<F, C2, C4> HasPacking for TowerBasisFp4<F, C2, C4>
where
    F: FieldCore + Valid + HasPacking + PowerBasisFp4MulBackend<C2> + 'static,
    C2: Fp2Config<F> + 'static,
    C4: TowerBasisFp4Config<F, C2> + 'static,
{
    type Packing = PackedTowerBasisFp4<F, C2, C4, F::Packing>;
}

/// Packed `PowerBasisFp4` elements stored as `[PF; 4]`.
pub struct PackedPowerBasisFp4<F: FieldCore, C: PowerBasisFp4Config<F>, PF: PackedField<Scalar = F>>
{
    /// Packed coefficients in power-basis order.
    pub coeffs: [PF; 4],
    _marker: std::marker::PhantomData<fn() -> (F, C)>,
}

impl<F, C, PF> Clone for PackedPowerBasisFp4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFp4Config<F>,
    PF: PackedField<Scalar = F>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, C, PF> Copy for PackedPowerBasisFp4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFp4Config<F>,
    PF: PackedField<Scalar = F>,
{
}

impl<F, C, PF> std::fmt::Debug for PackedPowerBasisFp4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFp4Config<F>,
    PF: PackedField<Scalar = F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedPowerBasisFp4")
            .finish_non_exhaustive()
    }
}

impl<F, C, PF> PackedPowerBasisFp4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFp4Config<F>,
    PF: PackedField<Scalar = F>,
{
    /// Create a packed value from packed power-basis coefficients.
    #[inline]
    pub fn new(coeffs: [PF; 4]) -> Self {
        Self {
            coeffs,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F, C, PF> PackedValue for PackedPowerBasisFp4<F, C, PF>
where
    F: FieldCore + Valid + 'static,
    C: PowerBasisFp4Config<F> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Value = PowerBasisFp4<F, C>;
    const WIDTH: usize = PF::WIDTH;

    fn from_fn<G>(mut f: G) -> Self
    where
        G: FnMut(usize) -> Self::Value,
    {
        let mut coeffs: [Vec<F>; 4] = std::array::from_fn(|_| Vec::with_capacity(PF::WIDTH));
        for i in 0..PF::WIDTH {
            let val = f(i);
            for (j, coeff) in val.coeffs.into_iter().enumerate() {
                coeffs[j].push(coeff);
            }
        }
        Self::new(std::array::from_fn(|j| PF::from_fn(|i| coeffs[j][i])))
    }

    fn extract(&self, lane: usize) -> Self::Value {
        PowerBasisFp4::new(std::array::from_fn(|j| self.coeffs[j].extract(lane)))
    }
}

impl<F, C, PF> Add for PackedPowerBasisFp4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFp4Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}

impl<F, C, PF> Sub for PackedPowerBasisFp4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFp4Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}

impl<F, C, PF> Mul for PackedPowerBasisFp4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFp4Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(PF::power_basis_fp4_mul::<C>(self.coeffs, rhs.coeffs))
    }
}

impl<F, C, PF> PackedField for PackedPowerBasisFp4<F, C, PF>
where
    F: FieldCore + Valid + PowerBasisFp4MulBackend<C> + 'static,
    C: PowerBasisFp4Config<F> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Scalar = PowerBasisFp4<F, C>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(std::array::from_fn(|i| PF::broadcast(value.coeffs[i])))
    }
}

impl<F, C> HasPacking for PowerBasisFp4<F, C>
where
    F: FieldCore + Valid + HasPacking + PowerBasisFp4MulBackend<C> + 'static,
    C: PowerBasisFp4Config<F> + 'static,
{
    type Packing = PackedPowerBasisFp4<F, C, F::Packing>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::ext::{Ext2, PowerBasisFp4, TowerBasisFp4, TwoNr, UnitNr};
    use crate::Fp64;
    use crate::RandomSampling;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Fp64<4294967197>;
    type E2 = Ext2<F>;
    type E4 = TowerBasisFp4<F, TwoNr, UnitNr>;
    type P4 = PowerBasisFp4<F, TwoNr>;
    type PE2 = PackedFp2<F, TwoNr, <F as HasPacking>::Packing>;
    type PE4 = PackedTowerBasisFp4<F, TwoNr, UnitNr, <F as HasPacking>::Packing>;
    type PP4 = PackedPowerBasisFp4<F, TwoNr, <F as HasPacking>::Packing>;

    #[test]
    fn packed_fp2_add() {
        let mut rng = StdRng::seed_from_u64(100);
        let width = <PE2 as PackedValue>::WIDTH;
        let a_elems: Vec<E2> = (0..width).map(|_| E2::random(&mut rng)).collect();
        let b_elems: Vec<E2> = (0..width).map(|_| E2::random(&mut rng)).collect();

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
        let a_elems: Vec<E2> = (0..width).map(|_| E2::random(&mut rng)).collect();
        let b_elems: Vec<E2> = (0..width).map(|_| E2::random(&mut rng)).collect();

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
        let a_elems: Vec<E4> = (0..width).map(|_| E4::random(&mut rng)).collect();
        let b_elems: Vec<E4> = (0..width).map(|_| E4::random(&mut rng)).collect();

        let pa = PE4::from_fn(|i| a_elems[i]);
        let pb = PE4::from_fn(|i| b_elems[i]);
        let pc = pa * pb;

        for i in 0..width {
            assert_eq!(
                pc.extract(i),
                a_elems[i] * b_elems[i],
                "packed TowerBasisFp4 mul mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn packed_power_basis_fp4_mul() {
        let mut rng = StdRng::seed_from_u64(350);
        let width = <PP4 as PackedValue>::WIDTH;
        let a_elems: Vec<P4> = (0..width).map(|_| P4::random(&mut rng)).collect();
        let b_elems: Vec<P4> = (0..width).map(|_| P4::random(&mut rng)).collect();

        let pa = PP4::from_fn(|i| a_elems[i]);
        let pb = PP4::from_fn(|i| b_elems[i]);
        let pc = pa * pb;

        for i in 0..width {
            assert_eq!(
                pc.extract(i),
                a_elems[i] * b_elems[i],
                "packed PowerBasisFp4 mul mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn pack_unpack_roundtrip_fp2() {
        let mut rng = StdRng::seed_from_u64(400);
        let width = <PE2 as PackedValue>::WIDTH;
        let elems: Vec<E2> = (0..width * 3).map(|_| E2::random(&mut rng)).collect();

        let packed = PE2::pack_slice(&elems);
        let unpacked = PE2::unpack_slice(&packed);

        assert_eq!(elems, unpacked);
    }
}
