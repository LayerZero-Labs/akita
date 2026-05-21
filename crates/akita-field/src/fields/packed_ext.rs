//! Packed extension field types using transpose-based packing.
//!
//! A `PackedFp2` stores `[PF; 2]` where `PF` is the packed base field.
//! Each `PF` lane contains the corresponding coefficient of an `Fp2` element.
//! This enables WIDTH-fold parallel arithmetic over `Fp2` using existing SIMD
//! base-field operations.

use crate::fields::ext::{
    Fp2, Fp2Config, PowerBasisFp4, PowerBasisFp4Config, PowerBasisFp4MulBackend, RingSubfieldFp4,
    RingSubfieldFp4MulBackend, TowerBasisFp4, TowerBasisFp4Config,
};
use crate::fields::packed::{HasPacking, PackedField, PackedValue};
use crate::{FieldCore, Invertible};
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

    #[inline(always)]
    fn inverse(self) -> Option<Self>
    where
        Self::Scalar: Invertible,
    {
        let norm = self.c0 * self.c0 - C::mul_non_residue(self.c1 * self.c1, PF::broadcast);
        let inv_norm = norm.inverse()?;
        let zero = PF::broadcast(F::zero());
        Some(Self::new(self.c0 * inv_norm, (zero - self.c1) * inv_norm))
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

    #[inline(always)]
    fn square(self) -> Self {
        let [c0, c1, c2, c3] = PF::tower_basis_fp4_mul::<C2, C4>(
            [
                self.coeffs[0].c0,
                self.coeffs[1].c0,
                self.coeffs[0].c1,
                self.coeffs[1].c1,
            ],
            [
                self.coeffs[0].c0,
                self.coeffs[1].c0,
                self.coeffs[0].c1,
                self.coeffs[1].c1,
            ],
        );
        Self::new(PackedFp2::new(c0, c2), PackedFp2::new(c1, c3))
    }

    #[inline(always)]
    fn inverse(self) -> Option<Self>
    where
        Self::Scalar: Invertible,
    {
        let v0 = self.coeffs[0].square();
        let v1 = self.coeffs[1].square();
        let nr = C4::non_residue();
        let nr_v1 = if nr.coeffs[0].is_zero() && nr.coeffs[1] == F::one() {
            PackedFp2::new(C2::mul_non_residue(v1.c1, PF::broadcast), v1.c0)
        } else {
            PackedFp2::broadcast(nr) * v1
        };
        let inv_norm = (v0 - nr_v1).inverse()?;
        let zero = PackedFp2::broadcast(Fp2::zero());
        Some(Self::new(
            self.coeffs[0] * inv_norm,
            (zero - self.coeffs[1]) * inv_norm,
        ))
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

    #[inline(always)]
    fn square(self) -> Self {
        Self::new(PF::power_basis_fp4_mul::<C>(self.coeffs, self.coeffs))
    }

    #[inline(always)]
    fn inverse(self) -> Option<Self>
    where
        Self::Scalar: Invertible,
    {
        let [a0, a1, a2, a3] = self.coeffs;
        let two = PF::broadcast(F::one() + F::one());

        let d0 =
            a0 * a0 + C::mul_w(a2 * a2, PF::broadcast) - C::mul_w(two * (a1 * a3), PF::broadcast);
        let d1 = two * (a0 * a2) - a1 * a1 - C::mul_w(a3 * a3, PF::broadcast);
        let inv_norm = (d0 * d0 - C::mul_w(d1 * d1, PF::broadcast)).inverse()?;
        let e0 = d0 * inv_norm;
        let e1 = (PF::broadcast(F::zero()) - d1) * inv_norm;

        Some(Self::new([
            a0 * e0 + C::mul_w(a2 * e1, PF::broadcast),
            PF::broadcast(F::zero()) - (a1 * e0 + C::mul_w(a3 * e1, PF::broadcast)),
            a0 * e1 + a2 * e0,
            PF::broadcast(F::zero()) - (a1 * e1 + a3 * e0),
        ]))
    }
}

impl<F, C> HasPacking for PowerBasisFp4<F, C>
where
    F: FieldCore + Valid + HasPacking + PowerBasisFp4MulBackend<C> + 'static,
    C: PowerBasisFp4Config<F> + 'static,
{
    type Packing = PackedPowerBasisFp4<F, C, F::Packing>;
}

/// Packed `RingSubfieldFp4` elements stored as `[PF; 4]`.
pub struct PackedRingSubfieldFp4<F: FieldCore, PF: PackedField<Scalar = F>> {
    /// Packed coefficients in `[1, e1, e2, e3]` order.
    pub coeffs: [PF; 4],
    _marker: std::marker::PhantomData<fn() -> F>,
}

impl<F, PF> Clone for PackedRingSubfieldFp4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, PF> Copy for PackedRingSubfieldFp4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
}

impl<F, PF> std::fmt::Debug for PackedRingSubfieldFp4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedRingSubfieldFp4")
            .finish_non_exhaustive()
    }
}

impl<F, PF> PackedRingSubfieldFp4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    /// Create a packed value from packed ring-subfield coefficients.
    #[inline]
    pub fn new(coeffs: [PF; 4]) -> Self {
        Self {
            coeffs,
            _marker: std::marker::PhantomData,
        }
    }

    /// Square using the packed ring-subfield backend hook.
    #[inline(always)]
    pub fn square(self) -> Self {
        Self::new(PF::ring_subfield_fp4_square(self.coeffs))
    }
}

impl<F, PF> PackedValue for PackedRingSubfieldFp4<F, PF>
where
    F: FieldCore + Valid + 'static,
    PF: PackedField<Scalar = F>,
{
    type Value = RingSubfieldFp4<F>;
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
        RingSubfieldFp4::new(std::array::from_fn(|j| self.coeffs[j].extract(lane)))
    }
}

impl<F, PF> Add for PackedRingSubfieldFp4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}

impl<F, PF> Sub for PackedRingSubfieldFp4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}

impl<F, PF> Mul for PackedRingSubfieldFp4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(PF::ring_subfield_fp4_mul(self.coeffs, rhs.coeffs))
    }
}

impl<F, PF> PackedField for PackedRingSubfieldFp4<F, PF>
where
    F: FieldCore + Valid + RingSubfieldFp4MulBackend + 'static,
    PF: PackedField<Scalar = F>,
{
    type Scalar = RingSubfieldFp4<F>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(std::array::from_fn(|i| PF::broadcast(value.coeffs[i])))
    }

    #[inline(always)]
    fn square(self) -> Self {
        Self::new(PF::ring_subfield_fp4_square(self.coeffs))
    }

    #[inline(always)]
    fn inverse(self) -> Option<Self>
    where
        Self::Scalar: Invertible,
    {
        Some(Self::new(PF::ring_subfield_fp4_inverse(self.coeffs)?))
    }
}

impl<F> HasPacking for RingSubfieldFp4<F>
where
    F: FieldCore + Valid + HasPacking + RingSubfieldFp4MulBackend + 'static,
{
    type Packing = PackedRingSubfieldFp4<F, F::Packing>;
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use crate::fields::ext::{Ext2, PowerBasisFp4, RingSubfieldFp4, TowerBasisFp4, TwoNr, UnitNr};
    use crate::Fp32;
    use crate::Fp64;
    use crate::Prime31Offset19;
    use crate::Prime32Offset99;
    use crate::Prime64Offset59;
    use crate::RandomSampling;
    use crate::RingCore;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Fp64<4294967197>;
    type E2 = Ext2<F>;
    type E4 = TowerBasisFp4<F, TwoNr, UnitNr>;
    type P4 = PowerBasisFp4<F, TwoNr>;
    type R4 = RingSubfieldFp4<F>;
    type PE2 = PackedFp2<F, TwoNr, <F as HasPacking>::Packing>;
    type PE4 = PackedTowerBasisFp4<F, TwoNr, UnitNr, <F as HasPacking>::Packing>;
    type PP4 = PackedPowerBasisFp4<F, TwoNr, <F as HasPacking>::Packing>;
    type PR4 = PackedRingSubfieldFp4<F, <F as HasPacking>::Packing>;
    type Mersenne31 = Fp32<{ (1u32 << 31) - 1 }>;
    type Generic30Offset16397 = Fp32<{ (1u32 << 30) - 16_397 }>;
    type Generic31Offset61 = Fp32<{ (1u32 << 31) - 61 }>;
    type Generic31Offset32787 = Fp32<{ (1u32 << 31) - 32_787 }>;
    type PP4Prime31 =
        PackedPowerBasisFp4<Prime31Offset19, TwoNr, <Prime31Offset19 as HasPacking>::Packing>;
    type PR4Prime31 =
        PackedRingSubfieldFp4<Prime31Offset19, <Prime31Offset19 as HasPacking>::Packing>;
    type PP4Mersenne31 =
        PackedPowerBasisFp4<Mersenne31, TwoNr, <Mersenne31 as HasPacking>::Packing>;
    type PR4Mersenne31 = PackedRingSubfieldFp4<Mersenne31, <Mersenne31 as HasPacking>::Packing>;
    type PP4Generic30Offset16397 = PackedPowerBasisFp4<
        Generic30Offset16397,
        TwoNr,
        <Generic30Offset16397 as HasPacking>::Packing,
    >;
    type PR4Generic30Offset16397 =
        PackedRingSubfieldFp4<Generic30Offset16397, <Generic30Offset16397 as HasPacking>::Packing>;
    type PP4Generic31Offset61 =
        PackedPowerBasisFp4<Generic31Offset61, TwoNr, <Generic31Offset61 as HasPacking>::Packing>;
    type PR4Generic31Offset61 =
        PackedRingSubfieldFp4<Generic31Offset61, <Generic31Offset61 as HasPacking>::Packing>;
    type PP4Generic31Offset32787 = PackedPowerBasisFp4<
        Generic31Offset32787,
        TwoNr,
        <Generic31Offset32787 as HasPacking>::Packing,
    >;
    type PR4Generic31Offset32787 =
        PackedRingSubfieldFp4<Generic31Offset32787, <Generic31Offset32787 as HasPacking>::Packing>;
    type R4Prime32 = RingSubfieldFp4<Prime32Offset99>;
    type PR4Prime32 =
        PackedRingSubfieldFp4<Prime32Offset99, <Prime32Offset99 as HasPacking>::Packing>;
    type E2Full = Fp2<Prime64Offset59, TwoNr>;
    type PE2Full = PackedFp2<Prime64Offset59, TwoNr, <Prime64Offset59 as HasPacking>::Packing>;

    fn fp32_ext_edge_values<const P: u32>() -> [Fp32<P>; 4] {
        [
            Fp32::<P>::from_canonical_u32(P - 1),
            Fp32::<P>::from_canonical_u32(P - 2),
            Fp32::<P>::from_canonical_u32((P - 1) / 2),
            Fp32::<P>::one(),
        ]
    }

    fn check_packed_power_basis_fp4_edge<const P: u32, PP4>()
    where
        PP4: PackedField<Scalar = PowerBasisFp4<Fp32<P>, TwoNr>>
            + PackedValue<Value = PowerBasisFp4<Fp32<P>, TwoNr>>,
    {
        let values = fp32_ext_edge_values::<P>();
        let elem = |offset: usize| {
            PowerBasisFp4::<Fp32<P>, TwoNr>::new(std::array::from_fn(|j| {
                values[(offset + j) % values.len()]
            }))
        };
        let a = PP4::from_fn(elem);
        let b = PP4::from_fn(|i| elem(i + 1));
        let product = a * b;
        let square = a.square();

        for lane in 0..PP4::WIDTH {
            let lhs = elem(lane);
            let rhs = elem(lane + 1);
            assert_eq!(
                product.extract(lane),
                lhs * rhs,
                "packed PowerBasisFp4 edge mul mismatch at lane {lane}"
            );
            assert_eq!(
                square.extract(lane),
                lhs.square(),
                "packed PowerBasisFp4 edge square mismatch at lane {lane}"
            );
        }
    }

    fn check_packed_ring_subfield_fp4_edge<const P: u32, PR4>()
    where
        PR4: PackedField<Scalar = RingSubfieldFp4<Fp32<P>>>
            + PackedValue<Value = RingSubfieldFp4<Fp32<P>>>,
    {
        let values = fp32_ext_edge_values::<P>();
        let elem = |offset: usize| {
            RingSubfieldFp4::<Fp32<P>>::new(std::array::from_fn(|j| {
                values[(offset + j) % values.len()]
            }))
        };
        let a = PR4::from_fn(elem);
        let b = PR4::from_fn(|i| elem(i + 1));
        let product = a * b;
        let square = a.square();

        for lane in 0..PR4::WIDTH {
            let lhs = elem(lane);
            let rhs = elem(lane + 1);
            assert_eq!(
                product.extract(lane),
                lhs * rhs,
                "packed RingSubfieldFp4 edge mul mismatch at lane {lane}"
            );
            assert_eq!(
                square.extract(lane),
                lhs.square(),
                "packed RingSubfieldFp4 edge square mismatch at lane {lane}"
            );
        }
    }

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
    fn packed_fp2_mul_full_word_fp64() {
        let mut rng = StdRng::seed_from_u64(201);
        let width = <PE2Full as PackedValue>::WIDTH;
        let a_elems: Vec<E2Full> = (0..width).map(|_| E2Full::random(&mut rng)).collect();
        let b_elems: Vec<E2Full> = (0..width).map(|_| E2Full::random(&mut rng)).collect();

        let pa = PE2Full::from_fn(|i| a_elems[i]);
        let pb = PE2Full::from_fn(|i| b_elems[i]);
        let pc = pa * pb;

        for i in 0..width {
            assert_eq!(
                pc.extract(i),
                a_elems[i] * b_elems[i],
                "full-word packed Fp2 mul mismatch at lane {i}"
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
    fn packed_power_basis_fp4_prime31_edge_lanes() {
        check_packed_power_basis_fp4_edge::<
            { crate::fields::pseudo_mersenne::PRIME31_OFFSET19_MODULUS },
            PP4Prime31,
        >();
    }

    #[test]
    fn packed_power_basis_fp4_mersenne31_edge_lanes() {
        check_packed_power_basis_fp4_edge::<{ (1u32 << 31) - 1 }, PP4Mersenne31>();
    }

    #[test]
    fn packed_power_basis_fp4_generic31_edge_lanes() {
        check_packed_power_basis_fp4_edge::<{ (1u32 << 31) - 61 }, PP4Generic31Offset61>();
    }

    #[test]
    fn packed_power_basis_fp4_large_generic30_edge_lanes() {
        check_packed_power_basis_fp4_edge::<{ (1u32 << 30) - 16_397 }, PP4Generic30Offset16397>();
    }

    #[test]
    fn packed_power_basis_fp4_large_generic31_edge_lanes() {
        check_packed_power_basis_fp4_edge::<{ (1u32 << 31) - 32_787 }, PP4Generic31Offset32787>();
    }

    #[test]
    fn packed_tower_basis_fp4_inverse() {
        let mut rng = StdRng::seed_from_u64(351);
        let width = <PE4 as PackedValue>::WIDTH;
        let elems: Vec<E4> = (0..width)
            .map(|_| {
                let x = E4::random(&mut rng);
                if x.is_zero() {
                    E4::one()
                } else {
                    x
                }
            })
            .collect();

        let packed = PE4::from_fn(|i| elems[i]);
        let inverted = packed.inverse().unwrap();

        for (i, elem) in elems.iter().enumerate() {
            assert_eq!(
                inverted.extract(i),
                elem.inverse().unwrap(),
                "packed TowerBasisFp4 inverse mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn packed_power_basis_fp4_inverse() {
        let mut rng = StdRng::seed_from_u64(352);
        let width = <PP4 as PackedValue>::WIDTH;
        let elems: Vec<P4> = (0..width)
            .map(|_| {
                let x = P4::random(&mut rng);
                if x.is_zero() {
                    P4::one()
                } else {
                    x
                }
            })
            .collect();

        let packed = PP4::from_fn(|i| elems[i]);
        let inverted = packed.inverse().unwrap();

        for (i, elem) in elems.iter().enumerate() {
            assert_eq!(
                inverted.extract(i),
                elem.inverse().unwrap(),
                "packed PowerBasisFp4 inverse mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn packed_ring_subfield_fp4_add() {
        let mut rng = StdRng::seed_from_u64(360);
        let width = <PR4 as PackedValue>::WIDTH;
        let a_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();
        let b_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();

        let pa = PR4::from_fn(|i| a_elems[i]);
        let pb = PR4::from_fn(|i| b_elems[i]);
        let pc = pa + pb;

        for i in 0..width {
            assert_eq!(
                pc.extract(i),
                a_elems[i] + b_elems[i],
                "packed RingSubfieldFp4 add mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn packed_ring_subfield_fp4_sub() {
        let mut rng = StdRng::seed_from_u64(361);
        let width = <PR4 as PackedValue>::WIDTH;
        let a_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();
        let b_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();

        let pa = PR4::from_fn(|i| a_elems[i]);
        let pb = PR4::from_fn(|i| b_elems[i]);
        let pc = pa - pb;

        for i in 0..width {
            assert_eq!(
                pc.extract(i),
                a_elems[i] - b_elems[i],
                "packed RingSubfieldFp4 sub mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn packed_ring_subfield_fp4_mul() {
        let mut rng = StdRng::seed_from_u64(362);
        let width = <PR4 as PackedValue>::WIDTH;
        let a_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();
        let b_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();

        let pa = PR4::from_fn(|i| a_elems[i]);
        let pb = PR4::from_fn(|i| b_elems[i]);
        let pc = pa * pb;

        for i in 0..width {
            assert_eq!(
                pc.extract(i),
                a_elems[i] * b_elems[i],
                "packed RingSubfieldFp4 mul mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn packed_ring_subfield_fp4_mul_prime32() {
        let mut rng = StdRng::seed_from_u64(365);
        let width = <PR4Prime32 as PackedValue>::WIDTH;
        let a_elems: Vec<R4Prime32> = (0..width).map(|_| R4Prime32::random(&mut rng)).collect();
        let b_elems: Vec<R4Prime32> = (0..width).map(|_| R4Prime32::random(&mut rng)).collect();

        let pa = PR4Prime32::from_fn(|i| a_elems[i]);
        let pb = PR4Prime32::from_fn(|i| b_elems[i]);
        let pc = pa * pb;

        for i in 0..width {
            assert_eq!(
                pc.extract(i),
                a_elems[i] * b_elems[i],
                "Prime32 packed RingSubfieldFp4 mul mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn packed_ring_subfield_fp4_prime31_edge_lanes() {
        check_packed_ring_subfield_fp4_edge::<
            { crate::fields::pseudo_mersenne::PRIME31_OFFSET19_MODULUS },
            PR4Prime31,
        >();
    }

    #[test]
    fn packed_ring_subfield_fp4_mersenne31_edge_lanes() {
        check_packed_ring_subfield_fp4_edge::<{ (1u32 << 31) - 1 }, PR4Mersenne31>();
    }

    #[test]
    fn packed_ring_subfield_fp4_generic31_edge_lanes() {
        check_packed_ring_subfield_fp4_edge::<{ (1u32 << 31) - 61 }, PR4Generic31Offset61>();
    }

    #[test]
    fn packed_ring_subfield_fp4_large_generic30_edge_lanes() {
        check_packed_ring_subfield_fp4_edge::<{ (1u32 << 30) - 16_397 }, PR4Generic30Offset16397>();
    }

    #[test]
    fn packed_ring_subfield_fp4_large_generic31_edge_lanes() {
        check_packed_ring_subfield_fp4_edge::<{ (1u32 << 31) - 32_787 }, PR4Generic31Offset32787>();
    }

    #[test]
    fn packed_ring_subfield_fp4_square() {
        let mut rng = StdRng::seed_from_u64(363);
        let width = <PR4 as PackedValue>::WIDTH;
        let elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();

        let packed = PR4::from_fn(|i| elems[i]);
        let squared = packed.square();

        for (i, elem) in elems.iter().enumerate() {
            assert_eq!(
                squared.extract(i),
                elem.square(),
                "packed RingSubfieldFp4 square mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn packed_ring_subfield_fp4_square_prime32() {
        let mut rng = StdRng::seed_from_u64(366);
        let width = <PR4Prime32 as PackedValue>::WIDTH;
        let elems: Vec<R4Prime32> = (0..width).map(|_| R4Prime32::random(&mut rng)).collect();

        let packed = PR4Prime32::from_fn(|i| elems[i]);
        let squared = packed.square();

        for (i, elem) in elems.iter().enumerate() {
            assert_eq!(
                squared.extract(i),
                elem.square(),
                "Prime32 packed RingSubfieldFp4 square mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn packed_ring_subfield_fp4_inverse() {
        let mut rng = StdRng::seed_from_u64(367);
        let width = <PR4 as PackedValue>::WIDTH;
        let elems: Vec<R4> = (0..width)
            .map(|_| {
                let x = R4::random(&mut rng);
                if x.is_zero() {
                    R4::one()
                } else {
                    x
                }
            })
            .collect();

        let packed = PR4::from_fn(|i| elems[i]);
        let inverted = packed.inverse().unwrap();

        for (i, elem) in elems.iter().enumerate() {
            assert_eq!(
                inverted.extract(i),
                elem.inverse().unwrap(),
                "packed RingSubfieldFp4 inverse mismatch at lane {i}"
            );
        }
    }

    #[test]
    fn packed_ring_subfield_fp4_broadcast() {
        let val = R4::new([
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
            F::from_u64(17),
        ]);
        let packed = PR4::broadcast(val);
        let width = <PR4 as PackedValue>::WIDTH;
        for i in 0..width {
            assert_eq!(packed.extract(i), val);
        }
    }

    #[test]
    fn packed_ring_subfield_fp4_pack_unpack() {
        let mut rng = StdRng::seed_from_u64(364);
        let width = <PR4 as PackedValue>::WIDTH;
        let elems: Vec<R4> = (0..width * 3).map(|_| R4::random(&mut rng)).collect();

        let packed = PR4::pack_slice(&elems);
        let unpacked = PR4::unpack_slice(&packed);

        assert_eq!(elems, unpacked);
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
