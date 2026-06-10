//! Packed extension field types using transpose-based packing.
//!
//! A `PackedFpExt2` stores `[PF; 2]` where `PF` is the packed base field.
//! Each `PF` lane contains the corresponding coefficient of an `FpExt2` element.
//! This enables WIDTH-fold parallel arithmetic over `FpExt2` using existing SIMD
//! base-field operations.

use crate::ext::{
    FpExt2, FpExt2Config, PowerBasisFpExt4, PowerBasisFpExt4Config, PowerBasisFpExt4MulBackend,
    RingSubfieldFpExt4, RingSubfieldFpExt4MulBackend, RingSubfieldFpExt8,
    RingSubfieldFpExt8MulBackend, TowerBasisFpExt4, TowerBasisFpExt4Config,
};
use crate::packed::{HasPacking, PackedField, PackedValue};
use crate::{FieldCore, Invertible};
use akita_serialization::Valid;
use core::ops::{Add, Mul, Sub};

/// Packed `FpExt2` elements stored in transpose layout: `[PF; 2]`.
///
/// If `PF` has width `W`, this represents `W` parallel `FpExt2` values.
pub struct PackedFpExt2<F: FieldCore, C: FpExt2Config<F>, PF: PackedField<Scalar = F>> {
    /// Degree-0 coefficient (packed across SIMD lanes).
    pub c0: PF,
    /// Degree-1 coefficient (packed across SIMD lanes).
    pub c1: PF,
    _marker: std::marker::PhantomData<fn() -> (F, C)>,
}

impl<F: FieldCore, C: FpExt2Config<F>, PF: PackedField<Scalar = F>> Clone
    for PackedFpExt2<F, C, PF>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldCore, C: FpExt2Config<F>, PF: PackedField<Scalar = F>> Copy
    for PackedFpExt2<F, C, PF>
{
}

impl<F: FieldCore, C: FpExt2Config<F>, PF: PackedField<Scalar = F>> std::fmt::Debug
    for PackedFpExt2<F, C, PF>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedFpExt2").finish_non_exhaustive()
    }
}

impl<F: FieldCore, C: FpExt2Config<F>, PF: PackedField<Scalar = F>> PackedFpExt2<F, C, PF> {
    /// Create a `PackedFpExt2` from its two packed coefficients.
    #[inline]
    pub fn new(c0: PF, c1: PF) -> Self {
        Self {
            c0,
            c1,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F, C, PF> PackedValue for PackedFpExt2<F, C, PF>
where
    F: FieldCore + Valid + 'static,
    C: FpExt2Config<F> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Value = FpExt2<F, C>;
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
        FpExt2::new(self.c0.extract(lane), self.c1.extract(lane))
    }
}

impl<F, C, PF> Add for PackedFpExt2<F, C, PF>
where
    F: FieldCore,
    C: FpExt2Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.c0 + rhs.c0, self.c1 + rhs.c1)
    }
}

impl<F, C, PF> Sub for PackedFpExt2<F, C, PF>
where
    F: FieldCore,
    C: FpExt2Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.c0 - rhs.c0, self.c1 - rhs.c1)
    }
}

impl<F, C, PF> Mul for PackedFpExt2<F, C, PF>
where
    F: FieldCore,
    C: FpExt2Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let (c0, c1) = PF::fp_ext2_mul::<C>(self.c0, self.c1, rhs.c0, rhs.c1);
        Self::new(c0, c1)
    }
}

impl<F, C, PF> PackedField for PackedFpExt2<F, C, PF>
where
    F: FieldCore + Valid + 'static,
    C: FpExt2Config<F> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Scalar = FpExt2<F, C>;

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

impl<F, C> HasPacking for FpExt2<F, C>
where
    F: FieldCore + Valid + HasPacking + 'static,
    C: FpExt2Config<F> + 'static,
{
    type Packing = PackedFpExt2<F, C, F::Packing>;
}

/// Packed `TowerBasisFpExt4` elements stored in transpose layout: `[PackedFpExt2; 2]`.
pub struct PackedTowerBasisFpExt4<
    F: FieldCore,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
    PF: PackedField<Scalar = F>,
> {
    /// Packed tower-basis coefficients `[b0, b1]`.
    pub coeffs: [PackedFpExt2<F, C2, PF>; 2],
    _marker: std::marker::PhantomData<fn() -> C4>,
}

impl<F, C2, C4, PF> Clone for PackedTowerBasisFpExt4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, C2, C4, PF> Copy for PackedTowerBasisFpExt4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
}

impl<F, C2, C4, PF> std::fmt::Debug for PackedTowerBasisFpExt4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedTowerBasisFpExt4")
            .finish_non_exhaustive()
    }
}

impl<F, C2, C4, PF> PackedTowerBasisFpExt4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    /// Create a `PackedTowerBasisFpExt4` from its two `PackedFpExt2` halves.
    #[inline]
    pub fn new(c0: PackedFpExt2<F, C2, PF>, c1: PackedFpExt2<F, C2, PF>) -> Self {
        Self {
            coeffs: [c0, c1],
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F, C2, C4, PF> PackedValue for PackedTowerBasisFpExt4<F, C2, C4, PF>
where
    F: FieldCore + Valid + 'static,
    C2: FpExt2Config<F> + 'static,
    C4: TowerBasisFpExt4Config<F, C2> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Value = TowerBasisFpExt4<F, C2, C4>;
    const WIDTH: usize = PF::WIDTH;

    fn from_fn<G>(mut f: G) -> Self
    where
        G: FnMut(usize) -> Self::Value,
    {
        let mut c0s: Vec<FpExt2<F, C2>> = Vec::with_capacity(PF::WIDTH);
        let mut c1s: Vec<FpExt2<F, C2>> = Vec::with_capacity(PF::WIDTH);
        for i in 0..PF::WIDTH {
            let val = f(i);
            c0s.push(val.coeffs[0]);
            c1s.push(val.coeffs[1]);
        }
        Self::new(
            PackedFpExt2::from_fn(|i| c0s[i]),
            PackedFpExt2::from_fn(|i| c1s[i]),
        )
    }

    fn extract(&self, lane: usize) -> Self::Value {
        TowerBasisFpExt4::new(self.coeffs[0].extract(lane), self.coeffs[1].extract(lane))
    }
}

impl<F, C2, C4, PF> Add for PackedTowerBasisFpExt4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
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

impl<F, C2, C4, PF> Sub for PackedTowerBasisFpExt4<F, C2, C4, PF>
where
    F: FieldCore,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
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

impl<F, C2, C4, PF> Mul for PackedTowerBasisFpExt4<F, C2, C4, PF>
where
    F: FieldCore + Valid + 'static,
    C2: FpExt2Config<F> + 'static,
    C4: TowerBasisFpExt4Config<F, C2>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let [c0, c1, c2, c3] = PF::tower_basis_fp_ext4_mul::<C2, C4>(
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
        Self::new(PackedFpExt2::new(c0, c2), PackedFpExt2::new(c1, c3))
    }
}

impl<F, C2, C4, PF> PackedField for PackedTowerBasisFpExt4<F, C2, C4, PF>
where
    F: FieldCore + Valid + PowerBasisFpExt4MulBackend<C2> + 'static,
    C2: FpExt2Config<F> + 'static,
    C4: TowerBasisFpExt4Config<F, C2> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Scalar = TowerBasisFpExt4<F, C2, C4>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(
            PackedFpExt2::broadcast(value.coeffs[0]),
            PackedFpExt2::broadcast(value.coeffs[1]),
        )
    }

    #[inline(always)]
    fn square(self) -> Self {
        let [c0, c1, c2, c3] = PF::tower_basis_fp_ext4_mul::<C2, C4>(
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
        Self::new(PackedFpExt2::new(c0, c2), PackedFpExt2::new(c1, c3))
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
            PackedFpExt2::new(C2::mul_non_residue(v1.c1, PF::broadcast), v1.c0)
        } else {
            PackedFpExt2::broadcast(nr) * v1
        };
        let inv_norm = (v0 - nr_v1).inverse()?;
        let zero = PackedFpExt2::broadcast(FpExt2::zero());
        Some(Self::new(
            self.coeffs[0] * inv_norm,
            (zero - self.coeffs[1]) * inv_norm,
        ))
    }
}

impl<F, C2, C4> HasPacking for TowerBasisFpExt4<F, C2, C4>
where
    F: FieldCore + Valid + HasPacking + PowerBasisFpExt4MulBackend<C2> + 'static,
    C2: FpExt2Config<F> + 'static,
    C4: TowerBasisFpExt4Config<F, C2> + 'static,
{
    type Packing = PackedTowerBasisFpExt4<F, C2, C4, F::Packing>;
}

/// Packed `PowerBasisFpExt4` elements stored as `[PF; 4]`.
pub struct PackedPowerBasisFpExt4<
    F: FieldCore,
    C: PowerBasisFpExt4Config<F>,
    PF: PackedField<Scalar = F>,
> {
    /// Packed coefficients in power-basis order.
    pub coeffs: [PF; 4],
    _marker: std::marker::PhantomData<fn() -> (F, C)>,
}

impl<F, C, PF> Clone for PackedPowerBasisFpExt4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFpExt4Config<F>,
    PF: PackedField<Scalar = F>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, C, PF> Copy for PackedPowerBasisFpExt4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFpExt4Config<F>,
    PF: PackedField<Scalar = F>,
{
}

impl<F, C, PF> std::fmt::Debug for PackedPowerBasisFpExt4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFpExt4Config<F>,
    PF: PackedField<Scalar = F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedPowerBasisFpExt4")
            .finish_non_exhaustive()
    }
}

impl<F, C, PF> PackedPowerBasisFpExt4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFpExt4Config<F>,
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

impl<F, C, PF> PackedValue for PackedPowerBasisFpExt4<F, C, PF>
where
    F: FieldCore + Valid + 'static,
    C: PowerBasisFpExt4Config<F> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Value = PowerBasisFpExt4<F, C>;
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
        PowerBasisFpExt4::new(std::array::from_fn(|j| self.coeffs[j].extract(lane)))
    }
}

impl<F, C, PF> Add for PackedPowerBasisFpExt4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFpExt4Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}

impl<F, C, PF> Sub for PackedPowerBasisFpExt4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFpExt4Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}

impl<F, C, PF> Mul for PackedPowerBasisFpExt4<F, C, PF>
where
    F: FieldCore,
    C: PowerBasisFpExt4Config<F>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(PF::power_basis_fp_ext4_mul::<C>(self.coeffs, rhs.coeffs))
    }
}

impl<F, C, PF> PackedField for PackedPowerBasisFpExt4<F, C, PF>
where
    F: FieldCore + Valid + PowerBasisFpExt4MulBackend<C> + 'static,
    C: PowerBasisFpExt4Config<F> + 'static,
    PF: PackedField<Scalar = F>,
{
    type Scalar = PowerBasisFpExt4<F, C>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(std::array::from_fn(|i| PF::broadcast(value.coeffs[i])))
    }

    #[inline(always)]
    fn square(self) -> Self {
        Self::new(PF::power_basis_fp_ext4_mul::<C>(self.coeffs, self.coeffs))
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

impl<F, C> HasPacking for PowerBasisFpExt4<F, C>
where
    F: FieldCore + Valid + HasPacking + PowerBasisFpExt4MulBackend<C> + 'static,
    C: PowerBasisFpExt4Config<F> + 'static,
{
    type Packing = PackedPowerBasisFpExt4<F, C, F::Packing>;
}

/// Packed `RingSubfieldFpExt4` elements stored as `[PF; 4]`.
pub struct PackedRingSubfieldFpExt4<F: FieldCore, PF: PackedField<Scalar = F>> {
    /// Packed coefficients in `[1, e1, e2, e3]` order.
    pub coeffs: [PF; 4],
    _marker: std::marker::PhantomData<fn() -> F>,
}

impl<F, PF> Clone for PackedRingSubfieldFpExt4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, PF> Copy for PackedRingSubfieldFpExt4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
}

impl<F, PF> std::fmt::Debug for PackedRingSubfieldFpExt4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedRingSubfieldFpExt4")
            .finish_non_exhaustive()
    }
}

impl<F, PF> PackedRingSubfieldFpExt4<F, PF>
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
        Self::new(PF::ring_subfield_fp_ext4_square(self.coeffs))
    }
}

impl<F, PF> PackedValue for PackedRingSubfieldFpExt4<F, PF>
where
    F: FieldCore + Valid + 'static,
    PF: PackedField<Scalar = F>,
{
    type Value = RingSubfieldFpExt4<F>;
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
        RingSubfieldFpExt4::new(std::array::from_fn(|j| self.coeffs[j].extract(lane)))
    }
}

impl<F, PF> Add for PackedRingSubfieldFpExt4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let [a0, a1, a2, a3] = self.coeffs;
        let [b0, b1, b2, b3] = rhs.coeffs;
        Self::new([a0 + b0, a1 + b1, a2 + b2, a3 + b3])
    }
}

impl<F, PF> Sub for PackedRingSubfieldFpExt4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let [a0, a1, a2, a3] = self.coeffs;
        let [b0, b1, b2, b3] = rhs.coeffs;
        Self::new([a0 - b0, a1 - b1, a2 - b2, a3 - b3])
    }
}

impl<F, PF> Mul for PackedRingSubfieldFpExt4<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(PF::ring_subfield_fp_ext4_mul(self.coeffs, rhs.coeffs))
    }
}

impl<F, PF> PackedField for PackedRingSubfieldFpExt4<F, PF>
where
    F: FieldCore + Valid + RingSubfieldFpExt4MulBackend + 'static,
    PF: PackedField<Scalar = F>,
{
    type Scalar = RingSubfieldFpExt4<F>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(std::array::from_fn(|i| PF::broadcast(value.coeffs[i])))
    }

    #[inline(always)]
    fn square(self) -> Self {
        Self::new(PF::ring_subfield_fp_ext4_square(self.coeffs))
    }

    #[inline(always)]
    fn inverse(self) -> Option<Self>
    where
        Self::Scalar: Invertible,
    {
        Some(Self::new(PF::ring_subfield_fp_ext4_inverse(self.coeffs)?))
    }
}

impl<F> HasPacking for RingSubfieldFpExt4<F>
where
    F: FieldCore + Valid + HasPacking + RingSubfieldFpExt4MulBackend + 'static,
{
    type Packing = PackedRingSubfieldFpExt4<F, F::Packing>;
}

/// Packed `RingSubfieldFpExt8` elements stored in transpose layout: `[PF; 8]`.
///
/// Each `PF` lane contains one coefficient of a degree-8 Chebyshev-basis element.
pub struct PackedRingSubfieldFpExt8<F: FieldCore, PF: PackedField<Scalar = F>> {
    /// Packed coefficients in `[1, e1, ..., e7]` order.
    pub coeffs: [PF; 8],
    _marker: std::marker::PhantomData<fn() -> F>,
}

impl<F, PF> Clone for PackedRingSubfieldFpExt8<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, PF> Copy for PackedRingSubfieldFpExt8<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
}

impl<F, PF> std::fmt::Debug for PackedRingSubfieldFpExt8<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedRingSubfieldFpExt8")
            .finish_non_exhaustive()
    }
}

impl<F, PF> PackedRingSubfieldFpExt8<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    /// Create a packed value from packed ring-subfield coefficients.
    #[inline]
    pub fn new(coeffs: [PF; 8]) -> Self {
        Self {
            coeffs,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F, PF> PackedValue for PackedRingSubfieldFpExt8<F, PF>
where
    F: FieldCore + Valid + 'static,
    PF: PackedField<Scalar = F>,
{
    type Value = RingSubfieldFpExt8<F>;
    const WIDTH: usize = PF::WIDTH;

    fn from_fn<G>(mut f: G) -> Self
    where
        G: FnMut(usize) -> Self::Value,
    {
        let mut coeffs: [Vec<F>; 8] = std::array::from_fn(|_| Vec::with_capacity(PF::WIDTH));
        for i in 0..PF::WIDTH {
            let val = f(i);
            for (j, coeff) in val.coeffs.into_iter().enumerate() {
                coeffs[j].push(coeff);
            }
        }
        Self::new(std::array::from_fn(|j| PF::from_fn(|i| coeffs[j][i])))
    }

    fn extract(&self, lane: usize) -> Self::Value {
        RingSubfieldFpExt8::new(std::array::from_fn(|j| self.coeffs[j].extract(lane)))
    }
}

impl<F, PF> Add for PackedRingSubfieldFpExt8<F, PF>
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

impl<F, PF> Sub for PackedRingSubfieldFpExt8<F, PF>
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

impl<F, PF> Mul for PackedRingSubfieldFpExt8<F, PF>
where
    F: FieldCore,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(PF::ring_subfield_fp_ext8_mul(self.coeffs, rhs.coeffs))
    }
}

impl<F, PF> PackedField for PackedRingSubfieldFpExt8<F, PF>
where
    F: FieldCore + Valid + RingSubfieldFpExt8MulBackend + 'static,
    PF: PackedField<Scalar = F>,
{
    type Scalar = RingSubfieldFpExt8<F>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::new(std::array::from_fn(|i| PF::broadcast(value.coeffs[i])))
    }

    #[inline(always)]
    fn square(self) -> Self {
        Self::new(PF::ring_subfield_fp_ext8_square(self.coeffs))
    }

    #[inline(always)]
    fn inverse(self) -> Option<Self>
    where
        Self::Scalar: Invertible,
    {
        // FpExt8 inversion uses Gaussian elimination — delegate lane by lane.
        let mut coeffs: [Vec<F>; 8] = std::array::from_fn(|_| Vec::with_capacity(PF::WIDTH));
        for lane in 0..PF::WIDTH {
            let scalar = self.extract(lane);
            let inv = scalar.inverse()?;
            for (j, c) in inv.coeffs.into_iter().enumerate() {
                coeffs[j].push(c);
            }
        }
        Some(Self::new(std::array::from_fn(|j| {
            PF::from_fn(|i| coeffs[j][i])
        })))
    }
}

impl<F> HasPacking for RingSubfieldFpExt8<F>
where
    F: FieldCore + Valid + HasPacking + RingSubfieldFpExt8MulBackend + 'static,
{
    type Packing = PackedRingSubfieldFpExt8<F, F::Packing>;
}

#[cfg(test)]
mod tests;
