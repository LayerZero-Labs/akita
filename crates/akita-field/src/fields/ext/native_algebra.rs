//! Native `num_traits`/`std` supertrait impls and core-algebra markers for the
//! extension field types (`FpExt2`, `TowerBasisFpExt4`, `PowerBasisFpExt4`,
//! `RingSubfieldFpExt4`, `RingSubfieldFpExt8`).
//!
//! These are the Jolt-free supertrait obligations of the native
//! [`AdditiveGroup`]/[`FieldCore`] hierarchy. The non-trivial `RingCore::square`
//! / `Invertible::inverse` impls stay co-located with each extension type.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::{Product, Sum};

use akita_serialization::Valid;
use num_traits::{One, Zero};

use super::{
    FpExt2, FpExt2Config, PowerBasisFpExt4, PowerBasisFpExt4Config, PowerBasisFpExt4MulBackend,
    RingSubfieldFpExt4, RingSubfieldFpExt4MulBackend, RingSubfieldFpExt8,
    RingSubfieldFpExt8MulBackend, TowerBasisFpExt4, TowerBasisFpExt4Config,
};
use crate::{AdditiveGroup, FieldCore};

// --- FpExt2 -----------------------------------------------------------------

impl<F: FieldCore, C: FpExt2Config<F>> Zero for FpExt2<F, C> {
    #[inline]
    fn zero() -> Self {
        Self::new(F::zero(), F::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coeffs[0].is_zero() && self.coeffs[1].is_zero()
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> One for FpExt2<F, C> {
    #[inline]
    fn one() -> Self {
        Self::new(F::one(), F::zero())
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> fmt::Display for FpExt2<F, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.coeffs[0], self.coeffs[1])
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> Hash for FpExt2<F, C> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coeffs[0].hash(state);
        self.coeffs[1].hash(state);
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> Sum for FpExt2<F, C> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, F: FieldCore, C: FpExt2Config<F>> Sum<&'a Self> for FpExt2<F, C> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> Product for FpExt2<F, C> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, F: FieldCore, C: FpExt2Config<F>> Product<&'a Self> for FpExt2<F, C> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * *x)
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> AdditiveGroup for FpExt2<F, C> {}
impl<F: FieldCore + Valid, C: FpExt2Config<F>> FieldCore for FpExt2<F, C> {}

// --- TowerBasisFpExt4 -------------------------------------------------------

impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Zero
    for TowerBasisFpExt4<F, C2, C4>
{
    #[inline]
    fn zero() -> Self {
        Self::new(FpExt2::zero(), FpExt2::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coeffs[0].is_zero() && self.coeffs[1].is_zero()
    }
}

impl<F, C2, C4> One for TowerBasisFpExt4<F, C2, C4>
where
    F: FieldCore + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
{
    #[inline]
    fn one() -> Self {
        Self::new(FpExt2::one(), FpExt2::zero())
    }
}

impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> fmt::Display
    for TowerBasisFpExt4<F, C2, C4>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.coeffs[0], self.coeffs[1])
    }
}

impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Hash
    for TowerBasisFpExt4<F, C2, C4>
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coeffs[0].hash(state);
        self.coeffs[1].hash(state);
    }
}

impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Sum
    for TowerBasisFpExt4<F, C2, C4>
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Sum<&'a Self>
    for TowerBasisFpExt4<F, C2, C4>
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}

impl<F, C2, C4> Product for TowerBasisFpExt4<F, C2, C4>
where
    F: FieldCore + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, F, C2, C4> Product<&'a Self> for TowerBasisFpExt4<F, C2, C4>
where
    F: FieldCore + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
{
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * *x)
    }
}

impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> AdditiveGroup
    for TowerBasisFpExt4<F, C2, C4>
{
}
impl<F, C2, C4> FieldCore for TowerBasisFpExt4<F, C2, C4>
where
    F: FieldCore + Valid + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
{
}

// --- PowerBasisFpExt4 -------------------------------------------------------

impl<F: FieldCore, C: PowerBasisFpExt4Config<F>> Zero for PowerBasisFpExt4<F, C> {
    #[inline]
    fn zero() -> Self {
        Self::new([F::zero(), F::zero(), F::zero(), F::zero()])
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|coeff| coeff.is_zero())
    }
}

impl<F: FieldCore + PowerBasisFpExt4MulBackend<C>, C: PowerBasisFpExt4Config<F>> One
    for PowerBasisFpExt4<F, C>
{
    #[inline]
    fn one() -> Self {
        Self::new([F::one(), F::zero(), F::zero(), F::zero()])
    }
}

impl<F: FieldCore, C: PowerBasisFpExt4Config<F>> fmt::Display for PowerBasisFpExt4<F, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, {}, {}, {})",
            self.coeffs[0], self.coeffs[1], self.coeffs[2], self.coeffs[3]
        )
    }
}

impl<F: FieldCore, C: PowerBasisFpExt4Config<F>> Hash for PowerBasisFpExt4<F, C> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coeffs.hash(state);
    }
}

impl<F: FieldCore, C: PowerBasisFpExt4Config<F>> Sum for PowerBasisFpExt4<F, C> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, F: FieldCore, C: PowerBasisFpExt4Config<F>> Sum<&'a Self> for PowerBasisFpExt4<F, C> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}

impl<F: FieldCore + PowerBasisFpExt4MulBackend<C>, C: PowerBasisFpExt4Config<F>> Product
    for PowerBasisFpExt4<F, C>
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, F: FieldCore + PowerBasisFpExt4MulBackend<C>, C: PowerBasisFpExt4Config<F>>
    Product<&'a Self> for PowerBasisFpExt4<F, C>
{
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * *x)
    }
}

impl<F: FieldCore, C: PowerBasisFpExt4Config<F>> AdditiveGroup for PowerBasisFpExt4<F, C> {}
impl<F, C> FieldCore for PowerBasisFpExt4<F, C>
where
    F: FieldCore + Valid + PowerBasisFpExt4MulBackend<C>,
    C: PowerBasisFpExt4Config<F>,
{
}

// --- RingSubfieldFpExt4 -----------------------------------------------------

impl<F: FieldCore> Zero for RingSubfieldFpExt4<F> {
    #[inline]
    fn zero() -> Self {
        Self::new([F::zero(), F::zero(), F::zero(), F::zero()])
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|coeff| coeff.is_zero())
    }
}

impl<F: FieldCore + RingSubfieldFpExt4MulBackend> One for RingSubfieldFpExt4<F> {
    #[inline]
    fn one() -> Self {
        Self::new([F::one(), F::zero(), F::zero(), F::zero()])
    }
}

impl<F: FieldCore> fmt::Display for RingSubfieldFpExt4<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, {}, {}, {})",
            self.coeffs[0], self.coeffs[1], self.coeffs[2], self.coeffs[3]
        )
    }
}

impl<F: FieldCore> Hash for RingSubfieldFpExt4<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coeffs.hash(state);
    }
}

impl<F: FieldCore> Sum for RingSubfieldFpExt4<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, F: FieldCore> Sum<&'a Self> for RingSubfieldFpExt4<F> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}

impl<F: FieldCore + RingSubfieldFpExt4MulBackend> Product for RingSubfieldFpExt4<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, F: FieldCore + RingSubfieldFpExt4MulBackend> Product<&'a Self> for RingSubfieldFpExt4<F> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * *x)
    }
}

impl<F: FieldCore + RingSubfieldFpExt4MulBackend> AdditiveGroup for RingSubfieldFpExt4<F> {}
impl<F: FieldCore + Valid + RingSubfieldFpExt4MulBackend> FieldCore for RingSubfieldFpExt4<F> {}

// --- RingSubfieldFpExt8 -----------------------------------------------------

impl<F: FieldCore> Zero for RingSubfieldFpExt8<F> {
    #[inline]
    fn zero() -> Self {
        Self::new([
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
        ])
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|coeff| coeff.is_zero())
    }
}

impl<F: FieldCore + RingSubfieldFpExt8MulBackend> One for RingSubfieldFpExt8<F> {
    #[inline]
    fn one() -> Self {
        Self::new([
            F::one(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
        ])
    }
}

impl<F: FieldCore> fmt::Display for RingSubfieldFpExt8<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, {}, {}, {}, {}, {}, {}, {})",
            self.coeffs[0],
            self.coeffs[1],
            self.coeffs[2],
            self.coeffs[3],
            self.coeffs[4],
            self.coeffs[5],
            self.coeffs[6],
            self.coeffs[7]
        )
    }
}

impl<F: FieldCore> Hash for RingSubfieldFpExt8<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coeffs.hash(state);
    }
}

impl<F: FieldCore> Sum for RingSubfieldFpExt8<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, F: FieldCore> Sum<&'a Self> for RingSubfieldFpExt8<F> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}

impl<F: FieldCore + RingSubfieldFpExt8MulBackend> Product for RingSubfieldFpExt8<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, F: FieldCore + RingSubfieldFpExt8MulBackend> Product<&'a Self> for RingSubfieldFpExt8<F> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * *x)
    }
}

impl<F: FieldCore + RingSubfieldFpExt8MulBackend> AdditiveGroup for RingSubfieldFpExt8<F> {}
impl<F: FieldCore + Valid + RingSubfieldFpExt8MulBackend> FieldCore for RingSubfieldFpExt8<F> {}
