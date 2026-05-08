//! Helpers for embedding base fields into extension fields.

use crate::fields::ext::{
    Fp2, Fp2Config, PowerBasisFp4, PowerBasisFp4Config, PowerBasisFp4MulBackend, RingSubfieldFp4,
    RingSubfieldFp4MulBackend, TowerBasisFp4, TowerBasisFp4Config, UnitNr,
};
use crate::{FieldCore, FromPrimitiveInt};
use akita_serialization::Valid;

/// Lift a base-field element into an extension field.
///
/// This is intentionally small: for extension towers we embed into the constant term.
pub trait LiftBase<F: FieldCore>: FieldCore {
    /// Embed `x ∈ F` as a constant in `Self`.
    fn lift_base(x: F) -> Self;
}

/// Multiply an extension-field element by a base-field scalar.
///
/// This avoids materializing the base scalar as an extension element and then
/// using a full extension multiply. For tower extensions this scales each
/// base-field coordinate directly.
pub trait MulBase<F: FieldCore>: FieldCore {
    /// Return `self * x`, where `x` is interpreted as a base-field scalar.
    fn mul_base(self, x: F) -> Self;
}

/// An algebraic extension of base field `F`.
///
/// Provides the extension degree and a constructor from a slice of base-field
/// coefficients (in the canonical basis `{1, u, u^2, ...}`).
pub trait ExtField<F: FieldCore>: FieldCore + LiftBase<F> + MulBase<F> + FromPrimitiveInt {
    /// Extension degree: `[Self : F]`.
    const EXT_DEGREE: usize;

    /// Construct from a coefficient slice `[c0, c1, ..., c_{d-1}]`.
    ///
    /// # Panics
    /// Panics if `coeffs.len() != Self::EXT_DEGREE`.
    fn from_base_slice(coeffs: &[F]) -> Self;

    /// Return base-field coefficients in the canonical basis.
    fn to_base_vec(&self) -> Vec<F>;
}

impl<F: FieldCore + FromPrimitiveInt> ExtField<F> for F {
    const EXT_DEGREE: usize = 1;

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 1);
        coeffs[0]
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        vec![*self]
    }
}

impl<F, C> ExtField<F> for Fp2<F, C>
where
    F: FieldCore + FromPrimitiveInt + Valid,
    C: Fp2Config<F>,
{
    const EXT_DEGREE: usize = 2;

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 2);
        Self::new(coeffs[0], coeffs[1])
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        vec![self.coeffs[0], self.coeffs[1]]
    }
}

impl<F, C2> ExtField<F> for TowerBasisFp4<F, C2, UnitNr>
where
    F: FieldCore + FromPrimitiveInt + Valid + PowerBasisFp4MulBackend<C2>,
    C2: Fp2Config<F>,
{
    const EXT_DEGREE: usize = 4;

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 4);
        Self::new(
            Fp2::new(coeffs[0], coeffs[2]),
            Fp2::new(coeffs[1], coeffs[3]),
        )
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        vec![
            self.coeffs[0].coeffs[0],
            self.coeffs[1].coeffs[0],
            self.coeffs[0].coeffs[1],
            self.coeffs[1].coeffs[1],
        ]
    }
}

impl<F, C2> LiftBase<Fp2<F, C2>> for TowerBasisFp4<F, C2, UnitNr>
where
    F: FieldCore + Valid + PowerBasisFp4MulBackend<C2>,
    C2: Fp2Config<F>,
{
    #[inline]
    fn lift_base(x: Fp2<F, C2>) -> Self {
        Self::new(x, Fp2::zero())
    }
}

impl<F, C2> MulBase<Fp2<F, C2>> for TowerBasisFp4<F, C2, UnitNr>
where
    F: FieldCore + Valid + PowerBasisFp4MulBackend<C2>,
    C2: Fp2Config<F>,
{
    #[inline]
    fn mul_base(self, x: Fp2<F, C2>) -> Self {
        Self::new(self.coeffs[0] * x, self.coeffs[1] * x)
    }
}

impl<F, C2> ExtField<Fp2<F, C2>> for TowerBasisFp4<F, C2, UnitNr>
where
    F: FieldCore + FromPrimitiveInt + Valid + PowerBasisFp4MulBackend<C2>,
    C2: Fp2Config<F>,
{
    const EXT_DEGREE: usize = 2;

    #[inline]
    fn from_base_slice(coeffs: &[Fp2<F, C2>]) -> Self {
        assert_eq!(coeffs.len(), 2);
        Self::new(coeffs[0], coeffs[1])
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<Fp2<F, C2>> {
        vec![self.coeffs[0], self.coeffs[1]]
    }
}

impl<F, C> ExtField<F> for PowerBasisFp4<F, C>
where
    F: FieldCore + FromPrimitiveInt + Valid + PowerBasisFp4MulBackend<C>,
    C: PowerBasisFp4Config<F>,
{
    const EXT_DEGREE: usize = 4;

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 4);
        Self::new([coeffs[0], coeffs[1], coeffs[2], coeffs[3]])
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        self.coeffs.to_vec()
    }
}

impl<F> ExtField<F> for RingSubfieldFp4<F>
where
    F: FieldCore + FromPrimitiveInt + Valid + RingSubfieldFp4MulBackend,
{
    const EXT_DEGREE: usize = 4;

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 4);
        Self::new([coeffs[0], coeffs[1], coeffs[2], coeffs[3]])
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        self.coeffs.to_vec()
    }
}

impl<F: FieldCore> LiftBase<F> for F {
    #[inline]
    fn lift_base(x: F) -> Self {
        x
    }
}

impl<F: FieldCore> MulBase<F> for F {
    #[inline]
    fn mul_base(self, x: F) -> Self {
        self * x
    }
}

impl<F, C> LiftBase<F> for Fp2<F, C>
where
    F: FieldCore + Valid,
    C: Fp2Config<F>,
{
    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new(x, F::zero())
    }
}

impl<F, C> MulBase<F> for Fp2<F, C>
where
    F: FieldCore + Valid,
    C: Fp2Config<F>,
{
    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(self.coeffs[0] * x, self.coeffs[1] * x)
    }
}

impl<F, C2, C4> LiftBase<F> for TowerBasisFp4<F, C2, C4>
where
    F: FieldCore + Valid + PowerBasisFp4MulBackend<C2>,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
{
    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new(Fp2::new(x, F::zero()), Fp2::new(F::zero(), F::zero()))
    }
}

impl<F, C2, C4> MulBase<F> for TowerBasisFp4<F, C2, C4>
where
    F: FieldCore + Valid + PowerBasisFp4MulBackend<C2>,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
{
    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(self.coeffs[0].mul_base(x), self.coeffs[1].mul_base(x))
    }
}

impl<F, C> LiftBase<F> for PowerBasisFp4<F, C>
where
    F: FieldCore + Valid + PowerBasisFp4MulBackend<C>,
    C: PowerBasisFp4Config<F>,
{
    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new([x, F::zero(), F::zero(), F::zero()])
    }
}

impl<F, C> MulBase<F> for PowerBasisFp4<F, C>
where
    F: FieldCore + Valid + PowerBasisFp4MulBackend<C>,
    C: PowerBasisFp4Config<F>,
{
    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] * x))
    }
}

impl<F> LiftBase<F> for RingSubfieldFp4<F>
where
    F: FieldCore + Valid + RingSubfieldFp4MulBackend,
{
    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new([x, F::zero(), F::zero(), F::zero()])
    }
}

impl<F> MulBase<F> for RingSubfieldFp4<F>
where
    F: FieldCore + Valid + RingSubfieldFp4MulBackend,
{
    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] * x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp32, NegOneNr, UnitNr};

    type F = Fp32<251>;
    type E2 = Fp2<F, NegOneNr>;
    type E4 = TowerBasisFp4<F, NegOneNr, UnitNr>;

    #[test]
    fn mul_base_matches_full_multiply_for_base_field() {
        let x = F::from_u64(7);
        let scalar = F::from_u64(11);

        assert_eq!(x.mul_base(scalar), x * scalar);
    }

    #[test]
    fn mul_base_matches_full_multiply_for_fp2() {
        let x = E2::new(F::from_u64(3), F::from_u64(5));
        let scalar = F::from_u64(11);

        assert_eq!(x.mul_base(scalar), x * E2::lift_base(scalar));
    }

    #[test]
    fn mul_base_matches_full_multiply_for_fp4() {
        let x = E4::new(
            E2::new(F::from_u64(3), F::from_u64(5)),
            E2::new(F::from_u64(7), F::from_u64(13)),
        );
        let scalar = F::from_u64(11);

        assert_eq!(x.mul_base(scalar), x * E4::lift_base(scalar));
    }

    #[test]
    fn fp4_mul_base_over_fp2_matches_full_multiply() {
        let x = E4::new(
            E2::new(F::from_u64(3), F::from_u64(5)),
            E2::new(F::from_u64(7), F::from_u64(13)),
        );
        let scalar = E2::new(F::from_u64(11), F::from_u64(17));

        assert_eq!(
            <E4 as MulBase<E2>>::mul_base(x, scalar),
            x * <E4 as LiftBase<E2>>::lift_base(scalar)
        );
    }

    #[test]
    fn fp4_lift_over_fp2_agrees_with_lift_over_base() {
        let scalar = F::from_u64(7);
        let via_base = <E4 as LiftBase<F>>::lift_base(scalar);
        let via_tower = <E4 as LiftBase<E2>>::lift_base(<E2 as LiftBase<F>>::lift_base(scalar));

        assert_eq!(via_base, via_tower);
    }

    #[test]
    fn fp4_ext_over_fp2_round_trips_through_base_slice() {
        let x = E4::new(
            E2::new(F::from_u64(3), F::from_u64(5)),
            E2::new(F::from_u64(7), F::from_u64(13)),
        );
        let coeffs = <E4 as ExtField<E2>>::to_base_vec(&x);
        let rebuilt = <E4 as ExtField<E2>>::from_base_slice(&coeffs);

        assert_eq!(rebuilt, x);
        assert_eq!(<E4 as ExtField<E2>>::EXT_DEGREE, 2);
    }

    /// `ExtField<F>::EXT_DEGREE` over the base prime field must equal
    /// `ExtField<ClaimField>::EXT_DEGREE * ExtField<F>::EXT_DEGREE` on the
    /// claim field. This is the chain
    /// `[ChallengeField : F] = [ChallengeField : ClaimField] * [ClaimField : F]`
    /// the field-role convention relies on.
    #[test]
    fn fp4_ext_degrees_chain_correctly() {
        assert_eq!(
            <E4 as ExtField<F>>::EXT_DEGREE,
            <E4 as ExtField<E2>>::EXT_DEGREE * <E2 as ExtField<F>>::EXT_DEGREE
        );
    }
}
