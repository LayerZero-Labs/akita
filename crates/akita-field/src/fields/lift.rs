//! Helpers for embedding base fields into extension fields.

use crate::fields::ext::{Fp2, Fp2Config, Fp4, Fp4Config};
use crate::{FieldCore, FromPrimitiveInt};
use akita_serialization::Valid;

/// Lift a base-field element into an extension field.
///
/// This is intentionally small: for extension towers we embed into the constant term.
pub trait LiftBase<F: FieldCore>: FieldCore {
    /// Embed `x ∈ F` as a constant in `Self`.
    fn lift_base(x: F) -> Self;
}

/// An algebraic extension of base field `F`.
///
/// Provides the extension degree and a constructor from a slice of base-field
/// coefficients (in the canonical basis `{1, u, u^2, ...}`).
pub trait ExtField<F: FieldCore>: FieldCore + LiftBase<F> + FromPrimitiveInt {
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
        vec![self.c0, self.c1]
    }
}

impl<F, C2, C4> ExtField<F> for Fp4<F, C2, C4>
where
    F: FieldCore + FromPrimitiveInt + Valid,
    C2: Fp2Config<F>,
    C4: Fp4Config<F, C2>,
{
    const EXT_DEGREE: usize = 4;

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 4);
        Self::new(
            Fp2::new(coeffs[0], coeffs[1]),
            Fp2::new(coeffs[2], coeffs[3]),
        )
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        vec![self.c0.c0, self.c0.c1, self.c1.c0, self.c1.c1]
    }
}

impl<F: FieldCore> LiftBase<F> for F {
    #[inline]
    fn lift_base(x: F) -> Self {
        x
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

impl<F, C2, C4> LiftBase<F> for Fp4<F, C2, C4>
where
    F: FieldCore + Valid,
    C2: Fp2Config<F>,
    C4: Fp4Config<F, C2>,
{
    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new(Fp2::new(x, F::zero()), Fp2::new(F::zero(), F::zero()))
    }
}
