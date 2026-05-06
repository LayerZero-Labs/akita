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
        Self::new(self.c0 * x, self.c1 * x)
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

impl<F, C2, C4> MulBase<F> for Fp4<F, C2, C4>
where
    F: FieldCore + Valid,
    C2: Fp2Config<F>,
    C4: Fp4Config<F, C2>,
{
    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(self.c0.mul_base(x), self.c1.mul_base(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp32, NegOneNr, UnitNr};

    type F = Fp32<251>;
    type E2 = Fp2<F, NegOneNr>;
    type E4 = Fp4<F, NegOneNr, UnitNr>;

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
}
