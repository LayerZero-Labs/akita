//! Helpers for embedding base fields into extension fields.

use crate::algebra::fields::ext::{Fp2, Fp2Config, Fp4, Fp4Config};
use crate::primitives::serialization::Valid;
use crate::FieldCore;

/// Lift a base-field element into an extension field.
///
/// This is intentionally small: for extension towers we embed into the constant term.
pub trait LiftBase<F: FieldCore>: FieldCore {
    /// Embed `x ∈ F` as a constant in `Self`.
    fn lift_base(x: F) -> Self;
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
