//! Scalar evaluation helpers for cyclotomic ring elements.

use super::CyclotomicRing;
use akita_field::{FieldCore, FromPrimitiveInt, MulBase};

/// Return the first `len` powers of `alpha`, starting with one.
pub fn scalar_powers<F: FieldCore>(alpha: F, len: usize) -> Vec<F> {
    let mut out = vec![F::zero(); len];
    let mut power = F::one();
    for val in out.iter_mut() {
        *val = power;
        power *= alpha;
    }
    out
}

/// Evaluate a cyclotomic ring element at the scalar `alpha`.
pub fn eval_ring_at<F: FieldCore, const D: usize>(r: &CyclotomicRing<F, D>, alpha: &F) -> F {
    let mut acc = F::zero();
    let mut power = F::one();
    for coeff in r.coefficients() {
        acc += *coeff * power;
        power *= *alpha;
    }
    acc
}

/// Evaluate a ring element against precomputed powers of `alpha`.
///
/// Ring coefficients live in `F`; the scalar powers may live in any field `E`
/// that supports multiplication by `F`. The ordinary base-field case is `E = F`.
///
/// # Panics
///
/// Panics in debug builds if `alpha_pows.len() != D`.
#[inline]
pub fn eval_ring_at_pows<F, E, const D: usize>(r: &CyclotomicRing<F, D>, alpha_pows: &[E]) -> E
where
    F: FieldCore,
    E: FieldCore + MulBase<F>,
{
    debug_assert_eq!(alpha_pows.len(), D);
    r.coefficients()
        .iter()
        .zip(alpha_pows.iter())
        .fold(E::zero(), |acc, (coeff, alpha_pow)| {
            acc + alpha_pow.mul_base(*coeff)
        })
}

/// Return the cyclotomic trace `D * coeff_0`.
pub fn trace<F: FieldCore + FromPrimitiveInt, const D: usize>(r: &CyclotomicRing<F, D>) -> F {
    F::from_u64(D as u64) * r.coefficients()[0]
}
