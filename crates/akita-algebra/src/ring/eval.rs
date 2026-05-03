//! Scalar evaluation helpers for cyclotomic ring elements.

use super::CyclotomicRing;
use akita_field::FieldCore;

/// Return the first `len` powers of `alpha`, starting with one.
pub fn scalar_powers<F: FieldCore>(alpha: F, len: usize) -> Vec<F> {
    let mut out = vec![F::zero(); len];
    let mut power = F::one();
    for val in out.iter_mut() {
        *val = power;
        power = power * alpha;
    }
    out
}

/// Evaluate a cyclotomic ring element at the scalar `alpha`.
pub fn eval_ring_at<F: FieldCore, const D: usize>(r: &CyclotomicRing<F, D>, alpha: &F) -> F {
    let mut acc = F::zero();
    let mut power = F::one();
    for coeff in r.coefficients() {
        acc += *coeff * power;
        power = power * *alpha;
    }
    acc
}

/// Evaluate a cyclotomic ring element against precomputed powers of `alpha`.
///
/// # Panics
///
/// Panics in debug builds if `alpha_pows.len() != D`.
#[inline]
pub fn eval_ring_at_pows<F: FieldCore, const D: usize>(
    r: &CyclotomicRing<F, D>,
    alpha_pows: &[F],
) -> F {
    debug_assert_eq!(alpha_pows.len(), D);
    r.coefficients()
        .iter()
        .zip(alpha_pows.iter())
        .fold(F::zero(), |acc, (coeff, alpha_pow)| {
            acc + *coeff * *alpha_pow
        })
}
