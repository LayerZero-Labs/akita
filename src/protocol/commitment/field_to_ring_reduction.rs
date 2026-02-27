//! Temporary helper for Hachi §3.1 conversion (k = 1).
//!
//! This implements Steps 3–5 from `task.md` for the k=1 (base-field) case:
//! - Pack coefficient blocks via `psi` (here: coefficient embedding).
//! - Pack the monomial vector for the inner variables.
//! - Compute `y_ring` and the trace identity check.
//!
//! We adopt LSB-first indexing (the same order used by `DenseMultilinearEvals`):
//! the lowest index bits correspond to the *first* variables.

#![allow(dead_code, clippy::type_complexity)]

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::protocol::opening_point::RingOpeningPoint;
use crate::{CanonicalField, FieldCore};

/// Reduce coefficient blocks into ring elements.
///
/// Note: this implementation assumes `k = 1` (base field).
///
/// - `coeffs` are evaluations on `{0,1}^n`, indexed in LSB-first order.
/// - The lowest `alpha = log2(D)` bits are packed into one ring element via
///   coefficient embedding (the k=1 case of `psi`).
/// - The output is a flat table of length `2^(num_vars - alpha)` representing
///   the ring polynomial's evaluation table over the outer variables.
pub(crate) fn reduce_coeffs_to_ring_elements<F: FieldCore, const D: usize>(
    num_vars: usize,
    coeffs: &[F],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    if D == 0 || !D.is_power_of_two() {
        return Err(HachiError::InvalidInput(format!(
            "ring degree D={D} is not a power of two"
        )));
    }
    let alpha = D.trailing_zeros() as usize;
    if num_vars < alpha {
        return Err(HachiError::InvalidInput(format!(
            "num_vars {num_vars} is smaller than alpha {alpha}"
        )));
    }

    let expected_len = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| HachiError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
    if coeffs.len() != expected_len {
        return Err(HachiError::InvalidSize {
            expected: expected_len,
            actual: coeffs.len(),
        });
    }

    let outer_vars = num_vars - alpha;
    let outer_len = 1usize
        .checked_shl(outer_vars as u32)
        .ok_or_else(|| HachiError::InvalidInput(format!("2^{outer_vars} does not fit usize")))?;

    let mut out = Vec::with_capacity(outer_len);
    for i in 0..outer_len {
        let coeffs = std::array::from_fn(|j| {
            let idx = i + (j << outer_vars);
            coeffs[idx]
        });
        out.push(CyclotomicRing::from_coefficients(coeffs));
    }
    Ok(out)
}

/// Build the Lagrange basis weights `(χ_j(point))_j` in LSB-first order.
pub(crate) fn lagrange_weights<F: FieldCore>(point: &[F]) -> Vec<F> {
    let len = 1usize << point.len();
    let mut weights = vec![F::zero(); len];
    multilinear_lagrange_basis(&mut weights, point);
    weights
}

/// Convert a field point into a ring opening point `(a, b)` using constant embedding.
pub(crate) fn ring_opening_point_from_field<F: FieldCore, const D: usize>(
    opening_point: &[F],
    r_vars: usize,
    m_vars: usize,
) -> Result<RingOpeningPoint<F, D>, HachiError> {
    let expected_len = r_vars
        .checked_add(m_vars)
        .ok_or_else(|| HachiError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() != expected_len {
        return Err(HachiError::InvalidPointDimension {
            expected: expected_len,
            actual: opening_point.len(),
        });
    }

    let b = lagrange_vector_from_field::<F, D>(&opening_point[..r_vars]);
    let a = lagrange_vector_from_field::<F, D>(&opening_point[r_vars..]);
    Ok(RingOpeningPoint { a, b })
}

fn lagrange_vector_from_field<F: FieldCore, const D: usize>(
    point: &[F],
) -> Vec<CyclotomicRing<F, D>> {
    lagrange_weights(point)
        .into_iter()
        .map(constant_ring::<F, D>)
        .collect()
}

fn constant_ring<F: FieldCore, const D: usize>(value: F) -> CyclotomicRing<F, D> {
    let mut coeffs = [F::zero(); D];
    coeffs[0] = value;
    CyclotomicRing::from_coefficients(coeffs)
}

/// Reduce inner openings (Lagrange vector) into a ring element.
///
/// Note: this implementation assumes `k = 1` (base field).
pub(crate) fn reduce_inner_openings_to_ring_elements<F: FieldCore, const D: usize>(
    inner_point: &[F],
) -> Result<CyclotomicRing<F, D>, HachiError> {
    let weights = lagrange_weights(inner_point);
    if weights.len() != D {
        return Err(HachiError::InvalidInput(format!(
            "inner basis length {} does not match D={D}",
            weights.len()
        )));
    }
    let coeffs = std::array::from_fn(|i| weights[i]);
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

/// Evaluate the packed ring polynomial `F` at the outer point (Lagrange basis).
pub(crate) fn evaluate_packed_ring_poly<F: FieldCore, const D: usize>(
    packed_coeffs: &[CyclotomicRing<F, D>],
    outer_point: &[F],
) -> CyclotomicRing<F, D> {
    let weights = lagrange_weights(outer_point);
    debug_assert_eq!(weights.len(), packed_coeffs.len());
    packed_coeffs
        .iter()
        .zip(weights.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, w_i)| {
            acc + f_i.scale(w_i)
        })
}

/// Trace map for k=1: `Tr_H(u) = d * ct(u)` for `R_q = F_q[X]/(X^d+1)`.
pub(crate) fn trace<F: CanonicalField, const D: usize>(u: &CyclotomicRing<F, D>) -> F {
    let d = F::from_u64(D as u64);
    u.coefficients()[0] * d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{D, F};

    fn eval_mle_from_coeffs(coeffs: &[F], point: &[F]) -> F {
        let weights = super::lagrange_weights(point);
        assert_eq!(weights.len(), coeffs.len());
        coeffs
            .iter()
            .zip(weights.iter())
            .fold(F::zero(), |acc, (c, w)| acc + (*c * *w))
    }

    #[test]
    fn k1_reduction_satisfies_trace_identity() {
        let num_vars = 8;
        let total_len = 1usize << num_vars;
        let coeffs: Vec<F> = (0..total_len)
            .map(|i| F::from_u64((i as u64) + 1))
            .collect();
        let point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i as u64) + 2)).collect();

        let claimed_y = eval_mle_from_coeffs(&coeffs, &point);

        let alpha = D.trailing_zeros() as usize;
        let outer_vars = num_vars - alpha;
        let ring_coeffs = reduce_coeffs_to_ring_elements::<F, D>(num_vars, &coeffs).unwrap();
        assert_eq!(ring_coeffs.len(), 1usize << outer_vars);

        let outer_point = &point[..outer_vars];
        let inner_point = &point[outer_vars..];
        let y_ring = evaluate_packed_ring_poly::<F, D>(&ring_coeffs, outer_point);
        let v = reduce_inner_openings_to_ring_elements::<F, D>(inner_point).unwrap();

        let trace_lhs = trace::<F, D>(&(y_ring * v.sigma_m1()));
        let trace_rhs = F::from_u64(D as u64) * claimed_y;
        assert_eq!(trace_lhs, trace_rhs);
    }
}
