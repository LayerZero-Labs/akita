//! Temporary helper for Hachi §3.1 conversion (k = 1).
//!
//! This implements Steps 3–5 from `task.md` for the k=1 (base-field) case:
//! - Pack coefficient blocks via `psi` (here: coefficient embedding).
//! - Pack the monomial vector for the inner variables.
//! - Compute `Y` and the trace identity check.
//!
//! We adopt LSB-first indexing (the same order used by `DenseMultilinearEvals`):
//! the lowest index bits correspond to the *first* variables.

#![allow(dead_code, clippy::type_complexity)]

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::opening_point::RingOpeningPoint;
use crate::{CanonicalField, FieldCore};

type ReduceClaimOutput<F, const D: usize> = (
    Vec<CyclotomicRing<F, D>>,
    CyclotomicRing<F, D>,
    CyclotomicRing<F, D>,
    F,
    F,
);

type ReduceClaimCheckedOutput<F, const D: usize> = (
    Vec<CyclotomicRing<F, D>>,
    CyclotomicRing<F, D>,
    CyclotomicRing<F, D>,
);

/// Reduce coefficient blocks into ring elements (k = 1).
///
/// - `coeffs` are the monomial-basis coefficients, indexed in LSB-first order.
/// - The lowest `alpha = log2(D)` bits are packed into one ring element via
///   coefficient embedding (the k=1 case of `psi`).
/// - The output is a flat table of length `2^(num_vars - alpha)` representing
///   the ring polynomial's coefficient table (monomial basis).
pub(crate) fn reduce_coeffs_to_ring_elements<F: FieldCore, const D: usize>(
    num_vars: usize,
    coeffs: &[F],
    k: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    assert_eq!(k, 1, "only k=1 is implemented");
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
    for chunk in coeffs.chunks_exact(D) {
        let coeffs = std::array::from_fn(|i| chunk[i]);
        out.push(CyclotomicRing::from_coefficients(coeffs));
    }
    debug_assert_eq!(out.len(), outer_len);
    Ok(out)
}

/// Build the monomial vector `(∏ x_t^{j_t})_j` in LSB-first order.
pub(crate) fn monomial_weights<F: FieldCore>(point: &[F]) -> Vec<F> {
    let mut weights = vec![F::one()];
    for x in point.iter().copied() {
        let prev_len = weights.len();
        weights.resize(prev_len * 2, F::zero());
        for i in 0..prev_len {
            let w = weights[i];
            weights[i + prev_len] = w * x;
        }
    }
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

    let b = monomial_vector_from_field::<F, D>(&opening_point[..r_vars]);
    let a = monomial_vector_from_field::<F, D>(&opening_point[r_vars..]);
    Ok(RingOpeningPoint { a, b })
}

fn monomial_vector_from_field<F: FieldCore, const D: usize>(
    point: &[F],
) -> Vec<CyclotomicRing<F, D>> {
    monomial_weights(point)
        .into_iter()
        .map(constant_ring::<F, D>)
        .collect()
}

fn constant_ring<F: FieldCore, const D: usize>(value: F) -> CyclotomicRing<F, D> {
    let mut coeffs = [F::zero(); D];
    coeffs[0] = value;
    CyclotomicRing::from_coefficients(coeffs)
}

/// Reduce inner openings (monomial vector) into a ring element (k = 1).
fn reduce_inner_openings_to_ring_elements<F: FieldCore, const D: usize>(
    inner_point: &[F],
    k: usize,
) -> Result<CyclotomicRing<F, D>, HachiError> {
    assert_eq!(k, 1, "only k=1 is implemented");
    let weights = monomial_weights(inner_point);
    if weights.len() != D {
        return Err(HachiError::InvalidInput(format!(
            "inner monomial length {} does not match D={D}",
            weights.len()
        )));
    }
    let coeffs = std::array::from_fn(|i| weights[i]);
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

/// Evaluate the packed ring polynomial `F` at the outer point (monomial basis).
fn evaluate_packed_ring_poly<F: FieldCore, const D: usize>(
    packed_coeffs: &[CyclotomicRing<F, D>],
    outer_point: &[F],
) -> CyclotomicRing<F, D> {
    let weights = monomial_weights(outer_point);
    debug_assert_eq!(weights.len(), packed_coeffs.len());
    packed_coeffs
        .iter()
        .zip(weights.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, w_i)| {
            acc + f_i.scale(w_i)
        })
}

/// Trace map for k=1: `Tr_H(u) = d * ct(u)` for `R_q = F_q[X]/(X^d+1)`.
fn trace_k1<F: CanonicalField, const D: usize>(u: &CyclotomicRing<F, D>) -> F {
    let d = F::from_u64(D as u64);
    u.coefficients()[0] * d
}

/// Verify the trace identity: `Tr_H(Y · σ_{-1}(v)) = (d/k) · y`.
fn verify_trace_identity<F: CanonicalField, const D: usize>(
    y_ring: &CyclotomicRing<F, D>,
    inner_point: &[F],
    claimed_y: F,
    k: usize,
) -> Result<(), HachiError> {
    assert_eq!(k, 1, "only k=1 is implemented");
    let v = reduce_inner_openings_to_ring_elements::<F, D>(inner_point, k)?;
    let trace_lhs = trace_k1::<F, D>(&((*y_ring) * v.sigma_m1()));
    let trace_rhs = F::from_u64(D as u64) * claimed_y;
    if trace_lhs != trace_rhs {
        return Err(HachiError::InvalidProof);
    }
    Ok(())
}

/// End-to-end §3.1 reduction for k=1 (base field), excluding the final PCS proof.
///
/// Returns:
/// - `packed_coeffs` = `(F_i)_i` packed into `R_q`
/// - `v` = packed monomial vector for inner variables
/// - `y_ring` = `Y` (ring element)
/// - `trace_lhs` / `trace_rhs` = sides of the trace identity check
fn reduce_mle_claim<F: CanonicalField, const D: usize>(
    num_vars: usize,
    coeffs: &[F],
    point: &[F],
    claimed_y: F,
    k: usize,
) -> Result<ReduceClaimOutput<F, D>, HachiError> {
    assert_eq!(k, 1, "only k=1 is implemented");
    if point.len() != num_vars {
        return Err(HachiError::InvalidPointDimension {
            expected: num_vars,
            actual: point.len(),
        });
    }
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

    let packed_coeffs = reduce_coeffs_to_ring_elements::<F, D>(num_vars, coeffs, k)?;

    let inner_point = &point[..alpha];
    let outer_point = &point[alpha..];

    let v = reduce_inner_openings_to_ring_elements::<F, D>(inner_point, k)?;
    let y_ring = evaluate_packed_ring_poly::<F, D>(&packed_coeffs, outer_point);

    let trace_lhs = trace_k1::<F, D>(&(y_ring * v.sigma_m1()));
    let trace_rhs = F::from_u64(D as u64) * claimed_y;

    Ok((packed_coeffs, v, y_ring, trace_lhs, trace_rhs))
}

/// Same as `reduce_mle_claim`, but enforces the trace identity.
fn reduce_mle_claim_checked<F: CanonicalField, const D: usize>(
    num_vars: usize,
    coeffs: &[F],
    point: &[F],
    claimed_y: F,
    k: usize,
) -> Result<ReduceClaimCheckedOutput<F, D>, HachiError> {
    let (packed_coeffs, v, y_ring, _trace_lhs, _trace_rhs) =
        reduce_mle_claim::<F, D>(num_vars, coeffs, point, claimed_y, k)?;
    let alpha = D.trailing_zeros() as usize;
    let inner_point = &point[..alpha];
    verify_trace_identity::<F, D>(&y_ring, inner_point, claimed_y, k)?;
    Ok((packed_coeffs, v, y_ring))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{D, F};

    fn eval_mle_from_coeffs(coeffs: &[F], point: &[F]) -> F {
        let weights = super::monomial_weights(point);
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

        let k = 1;
        let (packed, _v, _y_ring, trace_lhs, trace_rhs) =
            reduce_mle_claim::<F, D>(num_vars, &coeffs, &point, claimed_y, k).unwrap();

        let alpha = D.trailing_zeros() as usize;
        assert_eq!(packed.len(), 1usize << (num_vars - alpha));
        assert_eq!(trace_lhs, trace_rhs);

        // The checked variant should pass for a valid claim.
        assert!(reduce_mle_claim_checked::<F, D>(num_vars, &coeffs, &point, claimed_y, k).is_ok());
    }
}
