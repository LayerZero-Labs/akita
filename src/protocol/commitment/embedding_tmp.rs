//! Temporary helper for Hachi §3.1 conversion (k = 1).
//!
//! This implements Steps 3–5 from `task.md` for the k=1 (base-field) case:
//! - Pack coefficient blocks via `psi` (here: coefficient embedding).
//! - Pack the monomial vector for the inner variables.
//! - Compute `Y` and the trace identity check.
//!
//! We adopt LSB-first indexing (the same order used by `DenseMultilinearEvals`):
//! the lowest index bits correspond to the *first* variables.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::{CanonicalField, FieldCore};

/// Pack a multilinear coefficient table into ring coefficients (k = 1).
///
/// - `coeffs` are the monomial-basis coefficients, indexed in LSB-first order.
/// - The lowest `alpha = log2(D)` bits are packed into one ring element via
///   coefficient embedding (the k=1 case of `psi`).
/// - The output is a flat table of length `2^(num_vars - alpha)` representing
///   the ring polynomial's coefficient table (monomial basis).
pub fn pack_mle_evals_to_ring<F: FieldCore, const D: usize>(
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
    for chunk in coeffs.chunks_exact(D) {
        let coeffs = std::array::from_fn(|i| chunk[i]);
        out.push(CyclotomicRing::from_coefficients(coeffs));
    }
    debug_assert_eq!(out.len(), outer_len);
    Ok(out)
}

/// Build the monomial vector `(∏ x_t^{j_t})_j` in LSB-first order.
fn monomial_weights<F: FieldCore>(point: &[F]) -> Vec<F> {
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

/// Pack the inner-variable monomials into a single ring element (k = 1).
fn pack_inner_monomials_k1<F: FieldCore, const D: usize>(
    inner_point: &[F],
) -> Result<CyclotomicRing<F, D>, HachiError> {
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
fn evaluate_packed_ring_poly_k1<F: FieldCore, const D: usize>(
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

/// End-to-end §3.1 reduction for k=1 (base field), excluding the final PCS proof.
///
/// Returns:
/// - `packed_coeffs` = `(F_i)_i` packed into `R_q`
/// - `v` = packed monomial vector for inner variables
/// - `y_ring` = `Y` (ring element)
/// - `trace_lhs` / `trace_rhs` = sides of the trace identity check
pub fn reduce_mle_claim_k1<F: CanonicalField, const D: usize>(
    num_vars: usize,
    coeffs: &[F],
    point: &[F],
    claimed_y: F,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        CyclotomicRing<F, D>,
        CyclotomicRing<F, D>,
        F,
        F,
    ),
    HachiError,
> {
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

    let packed_coeffs = pack_mle_evals_to_ring::<F, D>(num_vars, coeffs)?;

    let inner_point = &point[..alpha];
    let outer_point = &point[alpha..];

    let v = pack_inner_monomials_k1::<F, D>(inner_point)?;
    let y_ring = evaluate_packed_ring_poly_k1::<F, D>(&packed_coeffs, outer_point);

    let trace_lhs = trace_k1::<F, D>(&(y_ring * v.sigma_m1()));
    let trace_rhs = F::from_u64(D as u64) * claimed_y;

    Ok((packed_coeffs, v, y_ring, trace_lhs, trace_rhs))
}

/// Same as `reduce_mle_claim_k1`, but enforces the trace identity.
pub fn reduce_mle_claim_k1_checked<F: CanonicalField, const D: usize>(
    num_vars: usize,
    coeffs: &[F],
    point: &[F],
    claimed_y: F,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        CyclotomicRing<F, D>,
        CyclotomicRing<F, D>,
    ),
    HachiError,
> {
    let (packed_coeffs, v, y_ring, trace_lhs, trace_rhs) =
        reduce_mle_claim_k1::<F, D>(num_vars, coeffs, point, claimed_y)?;
    if trace_lhs != trace_rhs {
        return Err(HachiError::InvalidProof);
    }
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

        let (packed, _v, _y_ring, trace_lhs, trace_rhs) =
            reduce_mle_claim_k1::<F, D>(num_vars, &coeffs, &point, claimed_y).unwrap();

        let alpha = D.trailing_zeros() as usize;
        assert_eq!(packed.len(), 1usize << (num_vars - alpha));
        assert_eq!(trace_lhs, trace_rhs);

        // The checked variant should pass for a valid claim.
        assert!(reduce_mle_claim_k1_checked::<F, D>(num_vars, &coeffs, &point, claimed_y).is_ok());
    }
}
