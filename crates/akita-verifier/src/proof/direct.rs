//! Verifier helpers for zero-fold proof payloads.

use akita_field::{AkitaError, ExtField, FieldCore};
use akita_types::{basis_weights, BasisMode, OpeningBatch, CleartextWitnessProof};

/// Check one zero-fold cleartext witness against one claimed opening.
///
/// Zero-fold cleartext witnesses are raw field-element tables. Under
/// [`BasisMode::Lagrange`] they are boolean-hypercube evaluations; under
/// [`BasisMode::Monomial`] they are multilinear coefficients.
///
/// # Errors
///
/// Returns an error if the witness length is not a power of two, does not
/// match the opening-point dimension, or is not field-element encoded.
pub fn cleartext_witness_opening_matches<F, E>(
    cleartext_witness: &CleartextWitnessProof<F>,
    opening_point: &[E],
    opening: &E,
    basis: BasisMode,
) -> Result<bool, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let witness = cleartext_witness
        .as_field_elements()
        .map(|witness| witness.coeffs())
        .ok_or(AkitaError::InvalidProof)?;
    if !witness.len().is_power_of_two() {
        return Err(AkitaError::InvalidProof);
    }
    let point_len = u32::try_from(opening_point.len()).map_err(|_| AkitaError::InvalidProof)?;
    let expected_len = 1usize
        .checked_shl(point_len)
        .ok_or(AkitaError::InvalidProof)?;
    if witness.len() != expected_len {
        return Err(AkitaError::InvalidProof);
    }
    let weights = basis_weights(opening_point, basis)?;
    let evaluation = witness
        .iter()
        .zip(weights.iter())
        .fold(E::zero(), |acc, (&coeff, &weight)| {
            acc + weight.mul_base(coeff)
        });
    Ok(evaluation == *opening)
}

/// Verify all zero-fold witness/opening claims using normalized opening_batch.
///
/// Cleartext witnesses are stored once per committed polynomial in slot order,
/// and every slot opens at the same public point.
/// # Errors
///
/// Returns an error if the opening batch summary is inconsistent with the flattened
/// witnesses/openings, routes a claim to a missing opening point, or any direct
/// witness does not match its opening.
pub(crate) fn verify_zero_fold_openings_with_opening_batch<F, E>(
    witnesses: &[CleartextWitnessProof<F>],
    opening_point: &[E],
    openings: &[E],
    opening_batch: &OpeningBatch,
    basis: BasisMode,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let num_claims = opening_batch.num_claims();
    let num_polynomials = opening_batch.num_polynomials();
    if witnesses.len() != num_polynomials
        || openings.len() != num_claims
        || opening_batch.claim_poly_indices().len() != num_claims
    {
        return Err(AkitaError::InvalidProof);
    }

    for (claim_idx, opening) in openings.iter().enumerate().take(num_claims) {
        let poly_idx = opening_batch.claim_poly_indices()[claim_idx];
        let witness = witnesses.get(poly_idx).ok_or(AkitaError::InvalidProof)?;
        if !cleartext_witness_opening_matches(witness, opening_point, opening, basis)? {
            return Err(AkitaError::InvalidProof);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp32, FpExt2, NegOneNr};
    use akita_types::FlatRingVec;

    type F = Fp32<251>;
    type E = FpExt2<F, NegOneNr>;

    #[test]
    fn root_direct_openings_accept_opening_batch() {
        let witnesses = vec![CleartextWitnessProof::FieldElements(
            FlatRingVec::from_coeffs(vec![F::from_u64(1), F::from_u64(2)]),
        )];
        let point = [E::new(F::from_u64(3), F::from_u64(4))];
        let opening = [E::new(F::from_u64(4), F::from_u64(4))];
        let opening_batch =
            OpeningBatch::same_point(1, 1).expect("valid single-point opening_batch");

        verify_zero_fold_openings_with_opening_batch(
            &witnesses,
            &point,
            &opening,
            &opening_batch,
            BasisMode::Lagrange,
        )
        .expect("extension-valued root-direct opening_batch claim should verify");
    }

    #[test]
    fn root_direct_single_point_batch_checks_each_witness() {
        let raw_poly = vec![F::from_u64(1), F::from_u64(2)];
        let witnesses = vec![
            CleartextWitnessProof::FieldElements(FlatRingVec::from_coeffs(raw_poly.clone())),
            CleartextWitnessProof::FieldElements(FlatRingVec::from_coeffs(raw_poly)),
        ];
        let point = [E::new(F::from_u64(3), F::from_u64(4))];
        let openings = [
            E::new(F::from_u64(4), F::from_u64(4)),
            E::new(F::from_u64(4), F::from_u64(4)),
        ];
        let opening_batch =
            OpeningBatch::same_point(1, 2).expect("valid single-point batch");

        verify_zero_fold_openings_with_opening_batch(
            &witnesses,
            &point,
            &openings,
            &opening_batch,
            BasisMode::Lagrange,
        )
        .expect("single-point root-direct batch should verify each witness");
    }
}
