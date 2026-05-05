//! Verifier helpers for root-direct proof payloads.

use akita_field::{AkitaError, FieldCore};
use akita_types::{
    basis_weights, checked_total_claims, BasisMode, DirectWitnessProof, MultiPointBatchShape,
};

/// Borrow the field-element payload from a direct witness.
///
/// # Errors
///
/// Returns an error if the witness is not encoded as field elements.
pub fn direct_witness_field_elements<F: FieldCore>(
    direct_witness: &DirectWitnessProof<F>,
) -> Result<&[F], AkitaError> {
    direct_witness
        .as_field_elements()
        .map(|witness| witness.coeffs())
        .ok_or(AkitaError::InvalidProof)
}

/// Check one root-direct witness against one claimed opening.
///
/// Root-direct witnesses are raw field-element tables. Under
/// [`BasisMode::Lagrange`] they are boolean-hypercube evaluations; under
/// [`BasisMode::Monomial`] they are multilinear coefficients.
///
/// # Errors
///
/// Returns an error if the witness length is not a power of two, does not
/// match the opening-point dimension, or is not field-element encoded.
pub fn direct_witness_opening_matches<F: FieldCore>(
    direct_witness: &DirectWitnessProof<F>,
    opening_point: &[F],
    opening: &F,
    basis: BasisMode,
) -> Result<bool, AkitaError> {
    let witness = direct_witness_field_elements(direct_witness)?;
    if !witness.len().is_power_of_two() {
        return Err(AkitaError::InvalidProof);
    }
    let expected_len = 1usize
        .checked_shl(opening_point.len() as u32)
        .ok_or(AkitaError::InvalidProof)?;
    if witness.len() != expected_len {
        return Err(AkitaError::InvalidProof);
    }
    let weights = basis_weights(opening_point, basis);
    let evaluation = witness
        .iter()
        .zip(weights.iter())
        .fold(F::zero(), |acc, (&coeff, &weight)| acc + coeff * weight);
    Ok(evaluation == *opening)
}

/// Verify all root-direct witness/opening claims in a flattened batch.
///
/// Commitment recomputation is intentionally left to the scheme crate until
/// commitment generation is split away from prover-only setup machinery.
///
/// # Errors
///
/// Returns an error if the batch shape is inconsistent, a claim routes to a
/// missing opening point, or any direct witness does not match its opening.
pub fn verify_root_direct_openings<F: FieldCore>(
    witnesses: &[DirectWitnessProof<F>],
    opening_points: &[&[F]],
    openings: &[F],
    batch_shape: &MultiPointBatchShape,
    basis: BasisMode,
) -> Result<(), AkitaError> {
    let num_claims = checked_total_claims(&batch_shape.claim_group_sizes, "batched_verify")
        .map_err(|_| AkitaError::InvalidProof)?;
    if witnesses.len() != num_claims
        || openings.len() != num_claims
        || batch_shape.claim_to_point.len() != num_claims
    {
        return Err(AkitaError::InvalidProof);
    }

    for (claim_idx, witness) in witnesses.iter().enumerate() {
        let point_idx = batch_shape.claim_to_point[claim_idx];
        if point_idx >= opening_points.len() {
            return Err(AkitaError::InvalidProof);
        }
        let opening_point = opening_points[point_idx];
        if !direct_witness_opening_matches(witness, opening_point, &openings[claim_idx], basis)? {
            return Err(AkitaError::InvalidProof);
        }
    }

    Ok(())
}
