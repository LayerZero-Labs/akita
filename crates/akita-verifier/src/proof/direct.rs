//! Verifier helpers for zero-fold proof payloads.

use akita_field::{AkitaError, ExtField, FieldCore};
use akita_types::{basis_weights, BasisMode, ClaimIncidenceSummary, CleartextWitnessProof};

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

/// Verify all zero-fold witness/opening claims using normalized incidence.
///
/// Cleartext witnesses are stored once per committed polynomial in group order,
/// while openings are stored once per claim. Multipoint openings of the same
/// committed polynomial therefore reuse the same witness through the
/// claim's `(group_idx, poly_idx)` route.
///
/// This is the zero-fold counterpart to incidence-driven schedule lookup:
/// claim-to-point routing comes from [`ClaimIncidenceSummary`] rather than the
/// temporary legacy batch-shape adapter.
///
/// # Errors
///
/// Returns an error if the incidence summary is inconsistent with the flattened
/// witnesses/openings, routes a claim to a missing opening point, or any direct
/// witness does not match its opening.
pub(crate) fn verify_zero_fold_openings_with_incidence<F, E>(
    witnesses: &[CleartextWitnessProof<F>],
    opening_points: &[&[E]],
    openings: &[E],
    incidence_summary: &ClaimIncidenceSummary,
    basis: BasisMode,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let num_claims = incidence_summary.num_claims();
    let num_polynomials = incidence_summary.num_polynomials();
    if witnesses.len() != num_polynomials
        || openings.len() != num_claims
        || incidence_summary.claim_to_point().len() != num_claims
        || incidence_summary.claim_poly_indices().len() != num_claims
    {
        return Err(AkitaError::InvalidProof);
    }

    let mut point_offsets = Vec::with_capacity(incidence_summary.num_polys_per_point().len());
    let mut next_offset = 0usize;
    for &polys_at_point in incidence_summary.num_polys_per_point() {
        point_offsets.push(next_offset);
        next_offset = next_offset
            .checked_add(polys_at_point)
            .ok_or(AkitaError::InvalidProof)?;
    }
    if next_offset != witnesses.len() {
        return Err(AkitaError::InvalidProof);
    }

    for (claim_idx, opening) in openings.iter().enumerate().take(num_claims) {
        let point_idx = incidence_summary.claim_to_point()[claim_idx];
        if point_idx >= opening_points.len() {
            return Err(AkitaError::InvalidProof);
        }
        let poly_idx = incidence_summary.claim_poly_indices()[claim_idx];
        let point_offset = *point_offsets
            .get(point_idx)
            .ok_or(AkitaError::InvalidProof)?;
        let witness_idx = point_offset
            .checked_add(poly_idx)
            .ok_or(AkitaError::InvalidProof)?;
        let witness = witnesses.get(witness_idx).ok_or(AkitaError::InvalidProof)?;
        let opening_point = opening_points[point_idx];
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
    fn cleartext_witness_opening_matches_extension_claim() {
        let witness = CleartextWitnessProof::FieldElements(FlatRingVec::from_coeffs(vec![
            F::from_u64(1),
            F::from_u64(2),
        ]));
        let point = [E::new(F::from_u64(3), F::from_u64(4))];
        let opening = E::new(F::from_u64(4), F::from_u64(4));

        assert!(
            cleartext_witness_opening_matches(&witness, &point, &opening, BasisMode::Lagrange)
                .expect("extension-valued direct opening should verify")
        );
    }

    #[test]
    fn root_direct_openings_accept_incidence_summary() {
        let witnesses = vec![CleartextWitnessProof::FieldElements(
            FlatRingVec::from_coeffs(vec![F::from_u64(1), F::from_u64(2)]),
        )];
        let point = [E::new(F::from_u64(3), F::from_u64(4))];
        let opening = [E::new(F::from_u64(4), F::from_u64(4))];
        let incidence_summary =
            ClaimIncidenceSummary::same_point(1, 1).expect("valid single-point incidence");

        verify_zero_fold_openings_with_incidence(
            &witnesses,
            &[&point[..]],
            &opening,
            &incidence_summary,
            BasisMode::Lagrange,
        )
        .expect("extension-valued root-direct incidence claim should verify");
    }

    #[test]
    fn root_direct_multipoint_each_point_has_its_own_witness() {
        // One-commitment-per-point: each point cites its own commitment and
        // contributes its own witness, even when the polynomial is identical.
        let raw_poly = vec![F::from_u64(1), F::from_u64(2)];
        let witnesses = vec![
            CleartextWitnessProof::FieldElements(FlatRingVec::from_coeffs(raw_poly.clone())),
            CleartextWitnessProof::FieldElements(FlatRingVec::from_coeffs(raw_poly)),
        ];
        let point_a = [E::new(F::from_u64(3), F::from_u64(4))];
        let point_b = [E::new(F::from_u64(5), F::from_u64(6))];
        let openings = [
            E::new(F::from_u64(4), F::from_u64(4)),
            E::new(F::from_u64(6), F::from_u64(6)),
        ];
        let incidence_summary = ClaimIncidenceSummary::from_point_polys(1, vec![1, 1])
            .expect("valid multipoint incidence");

        verify_zero_fold_openings_with_incidence(
            &witnesses,
            &[&point_a[..], &point_b[..]],
            &openings,
            &incidence_summary,
            BasisMode::Lagrange,
        )
        .expect("multipoint root-direct incidence should verify with per-point witnesses");
    }
}
