//! Verifier helpers for root-direct proof payloads.

use akita_field::{AkitaError, ExtField, FieldCore};
use akita_types::{basis_weights, BasisMode, ClaimIncidenceSummary, DirectWitnessProof};

/// Borrow the field-element payload from a direct witness.
///
/// # Errors
///
/// Returns an error if the witness is not encoded as field elements.
pub(crate) fn direct_witness_field_elements<F: FieldCore>(
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
pub fn direct_witness_opening_matches<F, E>(
    direct_witness: &DirectWitnessProof<F>,
    opening_point: &[E],
    opening: &E,
    basis: BasisMode,
) -> Result<bool, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
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
        .fold(E::zero(), |acc, (&coeff, &weight)| {
            acc + weight.mul_base(coeff)
        });
    Ok(evaluation == *opening)
}

/// Verify all root-direct witness/opening claims using normalized incidence.
///
/// Direct witnesses are stored once per committed polynomial in group order,
/// while openings are stored once per claim. Multipoint openings of the same
/// committed polynomial therefore reuse the same witness through the
/// claim's `(group_idx, poly_idx)` route.
///
/// This is the direct-root counterpart to incidence-driven schedule lookup:
/// claim-to-point routing comes from [`ClaimIncidenceSummary`] rather than the
/// temporary legacy batch-shape adapter.
///
/// # Errors
///
/// Returns an error if the incidence summary is inconsistent with the flattened
/// witnesses/openings, routes a claim to a missing opening point, or any direct
/// witness does not match its opening.
pub(crate) fn verify_root_direct_openings_with_incidence<F, E>(
    witnesses: &[DirectWitnessProof<F>],
    opening_points: &[&[E]],
    openings: &[E],
    incidence_summary: &ClaimIncidenceSummary,
    basis: BasisMode,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let num_claims = incidence_summary.num_claims;
    let num_polynomials = incidence_summary
        .num_polynomials()
        .map_err(|_| AkitaError::InvalidProof)?;
    if witnesses.len() != num_polynomials
        || openings.len() != num_claims
        || incidence_summary.claim_to_point.len() != num_claims
        || incidence_summary.claim_to_group.len() != num_claims
        || incidence_summary.claim_poly_indices.len() != num_claims
    {
        return Err(AkitaError::InvalidProof);
    }

    let mut group_offsets = Vec::with_capacity(incidence_summary.group_poly_counts.len());
    let mut next_offset = 0usize;
    for &group_size in &incidence_summary.group_poly_counts {
        group_offsets.push(next_offset);
        next_offset = next_offset
            .checked_add(group_size)
            .ok_or(AkitaError::InvalidProof)?;
    }
    if next_offset != witnesses.len() {
        return Err(AkitaError::InvalidProof);
    }

    for (claim_idx, opening) in openings.iter().enumerate().take(num_claims) {
        let point_idx = incidence_summary.claim_to_point[claim_idx];
        if point_idx >= opening_points.len() {
            return Err(AkitaError::InvalidProof);
        }
        let group_idx = incidence_summary.claim_to_group[claim_idx];
        let poly_idx = incidence_summary.claim_poly_indices[claim_idx];
        let group_offset = *group_offsets
            .get(group_idx)
            .ok_or(AkitaError::InvalidProof)?;
        let witness_idx = group_offset
            .checked_add(poly_idx)
            .ok_or(AkitaError::InvalidProof)?;
        let witness = witnesses.get(witness_idx).ok_or(AkitaError::InvalidProof)?;
        let opening_point = opening_points[point_idx];
        if !direct_witness_opening_matches(witness, opening_point, opening, basis)? {
            return Err(AkitaError::InvalidProof);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp2, Fp32, NegOneNr};
    use akita_types::{
        ClaimIncidence, ClaimIncidenceLimits, CommitmentGroupOccurrence, FlatRingVec,
        IncidenceClaim,
    };

    type F = Fp32<251>;
    type E = Fp2<F, NegOneNr>;

    #[test]
    fn direct_witness_opening_matches_extension_claim() {
        let witness = DirectWitnessProof::FieldElements(FlatRingVec::from_coeffs(vec![
            F::from_u64(1),
            F::from_u64(2),
        ]));
        let point = [E::new(F::from_u64(3), F::from_u64(4))];
        let opening = E::new(F::from_u64(4), F::from_u64(4));

        assert!(
            direct_witness_opening_matches(&witness, &point, &opening, BasisMode::Lagrange)
                .expect("extension-valued direct opening should verify")
        );
    }

    #[test]
    fn root_direct_openings_accept_incidence_summary() {
        let witnesses = vec![DirectWitnessProof::FieldElements(FlatRingVec::from_coeffs(
            vec![F::from_u64(1), F::from_u64(2)],
        ))];
        let point = [E::new(F::from_u64(3), F::from_u64(4))];
        let opening = [E::new(F::from_u64(4), F::from_u64(4))];
        let incidence_summary =
            ClaimIncidenceSummary::same_point(1, 1).expect("valid single-point incidence");

        verify_root_direct_openings_with_incidence(
            &witnesses,
            &[&point[..]],
            &opening,
            &incidence_summary,
            BasisMode::Lagrange,
        )
        .expect("extension-valued root-direct incidence claim should verify");
    }

    #[test]
    fn root_direct_multipoint_reuses_group_poly_witness() {
        let witnesses = vec![DirectWitnessProof::FieldElements(FlatRingVec::from_coeffs(
            vec![F::from_u64(1), F::from_u64(2)],
        ))];
        let point_a = [E::new(F::from_u64(3), F::from_u64(4))];
        let point_b = [E::new(F::from_u64(5), F::from_u64(6))];
        let openings = [
            E::new(F::from_u64(4), F::from_u64(4)),
            E::new(F::from_u64(6), F::from_u64(6)),
        ];
        let commitment = ();
        let incidence = ClaimIncidence {
            points: vec![&point_a[..], &point_b[..]],
            groups: vec![CommitmentGroupOccurrence {
                commitment: &commitment,
                poly_count: 1,
            }],
            claims: vec![
                IncidenceClaim {
                    point_idx: 0,
                    group_idx: 0,
                    poly_idx: 0,
                    claimed_eval: openings[0],
                },
                IncidenceClaim {
                    point_idx: 1,
                    group_idx: 0,
                    poly_idx: 0,
                    claimed_eval: openings[1],
                },
            ],
        };
        let incidence_summary = incidence
            .validate(ClaimIncidenceLimits {
                max_num_vars: 1,
                max_num_points: 2,
                max_num_claims: 2,
            })
            .expect("valid multipoint incidence");

        verify_root_direct_openings_with_incidence(
            &witnesses,
            &[&point_a[..], &point_b[..]],
            &openings,
            &incidence_summary,
            BasisMode::Lagrange,
        )
        .expect("multipoint root-direct incidence should reuse the same witness");
    }
}
