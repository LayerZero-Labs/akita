//! Verifier claim normalization helpers.

use akita_field::{AkitaError, FieldCore};
use akita_types::{
    validate_batched_inputs, AkitaExpandedSetup, ClaimIncidenceSummary, VerifierClaims,
};

/// Flattened and validated verifier claims.
pub struct PreparedVerifierClaims<'a, E: FieldCore, C> {
    /// Distinct opening points in caller order.
    pub opening_points: Vec<&'a [E]>,
    /// The single commitment over the entire polynomial bundle.
    pub commitment: C,
    /// Claimed openings flattened by opening point in claim order.
    pub openings: Vec<E>,
    /// Normalized incidence summary that owns canonical root claim routing.
    pub incidence_summary: ClaimIncidenceSummary,
}

/// Validate and flatten verifier claims into the canonical batch layout.
///
/// # Errors
///
/// Returns an error if the claims are empty, exceed setup capacity, use
/// inconsistent opening-point dimensions, contain empty per-point lists,
/// reference out-of-range polynomial indices, or overflow flattened claim
/// counts.
pub fn prepare_verifier_claims<'a, F, E, C>(
    setup: &AkitaExpandedSetup<F>,
    claims: &VerifierClaims<'a, E, C>,
) -> Result<PreparedVerifierClaims<'a, E, C>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
    C: Clone,
{
    validate_batched_inputs(
        setup,
        &claims.points,
        |c| c.point,
        |c| c.openings.len(),
        false,
    )?;

    if claims.points.is_empty() {
        return Err(AkitaError::InvalidProof);
    }
    let num_vars = claims.points[0].point.len();

    // Determine the global polynomial-index space from the maximum referenced
    // index. The verifier doesn't know the prover's `committed_polys` length
    // directly, so use the smallest size that covers every referenced index.
    let mut num_polys = 0usize;
    for claim in &claims.points {
        if claim.poly_indices.len() != claim.openings.len() {
            return Err(AkitaError::InvalidProof);
        }
        for &idx in &claim.poly_indices {
            num_polys = num_polys.max(idx.checked_add(1).ok_or(AkitaError::InvalidProof)?);
        }
    }
    if num_polys == 0 {
        return Err(AkitaError::InvalidProof);
    }

    let per_point_refs: Vec<&[usize]> = claims
        .points
        .iter()
        .map(|c| c.poly_indices.as_slice())
        .collect();
    let incidence_summary =
        ClaimIncidenceSummary::from_per_point_polys(num_vars, num_polys, &per_point_refs)
            .map_err(|_| AkitaError::InvalidProof)?;

    let openings: Vec<E> = claims
        .points
        .iter()
        .flat_map(|c| c.openings.iter().copied())
        .collect();

    Ok(PreparedVerifierClaims {
        opening_points: claims.points.iter().map(|c| c.point).collect(),
        commitment: claims.commitment.clone(),
        openings,
        incidence_summary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp2, Fp32, NegOneNr};
    use akita_types::{AkitaSetupSeed, FlatMatrix, PointClaim};

    type F = Fp32<251>;
    type E = Fp2<F, NegOneNr>;

    fn setup() -> AkitaExpandedSetup<F> {
        AkitaExpandedSetup {
            seed: AkitaSetupSeed {
                max_num_vars: 3,
                max_num_batched_polys: 4,
                max_num_points: 2,
                max_stride: 1,
                public_matrix_seed: [0u8; 32],
            },
            shared_matrix: FlatMatrix::from_flat_data(vec![F::zero()], 1),
        }
    }

    #[test]
    fn verifier_claim_preparation_accepts_extension_claim_scalars() {
        let point = [
            E::new(F::from_u64(1), F::from_u64(2)),
            E::new(F::from_u64(3), F::from_u64(4)),
        ];
        let openings = [
            E::new(F::from_u64(5), F::from_u64(6)),
            E::new(F::from_u64(7), F::from_u64(8)),
        ];
        let commitment = 11usize;
        let claims = VerifierClaims {
            commitment: &commitment,
            points: vec![PointClaim::all(&point[..], &openings[..])],
        };

        let prepared = prepare_verifier_claims(&setup(), &claims)
            .expect("extension-valued verifier claims should validate by shape");

        assert_eq!(prepared.opening_points, vec![&point[..]]);
        assert_eq!(prepared.openings, openings);
        assert_eq!(prepared.commitment, 11usize);
        assert_eq!(prepared.incidence_summary.num_claims, 2);
        assert_eq!(prepared.incidence_summary.num_polys, 2);
        assert_eq!(prepared.incidence_summary.num_points, 1);
    }
}
