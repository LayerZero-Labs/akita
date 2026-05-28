//! Verifier claim normalization helpers.

use akita_field::{AkitaError, FieldCore};
use akita_types::{
    validate_batched_inputs, AkitaExpandedSetup, ClaimIncidenceSummary, VerifierClaims,
};

/// Flattened and validated verifier claims.
pub(crate) struct PreparedVerifierClaims<'a, E: FieldCore, C> {
    /// Distinct opening points in caller order.
    pub opening_points: Vec<&'a [E]>,
    /// Commitments in opening-point order (one per point under the
    /// one-commitment-per-point invariant).
    pub commitments: Vec<C>,
    /// Claimed openings flattened by opening point, then by polynomial index.
    pub openings: Vec<E>,
    /// Normalized incidence summary that owns canonical root claim routing.
    pub incidence_summary: ClaimIncidenceSummary,
}

/// Validate and flatten verifier claims into the canonical batch layout.
///
/// # Errors
///
/// Returns an error if the claims are empty, exceed setup capacity, use
/// inconsistent opening-point dimensions, contain empty point payloads, or
/// overflow flattened claim counts.
pub(crate) fn prepare_verifier_claims<'a, F, E, C>(
    setup: &AkitaExpandedSetup<F>,
    claims: &VerifierClaims<'a, E, C>,
) -> Result<PreparedVerifierClaims<'a, E, C>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
    C: Clone,
{
    validate_batched_inputs(setup, claims, |payload| payload.openings.len(), false)?;

    // `validate_batched_inputs` already enforces the same limits and shape
    // constraints that `ClaimIncidence::validate` would, and verifier claims
    // are canonical by construction (each point lists its polys 0..k dense,
    // no duplicate edges), so skip the generic edge-set validation and
    // build the routing summary directly from per-point counts.
    let num_vars = claims[0].0.len();
    let num_polys_per_point: Vec<usize> = claims.iter().map(|(_, p)| p.openings.len()).collect();
    let summary = ClaimIncidenceSummary::from_point_polys(num_vars, num_polys_per_point)
        .map_err(|_| AkitaError::InvalidProof)?;

    let opening_points: Vec<&'a [E]> = claims.iter().map(|(point, _)| *point).collect();
    let commitments = claims
        .iter()
        .map(|(_, payload)| (*payload.commitment).clone())
        .collect();
    let openings = claims
        .iter()
        .flat_map(|(_, payload)| payload.openings.iter().copied())
        .collect();

    Ok(PreparedVerifierClaims {
        opening_points,
        commitments,
        openings,
        incidence_summary: summary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp2, Fp32, NegOneNr};
    use akita_types::{AkitaSetupSeed, CommittedOpenings, FlatMatrix};

    type F = Fp32<251>;
    type E = Fp2<F, NegOneNr>;

    fn setup() -> AkitaExpandedSetup<F> {
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupSeed {
                max_num_vars: 3,
                max_num_batched_polys: 4,
                max_num_points: 2,
                gen_ring_dim: 1,
                max_setup_len: 1,
                public_matrix_seed: [0u8; 32],
            },
            FlatMatrix::from_flat_data(vec![F::zero()], 1),
        )
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
        let claims = vec![(
            &point[..],
            CommittedOpenings {
                openings: &openings[..],
                commitment: &commitment,
            },
        )];

        let prepared = prepare_verifier_claims(&setup(), &claims)
            .expect("extension-valued verifier claims should validate by shape");

        assert_eq!(prepared.opening_points, vec![&point[..]]);
        assert_eq!(prepared.openings, openings);
        assert_eq!(prepared.commitments, vec![11usize]);
        assert_eq!(prepared.incidence_summary.num_claims(), 2);
        assert_eq!(prepared.incidence_summary.num_points(), 1);
    }
}
