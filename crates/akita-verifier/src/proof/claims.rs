//! Verifier claim normalization helpers.

use akita_field::{AkitaError, FieldCore};
use akita_types::{
    validate_batched_inputs, verifier_claims_to_incidence, AkitaExpandedSetup,
    AkitaRootBatchSummary, ClaimIncidenceLimits, ClaimIncidenceSummary, VerifierClaims,
};

/// Flattened and validated verifier claims.
pub(crate) struct PreparedVerifierClaims<'a, E: FieldCore, C> {
    /// Distinct opening points in caller order.
    pub opening_points: Vec<&'a [E]>,
    /// Commitments flattened by opening point and commitment group.
    pub commitments: Vec<C>,
    /// Claimed openings flattened by opening point, group, then claim.
    pub openings: Vec<E>,
    /// Normalized incidence summary that owns canonical root claim routing.
    pub incidence_summary: ClaimIncidenceSummary,
    /// Number of variables in each opening point.
    pub num_vars: usize,
    /// Total number of root claims represented by the layout.
    pub layout_num_claims: usize,
    /// Aggregate root batch summary used for schedule lookup.
    pub batch_summary: AkitaRootBatchSummary,
}

/// Validate and flatten verifier claims into the canonical batch layout.
///
/// # Errors
///
/// Returns an error if the claims are empty, exceed setup capacity, use
/// inconsistent opening-point dimensions, contain empty groups, or overflow
/// flattened claim counts.
pub(crate) fn prepare_verifier_claims<'a, F, E, C>(
    setup: &AkitaExpandedSetup<F>,
    claims: &VerifierClaims<'a, E, C>,
) -> Result<PreparedVerifierClaims<'a, E, C>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
    C: Clone,
{
    validate_batched_inputs(setup, claims, |group| group.openings.len(), false)?;

    let incidence = verifier_claims_to_incidence(claims);
    let summary = incidence
        .validate(ClaimIncidenceLimits {
            max_num_vars: setup.seed.max_num_vars,
            max_num_points: setup.seed.max_num_points,
            max_num_claims: setup.seed.max_num_batched_polys,
        })
        .map_err(|_| AkitaError::InvalidProof)?;

    let opening_points = incidence.points;
    let commitments = incidence
        .groups
        .iter()
        .map(|group| (*group.commitment).clone())
        .collect();
    let openings = incidence
        .claims
        .iter()
        .map(|claim| claim.claimed_eval)
        .collect();
    let num_vars = summary.num_vars;
    let layout_num_claims = summary.num_claims;
    let batch_summary = summary
        .root_batch_summary()
        .map_err(|_| AkitaError::InvalidProof)?;

    Ok(PreparedVerifierClaims {
        opening_points,
        commitments,
        openings,
        incidence_summary: summary,
        num_vars,
        layout_num_claims,
        batch_summary,
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
        let claims = vec![(
            &point[..],
            vec![CommittedOpenings {
                openings: &openings[..],
                commitment: &commitment,
            }],
        )];

        let prepared = prepare_verifier_claims(&setup(), &claims)
            .expect("extension-valued verifier claims should validate by shape");

        assert_eq!(prepared.opening_points, vec![&point[..]]);
        assert_eq!(prepared.openings, openings);
        assert_eq!(prepared.commitments, vec![11usize]);
        assert_eq!(prepared.incidence_summary.num_claims, 2);
        assert_eq!(prepared.incidence_summary.num_groups, 1);
        assert_eq!(prepared.incidence_summary.num_points, 1);
        assert_eq!(
            prepared.batch_summary,
            AkitaRootBatchSummary {
                num_claims: 2,
                num_commitment_groups: 1,
                num_points: 1,
            }
        );
        assert_eq!(prepared.layout_num_claims, 2);
    }
}
