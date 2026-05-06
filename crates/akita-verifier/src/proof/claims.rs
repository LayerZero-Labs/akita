//! Verifier claim normalization helpers.

use akita_field::{AkitaError, FieldCore};
use akita_types::{
    checked_total_claims, validate_batched_inputs, AkitaExpandedSetup, AkitaRootBatchSummary,
    MultiPointBatchShape, VerifierClaims,
};

/// Flattened and validated verifier claims.
pub struct PreparedVerifierClaims<'a, E: FieldCore, C> {
    /// Distinct opening points in caller order.
    pub opening_points: Vec<&'a [E]>,
    /// Commitments flattened by opening point and commitment group.
    pub commitments: Vec<C>,
    /// Claimed openings flattened by opening point, group, then claim.
    pub openings: Vec<E>,
    /// Multipoint batch routing shape.
    pub batch_shape: MultiPointBatchShape,
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
pub fn prepare_verifier_claims<'a, F, E, C>(
    setup: &AkitaExpandedSetup<F>,
    claims: &VerifierClaims<'a, E, C>,
) -> Result<PreparedVerifierClaims<'a, E, C>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
    C: Clone,
{
    validate_batched_inputs(setup, claims, |group| group.openings.len(), false)?;
    let opening_points: Vec<&'a [E]> = claims.iter().map(|(point, _)| *point).collect();
    let commitments: Vec<C> = claims
        .iter()
        .flat_map(|(_, groups)| {
            groups
                .iter()
                .map(|group| group.commitment.clone())
                .collect::<Vec<_>>()
        })
        .collect();
    let num_vars = opening_points[0].len();
    let batch_shape = MultiPointBatchShape {
        point_group_sizes: claims.iter().map(|(_, groups)| groups.len()).collect(),
        claim_group_sizes: claims
            .iter()
            .flat_map(|(_, groups)| groups.iter().map(|group| group.openings.len()))
            .collect(),
        claim_to_point: claims
            .iter()
            .enumerate()
            .flat_map(|(point_idx, (_, groups))| {
                groups
                    .iter()
                    .flat_map(move |group| std::iter::repeat_n(point_idx, group.openings.len()))
            })
            .collect(),
    };
    let openings: Vec<E> = claims
        .iter()
        .flat_map(|(_, groups)| {
            groups
                .iter()
                .flat_map(|group| group.openings.iter().copied())
                .collect::<Vec<_>>()
        })
        .collect();
    let layout_num_claims = checked_total_claims(&batch_shape.claim_group_sizes, "batched_verify")
        .map_err(|_| AkitaError::InvalidProof)?;
    let batch_summary = AkitaRootBatchSummary::from_claim_group_sizes(
        &batch_shape.claim_group_sizes,
        opening_points.len(),
    )
    .map_err(|_| AkitaError::InvalidProof)?;

    Ok(PreparedVerifierClaims {
        opening_points,
        commitments,
        openings,
        batch_shape,
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
        assert_eq!(prepared.layout_num_claims, 2);
    }
}
