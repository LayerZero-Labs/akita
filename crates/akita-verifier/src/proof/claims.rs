//! Verifier claim normalization helpers.

use akita_field::{AkitaError, FieldCore};
use akita_types::{
    validate_batched_inputs, verifier_claims_to_opening_batch, AkitaExpandedSetup, OpeningBatch,
    OpeningBatchLimits, VerifierClaims,
};

/// Flattened and validated verifier claims.
pub(crate) struct PreparedVerifierClaims<'a, E: FieldCore, C> {
    /// Shared opening point.
    pub opening_point: &'a [E],
    /// Batch commitment.
    pub commitment: C,
    /// Claimed openings in polynomial order.
    pub openings: Vec<E>,
    /// Normalized opening-batch summary that owns canonical root claim routing.
    pub opening_batch: OpeningBatch,
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

    let (opening_point, payload) = claims;
    let batch = verifier_claims_to_opening_batch(claims);
    let summary = batch
        .validate(OpeningBatchLimits {
            max_num_vars: setup.seed().max_num_vars,
            max_num_claims: setup.seed().max_num_batched_polys,
        })
        .map_err(|_| AkitaError::InvalidProof)?;

    let openings = payload.openings.to_vec();
    let commitment = (*payload.commitment).clone();

    Ok(PreparedVerifierClaims {
        opening_point,
        commitment,
        openings,
        opening_batch: summary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp32, FpExt2, NegOneNr};
    use akita_types::{AkitaSetupSeed, CommittedOpenings, FlatMatrix};

    type F = Fp32<251>;
    type E = FpExt2<F, NegOneNr>;

    fn setup() -> AkitaExpandedSetup<F> {
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupSeed {
                max_num_vars: 3,
                max_num_batched_polys: 4,
                gen_ring_dim: 1,
                max_setup_len: 1,
                #[cfg(feature = "zk")]
                max_zk_b_len: 1,
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
                public_matrix_seed: [0u8; 32],
            },
            FlatMatrix::from_flat_data(vec![F::zero()], 1),
            #[cfg(feature = "zk")]
            FlatMatrix::from_flat_data(vec![F::zero()], 1),
            #[cfg(feature = "zk")]
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
        let claims = (
            &point[..],
            CommittedOpenings {
                openings: &openings[..],
                commitment: &commitment,
            },
        );

        let prepared = prepare_verifier_claims(&setup(), &claims)
            .expect("extension-valued verifier claims should validate by shape");

        assert_eq!(prepared.opening_point, &point[..]);
        assert_eq!(prepared.openings, openings);
        assert_eq!(prepared.commitment, 11usize);
        assert_eq!(prepared.opening_batch.num_claims(), 2);
    }
}
