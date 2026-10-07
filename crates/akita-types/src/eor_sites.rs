//! Canonical proof-stream grammar for extension-opening reduction.

use crate::{tensor_opening_split, GrindingReplay};
use akita_error::{checked, AkitaError};
use akita_params::{GrindingSite, OpeningClaimsLayout};
use akita_transcript::{
    exchange_extension_group, public_extensions, ProtocolSiteId,
    SITE_FAMILY_EXTENSION_OPENING_REDUCTION,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

const STAGE_OPENINGS: u32 = 1;
const STAGE_PARTIALS: u32 = 2;
const STAGE_FINAL_CLAIMS: u32 = 3;

/// EOR has one sumcheck invocation at each fold level.
pub const EOR_SUMCHECK_INVOCATION: u32 = 0;

/// EOR prefix values shared by arithmetic proving and verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EorPrefix<E: Field> {
    /// Proof-supplied tensor column partials in canonical claim-major order.
    pub partials: Vec<E>,
    /// Tensor-row reduction point protected by the opening-point grind.
    pub eta: Vec<E>,
    /// Coefficients batching the individual opening claims.
    pub claim_coefficients: Vec<E>,
}

fn eor_site(level: u32, stage: u32) -> ProtocolSiteId {
    ProtocolSiteId {
        family: SITE_FAMILY_EXTENSION_OPENING_REDUCTION,
        level,
        stage,
        ..ProtocolSiteId::default()
    }
}

fn validate_shape<F, E>(
    opening_batch: &OpeningClaimsLayout,
    openings: &[E],
    partial_count: usize,
) -> Result<(usize, usize), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    opening_batch.check()?;
    let num_claims = opening_batch.num_total_polynomials();
    if openings.len() != num_claims {
        return Err(AkitaError::InvalidSize {
            expected: num_claims,
            actual: openings.len(),
        });
    }
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    let expected_partials = checked::product([width, num_claims])
        .ok_or_else(|| AkitaError::InvalidInput("eor partial count overflows usize".into()))?;
    if partial_count != expected_partials {
        return Err(AkitaError::InvalidSize {
            expected: expected_partials,
            actual: partial_count,
        });
    }
    Ok((split_bits, num_claims))
}

fn batch_challenges<E>(
    opening_batch: &OpeningClaimsLayout,
    level: u32,
    split_bits: usize,
    mut draw: impl FnMut(GrindingSite, usize) -> Result<Vec<E>, AkitaError>,
) -> Result<(Vec<E>, Vec<E>), AkitaError>
where
    E: Field,
{
    let eta = draw(GrindingSite::ExtensionOpeningPoint { level }, split_bits)?;
    let claim_coefficients = if opening_batch.requires_row_batch_challenge() {
        draw(
            GrindingSite::ExtensionOpeningClaimBatch { level },
            opening_batch.num_total_polynomials(),
        )?
    } else {
        vec![E::one()]
    };
    Ok((eta, claim_coefficients))
}

/// Exchange and bind the EOR prefix before its sumcheck rounds.
///
/// The prover passes its tensor column partials; the verifier passes
/// `width * num_claims` placeholders and receives them in place.
pub fn eor_prefix<F, E, G>(
    grinding: &mut G,
    opening_batch: &OpeningClaimsLayout,
    openings: &[E],
    mut partials: Vec<E>,
    level: u32,
) -> Result<EorPrefix<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    G: GrindingReplay,
{
    let (split_bits, _) = validate_shape::<F, E>(opening_batch, openings, partials.len())?;
    public_extensions::<F, E, _>(
        grinding.state_mut(),
        eor_site(level, STAGE_OPENINGS),
        openings,
    )?;
    exchange_extension_group::<F, E, _>(
        grinding.state_mut(),
        eor_site(level, STAGE_PARTIALS),
        &mut partials,
    )?;
    let (eta, claim_coefficients) =
        batch_challenges::<E>(opening_batch, level, split_bits, |site, count| {
            grinding.grinded_ext_challenges::<F, E>(site, count)
        })?;
    Ok(EorPrefix {
        partials,
        eta,
        claim_coefficients,
    })
}

/// Exchange the schedule-fixed EOR final-claim vector after sumcheck replay.
pub fn eor_final_claims<F, E, G>(
    grinding: &mut G,
    opening_batch: &OpeningClaimsLayout,
    final_claims: &mut [E],
    level: u32,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    G: GrindingReplay,
{
    if final_claims.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidSize {
            expected: opening_batch.num_total_polynomials(),
            actual: final_claims.len(),
        });
    }
    exchange_extension_group::<F, E, _>(
        grinding.state_mut(),
        eor_site(level, STAGE_FINAL_CLAIMS),
        final_claims,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProverGrinding, VerifierGrinding};
    use akita_params::{ChallengeFieldOrder, GrindingPlan, GrindingRun, PolynomialGroupLayout};
    use akita_transcript::{new_prover_channel, new_verifier_channel};
    use jolt_field::{FpExt4, Prime32Offset99, Ring, Zero};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    fn fixture() -> (GrindingPlan, OpeningClaimsLayout, Vec<E>, Vec<E>, Vec<E>) {
        let level = 3;
        let layout = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::new(12, 2),
            PolynomialGroupLayout::new(15, 1),
        ])
        .unwrap();
        let plan = GrindingPlan::new(
            vec![
                GrindingRun::proof_of_work(
                    GrindingSite::ExtensionOpeningPoint { level },
                    1,
                    ChallengeFieldOrder::from_full_capacity(128).unwrap(),
                )
                .unwrap(),
                GrindingRun::proof_of_work(
                    GrindingSite::ExtensionOpeningClaimBatch { level },
                    1,
                    ChallengeFieldOrder::from_full_capacity(128).unwrap(),
                )
                .unwrap(),
            ],
            ChallengeFieldOrder::from_full_capacity(128).unwrap(),
        )
        .unwrap();
        let openings = (1..=layout.num_total_polynomials())
            .map(|value| E::from_u64(value as u64))
            .collect::<Vec<_>>();
        let (_, width) = tensor_opening_split::<F, E>().unwrap();
        let partials = (0..width * layout.num_total_polynomials())
            .map(|value| E::from_u64((value + 10) as u64))
            .collect::<Vec<_>>();
        let final_claims = (0..layout.num_total_polynomials())
            .map(|value| E::from_u64((value + 100) as u64))
            .collect::<Vec<_>>();
        (plan, layout, openings, partials, final_claims)
    }

    #[test]
    fn eor_rejects_argument_lengths_before_reading_proof_atoms() {
        let (plan, layout, openings, partials, _) = fixture();
        let state = new_verifier_channel(b"native-eor", b"fixture", &[]).unwrap();
        let mut verifier = VerifierGrinding::new(state, &plan);
        assert!(matches!(
            eor_prefix::<F, E, _>(&mut verifier, &layout, &[], partials.clone(), 3),
            Err(AkitaError::InvalidSize {
                expected: 3,
                actual: 0
            })
        ));
        let expected = partials.len();
        assert!(matches!(
            eor_prefix::<F, E, _>(&mut verifier, &layout, &openings, vec![], 3),
            Err(AkitaError::InvalidSize { expected: count, actual: 0 }) if count == expected
        ));
        assert!(matches!(
            eor_final_claims::<F, E, _>(&mut verifier, &layout, &mut [], 3),
            Err(AkitaError::InvalidSize {
                expected: 3,
                actual: 0
            })
        ));
    }

    #[test]
    fn eor_grammar_roundtrips_without_structured_proof() {
        let level = 3;
        let (plan, layout, openings, partials, mut final_claims) = fixture();
        let state = new_prover_channel(b"native-eor", b"fixture").unwrap();
        let mut prover = ProverGrinding::new(state, &plan);
        let prover_prefix =
            eor_prefix::<F, E, _>(&mut prover, &layout, &openings, partials, level).unwrap();
        eor_final_claims::<F, E, _>(&mut prover, &layout, &mut final_claims, level).unwrap();
        let proof = prover.finish().unwrap();

        let state = new_verifier_channel(b"native-eor", b"fixture", &proof).unwrap();
        let mut verifier = VerifierGrinding::new(state, &plan);
        let verifier_prefix = eor_prefix::<F, E, _>(
            &mut verifier,
            &layout,
            &openings,
            vec![E::zero(); prover_prefix.partials.len()],
            level,
        )
        .unwrap();
        assert_eq!(verifier_prefix, prover_prefix);
        let mut received = vec![E::zero(); final_claims.len()];
        eor_final_claims::<F, E, _>(&mut verifier, &layout, &mut received, level).unwrap();
        assert_eq!(received, final_claims);
        verifier.finish().unwrap();
    }

    #[test]
    fn eor_rejects_truncation_before_final_claims() {
        let level = 3;
        let (plan, layout, openings, partials, mut final_claims) = fixture();
        let state = new_prover_channel(b"native-eor", b"fixture").unwrap();
        let mut prover = ProverGrinding::new(state, &plan);
        let partial_count = partials.len();
        eor_prefix::<F, E, _>(&mut prover, &layout, &openings, partials, level).unwrap();
        eor_final_claims::<F, E, _>(&mut prover, &layout, &mut final_claims, level).unwrap();
        let mut proof = prover.finish().unwrap();
        proof.pop();

        let state = new_verifier_channel(b"native-eor", b"fixture", &proof).unwrap();
        let mut verifier = VerifierGrinding::new(state, &plan);
        eor_prefix::<F, E, _>(
            &mut verifier,
            &layout,
            &openings,
            vec![E::zero(); partial_count],
            level,
        )
        .unwrap();
        let mut received = vec![E::zero(); layout.num_total_polynomials()];
        assert_eq!(
            eor_final_claims::<F, E, _>(&mut verifier, &layout, &mut received, level),
            Err(AkitaError::InvalidProof)
        );
    }
}
