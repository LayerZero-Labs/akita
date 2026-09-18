//! Canonical native proof-stream grammar for extension-opening reduction.

use crate::{
    tensor_opening_split, GrindingSite, NativeProverGrinding, NativeVerifierGrinding,
    OpeningClaimsLayout,
};
use akita_error::{checked, AkitaError};
use akita_transcript::{
    public_native_extensions_prover, public_native_extensions_verifier,
    receive_native_extension_group, send_native_extension_group, ProtocolSiteId,
    SITE_FAMILY_EXTENSION_OPENING_REDUCTION,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

const STAGE_OPENINGS: u32 = 1;
const STAGE_PARTIALS: u32 = 2;
const STAGE_FINAL_CLAIMS: u32 = 3;

/// EOR has one sumcheck invocation at each fold level.
pub const NATIVE_EOR_SUMCHECK_INVOCATION: u32 = 0;

/// Native EOR prefix values shared by arithmetic proving and verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeEorPrefix<E: Field> {
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
        return Err(AkitaError::InvalidProof);
    }
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    let expected_partials =
        checked::product([width, num_claims]).ok_or(AkitaError::InvalidProof)?;
    if partial_count != expected_partials {
        return Err(AkitaError::InvalidProof);
    }
    Ok((split_bits, num_claims))
}

fn prover_batch_challenges<F, E>(
    grinding: &mut NativeProverGrinding<'_>,
    opening_batch: &OpeningClaimsLayout,
    level: u32,
    split_bits: usize,
) -> Result<(Vec<E>, Vec<E>), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let eta = grinding.grinded_ext_challenges::<F, E>(
        GrindingSite::ExtensionOpeningPoint { level },
        split_bits,
    )?;
    let claim_coefficients = if opening_batch.requires_row_batch_challenge() {
        grinding.grinded_ext_challenges::<F, E>(
            GrindingSite::ExtensionOpeningClaimBatch { level },
            opening_batch.num_total_polynomials(),
        )?
    } else {
        vec![E::one()]
    };
    Ok((eta, claim_coefficients))
}

fn verifier_batch_challenges<F, E>(
    grinding: &mut NativeVerifierGrinding<'_, '_>,
    opening_batch: &OpeningClaimsLayout,
    level: u32,
    split_bits: usize,
) -> Result<(Vec<E>, Vec<E>), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let eta = grinding.grinded_ext_challenges::<F, E>(
        GrindingSite::ExtensionOpeningPoint { level },
        split_bits,
    )?;
    let claim_coefficients = if opening_batch.requires_row_batch_challenge() {
        grinding.grinded_ext_challenges::<F, E>(
            GrindingSite::ExtensionOpeningClaimBatch { level },
            opening_batch.num_total_polynomials(),
        )?
    } else {
        vec![E::one()]
    };
    Ok((eta, claim_coefficients))
}

/// Emit and bind the native EOR prefix before its sumcheck rounds.
pub fn native_eor_prover_prefix<F, E>(
    grinding: &mut NativeProverGrinding<'_>,
    opening_batch: &OpeningClaimsLayout,
    openings: &[E],
    partials: &[E],
    level: u32,
) -> Result<NativeEorPrefix<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let (split_bits, _) = validate_shape::<F, E>(opening_batch, openings, partials.len())?;
    public_native_extensions_prover::<F, E>(
        grinding.state_mut(),
        eor_site(level, STAGE_OPENINGS),
        openings,
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    send_native_extension_group::<F, E>(
        grinding.state_mut(),
        eor_site(level, STAGE_PARTIALS),
        partials,
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    let (eta, claim_coefficients) =
        prover_batch_challenges::<F, E>(grinding, opening_batch, level, split_bits)?;
    Ok(NativeEorPrefix {
        partials: partials.to_vec(),
        eta,
        claim_coefficients,
    })
}

/// Receive and bind the native EOR prefix before replaying its sumcheck.
pub fn native_eor_verifier_prefix<F, E>(
    grinding: &mut NativeVerifierGrinding<'_, '_>,
    opening_batch: &OpeningClaimsLayout,
    openings: &[E],
    level: u32,
) -> Result<NativeEorPrefix<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let (_, width) = tensor_opening_split::<F, E>()?;
    let partial_count = checked::product([width, opening_batch.num_total_polynomials()])
        .ok_or(AkitaError::InvalidProof)?;
    let (split_bits, _) = validate_shape::<F, E>(opening_batch, openings, partial_count)?;
    public_native_extensions_verifier::<F, E>(
        grinding.state_mut(),
        eor_site(level, STAGE_OPENINGS),
        openings,
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    let partials = receive_native_extension_group::<F, E>(
        grinding.state_mut(),
        eor_site(level, STAGE_PARTIALS),
        partial_count,
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    let (eta, claim_coefficients) =
        verifier_batch_challenges::<F, E>(grinding, opening_batch, level, split_bits)?;
    Ok(NativeEorPrefix {
        partials,
        eta,
        claim_coefficients,
    })
}

/// Emit the schedule-fixed EOR final-claim vector after sumcheck replay.
pub fn native_eor_prover_final_claims<F, E>(
    grinding: &mut NativeProverGrinding<'_>,
    opening_batch: &OpeningClaimsLayout,
    final_claims: &[E],
    level: u32,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    if final_claims.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }
    send_native_extension_group::<F, E>(
        grinding.state_mut(),
        eor_site(level, STAGE_FINAL_CLAIMS),
        final_claims,
    )
    .map_err(|_| AkitaError::InvalidProof)
}

/// Receive the schedule-fixed EOR final-claim vector after sumcheck replay.
pub fn native_eor_verifier_final_claims<F, E>(
    grinding: &mut NativeVerifierGrinding<'_, '_>,
    opening_batch: &OpeningClaimsLayout,
    level: u32,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    receive_native_extension_group::<F, E>(
        grinding.state_mut(),
        eor_site(level, STAGE_FINAL_CLAIMS),
        opening_batch.num_total_polynomials(),
    )
    .map_err(|_| AkitaError::InvalidProof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GrindingPlan, GrindingRun, PolynomialGroupLayout};
    use akita_transcript::{new_native_prover, new_native_verifier};
    use jolt_field::{FpExt4, Prime32Offset99, Ring};

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
                GrindingRun::proof_of_work(GrindingSite::ExtensionOpeningPoint { level }, 1, 128)
                    .unwrap(),
                GrindingRun::proof_of_work(
                    GrindingSite::ExtensionOpeningClaimBatch { level },
                    1,
                    128,
                )
                .unwrap(),
            ],
            128,
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
    fn native_eor_grammar_roundtrips_without_structured_proof() {
        let level = 3;
        let (plan, layout, openings, partials, final_claims) = fixture();
        let state = new_native_prover(b"native-eor", b"fixture").unwrap();
        let mut prover = NativeProverGrinding::new(state, &plan);
        let prover_prefix =
            native_eor_prover_prefix::<F, E>(&mut prover, &layout, &openings, &partials, level)
                .unwrap();
        native_eor_prover_final_claims::<F, E>(&mut prover, &layout, &final_claims, level).unwrap();
        let proof = prover.finish().unwrap();

        let state = new_native_verifier(b"native-eor", b"fixture", &proof).unwrap();
        let mut verifier = NativeVerifierGrinding::new(state, &plan);
        let verifier_prefix =
            native_eor_verifier_prefix::<F, E>(&mut verifier, &layout, &openings, level).unwrap();
        assert_eq!(verifier_prefix, prover_prefix);
        assert_eq!(
            native_eor_verifier_final_claims::<F, E>(&mut verifier, &layout, level).unwrap(),
            final_claims
        );
        verifier.finish().unwrap();
    }

    #[test]
    fn native_eor_rejects_truncation_before_final_claims() {
        let level = 3;
        let (plan, layout, openings, partials, final_claims) = fixture();
        let state = new_native_prover(b"native-eor", b"fixture").unwrap();
        let mut prover = NativeProverGrinding::new(state, &plan);
        native_eor_prover_prefix::<F, E>(&mut prover, &layout, &openings, &partials, level)
            .unwrap();
        native_eor_prover_final_claims::<F, E>(&mut prover, &layout, &final_claims, level).unwrap();
        let mut proof = prover.finish().unwrap();
        proof.pop();

        let state = new_native_verifier(b"native-eor", b"fixture", &proof).unwrap();
        let mut verifier = NativeVerifierGrinding::new(state, &plan);
        native_eor_verifier_prefix::<F, E>(&mut verifier, &layout, &openings, level).unwrap();
        assert_eq!(
            native_eor_verifier_final_claims::<F, E>(&mut verifier, &layout, level),
            Err(AkitaError::InvalidProof)
        );
    }
}
