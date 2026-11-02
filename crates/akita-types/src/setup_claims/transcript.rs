//! Setup-batch grammar. Statements are bound before randomization, and every
//! batch coefficient and reduction round is a grinding-priced query.
use super::SetupBatchStatement;
use crate::{GrindingReplay, ProverGrinding, VerifierGrinding};
use akita_error::AkitaError;
use akita_params::{ChallengeFieldOrder, GrindingPlan, GrindingSite};
use akita_transcript::{
    exchange_extension_group, new_prover_channel, new_verifier_channel, ProofChannel,
    ProtocolSiteId,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

const FAMILY: u32 = 0x53424154;
fn site(invocation: u32, round: u32, detail: u32) -> ProtocolSiteId {
    ProtocolSiteId {
        family: FAMILY,
        invocation,
        round,
        detail,
        ..ProtocolSiteId::default()
    }
}

/// Draw one independent coefficient per claim behind the batch's grinding query.
pub fn batch_coefficients<F, E, G>(
    grinding: &mut G,
    level: u32,
    count: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    G: GrindingReplay,
{
    grinding.grinded_ext_challenges::<F, E>(GrindingSite::SetupBatch { level }, count)
}

pub fn setup_evaluation<F, E, S>(state: &mut S, mut value: E) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    S: ProofChannel,
{
    exchange_extension_group::<F, E, _>(state, site(0, 0, 11), std::slice::from_mut(&mut value))?;
    Ok(value)
}

/// Public grinding plan of a standalone reduction over `statement`.
///
/// Rejects field towers whose challenge sampling is uncertified, or whose
/// queries cannot reach the 128-bit rate within the grinding limit.
pub fn standalone_grinding_plan<F, E>(
    statement: &SetupBatchStatement<E>,
) -> Result<GrindingPlan, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    if !akita_transcript::field_sampling_is_certified(
        F::NUM_BYTES,
        F::MODULUS_BITS,
        akita_params::TRANSCRIPT_GRINDING_QUERY_LIMIT,
    ) {
        return Err(AkitaError::InvalidSetup(
            "setup batch field-challenge sampling is not certified".into(),
        ));
    }
    let challenge_order = ChallengeFieldOrder::from_field(
        F::MODULUS_BITS,
        E::DEGREE,
        akita_params::field_modulus::<F>()?,
    )?;
    akita_params::derive_setup_batch_grinding_plan(
        challenge_order,
        statement.claims().len(),
        statement.polynomial().num_variables(),
    )
}

/// Instance bytes of a standalone reduction: its statement, then its plan digest.
fn standalone_instance<F, E>(
    statement: &SetupBatchStatement<E>,
    plan: &GrindingPlan,
) -> Result<Vec<u8>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let mut bytes = statement.descriptor_bytes::<F>()?;
    bytes.extend_from_slice(&plan.digest()?);
    Ok(bytes)
}

/// Start the prover side of a standalone reduction under `plan`.
pub fn standalone_prover<'plan, F, E>(
    session: &[u8],
    statement: &SetupBatchStatement<E>,
    plan: &'plan GrindingPlan,
) -> Result<ProverGrinding<'plan>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let state = new_prover_channel(session, &standalone_instance::<F, E>(statement, plan)?)?;
    Ok(ProverGrinding::new(state, plan))
}

/// Start the verifier side of a standalone reduction under `plan`.
pub fn standalone_verifier<'proof, 'plan, F, E>(
    session: &[u8],
    statement: &SetupBatchStatement<E>,
    plan: &'plan GrindingPlan,
    proof: &'proof [u8],
) -> Result<VerifierGrinding<'proof, 'plan>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let state = new_verifier_channel(
        session,
        &standalone_instance::<F, E>(statement, plan)?,
        proof,
    )?;
    Ok(VerifierGrinding::new(state, plan))
}

/// Bind a complete inherited or combined collection in the outer proof stream.
pub fn bind_outer_statement<F, E, S>(
    state: &mut S,
    level: u32,
    statement: &SetupBatchStatement<E>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    S: ProofChannel,
{
    let mut context = site(0, 0, 20);
    context.level = level;
    akita_transcript::public_bytes(state, context, &statement.descriptor_bytes::<F>()?)
}

/// The existing successor must open this batch's exact actual setup prefix.
pub fn bind_outer_prefix<F, E, G>(
    statement: &SetupBatchStatement<E>,
    slot: &crate::SetupPrefixVerifierSlot<F>,
    successor: &akita_params::CommittedGroupParams,
    grinding: &mut G,
    level: u32,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field,
    G: crate::GrindingReplay,
{
    use akita_serialization::AkitaSerialize;
    crate::setup_prefix_coverage_eval_len(
        None,
        &slot.id,
        successor,
        slot.id.natural_len,
        statement.coefficient_len(),
        "outer setup batch prefix mismatch",
    )?;
    if successor
        .setup_prefix()
        .and_then(|prefix| prefix.slot_id())
        .as_ref()
        != Some(&slot.id)
        || slot.id.n_prefix()? != statement.polynomial().field_len()
    {
        return Err(AkitaError::InvalidSetup(
            "outer batch does not match the successor setup prefix".into(),
        ));
    }
    let mut bytes = Vec::new();
    slot.id
        .serialize_compressed(&mut bytes)
        .map_err(|_| AkitaError::InvalidProof)?;
    crate::stage3_public_slot(grinding, level, &bytes)
}
