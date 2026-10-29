//! Standalone batch grammar. All claims are absorbed by the channel's instance before randomization.
use super::SetupBatchStatement;
use akita_error::AkitaError;
use akita_sumcheck::{SumcheckProverChannel, SumcheckRole, SumcheckVerifierChannel};
use akita_transcript::{
    exchange_extension_group, ext_challenge, new_prover_channel, new_verifier_channel,
    ProofChannel, ProtocolSiteId, ProverChannel, VerifierChannel,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};
use std::marker::PhantomData;

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

pub fn batch_coefficients<F, E, S>(state: &mut S, count: usize) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    S: ProofChannel,
{
    (0..count)
        .map(|index| {
            let index = u32::try_from(index).map_err(|_| AkitaError::InvalidProof)?;
            ext_challenge::<F, E, _>(state, site(index, 0, 10))
        })
        .collect()
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

pub struct SetupBatchProverChannel<F: Field, E> {
    pub state: ProverChannel,
    marker: PhantomData<(F, E)>,
}
impl<F: Field + CanonicalEncoding, E: ExtField<F>> SetupBatchProverChannel<F, E> {
    pub fn new(session: &[u8], statement: &SetupBatchStatement<E>) -> Result<Self, AkitaError> {
        Ok(Self {
            state: new_prover_channel(session, &statement.descriptor_bytes::<F>()?)?,
            marker: PhantomData,
        })
    }
}
impl<F: Field + CanonicalEncoding, E: ExtField<F>> SumcheckProverChannel<E>
    for SetupBatchProverChannel<F, E>
{
    fn state_mut(&mut self) -> &mut ProverChannel {
        &mut self.state
    }
    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        site(invocation, round, role as u32)
    }
    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        ext_challenge::<F, E, _>(
            &mut self.state,
            site(invocation, round, SumcheckRole::Challenge as u32),
        )
    }
}

pub struct SetupBatchVerifierChannel<'proof, F: Field, E> {
    pub state: VerifierChannel<'proof>,
    marker: PhantomData<(F, E)>,
}
impl<'proof, F: Field + CanonicalEncoding, E: ExtField<F>> SetupBatchVerifierChannel<'proof, F, E> {
    pub fn new(
        session: &[u8],
        statement: &SetupBatchStatement<E>,
        proof: &'proof [u8],
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            state: new_verifier_channel(session, &statement.descriptor_bytes::<F>()?, proof)?,
            marker: PhantomData,
        })
    }
}
impl<'proof, F: Field + CanonicalEncoding, E: ExtField<F>> SumcheckVerifierChannel<'proof, E>
    for SetupBatchVerifierChannel<'proof, F, E>
{
    fn state_mut(&mut self) -> &mut VerifierChannel<'proof> {
        &mut self.state
    }
    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        site(invocation, round, role as u32)
    }
    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        ext_challenge::<F, E, _>(
            &mut self.state,
            site(invocation, round, SumcheckRole::Challenge as u32),
        )
    }
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
