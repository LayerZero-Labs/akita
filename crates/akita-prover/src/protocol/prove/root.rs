use super::*;
use crate::SelectedProverOpeningData;
use akita_config::TrustedScheduleCatalog;
use jolt_field::AdditiveGroup;

/// Prove an ordered statement using reusable commitments retained by one backend.
///
/// Akita runs on the caller's transcript: its first operation absorbs the
/// instance descriptor as a public message, and every Akita message lands in
/// the caller's argument string. The caller owns the transcript's protocol id,
/// session binding, and `finish`.
#[allow(clippy::too_many_arguments)]
pub fn batched_prove<'a, Cfg, B, H>(
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    schedules: &TrustedScheduleCatalog<Cfg>,
    backend: &B,
    opening: SelectedProverOpeningData<'a, Cfg::ExtField, B::CommitmentHandle, Cfg::Field>,
    transcript: &mut ProverTranscript<H>,
    basis: BasisMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding + AkitaSerialize + Unreduced + PseudoMersenne + Ring + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<Cfg::Field>
        + AkitaSerialize
        + 'static,
    B: ProverBackend<Cfg::Field, Cfg::ExtField>,
    H: Sponge,
{
    let (selection, claims) = opening.into_low_level_parts();
    let resolved = schedules.resolve_selection(selection)?;
    let layout = claims.opening_layout();
    resolved.validate_opening_layout(layout)?;
    let schedule = resolved.schedule();
    let proof_session = backend.begin_proof(expanded, schedules, schedule, layout)?;
    let guard = crate::backend::ProofScope::admitted(backend, proof_session);
    let proof_session = guard.session();
    let context = backend.proof_context(proof_session, 0)?;
    let mut material = Vec::with_capacity(layout.num_groups());
    for index in 0..layout.num_groups() {
        let group = schedule.root.params.group_params(layout, index)?;
        material.push(backend.validate_commitment(
            proof_session,
            &context.for_group(index),
            claims.group(index)?,
            &group.profile,
            claims.opening_claims().group_commitment(index)?,
        )?);
    }
    let (grinding_plan, descriptor_bytes) = akita_config::transcript_instance_descriptor::<
        Cfg::Field,
        Cfg,
    >(expanded, layout, selection, schedule, basis)?;
    transcript.site(
        akita_params::ProtocolSiteId {
            family: akita_params::transcript_site::SITE_FAMILY_ROOT_STATEMENT,
            ..akita_params::ProtocolSiteId::default()
        }
        .into(),
    );
    transcript.public_bytes(&descriptor_bytes);
    let mut grinding = akita_types::ProverGrinding::new(transcript, &grinding_plan);
    claims.append_to(&schedule.root.params, &mut grinding)?;
    let root_claims = claims.map_groups(OpeningSource::Commitment)?;
    let (next_params, next_binding) = schedule.recursive_folds.first().map_or(
        (
            fold::FoldSuccessorParams::Terminal(&schedule.terminal),
            akita_params::NextWitnessBindingPolicy::TerminalInnerState,
        ),
        |next| {
            (
                fold::FoldSuccessorParams::Recursive(next),
                akita_params::NextWitnessBindingPolicy::OuterPayload,
            )
        },
    );
    let prepared = prepare_fold::<Cfg::Field, Cfg::ExtField, B, _>(
        backend,
        root_claims,
        material,
        false,
        &mut grinding,
        0,
        &schedule.root.params,
        basis,
        proof_session,
        schedule.root.output_witness_len,
        next_params.inner_ring_dimension(),
        false,
    )?;
    let root = prove_fold::<Cfg::Field, Cfg::ExtField, B, _>(
        expanded,
        prefix_slots,
        backend,
        proof_session,
        &mut grinding,
        0,
        &schedule.root.params,
        next_params,
        schedule.root.output_witness_len,
        next_binding,
        prepared,
    )?;
    let suffix = suffix::prove_suffix::<Cfg, B, _>(
        expanded,
        prefix_slots,
        backend,
        &mut grinding,
        root.next_state,
        schedule,
        proof_session,
    )?;
    if suffix.num_levels != schedule.num_fold_levels() {
        return Err(AkitaError::Internal(
            "proved suffix fold count differs from schedule".into(),
        ));
    }
    guard.finish()?;
    grinding.finish()
}
