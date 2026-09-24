use super::*;
use crate::SelectedProverOpeningData;
use akita_config::TrustedScheduleCatalog;
use jolt_field::AdditiveGroup;

/// Prove an ordered statement using reusable commitments retained by one backend.
#[allow(clippy::too_many_arguments)]
pub fn batched_prove<'a, Cfg, B>(
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    schedules: &TrustedScheduleCatalog<Cfg>,
    backend: &B,
    opening: SelectedProverOpeningData<'a, Cfg::ExtField, B::CommitmentHandle, Cfg::Field>,
    transcript_session: &[u8],
    basis: BasisMode,
) -> Result<Vec<u8>, AkitaError>
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
{
    let (selection, claims) = opening.into_low_level_parts();
    let resolved = schedules.resolve_selection(selection)?;
    let layout = claims.opening_layout();
    resolved.validate_opening_layout(layout)?;
    let schedule = resolved.schedule();
    schedule.validate_nonterminal_opening_execution(Cfg::EXT_DEGREE)?;
    let proof_session = backend.begin_proof(expanded, schedule, layout)?;
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
    let native = akita_transcript::new_native_prover(transcript_session, &descriptor_bytes)
        .map_err(|_| AkitaError::InvalidSetup("native transcript initialization failed".into()))?;
    let mut grinding = akita_types::NativeProverGrinding::new(native, &grinding_plan);
    claims.append_to_native(&schedule.root.params, &mut grinding)?;
    let root_claims = claims.map_groups(OpeningSource::Commitment)?;
    let (next_params, next_binding) = schedule.recursive_folds.first().map_or(
        (
            fold::FoldSuccessorParams::Terminal(&schedule.terminal),
            akita_types::NextWitnessBindingPolicy::TerminalInnerState,
        ),
        |next| {
            (
                fold::FoldSuccessorParams::Recursive(next),
                akita_types::NextWitnessBindingPolicy::OuterPayload,
            )
        },
    );
    let prepared = prepare_fold::<Cfg::Field, Cfg::ExtField, B>(
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
    let root = prove_fold::<Cfg::Field, Cfg::ExtField, B>(
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
    let suffix = suffix::prove_suffix::<Cfg, B>(
        expanded,
        prefix_slots,
        backend,
        &mut grinding,
        root.next_state,
        schedule,
        proof_session,
    )?;
    if suffix.num_levels != schedule.num_fold_levels() {
        return Err(AkitaError::InvalidProof);
    }
    guard.finish()?;
    grinding.finish()
}
