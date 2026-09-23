use super::*;
use crate::SelectedProverOpeningData;
use akita_config::TrustedScheduleCatalog;
use jolt_field::AdditiveGroup;

/// Prove an ordered statement using reusable commitments retained by one backend.
#[allow(clippy::too_many_arguments)]
pub fn batched_prove<'a, Cfg, T, B>(
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    schedules: &TrustedScheduleCatalog<Cfg>,
    backend: &B,
    opening: SelectedProverOpeningData<'a, Cfg::ExtField, B::CommitmentHandle, Cfg::Field>,
    transcript: &mut T,
    basis: BasisMode,
) -> Result<AkitaBatchedProof<Cfg::Field, Cfg::ExtField>, AkitaError>
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
    T: Transcript<Cfg::Field> + TranscriptChallengePreview,
    B: ProverBackend<Cfg::Field, Cfg::ExtField>,
{
    let (selection, claims) = opening.into_low_level_parts();
    let resolved = schedules.resolve_selection(selection)?;
    let layout = claims.opening_layout();
    resolved.validate_opening_layout(layout)?;
    let schedule = resolved.schedule();
    schedule.validate_nonterminal_opening_execution(Cfg::EXT_DEGREE)?;
    let session = backend.begin_proof(expanded, schedule, layout)?;
    let guard = crate::backend::ProofScope::admitted(backend, session);
    let session = guard.session();
    let context = backend.proof_context(session, 0)?;
    let mut material = Vec::with_capacity(layout.num_groups());
    for index in 0..layout.num_groups() {
        let group = schedule.root.params.group_params(layout, index)?;
        material.push(backend.validate_commitment(
            session,
            &context.for_group(index),
            claims.group(index)?,
            &group.profile,
            claims.opening_claims().group_commitment(index)?,
        )?);
    }
    let grinding = bind_transcript_instance_descriptor::<Cfg::Field, T, Cfg>(
        expanded, layout, selection, schedule, basis, transcript,
    )?;
    let mut transcript = akita_types::ProverGrindingTranscript::<T>::new(transcript, &grinding)?;
    claims.append_to_transcript(&schedule.root.params, &mut transcript)?;
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
    let prepared = prepare_fold::<Cfg::Field, Cfg::ExtField, _, B>(
        backend,
        root_claims,
        material,
        false,
        &mut transcript,
        0,
        &schedule.root.params,
        basis,
        session,
        schedule.root.output_witness_len,
        next_params.inner_ring_dimension(),
        false,
    )?;
    let root = prove_fold::<Cfg::Field, Cfg::ExtField, _, B>(
        expanded,
        prefix_slots,
        backend,
        session,
        &mut transcript,
        0,
        &schedule.root.params,
        next_params,
        schedule.root.output_witness_len,
        next_binding,
        prepared,
    )?;
    let suffix = suffix::prove_suffix::<Cfg, _, B>(
        expanded,
        prefix_slots,
        backend,
        &mut transcript,
        root.next_state,
        schedule,
        session,
    )?;
    let proof = AkitaBatchedProof {
        nonce_stream: transcript.finish()?,
        root: root.level_proof,
        recursive_folds: suffix.recursive_folds,
        terminal: suffix.terminal,
    };
    guard.finish()?;
    Ok(proof)
}
