use super::*;
use crate::SelectedProverOpeningData;
use akita_config::TrustedScheduleCatalog;
use jolt_algebra::AdditiveGroup;

#[allow(clippy::type_complexity)]
pub(super) fn resolve_root<'a, 's, Cfg, H>(
    expanded: &akita_types::AkitaSetupDescriptor,
    schedules: &'s TrustedScheduleCatalog<Cfg>,
    opening: SelectedProverOpeningData<'a, Cfg::ExtField, H, Cfg::Field>,
    basis: BasisMode,
    setup_batch_level: Option<u32>,
) -> Result<
    (
        &'s akita_config::ResolvedScheduleRow,
        Vec<akita_transcript::CommitmentDigest>,
        ProverOpeningData<'a, Cfg::ExtField, H, Cfg::Field>,
        akita_params::GrindingPlan,
        Vec<u8>,
    ),
    AkitaError,
>
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
    H: crate::CommitmentHandleMetadata,
{
    let (selection, digests, claims) = opening.into_low_level_parts();
    let resolved = schedules.resolve_selection(selection)?;
    resolved.validate_opening_layout(claims.opening_layout())?;
    let (grinding_plan, descriptor_bytes) =
        akita_config::transcript_instance_descriptor::<Cfg::Field, Cfg>(
            expanded,
            claims.opening_layout(),
            selection,
            resolved.schedule(),
            basis,
            setup_batch_level,
        )?;
    Ok((resolved, digests, claims, grinding_plan, descriptor_bytes))
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub(super) fn prove_root<Cfg, B>(
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    backend: &B,
    claims: &akita_types::OpeningClaims<'_, Cfg::ExtField, akita_types::Commitment<Cfg::Field>>,
    digests: &[akita_transcript::CommitmentDigest],
    layout: &OpeningClaimsLayout,
    handles: &[B::CommitmentHandle],
    schedule: &FoldSchedule,
    proof_session: &B::ProofSessionHandle,
    grinding: &mut akita_types::ProverGrinding<'_>,
    basis: BasisMode,
    handoff: &mut super::registry::FoldHandoff<'_, '_, '_, Cfg>,
) -> Result<
    ProveLevelOutput<Cfg::Field, Cfg::ExtField, B::CommitmentMaterialHandle, B::WitnessHandle>,
    AkitaError,
>
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
    B: ProverBackend<Cfg::Field, Cfg::ExtField> + 'static,
{
    let setup_batch = handoff
        .request
        .setup_batch
        .filter(|(target, _)| *target as usize == 0)
        .map(|(_, statement)| statement);
    prove_root_with::<Cfg, B, akita_transcript::ProverChannel>(
        expanded,
        prefix_slots,
        backend,
        claims,
        digests,
        layout,
        handles,
        schedule,
        proof_session,
        grinding,
        basis,
        setup_batch,
        |relation, commitment, binding, witness, material| {
            handoff.handoff_successor(
                backend,
                proof_session,
                0,
                &schedule.root.params,
                relation,
                commitment,
                binding,
                witness,
                material,
            )
        },
        &mut fold::RawPayload,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub(super) fn prove_root_with<Cfg, B, C>(
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    backend: &B,
    claims: &akita_types::OpeningClaims<'_, Cfg::ExtField, akita_types::Commitment<Cfg::Field>>,
    digests: &[akita_transcript::CommitmentDigest],
    layout: &OpeningClaimsLayout,
    handles: &[B::CommitmentHandle],
    schedule: &FoldSchedule,
    proof_session: &B::ProofSessionHandle,
    grinding: &mut akita_types::ProverGrinding<'_, C>,
    basis: BasisMode,
    setup_batch: Option<&akita_types::setup_claims::SetupBatchStatement<Cfg::ExtField>>,
    handoff: impl FnOnce(
        &RingRelationInstance<Cfg::Field>,
        &crate::backend::ValidatedRecursiveWitnessCommitPlan,
        &NextWitnessState<Cfg::Field>,
        &B::WitnessHandle,
        &B::CommitmentMaterialHandle,
    ) -> Result<(), AkitaError>,
    payload: &mut impl fold::SuccessorPayload<Cfg::Field>,
) -> Result<
    ProveLevelOutput<Cfg::Field, Cfg::ExtField, B::CommitmentMaterialHandle, B::WitnessHandle>,
    AkitaError,
>
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
    B: ProverBackend<Cfg::Field, Cfg::ExtField> + 'static,
    C: akita_types::ProverGrindingChannel + akita_transcript::PrivateProofChannel<Cfg::Field>,
{
    let root_claims = ProverOpeningData::from_parts(
        claims.clone(),
        layout.clone(),
        handles.iter().map(OpeningSource::Commitment).collect(),
    )?;
    let context = backend.proof_context(proof_session, 0)?;
    let mut material = Vec::with_capacity(layout.num_groups());
    for (index, handle) in handles.iter().enumerate() {
        let group = schedule.root.params.group_params(layout, index)?;
        material.push(backend.validate_commitment(
            proof_session,
            &context.for_group(index),
            handle,
            &group.profile,
            claims.group_commitment(index)?,
        )?);
    }
    root_claims.append_to(&schedule.root.params, digests, grinding)?;
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
    let prepared = prepare_fold::<Cfg::Field, Cfg::ExtField, B, C>(
        backend,
        root_claims,
        material,
        false,
        grinding,
        0,
        &schedule.root.params,
        basis,
        proof_session,
        schedule.root.output_witness_len,
        next_params.inner_ring_dimension(),
        false,
    )?;
    prove_fold::<Cfg::Field, Cfg::ExtField, B, C>(
        expanded,
        prefix_slots,
        backend,
        proof_session,
        grinding,
        0,
        &schedule.root.params,
        next_params,
        schedule.root.output_witness_len,
        next_binding,
        prepared,
        setup_batch,
        handoff,
        payload,
    )
}
