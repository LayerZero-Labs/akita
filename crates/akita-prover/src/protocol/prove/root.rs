use super::*;
use crate::SelectedProverOpeningData;
use akita_config::TrustedScheduleCatalog;
use jolt_field::AdditiveGroup;

#[allow(clippy::type_complexity)]
pub(super) fn resolve_root<'a, 's, Cfg, H>(
    expanded: &akita_types::AkitaSetupDescriptor,
    schedules: &'s TrustedScheduleCatalog<Cfg>,
    opening: SelectedProverOpeningData<'a, Cfg::ExtField, H, Cfg::Field>,
    basis: BasisMode,
) -> Result<
    (
        &'s akita_config::ResolvedScheduleRow,
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
    let (selection, claims) = opening.into_low_level_parts();
    let resolved = schedules.resolve_selection(selection)?;
    resolved.validate_opening_layout(claims.opening_layout())?;
    let (grinding_plan, descriptor_bytes) =
        akita_config::transcript_instance_descriptor::<Cfg::Field, Cfg>(
            expanded,
            claims.opening_layout(),
            selection,
            resolved.schedule(),
            basis,
        )?;
    Ok((resolved, claims, grinding_plan, descriptor_bytes))
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub(super) fn prove_root<Cfg, B>(
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    backend: &B,
    claims: &akita_types::OpeningClaims<'_, Cfg::ExtField, akita_types::Commitment<Cfg::Field>>,
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
    root_claims.append_to(&schedule.root.params, grinding)?;
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
    let prepared = prepare_fold::<Cfg::Field, Cfg::ExtField, B>(
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
    prove_fold::<Cfg, B>(
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
        handoff,
    )
}
