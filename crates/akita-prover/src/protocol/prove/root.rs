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
            |plan| {
                claims
                    .opening_claims()
                    .root_grinding_plan(plan, resolved.schedule())
            },
        )?;
    Ok((resolved, claims, grinding_plan, descriptor_bytes))
}

/// Extension seam: dispatch a root by its declared group relation kind.
/// Tracking issue: https://github.com/LayerZero-Labs/akita/issues/285.
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
    match claims.groups().first().map(|group| group.kind()) {
        #[cfg(feature = "labinius-trinomial")]
        Some(akita_types::opening_claims::GroupClaimKind::Binary { .. }) => {
            super::binary_root::prove_root(
                expanded,
                prefix_slots,
                backend,
                claims,
                layout,
                handles,
                schedule,
                proof_session,
                grinding,
                basis,
                handoff,
            )
        }
        Some(akita_types::opening_claims::GroupClaimKind::Ordinary) | None => {
            super::ordinary_root::prove_root(
                expanded,
                prefix_slots,
                backend,
                claims,
                layout,
                handles,
                schedule,
                proof_session,
                grinding,
                basis,
                handoff,
            )
        }
        // A dependency may expose another kind without enabling this crate's handler.
        #[allow(unreachable_patterns)]
        _ => Err(AkitaError::InvalidInput(
            "root group kind is not enabled in this protocol engine".into(),
        )),
    }
}
