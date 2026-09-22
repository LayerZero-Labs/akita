use super::*;
use akita_types::Commitment;

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_root_native<F, E>(
    setup: &AkitaVerifierSetup<F>,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    claims: &OpeningClaims<'_, E, &Commitment<F>>,
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    root_lp: &CommittedGroupParams,
    next_fold_params: Option<&FoldParams>,
    terminal: &TerminalFoldParams,
) -> Result<NativeFoldVerifyOutput<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    root_lp.validate_opening_batch(opening_batch)?;
    for group_index in 0..opening_batch.num_groups() {
        if !matches!(
            root_lp
                .group_params_geometry(opening_batch, group_index)?
                .opening_method(),
            akita_types::OpeningMethod::SubringCoefficientPacking { .. }
        ) {
            return Err(AkitaError::InvalidProof);
        }
    }
    let relation_geometry = RelationWitnessGeometry::for_level(root_lp, opening_batch, E::DEGREE)?;
    let relation_layout = relation_geometry.rhs_layout();
    for group_index in 0..opening_batch.num_groups() {
        let commitment = claims.group_commitment(group_index)?;
        let plan = relation_layout.compression_plan_for_group(group_index)?;
        if commitment.rows().coeff_len() != plan.terminal_coefficients() {
            return Err(AkitaError::InvalidProof);
        }
        let ring_dim = plan
            .maps()
            .last()
            .ok_or(AkitaError::InvalidProof)?
            .ring_dimension();
        akita_transcript::public_native_fields_verifier(
            grinding.state_mut(),
            akita_transcript::ProtocolSiteId {
                family: akita_transcript::SITE_FAMILY_ROOT_STATEMENT,
                stage: 1,
                group: u32::try_from(group_index).map_err(|_| AkitaError::InvalidProof)?,
                detail: u32::try_from(ring_dim).map_err(|_| AkitaError::InvalidProof)?,
                ..akita_transcript::ProtocolSiteId::default()
            },
            commitment.rows().coeffs(),
        )
        .map_err(|_| AkitaError::InvalidProof)?;
    }
    for (group_index, group) in claims.groups().iter().enumerate() {
        akita_transcript::public_native_extensions_verifier::<F, E>(
            grinding.state_mut(),
            akita_transcript::ProtocolSiteId {
                family: akita_transcript::SITE_FAMILY_ROOT_STATEMENT,
                stage: 2,
                group: u32::try_from(group_index).map_err(|_| AkitaError::InvalidProof)?,
                ..akita_transcript::ProtocolSiteId::default()
            },
            group.point(),
        )
        .map_err(|_| AkitaError::InvalidProof)?;
    }
    let openings = claims.flat_evaluations();
    let material = verify_coefficient_packing_root_prefix::<F, E>(
        claims,
        &openings,
        opening_batch,
        basis,
        root_lp,
    )?;
    akita_transcript::public_native_extensions_verifier::<F, E>(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
            stage: 2,
            ..akita_transcript::ProtocolSiteId::default()
        },
        &openings,
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    let payload_geometry = relation_layout.opening_payload_geometry()?;
    let opening_payload = akita_transcript::receive_native_field_group::<F>(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_OPENING_PAYLOAD,
            detail: u32::try_from(payload_geometry.transcript_ring_dimension())
                .map_err(|_| AkitaError::InvalidProof)?,
            ..akita_transcript::ProtocolSiteId::default()
        },
        payload_geometry.transmitted_coefficients(),
    )
    .map(RingVec::from_coeffs)
    .map_err(|_| AkitaError::InvalidProof)?;
    let prefix = finalize_native_claims::<F, E>(opening_batch, material, grinding, 0)?;
    let order = opening_batch.root_group_order()?;
    let commitment_payloads = order
        .into_iter()
        .map(|group_index| Ok(claims.group_commitment(group_index)?.rows().clone()))
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let witness_len = root_lp.output_witness_len::<F>(opening_batch, E::DEGREE)?;
    let (next_witness, next_witness_ring_dim, next_opening_source_len, stage3) =
        if let Some(next) = next_fold_params {
            let ring_dim = next.params.d_a();
            let coefficient_count = next
                .params
                .outer_payload_geometry()?
                .transmitted_coefficients();
            let committed_len = akita_types::witness_commitment_domain_len(witness_len, ring_dim)?;
            (
                NativeNextWitnessPlan::OuterPayload { coefficient_count },
                ring_dim,
                committed_len / ring_dim,
                matches!(
                    next.predecessor_setup_contribution_mode(),
                    SetupContributionMode::Recursive
                )
                .then_some(&next.params),
            )
        } else {
            let ring_dim = terminal.d_a();
            let coefficient_count = terminal
                .response_shape
                .layout
                .groups
                .first()
                .ok_or(AkitaError::InvalidProof)?
                .t_field_elems;
            let committed_len = akita_types::witness_commitment_domain_len(witness_len, ring_dim)?;
            (
                NativeNextWitnessPlan::TerminalT { coefficient_count },
                ring_dim,
                committed_len / ring_dim,
                None,
            )
        };
    verify_fold_native(
        setup,
        grinding,
        NativePreparedFoldReplay {
            lp: root_lp,
            level: 0,
            opening_payload,
            opening_shape: opening_batch.clone(),
            commitment_payloads,
            prefix,
            w_len: witness_len,
            next_witness,
            next_witness_ring_dim,
            next_opening_source_len,
            stage3,
            evaluation_trace_basis: basis,
        },
    )
}
