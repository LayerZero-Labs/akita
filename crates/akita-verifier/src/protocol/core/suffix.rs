use super::*;
use akita_types::OpeningClaimsLayout;

pub(super) struct NativeSuffixVerifierState<F: Field, E: Field> {
    pub opening_point: Vec<E>,
    pub opening: E,
    pub witness: RingVec<F>,
    pub basis: BasisMode,
    pub witness_len: usize,
    pub setup_prefix_opening: Option<SetupPrefixOpening<E>>,
}

fn suffix_commitment_payloads<F, E>(
    setup: &AkitaVerifierSetup<F>,
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    witness_commitment: &RingVec<F>,
) -> Result<Vec<RingVec<F>>, AkitaError>
where
    F: Field,
    E: ExtField<F>,
{
    let mut group_payloads = Vec::with_capacity(opening_batch.num_groups());
    if let Some(setup_prefix) = lp.setup_prefix() {
        let setup_prefix_id = setup_prefix.slot_id().ok_or_else(|| {
            AkitaError::InvalidSetup("selected setup-prefix group has no slot identity".to_string())
        })?;
        let slot = setup.prefix_slots().get(&setup_prefix_id).ok_or_else(|| {
            AkitaError::InvalidSetup(
                "planned setup-prefix slot is missing from verifier setup".to_string(),
            )
        })?;
        let mut coeffs = Vec::new();
        for row in &slot.commitment.rows {
            coeffs.extend_from_slice(row.coeffs());
        }
        group_payloads.push(RingVec::from_coeffs(coeffs));
    }
    group_payloads.push(RingVec::from_coeffs(witness_commitment.coeffs().to_vec()));
    if group_payloads.len() != opening_batch.num_groups() {
        return Err(AkitaError::InvalidProof);
    }

    let relation_geometry =
        RelationWitnessGeometry::for_level(lp, opening_batch, <E as ExtField<F>>::DEGREE)?;
    let relation_layout = relation_geometry.rhs_layout();
    let mut ordered = Vec::with_capacity(group_payloads.len());
    for (relation_group_index, group_index) in
        opening_batch.root_group_order()?.into_iter().enumerate()
    {
        let payload = group_payloads
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if payload.coeff_len()
            != relation_layout
                .group_payload_geometry(relation_group_index)?
                .transmitted_coefficients()
        {
            return Err(AkitaError::InvalidProof);
        }
        ordered.push(payload.clone());
    }
    Ok(ordered)
}

#[allow(clippy::too_many_arguments)]
fn prepare_fold_replay_native<'a, F, E>(
    setup: &'a AkitaVerifierSetup<F>,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    level: u32,
    current_state: &NativeSuffixVerifierState<F, E>,
    lp: &'a CommittedGroupParams,
    output_witness_len: usize,
    next_params: Option<&'a FoldParams>,
    terminal: &TerminalFoldParams,
) -> Result<NativePreparedFoldReplay<'a, F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    let payload_geometry = lp.outer_payload_geometry()?;
    if current_state.witness.coeff_len() != payload_geometry.transmitted_coefficients() {
        return Err(AkitaError::InvalidProof);
    }
    akita_transcript::public_native_fields_verifier(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
            level,
            stage: 3,
            detail: u32::try_from(payload_geometry.transcript_ring_dimension())
                .map_err(|_| AkitaError::InvalidProof)?,
            ..akita_transcript::ProtocolSiteId::default()
        },
        current_state.witness.coeffs(),
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    let recursive_num_vars = lp.recursive_opening_num_vars()?;
    if current_state.opening_point.len() > recursive_num_vars {
        return Err(AkitaError::InvalidProof);
    }
    let block_claims = match (&current_state.setup_prefix_opening, lp.setup_prefix()) {
        (Some((setup_prefix_point, setup_prefix_eval)), Some(_)) => {
            OpeningClaims::from_groups(vec![
                PolynomialGroupClaims::new(
                    setup_prefix_point.clone(),
                    vec![*setup_prefix_eval],
                    (),
                )?,
                PolynomialGroupClaims::new(
                    current_state.opening_point.clone(),
                    vec![current_state.opening],
                    (),
                )?,
            ])?
        }
        (None, None) => OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            current_state.opening_point.clone(),
            vec![current_state.opening],
            (),
        )?])?,
        _ => return Err(AkitaError::InvalidProof),
    };
    let opening_batch = block_claims.layout()?;
    let openings = (0..opening_batch.num_groups())
        .flat_map(|group_index| {
            block_claims
                .group_evaluations(group_index)
                .map(|values| values.to_vec())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let group_points = (0..opening_batch.num_groups())
        .map(|group_index| block_claims.group_point(group_index))
        .collect::<Result<Vec<_>, _>>()?;
    let material = if matches!(
        lp.opening_method(),
        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
    ) {
        verify_coefficient_packing_suffix_prefix_native::<F, E>(
            &block_claims,
            &openings,
            &opening_batch,
            current_state.basis,
            lp,
            grinding,
            level,
        )?
    } else if const { <E as ExtField<F>>::DEGREE == 1 } {
        let prepared =
            prepare_single_field_suffix_groups::<F, E>(&block_claims, lp, &opening_batch)?;
        for (group_index, point) in group_points.iter().enumerate() {
            akita_transcript::public_native_extensions_verifier::<F, E>(
                grinding.state_mut(),
                akita_transcript::ProtocolSiteId {
                    family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
                    level,
                    stage: 1,
                    group: u32::try_from(group_index).map_err(|_| AkitaError::InvalidProof)?,
                    ..akita_transcript::ProtocolSiteId::default()
                },
                point,
            )
            .map_err(|_| AkitaError::InvalidProof)?;
        }
        akita_transcript::public_native_extensions_verifier::<F, E>(
            grinding.state_mut(),
            akita_transcript::ProtocolSiteId {
                family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
                level,
                stage: 2,
                ..akita_transcript::ProtocolSiteId::default()
            },
            &openings,
        )
        .map_err(|_| AkitaError::InvalidProof)?;
        FoldClaimMaterial {
            prepared_points: prepared
                .into_iter()
                .map(PreparedFoldOpeningPoint::EvaluationTrace)
                .collect(),
            openings: openings.clone(),
            reduction_final_claims: None,
            reduction_factors: None,
        }
    } else {
        verify_extension_claim_suffix_prefix_native::<F, E>(
            &group_points,
            &openings,
            &opening_batch,
            current_state.basis,
            lp,
            grinding,
            level,
        )?
    };
    let relation_geometry = RelationWitnessGeometry::for_level(lp, &opening_batch, E::DEGREE)?;
    let opening_geometry = relation_geometry.rhs_layout().opening_payload_geometry()?;
    let opening_payload = akita_transcript::receive_native_field_group::<F>(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_OPENING_PAYLOAD,
            level,
            detail: u32::try_from(opening_geometry.transcript_ring_dimension())
                .map_err(|_| AkitaError::InvalidProof)?,
            ..akita_transcript::ProtocolSiteId::default()
        },
        opening_geometry.transmitted_coefficients(),
    )
    .map(RingVec::from_coeffs)
    .map_err(|_| AkitaError::InvalidProof)?;
    let prefix = finalize_native_claims::<F, E>(&opening_batch, material, grinding, level)?;
    let commitment_payloads =
        suffix_commitment_payloads::<F, E>(setup, lp, &opening_batch, &current_state.witness)?;
    let (next_witness, next_witness_ring_dim, next_opening_source_len, stage3) =
        if let Some(next) = next_params {
            let ring_dim = next.params.d_a();
            let coefficient_count = next
                .params
                .outer_payload_geometry()?
                .transmitted_coefficients();
            let committed_len =
                akita_types::witness_commitment_domain_len(output_witness_len, ring_dim)?;
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
            let committed_len =
                akita_types::witness_commitment_domain_len(output_witness_len, ring_dim)?;
            (
                NativeNextWitnessPlan::TerminalT { coefficient_count },
                ring_dim,
                committed_len / ring_dim,
                None,
            )
        };
    let challenge_field_bits = F::MODULUS_BITS
        .checked_mul(
            u32::try_from(E::DEGREE)
                .map_err(|_| AkitaError::InvalidSetup("extension degree overflow".into()))?,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("challenge field width overflow".into()))?;
    let level_layout = akita_types::native_nonterminal_level_layout(
        F::MODULUS_BITS,
        challenge_field_bits,
        lp,
        lp.relation_address_geometry(
            &opening_batch,
            E::DEGREE,
            next_witness_ring_dim,
            output_witness_len,
        )?,
        next_params.map(|next| &next.params),
    )?;
    Ok(NativePreparedFoldReplay {
        lp,
        level,
        opening_payload,
        opening_shape: opening_batch,
        commitment_payloads,
        prefix,
        w_len: output_witness_len,
        level_layout,
        next_witness,
        next_witness_ring_dim,
        next_opening_source_len,
        stage3,
        evaluation_trace_basis: current_state.basis,
    })
}

pub(super) fn verify_suffix_native<F, E>(
    setup: &AkitaVerifierSetup<F>,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    schedule: &FoldSchedule,
    mut current_state: NativeSuffixVerifierState<F, E>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    for (offset, step) in schedule.recursive_folds.iter().enumerate() {
        let level = u32::try_from(offset + 1).map_err(|_| AkitaError::InvalidProof)?;
        if current_state.witness_len != step.input_witness_len {
            return Err(AkitaError::InvalidProof);
        }
        let next = schedule.recursive_folds.get(offset + 1);
        let prepared = prepare_fold_replay_native::<F, E>(
            setup,
            grinding,
            level,
            &current_state,
            &step.params,
            step.output_witness_len,
            next,
            &schedule.terminal,
        )?;
        let output = verify_fold_native(setup, grinding, prepared)?;
        current_state = NativeSuffixVerifierState {
            opening_point: output.challenges,
            opening: output.opening,
            witness: output.next_witness,
            basis: BasisMode::Lagrange,
            witness_len: step.output_witness_len,
            setup_prefix_opening: output.setup_prefix_opening,
        };
    }
    verify_terminal_suffix_native(
        setup,
        grinding,
        u32::try_from(schedule.recursive_folds.len() + 1).map_err(|_| AkitaError::InvalidProof)?,
        &current_state,
        &schedule.terminal,
    )
}

fn verify_terminal_suffix_native<F, E>(
    setup: &AkitaVerifierSetup<F>,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    level: u32,
    current_state: &NativeSuffixVerifierState<F, E>,
    scheduled: &TerminalFoldParams,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    if current_state.witness_len != scheduled.input_witness_len
        || current_state.setup_prefix_opening.is_some()
    {
        return Err(AkitaError::InvalidProof);
    }
    let group = scheduled
        .response_shape
        .layout
        .groups
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    if scheduled.response_shape.layout.groups.len() != 1
        || current_state.witness.coeff_len() != group.t_field_elems
        || !current_state.witness.can_decode_vec(scheduled.d_a())
    {
        return Err(AkitaError::InvalidProof);
    }
    akita_transcript::public_native_fields_verifier(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_TERMINAL,
            level,
            stage: 2,
            ..akita_transcript::ProtocolSiteId::default()
        },
        current_state.witness.coeffs(),
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    let recursive_num_vars = scheduled.recursive_opening_num_vars()?;
    if current_state.opening_point.len() > recursive_num_vars {
        return Err(AkitaError::InvalidProof);
    }
    let opening_batch = OpeningClaimsLayout::new(current_state.opening_point.len(), 1)?;
    let (prepared_point, protocol_point, final_relation) = if const { <E as ExtField<F>>::DEGREE == 1 }
    {
        let prepared = akita_types::dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            scheduled.d_a(),
            |D| {
                prepare_opening_point::<F, E, D>(
                    &current_state.opening_point,
                    current_state.basis,
                    scheduled.blocks.positions_per_block,
                    scheduled.blocks.live_blocks,
                    scheduled.d_a().trailing_zeros() as usize,
                )
            }
        )?;
        (prepared, current_state.opening_point.clone(), None)
    } else {
        let replay = verify_extension_claim_terminal_suffix_native::<F, E>(
            &current_state.opening_point,
            current_state.opening,
            &opening_batch,
            current_state.basis,
            scheduled,
            grinding,
            level,
        )?;
        let group = replay
            .groups
            .into_iter()
            .next()
            .ok_or(AkitaError::InvalidProof)?;
        (group.prepared, group.protocol, replay.final_relation)
    };
    akita_transcript::public_native_extensions_verifier::<F, E>(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
            level,
            stage: 5,
            ..akita_transcript::ProtocolSiteId::default()
        },
        &protocol_point,
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    if final_relation.is_none() {
        akita_transcript::public_native_extensions_verifier::<F, E>(
            grinding.state_mut(),
            akita_transcript::ProtocolSiteId {
                family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
                level,
                stage: 4,
                ..akita_transcript::ProtocolSiteId::default()
            },
            std::slice::from_ref(&current_state.opening),
        )
        .map_err(|_| AkitaError::InvalidProof)?;
    }
    let row_coefficients = akita_types::verify_row_coefficients_native::<F, E>(
        &opening_batch,
        akita_types::GrindingSite::EvaluationBatch { level },
        grinding,
    )?;
    if row_coefficients.as_slice() != [E::one()] {
        return Err(AkitaError::InvalidProof);
    }
    let e_fields = akita_transcript::receive_native_field_group::<F>(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_TERMINAL,
            level,
            stage: 1,
            ..akita_transcript::ProtocolSiteId::default()
        },
        group.e_field_elems,
    )
    .map(RingVec::from_coeffs)
    .map_err(|_| AkitaError::InvalidProof)?;
    grinding.read_fold_response(akita_types::GrindingSite::FoldResponse { level })?;
    let operator_rejection = if scheduled.response_l2_sq_cap().is_some() {
        Some(
            akita_challenges::selective_l2_operator_norm_rejection(
                scheduled.d_a(),
                &scheduled.fold_challenge_config,
            )
            .ok_or(AkitaError::InvalidProof)?,
        )
    } else {
        None
    };
    let challenges = {
        let mut draw =
            akita_challenges::NativeVerifierFoldDraw::new(grinding.state_mut(), level, 0);
        draw.draw_folding_challenges_with_rejection(
            akita_challenges::FoldChallengeDrawDomain::EvaluationTrace,
            scheduled.d_a(),
            0,
            scheduled.blocks.live_blocks,
            1,
            &scheduled.fold_challenge_config,
            operator_rejection,
        )?
    };
    grinding.record_fold_challenges(level, 0, scheduled.blocks.live_blocks)?;
    let z_payload = akita_transcript::receive_native_bounded_bytes(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_TERMINAL,
            level,
            round: 3,
            ..akita_transcript::ProtocolSiteId::default()
        },
        group.z_payload_bytes,
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    scheduled.validate_terminal_linf_cap(group.z_linf_cap)?;
    let terminal_response = akita_types::TerminalResponse {
        layout: scheduled.response_shape.layout.clone(),
        z_payloads: vec![z_payload],
        e_fields,
        t_fields: current_state.witness.clone(),
    };
    super::terminal_direct::verify_terminal_ring_relations(
        setup,
        &challenges,
        &prepared_point.ring_multiplier_point,
        scheduled,
        &terminal_response,
    )?;
    let (target, scale) = match final_relation {
        Some((claims, factors)) => (
            *claims.first().ok_or(AkitaError::InvalidProof)?,
            *factors.first().ok_or(AkitaError::InvalidProof)?,
        ),
        None => (current_state.opening, E::one()),
    };
    super::terminal_direct::verify_terminal_trace(
        &prepared_point.ring_multiplier_point,
        scheduled,
        &terminal_response,
        &prepared_point,
        &[E::one()],
        None,
        scale,
        target,
    )
}
