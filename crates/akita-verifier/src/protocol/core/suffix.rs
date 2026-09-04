use super::*;
use akita_types::OpeningClaimsLayout;

/// Verifier state carried between suffix fold levels.
pub(super) struct SuffixVerifierState<'a, F: Field, E: Field> {
    /// Current opening point for the committed suffix witness.
    pub opening_point: Vec<E>,
    /// Claimed opening value for the current commitment.
    pub opening: E,
    pub witness: SuffixWitnessState<'a, F>,
    /// Basis used to interpret the current opening point.
    pub basis: BasisMode,
    /// Current suffix witness length in field elements.
    pub witness_len: usize,
    /// Optional setup-prefix opening carried from the previous stage-3 proof.
    pub setup_prefix_opening: Option<SetupPrefixOpening<E>>,
}

pub(super) enum SuffixWitnessState<'a, F: Field> {
    Commitment(&'a RingVec<F>),
    TerminalT(Vec<u8>),
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
        let slot = setup.prefix_slots.get(&setup_prefix_id).ok_or_else(|| {
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

struct FoldReplayPayload<'a, F: Field, E: Field> {
    extension_opening_reduction: Option<&'a ExtensionOpeningReductionProof<E>>,
    kind: FoldReplayKind<'a, F, E>,
}

enum FoldReplayKind<'a, F: Field, E: Field> {
    Recursive {
        v: &'a RingVec<F>,
        stage1: &'a AkitaStage1Proof<E>,
        stage2: &'a AkitaStage2Proof<F, E>,
        next_witness: PreparedNextWitness<'a, F>,
        next_witness_ring_dim: usize,
        stage3: Option<(&'a SetupSumcheckProof<E>, &'a CommittedGroupParams)>,
    },
}

/// Prepare one suffix fold level for relation verification.
///
/// Terminal levels absorb the cleartext final witness instead of a
/// next-witness commitment and run direct consistency/A and trace checks.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, the public trace check
/// fails, or the terminal witness replay is malformed.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_suffix<'a, F, E, T>(
    recursive_folds: &'a [FoldLevelProof<F, E>],
    terminal: &'a TerminalLevelProof<F, E>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    schedule: &FoldSchedule,
    mut current_state: SuffixVerifierState<'a, F, E>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
    T: akita_types::VerifierTranscriptGrinding<F>,
{
    for (offset, fold) in recursive_folds.iter().enumerate() {
        let level_index = offset + 1;
        let step = schedule
            .recursive_folds
            .get(offset)
            .ok_or(AkitaError::InvalidProof)?;
        if current_state.witness_len != step.input_witness_len {
            return Err(AkitaError::InvalidProof);
        }
        let current_lp = &step.params;
        let next_step = schedule.recursive_folds.get(offset + 1);
        let next_params = next_step.map(|next| &next.params);
        let next_witness_ring_dim =
            next_params.map_or(schedule.terminal.d_a(), CommittedGroupParams::d_a);
        let current_commitment = match &current_state.witness {
            SuffixWitnessState::Commitment(commitment) => *commitment,
            SuffixWitnessState::TerminalT(_) => return Err(AkitaError::InvalidProof),
        };
        if current_commitment.coeff_len()
            != current_lp
                .outer_payload_geometry()?
                .transmitted_coefficients()
        {
            return Err(AkitaError::InvalidProof);
        }

        let next_t_state = if next_step.is_none() {
            let witness = terminal.terminal_response();
            let t_state = raw_field_segment_bytes(&witness.t_fields)?;
            if t_state.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            Some(t_state)
        } else {
            None
        };
        let next_witness = match (fold.next_w_payload(), next_t_state.as_deref()) {
            (Some(commitment), None) => {
                let next_params = next_params.ok_or(AkitaError::InvalidProof)?;
                let ring_dim = next_params
                    .outer_payload_geometry()?
                    .transcript_ring_dimension();
                PreparedNextWitness::Commitment {
                    commitment,
                    ring_dim,
                }
            }
            (None, Some(t_state)) if !t_state.is_empty() => PreparedNextWitness::TerminalT(t_state),
            _ => return Err(AkitaError::InvalidProof),
        };
        let setup_contribution_mode = next_step.map_or(SetupContributionMode::Direct, |step| {
            step.predecessor_setup_contribution_mode()
        });
        let stage3 = fold.stage3_for_mode(setup_contribution_mode, next_params)?;
        let prepared = prepare_fold_replay::<F, E, T>(
            FoldReplayPayload {
                extension_opening_reduction: fold.extension_opening_reduction(),
                kind: FoldReplayKind::Recursive {
                    v: &fold.opening_payload,
                    stage1: &fold.stage1,
                    stage2: &fold.stage2,
                    next_witness,
                    next_witness_ring_dim,
                    stage3,
                },
            },
            setup,
            transcript,
            u32::try_from(level_index).map_err(|_| AkitaError::InvalidProof)?,
            &current_state,
            current_lp,
            step.output_witness_len,
        )?;
        let (challenges, setup_prefix_opening) =
            verify_fold::<F, E, T>(setup, transcript, prepared).map_err(|err| {
                AkitaError::InvalidInput(format!(
                    "suffix verify level {level_index} failed: {err:?}"
                ))
            })?;

        let next_commitment = fold.next_w_payload();
        let next_witness = match (next_commitment, next_t_state) {
            (Some(commitment), None) => SuffixWitnessState::Commitment(commitment),
            (None, Some(t_state)) => SuffixWitnessState::TerminalT(t_state),
            _ => return Err(AkitaError::InvalidProof),
        };
        current_state = SuffixVerifierState {
            opening_point: challenges,
            opening: fold.next_w_eval(),
            witness: next_witness,
            basis: BasisMode::Lagrange,
            witness_len: step.output_witness_len,
            setup_prefix_opening,
        };
    }

    let terminal_level = recursive_folds.len() + 1;
    if current_state.witness_len != schedule.terminal.input_witness_len {
        return Err(AkitaError::InvalidProof);
    }
    if !matches!(&current_state.witness, SuffixWitnessState::TerminalT(_)) {
        return Err(AkitaError::InvalidProof);
    }
    if terminal.terminal_response().num_elems()
        != schedule.terminal.response_shape.logical_num_elems()
    {
        return Err(AkitaError::InvalidProof);
    }
    verify_terminal_suffix::<F, E, T>(
        terminal,
        setup,
        transcript,
        u32::try_from(terminal_level).map_err(|_| AkitaError::InvalidProof)?,
        &current_state,
        &schedule.terminal,
    )
    .map_err(|err| {
        AkitaError::InvalidInput(format!(
            "suffix verify level {terminal_level} failed: {err:?}"
        ))
    })
}

fn verify_terminal_suffix<F, E, T>(
    proof: &TerminalLevelProof<F, E>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    level: u32,
    current_state: &SuffixVerifierState<'_, F, E>,
    scheduled: &TerminalFoldParams,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
    T: akita_types::VerifierTranscriptGrinding<F>,
{
    let params = &scheduled;
    let t_state = match &current_state.witness {
        SuffixWitnessState::TerminalT(bytes) if !bytes.is_empty() => bytes,
        _ => return Err(AkitaError::InvalidProof),
    };
    transcript.absorb_and_record_bytes(ABSORB_COMMITMENT, t_state);
    if raw_field_segment_bytes(&proof.terminal_response.t_fields)? != *t_state {
        return Err(AkitaError::InvalidProof);
    }
    if proof.terminal_response.layout != scheduled.response_shape.layout {
        return Err(AkitaError::InvalidProof);
    }
    let group = scheduled
        .response_shape
        .layout
        .groups
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    if scheduled.response_shape.layout.groups.len() != 1
        || params.validate_terminal_linf_cap(group.z_linf_cap).is_err()
    {
        return Err(AkitaError::InvalidProof);
    }
    let recursive_num_vars = params.recursive_opening_num_vars()?;
    if current_state.setup_prefix_opening.is_some() {
        return Err(AkitaError::InvalidProof);
    }
    if current_state.opening_point.len() > recursive_num_vars {
        return Err(AkitaError::InvalidProof);
    }
    let protocol_point = current_state.opening_point.clone();
    let opening_batch = OpeningClaimsLayout::new(protocol_point.len(), 1)?;
    let (prepared_points, final_relation) = if const { <E as ExtField<F>>::DEGREE == 1 } {
        if proof.extension_opening_reduction.is_some() {
            return Err(AkitaError::InvalidProof);
        }
        let prepared_points = prepare_single_field_terminal_suffix::<F, E, T>(
            &protocol_point,
            current_state.basis,
            &current_state.opening,
            params,
            transcript,
        )?;
        (prepared_points, None)
    } else {
        let replay = verify_extension_claim_terminal_suffix::<F, E, T>(
            proof.extension_opening_reduction.as_ref(),
            &protocol_point,
            &current_state.opening,
            &opening_batch,
            current_state.basis,
            params,
            transcript,
            level,
        )?;
        (
            replay
                .groups
                .into_iter()
                .map(|group| group.prepared)
                .collect(),
            replay.final_relation,
        )
    };
    let terminal_replay = prepare_terminal_witness_replay::<F, T>(
        transcript,
        proof.terminal_response(),
        scheduled.response_shape.logical_num_elems(),
    )?;
    let fold_response_nonce =
        transcript.read_fold_response(akita_types::GrindingSite::FoldResponse { level })?;
    let operator_rejection = if params.response_l2_sq_cap().is_some() {
        Some(
            akita_challenges::selective_l2_operator_norm_rejection(
                params.d_a(),
                &scheduled.fold_challenge_config,
            )
            .ok_or(AkitaError::InvalidProof)?,
        )
    } else {
        None
    };
    let challenges = LiveFoldDraw::<F, T>::new(transcript).draw_folding_challenges_with_rejection(
        akita_challenges::FoldChallengeDrawDomain::EvaluationTrace,
        params.d_a(),
        0,
        params.blocks.live_blocks,
        1,
        &scheduled.fold_challenge_config,
        fold_response_nonce,
        operator_rejection,
    )?;
    transcript.record_fold_challenges(level, 0, params.blocks.live_blocks)?;
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_replay.response);
    super::terminal_direct::verify_terminal_ring_relations(
        setup,
        &challenges,
        &prepared_points[0].ring_multiplier_point,
        params,
        proof.terminal_response(),
    )
    .map_err(|error| {
        AkitaError::InvalidInput(format!("terminal ring relation failed: {error:?}"))
    })?;
    let (target, scale) = match final_relation {
        Some((claims, factors)) => (
            *claims.first().ok_or(AkitaError::InvalidProof)?,
            *factors.first().ok_or(AkitaError::InvalidProof)?,
        ),
        None => (current_state.opening, E::one()),
    };
    super::terminal_direct::verify_terminal_trace(
        &prepared_points[0].ring_multiplier_point,
        params,
        proof.terminal_response(),
        &prepared_points[0],
        &[E::one()],
        None,
        scale,
        target,
    )
    .map_err(|error| AkitaError::InvalidInput(format!("terminal trace failed: {error:?}")))
}
#[inline(never)]
#[tracing::instrument(skip_all, name = "prepare_fold_replay")]
fn prepare_fold_replay<'a, F, E, T>(
    proof: FoldReplayPayload<'a, F, E>,
    setup: &'a AkitaVerifierSetup<F>,
    transcript: &mut T,
    level: u32,
    current_state: &'a SuffixVerifierState<'a, F, E>,
    lp: &'a CommittedGroupParams,
    output_witness_len: usize,
) -> Result<PreparedFoldReplay<'a, F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
    T: akita_types::VerifierTranscriptGrinding<F>,
{
    let role_dims = lp.role_dims();
    let commit_d = lp.outer_payload_geometry()?.transcript_ring_dimension();
    let alpha_bits = role_dims.d_a().trailing_zeros() as usize;
    if current_state.opening_point.len() < alpha_bits {
        return Err(AkitaError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    // Absorb the current suffix commitment as flat coefficients under the
    // schedule's ring dimension — byte-identical to the prover's absorb and to
    // the former typed `append_as_ring_commitment` path (S2 byte-identity test).
    match &current_state.witness {
        SuffixWitnessState::Commitment(commitment) => {
            commitment.append_flat_to_transcript::<T>(ABSORB_COMMITMENT, commit_d, transcript)?;
        }
        _ => return Err(AkitaError::InvalidProof),
    }
    let recursive_num_vars = lp.recursive_opening_num_vars()?;
    if current_state.opening_point.len() > recursive_num_vars {
        return Err(AkitaError::InvalidProof);
    }
    let witness_point = current_state.opening_point.clone();

    let block_claims = match (&current_state.setup_prefix_opening, lp.setup_prefix()) {
        (Some((setup_prefix_point, setup_prefix_eval)), Some(_)) => {
            let groups = vec![
                PolynomialGroupClaims::new(
                    setup_prefix_point.clone(),
                    vec![*setup_prefix_eval],
                    (),
                )?,
                PolynomialGroupClaims::new(witness_point.clone(), vec![current_state.opening], ())?,
            ];
            OpeningClaims::from_groups(groups)?
        }
        (None, None) => {
            let claims =
                PolynomialGroupClaims::new(witness_point, vec![current_state.opening], ())?;
            OpeningClaims::from_groups(vec![claims])?
        }
        _ => return Err(AkitaError::InvalidProof),
    };
    let opening_batch = block_claims.layout()?;
    let openings = (0..opening_batch.num_groups())
        .flat_map(|group_index| {
            block_claims
                .group_evaluations(group_index)
                .map(|evals| evals.to_vec())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    if openings.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }
    let claim_state = if matches!(
        lp.opening_method(),
        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
    ) {
        if proof.extension_opening_reduction.is_some() {
            return Err(AkitaError::InvalidProof);
        }
        verify_coefficient_packing_suffix_prefix::<F, E, T>(
            &block_claims,
            &openings,
            &opening_batch,
            current_state.basis,
            lp,
            transcript,
        )?
    } else if const { <E as ExtField<F>>::DEGREE == 1 } {
        if proof.extension_opening_reduction.is_some() {
            return Err(AkitaError::InvalidProof);
        }
        let prepared_points =
            prepare_single_field_suffix_groups::<F, E>(&block_claims, lp, &opening_batch)?;
        let group_points = (0..opening_batch.num_groups())
            .map(|group_index| block_claims.group_point(group_index))
            .collect::<Result<Vec<_>, _>>()?;
        absorb_protocol_opening_points(&group_points, transcript);
        append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
        FoldClaimMaterial {
            prepared_points: prepared_points
                .into_iter()
                .map(super::fold::PreparedFoldOpeningPoint::EvaluationTrace)
                .collect(),
            openings: openings.clone(),
            reduction_final_claims: None,
            reduction_factors: None,
        }
    } else {
        append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
        let group_points = (0..opening_batch.num_groups())
            .map(|group_index| block_claims.group_point(group_index))
            .collect::<Result<Vec<_>, _>>()?;
        verify_extension_claim_suffix_prefix::<F, E, T>(
            proof.extension_opening_reduction,
            &group_points,
            &openings,
            &opening_batch,
            current_state.basis,
            lp,
            transcript,
            level,
        )?
    };

    let witness_len = output_witness_len;
    let (opening_payload, payload) = match proof.kind {
        FoldReplayKind::Recursive {
            v,
            stage1,
            stage2,
            next_witness,
            next_witness_ring_dim,
            stage3,
        } => {
            if next_witness_ring_dim == 0 || !next_witness_ring_dim.is_power_of_two() {
                return Err(AkitaError::InvalidProof);
            }
            let committed_witness_len =
                akita_types::witness_commitment_domain_len(witness_len, next_witness_ring_dim)?;
            (
                v.clone(),
                PreparedFoldPayload::Recursive {
                    stage1,
                    stage2,
                    next_witness,
                    next_witness_ring_dim,
                    next_opening_source_len: committed_witness_len / next_witness_ring_dim,
                    stage3,
                },
            )
        }
    };
    let prefix = bind_opening_payload_and_finalize_claims(
        lp,
        &opening_batch,
        &opening_payload,
        claim_state,
        transcript,
        level,
    )?;
    let current_commitment = match &current_state.witness {
        SuffixWitnessState::Commitment(commitment) => *commitment,
        SuffixWitnessState::TerminalT(_) => return Err(AkitaError::InvalidProof),
    };
    let commitment_payloads =
        suffix_commitment_payloads::<F, E>(setup, lp, &opening_batch, current_commitment)?;
    Ok(PreparedFoldReplay {
        lp,
        level,
        opening_payload,
        opening_shape: opening_batch,
        commitment_payloads,
        prefix,
        w_len: witness_len,
        payload,
        evaluation_trace_basis: current_state.basis,
    })
}
