use super::*;
use akita_types::{BatchedStage3Geometry, OpeningClaimsLayout, RingView};

fn absorb_prepared_opening_points<F, E, T>(
    prepared_points: &[PreparedOpeningPoint<F, E>],
    transcript: &mut T,
) where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + AkitaSerialize,
    T: Transcript<F>,
{
    for prepared in prepared_points {
        for coordinate in &prepared.padded_point {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, coordinate);
        }
    }
}

/// Verifier state carried between suffix fold levels.
pub(super) struct SuffixVerifierState<'a, F: FieldCore, E: FieldCore> {
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

pub(super) enum SuffixWitnessState<'a, F: FieldCore> {
    Commitment(&'a RingVec<F>),
    TerminalT(Vec<u8>),
}

fn prepare_suffix_group_points<F, E>(
    protocol_point: &[E],
    block_claims: &OpeningClaims<'_, E>,
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    role_d_a: usize,
    alpha_bits: usize,
) -> Result<Vec<PreparedOpeningPoint<F, E>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
{
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        role_d_a,
        |D| {
            let mut prepared_points = Vec::with_capacity(opening_batch.num_groups());
            for group_index in 0..opening_batch.num_groups() {
                let group_lp = lp.group_params(opening_batch, group_index)?;
                let target_len = alpha_bits
                    .checked_add(group_lp.position_index_bits())
                    .and_then(|n| n.checked_add(group_lp.block_index_bits()))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("group opening point length overflow".to_string())
                    })?;
                let point_vars = block_claims.group_point_vars(group_index)?;
                if point_vars.num_vars() != target_len {
                    return Err(AkitaError::InvalidInput(format!(
                        "suffix group point width mismatch: group={group_index}, \
                         groups={}, setup_prefix={}, target_len={target_len}, actual_len={}",
                        opening_batch.num_groups(),
                        lp.setup_prefix.is_some(),
                        point_vars.num_vars()
                    )));
                }
                let group_protocol_point = point_vars
                    .indices()
                    .iter()
                    .map(|&idx| {
                        protocol_point
                            .get(idx)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                prepared_points.push(prepare_opening_point::<F, E, D>(
                    &group_protocol_point,
                    BasisMode::Lagrange,
                    group_lp.num_positions_per_block(),
                    group_lp.num_live_blocks(),
                    alpha_bits,
                )?);
            }
            Ok(prepared_points)
        }
    )
}

fn suffix_commitment_rows<F: FieldCore>(
    setup: &AkitaVerifierSetup<F>,
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    witness_commitment: &RingVec<F>,
) -> Result<RingVec<F>, AkitaError> {
    let mut group_rows = Vec::with_capacity(opening_batch.num_groups());
    if let Some(setup_prefix_id) = lp.setup_prefix.as_ref() {
        let slot = setup.prefix_slots.get(setup_prefix_id).ok_or_else(|| {
            AkitaError::InvalidSetup(
                "planned setup-prefix slot is missing from verifier setup".to_string(),
            )
        })?;
        let mut coeffs = Vec::new();
        for row in &slot.commitment.rows {
            coeffs.extend_from_slice(row.coeffs());
        }
        group_rows.push(RingVec::from_coeffs(coeffs));
    }
    group_rows.push(RingVec::from_coeffs(witness_commitment.coeffs().to_vec()));
    if group_rows.len() != opening_batch.num_groups() {
        return Err(AkitaError::InvalidProof);
    }

    let commitment_ring_dim = lp.role_dims().d_b();
    let mut group_order = (0..opening_batch.num_groups())
        .map(|group_index| {
            let range = lp.commitment_row_range(opening_batch, group_index)?;
            Ok((range.start, range.len(), group_index))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    group_order.sort_by_key(|(start, _, _)| *start);

    let mut coeffs = Vec::new();
    for (_, expected_rows, group_index) in group_order {
        let rows = group_rows
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if !rows.can_decode_vec(commitment_ring_dim) {
            return Err(AkitaError::InvalidProof);
        }
        let actual_rows = rows.coeff_len() / commitment_ring_dim;
        if actual_rows != expected_rows {
            return Err(AkitaError::InvalidProof);
        }
        coeffs.extend_from_slice(rows.coeffs());
    }
    Ok(RingVec::from_coeffs(coeffs))
}

struct FoldReplayPayload<'a, F: FieldCore, E: FieldCore> {
    extension_opening_reduction: Option<&'a ExtensionOpeningReductionProof<E>>,
    fold_grind_nonce: u32,
    kind: FoldReplayKind<'a, F, E>,
}

enum FoldReplayKind<'a, F: FieldCore, E: FieldCore> {
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
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField + HalvingField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F>,
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
        let current_lp = &step.params.witness;
        let next_step = schedule.recursive_folds.get(offset + 1);
        let next_params = next_step.map(|next| &next.params.witness);
        let next_witness_ring_dim = next_params.map_or(
            schedule.terminal.params.witness.d_a(),
            CommittedGroupParams::d_a,
        );
        let role_dims = current_lp.role_dims();
        let current_commitment = match &current_state.witness {
            SuffixWitnessState::Commitment(commitment) => *commitment,
            SuffixWitnessState::TerminalT(_) => return Err(AkitaError::InvalidProof),
        };
        if !current_commitment.can_decode_vec(role_dims.d_b())
            || !fold.v.can_decode_vec(role_dims.d_d())
        {
            return Err(AkitaError::InvalidProof);
        }
        let commitment_view = RingView::new(current_commitment.coeffs(), role_dims.d_b())?;
        if commitment_view.num_rings() != current_lp.outer_commit_matrix.output_rank() {
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
        let next_witness = match (fold.next_w_commitment(), next_t_state.as_deref()) {
            (Some(commitment), None) => PreparedNextWitness::Commitment {
                commitment,
                ring_dim: next_params
                    .ok_or(AkitaError::InvalidProof)?
                    .role_dims()
                    .d_b(),
            },
            (None, Some(t_state)) if !t_state.is_empty() => PreparedNextWitness::TerminalT(t_state),
            _ => return Err(AkitaError::InvalidProof),
        };
        let setup_contribution_mode = next_step.map_or(SetupContributionMode::Direct, |step| {
            step.params.predecessor_setup_contribution_mode()
        });
        let stage3 = fold.stage3_for_mode(setup_contribution_mode, next_params)?;
        let prepared = prepare_fold_replay::<F, E, T>(
            FoldReplayPayload {
                extension_opening_reduction: fold.extension_opening_reduction(),
                fold_grind_nonce: fold.fold_grind_nonce,
                kind: FoldReplayKind::Recursive {
                    v: &fold.v,
                    stage1: &fold.stage1,
                    stage2: &fold.stage2,
                    next_witness,
                    next_witness_ring_dim,
                    stage3,
                },
            },
            setup,
            transcript,
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

        let next_commitment = fold.next_w_commitment();
        let next_witness = match (next_commitment, next_t_state) {
            (Some(commitment), None) => SuffixWitnessState::Commitment(commitment),
            (None, Some(t_state)) => SuffixWitnessState::TerminalT(t_state),
            _ => return Err(AkitaError::InvalidProof),
        };
        current_state = SuffixVerifierState {
            opening_point: challenges,
            opening: fold
                .stage3_sumcheck_proof()
                .map_or_else(|| fold.next_w_eval(), |proof| proof.next_w_eval),
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
        != schedule.terminal.params.response_shape.logical_num_elems()
    {
        return Err(AkitaError::InvalidProof);
    }
    verify_terminal_suffix::<F, E, T>(
        terminal,
        setup,
        transcript,
        &current_state,
        &schedule.terminal.params,
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
    current_state: &SuffixVerifierState<'_, F, E>,
    scheduled: &TerminalFoldParams,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField + HalvingField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let params = &scheduled.witness;
    let t_state = match &current_state.witness {
        SuffixWitnessState::TerminalT(bytes) if !bytes.is_empty() => bytes,
        _ => return Err(AkitaError::InvalidProof),
    };
    transcript.absorb_and_record_bytes(ABSORB_COMMITMENT, t_state);
    if raw_field_segment_bytes(&proof.terminal_response.t_fields)? != *t_state {
        return Err(AkitaError::InvalidProof);
    }
    let (honest_cap, _) = params.response_linf_bounds(&scheduled.sparse_challenge_config)?;
    let expected_shape = TerminalResponseShape::derive(params, honest_cap)?;
    if scheduled.response_shape != expected_shape
        || proof.terminal_response.layout != expected_shape.layout
    {
        return Err(AkitaError::InvalidProof);
    }
    params.validate_fold_grind_nonce(&scheduled.sparse_challenge_config, proof.fold_grind_nonce)?;

    let alpha_bits = params.d_a().trailing_zeros() as usize;
    let recursive_num_vars = params.recursive_opening_num_vars()?;
    if current_state.opening_point.len() > recursive_num_vars
        || current_state.setup_prefix_opening.is_some()
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut protocol_point = current_state.opening_point.clone();
    protocol_point.resize(recursive_num_vars, E::zero());
    let opening_batch = OpeningClaimsLayout::new(recursive_num_vars, 1)?;
    let FoldEorReplay {
        prepared_points,
        final_relation,
        ..
    } = verify_fold_eor_geometry::<F, E, T>(
        proof.extension_opening_reduction.as_ref(),
        &protocol_point,
        &[current_state.opening],
        &[E::one()],
        &opening_batch,
        current_state.basis,
        params.d_a(),
        params.num_positions_per_block,
        params.num_live_blocks,
        <E as ExtField<F>>::EXT_DEGREE != 1,
        transcript,
    )
    .map_err(|error| {
        AkitaError::InvalidInput(format!(
            "terminal extension-opening replay failed: {error:?}"
        ))
    })?;
    let prepared_points = if proof.extension_opening_reduction.is_some() {
        prepared_points
    } else {
        vec![dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            params.d_a(),
            |D| prepare_opening_point::<F, E, D>(
                &protocol_point,
                current_state.basis,
                params.num_positions_per_block,
                params.num_live_blocks,
                alpha_bits,
            )
        )?]
    };
    absorb_prepared_opening_points(&prepared_points, transcript);
    if final_relation.is_none() {
        append_claim_values_to_transcript::<F, E, T>(
            std::slice::from_ref(&current_state.opening),
            transcript,
        );
    }
    let terminal_replay = prepare_terminal_witness_replay::<F, T>(
        transcript,
        proof.terminal_response(),
        scheduled.response_shape.logical_num_elems(),
    )?;
    let challenges = LiveFoldDraw::<F, T>::new(transcript).draw_folding_challenges(
        params.d_a(),
        0,
        params.num_live_blocks,
        1,
        &scheduled.sparse_challenge_config,
        &TensorChallengeShape::Flat,
        witness_fold_challenge_labels(),
        proof.fold_grind_nonce,
    )?;
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_replay.response);
    super::terminal_direct::verify_terminal_ring_relations(
        setup,
        &challenges,
        &prepared_points[0].ring_multiplier_point,
        params,
        &scheduled.sparse_challenge_config,
        proof.terminal_response(),
    )
    .map_err(|error| {
        AkitaError::InvalidInput(format!("terminal ring relation failed: {error:?}"))
    })?;
    let (target, scale) = match final_relation {
        Some((claim, factors)) => (claim, *factors.first().ok_or(AkitaError::InvalidProof)?),
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
    current_state: &'a SuffixVerifierState<'a, F, E>,
    lp: &'a CommittedGroupParams,
    output_witness_len: usize,
) -> Result<PreparedFoldReplay<'a, F, E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField + HalvingField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let role_dims = lp.role_dims();
    let commit_d = role_dims.d_b();
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
    let block_claims = match (
        &current_state.setup_prefix_opening,
        lp.setup_prefix.as_ref(),
    ) {
        (Some((setup_prefix_point, setup_prefix_eval)), Some(setup_prefix_id)) => {
            let (shared_point, setup_offset) = BatchedStage3Geometry::shared_suffix_point(
                setup_prefix_point,
                current_state.opening_point.as_slice(),
            )?;
            let setup_point_vars = BatchedStage3Geometry::setup_prefix_point_vars(
                setup_prefix_point.len(),
                setup_prefix_id,
                setup_offset,
                shared_point.len(),
            )?;
            let groups = vec![
                PolynomialGroupClaims::new(setup_point_vars, vec![*setup_prefix_eval], ())?,
                PolynomialGroupClaims::new(
                    PointVariableSelection::suffix(
                        current_state.opening_point.len(),
                        shared_point.len(),
                    )?,
                    vec![current_state.opening],
                    (),
                )?,
            ];
            OpeningClaims::from_groups_allow_custom_routing(shared_point, groups)?
        }
        (None, None) => {
            let mut padded_point = current_state.opening_point.clone();
            padded_point.resize(recursive_num_vars, E::zero());
            let claims = PolynomialGroupClaims::new(
                PointVariableSelection::prefix(recursive_num_vars, recursive_num_vars)?,
                vec![current_state.opening],
                (),
            )?;
            OpeningClaims::from_groups(padded_point, vec![claims])?
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
    let row_coefficients = vec![E::one(); opening_batch.num_total_polynomials()];
    let requires_extension_reduction =
        <E as ExtField<F>>::EXT_DEGREE != 1 && lp.setup_prefix.is_none();
    let FoldEorReplay {
        prepared_points,
        reduction_challenges: _,
        final_relation: eor_trace_final,
        ..
    } = verify_fold_eor::<F, E, T>(
        proof.extension_opening_reduction,
        block_claims.point(),
        &openings,
        &row_coefficients,
        &opening_batch,
        current_state.basis,
        lp,
        requires_extension_reduction,
        transcript,
    )?;
    if proof.extension_opening_reduction.is_some() && opening_batch.num_groups() != 1 {
        return Err(AkitaError::InvalidProof);
    }
    let prepared_points = if proof.extension_opening_reduction.is_some() {
        prepared_points
    } else {
        prepare_suffix_group_points::<F, E>(
            block_claims.point(),
            &block_claims,
            lp,
            &opening_batch,
            role_dims.d_a(),
            alpha_bits,
        )?
    };
    absorb_prepared_opening_points(&prepared_points, transcript);

    let witness_len = output_witness_len;
    let (trace_eval_target, trace_eval_scale) = match eor_trace_final.as_ref() {
        Some((final_claim, factors_by_point)) => (
            *final_claim,
            *factors_by_point.first().ok_or(AkitaError::InvalidProof)?,
        ),
        None => (
            opening_batch.batched_eval_target(&row_coefficients, &openings)?,
            E::one(),
        ),
    };

    let fold_grind_nonce = proof.fold_grind_nonce;
    let (v_storage, payload, next_opening_ring_dim) = match proof.kind {
        FoldReplayKind::Recursive {
            v,
            stage1,
            stage2,
            next_witness,
            next_witness_ring_dim,
            stage3,
        } => {
            if next_witness_ring_dim == 0 || !witness_len.is_multiple_of(next_witness_ring_dim) {
                return Err(AkitaError::InvalidProof);
            }
            (
                v.clone(),
                PreparedFoldPayload::Recursive {
                    stage1,
                    stage2,
                    next_witness,
                    next_witness_ring_dim,
                    next_opening_source_len: witness_len / next_witness_ring_dim,
                    stage3,
                },
                next_witness_ring_dim,
            )
        }
    };
    let current_commitment = match &current_state.witness {
        SuffixWitnessState::Commitment(commitment) => *commitment,
        SuffixWitnessState::TerminalT(_) => return Err(AkitaError::InvalidProof),
    };
    let commitment_rows = suffix_commitment_rows(setup, lp, &opening_batch, current_commitment)?;
    if !witness_len.is_multiple_of(next_opening_ring_dim) {
        return Err(AkitaError::InvalidProof);
    }
    Ok(PreparedFoldReplay {
        lp,
        fold_grind_nonce,
        v: v_storage,
        opening_shape: opening_batch,
        commitment_rows,
        row_coefficients,
        group_ring_opening_points: prepared_points
            .iter()
            .map(|point| point.ring_opening_point.clone())
            .collect(),
        group_ring_multiplier_points: prepared_points
            .iter()
            .map(|point| point.ring_multiplier_point.clone())
            .collect(),
        witness_len,
        payload,
        trace_prepared_points: Some(prepared_points),
        trace_block_opening: None,
        trace_eval_target,
        trace_eval_scale,
        trace_claim_scales: None,
        trace_basis: current_state.basis,
    })
}
