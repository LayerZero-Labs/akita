use super::*;
use akita_types::{OpeningClaimsLayout, RingView};

/// Verifier state carried between suffix fold levels.
pub(super) struct SuffixVerifierState<'a, F: FieldCore, E: FieldCore> {
    /// Current opening point for the committed suffix witness.
    pub opening_point: Vec<E>,
    /// Claimed opening value for the current commitment.
    pub opening: E,
    /// Current suffix witness commitment.
    pub commitment: &'a RingVec<F>,
    /// Basis used to interpret the current opening point.
    pub basis: BasisMode,
    /// Current suffix witness length in field elements.
    pub w_len: usize,
    /// Optional setup-prefix opening carried from the previous stage-3 proof.
    pub setup_prefix_opening: Option<SetupPrefixOpening<E>>,
}

fn setup_prefix_column_major_indices(
    setup_prefix_point_len: usize,
    setup_prefix_id: &akita_types::SetupPrefixSlotId,
    offset: usize,
    shared_point_len: usize,
) -> Result<PointVariableSelection, AkitaError> {
    let ring_bits = setup_prefix_id.d_setup.trailing_zeros() as usize;
    let params = &setup_prefix_id.commitment_params;
    let expected = ring_bits
        .checked_add(params.layout.r_vars)
        .and_then(|n| n.checked_add(params.layout.m_vars))
        .ok_or_else(|| AkitaError::InvalidSetup("setup-prefix point length overflow".into()))?;
    if setup_prefix_point_len != expected {
        return Err(AkitaError::InvalidInput(format!(
            "setup-prefix point width mismatch: expected={expected}, \
             actual={setup_prefix_point_len}, ring_bits={ring_bits}, \
             m_vars={}, r_vars={}, natural_len={}",
            params.layout.m_vars, params.layout.r_vars, setup_prefix_id.natural_len,
        )));
    }
    let mut indices = Vec::with_capacity(expected);
    indices.extend(offset..offset + ring_bits);
    indices.extend(
        offset + ring_bits + params.layout.m_vars
            ..offset + ring_bits + params.layout.m_vars + params.layout.r_vars,
    );
    indices.extend(offset + ring_bits..offset + ring_bits + params.layout.m_vars);
    PointVariableSelection::new(indices, shared_point_len)
}

fn shared_stage3_point<E: FieldCore>(
    setup_prefix_point: &[E],
    witness_point: &[E],
) -> Result<(Vec<E>, usize), AkitaError> {
    if setup_prefix_point.len() >= witness_point.len() {
        if &setup_prefix_point[setup_prefix_point.len() - witness_point.len()..] != witness_point {
            return Err(AkitaError::InvalidInput(
                "stage-3 suffix opening points are inconsistent".to_string(),
            ));
        }
        Ok((setup_prefix_point.to_vec(), 0))
    } else {
        if &witness_point[witness_point.len() - setup_prefix_point.len()..] != setup_prefix_point {
            return Err(AkitaError::InvalidInput(
                "stage-3 suffix opening points are inconsistent".to_string(),
            ));
        }
        Ok((
            witness_point.to_vec(),
            witness_point.len() - setup_prefix_point.len(),
        ))
    }
}

fn prepare_suffix_group_points<F, E>(
    protocol_point: &[E],
    fold_claims: &OpeningClaims<'_, E>,
    lp: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    role_d_a: usize,
    alpha_bits: usize,
    block_order: BlockOrder,
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
                let group_lp = lp.root_group_params(opening_batch, group_index)?;
                let target_len = alpha_bits
                    .checked_add(group_lp.m_vars())
                    .and_then(|n| n.checked_add(group_lp.r_vars()))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("group opening point length overflow".to_string())
                    })?;
                let point_vars = fold_claims.group_point_vars(group_index)?;
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
                    group_lp.m_vars(),
                    group_lp.r_vars(),
                    alpha_bits,
                    block_order,
                )?);
            }
            Ok(prepared_points)
        }
    )
}

fn suffix_commitment_rows<F: FieldCore>(
    setup: &AkitaVerifierSetup<F>,
    lp: &LevelParams,
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

    let commitment_ring_dim = lp.role_dims().d_a();
    let mut group_order = (0..opening_batch.num_groups())
        .map(|group_index| {
            let range = lp.root_commitment_row_range(
                opening_batch,
                group_index,
                RelationMatrixRowLayout::WithDBlock,
            )?;
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

/// Prepare one suffix fold level for relation verification.
///
/// Terminal levels absorb the cleartext final witness instead of a
/// next-witness commitment and run stage-2 in relation-only mode.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, the public trace check
/// fails, or the terminal witness replay is malformed.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[tracing::instrument(skip_all, name = "prepare_fold_replay")]
fn prepare_fold_replay<'a, F, E, T>(
    proof: &'a AkitaLevelProof<F, E>,
    setup: &'a AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &'a SuffixVerifierState<'a, F, E>,
    scheduled: &'a ExecutionSchedule,
    block_order: BlockOrder,
    _setup_contribution_mode: SetupContributionMode,
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
    let lp = &scheduled.params;
    let role_dims = lp.role_dims();
    let commit_d = role_dims.d_b();
    let next_fold_level_params = (!scheduled.is_terminal).then_some(&scheduled.next_params);
    let relation_matrix_row_layout = proof.relation_matrix_row_layout();
    let alpha_bits = role_dims.d_a().trailing_zeros() as usize;
    if current_state.opening_point.len() < alpha_bits {
        return Err(AkitaError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    // Absorb the current suffix commitment as flat coefficients under the
    // schedule's ring dimension — byte-identical to the prover's absorb and to
    // the former typed `append_as_ring_commitment` path (S2 byte-identity test).
    current_state.commitment.append_flat_to_transcript::<T>(
        ABSORB_COMMITMENT,
        commit_d,
        transcript,
    )?;
    let recursive_num_vars = lp.recursive_opening_num_vars()?;
    let fold_claims = match (
        &current_state.setup_prefix_opening,
        lp.setup_prefix.as_ref(),
    ) {
        (Some((setup_prefix_point, setup_prefix_eval)), Some(setup_prefix_id)) => {
            let (shared_point, setup_offset) =
                shared_stage3_point(setup_prefix_point, current_state.opening_point.as_slice())?;
            let setup_point_vars = setup_prefix_column_major_indices(
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
    let opening_batch = fold_claims.layout()?;
    let openings = (0..opening_batch.num_groups())
        .flat_map(|group_index| {
            fold_claims
                .group_evaluations(group_index)
                .map(|evals| evals.to_vec())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    if openings.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }
    let row_coefficients = vec![E::one(); opening_batch.num_total_polynomials()];
    let FoldEorReplay {
        prepared_points,
        reduction_challenges: _,
        final_relation: eor_trace_final,
        ..
    } = verify_fold_eor::<F, E, T>(
        proof.extension_opening_reduction(),
        fold_claims.point(),
        &openings,
        &row_coefficients,
        &opening_batch,
        current_state.basis,
        lp,
        block_order,
        true,
        transcript,
    )?;
    if proof.extension_opening_reduction().is_some() && opening_batch.num_groups() != 1 {
        return Err(AkitaError::InvalidProof);
    }
    let prepared_points = if proof.extension_opening_reduction().is_some() {
        prepared_points
    } else {
        prepare_suffix_group_points::<F, E>(
            fold_claims.point(),
            &fold_claims,
            lp,
            &opening_batch,
            role_dims.d_a(),
            alpha_bits,
            block_order,
        )?
    };
    for prepared_point in &prepared_points {
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
    }

    let w_len = scheduled.next_w_len;
    let terminal_replay = if proof.final_w_len().is_some() {
        let final_w_len = proof.final_w_len().ok_or(AkitaError::InvalidProof)?;
        let final_witness = proof
            .stage2()
            .final_witness()
            .ok_or(AkitaError::InvalidProof)?;
        if final_witness.as_segment_typed().is_none() {
            return Err(AkitaError::InvalidProof);
        }
        let layout = TerminalWitnessSegmentLayout {
            e_hat_digit_offset: 0,
            e_hat_digit_count: 1,
        };
        Some(prepare_terminal_witness_replay::<F, T>(
            transcript,
            final_witness,
            final_w_len,
            layout,
        )?)
    } else {
        None
    };
    let stage1_proof = proof.stage1_proof();
    let next_w_commitment = proof.next_w_commitment_opt();
    let stage3 = proof.stage3_for_mode(lp.setup_contribution_mode, next_fold_level_params)?;
    let stage2 = proof.stage2();
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

    let fold_grind_nonce = proof.fold_grind_nonce();
    let v_storage = match proof {
        AkitaLevelProof::Intermediate { v, .. } => v.clone(),
        AkitaLevelProof::Terminal { .. } => RingVec::from_coeffs(Vec::new()),
    };
    let commitment_rows =
        suffix_commitment_rows(setup, lp, &opening_batch, current_state.commitment)?;
    Ok(PreparedFoldReplay {
        lp,
        relation_matrix_row_layout,
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
        w_len,
        stage1: stage1_proof,
        stage2,
        next_w_commitment,
        next_ring_dim: (!scheduled.is_terminal).then_some(scheduled.next_params.role_dims().d_b()),
        terminal_replay,
        stage3,
        trace_prepared_points: Some(prepared_points),
        trace_block_opening: None,
        trace_eval_target,
        trace_eval_scale,
        trace_claim_scales: None,
        trace_basis: current_state.basis,
        block_order,
    })
}

/// Verify all suffix fold levels after the root proof.
///
/// The supplied `schedule` is the already-selected public schedule for this
/// proof shape. This function checks that each proof level matches that
/// schedule, dispatches to the corresponding ring dimension, and threads the
/// verifier state to the next suffix commitment.
///
/// # Errors
///
/// Returns an error if the schedule is malformed for the supplied proof,
/// decoded proof dimensions do not match, any fold-level verifier rejects, or
/// the suffix witness handoff has the wrong shape.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_suffix<'a, F, E, T>(
    steps: &'a [AkitaLevelProof<F, E>],
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    schedule: &Schedule,
    mut current_state: SuffixVerifierState<'a, F, E>,
    setup_contribution_mode: SetupContributionMode,
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
    for (offset, step) in steps.iter().enumerate() {
        let level_index = offset + 1;
        let scheduled = schedule.get_execution_schedule(level_index)?;
        scheduled.validate_current_w_len(current_state.w_len)?;
        let current_lp = &scheduled.params;
        let next_params = &scheduled.next_params;
        let next_w_len = scheduled.next_w_len;
        let _carried_setup_prefix = &current_state.setup_prefix_opening;
        let role_dims = current_lp.role_dims();
        let commit_d = role_dims.d_b();
        let witness_d = role_dims.d_a();
        match step {
            AkitaLevelProof::Intermediate { .. } => {
                let level_proof = step;
                if scheduled.is_terminal {
                    return Err(AkitaError::InvalidProof);
                }
                if !current_state.commitment.can_decode_vec(commit_d)
                    || !level_proof.v().can_decode_vec(witness_d)
                {
                    return Err(AkitaError::InvalidProof);
                }
                let commitment_view = RingView::new(current_state.commitment.coeffs(), commit_d)?;
                if commitment_view.num_rings() != current_lp.b_key.row_len() {
                    return Err(AkitaError::InvalidProof);
                }

                let (challenges, setup_prefix_opening) = {
                    let prepared = prepare_fold_replay::<F, E, T>(
                        level_proof,
                        setup,
                        transcript,
                        &current_state,
                        &scheduled,
                        BlockOrder::ColumnMajor,
                        setup_contribution_mode,
                    )?;
                    verify_fold::<F, E, T>(setup, transcript, prepared).map_err(|err| {
                        AkitaError::InvalidInput(format!(
                            "suffix verify level {level_index} failed: {err:?}"
                        ))
                    })?
                };

                let next_commit_d = next_params.role_dims().d_b();
                if next_commit_d == 0
                    || !level_proof
                        .next_w_commitment()
                        .can_decode_vec(next_commit_d)
                {
                    return Err(AkitaError::InvalidProof);
                }
                current_state = SuffixVerifierState {
                    opening_point: challenges,
                    opening: level_proof
                        .stage3_sumcheck_proof()
                        .map_or_else(|| level_proof.next_w_eval(), |proof| proof.next_w_eval),
                    commitment: level_proof.next_w_commitment(),
                    basis: BasisMode::Lagrange,
                    w_len: next_w_len,
                    setup_prefix_opening,
                };
            }
            AkitaLevelProof::Terminal { .. } => {
                let terminal_proof = step;
                if !current_state.commitment.can_decode_vec(commit_d) {
                    return Err(AkitaError::InvalidProof);
                }
                if terminal_proof
                    .final_w_len()
                    .ok_or(AkitaError::InvalidProof)?
                    != next_w_len
                {
                    return Err(AkitaError::InvalidProof);
                }
                let prepared = prepare_fold_replay::<F, E, T>(
                    terminal_proof,
                    setup,
                    transcript,
                    &current_state,
                    &scheduled,
                    BlockOrder::ColumnMajor,
                    setup_contribution_mode,
                )?;
                verify_fold::<F, E, T>(setup, transcript, prepared).map_err(|err| {
                    AkitaError::InvalidInput(format!(
                        "suffix verify level {level_index} failed: {err:?}"
                    ))
                })?;
                // The trailing-Direct witness shape is already validated in
                // `verify_folded_batched_proof` before this loop.
                if !scheduled.is_terminal {
                    return Err(AkitaError::InvalidProof);
                }
            }
        }
    }

    Ok(())
}
