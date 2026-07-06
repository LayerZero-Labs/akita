use super::*;
use akita_types::{terminal_witness_segment_layout, OpeningClaimsLayout, RingView};

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
    transcript: &mut T,
    current_state: &'a SuffixVerifierState<'a, F, E>,
    scheduled: &'a ExecutionSchedule,
    block_order: BlockOrder,
    setup_contribution_mode: SetupContributionMode,
) -> Result<PreparedFoldReplay<'a, F, E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField + HalvingField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let lp = &scheduled.params;
    let role_dims = lp.role_dims();
    let commit_d = role_dims.d_b();
    let next_fold_level_params = (!scheduled.is_terminal).then_some(&scheduled.next_params);
    let m_row_layout = proof.m_row_layout();
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
    let num_claims = 1usize;
    let num_vars = lp.recursive_opening_num_vars()?;
    let opening_batch = OpeningClaimsLayout::new(num_vars, num_claims)?;
    let openings = vec![current_state.opening];
    let row_coefficients = vec![E::one()];
    let FoldEorReplay {
        prepared_points,
        reduction_challenges: _,
        final_relation: eor_trace_final,
        ..
    } = verify_fold_eor::<F, E, T>(
        proof.extension_opening_reduction(),
        current_state.opening_point.as_slice(),
        &openings,
        &row_coefficients,
        &opening_batch,
        current_state.basis,
        lp,
        block_order,
        true,
        transcript,
    )?;
    if prepared_points.len() != 1 {
        return Err(AkitaError::InvalidProof);
    }
    let prepared_point = &prepared_points[0];
    for pt in &prepared_point.padded_point {
        append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
    }

    let w_len = match proof.final_w_len() {
        Some(final_w_len) => final_w_len,
        None => {
            let nc = lp.witness_chunk.num_chunks;
            let ring = if nc > 1 {
                akita_types::w_ring_element_count_for_chunks(
                    F::modulus_bits(),
                    lp,
                    num_claims,
                    MRowLayout::WithDBlock,
                    nc,
                )?
            } else {
                w_ring_element_count_with_counts_for_layout::<F>(
                    lp,
                    num_claims,
                    num_claims,
                    MRowLayout::WithDBlock,
                )?
            };
            ring.checked_mul(role_dims.d_a()).ok_or_else(|| {
                AkitaError::InvalidSetup("next witness length overflow".to_string())
            })?
        }
    };
    let terminal_replay = if proof.final_w_len().is_some() {
        let layout =
            terminal_witness_segment_layout(lp, num_claims, num_claims, F::modulus_bits())?;
        let final_witness = proof
            .stage2()
            .final_witness()
            .ok_or(AkitaError::InvalidProof)?;
        Some(prepare_terminal_witness_replay::<F, T>(
            transcript,
            final_witness,
            w_len,
            layout,
        )?)
    } else {
        None
    };
    let stage1_proof = proof.stage1_proof();
    let next_w_commitment = proof.next_w_commitment_opt();
    let stage3 = proof.stage3_for_mode(setup_contribution_mode, next_fold_level_params)?;
    let stage2 = proof.stage2();
    let (trace_eval_target, trace_eval_scale) = match eor_trace_final.as_ref() {
        Some((final_claim, factors_by_point)) => (
            *final_claim,
            *factors_by_point.first().ok_or(AkitaError::InvalidProof)?,
        ),
        None => (current_state.opening, E::one()),
    };

    let fold_grind_nonce = proof.fold_grind_nonce();
    let v_storage = match proof {
        AkitaLevelProof::Intermediate { v, .. } => v.clone(),
        AkitaLevelProof::Terminal { .. } => RingVec::from_coeffs(Vec::new()),
    };
    let replay_opening_batch = OpeningClaims::from_groups(
        current_state.opening_point.as_slice(),
        vec![PolynomialGroupClaims::new(
            PointVariableSelection::prefix(
                opening_batch.max_num_vars(),
                current_state.opening_point.len(),
            )?,
            openings,
            current_state.commitment,
        )?],
    )?;
    Ok(PreparedFoldReplay {
        lp,
        m_row_layout,
        fold_grind_nonce,
        v: v_storage,
        opening_batch: replay_opening_batch,
        row_coefficients,
        ring_opening_point: prepared_point.ring_opening_point.clone(),
        ring_multiplier_point: prepared_point.ring_multiplier_point.clone(),
        w_len,
        stage1: stage1_proof,
        stage2,
        next_w_commitment,
        next_ring_dim: (!scheduled.is_terminal).then_some(scheduled.next_params.role_dims().d_b()),
        terminal_replay,
        stage3,
        trace_prepared_point: Some(prepared_point.clone()),
        trace_block_opening: None,
        trace_eval_target,
        trace_eval_scale,
        trace_claim_scales: None,
        trace_basis: current_state.basis,
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
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    for (offset, step) in steps.iter().enumerate() {
        let level_index = offset + 1;
        let scheduled = schedule.get_execution_schedule(level_index)?;
        scheduled.validate_current_w_len(current_state.w_len)?;
        let current_lp = &scheduled.params;
        let next_params = &scheduled.next_params;
        let next_w_len = scheduled.next_w_len;
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

                let challenges = {
                    let prepared = prepare_fold_replay::<F, E, T>(
                        level_proof,
                        transcript,
                        &current_state,
                        &scheduled,
                        BlockOrder::ColumnMajor,
                        setup_contribution_mode,
                    )?;
                    verify_fold::<F, E, T>(setup, transcript, prepared)?
                };

                let next_commit_d = next_params.role_dims().d_b();
                if next_commit_d == 0
                    || !level_proof
                        .next_w_commitment()
                        .can_decode_vec(next_commit_d)
                {
                    return Err(AkitaError::InvalidProof);
                }
                let next_chunks = current_lp.witness_chunk.num_chunks;
                let computed_next_w_ring = if next_chunks > 1 {
                    akita_types::w_ring_element_count_for_chunks(
                        F::modulus_bits(),
                        current_lp,
                        1,
                        MRowLayout::WithDBlock,
                        next_chunks,
                    )?
                } else {
                    w_ring_element_count_with_counts_for_layout::<F>(
                        current_lp,
                        1,
                        1,
                        MRowLayout::WithDBlock,
                    )?
                };
                let computed_next_w_len =
                    computed_next_w_ring.checked_mul(witness_d).ok_or_else(|| {
                        AkitaError::InvalidSetup("next witness length overflow".to_string())
                    })?;
                scheduled.validate_next_w_len(computed_next_w_len)?;
                current_state = SuffixVerifierState {
                    opening_point: challenges,
                    opening: level_proof
                        .stage3_sumcheck_proof()
                        .map_or_else(|| level_proof.next_w_eval(), |proof| proof.next_w_eval),
                    commitment: level_proof.next_w_commitment(),
                    basis: BasisMode::Lagrange,
                    w_len: next_w_len,
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
                    transcript,
                    &current_state,
                    &scheduled,
                    BlockOrder::ColumnMajor,
                    setup_contribution_mode,
                )?;
                verify_fold::<F, E, T>(setup, transcript, prepared)?;
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
