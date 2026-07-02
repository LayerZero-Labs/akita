use super::*;
use akita_types::dispatch_ring_dim_result;
use akita_types::{terminal_witness_segment_layout, OpeningBatchShape};

/// Verifier state carried between suffix fold levels.
pub(super) struct SuffixVerifierState<'a, F: FieldCore, L: FieldCore> {
    /// Current opening point for the committed suffix witness.
    pub opening_point: Vec<L>,
    /// Claimed opening value for the current commitment.
    pub opening: L,
    /// Current suffix witness commitment.
    pub commitment: &'a FlatRingVec<F>,
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
#[tracing::instrument(skip_all, name = "prepare_fold_data")]
fn prepare_fold_data<'a, F, L, T, const D: usize>(
    proof: &'a AkitaLevelProof<F, L>,
    transcript: &mut T,
    current_state: &'a SuffixVerifierState<'a, F, L>,
    scheduled: &'a ExecutionSchedule,
    block_order: BlockOrder,
    setup_contribution_mode: SetupContributionMode,
) -> Result<PreparedFoldReplay<'a, F, L, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField + HalvingField,
    L: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let lp = &scheduled.params;
    let next_fold_level_params = (!scheduled.is_terminal).then_some(&scheduled.next_params);
    let alpha_bits = validate_level_dispatch::<D>(lp)?;
    let m_row_layout = scheduled_m_row_layout(scheduled);
    let (v_typed, compressed_v_payload) = match (m_row_layout, scheduled.compression.v.as_ref()) {
        (MRowLayout::WithDBlock, None) => (proof.v_as_ring_slice::<D>()?.to_vec(), None),
        (MRowLayout::WithoutDBlock | MRowLayout::WithoutCommitmentBlocks, Some(plan)) => {
            let AkitaLevelProof::Intermediate { v, .. } = proof else {
                return Err(AkitaError::InvalidProof);
            };
            if v.coeff_len() != plan.public_len {
                return Err(AkitaError::InvalidProof);
            }
            (Vec::new(), Some(v))
        }
        (MRowLayout::WithoutDBlock | MRowLayout::WithoutCommitmentBlocks, None) => {
            (Vec::new(), None)
        }
        (MRowLayout::WithDBlock, Some(_)) => return Err(AkitaError::InvalidProof),
    };
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;
    if current_state.opening_point.len() < alpha_bits {
        return Err(AkitaError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    current_state
        .commitment
        .append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;
    let num_claims = 1usize;
    let num_vars = lp.recursive_opening_num_vars()?;
    let opening_batch = OpeningBatchShape::new(num_vars, num_claims)?;
    let openings = vec![current_state.opening];
    let row_coefficients = vec![L::one()];
    let FoldEorReplay {
        prepared_points,
        reduction_challenges: _,
        final_relation: eor_trace_final,
        ..
    } = verify_fold_eor::<F, L, T, D>(
        proof.extension_opening_reduction(),
        &[],
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
        append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
    }

    let w_len = match proof.final_w_len() {
        Some(final_w_len) => final_w_len,
        None => scheduled.next_w_len,
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
        None => (current_state.opening, L::one()),
    };

    let fold_grind_nonce = proof.fold_grind_nonce();
    let replay_opening_batch = VerifierOpeningBatch::from_shape_and_groups(
        current_state.opening_point.as_slice(),
        opening_batch,
        vec![CommitmentGroup {
            claims: openings,
            commitment: commitment_u,
        }],
    )?;
    Ok(PreparedFoldReplay {
        lp,
        m_row_layout,
        fold_grind_nonce,
        v: v_typed,
        v_compression: scheduled.compression.v.as_ref(),
        compressed_v_payload,
        opening_batch: replay_opening_batch,
        row_coefficients,
        ring_opening_point: prepared_point.ring_opening_point.clone(),
        ring_multiplier_point: prepared_point.ring_multiplier_point.clone(),
        w_len,
        stage1: stage1_proof,
        stage2,
        next_w_commitment,
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
pub(super) fn verify_suffix<'a, F, L, T>(
    steps: &'a [AkitaLevelProof<F, L>],
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    schedule: &Schedule,
    mut current_state: SuffixVerifierState<'a, F, L>,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField + HalvingField,
    L: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    for (offset, step) in steps.iter().enumerate() {
        let level_index = offset + 1;
        let scheduled = schedule.get_execution_schedule(level_index)?;
        reject_active_b_side_compression(&scheduled)?;
        scheduled.validate_current_w_len(current_state.w_len)?;
        let current_lp = &scheduled.params;
        let next_params = &scheduled.next_params;
        let next_w_len = scheduled.next_w_len;
        let level_d = current_lp.ring_dimension;

        match step {
            AkitaLevelProof::Intermediate { .. } => {
                let level_proof = step;
                if scheduled.is_terminal {
                    return Err(AkitaError::InvalidProof);
                }
                let m_row_layout = scheduled_m_row_layout(&scheduled);
                if !current_state.commitment.can_decode_vec(level_d)
                    || match (m_row_layout, scheduled.compression.v.as_ref()) {
                        (MRowLayout::WithDBlock, None) => !level_proof.v().can_decode_vec(level_d),
                        (
                            MRowLayout::WithoutDBlock | MRowLayout::WithoutCommitmentBlocks,
                            Some(plan),
                        ) => level_proof.v().coeff_len() != plan.public_len,
                        (MRowLayout::WithoutDBlock | MRowLayout::WithoutCommitmentBlocks, None) => {
                            false
                        }
                        (MRowLayout::WithDBlock, Some(_)) => true,
                    }
                {
                    return Err(AkitaError::InvalidProof);
                }

                let challenges = dispatch_ring_dim_result!(level_d, |D_LEVEL| {
                    let prepared = prepare_fold_data::<F, L, T, D_LEVEL>(
                        level_proof,
                        transcript,
                        &current_state,
                        &scheduled,
                        BlockOrder::ColumnMajor,
                        setup_contribution_mode,
                    )?;
                    verify_fold::<F, L, T, D_LEVEL>(setup, transcript, prepared)
                })?;

                let next_level_d = next_params.ring_dimension;
                if next_level_d == 0
                    || !level_proof.next_w_commitment().can_decode_vec(next_level_d)
                {
                    return Err(AkitaError::InvalidProof);
                }
                scheduled.validate_next_w_len(next_w_len)?;
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
                if !current_state.commitment.can_decode_vec(level_d) {
                    return Err(AkitaError::InvalidProof);
                }
                if terminal_proof
                    .final_w_len()
                    .ok_or(AkitaError::InvalidProof)?
                    != next_w_len
                {
                    return Err(AkitaError::InvalidProof);
                }
                dispatch_ring_dim_result!(level_d, |D_LEVEL| {
                    let prepared = prepare_fold_data::<F, L, T, D_LEVEL>(
                        terminal_proof,
                        transcript,
                        &current_state,
                        &scheduled,
                        BlockOrder::ColumnMajor,
                        setup_contribution_mode,
                    )?;
                    verify_fold::<F, L, T, D_LEVEL>(setup, transcript, prepared)
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
