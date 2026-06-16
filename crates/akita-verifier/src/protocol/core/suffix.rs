use super::*;
#[cfg(feature = "zk")]
use akita_r1cs::zk_ext_mask_lc_at;
#[cfg(feature = "zk")]
use akita_types::dispatch_ring_dim_result;
#[cfg(not(feature = "zk"))]
use akita_types::dispatch_ring_dim_result;
use akita_types::OpeningBatch;

/// Verifier state carried between suffix fold levels.
pub(super) struct SuffixVerifierState<'a, F: FieldCore, L: FieldCore> {
    /// Current opening point for the committed suffix witness.
    pub opening_point: Vec<L>,
    /// Claimed opening value for the current commitment.
    pub opening: L,
    /// Hidden mask added to `opening` in the public proof.
    #[cfg(feature = "zk")]
    pub opening_mask: ZkR1csLinearCombination<L>,
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
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<L>,
) -> Result<PreparedFoldReplay<'a, F, L, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    L: RingSubfieldEncoding<F>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
{
    let lp = &scheduled.params;
    let next_fold_level_params = (!scheduled.is_terminal).then_some(&scheduled.next_params);
    let alpha_bits = validate_level_dispatch::<D>(lp)?;
    let m_row_layout = proof.m_row_layout();
    let v_typed = proof.v_as_ring_slice::<D>()?;
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;
    if current_state.opening_point.len() < alpha_bits {
        return Err(AkitaError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    current_state
        .commitment
        .append_as_ring_slice::<T, D>(ABSORB_COMMITMENT, transcript)?;
    let num_claims = 1usize;
    let num_vars = lp.recursive_opening_num_vars()?;
    let opening_batch = OpeningBatch::same_point(num_vars, num_claims)?;
    let openings = vec![current_state.opening];
    if proof.extension_opening_reduction().is_some() {
        append_claim_values_to_transcript::<F, L, T>(&openings, transcript);
    }
    let row_coefficients = vec![L::one()];
    #[cfg(feature = "zk")]
    let opening_masks = vec![Some(&current_state.opening_mask)];
    let FoldEorReplay {
        prepared_points,
        #[cfg(not(feature = "zk"))]
            reduction_challenges: _,
        #[cfg(feature = "zk")]
            reduction_challenges: _,
        #[cfg(not(feature = "zk"))]
            final_relation: eor_trace_final,
        #[cfg(feature = "zk")]
            final_relation: zk_eor_final,
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
        #[cfg(feature = "zk")]
        &opening_masks,
        #[cfg(feature = "zk")]
        "suffix extension-opening partial claim",
        #[cfg(feature = "zk")]
        zk_hiding_cursor,
        #[cfg(feature = "zk")]
        zk_relations,
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
        None => w_ring_element_count_with_counts::<F>(lp, 1, 1, num_claims, num_claims)?
            .checked_mul(D)
            .ok_or_else(|| AkitaError::InvalidSetup("next witness length overflow".to_string()))?,
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
    #[cfg(not(feature = "zk"))]
    let (trace_eval_target, trace_eval_scale) = match eor_trace_final.as_ref() {
        Some((final_claim, factors_by_point)) => (
            *final_claim,
            *factors_by_point.first().ok_or(AkitaError::InvalidProof)?,
        ),
        None => (current_state.opening, L::one()),
    };
    #[cfg(feature = "zk")]
    let (trace_eval_target, trace_eval_scale, trace_eval_target_mask) =
        if let Some((final_claim, factors_by_point)) = zk_eor_final.as_ref() {
            (
                final_claim.public,
                *factors_by_point.first().ok_or(AkitaError::InvalidProof)?,
                final_claim.mask.clone(),
            )
        } else {
            (
                current_state.opening,
                L::one(),
                current_state.opening_mask.clone(),
            )
        };

    Ok(PreparedFoldReplay {
        lp,
        m_row_layout,
        v: v_typed.to_vec(),
        commitment_rows: commitment_u,
        row_coefficients,
        opening_batch,
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
        #[cfg(feature = "zk")]
        trace_eval_target_mask,
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
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<L>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    L: RingSubfieldEncoding<F>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
{
    for (offset, step) in steps.iter().enumerate() {
        let level_index = offset + 1;
        let scheduled = schedule.get_execution_schedule(level_index)?;
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
                if !current_state.commitment.can_decode_vec(level_d)
                    || !level_proof.v().can_decode_vec(level_d)
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
                        #[cfg(feature = "zk")]
                        zk_hiding_cursor,
                        #[cfg(feature = "zk")]
                        zk_relations,
                    )?;
                    verify_fold::<F, L, T, D_LEVEL>(
                        setup,
                        transcript,
                        prepared,
                        #[cfg(feature = "zk")]
                        zk_hiding_cursor,
                        #[cfg(feature = "zk")]
                        zk_relations,
                    )
                })?;

                let next_level_d = next_params.ring_dimension;
                if next_level_d == 0
                    || !level_proof.next_w_commitment().can_decode_vec(next_level_d)
                {
                    return Err(AkitaError::InvalidProof);
                }
                let computed_next_w_len =
                    w_ring_element_count_with_counts::<F>(current_lp, 1, 1, 1, 1)?
                        .checked_mul(level_d)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("next witness length overflow".to_string())
                        })?;
                scheduled.validate_next_w_len(computed_next_w_len)?;
                current_state = SuffixVerifierState {
                    opening_point: challenges,
                    opening: level_proof.next_w_eval(),
                    #[cfg(feature = "zk")]
                    opening_mask: zk_ext_mask_lc_at::<F, L>(
                        *zk_hiding_cursor - <L as ExtField<F>>::EXT_DEGREE,
                    ),
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
                        #[cfg(feature = "zk")]
                        zk_hiding_cursor,
                        #[cfg(feature = "zk")]
                        zk_relations,
                    )?;
                    verify_fold::<F, L, T, D_LEVEL>(
                        setup,
                        transcript,
                        prepared,
                        #[cfg(feature = "zk")]
                        zk_hiding_cursor,
                        #[cfg(feature = "zk")]
                        zk_relations,
                    )
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
