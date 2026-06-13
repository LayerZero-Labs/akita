use super::*;
#[cfg(feature = "zk")]
use akita_r1cs::zk_ext_mask_lc_at;
#[cfg(feature = "zk")]
use akita_types::dispatch_ring_dim_result;
#[cfg(not(feature = "zk"))]
use akita_types::dispatch_ring_dim_result;
use akita_types::{ClaimIncidenceSummary, CommitmentRouting};

/// Prepare one recursive fold level for relation verification.
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
    next_fold_level_params: Option<&'a LevelParams>,
    transcript: &mut T,
    current_state: &'a RecursiveVerifierState<'a, F, L>,
    lp: &'a LevelParams,
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
    let incidence = ClaimIncidenceSummary::from_point_polys(num_vars, vec![1; num_claims])?;
    let row_coefficients = vec![L::one()];
    let challenge_points = vec![current_state.opening_point.clone()];
    let openings = vec![current_state.opening];
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
        ..
    } = verify_fold_eor::<F, L, T, D>(
        proof.extension_opening_reduction(),
        &[],
        &challenge_points,
        &openings,
        &row_coefficients,
        &incidence,
        current_state.basis,
        lp,
        block_order,
        true,
        transcript,
        #[cfg(feature = "zk")]
        &opening_masks,
        #[cfg(feature = "zk")]
        "recursive extension-opening partial claim",
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

    let ring_opening_points = vec![prepared_point.ring_opening_point.clone()];
    let ring_multiplier_points = vec![prepared_point.ring_multiplier_point.clone()];
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
    let commitment_routing = CommitmentRouting::from_recursive_multipoint(num_claims)?;
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
    let (trace_eval_target, trace_eval_scale) = (current_state.opening, L::one());

    Ok(PreparedFoldReplay {
        lp,
        m_row_layout,
        v: v_typed.to_vec(),
        commitment_rows: commitment_u,
        row_coefficients,
        incidence,
        commitment_routing,
        ring_opening_points,
        ring_multiplier_points,
        w_len,
        stage1: stage1_proof,
        stage2,
        next_w_commitment,
        terminal_replay,
        stage3,
        trace_prepared_points: vec![prepared_point.clone()],
        trace_block_openings: Vec::new(),
        trace_eval_target,
        trace_eval_scale,
        trace_claim_scales: None,
        trace_basis: current_state.basis,
    })
}

/// Verify all recursive fold levels after the root proof.
///
/// The supplied `schedule` is the already-selected public schedule for this
/// proof shape. This function checks that each proof level matches that
/// schedule, dispatches to the corresponding ring dimension, and threads the
/// verifier state to the next recursive commitment.
///
/// # Errors
///
/// Returns an error if the schedule is malformed for the supplied proof,
/// decoded proof dimensions do not match, any fold-level verifier rejects, or
/// the recursive witness handoff has the wrong shape.
#[allow(clippy::too_many_arguments)]
fn verify_suffix<'a, F, L, T>(
    steps: &'a [AkitaLevelProof<F, L>],
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    schedule: &Schedule,
    num_vars: usize,
    mut current_state: RecursiveVerifierState<'a, F, L>,
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
        let Some(Step::Fold(scheduled_step)) = schedule.steps.get(level_index) else {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule is missing fold step at level {level_index}"
            )));
        };
        let next_w_len = scheduled_step.next_w_len;
        let schedule_inputs = AkitaScheduleInputs {
            num_vars,
            level: level_index,
            current_w_len: current_state.w_len,
        };
        let (current_lp, next_params) = scheduled_fold_execution(
            schedule,
            level_index,
            schedule_inputs,
            current_state.log_basis,
        )?;
        let successor_is_fold = matches!(schedule.steps.get(level_index + 1), Some(Step::Fold(_)));
        let level_d = current_lp.ring_dimension;

        match step {
            AkitaLevelProof::Intermediate { .. } => {
                let level_proof = step;
                if !successor_is_fold {
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
                        Some(&next_params),
                        transcript,
                        &current_state,
                        &current_lp,
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
                    w_ring_element_count_with_counts::<F>(&current_lp, 1, 1, 1, 1)?
                        .checked_mul(level_d)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("next witness length overflow".to_string())
                        })?;
                if computed_next_w_len != next_w_len {
                    return Err(AkitaError::InvalidProof);
                }
                current_state = RecursiveVerifierState {
                    opening_point: challenges,
                    opening: level_proof.next_w_eval(),
                    #[cfg(feature = "zk")]
                    opening_mask: zk_ext_mask_lc_at::<F, L>(
                        *zk_hiding_cursor - <L as ExtField<F>>::EXT_DEGREE,
                    ),
                    commitment: level_proof.next_w_commitment(),
                    basis: BasisMode::Lagrange,
                    w_len: next_w_len,
                    log_basis: next_params.log_basis,
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
                        None,
                        transcript,
                        &current_state,
                        &current_lp,
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
                // Invariant: a terminal step implies the scheduled successor is
                // a Direct step. The trailing-Direct witness shape is already
                // validated in `verify_folded_batched_proof` before this loop.
                if successor_is_fold {
                    return Err(AkitaError::InvalidProof);
                }
            }
        }
    }

    Ok(())
}

/// Verify the folded-root branch of a batched opening proof.
///
/// The caller owns config-backed schedule selection and passes the derived
/// root verifier layout plus the first recursive-level params. This function
/// owns the fold-root proof-shape checks, root opening preparation, root
/// transcript replay, and recursive suffix handoff.
///
/// # Errors
///
/// Returns an error if the proof is not a folded-root proof, the schedule does
/// not match the proof shape, the root proof rejects, or a recursive suffix
/// level rejects.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_folded_batched_proof<F, E, C, T, const D: usize>(
    proof: &AkitaBatchedProof<F, C>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    opening_points: &[&[E]],
    openings: &[E],
    commitments: &[RingCommitment<F, D>],
    incidence_summary: &ClaimIncidenceSummary,
    basis: BasisMode,
    schedule: &Schedule,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
{
    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        return Err(AkitaError::InvalidProof);
    };
    let root_lp = &root_step.params;
    let total_fold_levels = schedule_num_fold_levels(schedule);
    let terminal_direct = schedule
        .steps
        .last()
        .and_then(|step| match step {
            Step::Direct(direct) => Some(direct),
            Step::Fold(_) => None,
        })
        .ok_or(AkitaError::InvalidProof)?;

    #[cfg(feature = "zk")]
    let mut zk_relations = ZkRelationAccumulator::new();
    #[cfg(feature = "zk")]
    {
        if proof.zk_hiding.u_blind.is_empty() || proof.zk_hiding.hiding_witness.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        verify_zk_hiding_commitment::<F, D>(setup, root_lp, &proof.zk_hiding)?;
        transcript.append_serde(ABSORB_ZK_HIDING_COMMITMENT, &proof.zk_hiding.u_blind);
    }
    #[cfg(feature = "zk")]
    let mut zk_hiding_cursor = 0usize;

    match &proof.root {
        akita_types::AkitaBatchedRootProof::ZeroFold { .. } => Err(AkitaError::InvalidProof),
        akita_types::AkitaBatchedRootProof::Terminal(terminal) => {
            // 1-fold case: the root itself is the terminal fold. No recursive
            // suffix follows.
            if total_fold_levels != 1 {
                return Err(AkitaError::InvalidProof);
            }
            if terminal.final_witness().shape() != terminal_direct.witness_shape {
                return Err(AkitaError::InvalidProof);
            }
            verify_root::<F, E, C, T, D>(
                &proof.root,
                setup,
                transcript,
                opening_points,
                openings,
                commitments,
                incidence_summary,
                basis,
                root_lp,
                setup_contribution_mode,
                None,
                root_step.next_w_len,
                #[cfg(feature = "zk")]
                &mut zk_hiding_cursor,
                #[cfg(feature = "zk")]
                &mut zk_relations,
            )?;
            Ok(())
        }
        akita_types::AkitaBatchedRootProof::Fold(fold_root) => {
            let expected_recursive_levels = total_fold_levels
                .checked_sub(1)
                .ok_or(AkitaError::InvalidProof)?;
            if proof.steps.len() != expected_recursive_levels {
                return Err(AkitaError::InvalidProof);
            }

            let terminal_step = proof
                .steps
                .last()
                .and_then(|step| match step {
                    AkitaLevelProof::Terminal { .. } => Some(step),
                    AkitaLevelProof::Intermediate { .. } => None,
                })
                .ok_or(AkitaError::InvalidProof)?;
            if terminal_step
                .stage2()
                .final_witness()
                .ok_or(AkitaError::InvalidProof)?
                .shape()
                != terminal_direct.witness_shape
            {
                return Err(AkitaError::InvalidProof);
            }

            let first_recursive_params =
                scheduled_next_level_params(schedule, 1).map_err(|_| AkitaError::InvalidProof)?;
            let root_stage2 = fold_root
                .stage2
                .as_intermediate()
                .ok_or(AkitaError::InvalidProof)?;
            let root_challenges = verify_root::<F, E, C, T, D>(
                &proof.root,
                setup,
                transcript,
                opening_points,
                openings,
                commitments,
                incidence_summary,
                basis,
                root_lp,
                setup_contribution_mode,
                Some(&first_recursive_params),
                root_step.next_w_len,
                #[cfg(feature = "zk")]
                &mut zk_hiding_cursor,
                #[cfg(feature = "zk")]
                &mut zk_relations,
            )?;

            let first_level_d = first_recursive_params.ring_dimension;
            if !root_stage2.next_w_commitment.can_decode_vec(first_level_d) {
                return Err(AkitaError::InvalidProof);
            }

            let current_state = RecursiveVerifierState {
                opening_point: root_challenges,
                opening: root_stage2.next_w_eval(),
                #[cfg(feature = "zk")]
                opening_mask: zk_ext_mask_lc_at::<F, C>(
                    zk_hiding_cursor - <C as ExtField<F>>::EXT_DEGREE,
                ),
                commitment: &root_stage2.next_w_commitment,
                basis: BasisMode::Lagrange,
                w_len: root_step.next_w_len,
                log_basis: first_recursive_params.log_basis,
            };
            verify_suffix::<F, C, T>(
                &proof.steps,
                setup,
                transcript,
                schedule,
                incidence_summary.num_vars(),
                current_state,
                setup_contribution_mode,
                #[cfg(feature = "zk")]
                &mut zk_hiding_cursor,
                #[cfg(feature = "zk")]
                &mut zk_relations,
            )?;
            Ok(())
        }
    }?;

    #[cfg(feature = "zk")]
    {
        if zk_hiding_cursor != proof.zk_hiding.hiding_witness.len() {
            return Err(AkitaError::InvalidProof);
        }
        let lifted = lift_hiding_witness::<F, C>(&proof.zk_hiding.hiding_witness);
        zk_relations.verify_all(&lifted)?;
    }

    Ok(())
}
