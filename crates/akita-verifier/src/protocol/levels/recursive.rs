#[cfg(not(feature = "zk"))]
use super::extension_opening_reduction::ExtensionOpeningReductionVerifier;
use super::*;
#[cfg(feature = "zk")]
use akita_algebra::EqPolynomial;
#[cfg(feature = "zk")]
use akita_r1cs::{zk_ext_mask_lc, zk_ext_mask_lc_at, zk_row_masks_from_column_masks};
#[cfg(feature = "zk")]
use akita_types::tensor_equality_factor_eval_at_point;
#[cfg(not(feature = "zk"))]
use akita_types::{check_tensor_extension_opening_claim, recover_ring_subfield_inner_product};
use akita_types::{
    tensor_reduction_claim_from_rows, tensor_row_partials_from_columns, ClaimIncidenceSummary,
    CommitmentRouting, RingRelationInstance,
};

enum RecursiveFoldProofView<'a, F: FieldCore, L: FieldCore> {
    Intermediate {
        proof: &'a AkitaLevelProof<F, L>,
        next_fold_level_params: &'a LevelParams,
    },
    Terminal {
        proof: &'a TerminalLevelProof<F, L>,
        final_w_len: usize,
    },
}

impl<F: FieldCore, L: FieldCore> RecursiveFoldProofView<'_, F, L> {
    fn y_rings_typed<const D: usize>(&self) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        match self {
            Self::Intermediate { proof, .. } => proof.try_y_rings_typed::<D>(),
            Self::Terminal { proof, .. } => proof.try_y_rings_typed::<D>(),
        }
    }
    fn extension_opening_reduction(&self) -> Option<&ExtensionOpeningReductionProof<L>> {
        match self {
            Self::Intermediate { proof, .. } => proof.extension_opening_reduction.as_ref(),
            Self::Terminal { proof, .. } => proof.extension_opening_reduction.as_ref(),
        }
    }

    fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal { .. })
    }

    fn next_fold_level_params<'a>(&'a self, current_lp: &'a LevelParams) -> &'a LevelParams {
        match self {
            Self::Intermediate {
                next_fold_level_params,
                ..
            } => next_fold_level_params,
            Self::Terminal { .. } => current_lp,
        }
    }
}

/// Verify one recursive fold level.
///
/// The returned challenges become the opening point for the next level. Terminal
/// levels absorb the cleartext final witness instead of a next-witness
/// commitment and run stage-2 in relation-only mode.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, the public trace check
/// fails, ring-switch replay fails, or a sumcheck verifier rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[tracing::instrument(skip_all, name = "verify_recursive_level")]
fn verify_recursive_level<F, L, T, const D: usize>(
    proof: RecursiveFoldProofView<'_, F, L>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F, L>,
    lp: &LevelParams,
    block_order: BlockOrder,
    setup_contribution_mode: SetupContributionMode,
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<L>,
) -> Result<Vec<L>, AkitaError>
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
    let is_last = proof.is_terminal();
    let next_fold_level_params = proof.next_fold_level_params(lp);
    let y_rings = proof.y_rings_typed::<D>()?;
    let v_typed_owned: Vec<CyclotomicRing<F, D>>;
    let v_typed: &[CyclotomicRing<F, D>] = match &proof {
        RecursiveFoldProofView::Intermediate { proof, .. } => proof.v.as_ring_slice::<D>()?,
        RecursiveFoldProofView::Terminal { .. } => {
            v_typed_owned = Vec::new();
            &v_typed_owned
        }
    };
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;
    if current_state.opening_point.len() < alpha_bits {
        return Err(AkitaError::InvalidSetup(
            "opening point length underflow".to_string(),
        ));
    }
    current_state
        .commitment
        .append_as_ring_slice::<T, D>(ABSORB_COMMITMENT, transcript)?;
    if y_rings.len() != 1 {
        return Err(AkitaError::InvalidProof);
    }
    // The zk EOR final relation consumes the shared y-ring opening masks, so it
    // stays in this outer flow rather than inside the sumcheck driver. It carries
    // `(final_claim_lc, factor)` for that deferred relation.
    #[cfg(feature = "zk")]
    let mut zk_eor_final: Option<(ZkR1csLinearCombination<L>, L)> = None;
    let reduction_check = if <L as ExtField<F>>::EXT_DEGREE == 1 {
        if proof.extension_opening_reduction().is_some() {
            return Err(AkitaError::InvalidProof);
        }
        None
    } else {
        let reduction = proof
            .extension_opening_reduction()
            .ok_or(AkitaError::InvalidProof)?;
        let (split_bits, width) = {
            let width = <L as ExtField<F>>::EXT_DEGREE;
            if width == 0 || !width.is_power_of_two() {
                return Err(AkitaError::InvalidProof);
            }
            (width.trailing_zeros() as usize, width)
        };
        if split_bits > current_state.opening_point.len() || reduction.partials.len() != width {
            return Err(AkitaError::InvalidProof);
        }
        #[cfg(not(feature = "zk"))]
        check_tensor_extension_opening_claim::<F, L>(
            &current_state.opening_point,
            current_state.opening,
            &reduction.partials,
        )?;
        #[cfg(feature = "zk")]
        let partial_masks = (0..width)
            .map(|_| zk_ext_mask_lc::<F, L>(zk_hiding_cursor))
            .collect::<Vec<_>>();
        #[cfg(feature = "zk")]
        {
            let head_weights =
                EqPolynomial::<L>::evals(&current_state.opening_point[..split_bits])?;
            let true_opening = ZkRelationAccumulator::unmask_lc(
                current_state.opening,
                &current_state.opening_mask,
            );
            let mut residual = ZkR1csLinearCombination::zero();
            residual.add_scaled(-L::one(), &true_opening);
            for ((&partial, mask), weight) in reduction
                .partials
                .iter()
                .zip(partial_masks.iter())
                .zip(head_weights)
            {
                let true_partial = ZkRelationAccumulator::unmask_lc(partial, mask);
                residual.add_scaled(weight, &true_partial);
            }
            zk_push_linear_zero(
                zk_relations,
                "recursive extension-opening partial claim",
                residual,
            )?;
        }
        for partial in &reduction.partials {
            append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
        }
        let row_partials = tensor_row_partials_from_columns::<F, L>(&reduction.partials)?;
        let eta = (0..split_bits)
            .map(|_| sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
            .collect::<Vec<_>>();
        let input_claim = tensor_reduction_claim_from_rows::<F, L>(&row_partials, &eta)?;
        #[cfg(feature = "zk")]
        let input_claim_mask = {
            let mut input_claim_mask = ZkR1csLinearCombination::zero();
            let row_masks = zk_row_masks_from_column_masks::<F, L>(&partial_masks)?;
            for (weight, row_mask) in EqPolynomial::<L>::evals(&eta)?.into_iter().zip(row_masks) {
                input_claim_mask.add_scaled(weight, &row_mask);
            }
            input_claim_mask
        };
        let tail_point = &current_state.opening_point[split_bits..];
        #[cfg(not(feature = "zk"))]
        {
            let basis = current_state.basis;
            let eor_verifier = ExtensionOpeningReductionVerifier::<F, L, D>::new(
                tail_point.len(),
                input_claim,
                eta,
                vec![(&y_rings[0], tail_point.to_vec())],
                Box::new(
                    move |rho: &[L]| -> Result<CyclotomicRing<F, D>, AkitaError> {
                        let protocol_point = ring_subfield_packed_extension_opening_point::<F, L, D>(
                            rho.len(),
                            rho,
                        )?;
                        Ok(prepare_recursive_opening_point_ext::<F, L, D>(
                            &protocol_point,
                            basis,
                            lp,
                            alpha_bits,
                            block_order,
                        )?
                        .inner_reduction)
                    },
                ),
            );
            let rho = eor_verifier.verify::<F, T, _>(&reduction.sumcheck, transcript, |tr| {
                sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;
            Some(rho)
        }
        #[cfg(feature = "zk")]
        {
            let (final_claim_lc, challenges) =
                verify_zk_extension_opening_reduction_sumcheck::<F, L, T, _>(
                    input_claim,
                    tail_point.len(),
                    &reduction.sumcheck_proof_masked,
                    input_claim_mask,
                    transcript,
                    |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                    zk_relations,
                    zk_hiding_cursor,
                )?;
            let factor =
                tensor_equality_factor_eval_at_point::<F, L>(tail_point, &eta, &challenges)?;
            zk_eor_final = Some((final_claim_lc, factor));
            Some(challenges)
        }
    };
    let protocol_point = match &reduction_check {
        Some(rho) => ring_subfield_packed_extension_opening_point::<F, L, D>(rho.len(), rho)?,
        None => current_state.opening_point.clone(),
    };
    let prepared_points = vec![prepare_recursive_opening_point_ext::<F, L, D>(
        &protocol_point,
        current_state.basis,
        lp,
        alpha_bits,
        block_order,
    )?];
    for prepared_point in &prepared_points {
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    // See the root verifier comment at `levels.rs` on absorbing `y_rings` before
    // EOR for the recommended (transcript-breaking) reorder and the invariant
    // it relies on.
    for y_ring in &y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }
    #[cfg(feature = "zk")]
    let y_masks = zk_base_mask_lcs::<L>(y_rings.len() * D, zk_hiding_cursor);

    #[cfg(not(feature = "zk"))]
    let internal_claims = y_rings
        .iter()
        .zip(prepared_points.iter())
        .map(|(y_ring, prepared_point)| {
            recover_ring_subfield_inner_product::<F, L, D>(y_ring, &prepared_point.inner_reduction)
        })
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(feature = "zk")]
    {
        let y_opening = zk_recovered_y_ring_lc::<F, L, D>(
            &y_rings[0],
            y_masks.get(..D).ok_or(AkitaError::InvalidProof)?,
            &prepared_points[0].inner_reduction,
        )?;
        if let Some((final_claim, factor)) = &zk_eor_final {
            let mut residual = final_claim.clone();
            residual.add_scaled(-*factor, &y_opening);
            zk_push_linear_zero(
                zk_relations,
                "recursive extension-opening reduction output",
                residual,
            )?;
        } else {
            let true_opening = ZkRelationAccumulator::unmask_lc(
                current_state.opening,
                &current_state.opening_mask,
            );
            let mut residual = y_opening;
            residual.add_scaled(-L::one(), &true_opening);
            zk_push_linear_zero(zk_relations, "recursive y-ring opening relation", residual)?;
        }
    }
    // When `reduction_check` is `Some`, the non-zk EOR final relation is enforced
    // inside the sumcheck driver via `expected_output_claim`.
    #[cfg(not(feature = "zk"))]
    if reduction_check.is_none() && internal_claims[0] != current_state.opening {
        return Err(AkitaError::InvalidProof);
    }

    let ring_opening_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_opening_point.clone())
        .collect::<Vec<_>>();
    let ring_multiplier_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_multiplier_point.clone())
        .collect::<Vec<_>>();
    let num_claims = y_rings.len();
    let w_len = match &proof {
        RecursiveFoldProofView::Terminal { final_w_len, .. } => *final_w_len,
        RecursiveFoldProofView::Intermediate { .. } => {
            w_ring_element_count_with_counts::<F>(lp, 1, 1, num_claims, num_claims)?
                .checked_mul(D)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("next witness length overflow".to_string())
                })?
        }
    };
    let terminal_replay = if let RecursiveFoldProofView::Terminal {
        proof: terminal_proof,
        ..
    } = &proof
    {
        let layout =
            terminal_witness_segment_layout(lp, num_claims, num_claims, F::modulus_bits())?;
        Some(prepare_terminal_witness_replay::<F, T>(
            transcript,
            &terminal_proof.final_witness,
            w_len,
            layout,
        )?)
    } else {
        None
    };
    let stage1_challenges = derive_stage1_challenges::<F, T, D>(
        transcript,
        v_typed,
        lp.num_blocks,
        num_claims,
        lp,
        if is_last {
            MRowLayout::WithoutDBlock
        } else {
            MRowLayout::WithDBlock
        },
    )?;
    tracing::debug!(w_len, is_last, "verify ring_switch");
    let gamma = vec![L::one(); num_claims];
    let m_row_layout = if is_last {
        MRowLayout::WithoutDBlock
    } else {
        MRowLayout::WithDBlock
    };
    let num_vars = lp.recursive_opening_num_vars()?;
    let incidence = ClaimIncidenceSummary::from_point_polys(num_vars, vec![1; num_claims])?;
    let commitment_routing = CommitmentRouting::from_recursive_multipoint(num_claims)?;
    let (gamma_base, row_coefficient_rings) =
        RingRelationInstance::<F, D>::gamma_and_row_rings_from_coefficients::<L>(&gamma)?;
    let relation_instance = RingRelationInstance::new(
        m_row_layout,
        stage1_challenges.clone(),
        ring_opening_points.clone(),
        ring_multiplier_points.clone(),
        incidence,
        commitment_routing,
        gamma_base,
        row_coefficient_rings,
        y_rings.clone(),
        v_typed.to_vec(),
    )?;
    relation_instance.check_v_shape_for_level(lp)?;
    let ring_switch_replay = crate::protocol::ring_switch::RingSwitchReplay {
        relation: &relation_instance,
        row_coefficients: &gamma,
        lp,
    };

    let rs = match &proof {
        RecursiveFoldProofView::Intermediate { proof, .. } => ring_switch_verifier::<F, L, T, D>(
            &ring_switch_replay,
            w_len,
            proof.next_w_commitment(),
            transcript,
        )?,
        RecursiveFoldProofView::Terminal { .. } => {
            let replay = terminal_replay.as_ref().ok_or(AkitaError::InvalidProof)?;
            ring_switch_verifier_terminal::<F, L, T, D>(
                &ring_switch_replay,
                w_len,
                transcript,
                &replay.parts,
            )?
        }
    };
    let relation_claim = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        v_typed,
        commitment_u,
        &y_rings,
    )?;
    #[cfg(feature = "zk")]
    let relation_claim_mask =
        zk_relation_claim_mask_from_y_masks::<L, D>(&rs.tau1, rs.alpha, y_rings.len(), &y_masks)?;
    let stage1_proof = match &proof {
        RecursiveFoldProofView::Intermediate { proof, .. } => Some(&proof.stage1),
        RecursiveFoldProofView::Terminal { .. } => None,
    };
    let stage1_replay = verify_stage1_or_terminal::<F, L, T>(
        stage1_proof,
        &rs.tau0,
        rs.col_bits,
        rs.ring_bits,
        rs.b,
        transcript,
        #[cfg(feature = "zk")]
        zk_hiding_cursor,
        #[cfg(feature = "zk")]
        zk_relations,
    )?;
    let stage3_sumcheck_proof = match &proof {
        RecursiveFoldProofView::Intermediate { proof, .. } => stage3_sumcheck_proof_for_mode(
            setup_contribution_mode,
            proof.stage3_sumcheck_proof.as_ref(),
        )?,
        RecursiveFoldProofView::Terminal { .. } => None,
    };
    let stage2_replay = match &proof {
        RecursiveFoldProofView::Intermediate { proof, .. } => Stage2ProofReplay::Intermediate {
            next_w_eval: proof.stage2.next_w_eval(),
            #[cfg(not(feature = "zk"))]
            sumcheck: &proof.stage2.sumcheck_proof,
            #[cfg(feature = "zk")]
            sumcheck_masked: &proof.stage2.sumcheck_proof_masked,
        },
        RecursiveFoldProofView::Terminal {
            proof: terminal_proof,
            ..
        } => Stage2ProofReplay::Terminal {
            final_witness: &terminal_proof.final_witness,
            physical_w_len: w_len,
            #[cfg(not(feature = "zk"))]
            sumcheck: &terminal_proof.stage2_sumcheck,
            #[cfg(feature = "zk")]
            sumcheck_masked: &terminal_proof.stage2_sumcheck_proof_masked,
        },
    };
    let stage2_input = Stage2ReplayInput {
        setup,
        stage2: stage2_replay,
        stage1: stage1_replay,
        rs,
        relation_claim,
        #[cfg(feature = "zk")]
        relation_claim_mask,
        setup_sumcheck_proof: stage3_sumcheck_proof,
        next_fold_level_params,
        opening_points: &ring_opening_points,
        ring_multiplier_points: &ring_multiplier_points,
        v: v_typed,
        u: commitment_u,
        y_rings: &y_rings,
    };
    verify_stage2_and_setup_replay::<F, L, T, D>(
        transcript,
        stage2_input,
        #[cfg(feature = "zk")]
        zk_hiding_cursor,
        #[cfg(feature = "zk")]
        zk_relations,
    )
}

fn scheduled_recursive_verify_level<F: FieldCore, L: FieldCore>(
    schedule: &Schedule,
    level: usize,
    current_state: &RecursiveVerifierState<'_, F, L>,
) -> Result<(LevelParams, usize, Option<LevelParams>), AkitaError> {
    let Some(Step::Fold(step)) = schedule.steps.get(level) else {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule is missing fold step at level {level}"
        )));
    };
    if step.current_w_len != current_state.w_len || step.params.log_basis != current_state.log_basis
    {
        return Err(AkitaError::InvalidSetup(
            "scheduled recursive level did not match runtime state".to_string(),
        ));
    }
    let next_level_params = match schedule.steps.get(level + 1) {
        Some(Step::Fold(next_step)) => Some(next_step.params.clone()),
        Some(Step::Direct(_)) => None,
        None => {
            return Err(AkitaError::InvalidSetup(
                "schedule is missing successor step".to_string(),
            ))
        }
    };
    Ok((step.params.clone(), step.next_w_len, next_level_params))
}

macro_rules! dispatch_verifier_ring_dim_result {
    ($d:expr, |$D:ident| $body:expr) => {{
        match $d {
            32 => {
                const $D: usize = 32;
                $body
            }
            64 => {
                const $D: usize = 64;
                $body
            }
            128 => {
                const $D: usize = 128;
                $body
            }
            256 => {
                const $D: usize = 256;
                $body
            }
            _ => Err(AkitaError::InvalidProof),
        }
    }};
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
pub(crate) fn verify_batched_recursive_suffix<'a, F, L, T, const D: usize>(
    proof: &'a AkitaBatchedProof<F, L>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    schedule: &Schedule,
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
    let num_steps = proof.steps.len();
    if num_steps == 0 {
        // Root was itself the terminal — no suffix to verify.
        return Ok(());
    }
    for (offset, step) in proof.steps.iter().enumerate() {
        let level_index = offset + 1;
        let is_last = offset == num_steps - 1;
        let (current_lp, next_w_len, scheduled_next_params) =
            scheduled_recursive_verify_level(schedule, level_index, &current_state)?;
        let level_d = current_lp.ring_dimension;

        match step {
            AkitaProofStep::Intermediate(level_proof) => {
                let scheduled_next_params =
                    scheduled_next_params.ok_or(AkitaError::InvalidProof)?;
                if is_last {
                    // The terminal slot must be a Terminal variant.
                    return Err(AkitaError::InvalidProof);
                }
                if !current_state.commitment.can_decode_vec(level_d)
                    || !level_proof.y_ring.can_decode_vec(level_d)
                    || !level_proof.v.can_decode_vec(level_d)
                {
                    return Err(AkitaError::InvalidProof);
                }

                let challenges = dispatch_verifier_ring_dim_result!(level_d, |D_LEVEL| {
                    verify_recursive_level::<F, L, T, D_LEVEL>(
                        RecursiveFoldProofView::Intermediate {
                            proof: level_proof,
                            next_fold_level_params: &scheduled_next_params,
                        },
                        setup,
                        transcript,
                        &current_state,
                        &current_lp,
                        BlockOrder::ColumnMajor,
                        setup_contribution_mode,
                        #[cfg(feature = "zk")]
                        zk_hiding_cursor,
                        #[cfg(feature = "zk")]
                        zk_relations,
                    )
                })?;

                let next_level_d = scheduled_next_params.ring_dimension;
                if next_level_d == 0
                    || !level_proof.next_w_commitment().can_decode_vec(next_level_d)
                {
                    return Err(AkitaError::InvalidProof);
                }
                let y_ring_count = level_proof.y_ring.coeff_len() / level_d;
                let computed_next_w_len = w_ring_element_count_with_counts::<F>(
                    &current_lp,
                    1,
                    1,
                    y_ring_count,
                    y_ring_count,
                )?
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
                    log_basis: scheduled_next_params.log_basis,
                };
            }
            AkitaProofStep::Terminal(terminal_proof) => {
                if !is_last {
                    return Err(AkitaError::InvalidProof);
                }
                if !current_state.commitment.can_decode_vec(level_d)
                    || !terminal_proof.y_rings.can_decode_vec(level_d)
                {
                    return Err(AkitaError::InvalidProof);
                }
                dispatch_verifier_ring_dim_result!(level_d, |D_LEVEL| {
                    verify_recursive_level::<F, L, T, D_LEVEL>(
                        RecursiveFoldProofView::Terminal {
                            proof: terminal_proof,
                            final_w_len: next_w_len,
                        },
                        setup,
                        transcript,
                        &current_state,
                        &current_lp,
                        BlockOrder::ColumnMajor,
                        setup_contribution_mode,
                        #[cfg(feature = "zk")]
                        zk_hiding_cursor,
                        #[cfg(feature = "zk")]
                        zk_relations,
                    )
                })?;
                // Invariant: a terminal step implies the scheduled successor
                // is a Direct step (not a Fold), which `scheduled_recursive_verify_level`
                // signals by returning `None`. The trailing-`Direct` witness
                // shape is already validated in `verify_fold_batched_proof`
                // before this loop runs.
                if scheduled_next_params.is_some() {
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
pub(crate) fn verify_fold_batched_proof<F, E, C, T, const D: usize>(
    proof: &AkitaBatchedProof<F, C>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    opening_points: &[&[E]],
    openings: &[E],
    commitments: &[RingCommitment<F, D>],
    incidence_summary: &ClaimIncidenceSummary,
    basis: BasisMode,
    schedule: &Schedule,
    root_lp: &LevelParams,
    next_level_params: &LevelParams,
    root_next_fold_level_params: &LevelParams,
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
            if total_fold_levels != 1 || !proof.steps.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            if terminal.final_witness.shape() != terminal_direct.witness_shape {
                return Err(AkitaError::InvalidProof);
            }
            let y_coeff_len = terminal.y_rings.coeff_len();
            if !y_coeff_len.is_multiple_of(D) || y_coeff_len / D != opening_points.len() {
                return Err(AkitaError::InvalidProof);
            }
            verify_terminal_root_level::<F, E, C, T, D>(
                &terminal.y_rings,
                terminal.extension_opening_reduction.as_ref(),
                #[cfg(not(feature = "zk"))]
                &terminal.stage2_sumcheck,
                #[cfg(feature = "zk")]
                &terminal.stage2_sumcheck_proof_masked,
                &terminal.final_witness,
                root_step.next_w_len,
                setup,
                transcript,
                opening_points,
                openings,
                commitments,
                incidence_summary,
                basis,
                #[cfg(feature = "zk")]
                &mut zk_hiding_cursor,
                #[cfg(feature = "zk")]
                &mut zk_relations,
                root_lp,
                &root_step.params,
            )?;
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
        akita_types::AkitaBatchedRootProof::Fold(fold_root) => {
            let expected_recursive_levels = total_fold_levels
                .checked_sub(1)
                .ok_or(AkitaError::InvalidProof)?;
            if proof.steps.len() != expected_recursive_levels {
                return Err(AkitaError::InvalidProof);
            }
            let y_coeff_len = fold_root.y_rings.coeff_len();
            if !y_coeff_len.is_multiple_of(D) || y_coeff_len / D != opening_points.len() {
                return Err(AkitaError::InvalidProof);
            }

            let terminal_step = proof
                .steps
                .last()
                .and_then(|step| match step {
                    AkitaProofStep::Terminal(t) => Some(t),
                    AkitaProofStep::Intermediate(_) => None,
                })
                .ok_or(AkitaError::InvalidProof)?;
            if terminal_step.final_witness.shape() != terminal_direct.witness_shape {
                return Err(AkitaError::InvalidProof);
            }

            let root_challenges = verify_intermediate_root_level::<F, E, C, T, D>(
                &fold_root.y_rings,
                fold_root.extension_opening_reduction.as_ref(),
                &fold_root.v,
                &fold_root.stage1,
                &fold_root.stage2,
                fold_root.stage3_sumcheck_proof.as_ref(),
                setup,
                transcript,
                opening_points,
                openings,
                commitments,
                incidence_summary,
                basis,
                setup_contribution_mode,
                #[cfg(feature = "zk")]
                &mut zk_hiding_cursor,
                #[cfg(feature = "zk")]
                &mut zk_relations,
                root_lp,
                &root_step.params,
                root_next_fold_level_params,
            )?;

            let first_level_d = next_level_params.ring_dimension;
            if !fold_root
                .stage2
                .next_w_commitment
                .can_decode_vec(first_level_d)
            {
                return Err(AkitaError::InvalidProof);
            }

            let current_state = RecursiveVerifierState {
                opening_point: root_challenges,
                opening: fold_root.stage2.next_w_eval(),
                #[cfg(feature = "zk")]
                opening_mask: zk_ext_mask_lc_at::<F, C>(
                    zk_hiding_cursor - <C as ExtField<F>>::EXT_DEGREE,
                ),
                commitment: &fold_root.stage2.next_w_commitment,
                basis: BasisMode::Lagrange,
                w_len: root_step.next_w_len,
                log_basis: next_level_params.log_basis,
            };
            verify_batched_recursive_suffix::<F, C, T, D>(
                proof,
                setup,
                transcript,
                schedule,
                current_state,
                setup_contribution_mode,
                #[cfg(feature = "zk")]
                &mut zk_hiding_cursor,
                #[cfg(feature = "zk")]
                &mut zk_relations,
            )?;
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
    }
}
