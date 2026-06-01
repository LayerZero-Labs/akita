use super::*;

/// Verify one intermediate recursive fold level.
///
/// The returned challenges become the opening point for the next level.
///
/// # Errors
///
/// Returns an error if the level proof shape is inconsistent, the public
/// trace check fails, ring-switch replay fails, or either sumcheck verifier
/// rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[tracing::instrument(skip_all, name = "verify_intermediate_level")]
pub(crate) fn verify_intermediate_level<F, L, T, const D: usize>(
    level_proof: &AkitaLevelProof<F, L>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F, L>,
    lp: &LevelParams,
    block_order: BlockOrder,
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
    verify_one_level_inner::<F, L, T, D>(
        FoldProofView::Intermediate(level_proof),
        setup,
        transcript,
        current_state,
        None,
        lp,
        block_order,
        #[cfg(feature = "zk")]
        zk_hiding_cursor,
        #[cfg(feature = "zk")]
        zk_relations,
    )
}

/// Verify one terminal recursive fold level.
///
/// At the terminal level the cleartext `final_witness` is absorbed via
/// [`ABSORB_SUMCHECK_W`] in place of a next-witness commitment, stage-1 is
/// skipped (packed-digit range is structurally enforced), and stage-2 runs
/// in relation-only mode.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, the public trace
/// check fails, ring-switch replay fails, or the stage-2 sumcheck verifier
/// rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[tracing::instrument(skip_all, name = "verify_terminal_level")]
pub(crate) fn verify_terminal_level<F, L, T, const D: usize>(
    terminal_proof: &TerminalLevelProof<F, L>,
    final_w_len: usize,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F, L>,
    lp: &LevelParams,
    block_order: BlockOrder,
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
    verify_one_level_inner::<F, L, T, D>(
        FoldProofView::Terminal(terminal_proof),
        setup,
        transcript,
        current_state,
        Some(final_w_len),
        lp,
        block_order,
        #[cfg(feature = "zk")]
        zk_hiding_cursor,
        #[cfg(feature = "zk")]
        zk_relations,
    )
}

enum FoldProofView<'a, F: FieldCore, L: FieldCore> {
    Intermediate(&'a AkitaLevelProof<F, L>),
    Terminal(&'a TerminalLevelProof<F, L>),
}

impl<F: FieldCore, L: FieldCore> FoldProofView<'_, F, L> {
    fn y_rings_typed<const D: usize>(&self) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        match self {
            Self::Intermediate(proof) => proof.try_y_rings_typed::<D>(),
            Self::Terminal(proof) => proof.try_y_rings_typed::<D>(),
        }
    }
    fn extension_opening_reduction(&self) -> Option<&ExtensionOpeningReductionProof<L>> {
        match self {
            Self::Intermediate(proof) => proof.extension_opening_reduction.as_ref(),
            Self::Terminal(proof) => proof.extension_opening_reduction.as_ref(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_one_level_inner<F, L, T, const D: usize>(
    proof: FoldProofView<'_, F, L>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F, L>,
    final_w_len: Option<usize>,
    lp: &LevelParams,
    block_order: BlockOrder,
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
    let is_last = matches!(proof, FoldProofView::Terminal(_));
    let y_rings = proof.y_rings_typed::<D>()?;
    let v_typed_owned: Vec<CyclotomicRing<F, D>>;
    let v_typed: &[CyclotomicRing<F, D>] = match &proof {
        FoldProofView::Intermediate(level_proof) => level_proof.v.as_ring_slice::<D>()?,
        FoldProofView::Terminal(_) => {
            v_typed_owned = Vec::new();
            &v_typed_owned
        }
    };
    let commitment = current_state.common_commitment()?;
    let carried_openings = current_state.carried_openings.as_slice();
    if carried_openings.is_empty()
        || carried_openings
            .iter()
            .any(|claim| claim.natural_len > claim.padded_len)
    {
        return Err(AkitaError::InvalidProof);
    }
    #[cfg(feature = "zk")]
    if carried_openings.len() != 1 {
        return Err(AkitaError::InvalidProof);
    }
    let commitment_u = commitment.as_ring_slice::<D>()?;
    commitment.append_as_ring_slice::<T, D>(ABSORB_COMMITMENT, transcript)?;
    if y_rings.len() != carried_openings.len() {
        return Err(AkitaError::InvalidProof);
    }
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
        if carried_openings.len() != 1 {
            return Err(AkitaError::InvalidProof);
        }
        let claim = &carried_openings[0];
        if split_bits > claim.opening_point.len() || reduction.partials.len() != width {
            return Err(AkitaError::InvalidProof);
        }
        #[cfg(not(feature = "zk"))]
        check_tensor_extension_opening_claim::<F, L>(
            &claim.opening_point,
            claim.opening,
            &reduction.partials,
        )?;
        #[cfg(feature = "zk")]
        let partial_masks = (0..width)
            .map(|_| zk_ext_mask_lc::<F, L>(zk_hiding_cursor))
            .collect::<Vec<_>>();
        #[cfg(feature = "zk")]
        {
            let head_weights = EqPolynomial::<L>::evals(&claim.opening_point[..split_bits])?;
            let true_opening = ZkRelationAccumulator::unmask_lc(claim.opening, &claim.opening_mask);
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
        let tail_point = &claim.opening_point[split_bits..];
        #[cfg(not(feature = "zk"))]
        let result = ExtensionOpeningReductionSumcheck::new(input_claim, tail_point.len())
            .verify::<F, _, _>(&reduction.sumcheck, transcript, |tr| {
                sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;
        #[cfg(feature = "zk")]
        let (final_claim_lc, challenges) =
            ExtensionOpeningReductionSumcheck::new(input_claim, tail_point.len())
                .verify_zk::<F, _, _>(
                    &reduction.sumcheck_proof_masked,
                    input_claim_mask,
                    transcript,
                    |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                    zk_relations,
                    zk_hiding_cursor,
                )?;
        #[cfg(feature = "zk")]
        let result_challenges = challenges;
        #[cfg(not(feature = "zk"))]
        let result_challenges = result.challenges;
        let factor =
            tensor_equality_factor_eval_at_point::<F, L>(tail_point, &eta, &result_challenges)?;
        #[cfg(feature = "zk")]
        {
            Some((final_claim_lc, factor, result_challenges))
        }
        #[cfg(not(feature = "zk"))]
        {
            Some((result.final_claim, factor, result_challenges))
        }
    };
    let mut prepared_points = Vec::with_capacity(carried_openings.len());
    for (claim_idx, claim) in carried_openings.iter().enumerate() {
        let protocol_point = match (&reduction_check, claim_idx) {
            (Some((_final_claim, _factor, rho)), 0) => {
                ring_subfield_packed_extension_opening_point::<F, L, D>(rho.len(), rho)?
            }
            (Some(_), _) => return Err(AkitaError::InvalidProof),
            (None, _) => claim.opening_point.clone(),
        };
        prepared_points.push(prepare_recursive_opening_point_ext::<F, L, D>(
            &protocol_point,
            claim.basis,
            lp,
            alpha_bits,
            block_order,
        )?);
    }
    for prepared_point in &prepared_points {
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
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
        match &reduction_check {
            Some((final_claim, factor, _rho)) => {
                let mut residual = final_claim.clone();
                residual.add_scaled(-*factor, &y_opening);
                zk_push_linear_zero(
                    zk_relations,
                    "recursive extension-opening reduction output",
                    residual,
                )?;
            }
            None => {
                let claim = &carried_openings[0];
                let true_opening =
                    ZkRelationAccumulator::unmask_lc(claim.opening, &claim.opening_mask);
                let mut residual = y_opening;
                residual.add_scaled(-L::one(), &true_opening);
                zk_push_linear_zero(zk_relations, "recursive y-ring opening relation", residual)?;
            }
        }
    }
    #[cfg(not(feature = "zk"))]
    match reduction_check {
        Some((final_claim, factor, _rho)) => {
            check_extension_opening_reduction_output(final_claim, internal_claims[0], factor)?;
        }
        None => {
            for (claim, &internal_claim) in carried_openings.iter().zip(internal_claims.iter()) {
                if internal_claim != claim.opening {
                    return Err(AkitaError::InvalidProof);
                }
            }
        }
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
    let w_len = if is_last {
        final_w_len.ok_or(AkitaError::InvalidProof)?
    } else {
        w_ring_element_count_with_counts::<F>(lp, 1, 1, num_claims, num_claims)?
            .checked_mul(D)
            .ok_or_else(|| AkitaError::InvalidSetup("next witness length overflow".to_string()))?
    };
    let terminal_replay = if let FoldProofView::Terminal(terminal_proof) = &proof {
        let layout = terminal_witness_segment_layout(lp, num_claims, num_claims)?;
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
            MRowLayout::Terminal
        } else {
            MRowLayout::Intermediate
        },
    )?;
    tracing::debug!(w_len, is_last, "verify ring_switch");
    let claim_to_point = (0..num_claims).collect::<Vec<_>>();
    let claim_to_point_poly = (0..num_claims).collect::<Vec<_>>();
    let claim_poly_indices = vec![0usize; num_claims];
    let gamma = vec![L::one(); num_claims];
    let num_polys_per_point = vec![1usize; num_claims];

    let rs = match &proof {
        FoldProofView::Intermediate(level_proof) => ring_switch_verifier::<F, L, T, { D }>(
            &ring_opening_points,
            &ring_multiplier_points,
            &claim_to_point,
            &stage1_challenges,
            w_len,
            level_proof.next_w_commitment(),
            transcript,
            lp,
            &num_polys_per_point,
            &claim_to_point_poly,
            &claim_poly_indices,
            &gamma,
            num_claims,
        )?,
        FoldProofView::Terminal(_) => {
            let replay = terminal_replay.as_ref().ok_or(AkitaError::InvalidProof)?;
            ring_switch_verifier_terminal::<F, L, T, { D }>(
                &ring_opening_points,
                &ring_multiplier_points,
                &claim_to_point,
                &stage1_challenges,
                w_len,
                transcript,
                &replay.parts,
                lp,
                &num_polys_per_point,
                &claim_to_point_poly,
                &claim_poly_indices,
                &gamma,
                num_claims,
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
    #[cfg(feature = "zk")]
    let mut s_claim_mask = ZkR1csLinearCombination::<L>::zero();
    let (batching_coeff, s_claim, r_stage1) = match &proof {
        FoldProofView::Intermediate(level_proof) => {
            let stage1 = &level_proof.stage1;
            let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
            let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
            #[cfg(not(feature = "zk"))]
            let r_stage1 = {
                let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
                stage1_verifier.verify::<F, T>(stage1, transcript)?
            };
            #[cfg(feature = "zk")]
            let r_stage1 = {
                let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
                let (r, mask) = stage1_verifier.verify::<F, T>(
                    stage1,
                    transcript,
                    zk_relations,
                    zk_hiding_cursor,
                )?;
                s_claim_mask = mask;
                r
            };
            transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
            let batching_coeff: L =
                sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
            (batching_coeff, stage1.s_claim, r_stage1)
        }
        FoldProofView::Terminal(_) => {
            let r_stage1 = vec![L::zero(); rs.col_bits + rs.ring_bits];
            (L::zero(), L::zero(), r_stage1)
        }
    };
    let stage2_input_claim = batching_coeff * s_claim + relation_claim;
    let row_eval_source = Stage2RowEvalSource::new(rs.prepared_row_eval);
    #[cfg(feature = "zk")]
    let stage2_next_w_eval_mask_cursor =
        *zk_hiding_cursor + (rs.col_bits + rs.ring_bits) * 3 * <L as ExtField<F>>::EXT_DEGREE;
    let stage2_verifier = match &proof {
        FoldProofView::Terminal(terminal_proof) => AkitaStage2Verifier::new_with_direct_witness(
            batching_coeff,
            s_claim,
            #[cfg(feature = "zk")]
            s_claim_mask,
            #[cfg(feature = "zk")]
            relation_claim_mask,
            &terminal_proof.final_witness,
            w_len,
            r_stage1,
            rs.alpha_evals_y,
            row_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &ring_multiplier_points,
            &rs.tau1,
            v_typed,
            commitment_u,
            &y_rings,
            Some(relation_claim),
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )?,
        FoldProofView::Intermediate(level_proof) => AkitaStage2Verifier::new_with_claimed_w_eval(
            batching_coeff,
            s_claim,
            #[cfg(feature = "zk")]
            s_claim_mask,
            #[cfg(feature = "zk")]
            relation_claim_mask,
            level_proof.stage2.next_w_eval(),
            #[cfg(feature = "zk")]
            zk_ext_mask_lc_at::<F, L>(stage2_next_w_eval_mask_cursor),
            r_stage1,
            rs.alpha_evals_y,
            row_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &ring_multiplier_points,
            &rs.tau1,
            v_typed,
            commitment_u,
            &y_rings,
            Some(relation_claim),
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )?,
    };
    if stage2_input_claim != SumcheckInstanceVerifier::input_claim(&stage2_verifier) {
        return Err(AkitaError::InvalidProof);
    }

    #[cfg(not(feature = "zk"))]
    let stage2_sumcheck_ref = match &proof {
        FoldProofView::Intermediate(level_proof) => &level_proof.stage2.sumcheck_proof,
        FoldProofView::Terminal(terminal_proof) => &terminal_proof.stage2_sumcheck,
    };
    #[cfg(feature = "zk")]
    let stage2_sumcheck_masked_ref = match &proof {
        FoldProofView::Intermediate(level_proof) => &level_proof.stage2.sumcheck_proof_masked,
        FoldProofView::Terminal(terminal_proof) => &terminal_proof.stage2_sumcheck_proof_masked,
    };
    let challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        #[cfg(not(feature = "zk"))]
        {
            stage2_verifier.verify::<F, T, _>(stage2_sumcheck_ref, transcript, |tr| {
                sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?
        }
        #[cfg(feature = "zk")]
        {
            let challenges = stage2_verifier.verify_zk::<F, T, _>(
                stage2_sumcheck_masked_ref,
                transcript,
                zk_relations,
                zk_hiding_cursor,
                |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
            )?;
            if matches!(proof, FoldProofView::Intermediate(_)) {
                *zk_hiding_cursor += <L as ExtField<F>>::EXT_DEGREE;
            }
            challenges
        }
    };
    if let FoldProofView::Intermediate(level_proof) = &proof {
        let next_w_eval = level_proof.stage2.next_w_eval();
        transcript.record_wire_serde(ABSORB_STAGE2_NEXT_W_EVAL, &next_w_eval);
        transcript.append_serde(ABSORB_STAGE2_NEXT_W_EVAL, &next_w_eval);
    }
    Ok(challenges)
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
    if step.current_w_len != current_state.common_padded_len()?
        || step.params.log_basis != current_state.log_basis
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

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_verify_intermediate_level<F, L, T>(
    level_d: usize,
    level_proof: &AkitaLevelProof<F, L>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F, L>,
    lp: &LevelParams,
    block_order: BlockOrder,
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
    macro_rules! dispatch {
        ($d:literal) => {
            verify_intermediate_level::<F, L, T, $d>(
                level_proof,
                setup,
                transcript,
                current_state,
                lp,
                block_order,
                #[cfg(feature = "zk")]
                zk_hiding_cursor,
                #[cfg(feature = "zk")]
                zk_relations,
            )
        };
    }
    match level_d {
        32 => dispatch!(32),
        64 => dispatch!(64),
        128 => dispatch!(128),
        256 => dispatch!(256),
        512 => dispatch!(512),
        1024 => dispatch!(1024),
        _ => Err(AkitaError::InvalidProof),
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_verify_terminal_level<F, L, T>(
    level_d: usize,
    terminal_proof: &TerminalLevelProof<F, L>,
    final_w_len: usize,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F, L>,
    lp: &LevelParams,
    block_order: BlockOrder,
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
    macro_rules! dispatch {
        ($d:literal) => {
            verify_terminal_level::<F, L, T, $d>(
                terminal_proof,
                final_w_len,
                setup,
                transcript,
                current_state,
                lp,
                block_order,
                #[cfg(feature = "zk")]
                zk_hiding_cursor,
                #[cfg(feature = "zk")]
                zk_relations,
            )
        };
    }
    match level_d {
        32 => dispatch!(32),
        64 => dispatch!(64),
        128 => dispatch!(128),
        256 => dispatch!(256),
        512 => dispatch!(512),
        1024 => dispatch!(1024),
        _ => Err(AkitaError::InvalidProof),
    }
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
                if is_last {
                    // The terminal slot must be a Terminal variant.
                    return Err(AkitaError::InvalidProof);
                }
                if !current_state.common_commitment()?.can_decode_vec(level_d)
                    || !level_proof.y_ring.can_decode_vec(level_d)
                    || !level_proof.v.can_decode_vec(level_d)
                {
                    return Err(AkitaError::InvalidProof);
                }

                let challenges = if level_d == D {
                    verify_intermediate_level::<F, L, T, D>(
                        level_proof,
                        setup,
                        transcript,
                        &current_state,
                        &current_lp,
                        BlockOrder::ColumnMajor,
                        #[cfg(feature = "zk")]
                        zk_hiding_cursor,
                        #[cfg(feature = "zk")]
                        zk_relations,
                    )?
                } else {
                    dispatch_verify_intermediate_level::<F, L, T>(
                        level_d,
                        level_proof,
                        setup,
                        transcript,
                        &current_state,
                        &current_lp,
                        BlockOrder::ColumnMajor,
                        #[cfg(feature = "zk")]
                        zk_hiding_cursor,
                        #[cfg(feature = "zk")]
                        zk_relations,
                    )?
                };

                let scheduled_next_params =
                    scheduled_next_params.ok_or(AkitaError::InvalidProof)?;
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
                    carried_openings: vec![RecursiveVerifierCarriedOpening {
                        opening_point: challenges,
                        opening: level_proof.next_w_eval(),
                        #[cfg(feature = "zk")]
                        opening_mask: zk_ext_mask_lc_at::<F, L>(
                            *zk_hiding_cursor - <L as ExtField<F>>::EXT_DEGREE,
                        ),
                        commitment: level_proof.next_w_commitment(),
                        basis: BasisMode::Lagrange,
                        natural_len: next_w_len,
                        padded_len: next_w_len,
                        kind: CarriedOpeningKind::RecursiveWitness,
                    }],
                    log_basis: scheduled_next_params.log_basis,
                };
            }
            AkitaProofStep::Terminal(terminal_proof) => {
                if !is_last {
                    return Err(AkitaError::InvalidProof);
                }
                if !current_state.common_commitment()?.can_decode_vec(level_d)
                    || !terminal_proof.y_rings.can_decode_vec(level_d)
                {
                    return Err(AkitaError::InvalidProof);
                }
                if level_d == D {
                    verify_terminal_level::<F, L, T, D>(
                        terminal_proof,
                        next_w_len,
                        setup,
                        transcript,
                        &current_state,
                        &current_lp,
                        BlockOrder::ColumnMajor,
                        #[cfg(feature = "zk")]
                        zk_hiding_cursor,
                        #[cfg(feature = "zk")]
                        zk_relations,
                    )?
                } else {
                    dispatch_verify_terminal_level::<F, L, T>(
                        level_d,
                        terminal_proof,
                        next_w_len,
                        setup,
                        transcript,
                        &current_state,
                        &current_lp,
                        BlockOrder::ColumnMajor,
                        #[cfg(feature = "zk")]
                        zk_hiding_cursor,
                        #[cfg(feature = "zk")]
                        zk_relations,
                    )?
                };
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
        akita_types::AkitaBatchedRootProof::Direct { .. } => Err(AkitaError::InvalidProof),
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

            let first_level_d = next_level_params.ring_dimension;
            if !fold_root
                .stage2
                .next_w_commitment
                .can_decode_vec(first_level_d)
            {
                return Err(AkitaError::InvalidProof);
            }

            let current_state = RecursiveVerifierState {
                carried_openings: vec![RecursiveVerifierCarriedOpening {
                    opening_point: root_challenges,
                    opening: fold_root.stage2.next_w_eval(),
                    #[cfg(feature = "zk")]
                    opening_mask: zk_ext_mask_lc_at::<F, C>(
                        zk_hiding_cursor - <C as ExtField<F>>::EXT_DEGREE,
                    ),
                    commitment: &fold_root.stage2.next_w_commitment,
                    basis: BasisMode::Lagrange,
                    natural_len: root_step.next_w_len,
                    padded_len: root_step.next_w_len,
                    kind: CarriedOpeningKind::RecursiveWitness,
                }],
                log_basis: next_level_params.log_basis,
            };
            verify_batched_recursive_suffix::<F, C, T, D>(
                proof,
                setup,
                transcript,
                schedule,
                current_state,
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
