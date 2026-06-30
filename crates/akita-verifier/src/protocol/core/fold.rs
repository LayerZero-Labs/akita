//! Shared per-fold verifier replay (EOR, stage-1/2/3, ring switch).

use super::*;

pub(in crate::protocol::core) struct FoldEorReplay<F: FieldCore, C: FieldCore, const D: usize> {
    pub(in crate::protocol::core) prepared_points: Vec<PreparedOpeningPoint<F, C, D>>,
    pub(in crate::protocol::core) reduction_challenges: Option<Vec<C>>,
    pub(in crate::protocol::core) final_relation: Option<(C, Vec<C>)>,
}

#[derive(Clone, Copy)]
struct EorReductionShape {
    split_bits: usize,
    width: usize,
    num_rounds: usize,
}

fn eor_reduction_shape<F, C>(
    opening_num_vars: usize,
    partials_len: usize,
    num_claims: usize,
) -> Result<EorReductionShape, AkitaError>
where
    F: FieldCore,
    C: ExtField<F>,
{
    let (split_bits, width) =
        tensor_opening_split::<F, C>().map_err(|_| AkitaError::InvalidProof)?;
    let num_rounds = opening_num_vars
        .checked_sub(split_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let expected_partials = width
        .checked_mul(num_claims)
        .ok_or(AkitaError::InvalidProof)?;
    if width == 1 || partials_len != expected_partials {
        return Err(AkitaError::InvalidProof);
    }
    Ok(EorReductionShape {
        split_bits,
        width,
        num_rounds,
    })
}

fn eor_input_claim_from_partials<F, C>(
    partials: &[C],
    shape: EorReductionShape,
    eta: &[C],
    row_coefficients: &[C],
) -> Result<C, AkitaError>
where
    F: FieldCore,
    C: ExtField<F>,
{
    if shape.width == 0
        || !partials.len().is_multiple_of(shape.width)
        || row_coefficients.len() != partials.len() / shape.width
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut input_claim = C::zero();
    for (&row_coefficient, partials) in row_coefficients
        .iter()
        .zip(partials.chunks_exact(shape.width))
    {
        let row_partials = tensor_row_partials_from_columns::<F, C>(partials)?;
        let claim = tensor_reduction_claim_from_rows::<F, C>(&row_partials, eta)?;
        input_claim += row_coefficient * claim;
    }
    Ok(input_claim)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn verify_fold_eor<F, C, T, const D: usize>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<C>>,
    _y_rings: &[CyclotomicRing<F, D>],
    challenge_point: &[C],
    openings: &[C],
    row_coefficients: &[C],
    opening_batch: &OpeningBatchShape,
    basis: BasisMode,
    lp: &LevelParams,
    block_order: BlockOrder,
    requires_reduction: bool,
    transcript: &mut T,
) -> Result<FoldEorReplay<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
    C: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let num_claims = opening_batch.num_polynomials();
    if openings.len() != num_claims || row_coefficients.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    let alpha_bits = lp.ring_dimension.trailing_zeros() as usize;
    let mut eor_trace_final: Option<(C, Vec<C>)> = None;
    let reduction_check = if let Some(reduction) = extension_opening_reduction {
        if <C as ExtField<F>>::EXT_DEGREE == 1 {
            return Err(AkitaError::InvalidProof);
        }
        let shape = eor_reduction_shape::<F, C>(
            opening_batch.num_vars(),
            reduction.partials.len(),
            num_claims,
        )?;
        if challenge_point.len() > opening_batch.num_vars() {
            return Err(AkitaError::InvalidProof);
        }
        let mut eor_point = challenge_point.to_vec();
        eor_point.resize(opening_batch.num_vars(), C::zero());
        for (claim_idx, opening) in openings.iter().copied().enumerate().take(num_claims) {
            let partial_start = claim_idx * shape.width;
            let partial_end = partial_start + shape.width;
            let partials = &reduction.partials[partial_start..partial_end];
            let expected =
                derive_tensor_extension_opening_claim_from_partials::<F, C>(&eor_point, partials)?;
            if expected != opening {
                return Err(AkitaError::InvalidProof);
            }
            for partial in partials {
                append_ext_field::<F, C, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
            }
        }
        let eta = (0..shape.split_bits)
            .map(|_| sample_ext_challenge::<F, C, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
            .collect::<Vec<_>>();
        let input_claim = eor_input_claim_from_partials::<F, C>(
            &reduction.partials,
            shape,
            &eta,
            row_coefficients,
        )?;
        let (final_claim, rho) = verify_extension_opening_reduction_sumcheck::<F, T, C, _>(
            input_claim,
            shape.num_rounds,
            &reduction.sumcheck,
            transcript,
            |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
        )?;
        let final_factor = tensor_equality_factor_eval_at_point::<F, C>(
            &eor_point[shape.split_bits..],
            &eta,
            &rho,
        )?;
        eor_trace_final = Some((final_claim, vec![final_factor]));
        Some(rho)
    } else if requires_reduction && <C as ExtField<F>>::EXT_DEGREE != 1 {
        return Err(AkitaError::InvalidProof);
    } else {
        None
    };

    let prepared_points = if let Some(rho) = &reduction_check {
        let protocol_point =
            ring_subfield_packed_extension_opening_point::<F, C, D>(rho.len(), rho)?;
        let prepared =
            prepare_opening_point::<F, C, D>(&protocol_point, basis, lp, alpha_bits, block_order)?;
        vec![prepared]
    } else {
        vec![prepare_opening_point::<F, C, D>(
            challenge_point,
            basis,
            lp,
            alpha_bits,
            block_order,
        )?]
    };
    Ok(FoldEorReplay {
        prepared_points,
        reduction_challenges: reduction_check,
        final_relation: eor_trace_final,
    })
}

pub(in crate::protocol::core) struct PreparedFoldReplay<
    'a,
    F: FieldCore,
    E: FieldCore,
    const D: usize,
> {
    pub(in crate::protocol::core) lp: &'a LevelParams,
    pub(in crate::protocol::core) m_row_layout: MRowLayout,
    pub(in crate::protocol::core) fold_grind_nonce: u32,
    pub(in crate::protocol::core) v: Vec<CyclotomicRing<F, D>>,
    pub(in crate::protocol::core) opening_batch:
        VerifierOpeningBatch<'a, E, &'a [CyclotomicRing<F, D>]>,
    pub(in crate::protocol::core) row_coefficients: Vec<E>,
    pub(in crate::protocol::core) ring_opening_point: RingOpeningPoint<F>,
    pub(in crate::protocol::core) ring_multiplier_point: RingMultiplierOpeningPoint<F, D>,
    pub(in crate::protocol::core) w_len: usize,
    pub(in crate::protocol::core) stage1: Option<&'a AkitaStage1Proof<E>>,
    pub(in crate::protocol::core) stage2: &'a AkitaStage2Proof<F, E>,
    pub(in crate::protocol::core) next_w_commitment: Option<&'a FlatRingVec<F>>,
    pub(in crate::protocol::core) terminal_replay: Option<TerminalWitnessTranscriptParts>,
    pub(in crate::protocol::core) stage3: Option<(&'a SetupSumcheckProof<E>, &'a LevelParams)>,
    pub(in crate::protocol::core) trace_prepared_point: Option<PreparedOpeningPoint<F, E, D>>,
    pub(in crate::protocol::core) trace_block_opening: Option<Vec<E>>,
    pub(in crate::protocol::core) trace_eval_target: E,
    pub(in crate::protocol::core) trace_eval_scale: E,
    pub(in crate::protocol::core) trace_claim_scales: Option<Vec<E>>,
    pub(in crate::protocol::core) trace_basis: BasisMode,
}

struct Stage1Replay<E: FieldCore> {
    batching_coeff: E,
    s_claim: E,
    stage1_point: Vec<E>,
}

fn verify_stage1<F, E, T>(
    proof: Option<&AkitaStage1Proof<E>>,
    rs: &RingSwitchVerifyOutput<E>,
    transcript: &mut T,
) -> Result<Stage1Replay<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let num_rounds = rs
        .col_bits
        .checked_add(rs.ring_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("stage-1 variable count overflow".to_string()))?;
    let tau0 = if rs.tau0.is_empty() {
        None
    } else {
        Some(rs.tau0.as_slice())
    };
    let stage1 = match (proof, tau0) {
        (Some(proof), Some(tau0)) => Some((proof, tau0)),
        (None, None) => None,
        _ => return Err(AkitaError::InvalidProof),
    };
    if let Some((proof, tau0)) = stage1 {
        if tau0.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: tau0.len(),
            });
        }
        let tau0_reordered = reorder_stage1_coords(tau0, rs.col_bits, rs.ring_bits);
        let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
        let stage1_point = {
            let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
            stage1_verifier.verify::<F, T>(proof, transcript)?
        };
        transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &proof.s_claim);
        let batching_coeff: E =
            sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
        return Ok(Stage1Replay {
            batching_coeff,
            s_claim: proof.s_claim,
            stage1_point,
        });
    }

    let relation_only = RelationOnlyStage2Inputs::new(num_rounds);
    Ok(Stage1Replay {
        batching_coeff: relation_only.batching_coeff,
        s_claim: relation_only.s_claim,
        stage1_point: relation_only.stage1_point,
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_stage2<F, E, T, const D: usize>(
    transcript: &mut T,
    setup: &AkitaVerifierSetup<F>,
    stage2: &AkitaStage2Proof<F, E>,
    physical_w_len: usize,
    stage1: Stage1Replay<E>,
    rs: &RingSwitchVerifyOutput<E>,
    relation_claim: E,
    lp: &LevelParams,
    num_segments: usize,
    setup_claim: Option<E>,
    ring_opening_point: &RingOpeningPoint<F>,
    ring_multiplier_point: &RingMultiplierOpeningPoint<F, D>,
    trace: Option<TraceClaim<F, E, D>>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let witness_oracle = match stage2 {
        AkitaStage2Proof::Terminal(proof) => stage2_cleartext_oracle::<F, E, D>(
            &proof.final_witness,
            physical_w_len,
            lp,
            num_segments,
        )?,
        AkitaStage2Proof::Intermediate(proof) => Stage2WitnessOracle::ClaimedEval {
            eval: proof.next_w_eval(),
        },
    };
    let stage2_verifier = AkitaStage2Verifier::new(
        stage1.batching_coeff,
        stage1.s_claim,
        witness_oracle,
        stage1.stage1_point,
        rs.alpha_evals_y.clone(),
        rs.prepared_row_eval.clone(),
        setup_claim,
        &setup.expanded,
        ring_opening_point,
        ring_multiplier_point,
        relation_claim,
        rs.alpha,
        rs.col_bits,
        rs.ring_bits,
        trace,
    )?;

    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        stage2_verifier.verify::<F, T, _>(stage2.sumcheck(), transcript, |tr| {
            sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?
    };
    if let AkitaStage2Proof::Intermediate(proof) = stage2 {
        transcript.absorb_and_record_serde(ABSORB_STAGE2_NEXT_W_EVAL, &proof.next_w_eval());
    }
    Ok(sumcheck_challenges)
}

fn verify_stage3<F, E, T, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    rs: &RingSwitchVerifyOutput<E>,
    sumcheck_challenges: &[E],
    stage2_next_w_eval: E,
    stage3: Option<(&SetupSumcheckProof<E>, &LevelParams)>,
) -> Result<Option<Vec<E>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    if let Some((proof, next_fold_level_params)) = stage3 {
        let witness_rounds = rs.col_bits.checked_add(rs.ring_bits).ok_or_else(|| {
            AkitaError::InvalidSetup("stage-3 witness round count overflow".to_string())
        })?;
        if sumcheck_challenges.len() != witness_rounds {
            return Err(AkitaError::InvalidSize {
                expected: witness_rounds,
                actual: sumcheck_challenges.len(),
            });
        }
        let setup_x_challenges = sumcheck_challenges
            .get(rs.ring_bits..)
            .ok_or(AkitaError::InvalidProof)?;
        let eta = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
        let verifier = SetupSumcheckVerifier::new::<F, D>(
            &rs.prepared_row_eval,
            setup_x_challenges,
            rs.alpha,
        )?;
        let rho_w = verifier.verify_batched_stage3::<F, T, D>(
            setup,
            next_fold_level_params,
            proof,
            stage2_next_w_eval,
            sumcheck_challenges,
            witness_rounds,
            eta,
            transcript,
        )?;
        transcript.absorb_and_record_serde(ABSORB_STAGE3_NEXT_W_EVAL, &proof.next_w_eval);
        return Ok(Some(rho_w));
    }
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(in crate::protocol::core) fn verify_fold<F, E, T, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    prepared: PreparedFoldReplay<'_, F, E, D>,
    a_ones_table: &FoldAOnesTable<F>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let opening_shape = prepared.opening_batch.to_shape();
    let commitment_rows = prepared
        .opening_batch
        .single_group_commitment()
        .ok_or(AkitaError::InvalidProof)?;
    validate_fold_grind_nonce(
        &prepared.lp.fold_witness_grind_contract(
            opening_shape.num_polynomials(),
            FoldLinfProtocolBinding::CURRENT.max_grind_attempts,
        )?,
        prepared.fold_grind_nonce,
    )?;
    let stage1_challenges = derive_stage1_challenges::<F, T, D>(
        transcript,
        &prepared.v,
        prepared.lp.num_blocks,
        opening_shape.num_polynomials(),
        prepared.lp,
        prepared.m_row_layout,
        prepared.fold_grind_nonce,
    )?;
    let (gamma, row_coefficient_rings) =
        RingRelationInstance::<F, D>::gamma_and_row_rings_from_coefficients::<E>(
            &prepared.row_coefficients,
        )?;
    let n_d_active = match prepared.m_row_layout {
        MRowLayout::WithDBlock => prepared.lp.d_key.row_len(),
        MRowLayout::WithoutDBlock => 0,
    };
    let y_v_slice = match prepared.m_row_layout {
        MRowLayout::WithDBlock => prepared.v.as_slice(),
        MRowLayout::WithoutDBlock => &[],
    };
    let num_digits_fold = prepared
        .lp
        .num_digits_fold(opening_shape.num_polynomials(), F::modulus_bits())?;
    let committed_shift = fold_response_shift(prepared.lp.log_basis, num_digits_fold);
    let consistency_shift_row = fold_shift_consistency_row::<F, D>(
        &prepared.ring_multiplier_point,
        prepared.lp.block_len,
        prepared.lp.num_digits_commit,
        prepared.lp.log_basis,
        committed_shift,
    )?;
    let a_shift_rows = a_ones_table.a_shift_rows::<D>(prepared.lp, committed_shift)?;
    let relation_y = generate_y::<F, D>(
        consistency_shift_row,
        y_v_slice,
        commitment_rows,
        a_shift_rows.as_ref(),
        n_d_active,
        prepared.lp.effective_commit_rows(),
        prepared.lp.b_inner_rows_per_group(),
        prepared.lp.a_key.row_len(),
    )?;
    let relation_instance = RingRelationInstance::new(
        prepared.m_row_layout,
        stage1_challenges,
        prepared.ring_opening_point,
        prepared.ring_multiplier_point,
        opening_shape,
        gamma,
        row_coefficient_rings,
        relation_y,
        prepared.v,
    )?;
    let ring_switch_replay = RingSwitchReplay {
        relation: &relation_instance,
        row_coefficients: &prepared.row_coefficients,
        lp: prepared.lp,
    };
    let rs = match prepared.stage2 {
        AkitaStage2Proof::Intermediate(_) => {
            let next_w_commitment = prepared.next_w_commitment.ok_or(AkitaError::InvalidProof)?;
            ring_switch_verifier::<F, E, T, D>(
                &ring_switch_replay,
                prepared.w_len,
                next_w_commitment,
                transcript,
            )?
        }
        AkitaStage2Proof::Terminal(_) => {
            let replay = prepared
                .terminal_replay
                .as_ref()
                .ok_or(AkitaError::InvalidProof)?;
            ring_switch_verifier_terminal::<F, E, T, D>(
                &ring_switch_replay,
                prepared.w_len,
                transcript,
                replay,
            )?
        }
    };
    let num_commitments = commitment_rows.len() / prepared.lp.effective_commit_rows();
    let relation_claim = relation_claim_from_fold_active_rows_for_level_extension::<F, E, D>(
        prepared.lp,
        prepared.m_row_layout,
        num_commitments,
        &rs.tau1,
        rs.alpha,
        &consistency_shift_row,
        &relation_instance.v,
        commitment_rows,
        a_shift_rows.as_ref(),
    )?;
    let stage1_replay = verify_stage1::<F, E, T>(prepared.stage1, &rs, transcript)?;
    let is_terminal_stage2 = matches!(prepared.stage2, AkitaStage2Proof::Terminal(_));
    let trace_gamma = if is_terminal_stage2 {
        sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH)
    } else {
        stage1_replay.batching_coeff
    };
    let trace_coeff = stage2_trace_coeff(
        stage1_replay.batching_coeff,
        trace_gamma,
        is_terminal_stage2,
    );
    ensure_trace_stage2_supported(<E as ExtField<F>>::EXT_DEGREE)?;
    let trace_wire = if prepared.trace_prepared_point.is_none() {
        None
    } else if prepared.trace_block_opening.is_none() {
        let segment = relation_instance.segment_layout(prepared.lp)?;
        let layout = trace_weight_layout_from_segment(
            prepared.lp,
            &segment,
            rs.col_bits,
            rs.ring_bits,
            prepared.lp.num_blocks,
        )?;
        let prepared_point = prepared
            .trace_prepared_point
            .as_ref()
            .ok_or(AkitaError::InvalidProof)?;
        Some(TraceClaim {
            layout,
            trace_coeff,
            trace_opening_claim: trace_coeff * prepared.trace_eval_target,
            trace_terms: trace_terms_recursive(
                prepared_point,
                prepared.lp,
                prepared.trace_basis,
                prepared.trace_eval_scale,
            )?,
        })
    } else {
        let segment = relation_instance.segment_layout(prepared.lp)?;
        let num_trace_blocks = relation_instance
            .opening_batch()
            .num_polynomials()
            .checked_mul(prepared.lp.num_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("trace block count overflow".to_string()))?;
        let layout = trace_weight_layout_from_segment(
            prepared.lp,
            &segment,
            rs.col_bits,
            rs.ring_bits,
            num_trace_blocks,
        )?;
        Some(build_trace_claim_root::<F, E, D>(
            layout,
            prepared.lp,
            relation_instance.opening_batch(),
            prepared
                .trace_prepared_point
                .as_ref()
                .ok_or(AkitaError::InvalidProof)?,
            prepared
                .trace_block_opening
                .as_ref()
                .ok_or(AkitaError::InvalidProof)?,
            prepared.trace_basis,
            &prepared.row_coefficients,
            trace_coeff,
            prepared.trace_eval_target,
            prepared.trace_claim_scales.as_deref(),
        )?)
    };
    let setup_claim = prepared.stage3.as_ref().map(|(proof, _)| proof.claim);
    let sumcheck_challenges = verify_stage2::<F, E, T, D>(
        transcript,
        setup,
        prepared.stage2,
        prepared.w_len,
        stage1_replay,
        &rs,
        relation_claim,
        prepared.lp,
        1,
        setup_claim,
        relation_instance.opening_point(),
        relation_instance.ring_multiplier_point(),
        trace_wire,
    )?;
    let stage2_next_w_eval = if prepared.stage3.is_some() {
        prepared.stage2.next_w_eval()
    } else {
        E::zero()
    };
    let stage3_challenges = verify_stage3::<F, E, T, D>(
        setup,
        transcript,
        &rs,
        &sumcheck_challenges,
        stage2_next_w_eval,
        prepared.stage3,
    )?;
    Ok(stage3_challenges.unwrap_or(sumcheck_challenges))
}
