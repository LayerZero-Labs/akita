//! Shared per-fold verifier replay (EOR, stage-1/2/3, ring switch).

mod extension_claim;
mod single_field;

use super::*;
use akita_types::{dispatch_for_field, DigitRangeEqualityPoint, DigitRangePlan};

pub(in crate::protocol::core) use extension_claim::{
    verify_extension_claim_root_prefix, verify_extension_claim_suffix_prefix,
    verify_extension_claim_terminal_suffix,
};
pub(in crate::protocol::core) use single_field::{
    absorb_protocol_opening_points, prepare_single_field_suffix_groups,
    prepare_single_field_terminal_suffix, verify_single_field_root_prefix,
};

/// Common prepared fold prefix produced by the single-field and
/// extension-claim geometry modules, consumed by root and suffix finishing
/// logic.
pub(in crate::protocol::core) struct FoldPrefix<F: FieldCore, E: FieldCore> {
    pub(in crate::protocol::core) prepared_points: Vec<PreparedOpeningPoint<F, E>>,
    pub(in crate::protocol::core) row_coefficients: Vec<E>,
    pub(in crate::protocol::core) trace_eval_target: E,
    pub(in crate::protocol::core) trace_claim_coefficients: Vec<E>,
}

pub(in crate::protocol::core) struct PreparedFoldReplay<'a, F: FieldCore, E: FieldCore> {
    pub(in crate::protocol::core) lp: &'a CommittedGroupParams,
    pub(in crate::protocol::core) fold_grind_nonce: u32,
    pub(in crate::protocol::core) v: RingVec<F>,
    /// Normalized opening geometry (one group for scalar/suffix folds, `G`
    /// groups for multi-group roots).
    pub(in crate::protocol::core) opening_shape: OpeningClaimsLayout,
    /// Sent commitment rows concatenated in M-row (final-first
    /// `root_group_order`) order — the single group's rows for scalar/suffix
    /// folds, `concat_g u_g` for multi-group roots. Matches the prover's
    /// `RingRelationProver` commitment-row concatenation and
    /// `relation_rhs_layout_for` block order.
    pub(in crate::protocol::core) commitment_rows: RingVec<F>,
    pub(in crate::protocol::core) prefix: FoldPrefix<F, E>,
    pub(in crate::protocol::core) w_len: usize,
    pub(in crate::protocol::core) payload: PreparedFoldPayload<'a, F, E>,
    pub(in crate::protocol::core) evaluation_trace_basis: BasisMode,
}

#[derive(Clone, Copy)]
pub(in crate::protocol::core) enum PreparedNextWitness<'a, F: FieldCore> {
    Commitment {
        commitment: &'a RingVec<F>,
        ring_dim: usize,
    },
    TerminalT(&'a [u8]),
}

pub(in crate::protocol::core) enum PreparedFoldPayload<'a, F: FieldCore, E: FieldCore> {
    Recursive {
        stage1: &'a AkitaStage1Proof<E>,
        stage2: &'a AkitaStage2Proof<F, E>,
        next_witness: PreparedNextWitness<'a, F>,
        next_witness_ring_dim: usize,
        next_opening_source_len: usize,
        stage3: Option<(&'a SetupSumcheckProof<E>, &'a CommittedGroupParams)>,
    },
}

struct Stage1Replay<E: FieldCore> {
    batching_coeff: E,
    range_image_evaluation: E,
    stage1_point: Vec<E>,
}

fn verify_stage1<F, E, T>(
    proof: &AkitaStage1Proof<E>,
    rs: &RingSwitchVerifyOutput<E>,
    transcript: &mut T,
) -> Result<Stage1Replay<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let num_rounds = rs.relation_address_geometry.relation_point_variable_count();
    if rs.tau0.len() != num_rounds {
        return Err(AkitaError::InvalidSize {
            expected: num_rounds,
            actual: rs.tau0.len(),
        });
    }
    let digit_range_equality_col_bits = rs
        .tau0
        .len()
        .checked_sub(rs.digit_range_equality_low_variable_count)
        .ok_or(AkitaError::InvalidProof)?;
    let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
        &rs.tau0,
        digit_range_equality_col_bits,
        rs.digit_range_equality_low_variable_count,
    )?;
    let plan = DigitRangePlan::new(rs.b)?;
    let stage1_verifier = AkitaStage1Verifier::new(equality_point, plan);
    let stage1_point = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify::<F, T>(proof, transcript)?
    };
    transcript.append_serde(ABSORB_RANGE_IMAGE_EVALUATION, &proof.range_image_evaluation);
    let batching_coeff: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
    Ok(Stage1Replay {
        batching_coeff,
        range_image_evaluation: proof.range_image_evaluation,
        stage1_point,
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_stage2<F, E, T>(
    transcript: &mut T,
    setup: &AkitaVerifierSetup<F>,
    stage2: &AkitaStage2Proof<F, E>,
    stage1: Stage1Replay<E>,
    rs: &RingSwitchVerifyOutput<E>,
    relation_claim: E,
    setup_claim: Option<E>,
    evaluation_trace: PreparedEvaluationTrace<E>,
    evaluation_trace_row_weight: E,
    evaluation_trace_opening_claim: E,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let witness_eval = stage2.next_w_eval();
    verify_stage2_kernel::<F, E, T>(
        transcript,
        setup,
        stage2,
        stage1,
        rs,
        relation_claim,
        witness_eval,
        setup_claim,
        evaluation_trace,
        evaluation_trace_row_weight,
        evaluation_trace_opening_claim,
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_stage2_kernel<F, E, T>(
    transcript: &mut T,
    setup: &AkitaVerifierSetup<F>,
    stage2: &AkitaStage2Proof<F, E>,
    stage1: Stage1Replay<E>,
    rs: &RingSwitchVerifyOutput<E>,
    relation_claim: E,
    witness_eval: E,
    setup_claim: Option<E>,
    evaluation_trace: PreparedEvaluationTrace<E>,
    evaluation_trace_row_weight: E,
    evaluation_trace_opening_claim: E,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let stage2_verifier = AkitaStage2Verifier::<F, E>::new(
        stage1.batching_coeff,
        stage1.range_image_evaluation,
        witness_eval,
        stage1.stage1_point,
        &rs.relation_matrix_evaluator,
        &setup.expanded,
        rs.alpha,
        setup_claim,
        relation_claim,
        rs.relation_address_geometry.relation_lane_variable_count(),
        rs.relation_address_geometry
            .relation_coefficient_variable_count(),
        evaluation_trace,
        evaluation_trace_row_weight,
        evaluation_trace_opening_claim,
    )?;

    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        stage2_verifier.verify::<F, T, _>(&stage2.sumcheck_proof, transcript, |tr| {
            sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?
    };
    transcript.absorb_and_record_serde(ABSORB_STAGE2_NEXT_W_EVAL, &stage2.next_w_eval());
    Ok(sumcheck_challenges)
}

fn verify_stage3<F, E, T>(
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    rs: &RingSwitchVerifyOutput<E>,
    sumcheck_challenges: &[E],
    stage3: Option<(&SetupSumcheckProof<E>, &CommittedGroupParams)>,
) -> Result<Option<(Vec<E>, E)>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    if let Some((proof, next_fold_level_params)) = stage3 {
        let setup_coefficient_bits = rs
            .relation_address_geometry
            .relation_coefficient_variable_count();
        let setup_x_challenges = sumcheck_challenges
            .get(setup_coefficient_bits..)
            .ok_or(AkitaError::InvalidProof)?;
        let verifier = SetupSumcheckVerifier::new::<F>(
            &rs.relation_matrix_evaluator,
            setup_x_challenges,
            rs.alpha,
        )?;
        let setup_point =
            verifier.verify_stage3::<F, T>(setup, next_fold_level_params, proof, transcript)?;
        return Ok(next_fold_level_params
            .setup_prefix
            .as_ref()
            .map(|_| (setup_point, proof.setup_prefix_eval)));
    }
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(in crate::protocol::core) fn verify_fold<F, E, T>(
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    prepared: PreparedFoldReplay<'_, F, E>,
) -> Result<FoldVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let opening_shape = prepared.opening_shape.clone();
    let num_groups = opening_shape.num_groups();
    let commitment_rows = &prepared.commitment_rows;
    let prefix = &prepared.prefix;
    let role_dims = prepared.lp.role_dims();
    let _fold_span = tracing::info_span!(
        "verify_fold",
        d_a = role_dims.d_a(),
        d_b = role_dims.d_b(),
        d_d = role_dims.d_d(),
        groups = num_groups,
    )
    .entered();
    {
        let _span = tracing::info_span!("fold_validate_inputs").entered();
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Outer),
            F,
            role_dims.d_b(),
            |D| commitment_rows.as_ring_slice::<D>().map(|_| ())
        )?;
        prepared.lp.validate_fold_grind_nonce(
            &opening_shape,
            FoldLinfProtocolBinding::CURRENT.max_grind_attempts,
            prepared.fold_grind_nonce,
        )?;
        if !prepared.v.coeffs().is_empty() {
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Opening),
                F,
                role_dims.d_d(),
                |D| prepared.v.as_ring_slice::<D>().map(|_| ())
            )?;
        }
        if prefix.prepared_points.len() != num_groups {
            return Err(AkitaError::InvalidProof);
        }
    }
    let group_challenges = {
        let _span = tracing::info_span!("fold_derive_stage1_challenges").entered();
        derive_multi_group_stage1_challenges::<F, T>(
            transcript,
            prepared.v.coeffs(),
            role_dims.d_d(),
            &opening_shape,
            prepared.lp,
            prepared.fold_grind_nonce,
        )?
    };
    let (relation_rhs_layout, relation_instance) = {
        let _span = tracing::info_span!("fold_prepare_relation").entered();
        let (gamma, row_coefficient_rings) = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            role_dims.d_a(),
            |D| {
                RingRelationInstance::<F>::gamma_and_row_rings_from_coefficients::<D, E>(
                    &prefix.row_coefficients,
                )
            }
        )?;
        let relation_rhs_layout = relation_rhs_layout_for(prepared.lp, &opening_shape)?;
        let relation_rhs =
            assemble_relation_rhs::<F>(&relation_rhs_layout, &prepared.v, commitment_rows)?;
        let group_ring_opening_points = prefix
            .prepared_points
            .iter()
            .map(|prepared| prepared.ring_opening_point.clone())
            .collect::<Vec<_>>();
        let group_ring_multiplier_points = prefix
            .prepared_points
            .iter()
            .map(|prepared| prepared.ring_multiplier_point.clone())
            .collect::<Vec<_>>();
        let relation_instance = RingRelationInstance::new(
            group_challenges,
            group_ring_opening_points,
            group_ring_multiplier_points,
            opening_shape.clone(),
            gamma,
            row_coefficient_rings,
            relation_rhs,
            prepared.v,
            role_dims,
        )?;
        relation_instance.check_v_shape_for_level(prepared.lp)?;
        (relation_rhs_layout, relation_instance)
    };
    let (stage1, stage2, next_witness, next_witness_ring_dim, next_opening_source_len, stage3) =
        match prepared.payload {
            PreparedFoldPayload::Recursive {
                stage1,
                stage2,
                next_witness,
                next_witness_ring_dim,
                next_opening_source_len,
                stage3,
            } => (
                stage1,
                stage2,
                next_witness,
                next_witness_ring_dim,
                next_opening_source_len,
                stage3,
            ),
        };
    let ring_switch_replay = RingSwitchReplay {
        setup: &setup.expanded,
        relation: &relation_instance,
        row_coefficients: &prefix.row_coefficients,
        lp: prepared.lp,
        opening_source_len: next_opening_source_len,
        opening_ring_dim: next_witness_ring_dim,
    };
    {
        let _span = tracing::info_span!("fold_bind_next_witness").entered();
        match next_witness {
            PreparedNextWitness::Commitment {
                commitment,
                ring_dim,
            } => {
                if ring_dim == 0 || !commitment.can_decode_vec(ring_dim) {
                    return Err(AkitaError::InvalidProof);
                }
                transcript.absorb_and_record_serde(ABSORB_NEXT_LEVEL_WITNESS_BINDING, commitment);
            }
            PreparedNextWitness::TerminalT(t_state) if !t_state.is_empty() => {
                transcript.absorb_and_record_bytes(ABSORB_NEXT_LEVEL_WITNESS_BINDING, t_state);
            }
            PreparedNextWitness::TerminalT(_) => return Err(AkitaError::InvalidProof),
        }
    }
    let rs = ring_switch_verifier::<F, E, T>(&ring_switch_replay, prepared.w_len, transcript)?;
    let relation_claim = relation_claim_from_layout_extension::<F, E>(
        &relation_rhs_layout,
        &rs.tau1,
        rs.alpha,
        relation_instance.v(),
        commitment_rows,
    )?;
    let stage1_replay = verify_stage1::<F, E, T>(stage1, &rs, transcript)?;
    // EvaluationTrace is the last padded relation row: weight openings by
    // `eq(tau1, EvaluationTrace_row_index)`.
    let opening_batch = relation_instance.opening_batch();
    let evaluation_trace_row = prepared.lp.evaluation_trace_row_index(opening_batch)?;
    let evaluation_trace_weight = evaluation_trace_row_weight(evaluation_trace_row, &rs.tau1)?;
    ensure_trace_stage2_supported(<E as ExtField<F>>::EXT_DEGREE)?;
    let trace_domain = rs.relation_address_geometry.digit_witness_domain();
    if trace_domain.live_len() != prepared.w_len {
        return Err(AkitaError::InvalidSize {
            expected: trace_domain.live_len(),
            actual: prepared.w_len,
        });
    }
    let trace_witness_layout = rs.relation_matrix_evaluator.witness_layout()?;
    let trace_preparation_span = tracing::info_span!(
        "stage2_evaluation_trace_preparation",
        claims = opening_batch.num_total_polynomials(),
        groups = opening_batch.num_groups(),
        chunks = trace_witness_layout.units().len(),
        coefficient_block_len = rs
            .relation_address_geometry
            .relation_coefficient_block_len(),
    )
    .entered();
    let evaluation_trace = prepare_evaluation_trace::<F, E>(&EvaluationTraceInputs {
        digit_witness_domain: trace_domain,
        relation_coefficient_block_len: rs
            .relation_address_geometry
            .relation_coefficient_block_len(),
        witness_layout: trace_witness_layout,
        level_params: prepared.lp,
        opening_batch,
        prepared_points: &prefix.prepared_points,
        claim_coefficients: &prefix.trace_claim_coefficients,
        basis: prepared.evaluation_trace_basis,
    })?;
    drop(trace_preparation_span);
    let evaluation_trace_opening_claim = evaluation_trace_weight * prefix.trace_eval_target;
    let setup_claim = stage3.as_ref().map(|(proof, _)| proof.claim);
    let sumcheck_challenges = verify_stage2::<F, E, T>(
        transcript,
        setup,
        stage2,
        stage1_replay,
        &rs,
        relation_claim,
        setup_claim,
        evaluation_trace,
        evaluation_trace_weight,
        evaluation_trace_opening_claim,
    )?;
    let setup_prefix_opening =
        verify_stage3::<F, E, T>(setup, transcript, &rs, &sumcheck_challenges, stage3)?;
    Ok((sumcheck_challenges, setup_prefix_opening))
}
