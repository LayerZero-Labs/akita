//! Shared per-fold verifier replay (EOR, stage-1/2/3, ring switch).

use super::*;
use akita_types::{dispatch_for_field, DigitRangeEqualityPoint, DigitRangePlan};

pub(in crate::protocol::core) struct FoldEorReplay<F: FieldCore, E: FieldCore> {
    pub(in crate::protocol::core) prepared_points: Vec<PreparedOpeningPoint<F, E>>,
    pub(in crate::protocol::core) final_relation: Option<(E, Vec<E>)>,
}

#[derive(Clone, Copy)]
struct EorReductionShape {
    split_bits: usize,
    width: usize,
    num_rounds: usize,
}

struct EorSumcheckReplay<E: FieldCore> {
    rho: Vec<E>,
    final_claim: E,
    final_factors: Vec<E>,
}

fn eor_reduction_shape<F, E>(
    opening_num_vars: usize,
    partials_len: usize,
    num_claims: usize,
) -> Result<EorReductionShape, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let (split_bits, width) =
        tensor_opening_split::<F, E>().map_err(|_| AkitaError::InvalidProof)?;
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

fn eor_input_claim_from_partials<F, E>(
    partials: &[E],
    shape: EorReductionShape,
    eta: &[E],
    row_coefficients: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    if shape.width == 0
        || !partials.len().is_multiple_of(shape.width)
        || row_coefficients.len() != partials.len() / shape.width
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut input_claim = E::zero();
    for (&row_coefficient, partials) in row_coefficients
        .iter()
        .zip(partials.chunks_exact(shape.width))
    {
        let row_partials = tensor_row_partials_from_columns::<F, E>(partials)?;
        let claim = tensor_reduction_claim_from_rows::<F, E>(&row_partials, eta)?;
        input_claim += row_coefficient * claim;
    }
    Ok(input_claim)
}

fn verify_eor_sumcheck<F, E, T>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    group_points: &[&[E]],
    openings: &[E],
    row_coefficients: &[E],
    opening_batch: &OpeningClaimsLayout,
    requires_reduction: bool,
    transcript: &mut T,
) -> Result<Option<EorSumcheckReplay<E>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    let num_claims = opening_batch.num_total_polynomials();
    if openings.len() != num_claims
        || row_coefficients.len() != num_claims
        || group_points.len() != opening_batch.num_groups()
    {
        return Err(AkitaError::InvalidProof);
    }
    let Some(reduction) = extension_opening_reduction else {
        if requires_reduction && <E as ExtField<F>>::EXT_DEGREE != 1 {
            return Err(AkitaError::InvalidProof);
        }
        return Ok(None);
    };
    if <E as ExtField<F>>::EXT_DEGREE == 1 {
        return Err(AkitaError::InvalidProof);
    }
    let shape = eor_reduction_shape::<F, E>(
        opening_batch.max_num_vars(),
        reduction.partials.len(),
        num_claims,
    )?;
    let mut claim_offset = 0usize;
    for (group_index, group_point) in group_points.iter().enumerate() {
        let group_layout = opening_batch.group_layout(group_index)?;
        if group_point.len() != group_layout.num_vars() || group_point.len() < shape.split_bits {
            return Err(AkitaError::InvalidProof);
        }
        for opening in openings
            .get(claim_offset..)
            .ok_or(AkitaError::InvalidProof)?
            .iter()
            .take(group_layout.num_polynomials())
        {
            let partial_start = claim_offset
                .checked_mul(shape.width)
                .ok_or(AkitaError::InvalidProof)?;
            let partial_end = partial_start
                .checked_add(shape.width)
                .ok_or(AkitaError::InvalidProof)?;
            let partials = reduction
                .partials
                .get(partial_start..partial_end)
                .ok_or(AkitaError::InvalidProof)?;
            let expected =
                derive_tensor_extension_opening_claim_from_partials::<F, E>(group_point, partials)?;
            if expected != *opening {
                return Err(AkitaError::InvalidProof);
            }
            for partial in partials {
                append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
            }
            claim_offset = claim_offset
                .checked_add(1)
                .ok_or(AkitaError::InvalidProof)?;
        }
    }
    if claim_offset != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let eta = (0..shape.split_bits)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
        .collect::<Vec<_>>();
    let input_claim =
        eor_input_claim_from_partials::<F, E>(&reduction.partials, shape, &eta, row_coefficients)?;
    let (final_claim, rho) = verify_extension_opening_reduction_sumcheck::<F, T, E, _>(
        input_claim,
        shape.num_rounds,
        &reduction.sumcheck,
        transcript,
        |tr| sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND),
    )?;
    let mut final_factors = Vec::with_capacity(group_points.len());
    for group_point in group_points {
        let tail_point = group_point
            .get(shape.split_bits..)
            .ok_or(AkitaError::InvalidProof)?;
        let local_rho = rho
            .get(..tail_point.len())
            .ok_or(AkitaError::InvalidProof)?;
        let mut factor = tensor_equality_factor_eval_at_point::<F, E>(tail_point, &eta, local_rho)?;
        for &extra_challenge in rho
            .get(tail_point.len()..)
            .ok_or(AkitaError::InvalidProof)?
        {
            factor *= E::one() - extra_challenge;
        }
        final_factors.push(factor);
    }
    Ok(Some(EorSumcheckReplay {
        rho,
        final_claim,
        final_factors,
    }))
}

/// Verify the terminal fold's single-group extension-opening reduction.
///
/// Terminal proofs carry their geometry directly rather than through
/// `CommittedGroupParams`, so their replay remains an explicit terminal
/// boundary instead of being disguised as an ordinary committed-group fold.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn verify_terminal_fold_eor<F, E, T>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    challenge_point: &[E],
    openings: &[E],
    row_coefficients: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    d_a: usize,
    num_positions_per_block: usize,
    num_live_blocks: usize,
    requires_reduction: bool,
    transcript: &mut T,
) -> Result<FoldEorReplay<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
        verify_terminal_fold_eor_kernel::<F, E, T, D>(
            extension_opening_reduction,
            challenge_point,
            openings,
            row_coefficients,
            opening_batch,
            basis,
            num_positions_per_block,
            num_live_blocks,
            d_a.trailing_zeros() as usize,
            requires_reduction,
            transcript,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_terminal_fold_eor_kernel<F, E, T, const D: usize>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    challenge_point: &[E],
    openings: &[E],
    row_coefficients: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    num_positions_per_block: usize,
    num_live_blocks: usize,
    alpha_bits: usize,
    requires_reduction: bool,
    transcript: &mut T,
) -> Result<FoldEorReplay<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    if challenge_point.len() > opening_batch.max_num_vars() || opening_batch.num_groups() != 1 {
        return Err(AkitaError::InvalidProof);
    }
    let mut eor_point = challenge_point.to_vec();
    eor_point.resize(opening_batch.max_num_vars(), E::zero());
    let replay = verify_eor_sumcheck::<F, E, T>(
        extension_opening_reduction,
        &[eor_point.as_slice()],
        openings,
        row_coefficients,
        opening_batch,
        requires_reduction,
        transcript,
    )?;
    let prepared_points = if let Some(replay) = &replay {
        let protocol_point =
            ring_subfield_packed_extension_opening_point::<F, E, D>(replay.rho.len(), &replay.rho)?;
        let prepared = prepare_opening_point::<F, E, D>(
            &protocol_point,
            basis,
            num_positions_per_block,
            num_live_blocks,
            alpha_bits,
        )?;
        vec![prepared]
    } else {
        Vec::new()
    };
    Ok(FoldEorReplay {
        prepared_points,
        final_relation: replay.map(|replay| (replay.final_claim, replay.final_factors)),
    })
}

#[allow(clippy::too_many_arguments)]
/// Verify one fold's extension-opening reduction over all opening groups.
///
/// Every group retains its native opening point and committed geometry. The
/// groups share one batched sumcheck challenge sequence.
pub(in crate::protocol::core) fn verify_fold_eor<F, E, T>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    group_points: &[&[E]],
    openings: &[E],
    row_coefficients: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    lp: &CommittedGroupParams,
    requires_reduction: bool,
    transcript: &mut T,
) -> Result<FoldEorReplay<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let replay = verify_eor_sumcheck::<F, E, T>(
        extension_opening_reduction,
        group_points,
        openings,
        row_coefficients,
        opening_batch,
        requires_reduction,
        transcript,
    )?;
    let mut prepared_points = Vec::new();
    if let Some(replay) = &replay {
        prepared_points.reserve(group_points.len());
        for (group_index, group_point) in group_points.iter().enumerate() {
            let group_lp = lp.group_params(opening_batch, group_index)?;
            let group_dims = lp.group_role_dims(opening_batch, group_index)?;
            let alpha_bits = group_dims.d_a().trailing_zeros() as usize;
            let tail_vars = group_point
                .len()
                .checked_sub(tensor_opening_split::<F, E>()?.0)
                .ok_or(AkitaError::InvalidProof)?;
            let local_rho = replay
                .rho
                .get(..tail_vars)
                .ok_or(AkitaError::InvalidProof)?;
            let prepared = dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Inner),
                F,
                group_dims.d_a(),
                |D| {
                    let protocol_point = ring_subfield_packed_extension_opening_point::<F, E, D>(
                        local_rho.len(),
                        local_rho,
                    )?;
                    prepare_opening_point::<F, E, D>(
                        &protocol_point,
                        basis,
                        group_lp.num_positions_per_block(),
                        group_lp.num_live_blocks(),
                        alpha_bits,
                    )
                }
            )?;
            prepared_points.push(prepared);
        }
    }
    Ok(FoldEorReplay {
        prepared_points,
        final_relation: replay.map(|replay| (replay.final_claim, replay.final_factors)),
    })
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
    pub(in crate::protocol::core) row_coefficients: Vec<E>,
    /// Per-group ring opening points in `OpeningClaims` order.
    pub(in crate::protocol::core) group_ring_opening_points: Vec<RingOpeningPoint<F>>,
    /// Per-group ring multiplier points in `OpeningClaims` order.
    pub(in crate::protocol::core) group_ring_multiplier_points: Vec<RingMultiplierOpeningPoint<F>>,
    pub(in crate::protocol::core) w_len: usize,
    pub(in crate::protocol::core) payload: PreparedFoldPayload<'a, F, E>,
    /// Per-group prepared opening points in `OpeningClaims` order (one element
    /// for scalar/suffix folds). Reused for the fused trace term.
    pub(in crate::protocol::core) evaluation_trace_points: Vec<PreparedOpeningPoint<F, E>>,
    pub(in crate::protocol::core) evaluation_trace_claim: E,
    pub(in crate::protocol::core) evaluation_trace_claim_coefficients: Vec<E>,
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
    let carrier_ring_dimension = rs.relation_address_geometry.carrier_ring_dimension();
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        carrier_ring_dimension,
        |D| {
            verify_stage2_kernel::<F, E, T, D>(
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
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_stage2_kernel<F, E, T, const D: usize>(
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
    let stage2_verifier = AkitaStage2Verifier::<F, E, D>::new(
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
            .common_relation_witness_variable_count(),
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
            .common_relation_witness_variable_count();
        let setup_x_challenges = sumcheck_challenges
            .get(setup_coefficient_bits..)
            .ok_or(AkitaError::InvalidProof)?;
        let verifier = SetupSumcheckVerifier::new::<F>(
            &rs.relation_matrix_evaluator,
            setup_x_challenges,
            &rs.tau1,
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
        prepared
            .lp
            .fold_witness_grind_batch_contract(
                &opening_shape,
                FoldLinfProtocolBinding::CURRENT.max_grind_attempts,
            )?
            .validate_nonce(prepared.fold_grind_nonce)?;
        if !prepared.v.coeffs().is_empty() {
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Opening),
                F,
                role_dims.d_d(),
                |D| prepared.v.as_ring_slice::<D>().map(|_| ())
            )?;
        }
        if prepared.group_ring_opening_points.len() != num_groups
            || prepared.group_ring_multiplier_points.len() != num_groups
        {
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
                    &prepared.row_coefficients,
                )
            }
        )?;
        let relation_rhs_layout = relation_rhs_layout_for(prepared.lp, &opening_shape)?;
        let relation_rhs =
            assemble_relation_rhs::<F>(&relation_rhs_layout, &prepared.v, commitment_rows)?;
        let relation_instance = RingRelationInstance::new(
            group_challenges,
            prepared.group_ring_opening_points,
            prepared.group_ring_multiplier_points,
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
        row_coefficients: &prepared.row_coefficients,
        lp: prepared.lp,
        opening_source_len: next_opening_source_len,
        opening_ring_dim: next_witness_ring_dim,
    };
    let carrier_ring_dimension = prepared.lp.relation_witness_carrier_ring_dimension();
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
    let rs = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        carrier_ring_dimension,
        |D| ring_switch_verifier::<F, E, T, D>(&ring_switch_replay, prepared.w_len, transcript)
    )?;
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
    let evaluation_trace_points = &prepared.evaluation_trace_points;
    let trace_preparation_span = tracing::info_span!(
        "stage2_evaluation_trace_preparation",
        claims = opening_batch.num_total_polynomials(),
        groups = opening_batch.num_groups(),
        chunks = trace_witness_layout.units().len(),
        source_ring_dimension = carrier_ring_dimension,
    )
    .entered();
    let evaluation_trace = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        carrier_ring_dimension,
        |D| {
            prepare_evaluation_trace::<F, E, D>(&EvaluationTraceInputs {
                digit_witness_domain: trace_domain,
                witness_layout: trace_witness_layout,
                carrier_ring_dimension,
                level_params: prepared.lp,
                opening_batch,
                prepared_points: evaluation_trace_points,
                claim_coefficients: &prepared.evaluation_trace_claim_coefficients,
                basis: prepared.evaluation_trace_basis,
            })
        }
    )?;
    drop(trace_preparation_span);
    let evaluation_trace_opening_claim = evaluation_trace_weight * prepared.evaluation_trace_claim;
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

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::CompressedUniPoly;
    use akita_field::{FpExt4, Prime32Offset99};
    use akita_sumcheck::SumcheckProof;
    use akita_transcript::AkitaTranscript;
    use akita_types::{PolynomialGroupLayout, EXTENSION_OPENING_REDUCTION_DEGREE};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    fn extension_point(num_vars: usize, offset: u64) -> Vec<E> {
        (0..num_vars)
            .map(|index| {
                let value = offset + index as u64;
                E::from_base_slice(&[
                    F::from_u64(value + 1),
                    F::from_u64(value + 2),
                    F::from_u64(value + 3),
                    F::from_u64(value + 4),
                ])
            })
            .collect()
    }

    #[test]
    fn recursive_setup_prefix_groups_share_one_max_tail_eor_and_reject_tampering() {
        const SETUP_PREFIX_VARS: usize = 12;
        const WITNESS_VARS: usize = 20;
        let opening_batch = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::singleton(SETUP_PREFIX_VARS),
            PolynomialGroupLayout::singleton(WITNESS_VARS),
        ])
        .expect("recursive setup-prefix opening layout");
        let group_points = [
            extension_point(SETUP_PREFIX_VARS, 10),
            extension_point(WITNESS_VARS, 100),
        ];
        let group_point_refs = group_points.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let openings = vec![E::zero(); opening_batch.num_total_polynomials()];
        let row_coefficients = vec![E::one(); opening_batch.num_total_polynomials()];
        let (split_bits, width) = tensor_opening_split::<F, E>().expect("tensor split");
        let max_tail_rounds = WITNESS_VARS - split_bits;
        let reduction = ExtensionOpeningReductionProof {
            partials: vec![E::zero(); width * opening_batch.num_total_polynomials()],
            sumcheck: SumcheckProof {
                round_polys: (0..max_tail_rounds)
                    .map(|_| CompressedUniPoly {
                        coeffs_except_linear_term: vec![
                            E::zero();
                            EXTENSION_OPENING_REDUCTION_DEGREE
                        ],
                    })
                    .collect(),
            },
        };

        let mut transcript = AkitaTranscript::<F>::new(b"test/recursive-setup-prefix-grouped-eor");
        let replay = verify_eor_sumcheck::<F, E, _>(
            Some(&reduction),
            &group_point_refs,
            &openings,
            &row_coefficients,
            &opening_batch,
            true,
            &mut transcript,
        )
        .expect("recursive setup-prefix grouped EOR")
        .expect("extension claims require EOR");
        assert_eq!(replay.rho.len(), max_tail_rounds);
        assert_eq!(replay.final_factors.len(), 2);

        let mut tampered = reduction.clone();
        tampered.partials[0] += E::one();
        let mut tampered_transcript =
            AkitaTranscript::<F>::new(b"test/recursive-setup-prefix-grouped-eor");
        let tampered_result = verify_eor_sumcheck::<F, E, _>(
            Some(&tampered),
            &group_point_refs,
            &openings,
            &row_coefficients,
            &opening_batch,
            true,
            &mut tampered_transcript,
        );
        assert!(
            matches!(tampered_result, Err(AkitaError::InvalidProof)),
            "tampered setup-prefix EOR partial must reject"
        );

        let mut missing_transcript =
            AkitaTranscript::<F>::new(b"test/recursive-setup-prefix-grouped-eor");
        let missing_result = verify_eor_sumcheck::<F, E, _>(
            None,
            &group_point_refs,
            &openings,
            &row_coefficients,
            &opening_batch,
            true,
            &mut missing_transcript,
        );
        assert!(
            matches!(missing_result, Err(AkitaError::InvalidProof)),
            "missing setup-prefix EOR must reject for extension claims"
        );
    }
}
