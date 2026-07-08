//! Shared per-fold verifier replay (EOR, stage-1/2/3, ring switch).

use super::*;
use akita_types::dispatch_for_field;

pub(in crate::protocol::core) struct FoldEorReplay<F: FieldCore, E: FieldCore> {
    pub(in crate::protocol::core) prepared_points: Vec<PreparedOpeningPoint<F, E>>,
    pub(in crate::protocol::core) reduction_challenges: Option<Vec<E>>,
    pub(in crate::protocol::core) final_relation: Option<(E, Vec<E>)>,
}

#[derive(Clone, Copy)]
struct EorReductionShape {
    split_bits: usize,
    width: usize,
    num_rounds: usize,
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

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn verify_fold_eor<F, E, T>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    challenge_point: &[E],
    openings: &[E],
    row_coefficients: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    lp: &LevelParams,
    block_order: BlockOrder,
    requires_reduction: bool,
    transcript: &mut T,
) -> Result<FoldEorReplay<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let d_a = lp.role_dims().d_a();
    dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
        verify_fold_eor_kernel::<F, E, T, D>(
            extension_opening_reduction,
            challenge_point,
            openings,
            row_coefficients,
            opening_batch,
            basis,
            lp.m_vars,
            lp.r_vars,
            d_a.trailing_zeros() as usize,
            block_order,
            requires_reduction,
            transcript,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_fold_eor_kernel<F, E, T, const D: usize>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    challenge_point: &[E],
    openings: &[E],
    row_coefficients: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    m_vars: usize,
    r_vars: usize,
    alpha_bits: usize,
    block_order: BlockOrder,
    requires_reduction: bool,
    transcript: &mut T,
) -> Result<FoldEorReplay<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let num_claims = opening_batch.num_total_polynomials();
    if openings.len() != num_claims || row_coefficients.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    let mut eor_trace_final: Option<(E, Vec<E>)> = None;
    let reduction_check = if let Some(reduction) = extension_opening_reduction {
        if <E as ExtField<F>>::EXT_DEGREE == 1 {
            return Err(AkitaError::InvalidProof);
        }
        let shape = eor_reduction_shape::<F, E>(
            opening_batch.max_num_vars(),
            reduction.partials.len(),
            num_claims,
        )?;
        if challenge_point.len() > opening_batch.max_num_vars() {
            return Err(AkitaError::InvalidProof);
        }
        let mut eor_point = challenge_point.to_vec();
        eor_point.resize(opening_batch.max_num_vars(), E::zero());
        for (claim_idx, opening) in openings.iter().copied().enumerate().take(num_claims) {
            let partial_start = claim_idx * shape.width;
            let partial_end = partial_start + shape.width;
            let partials = &reduction.partials[partial_start..partial_end];
            let expected =
                derive_tensor_extension_opening_claim_from_partials::<F, E>(&eor_point, partials)?;
            if expected != opening {
                return Err(AkitaError::InvalidProof);
            }
            for partial in partials {
                append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
            }
        }
        let eta = (0..shape.split_bits)
            .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
            .collect::<Vec<_>>();
        let input_claim = eor_input_claim_from_partials::<F, E>(
            &reduction.partials,
            shape,
            &eta,
            row_coefficients,
        )?;
        let (final_claim, rho) = verify_extension_opening_reduction_sumcheck::<F, T, E, _>(
            input_claim,
            shape.num_rounds,
            &reduction.sumcheck,
            transcript,
            |tr| sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND),
        )?;
        let final_factor = tensor_equality_factor_eval_at_point::<F, E>(
            &eor_point[shape.split_bits..],
            &eta,
            &rho,
        )?;
        eor_trace_final = Some((final_claim, vec![final_factor]));
        Some(rho)
    } else if requires_reduction && <E as ExtField<F>>::EXT_DEGREE != 1 {
        return Err(AkitaError::InvalidProof);
    } else {
        None
    };

    let prepared_points = if let Some(rho) = &reduction_check {
        let protocol_point =
            ring_subfield_packed_extension_opening_point::<F, E, D>(rho.len(), rho)?;
        let prepared = prepare_opening_point::<F, E, D>(
            &protocol_point,
            basis,
            m_vars,
            r_vars,
            alpha_bits,
            block_order,
        )?;
        vec![prepared]
    } else {
        vec![prepare_opening_point::<F, E, D>(
            challenge_point,
            basis,
            m_vars,
            r_vars,
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

pub(in crate::protocol::core) struct PreparedFoldReplay<'a, F: FieldCore, E: FieldCore> {
    pub(in crate::protocol::core) lp: &'a LevelParams,
    pub(in crate::protocol::core) relation_matrix_row_layout: RelationMatrixRowLayout,
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
    pub(in crate::protocol::core) stage1: Option<&'a AkitaStage1Proof<E>>,
    pub(in crate::protocol::core) stage2: &'a AkitaStage2Proof<F, E>,
    pub(in crate::protocol::core) next_w_commitment: Option<&'a RingVec<F>>,
    /// Schedule ring dimension of the next fold level. `Some` for
    /// intermediate levels (the dimension `next_w_commitment` is shaped at,
    /// which may differ from the current level's `D` in mixed-D schedules);
    /// `None` for terminal levels.
    pub(in crate::protocol::core) next_ring_dim: Option<usize>,
    pub(in crate::protocol::core) terminal_replay: Option<TerminalWitnessTranscriptParts>,
    pub(in crate::protocol::core) stage3: Option<(&'a SetupSumcheckProof<E>, &'a LevelParams)>,
    /// Per-group prepared opening points in `OpeningClaims` order (one element
    /// for scalar/suffix folds). Reused for the fused trace term.
    pub(in crate::protocol::core) trace_prepared_points: Option<Vec<PreparedOpeningPoint<F, E>>>,
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
fn verify_stage2<F, E, T>(
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
    trace: Option<TraceWireAtRoleA<'_, F, E>>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let witness_oracle = match stage2 {
        AkitaStage2Proof::Terminal(proof) => {
            stage2_cleartext_oracle::<F, E>(&proof.final_witness, physical_w_len, lp, num_segments)?
        }
        AkitaStage2Proof::Intermediate(proof) => Stage2WitnessOracle::ClaimedEval {
            eval: proof.next_w_eval(),
        },
    };
    let d_a = lp.role_dims().d_a();
    dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
        verify_stage2_kernel::<F, E, T, D>(
            transcript,
            setup,
            stage2,
            stage1,
            rs,
            relation_claim,
            witness_oracle,
            setup_claim,
            trace.map(|wire| wire.into_claim::<D>()).transpose()?,
        )
    })
}

enum TraceWireAtRoleA<'a, F: FieldCore, E: FieldCore> {
    Recursive {
        lp: &'a LevelParams,
        layout: akita_types::TraceWeightLayout,
        trace_coeff: E,
        trace_opening_claim: E,
        prepared_point: PreparedOpeningPoint<F, E>,
        trace_basis: BasisMode,
        trace_eval_scale: E,
    },
    Root {
        lp: &'a LevelParams,
        layout: akita_types::TraceWeightLayout,
        prepared_point: PreparedOpeningPoint<F, E>,
        trace_block_opening: Vec<E>,
        trace_basis: BasisMode,
        row_coefficients: Vec<E>,
        trace_coeff: E,
        trace_eval_target: E,
        trace_claim_scales: Option<Vec<E>>,
        opening_batch: OpeningClaimsLayout,
    },
    MultiGroupRoot {
        lp: &'a LevelParams,
        layout: akita_types::TraceWeightLayout,
        prepared_points: Vec<PreparedOpeningPoint<F, E>>,
        row_coefficients: Vec<E>,
        trace_coeff: E,
        trace_eval_target: E,
        trace_claim_scales: Option<Vec<E>>,
        opening_batch: OpeningClaimsLayout,
        live_x_cols: usize,
    },
}

impl<'a, F, E> TraceWireAtRoleA<'a, F, E>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    fn into_claim<const D: usize>(self) -> Result<TraceClaim<F, E, D>, AkitaError> {
        match self {
            Self::Recursive {
                lp,
                layout,
                trace_coeff,
                trace_opening_claim,
                prepared_point,
                trace_basis,
                trace_eval_scale,
            } => Ok(TraceClaim {
                layout,
                trace_coeff,
                trace_opening_claim,
                trace_terms: trace_terms_recursive(
                    &prepared_point,
                    lp,
                    trace_basis,
                    trace_eval_scale,
                )?,
                dense_evals: None,
            }),
            Self::Root {
                lp,
                layout,
                prepared_point,
                trace_block_opening,
                trace_basis,
                row_coefficients,
                trace_coeff,
                trace_eval_target,
                trace_claim_scales,
                opening_batch,
            } => build_trace_claim_root::<F, E, D>(
                layout,
                lp,
                &opening_batch,
                &prepared_point,
                &trace_block_opening,
                trace_basis,
                &row_coefficients,
                trace_coeff,
                trace_eval_target,
                trace_claim_scales.as_deref(),
            ),
            Self::MultiGroupRoot {
                lp,
                layout,
                prepared_points,
                row_coefficients,
                trace_coeff,
                trace_eval_target,
                trace_claim_scales,
                opening_batch,
                live_x_cols,
            } => build_trace_claim_multi_group_root::<F, E, D>(
                layout,
                lp,
                &opening_batch,
                &prepared_points,
                &row_coefficients,
                trace_claim_scales.as_deref(),
                trace_coeff,
                trace_eval_target,
                live_x_cols,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_stage2_kernel<F, E, T, const D: usize>(
    transcript: &mut T,
    setup: &AkitaVerifierSetup<F>,
    stage2: &AkitaStage2Proof<F, E>,
    stage1: Stage1Replay<E>,
    rs: &RingSwitchVerifyOutput<E>,
    relation_claim: E,
    witness_oracle: Stage2WitnessOracle<'_, F, E>,
    setup_claim: Option<E>,
    trace: Option<TraceClaim<F, E, D>>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let stage2_verifier = AkitaStage2Verifier::new(
        stage1.batching_coeff,
        stage1.s_claim,
        witness_oracle,
        stage1.stage1_point,
        rs.alpha_evals_y.clone(),
        rs.relation_matrix_evaluator.clone(),
        setup_claim,
        &setup.expanded,
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

fn verify_stage3<F, E, T>(
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    rs: &RingSwitchVerifyOutput<E>,
    sumcheck_challenges: &[E],
    stage2_next_w_eval: E,
    stage3: Option<(&SetupSumcheckProof<E>, &LevelParams)>,
    role_d_a: usize,
) -> Result<Option<Vec<E>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
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
        let rho_w = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            role_d_a,
            |D| {
                let verifier = SetupSumcheckVerifier::new::<F, D>(
                    &rs.relation_matrix_evaluator,
                    setup_x_challenges,
                    rs.alpha,
                )?;
                let rho_w = verifier.verify_batched_stage3::<F, T>(
                    setup,
                    next_fold_level_params,
                    role_d_a,
                    proof,
                    stage2_next_w_eval,
                    sumcheck_challenges,
                    witness_rounds,
                    eta,
                    transcript,
                )?;
                transcript.absorb_and_record_serde(ABSORB_STAGE3_NEXT_W_EVAL, &proof.next_w_eval);
                Ok(rho_w)
            }
        )?;
        return Ok(Some(rho_w));
    }
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(in crate::protocol::core) fn verify_fold<F, E, T>(
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    prepared: PreparedFoldReplay<'_, F, E>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let opening_shape = prepared.opening_shape.clone();
    let num_groups = opening_shape.num_groups();
    let commitment_rows = &prepared.commitment_rows;
    let role_dims = prepared.lp.role_dims();
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Outer),
        F,
        role_dims.d_b(),
        |D| commitment_rows.as_ring_slice::<D>().map(|_| ())
    )?;
    validate_fold_grind_nonce(
        &prepared.lp.fold_witness_grind_contract(
            opening_shape.num_total_polynomials(),
            FoldLinfProtocolBinding::CURRENT.max_grind_attempts,
        )?,
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
    if prepared.group_ring_opening_points.len() != num_groups
        || prepared.group_ring_multiplier_points.len() != num_groups
    {
        return Err(AkitaError::InvalidProof);
    }
    let group_challenges = derive_multi_group_stage1_challenges::<F, T>(
        transcript,
        prepared.v.coeffs(),
        role_dims.d_a(),
        &opening_shape,
        prepared.lp,
        prepared.relation_matrix_row_layout,
        prepared.fold_grind_nonce,
    )?;
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
    let relation_rhs_layout = relation_rhs_layout_for(
        prepared.lp,
        &opening_shape,
        prepared.relation_matrix_row_layout,
    )?;
    let relation_rhs = assemble_relation_rhs::<F>(
        role_dims,
        &relation_rhs_layout,
        &prepared.v,
        commitment_rows,
    )?;
    let relation_instance = RingRelationInstance::new(
        prepared.relation_matrix_row_layout,
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
    let ring_switch_replay = RingSwitchReplay {
        relation: &relation_instance,
        row_coefficients: &prepared.row_coefficients,
        lp: prepared.lp,
    };
    let d_a = role_dims.d_a();
    let rs =
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            d_a,
            |D| match prepared.stage2 {
                AkitaStage2Proof::Intermediate(_) => {
                    let next_w_commitment =
                        prepared.next_w_commitment.ok_or(AkitaError::InvalidProof)?;
                    let next_ring_dim = prepared.next_ring_dim.ok_or(AkitaError::InvalidProof)?;
                    ring_switch_verifier::<F, E, T, D>(
                        &ring_switch_replay,
                        prepared.w_len,
                        next_w_commitment,
                        next_ring_dim,
                        transcript,
                    )
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
                    )
                }
            }
        )?;
    let relation_claim = relation_claim_from_layout_extension::<F, E>(
        relation_instance.role_dims(),
        &relation_rhs_layout,
        &rs.tau1,
        rs.alpha,
        relation_instance.v(),
        commitment_rows,
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
    let trace_wire = if prepared.trace_prepared_points.is_none() {
        None
    } else if prepared.trace_block_opening.is_none() {
        let segment_layout = relation_instance.segment_layout(prepared.lp, None)?;
        let layout = trace_weight_layout_from_segment(
            prepared.lp,
            &segment_layout,
            rs.col_bits,
            rs.ring_bits,
            prepared.lp.num_blocks,
        )?;
        let prepared_point = prepared
            .trace_prepared_points
            .as_ref()
            .and_then(|points| points.first())
            .ok_or(AkitaError::InvalidProof)?;
        Some(TraceWireAtRoleA::Recursive {
            lp: prepared.lp,
            layout,
            trace_coeff,
            trace_opening_claim: trace_coeff * prepared.trace_eval_target,
            prepared_point: prepared_point.clone(),
            trace_basis: prepared.trace_basis,
            trace_eval_scale: prepared.trace_eval_scale,
        })
    } else if prepared.lp.has_precommitted_groups() {
        // Grouped root: dense trace-weight table (per-group `num_blocks`,
        // `num_digits_open`, and group-major e-hat offset). The layout is inert
        // for the dense path; size it against the scalar block count so
        // `trace_weight_layout_from_segment` accepts it.
        let segment_layout = relation_instance.segment_layout(prepared.lp, None)?;
        let num_trace_blocks = relation_instance
            .opening_batch()
            .num_total_polynomials()
            .checked_mul(prepared.lp.num_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("trace block count overflow".to_string()))?;
        let layout = trace_weight_layout_from_segment(
            prepared.lp,
            &segment_layout,
            rs.col_bits,
            rs.ring_bits,
            num_trace_blocks,
        )?;
        if d_a == 0 || !prepared.w_len.is_multiple_of(d_a) {
            return Err(AkitaError::InvalidProof);
        }
        let live_x_cols = prepared.w_len / d_a;
        let col_bits = u32::try_from(rs.col_bits).map_err(|_| {
            AkitaError::InvalidSetup("multi-group trace column bits overflow".to_string())
        })?;
        let max_live_x_cols = 1usize.checked_shl(col_bits).ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group trace column bound overflow".to_string())
        })?;
        if live_x_cols > max_live_x_cols {
            return Err(AkitaError::InvalidSize {
                expected: max_live_x_cols,
                actual: live_x_cols,
            });
        }
        let prepared_points = prepared
            .trace_prepared_points
            .as_ref()
            .ok_or(AkitaError::InvalidProof)?
            .clone();
        Some(TraceWireAtRoleA::MultiGroupRoot {
            lp: prepared.lp,
            layout,
            prepared_points,
            row_coefficients: prepared.row_coefficients.clone(),
            trace_coeff,
            trace_eval_target: prepared.trace_eval_target,
            trace_claim_scales: prepared.trace_claim_scales.clone(),
            opening_batch: relation_instance.opening_batch().clone(),
            live_x_cols,
        })
    } else {
        let segment_layout = relation_instance.segment_layout(prepared.lp, None)?;
        let num_trace_blocks = relation_instance
            .opening_batch()
            .num_total_polynomials()
            .checked_mul(prepared.lp.num_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("trace block count overflow".to_string()))?;
        let layout = trace_weight_layout_from_segment(
            prepared.lp,
            &segment_layout,
            rs.col_bits,
            rs.ring_bits,
            num_trace_blocks,
        )?;
        Some(TraceWireAtRoleA::Root {
            lp: prepared.lp,
            layout,
            prepared_point: prepared
                .trace_prepared_points
                .as_ref()
                .and_then(|points| points.first())
                .ok_or(AkitaError::InvalidProof)?
                .clone(),
            trace_block_opening: prepared
                .trace_block_opening
                .as_ref()
                .ok_or(AkitaError::InvalidProof)?
                .clone(),
            trace_basis: prepared.trace_basis,
            row_coefficients: prepared.row_coefficients.clone(),
            trace_coeff,
            trace_eval_target: prepared.trace_eval_target,
            trace_claim_scales: prepared.trace_claim_scales.clone(),
            opening_batch: relation_instance.opening_batch().clone(),
        })
    };
    let setup_claim = prepared.stage3.as_ref().map(|(proof, _)| proof.claim);
    let sumcheck_challenges = verify_stage2::<F, E, T>(
        transcript,
        setup,
        prepared.stage2,
        prepared.w_len,
        stage1_replay,
        &rs,
        relation_claim,
        prepared.lp,
        num_groups,
        setup_claim,
        trace_wire,
    )?;
    let stage2_next_w_eval = if prepared.stage3.is_some() {
        prepared.stage2.next_w_eval()
    } else {
        E::zero()
    };
    let stage3_challenges = verify_stage3::<F, E, T>(
        setup,
        transcript,
        &rs,
        &sumcheck_challenges,
        stage2_next_w_eval,
        prepared.stage3,
        d_a,
    )?;
    Ok(stage3_challenges.unwrap_or(sumcheck_challenges))
}
