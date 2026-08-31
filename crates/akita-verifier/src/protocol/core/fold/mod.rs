//! Shared per-fold verifier replay (EOR, stage-1/2/3, ring switch).

mod coefficient_packing;
mod extension_claim;
mod single_field;

use super::*;
use crate::stages::stage2::{Stage2CompressionOracle, Stage2OpeningSemantics};
use akita_algebra::offset_eq::EqPairTensorFamily;
use akita_types::{
    dispatch_for_field, DigitRangeEqualityPoint, DigitRangePlan, OpeningFamily,
    RingRelationGroupOpening,
};

pub(in crate::protocol::core) use coefficient_packing::{
    verify_coefficient_packing_root_prefix, verify_coefficient_packing_suffix_prefix,
};
pub(in crate::protocol::core) use extension_claim::{
    verify_extension_claim_suffix_prefix, verify_extension_claim_terminal_suffix,
};
pub(in crate::protocol::core) use single_field::{
    absorb_protocol_opening_points, prepare_single_field_suffix_groups,
    prepare_single_field_terminal_suffix,
};

/// Common prepared fold prefix produced by the single-field and
/// extension-claim geometry modules, consumed by root and suffix finishing
/// logic.
pub(in crate::protocol::core) struct FoldPrefix<F: Field, E: Field> {
    pub(in crate::protocol::core) prepared_points: Vec<PreparedFoldOpeningPoint<F, E>>,
    pub(in crate::protocol::core) row_coefficients: Vec<E>,
    pub(in crate::protocol::core) trace_eval_target: E,
    pub(in crate::protocol::core) trace_claim_coefficients: Vec<E>,
    pub(in crate::protocol::core) scalar_openings: Vec<E>,
}

pub(in crate::protocol::core) type PreparedFoldOpeningPoint<F, E> = OpeningFamily<
    PreparedOpeningPoint<F, E>,
    akita_types::PreparedSubringCoefficientPackingPoint<E>,
>;

/// Fold material fixed before the shared opening payload is absorbed.
pub(in crate::protocol::core) struct FoldClaimMaterial<F: Field, E: Field> {
    pub(in crate::protocol::core) prepared_points: Vec<PreparedFoldOpeningPoint<F, E>>,
    pub(in crate::protocol::core) openings: Vec<E>,
    pub(in crate::protocol::core) reduction_final_claims: Option<Vec<E>>,
    pub(in crate::protocol::core) reduction_factors: Option<Vec<E>>,
}

pub(in crate::protocol::core) fn bind_opening_payload_and_finalize_claims<F, E, T>(
    lp: &CommittedGroupParams,
    opening_shape: &OpeningClaimsLayout,
    opening_payload: &RingVec<F>,
    material: FoldClaimMaterial<F, E>,
    transcript: &mut T,
    level: u32,
) -> Result<FoldPrefix<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: ExtField<F>,
    T: akita_types::VerifierTranscriptGrinding<F>,
{
    let geometry = RelationWitnessGeometry::for_level(lp, opening_shape, E::DEGREE)?
        .rhs_layout()
        .opening_payload_geometry()?;
    if opening_payload.coeff_len() != geometry.transmitted_coefficients() {
        return Err(AkitaError::InvalidProof);
    }
    opening_payload.append_flat_to_transcript(
        ABSORB_OPENING_PAYLOAD,
        geometry.transcript_ring_dimension(),
        transcript,
    )?;
    if material.openings.len() != opening_shape.num_total_polynomials()
        || material.prepared_points.len() != opening_shape.num_groups()
    {
        return Err(AkitaError::InvalidProof);
    }
    let row_coefficients = akita_types::sample_row_coefficients::<F, E, T>(
        opening_shape,
        akita_types::GrindingSite::EvaluationBatch { level },
        transcript,
    )?;
    let trace_claim_coefficients = material.reduction_factors.as_ref().map_or_else(
        || Ok(row_coefficients.clone()),
        |factors| opening_shape.scale_row_coefficients_by_group(&row_coefficients, factors),
    )?;
    let trace_eval_target = if let Some(final_claims) = &material.reduction_final_claims {
        if final_claims.len() != row_coefficients.len() || material.reduction_factors.is_none() {
            return Err(AkitaError::InvalidProof);
        }
        final_claims
            .iter()
            .zip(&row_coefficients)
            .fold(E::zero(), |acc, (&claim, &coefficient)| {
                acc + coefficient * claim
            })
    } else {
        if material.reduction_factors.is_some() {
            return Err(AkitaError::InvalidProof);
        }
        opening_shape.batched_eval_target(&trace_claim_coefficients, &material.openings)?
    };
    Ok(FoldPrefix {
        prepared_points: material.prepared_points,
        row_coefficients,
        trace_eval_target,
        trace_claim_coefficients,
        scalar_openings: material.openings,
    })
}

pub(in crate::protocol::core) struct PreparedFoldReplay<'a, F: Field, E: Field> {
    pub(in crate::protocol::core) lp: &'a CommittedGroupParams,
    pub(in crate::protocol::core) level: u32,
    pub(in crate::protocol::core) opening_payload: RingVec<F>,
    /// Normalized opening geometry (one group for scalar/suffix folds, `G`
    /// groups for multi-group roots).
    pub(in crate::protocol::core) opening_shape: OpeningClaimsLayout,
    /// Terminal F payloads in relation (final-first) group order.
    pub(in crate::protocol::core) commitment_payloads: Vec<RingVec<F>>,
    pub(in crate::protocol::core) prefix: FoldPrefix<F, E>,
    pub(in crate::protocol::core) w_len: usize,
    pub(in crate::protocol::core) payload: PreparedFoldPayload<'a, F, E>,
    pub(in crate::protocol::core) evaluation_trace_basis: BasisMode,
}

#[derive(Clone, Copy)]
pub(in crate::protocol::core) enum PreparedNextWitness<'a, F: Field> {
    Commitment {
        commitment: &'a RingVec<F>,
        ring_dim: usize,
    },
    TerminalT(&'a [u8]),
}

pub(in crate::protocol::core) enum PreparedFoldPayload<'a, F: Field, E: Field> {
    Recursive {
        stage1: &'a AkitaStage1Proof<E>,
        stage2: &'a AkitaStage2Proof<F, E>,
        next_witness: PreparedNextWitness<'a, F>,
        next_witness_ring_dim: usize,
        next_opening_source_len: usize,
        stage3: Option<(&'a SetupSumcheckProof<E>, &'a CommittedGroupParams)>,
    },
}

struct Stage1Replay<'a, E: Field> {
    batching_coeff: E,
    compression: Stage2CompressionOracle<'a, E>,
    range_image_evaluation: E,
    stage1_point: Vec<E>,
    physical_l2_claim: E,
    physical_l2_families: Vec<EqPairTensorFamily<E>>,
}

fn verify_stage1<'a, F, E, T>(
    proof: &AkitaStage1Proof<E>,
    rs: &'a RingSwitchVerifyOutput<E>,
    lp: &CommittedGroupParams,
    relation_plan: &RelationRangeImagePlan,
    transcript: &mut T,
    level: u32,
) -> Result<Stage1Replay<'a, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: ExtField<F> + FpExtEncoding<F> + Ring + AkitaSerialize,
    T: akita_types::VerifierTranscriptGrinding<F>,
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
    let physical_plan = PhysicalResponsePlan::new(lp, relation_plan)?;
    let (stage1_point, physical_l2_virtual_evaluations, physical_plan) =
        match (physical_plan.as_ref(), proof.norm_proof.as_ref()) {
            (None, None) => {
                let point = {
                    let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
                    stage1_verifier.verify::<F, T>(proof, transcript, level)?
                };
                (point, None, None)
            }
            (Some(plan), Some(norm_proof)) => {
                let InnerCommitSecurityRoute::L2 {
                    response_l2_sq_cap, ..
                } = lp.inner().matrix.security_route()
                else {
                    return Err(AkitaError::InvalidSetup(
                        "physical response plan exists for a non-L2 route".into(),
                    ));
                };
                let leaf = stage1_verifier.verify_product_prefix::<F, T>(
                    &proof.stages,
                    transcript,
                    level,
                )?;
                let replay = verify_physical_l2_norm::<F, E, T>(
                    plan,
                    norm_proof,
                    PhysicalL2RangeClaim {
                        equality_point: &leaf.equality_point,
                        input_claim: leaf.input_claim,
                        leaf_coefficients: &leaf.polynomial_coefficients,
                        image_evaluation: proof.range_image_evaluation,
                    },
                    lp.inner().matrix.sis_modulus_profile(),
                    response_l2_sq_cap,
                    transcript,
                    level,
                )?;
                (replay.point, Some(replay.virtual_evaluations), Some(plan))
            }
            _ => return Err(AkitaError::InvalidProof),
        };
    transcript.append_serde(ABSORB_RANGE_IMAGE_EVALUATION, &proof.range_image_evaluation);
    let (physical_l2_claim, physical_l2_families) =
        match (physical_l2_virtual_evaluations, physical_plan) {
            (Some(evaluations), Some(plan)) => {
                transcript.grind_query(akita_types::GrindingSite::L2VirtualBatch { level })?;
                let eta = sample_ext_challenge::<F, E, T>(
                    transcript,
                    akita_transcript::labels::CHALLENGE_L2_VIRTUAL_BATCH,
                );
                let mut batching = Vec::with_capacity(evaluations.len());
                let mut power = E::one();
                let mut claim = E::zero();
                for evaluation in evaluations {
                    batching.push(power);
                    claim += evaluation * power;
                    power *= eta;
                }
                (claim, plan.virtualization_families(&batching)?)
            }
            (None, None) => (E::zero(), Vec::new()),
            _ => return Err(AkitaError::InvalidProof),
        };
    let compression = match &rs.compression {
        crate::protocol::ring_switch::PreparedStage2Compression::Raw => {
            Stage2CompressionOracle::Raw
        }
        crate::protocol::ring_switch::PreparedStage2Compression::QuotientLift {
            weights,
            support,
        } => {
            transcript.grind_query(akita_types::GrindingSite::CompressionBinary { level })?;
            Stage2CompressionOracle::QuotientLift {
                weights,
                support,
                binary_batching: sample_ext_challenge::<F, E, T>(
                    transcript,
                    CHALLENGE_COMPRESSION_BINARY,
                ),
            }
        }
        crate::protocol::ring_switch::PreparedStage2Compression::ReducedEvaluation {
            weights,
            support,
        } => {
            transcript.grind_query(akita_types::GrindingSite::CompressionBinary { level })?;
            Stage2CompressionOracle::ReducedEvaluation {
                weights,
                support,
                binary_batching: sample_ext_challenge::<F, E, T>(
                    transcript,
                    CHALLENGE_COMPRESSION_BINARY,
                ),
            }
        }
    };
    transcript.grind_query(akita_types::GrindingSite::Stage2Batch { level })?;
    let batching_coeff: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
    Ok(Stage1Replay {
        batching_coeff,
        compression,
        range_image_evaluation: proof.range_image_evaluation,
        stage1_point,
        physical_l2_claim,
        physical_l2_families,
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_stage2<F, E, T>(
    transcript: &mut T,
    level: u32,
    setup: &AkitaVerifierSetup<F>,
    stage2: &AkitaStage2Proof<F, E>,
    stage1: Stage1Replay<'_, E>,
    rs: &RingSwitchVerifyOutput<E>,
    relation_claim: E,
    setup_claim: Option<E>,
    opening_semantics: Stage2OpeningSemantics<'_, E>,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
    T: akita_types::VerifierTranscriptGrinding<F>,
{
    let witness_eval = stage2.next_w_eval();
    let stage2_verifier = AkitaStage2Verifier::<F, E>::new(
        stage1.batching_coeff,
        stage1.range_image_evaluation,
        witness_eval,
        stage1.stage1_point,
        &rs.relation_matrix_evaluator,
        stage1.compression,
        &setup.expanded,
        rs.alpha,
        setup_claim,
        relation_claim,
        rs.relation_address_geometry.relation_lane_variable_count(),
        rs.relation_address_geometry
            .relation_coefficient_variable_count(),
        opening_semantics,
        stage1.physical_l2_claim,
        stage1.physical_l2_families,
    )?;

    let mut round = 0u32;
    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        stage2_verifier.verify::<F, T, _>(&stage2.sumcheck_proof, transcript, |tr| {
            let challenge = akita_types::sample_grinded_sumcheck_challenge::<F, E, T>(
                tr,
                akita_types::SumcheckProtocol::Stage2,
                level,
                0,
                round,
            )?;
            round = round.checked_add(1).ok_or(AkitaError::InvalidProof)?;
            Ok(challenge)
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
    level: u32,
) -> Result<Option<(Vec<E>, E)>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
    T: akita_types::VerifierTranscriptGrinding<F>,
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
        let setup_point = verifier.verify_stage3::<F, T>(
            setup,
            next_fold_level_params,
            proof,
            transcript,
            level,
        )?;
        return Ok(next_fold_level_params
            .setup_prefix()
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
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
    T: akita_types::VerifierTranscriptGrinding<F>,
{
    let opening_shape = prepared.opening_shape.clone();
    let num_groups = opening_shape.num_groups();
    let commitment_payloads = &prepared.commitment_payloads;
    let prefix = &prepared.prefix;
    let role_dims = prepared.lp.role_dims();
    let relation_geometry =
        RelationWitnessGeometry::for_level(prepared.lp, &opening_shape, E::DEGREE).map_err(
            |error| {
                AkitaError::InvalidInput(format!("compressed relation layout failed: {error:?}"))
            },
        )?;
    let relation_rhs_layout = relation_geometry.rhs_layout();
    if commitment_payloads.len() != num_groups {
        return Err(AkitaError::InvalidInput(
            "commitment payload group count mismatch".into(),
        ));
    }
    for (relation_group_index, payload) in commitment_payloads.iter().enumerate() {
        if payload.coeff_len()
            != relation_rhs_layout
                .group_payload_geometry(relation_group_index)?
                .transmitted_coefficients()
        {
            return Err(AkitaError::InvalidInput(
                "commitment payload length mismatch".into(),
            ));
        }
    }
    let _fold_span = tracing::info_span!(
        "verify_fold",
        d_a = role_dims.d_a(),
        d_b = role_dims.d_b(),
        d_d = role_dims.d_d(),
        groups = num_groups,
    )
    .entered();
    let fold_grind_nonce =
        transcript.read_fold_response(akita_types::GrindingSite::FoldResponse {
            level: prepared.level,
        })?;
    {
        let _span = tracing::info_span!("fold_validate_inputs").entered();
        prepared.lp.validate_opening_batch(&opening_shape)?;
        if prefix.prepared_points.len() != num_groups {
            return Err(AkitaError::InvalidProof);
        }
    }
    let group_challenges = {
        let _span = tracing::info_span!("fold_derive_stage1_challenges").entered();
        derive_multi_group_stage1_challenges::<F, E, T>(
            transcript,
            prepared.level,
            &opening_shape,
            prepared.lp,
            fold_grind_nonce,
        )
        .map_err(|error| {
            AkitaError::InvalidInput(format!("fold challenge replay failed: {error:?}"))
        })?
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
        let commitment_rows = RingVec::from_coeffs(
            commitment_payloads
                .iter()
                .flat_map(|payload| payload.coeffs().iter().copied())
                .collect(),
        );
        let relation_rhs = if prepared.lp.payload_mode.is_compressed() {
            let group_payloads = commitment_payloads
                .iter()
                .map(|payload| payload.coeffs())
                .collect::<Vec<_>>();
            assemble_compressed_relation_rhs::<F>(
                relation_rhs_layout,
                &group_payloads,
                prepared.opening_payload.coeffs(),
            )?
        } else {
            assemble_relation_rhs::<F>(
                relation_rhs_layout,
                &prepared.opening_payload,
                &commitment_rows,
            )?
        };
        let group_openings = group_challenges
            .into_iter()
            .zip(&prefix.prepared_points)
            .map(|(challenges, prepared)| match (challenges, prepared) {
                (
                    OpeningFamily::EvaluationTrace(challenges),
                    PreparedFoldOpeningPoint::EvaluationTrace(point),
                ) => Ok(RingRelationGroupOpening::evaluation_trace(
                    challenges,
                    point.ring_multiplier_point.clone(),
                )),
                (
                    OpeningFamily::SubringCoefficientPacking(challenges),
                    PreparedFoldOpeningPoint::SubringCoefficientPacking(point),
                ) if point.geometry() == challenges.geometry() => {
                    Ok(RingRelationGroupOpening::coefficient_packing(challenges))
                }
                _ => Err(AkitaError::InvalidProof),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let relation_instance = RingRelationInstance::new(
            group_openings,
            E::DEGREE,
            opening_shape.clone(),
            gamma,
            row_coefficient_rings,
            relation_rhs,
            if prepared.lp.payload_mode.is_compressed() {
                RingVec::from_coeffs(Vec::new())
            } else {
                prepared.opening_payload.clone()
            },
            role_dims,
        )
        .map_err(|error| {
            AkitaError::InvalidInput(format!("relation instance failed: {error:?}"))
        })?;
        if !prepared.lp.payload_mode.is_compressed() {
            relation_instance.check_v_shape_for_level(prepared.lp)?;
        }
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
    let rs = ring_switch_verifier::<F, E, T>(
        &ring_switch_replay,
        prepared.w_len,
        transcript,
        prepared.level,
    )
    .map_err(|error| {
        AkitaError::InvalidInput(format!("compressed ring-switch replay failed: {error:?}"))
    })?;
    let relation_claim = relation_claim_from_compressed_rhs_extension::<F, E>(
        relation_rhs_layout,
        &rs.tau1,
        rs.alpha,
        relation_instance.rhs(),
    )
    .map_err(|error| {
        AkitaError::InvalidInput(format!("compressed relation claim failed: {error:?}"))
    })?;
    // EvaluationTrace is the last padded relation row: weight openings by
    // `eq(tau1, EvaluationTrace_row_index)`.
    let opening_batch = relation_instance.opening_batch();
    let relation_range_image_plan = RelationRangeImagePlan::new(
        relation_geometry.clone(),
        rs.relation_address_geometry,
        DigitRangePlan::new(rs.b)?,
        rs.relation_matrix_evaluator.witness_layout()?.clone(),
        opening_batch,
    )?;
    let prepared_packing_points = prefix
        .prepared_points
        .iter()
        .enumerate()
        .filter_map(|(group_index, point)| match point {
            PreparedFoldOpeningPoint::SubringCoefficientPacking(point) => {
                Some((group_index, point))
            }
            PreparedFoldOpeningPoint::EvaluationTrace(_) => None,
        })
        .collect::<Vec<_>>();
    let coefficient_packing_batch = if prepared_packing_points.is_empty() {
        None
    } else {
        Some(
            akita_types::prepare_coefficient_packing_verifier_batch_semantics(
                akita_types::CoefficientPackingBatchSemanticInputs {
                    level_params: prepared.lp,
                    opening_batch,
                    relation_plan: &relation_range_image_plan,
                    relation: &relation_instance,
                    prepared_points: &prepared_packing_points,
                    alpha: rs.alpha,
                    tau1: &rs.tau1,
                    claim_coefficients: &prefix.trace_claim_coefficients,
                },
            )?,
        )
    };
    let stage1_replay = verify_stage1::<F, E, T>(
        stage1,
        &rs,
        prepared.lp,
        &relation_range_image_plan,
        transcript,
        prepared.level,
    )
    .map_err(|error| {
        AkitaError::InvalidInput(format!("compressed stage-1 replay failed: {error:?}"))
    })?;
    let trace_domain = rs.relation_address_geometry.digit_witness_domain();
    if trace_domain.live_len() != prepared.w_len {
        return Err(AkitaError::InvalidSize {
            expected: trace_domain.live_len(),
            actual: prepared.w_len,
        });
    }
    let stage2_span = tracing::info_span!(
        "stage2_verifier",
        level = prepared.level,
        relation_mode = ?prepared.lp.ring_relation_mode,
        reduced = prepared.lp.ring_relation_mode.is_reduced_evaluation(),
    )
    .entered();
    let opening_semantics = if let Some(batch) = &coefficient_packing_batch {
        if prefix
            .prepared_points
            .iter()
            .any(|point| matches!(point, PreparedFoldOpeningPoint::EvaluationTrace(_)))
        {
            return Err(AkitaError::InvalidProof);
        }
        let mut authenticated_total = E::zero();
        let mut group_openings = Vec::with_capacity(batch.groups().len());
        for semantics in batch.groups() {
            let claim_range = semantics.group_claim_range();
            let openings = prefix
                .scalar_openings
                .get(claim_range.clone())
                .ok_or(AkitaError::InvalidProof)?;
            let coefficients = prefix
                .trace_claim_coefficients
                .get(claim_range)
                .ok_or(AkitaError::InvalidProof)?;
            let authenticated = openings
                .iter()
                .zip(coefficients)
                .fold(E::zero(), |sum, (&opening, &coefficient)| {
                    sum + opening * coefficient
                });
            authenticated_total += authenticated;
            group_openings.push((semantics.group_index(), authenticated));
        }
        if authenticated_total != prefix.trace_eval_target {
            return Err(AkitaError::InvalidProof);
        }
        Stage2OpeningSemantics::packing(batch, &group_openings)?
    } else {
        let evaluation_trace_row = prepared.lp.evaluation_trace_row_index(opening_batch)?;
        let evaluation_trace_weight = relation_row_weight(evaluation_trace_row, &rs.tau1)?;
        ensure_trace_stage2_supported(<E as ExtField<F>>::DEGREE)?;
        let trace_witness_layout = relation_range_image_plan.witness_layout();
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
        let evaluation_trace_points = prefix
            .prepared_points
            .iter()
            .map(|point| match point {
                OpeningFamily::EvaluationTrace(point) => Ok(point.clone()),
                OpeningFamily::SubringCoefficientPacking(_) => Err(AkitaError::InvalidProof),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let evaluation_trace = prepare_evaluation_trace::<F, E>(&EvaluationTraceInputs {
            digit_witness_domain: trace_domain,
            relation_coefficient_block_len: rs
                .relation_address_geometry
                .relation_coefficient_block_len(),
            witness_layout: trace_witness_layout,
            level_params: prepared.lp,
            opening_batch,
            prepared_points: &evaluation_trace_points,
            claim_coefficients: &prefix.trace_claim_coefficients,
            basis: prepared.evaluation_trace_basis,
        })?;
        drop(trace_preparation_span);
        Stage2OpeningSemantics::evaluation_trace(
            evaluation_trace,
            evaluation_trace_weight,
            evaluation_trace_weight * prefix.trace_eval_target,
        )
    };
    let setup_claim = stage3.as_ref().map(|(proof, _)| proof.claim);
    let sumcheck_challenges = verify_stage2::<F, E, T>(
        transcript,
        prepared.level,
        setup,
        stage2,
        stage1_replay,
        &rs,
        relation_claim,
        setup_claim,
        opening_semantics,
    )
    .map_err(|error| {
        AkitaError::InvalidInput(format!("compressed stage-2 replay failed: {error:?}"))
    })?;
    drop(stage2_span);
    let setup_prefix_opening = verify_stage3::<F, E, T>(
        setup,
        transcript,
        &rs,
        &sumcheck_challenges,
        stage3,
        prepared.level,
    )?;
    Ok((sumcheck_challenges, setup_prefix_opening))
}
