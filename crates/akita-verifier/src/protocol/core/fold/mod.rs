//! Shared per-fold verifier replay (EOR, stage-1/2/3, ring switch).

mod coefficient_packing;
mod extension_claim;
mod single_field;

use super::*;
use crate::stages::stage2::{Stage2CompressionOracle, Stage2OpeningSemantics};
use akita_algebra::offset_eq::EqPairTensorFamily;
use akita_types::{
    batch_l2_virtual_evaluations, dispatch_for_field, DigitRangeEqualityPoint, DigitRangePlan,
    OpeningFamily, RingRelationGroupOpening,
};

pub(in crate::protocol::core) use coefficient_packing::{
    verify_coefficient_packing_root_prefix, verify_coefficient_packing_suffix_prefix_native,
};
pub(in crate::protocol::core) use extension_claim::{
    verify_extension_claim_suffix_prefix_native, verify_extension_claim_terminal_suffix_native,
};
pub(in crate::protocol::core) use single_field::prepare_single_field_suffix_groups;

/// Common prepared fold prefix consumed by root and suffix finishing logic.
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

pub(in crate::protocol::core) fn finalize_native_claims<F, E>(
    opening_shape: &OpeningClaimsLayout,
    material: FoldClaimMaterial<F, E>,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    level: u32,
) -> Result<FoldPrefix<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    if material.openings.len() != opening_shape.num_total_polynomials()
        || material.prepared_points.len() != opening_shape.num_groups()
    {
        return Err(AkitaError::InvalidProof);
    }
    let row_coefficients = akita_types::verify_row_coefficients_native::<F, E>(
        opening_shape,
        akita_types::GrindingSite::EvaluationBatch { level },
        grinding,
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

pub(in crate::protocol::core) struct NativePreparedFoldReplay<'a, F: Field, E: Field> {
    pub(in crate::protocol::core) lp: &'a CommittedGroupParams,
    pub(in crate::protocol::core) level: u32,
    pub(in crate::protocol::core) opening_payload: RingVec<F>,
    pub(in crate::protocol::core) opening_shape: OpeningClaimsLayout,
    pub(in crate::protocol::core) commitment_payloads: Vec<RingVec<F>>,
    pub(in crate::protocol::core) prefix: FoldPrefix<F, E>,
    pub(in crate::protocol::core) w_len: usize,
    pub(in crate::protocol::core) level_layout: akita_types::NativeNonterminalLevelLayout,
    pub(in crate::protocol::core) next_witness: NativeNextWitnessPlan,
    pub(in crate::protocol::core) next_witness_ring_dim: usize,
    pub(in crate::protocol::core) next_opening_source_len: usize,
    pub(in crate::protocol::core) stage3: Option<&'a CommittedGroupParams>,
    pub(in crate::protocol::core) evaluation_trace_basis: BasisMode,
}

#[derive(Clone, Copy)]
pub(in crate::protocol::core) enum NativeNextWitnessPlan {
    OuterPayload { coefficient_count: usize },
    TerminalT { coefficient_count: usize },
}

pub(in crate::protocol::core) struct NativeFoldVerifyOutput<F: Field, E: Field> {
    pub(in crate::protocol::core) challenges: Vec<E>,
    pub(in crate::protocol::core) setup_prefix_opening: Option<SetupPrefixOpening<E>>,
    pub(in crate::protocol::core) next_witness: RingVec<F>,
    pub(in crate::protocol::core) opening: E,
}

struct Stage1Replay<'a, E: Field> {
    batching_coeff: E,
    compression: Stage2CompressionOracle<'a, E>,
    range_image_evaluation: E,
    stage1_point: Vec<E>,
    physical_l2_claim: E,
    physical_l2_families: Vec<EqPairTensorFamily<E>>,
}

fn verify_stage1_native<'a, F, E>(
    rs: &'a RingSwitchVerifyOutput<E>,
    lp: &CommittedGroupParams,
    relation_plan: &RelationRangeImagePlan,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    level: u32,
    layout: &akita_types::NativeNonterminalLevelLayout,
) -> Result<Stage1Replay<'a, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: ExtField<F> + FpExtEncoding<F> + Ring + AkitaSerialize,
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
    let stage1_verifier = AkitaStage1Verifier::new(equality_point, DigitRangePlan::new(rs.b)?);
    let (stage1_stages, stage1_norm) = DigitRangePlan::new(rs.b)?
        .proof_shapes_for_route(num_rounds, lp.inner().matrix.security_route())?;
    if stage1_stages != layout.stage1_stages() || stage1_norm.as_ref() != layout.stage1_norm() {
        return Err(AkitaError::InvalidSetup(
            "native Stage 1 replay disagrees with the level grammar".into(),
        ));
    }
    let physical_plan = PhysicalResponsePlan::new(lp, relation_plan)?;
    let physical = physical_plan
        .as_ref()
        .map(|plan| {
            let InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap, ..
            } = lp.inner().matrix.security_route()
            else {
                return Err(AkitaError::InvalidSetup(
                    "physical response plan exists for a non-L2 route".into(),
                ));
            };
            Ok((
                plan,
                lp.inner().matrix.sis_modulus_profile(),
                response_l2_sq_cap,
            ))
        })
        .transpose()?;
    let replay = stage1_verifier.verify_native::<F>(grinding, physical, level)?;
    let (physical_l2_claim, physical_l2_families) = match (
        replay.physical_l2_virtual_evaluations,
        physical_plan.as_ref(),
    ) {
        (Some(evaluations), Some(plan)) => {
            let eta = grinding.grinded_ext_challenge::<F, E>(
                akita_types::GrindingSite::L2VirtualBatch { level },
            )?;
            let (claim, batching) = batch_l2_virtual_evaluations(eta, &evaluations);
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
        } => Stage2CompressionOracle::QuotientLift {
            weights,
            support,
            binary_batching: grinding.grinded_ext_challenge::<F, E>(
                akita_types::GrindingSite::CompressionBinary { level },
            )?,
        },
        crate::protocol::ring_switch::PreparedStage2Compression::ReducedEvaluation {
            weights,
            support,
        } => Stage2CompressionOracle::ReducedEvaluation {
            weights,
            support,
            binary_batching: grinding.grinded_ext_challenge::<F, E>(
                akita_types::GrindingSite::CompressionBinary { level },
            )?,
        },
    };
    let batching_coeff =
        grinding.grinded_ext_challenge::<F, E>(akita_types::GrindingSite::Stage2Batch { level })?;
    Ok(Stage1Replay {
        batching_coeff,
        compression,
        range_image_evaluation: replay.range_image_evaluation,
        stage1_point: replay.point,
        physical_l2_claim,
        physical_l2_families,
    })
}

struct Stage2RoundReplay<E: Field> {
    output_claim: E,
    challenges: Vec<E>,
    witness_eval: E,
}

fn replay_stage2_native<F, E>(
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    level: u32,
    input_claim: E,
    shape: akita_sumcheck::NativeSumcheckShape,
) -> Result<Stage2RoundReplay<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let mut channel = akita_types::NativeGrindingSumcheckVerifier::<F, E>::new(
        grinding,
        akita_types::SumcheckProtocol::Stage2,
        level,
        0,
    );
    let replay = akita_sumcheck::verify_sumcheck_rounds_native::<F, E, _>(
        &mut channel,
        0,
        input_claim,
        shape,
    )?;
    let witness_eval = akita_types::native_stage2_verifier_w_eval::<F, E>(grinding, level)?;
    Ok(Stage2RoundReplay {
        output_claim: replay.output_claim,
        challenges: replay.challenges,
        witness_eval,
    })
}
#[allow(clippy::too_many_arguments)]
fn validate_stage2_replay<F, E>(
    setup: &AkitaVerifierSetup<F>,
    stage1: Stage1Replay<'_, E>,
    rs: &RingSwitchVerifyOutput<E>,
    relation_claim: E,
    setup_claim: Option<E>,
    opening_semantics: Stage2OpeningSemantics<'_, E>,
    replay: Stage2RoundReplay<E>,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    let witness_eval = replay.witness_eval;
    let stage2_verifier = AkitaStage2Verifier::<F, E>::new(
        stage1.batching_coeff,
        stage1.range_image_evaluation,
        witness_eval,
        stage1.stage1_point,
        &rs.relation_matrix_evaluator,
        stage1.compression,
        setup.expanded(),
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

    let expected = akita_sumcheck::SumcheckInstanceVerifier::expected_output_claim(
        &stage2_verifier,
        &replay.challenges,
    )?;
    if replay.output_claim != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(replay.challenges)
}

/// Replay one complete fold directly from the native Spongefish argument.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(in crate::protocol::core) fn verify_fold_native<F, E>(
    setup: &AkitaVerifierSetup<F>,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    prepared: NativePreparedFoldReplay<'_, F, E>,
) -> Result<NativeFoldVerifyOutput<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    let opening_shape = prepared.opening_shape.clone();
    if prepared.opening_payload.coeff_len() != prepared.level_layout.opening_payload_coeffs() {
        return Err(AkitaError::InvalidProof);
    }
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
    prepared.lp.validate_opening_batch(&opening_shape)?;
    if prefix.prepared_points.len() != num_groups {
        return Err(AkitaError::InvalidProof);
    }
    grinding.read_fold_response(akita_types::GrindingSite::FoldResponse {
        level: prepared.level,
    })?;
    let group_challenges = derive_multi_group_stage1_challenges_native::<F, E>(
        grinding,
        prepared.level,
        &opening_shape,
        prepared.lp,
    )?;
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
        .map(|(challenges, point)| match (challenges, point) {
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
    )?;
    if !prepared.lp.payload_mode.is_compressed() {
        relation_instance.check_v_shape_for_level(prepared.lp)?;
    }
    let next_witness = match prepared.next_witness {
        NativeNextWitnessPlan::OuterPayload { coefficient_count } => {
            if coefficient_count != prepared.level_layout.next_outer_payload_coeffs() {
                return Err(AkitaError::InvalidSetup(
                    "native successor payload disagrees with the level grammar".into(),
                ));
            }
            akita_transcript::receive_native_field_group::<F>(
                grinding.state_mut(),
                akita_transcript::ProtocolSiteId {
                    family: akita_transcript::SITE_FAMILY_NEXT_WITNESS,
                    level: prepared.level,
                    stage: 1,
                    ..akita_transcript::ProtocolSiteId::default()
                },
                coefficient_count,
            )
        }
        NativeNextWitnessPlan::TerminalT { coefficient_count } => {
            akita_transcript::receive_native_field_group::<F>(
                grinding.state_mut(),
                akita_transcript::ProtocolSiteId {
                    family: akita_transcript::SITE_FAMILY_NEXT_WITNESS,
                    level: prepared.level,
                    stage: 2,
                    ..akita_transcript::ProtocolSiteId::default()
                },
                coefficient_count,
            )
        }
    }
    .map(RingVec::from_coeffs)
    .map_err(|_| AkitaError::InvalidProof)?;
    if prepared.next_witness_ring_dim == 0
        || matches!(
            prepared.next_witness,
            NativeNextWitnessPlan::TerminalT { .. }
        ) && !next_witness.can_decode_vec(prepared.next_witness_ring_dim)
    {
        return Err(AkitaError::InvalidProof);
    }
    let ring_switch_replay = RingSwitchReplay {
        setup: setup.expanded(),
        relation: &relation_instance,
        row_coefficients: &prefix.row_coefficients,
        lp: prepared.lp,
        opening_source_len: prepared.next_opening_source_len,
        opening_ring_dim: prepared.next_witness_ring_dim,
    };
    let rs = ring_switch_verifier_native::<F, E>(
        &ring_switch_replay,
        prepared.w_len,
        grinding,
        prepared.level,
    )?;
    let relation_claim = relation_claim_from_compressed_rhs_extension::<F, E>(
        relation_rhs_layout,
        &rs.tau1,
        rs.alpha,
        relation_instance.rhs(),
    )?;
    let opening_batch = relation_instance.opening_batch();
    let relation_range_image_plan = RelationRangeImagePlan::new(
        relation_geometry,
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
    let stage1_replay = verify_stage1_native::<F, E>(
        &rs,
        prepared.lp,
        &relation_range_image_plan,
        grinding,
        prepared.level,
        &prepared.level_layout,
    )?;
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
            witness_layout: relation_range_image_plan.witness_layout(),
            level_params: prepared.lp,
            opening_batch,
            prepared_points: &evaluation_trace_points,
            claim_coefficients: &prefix.trace_claim_coefficients,
            basis: prepared.evaluation_trace_basis,
        })?;
        Stage2OpeningSemantics::evaluation_trace(
            evaluation_trace,
            evaluation_trace_weight,
            evaluation_trace_weight * prefix.trace_eval_target,
        )
    };
    let input_claim = stage1_replay.batching_coeff * stage1_replay.range_image_evaluation
        + relation_claim
        + opening_semantics.opening_claim()
        + stage1_replay.physical_l2_claim;
    let stage2_replay = replay_stage2_native::<F, E>(
        grinding,
        prepared.level,
        input_claim,
        prepared.level_layout.stage2_sumcheck(),
    )?;
    let (setup_claim, setup_prefix_opening) = if let Some(next_params) = prepared.stage3 {
        let setup_coefficient_bits = rs
            .relation_address_geometry
            .relation_coefficient_variable_count();
        let setup_x_challenges = stage2_replay
            .challenges
            .get(setup_coefficient_bits..)
            .ok_or(AkitaError::InvalidProof)?;
        let verifier = SetupSumcheckVerifier::new::<F>(
            &rs.relation_matrix_evaluator,
            setup_x_challenges,
            rs.alpha,
        )?;
        let replay =
            verifier.verify_stage3_native::<F>(setup, next_params, grinding, prepared.level)?;
        (
            Some(replay.claim),
            Some((replay.challenges, replay.setup_prefix_eval)),
        )
    } else {
        (None, None)
    };
    let opening = stage2_replay.witness_eval;
    let challenges = validate_stage2_replay(
        setup,
        stage1_replay,
        &rs,
        relation_claim,
        setup_claim,
        opening_semantics,
        stage2_replay,
    )?;
    drop(stage2_span);
    Ok(NativeFoldVerifyOutput {
        challenges,
        setup_prefix_opening,
        next_witness,
        opening,
    })
}
