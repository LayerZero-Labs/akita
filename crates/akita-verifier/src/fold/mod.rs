//! Root and suffix fold verifier replay for Akita proofs.
//!
//! This module owns the shared per-fold replay engine plus path-specific prep
//! in `verify`, `root`, and `suffix`. FoldSchedule/config dispatch stays with
//! the scheme crate until the verifier-facing config boundary is extracted.

mod challenges;
mod root;
mod suffix;
mod terminal;
mod verify;

use crate::relation::evaluation_trace::prepare_evaluation_trace;
use crate::relation::RingSwitchReplay;
use crate::stages::opening_claims::{
    finalize_native_claims, prepare_single_field_suffix_groups,
    verify_coefficient_packing_root_prefix, verify_coefficient_packing_suffix_prefix_native,
    verify_extension_claim_suffix_prefix_native, FoldClaimMaterial, FoldPrefix,
    PreparedFoldOpeningPoint,
};
use crate::stages::ring_switch::ring_switch_verifier_native;
use crate::stages::stage1::verify_stage1_native;
use crate::stages::stage2::{replay_stage2_native, validate_stage2_replay, Stage2OpeningSemantics};
use crate::stages::SetupSumcheckVerifier;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::NativeGrinding;
use akita_types::{
    assemble_compressed_relation_rhs, assemble_relation_rhs, dispatch_for_field,
    ensure_trace_stage2_supported, proof::relation::relation_row_weight,
    relation_claim_from_compressed_rhs_extension, AkitaVerifierSetup, BasisMode,
    CommittedGroupParams, DigitRangePlan, EvaluationTraceInputs, FoldParams, FoldSchedule,
    FpExtEncoding, OpeningClaims, OpeningClaimsLayout, OpeningFamily, PolynomialGroupClaims,
    RelationRangeImagePlan, RelationWitnessGeometry, RingRelationGroupOpening,
    RingRelationInstance, RingVec, SetupContributionMode, TerminalFoldParams,
};
use challenges::derive_multi_group_stage1_challenges_native;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, PseudoMersenne, Ring};

pub(crate) type SetupPrefixOpening<E> = (Vec<E>, E);

pub(crate) struct NativePreparedFoldReplay<'a, F: Field, E: Field> {
    pub(crate) lp: &'a CommittedGroupParams,
    pub(crate) level: u32,
    pub(crate) opening_payload: RingVec<F>,
    pub(crate) opening_shape: OpeningClaimsLayout,
    pub(crate) commitment_payloads: Vec<RingVec<F>>,
    pub(crate) prefix: FoldPrefix<F, E>,
    pub(crate) w_len: usize,
    pub(crate) level_layout: akita_types::NativeNonterminalLevelLayout,
    pub(crate) next_witness: NativeNextWitnessPlan,
    pub(crate) next_witness_ring_dim: usize,
    pub(crate) next_opening_source_len: usize,
    pub(crate) stage3: Option<&'a CommittedGroupParams>,
    pub(crate) evaluation_trace_basis: BasisMode,
}

#[derive(Clone, Copy)]
pub(crate) enum NativeNextWitnessPlan {
    OuterPayload { coefficient_count: usize },
    TerminalT { coefficient_count: usize },
}

pub(crate) struct NativeFoldVerifyOutput<F: Field, E: Field> {
    pub(crate) challenges: Vec<E>,
    pub(crate) setup_prefix_opening: Option<SetupPrefixOpening<E>>,
    pub(crate) next_witness: RingVec<F>,
    pub(crate) opening: E,
}

/// Replay one complete fold directly from the native Spongefish argument.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn verify_fold_native<F, E>(
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
        role_dims,
    )?;
    if !prepared.lp.payload_mode.is_compressed() {
        RingRelationInstance::check_v_shape_for_level(&prepared.opening_payload, prepared.lp)?;
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
                akita_types::NativeFoldSite::NextWitnessPayload {
                    level: prepared.level,
                }
                .id()?,
                coefficient_count,
            )
        }
        NativeNextWitnessPlan::TerminalT { coefficient_count } => {
            akita_transcript::receive_native_field_group::<F>(
                grinding.state_mut(),
                akita_types::NativeFoldSite::NextWitnessInnerState {
                    level: prepared.level,
                }
                .id()?,
                coefficient_count,
            )
        }
    }
    .map(RingVec::from_coeffs)?;
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
            crate::coefficient_packing_relation::prepare_coefficient_packing_verifier_batch_semantics(
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
