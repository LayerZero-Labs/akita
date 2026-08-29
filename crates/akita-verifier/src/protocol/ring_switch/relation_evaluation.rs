//! Mode-bound prepared relation evaluation for every role geometry.
//!
//! Quotient lifting factors the common low alpha coordinates and evaluates its
//! explicit quotient tail. Reduced evaluation prepares exact terminal
//! coefficient functionals, performs the same structured/direct setup work,
//! and returns that already-complete flat MLE without either lifted-only step.

use super::{
    prepared_relation_point::PreparedRelationPoint, PreparedRelationGroups, RelationMatrixEvaluator,
};
use akita_algebra::offset_eq::OffsetEqWindow;
use akita_error::AkitaError;
use akita_types::{
    gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup, FpExtEncoding,
    RelationAddressGeometry, RelationQuotientLayout, RelationRowFamily, RelationWitnessGeometry,
    RingRelationMode,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

pub(super) fn evaluate_relation_at_point<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    point: &[E],
    setup: &AkitaExpandedSetup<F>,
    alpha: E,
    deferred_setup_claim: Option<E>,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
{
    let context = evaluator
        .flat_context
        .as_ref()
        .ok_or(AkitaError::InvalidProof)?;
    let mode = match &evaluator.groups {
        PreparedRelationGroups::QuotientLift(_) => RingRelationMode::QuotientLift,
        PreparedRelationGroups::ReducedEvaluation(_) => RingRelationMode::ReducedEvaluation,
    };
    if context.level_params.ring_relation_mode != mode {
        return Err(AkitaError::InvalidSetup(
            "prepared relation groups disagree with the authenticated mode".into(),
        ));
    }
    if mode.is_reduced_evaluation() && deferred_setup_claim.is_some() {
        return Err(AkitaError::InvalidProof);
    }
    match (mode, context.witness_layout.relation_quotient_layout()) {
        (RingRelationMode::QuotientLift, RelationQuotientLayout::QuotientLift { .. })
        | (RingRelationMode::ReducedEvaluation, RelationQuotientLayout::ReducedEvaluation) => {}
        _ => {
            return Err(AkitaError::InvalidSetup(
                "relation evaluator mode disagrees with its witness layout".into(),
            ));
        }
    }
    let row_families = if mode == RingRelationMode::QuotientLift {
        Some(
            RelationWitnessGeometry::for_level(
                &context.level_params,
                &context.opening_batch,
                context.extension_degree,
            )?
            .rhs_layout()
            .row_families()?,
        )
    } else {
        None
    };
    let quotient_row_dims = row_families
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .filter(|family| {
            !matches!(
                family,
                RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
            )
        })
        .map(|family| family.geometry().polynomial_modulus_dimension())
        .collect::<Vec<_>>();
    let prepared_point = PreparedRelationPoint::new(
        point,
        alpha,
        evaluator.relation_address_geometry,
        &quotient_row_dims,
        mode,
    )?;
    if evaluator.relation_address_geometry != prepared_point.relation_address_geometry() {
        return Err(AkitaError::InvalidProof);
    }
    // The setup projection and flat relation point use the same common base of
    // the current commitment roles. Outgoing witness packaging affects only
    // the checked flat live length. The same plan therefore owns the mixed
    // E/T/Z contraction, direct setup scan, and deferred Stage-3 geometry.
    let fold_gadget = evaluator.setup_contribution_fold_gadget::<F>()?;
    let mut plan = {
        let _span = tracing::info_span!("relation_setup_plan").entered();
        let fold_gadget = fold_gadget.as_deref().unwrap_or(&[]);
        evaluator.setup_contribution_plan::<F>(
            prepared_point.relation_address().clone(),
            (!fold_gadget.is_empty()).then_some(fold_gadget),
        )?
    };
    if deferred_setup_claim.is_none() {
        let _span =
            tracing::info_span!("relation_setup_weights", required = plan.required()).entered();
        plan.materialize_direct_scan(prepared_point.coefficient_functional())?;
    }

    let mut structured_evaluation = E::zero();
    {
        let _span = tracing::info_span!("relation_structured_groups").entered();
        match &evaluator.groups {
            PreparedRelationGroups::QuotientLift(groups) => {
                for group in groups {
                    structured_evaluation += plan
                        .evaluate_structured_group::<F>(
                            group.group_id,
                            &group.multipliers.c_alphas,
                            &group.multipliers.opening_a_evals,
                            alpha,
                        )
                        .map_err(|error| {
                            AkitaError::InvalidInput(format!(
                                "relation group {} contraction failed: {error:?}",
                                group.group_id
                            ))
                        })?;
                }
            }
            PreparedRelationGroups::ReducedEvaluation(groups) => {
                for group in groups {
                    structured_evaluation += plan
                        .evaluate_reduced_structured_group::<F>(
                            group.group_id,
                            &group.multipliers.challenges,
                            &group.multipliers.opening,
                        )
                        .map_err(|error| {
                            AkitaError::InvalidInput(format!(
                                "relation group {} contraction failed: {error:?}",
                                group.group_id
                            ))
                        })?;
                }
            }
        }
    }

    let setup_evaluation = if let Some(claim) = deferred_setup_claim {
        claim
    } else {
        let _span =
            tracing::info_span!("relation_setup_scan", required = plan.required()).entered();
        plan.evaluate_direct::<F>(setup)?
    };
    let relation_evaluation = structured_evaluation + setup_evaluation;
    match mode {
        RingRelationMode::QuotientLift => {
            let quotient_evaluation = evaluate_quotient_tail::<F, E>(
                evaluator,
                &prepared_point,
                row_families.as_deref().ok_or(AkitaError::InvalidProof)?,
            )
            .map_err(|error| {
                AkitaError::InvalidInput(format!("relation quotient failed: {error:?}"))
            })?;
            if deferred_setup_claim.is_some() {
                evaluator.cache_setup_contribution_plan(prepared_point.address_point(), plan)?;
            }
            Ok(prepared_point.common_alpha_evaluation()?
                * (relation_evaluation + quotient_evaluation))
        }
        RingRelationMode::ReducedEvaluation => Ok(relation_evaluation),
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_quotient_tail<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    prepared_point: &PreparedRelationPoint<E>,
    row_families: &[RelationRowFamily],
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F>,
{
    let context = evaluator
        .flat_context
        .as_ref()
        .ok_or(AkitaError::InvalidProof)?;
    let rows = row_families.len();
    if rows
        != context
            .level_params
            .relation_matrix_row_count(context.opening_batch.num_groups())?
    {
        return Err(AkitaError::InvalidSetup(
            "relation quotient row dimensions disagree with the matrix layout".into(),
        ));
    }
    let levels = r_decomp_levels::<F>(evaluator.log_basis);
    let quotient_gadget = gadget_row_scalars::<F>(levels, evaluator.log_basis);
    let mut evaluation = E::zero();
    for (row, family) in row_families.iter().enumerate() {
        if matches!(
            family,
            RelationRowFamily::CompressionF { .. }
                | RelationRowFamily::CompressionH { .. }
                | RelationRowFamily::Consistency {
                    opening_method: akita_types::OpeningMethod::SubringCoefficientPacking { .. },
                    ..
                }
        ) {
            continue;
        }
        let row_dimension = family.geometry().polynomial_modulus_dimension();
        let role_factors = prepared_point.for_dimension(row_dimension)?;
        let denominator = role_factors
            .powers
            .last()
            .copied()
            .ok_or(AkitaError::InvalidProof)?
            * prepared_point.alpha()
            + E::one();
        let row_weight = evaluator
            .eq_tau1
            .get(row)
            .copied()
            .ok_or(AkitaError::InvalidProof)?;
        let mut row_evaluation = E::zero();
        for (digit, &gadget) in quotient_gadget.iter().enumerate() {
            let physical_coefficient = context
                .witness_layout
                .r_coefficient_index(row, digit, 0, 0)?;
            let lane_start = canonical_relation_lane_index(
                evaluator.relation_address_geometry,
                physical_coefficient,
            )?;
            let lane_evaluation = evaluate_lane_segment(
                prepared_point.relation_address().equality_window(),
                lane_start,
                &role_factors.lane_powers,
            )?;
            row_evaluation += lane_evaluation.mul_base(gadget);
        }
        evaluation -= row_evaluation * row_weight * denominator;
    }
    Ok(evaluation)
}

fn evaluate_lane_segment<E: Field>(
    equality_window: &OffsetEqWindow<E>,
    lane_start: usize,
    lane_alpha_powers: &[E],
) -> Result<E, AkitaError> {
    lane_alpha_powers
        .iter()
        .enumerate()
        .try_fold(E::zero(), |sum, (lane, &alpha_power)| {
            let index = lane_start
                .checked_add(lane)
                .ok_or_else(|| AkitaError::InvalidSetup("relation lane address overflow".into()))?;
            Ok(sum + equality_window.eval(index) * alpha_power)
        })
}

fn canonical_relation_lane_index(
    geometry: RelationAddressGeometry,
    physical_coefficient: usize,
) -> Result<usize, AkitaError> {
    let coeff_count = geometry.relation_coefficient_block_len();
    if physical_coefficient >= geometry.digit_witness_domain().live_len()
        || !physical_coefficient.is_multiple_of(coeff_count)
    {
        return Err(AkitaError::InvalidProof);
    }
    Ok(physical_coefficient / coeff_count)
}
