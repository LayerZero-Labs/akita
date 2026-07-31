//! Succinct prepared relation evaluation for every role geometry.
//!
//! The common low alpha coordinates are factored once. The setup plan then
//! selects contiguous q=1 or projected-lane q>1 kernels without changing the
//! verifier formula or control flow.

use super::{prepared_relation_point::PreparedRelationPoint, RelationMatrixEvaluator};
use akita_algebra::offset_eq::OffsetEqWindow;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, MulBaseUnreduced,
};
use akita_types::{
    gadget_row_scalars, r_decomp_levels, relation_rhs_layout_for, AkitaExpandedSetup,
    FpExtEncoding, RelationAddressGeometry,
};

pub(super) fn evaluate_relation_at_point<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    point: &[E],
    setup: &AkitaExpandedSetup<F>,
    alpha: E,
    deferred_setup_claim: Option<E>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    let context = evaluator
        .flat_context
        .as_ref()
        .ok_or(AkitaError::InvalidProof)?;
    let quotient_row_dims =
        relation_rhs_layout_for(&context.level_params, &context.opening_batch)?.row_ring_dims()?;
    let prepared_point = PreparedRelationPoint::new(
        point,
        alpha,
        evaluator.relation_address_geometry,
        &quotient_row_dims,
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

    let mut structured_evaluation = E::zero();
    {
        let _span = tracing::info_span!("relation_structured_groups").entered();
        for group in &evaluator.groups {
            let block_challenges = group.structured_block_challenges::<F>()?;
            structured_evaluation += plan
                .evaluate_structured_group::<F>(
                    group.group_id,
                    &block_challenges,
                    &group.opening_a_evals,
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

    let setup_evaluation = if let Some(claim) = deferred_setup_claim {
        claim
    } else {
        let _span =
            tracing::info_span!("relation_setup_scan", required = plan.required()).entered();
        plan.materialize_direct_scan(alpha)?;
        plan.evaluate_direct::<F>(
            setup,
            prepared_point.inner().powers.as_ref(),
            prepared_point.outer().powers.as_ref(),
            prepared_point.opening().powers.as_ref(),
        )?
    };
    let quotient_evaluation =
        evaluate_quotient_tail::<F, E>(evaluator, &prepared_point).map_err(|error| {
            AkitaError::InvalidInput(format!("relation quotient failed: {error:?}"))
        })?;

    let relation_evaluation = structured_evaluation + setup_evaluation + quotient_evaluation;
    evaluator.cache_setup_contribution_plan(prepared_point.address_point(), plan)?;
    Ok(prepared_point.common_alpha_evaluation() * relation_evaluation)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_quotient_tail<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    prepared_point: &PreparedRelationPoint<E>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    let context = evaluator
        .flat_context
        .as_ref()
        .ok_or(AkitaError::InvalidProof)?;
    let quotient_row_dims =
        relation_rhs_layout_for(&context.level_params, &context.opening_batch)?.row_ring_dims()?;
    let rows = quotient_row_dims.len();
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
    for (row, &row_dimension) in quotient_row_dims.iter().enumerate() {
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
        for (digit, &gadget) in quotient_gadget.iter().enumerate() {
            let witness_column = context.witness_layout.r_index(levels, row, digit)?;
            let lane_start = canonical_relation_lane_index(
                evaluator.relation_address_geometry,
                witness_column,
                0,
            )?;
            let lane_evaluation = evaluate_lane_segment(
                prepared_point.relation_address().equality_window(),
                lane_start,
                &role_factors.lane_powers,
            )?;
            evaluation -= lane_evaluation * row_weight * denominator * E::lift_base(gadget);
        }
    }
    Ok(evaluation)
}

fn evaluate_lane_segment<E: FieldCore>(
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
    witness_column: usize,
    inner_lane: usize,
) -> Result<usize, AkitaError> {
    let inner_ring_dimension = geometry.carrier_ring_dimension();
    let coeff_count = geometry.relation_coefficient_block_len();
    let lanes_per_inner_column = inner_ring_dimension
        .checked_div(coeff_count)
        .filter(|count| *count != 0)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid common relation lane width".into()))?;
    if inner_lane >= lanes_per_inner_column {
        return Err(AkitaError::InvalidProof);
    }
    let physical_coefficient = witness_column
        .checked_mul(inner_ring_dimension)
        .and_then(|base| {
            inner_lane
                .checked_mul(coeff_count)
                .and_then(|offset| base.checked_add(offset))
        })
        .ok_or_else(|| AkitaError::InvalidSetup("relation lane address overflow".into()))?;
    if physical_coefficient >= geometry.digit_witness_domain().live_len()
        || !physical_coefficient.is_multiple_of(coeff_count)
    {
        return Err(AkitaError::InvalidProof);
    }
    Ok(physical_coefficient / coeff_count)
}
