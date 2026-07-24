//! Succinct lane-factored relation evaluation.
//!
//! This path is used when the flat relation point has more than one coefficient
//! lane per witness column. It factors the common low alpha coordinates once,
//! contracts E/T intervals, and scans setup matrices in their native role
//! rings. It never constructs prover relation events or a dense relation table.

use super::{
    group_block_challenges, prepared_relation_point::PreparedRelationPoint, RelationMatrixEvaluator,
};
use akita_algebra::offset_eq::OffsetEqWindow;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, MulBaseUnreduced,
};
use akita_types::{
    checked_opening_source_index, gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup,
    FpExtEncoding, SetupContributionPlan,
};

pub(super) fn evaluate_lane_factored_relation_at_point<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    setup: &AkitaExpandedSetup<F>,
    prepared_point: &PreparedRelationPoint<'_, E>,
    setup_plan: &SetupContributionPlan<E>,
    deferred_setup_claim: Option<E>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    let mut structured_evaluation = E::zero();
    for group in &evaluator.groups {
        let block_challenges = group_block_challenges::<F, E>(group)?;
        structured_evaluation += setup_plan
            .evaluate_structured_group::<F>(
                group.group_id,
                &block_challenges,
                &group.opening_a_evals,
                prepared_point.alpha(),
            )
            .map_err(|error| {
                AkitaError::InvalidInput(format!(
                    "mixed relation group {} contraction failed: {error:?}",
                    group.group_id
                ))
            })?;
    }

    let setup_evaluation = if let Some(claim) = deferred_setup_claim {
        claim
    } else {
        let _span = tracing::info_span!("mixed_relation_setup_scan").entered();
        setup_plan
            .evaluate_direct::<F>(setup, prepared_point.alpha())
            .map_err(|error| {
                AkitaError::InvalidInput(format!("mixed relation setup scan failed: {error:?}"))
            })?
    };
    let quotient_evaluation =
        evaluate_quotient_tail::<F, E>(evaluator, prepared_point).map_err(|error| {
            AkitaError::InvalidInput(format!("mixed relation quotient failed: {error:?}"))
        })?;

    Ok(prepared_point.coeff_eval()
        * (structured_evaluation + setup_evaluation + quotient_evaluation))
}

fn evaluate_quotient_tail<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    prepared_point: &PreparedRelationPoint<'_, E>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    let context = evaluator
        .flat_context
        .as_ref()
        .ok_or(AkitaError::InvalidProof)?;
    let rows = context
        .level_params
        .relation_matrix_row_count(context.opening_batch.num_groups())?;
    let levels = r_decomp_levels::<F>(evaluator.log_basis);
    let quotient_gadget = gadget_row_scalars::<F>(levels, evaluator.log_basis);
    let d_row_start = rows
        .checked_sub(context.level_params.open_commit_matrix.output_rank())
        .ok_or(AkitaError::InvalidProof)?;
    let b_row_ranges = (0..context.opening_batch.num_groups())
        .map(|group| {
            context
                .level_params
                .commitment_row_range(&context.opening_batch, group)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut evaluation = E::zero();
    for row in 0..rows {
        let row_dimension = if row >= d_row_start {
            evaluator.role_dims.d_d()
        } else if b_row_ranges.iter().any(|range| range.contains(&row)) {
            evaluator.role_dims.d_b()
        } else {
            evaluator.role_dims.d_a()
        };
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
                context.opening_source_len,
                context.opening_ring_dim,
                evaluator.role_dims.d_a(),
                prepared_point.coeff_count(),
                witness_column,
                0,
            )?;
            let lane_evaluation = evaluate_lane_segment(
                prepared_point.equality_window(),
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

#[allow(clippy::too_many_arguments)]
fn canonical_relation_lane_index(
    opening_source_len: usize,
    opening_ring_dimension: usize,
    inner_ring_dimension: usize,
    coeff_count: usize,
    witness_column: usize,
    inner_lane: usize,
) -> Result<usize, AkitaError> {
    let lanes_per_inner_column = inner_ring_dimension
        .checked_div(coeff_count)
        .filter(|count| *count != 0)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid common relation lane width".into()))?;
    if inner_lane >= lanes_per_inner_column || opening_ring_dimension == 0 {
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
    checked_opening_source_index(
        opening_source_len,
        physical_coefficient / opening_ring_dimension,
    )?;
    if !physical_coefficient.is_multiple_of(coeff_count) {
        return Err(AkitaError::InvalidProof);
    }
    Ok(physical_coefficient / coeff_count)
}
