//! Succinct lane-factored relation evaluation.
//!
//! This path is used when the flat relation point has more than one coefficient
//! lane per witness column. It factors the common low alpha coordinates once,
//! contracts E/T intervals, and scans setup matrices in their native role
//! rings. It never constructs prover relation events or a dense relation table.

use super::{
    prepared_relation_point::PreparedRelationPoint, RelationMatrixEvaluator,
    RelationMatrixGroupEvaluator,
};
use crate::protocol::validate_log_basis;
use akita_algebra::offset_eq::{eval_affine_digit_interval, OffsetEqWindow};
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, LiftBase, MulBase, MulBaseUnreduced,
};
use akita_types::{
    checked_opening_source_index, gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup,
    FpExtEncoding, SetupProjectionGeometry, WitnessUnitLayout,
};

#[cfg(test)]
use akita_types::opening_domain_len;

pub(super) fn evaluate_mixed_relation_at_point<F, E>(
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
    if deferred_setup_claim.is_some() {
        return Err(AkitaError::InvalidProof);
    }
    let context = evaluator
        .flat_context
        .as_ref()
        .ok_or(AkitaError::InvalidProof)?;
    let prepared_point = PreparedRelationPoint::new(
        point,
        alpha,
        evaluator.role_dims,
        context.opening_ring_dim,
        context.opening_source_len,
    )?;
    let inner_ring_dimension = evaluator.role_dims.d_a();
    let coeff_count = prepared_point.coeff_count();
    let lanes_per_inner_column = prepared_point.inner().lane_powers.len();

    let mut constraint_evaluation = E::zero();
    for group in &evaluator.groups {
        let units = context.witness_layout.units_for_group(group.group_id)?;
        constraint_evaluation += evaluate_group_constraints::<F, E>(
            group,
            &units,
            context.opening_source_len,
            context.opening_ring_dim,
            inner_ring_dimension,
            coeff_count,
            prepared_point.address_point(),
            prepared_point.equality_window(),
            lanes_per_inner_column,
            &prepared_point.inner().lane_powers,
            evaluator.eq_tau1.as_ref(),
        )
        .map_err(|error| {
            AkitaError::InvalidInput(format!(
                "mixed relation group {} contraction failed: {error:?}",
                group.group_id
            ))
        })?;
    }

    let setup_evaluation = {
        let _span = tracing::info_span!("mixed_relation_setup_scan").entered();
        evaluate_setup_contribution::<F, E>(evaluator, setup, &prepared_point).map_err(|error| {
            AkitaError::InvalidInput(format!("mixed relation setup scan failed: {error:?}"))
        })?
    };
    let quotient_evaluation =
        evaluate_quotient_tail::<F, E>(evaluator, &prepared_point).map_err(|error| {
            AkitaError::InvalidInput(format!("mixed relation quotient failed: {error:?}"))
        })?;

    Ok(prepared_point.coeff_eval()
        * (constraint_evaluation + setup_evaluation + quotient_evaluation))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_group_constraints<F, E>(
    group: &RelationMatrixGroupEvaluator<E>,
    units: &[&WitnessUnitLayout],
    opening_source_len: usize,
    opening_ring_dimension: usize,
    inner_ring_dimension: usize,
    coeff_count: usize,
    lane_and_column_point: &[E],
    equality_window: &OffsetEqWindow<E>,
    lanes_per_inner_column: usize,
    inner_lane_alpha_powers: &[E],
    relation_row_weights: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FieldCore + LiftBase<F> + MulBase<F>,
{
    validate_log_basis(group.log_basis_inner)?;
    validate_log_basis(group.log_basis_outer)?;
    validate_log_basis(group.log_basis_open)?;
    if inner_lane_alpha_powers.len() != lanes_per_inner_column {
        return Err(AkitaError::InvalidProof);
    }
    let opening_gadget = gadget_row_scalars::<F>(group.depth_open, group.log_basis_open)
        .into_iter()
        .map(<E as LiftBase<F>>::lift_base)
        .collect::<Vec<_>>();
    let t_commitment_gadget = gadget_row_scalars::<F>(group.depth_commit, group.log_basis_outer);
    let witness_gadget = gadget_row_scalars::<F>(group.depth_witness, group.log_basis_inner);
    let consistency_weight = relation_row_weights
        .first()
        .copied()
        .ok_or(AkitaError::InvalidProof)?;
    let a_row_end = group
        .a_row_start
        .checked_add(group.n_a)
        .ok_or_else(|| AkitaError::InvalidSetup("A row range overflow".into()))?;
    let a_row_weights = relation_row_weights
        .get(group.a_row_start..a_row_end)
        .ok_or(AkitaError::InvalidProof)?;
    let claim_factors = (0..group.num_claims)
        .map(|claim| {
            group
                .c_alphas
                .affine_factors::<F>(claim, group.num_live_blocks)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let e_digit_lane_weights = opening_gadget
        .iter()
        .flat_map(|&digit_weight| {
            inner_lane_alpha_powers
                .iter()
                .map(move |&lane_weight| consistency_weight * digit_weight * lane_weight)
        })
        .collect::<Vec<_>>();
    let t_digit_lane_weights = a_row_weights
        .iter()
        .flat_map(|&row_weight| {
            t_commitment_gadget.iter().flat_map(move |&digit_weight| {
                inner_lane_alpha_powers.iter().map(move |&lane_weight| {
                    row_weight * <E as LiftBase<F>>::lift_base(digit_weight) * lane_weight
                })
            })
        })
        .collect::<Vec<_>>();
    let e_block_stride = group
        .depth_open
        .checked_mul(lanes_per_inner_column)
        .ok_or_else(|| AkitaError::InvalidSetup("E lane stride overflow".into()))?;
    let t_block_stride = group
        .n_a
        .checked_mul(group.depth_commit)
        .and_then(|stride| stride.checked_mul(lanes_per_inner_column))
        .ok_or_else(|| AkitaError::InvalidSetup("T lane stride overflow".into()))?;

    let mut evaluation = E::zero();
    for unit in units {
        for (claim, factors) in claim_factors.iter().enumerate() {
            let e_column = unit.e_index(
                group.num_claims,
                group.depth_open,
                claim,
                unit.global_block_start(),
                0,
            )?;
            let e_lane_start = canonical_relation_lane_index(
                opening_source_len,
                opening_ring_dimension,
                inner_ring_dimension,
                coeff_count,
                e_column,
                0,
            )?;
            evaluation += eval_affine_digit_interval(
                lane_and_column_point,
                e_lane_start,
                unit.global_block_start(),
                unit.num_live_blocks(),
                e_block_stride,
                &e_digit_lane_weights,
                &factors.high,
                &factors.low,
            )?;

            let t_column = unit.t_index(
                group.num_claims,
                group.n_a,
                group.depth_commit,
                claim,
                unit.global_block_start(),
                0,
                0,
            )?;
            let t_lane_start = canonical_relation_lane_index(
                opening_source_len,
                opening_ring_dimension,
                inner_ring_dimension,
                coeff_count,
                t_column,
                0,
            )?;
            evaluation += eval_affine_digit_interval(
                lane_and_column_point,
                t_lane_start,
                unit.global_block_start(),
                unit.num_live_blocks(),
                t_block_stride,
                &t_digit_lane_weights,
                &factors.high,
                &factors.low,
            )?;
        }
    }

    let fold_gadget = gadget_row_scalars::<F>(group.depth_fold, group.log_basis_open);
    for unit in units {
        for (position, &opening_evaluation) in group.opening_a_evals.iter().enumerate() {
            for (commit_digit, &commit_weight) in witness_gadget.iter().enumerate() {
                for (fold_digit, &fold_weight) in fold_gadget.iter().enumerate() {
                    let z_column = unit.z_index(
                        group.opening_a_evals.len(),
                        group.depth_witness,
                        group.depth_fold,
                        position,
                        commit_digit,
                        fold_digit,
                    )?;
                    let z_lane_start = canonical_relation_lane_index(
                        opening_source_len,
                        opening_ring_dimension,
                        inner_ring_dimension,
                        coeff_count,
                        z_column,
                        0,
                    )?;
                    let lane_equality = evaluate_lane_segment(
                        equality_window,
                        z_lane_start,
                        inner_lane_alpha_powers,
                    )?;
                    evaluation -= lane_equality
                        * consistency_weight
                        * opening_evaluation
                        * <E as LiftBase<F>>::lift_base(commit_weight)
                        * <E as LiftBase<F>>::lift_base(fold_weight);
                }
            }
        }
    }
    Ok(evaluation)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_setup_contribution<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    setup: &AkitaExpandedSetup<F>,
    prepared_point: &PreparedRelationPoint<'_, E>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    let context = evaluator
        .flat_context
        .as_ref()
        .ok_or(AkitaError::InvalidProof)?;
    let role_dims = evaluator.role_dims;
    let inner_ring_dimension = role_dims.d_a();
    let outer_ring_dimension = role_dims.d_b();
    let opening_ring_dimension = role_dims.d_d();
    let coeff_count = prepared_point.coeff_count();
    let equality_window = prepared_point.equality_window();
    let (outer_subcolumns, opening_subcolumns) =
        SetupProjectionGeometry::witness_subcolumn_ratios(role_dims)?;
    let outer_lanes = prepared_point.outer().lane_powers.len();
    let opening_lanes = prepared_point.opening().lane_powers.len();
    let inner_alpha_powers = prepared_point.inner().powers.as_ref();
    let outer_alpha_powers = prepared_point.outer().powers.as_ref();
    let opening_alpha_powers = prepared_point.opening().powers.as_ref();
    let inner_lane_alpha_powers = prepared_point.inner().lane_powers.as_ref();
    let outer_lane_alpha_powers = prepared_point.outer().lane_powers.as_ref();
    let opening_lane_alpha_powers = prepared_point.opening().lane_powers.as_ref();
    let rows = context.level_params.relation_matrix_row_count_for(
        context.opening_batch.num_groups(),
        context.relation_matrix_row_layout,
    )?;

    let active_d_rows = context
        .level_params
        .n_d_active_for(context.relation_matrix_row_layout);
    let d_row_start = rows
        .checked_sub(active_d_rows)
        .ok_or(AkitaError::InvalidProof)?;
    let d_row_weights = evaluator
        .eq_tau1
        .get(d_row_start..rows)
        .ok_or(AkitaError::InvalidProof)?;
    let mut d_column_weights = Vec::new();
    for group in &evaluator.groups {
        let units = context.witness_layout.units_for_group(group.group_id)?;
        let group_native_columns = group
            .num_claims
            .checked_mul(group.num_live_blocks)
            .and_then(|count| count.checked_mul(opening_subcolumns))
            .and_then(|count| count.checked_mul(group.depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("D column count overflow".into()))?;
        let group_start = d_column_weights.len();
        let group_end = group_start
            .checked_add(group_native_columns)
            .ok_or_else(|| AkitaError::InvalidSetup("D column range overflow".into()))?;
        d_column_weights.resize(group_end, E::zero());
        for unit in units {
            for claim in 0..group.num_claims {
                for local_block in 0..unit.num_live_blocks() {
                    let global_block = unit
                        .global_block_start()
                        .checked_add(local_block)
                        .ok_or_else(|| AkitaError::InvalidSetup("D block overflow".into()))?;
                    let logical_block = claim
                        .checked_mul(group.num_live_blocks)
                        .and_then(|base| base.checked_add(global_block))
                        .ok_or_else(|| AkitaError::InvalidSetup("D block index overflow".into()))?;
                    for opening_subcolumn in 0..opening_subcolumns {
                        for digit in 0..group.depth_open {
                            let witness_column = unit.e_index(
                                group.num_claims,
                                group.depth_open,
                                claim,
                                global_block,
                                digit,
                            )?;
                            let lane_start = canonical_relation_lane_index(
                                context.opening_source_len,
                                context.opening_ring_dim,
                                inner_ring_dimension,
                                coeff_count,
                                witness_column,
                                opening_subcolumn * opening_lanes,
                            )?;
                            let weight = evaluate_lane_segment(
                                equality_window,
                                lane_start,
                                opening_lane_alpha_powers,
                            )?;
                            let local_column = logical_block
                                .checked_mul(opening_subcolumns)
                                .and_then(|base| base.checked_add(opening_subcolumn))
                                .and_then(|base| base.checked_mul(group.depth_open))
                                .and_then(|base| base.checked_add(digit))
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup("D column index overflow".into())
                                })?;
                            let native_column =
                                group_start.checked_add(local_column).ok_or_else(|| {
                                    AkitaError::InvalidSetup("D native column overflow".into())
                                })?;
                            *d_column_weights
                                .get_mut(native_column)
                                .ok_or(AkitaError::InvalidProof)? = weight;
                        }
                    }
                }
            }
        }
    }
    let d_evaluation = if active_d_rows == 0 {
        E::zero()
    } else {
        evaluate_weighted_setup_matrix(
            setup,
            active_d_rows,
            &d_column_weights,
            opening_ring_dimension,
            d_row_weights,
            opening_alpha_powers,
        )?
    };

    let mut grouped_evaluation = E::zero();
    for group in &evaluator.groups {
        let units = context.witness_layout.units_for_group(group.group_id)?;
        let b_row_end = context
            .level_params
            .commitment_row_range(
                &context.opening_batch,
                group.group_id,
                context.relation_matrix_row_layout,
            )?
            .end;
        let b_row_weights = evaluator
            .eq_tau1
            .get(group.b_row_start..b_row_end)
            .ok_or(AkitaError::InvalidProof)?;
        if !b_row_weights.is_empty() {
            let semantic_t_columns = group
                .num_claims
                .checked_mul(group.num_live_blocks)
                .and_then(|count| count.checked_mul(group.n_a))
                .and_then(|count| count.checked_mul(group.depth_commit))
                .ok_or_else(|| AkitaError::InvalidSetup("B column count overflow".into()))?;
            let mut b_column_weights = vec![
                E::zero();
                semantic_t_columns
                    .checked_mul(outer_subcolumns)
                    .ok_or_else(|| AkitaError::InvalidSetup(
                        "B native column count overflow".into()
                    ))?
            ];
            for unit in &units {
                for claim in 0..group.num_claims {
                    for local_block in 0..unit.num_live_blocks() {
                        let global_block = unit
                            .global_block_start()
                            .checked_add(local_block)
                            .ok_or_else(|| AkitaError::InvalidSetup("B block overflow".into()))?;
                        let block_claim = claim
                            .checked_mul(group.num_live_blocks)
                            .and_then(|base| base.checked_add(global_block))
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("B block index overflow".into())
                            })?;
                        for a_row in 0..group.n_a {
                            for digit in 0..group.depth_commit {
                                let witness_column = unit.t_index(
                                    group.num_claims,
                                    group.n_a,
                                    group.depth_commit,
                                    claim,
                                    global_block,
                                    a_row,
                                    digit,
                                )?;
                                let semantic_column = block_claim
                                    .checked_mul(group.n_a)
                                    .and_then(|base| base.checked_add(a_row))
                                    .and_then(|base| base.checked_mul(group.depth_commit))
                                    .and_then(|base| base.checked_add(digit))
                                    .ok_or_else(|| {
                                        AkitaError::InvalidSetup("B column index overflow".into())
                                    })?;
                                for outer_subcolumn in 0..outer_subcolumns {
                                    let first_lane = outer_subcolumn * outer_lanes;
                                    let lane_start = canonical_relation_lane_index(
                                        context.opening_source_len,
                                        context.opening_ring_dim,
                                        inner_ring_dimension,
                                        coeff_count,
                                        witness_column,
                                        first_lane,
                                    )?;
                                    let native_column = semantic_column
                                        .checked_mul(outer_subcolumns)
                                        .and_then(|base| base.checked_add(outer_subcolumn))
                                        .ok_or_else(|| {
                                            AkitaError::InvalidSetup(
                                                "B native column index overflow".into(),
                                            )
                                        })?;
                                    *b_column_weights
                                        .get_mut(native_column)
                                        .ok_or(AkitaError::InvalidProof)? = evaluate_lane_segment(
                                        equality_window,
                                        lane_start,
                                        outer_lane_alpha_powers,
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
            grouped_evaluation += evaluate_weighted_setup_matrix(
                setup,
                b_row_weights.len(),
                &b_column_weights,
                outer_ring_dimension,
                b_row_weights,
                outer_alpha_powers,
            )?;
        }

        let a_row_end = group
            .a_row_start
            .checked_add(group.n_a)
            .ok_or_else(|| AkitaError::InvalidSetup("A row range overflow".into()))?;
        let a_row_weights = evaluator
            .eq_tau1
            .get(group.a_row_start..a_row_end)
            .ok_or(AkitaError::InvalidProof)?;
        let group_params = context
            .level_params
            .group_params(&context.opening_batch, group.group_id)?;
        let active_a_columns = group
            .opening_a_evals
            .len()
            .checked_mul(group.depth_witness)
            .ok_or_else(|| AkitaError::InvalidSetup("A column count overflow".into()))?;
        let a_columns = group_params.a_col_len();
        if active_a_columns > a_columns {
            return Err(AkitaError::InvalidProof);
        }
        let mut a_column_weights = vec![E::zero(); a_columns];
        let fold_gadget = gadget_row_scalars::<F>(group.depth_fold, group.log_basis_open);
        for unit in units {
            for position in 0..group.opening_a_evals.len() {
                for commit_digit in 0..group.depth_witness {
                    let a_column = position
                        .checked_mul(group.depth_witness)
                        .and_then(|base| base.checked_add(commit_digit))
                        .ok_or_else(|| AkitaError::InvalidSetup("A column overflow".into()))?;
                    for (fold_digit, &fold_weight) in fold_gadget.iter().enumerate() {
                        let witness_column = unit.z_index(
                            group.opening_a_evals.len(),
                            group.depth_witness,
                            group.depth_fold,
                            position,
                            commit_digit,
                            fold_digit,
                        )?;
                        let lane_start = canonical_relation_lane_index(
                            context.opening_source_len,
                            context.opening_ring_dim,
                            inner_ring_dimension,
                            coeff_count,
                            witness_column,
                            0,
                        )?;
                        let column_weight = a_column_weights
                            .get_mut(a_column)
                            .ok_or(AkitaError::InvalidProof)?;
                        *column_weight -= E::lift_base(fold_weight)
                            * evaluate_lane_segment(
                                equality_window,
                                lane_start,
                                inner_lane_alpha_powers,
                            )?;
                    }
                }
            }
        }
        grouped_evaluation += evaluate_weighted_setup_matrix(
            setup,
            group.n_a,
            &a_column_weights,
            inner_ring_dimension,
            a_row_weights,
            inner_alpha_powers,
        )?;
    }
    Ok(d_evaluation + grouped_evaluation)
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
    let rows = context.level_params.relation_matrix_row_count_for(
        context.opening_batch.num_groups(),
        context.relation_matrix_row_layout,
    )?;
    let levels = r_decomp_levels::<F>(evaluator.log_basis);
    let quotient_gadget = gadget_row_scalars::<F>(levels, evaluator.log_basis);
    let d_row_start = rows
        .checked_sub(
            context
                .level_params
                .n_d_active_for(context.relation_matrix_row_layout),
        )
        .ok_or(AkitaError::InvalidProof)?;
    let b_row_ranges = (0..context.opening_batch.num_groups())
        .map(|group| {
            context.level_params.commitment_row_range(
                &context.opening_batch,
                group,
                context.relation_matrix_row_layout,
            )
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

fn evaluate_weighted_setup_matrix<F, E>(
    setup: &AkitaExpandedSetup<F>,
    row_count: usize,
    column_weights: &[E],
    ring_dimension: usize,
    row_weights: &[E],
    alpha_powers: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    if row_weights.len() != row_count || alpha_powers.len() != ring_dimension {
        return Err(AkitaError::InvalidProof);
    }
    let view =
        setup
            .shared_matrix
            .ring_view_dyn(row_count, column_weights.len(), ring_dimension)?;
    let rows = (0..row_count)
        .map(|row| view.row_flat(row))
        .collect::<Result<Vec<_>, _>>()?;
    cfg_fold_reduce!(
        0..row_count,
        || Ok(E::zero()),
        |acc: Result<E, AkitaError>, row| {
            let coefficients = *rows.get(row).ok_or(AkitaError::InvalidProof)?;
            let row_weight = *row_weights.get(row).ok_or(AkitaError::InvalidProof)?;
            let mut row_evaluation = E::zero();
            for (column, &column_weight) in column_weights.iter().enumerate() {
                if column_weight.is_zero() {
                    continue;
                }
                let start = column
                    .checked_mul(ring_dimension)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup column overflow".into()))?;
                let end = start
                    .checked_add(ring_dimension)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup column overflow".into()))?;
                let ring = coefficients
                    .get(start..end)
                    .ok_or(AkitaError::InvalidProof)?;
                row_evaluation +=
                    column_weight * eval_flat_ring_at_pows_fast::<F, E>(ring, alpha_powers);
            }
            Ok(acc? + row_weight * row_evaluation)
        },
        |left: Result<E, AkitaError>, right: Result<E, AkitaError>| Ok(left? + right?)
    )
}
