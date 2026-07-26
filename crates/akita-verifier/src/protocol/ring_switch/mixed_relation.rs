//! Succinct lane-factored relation evaluation.
//!
//! This path is used when the flat relation point has more than one coefficient
//! lane per witness column. It factors the common low alpha coordinates once,
//! contracts E/T intervals, and scans setup matrices in their native role
//! rings. It never constructs prover relation events or a dense relation table.

use super::{prepared_relation_point::PreparedRelationPoint, RelationMatrixEvaluator};
use akita_algebra::offset_eq::OffsetEqWindow;
#[cfg(test)]
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
#[cfg(test)]
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, MulBaseUnreduced,
};
#[cfg(test)]
use akita_types::SetupProjectionGeometry;
use akita_types::{
    checked_opening_source_index, gadget_row_scalars, r_decomp_levels, relation_rhs_layout_for,
    AkitaExpandedSetup, FpExtEncoding,
};

pub(super) fn evaluate_lane_factored_relation_at_point<F, E>(
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
    // The setup projection still uses the common base of the commitment roles,
    // while the checked plan spans address the potentially finer coefficient
    // block shared with the outgoing witness. The same plan therefore owns the
    // mixed E/T/Z contraction, direct setup scan, and deferred Stage-3 geometry.
    let fold_gadget = evaluator.setup_contribution_fold_gadget::<F>()?;
    let plan = {
        let _span = tracing::info_span!("mixed_relation_setup_plan").entered();
        let fold_gadget = fold_gadget.as_deref().unwrap_or(&[]);
        if deferred_setup_claim.is_some() {
            evaluator.setup_contribution_plan_deferred::<F>(
                prepared_point.address_point(),
                (!fold_gadget.is_empty()).then_some(fold_gadget),
                alpha,
            )?
        } else {
            evaluator.setup_contribution_plan::<F>(
                prepared_point.address_point(),
                (!fold_gadget.is_empty()).then_some(fold_gadget),
                alpha,
            )?
        }
    };

    let mut structured_evaluation = E::zero();
    {
        let _span = tracing::info_span!("mixed_relation_structured_groups").entered();
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
                        "mixed relation group {} contraction failed: {error:?}",
                        group.group_id
                    ))
                })?;
        }
    }

    let setup_evaluation = if let Some(claim) = deferred_setup_claim {
        claim
    } else {
        let _span =
            tracing::info_span!("mixed_relation_setup_scan", required = plan.required()).entered();
        plan.evaluate_direct::<F>(
            setup,
            prepared_point.inner().powers.as_ref(),
            prepared_point.outer().powers.as_ref(),
            prepared_point.opening().powers.as_ref(),
        )?
    };
    let quotient_evaluation =
        evaluate_quotient_tail::<F, E>(evaluator, &prepared_point).map_err(|error| {
            AkitaError::InvalidInput(format!("mixed relation quotient failed: {error:?}"))
        })?;

    let relation_evaluation = structured_evaluation + setup_evaluation + quotient_evaluation;
    evaluator.cache_setup_contribution_plan(prepared_point.address_point(), plan)?;
    Ok(prepared_point.common_alpha_evaluation() * relation_evaluation)
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
#[allow(dead_code)]
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
    let coeff_count = prepared_point.common_relation_witness_coeff_count();
    let equality_window = prepared_point.equality_window();
    let (outer_subcolumns, opening_subcolumns) =
        SetupProjectionGeometry::a_carrier_subcolumn_counts(role_dims)?;
    let outer_lanes = prepared_point.outer().lane_powers.len();
    let opening_lanes = prepared_point.opening().lane_powers.len();
    let inner_alpha_powers = prepared_point.inner().powers.as_ref();
    let outer_alpha_powers = prepared_point.outer().powers.as_ref();
    let opening_alpha_powers = prepared_point.opening().powers.as_ref();
    let inner_lane_alpha_powers = prepared_point.inner().lane_powers.as_ref();
    let outer_lane_alpha_powers = prepared_point.outer().lane_powers.as_ref();
    let opening_lane_alpha_powers = prepared_point.opening().lane_powers.as_ref();
    let rows = context
        .level_params
        .relation_matrix_row_count(context.opening_batch.num_groups())?;

    let active_d_rows = context.level_params.open_commit_matrix.output_rank();
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
        let num_claims = group.num_claims;
        let num_live_blocks = group.num_live_blocks;
        let depth_open = group.depth_open;
        let group_native_columns = num_claims
            .checked_mul(num_live_blocks)
            .and_then(|count| count.checked_mul(opening_subcolumns))
            .and_then(|count| count.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("D column count overflow".into()))?;
        // Build the D-role column weights in parallel over the native column
        // index (same bijection as the serial form; uncovered blocks stay zero).
        let group_weights = cfg_into_iter!(0..group_native_columns)
            .map(|local_column| -> Result<E, AkitaError> {
                let digit = local_column % depth_open;
                let t = local_column / depth_open;
                let opening_subcolumn = t % opening_subcolumns;
                let logical_block = t / opening_subcolumns;
                let claim = logical_block / num_live_blocks;
                let global_block = logical_block % num_live_blocks;
                let Some(unit) = units.iter().copied().find(|u| {
                    let start = u.global_block_start();
                    global_block >= start && global_block - start < u.num_live_blocks()
                }) else {
                    return Ok(E::zero());
                };
                let witness_column =
                    unit.e_index(num_claims, depth_open, claim, global_block, digit)?;
                let lane_start = canonical_relation_lane_index(
                    context.opening_source_len,
                    context.opening_ring_dim,
                    inner_ring_dimension,
                    coeff_count,
                    witness_column,
                    opening_subcolumn * opening_lanes,
                )?;
                evaluate_lane_segment(equality_window, lane_start, opening_lane_alpha_powers)
            })
            .collect::<Result<Vec<E>, AkitaError>>()?;
        d_column_weights.extend(group_weights);
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
            .commitment_row_range(&context.opening_batch, group.group_id)?
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
            let b_native_columns = semantic_t_columns
                .checked_mul(outer_subcolumns)
                .ok_or_else(|| AkitaError::InvalidSetup("B native column count overflow".into()))?;
            // Build the B-role column weights in parallel over the native column
            // index. Each native column is written exactly once in the serial
            // form, so inverting `native_column → (claim, block, a_row, digit,
            // outer_subcolumn)` and mapping independently produces the identical
            // vector. Blocks not covered by any unit stay zero (as before).
            let num_claims = group.num_claims;
            let num_live_blocks = group.num_live_blocks;
            let n_a = group.n_a;
            let depth_commit = group.depth_commit;
            let b_column_weights = cfg_into_iter!(0..b_native_columns)
                .map(|native_column| -> Result<E, AkitaError> {
                    let outer_subcolumn = native_column % outer_subcolumns;
                    let semantic_column = native_column / outer_subcolumns;
                    let digit = semantic_column % depth_commit;
                    let rest = semantic_column / depth_commit;
                    let a_row = rest % n_a;
                    let block_claim = rest / n_a;
                    let claim = block_claim / num_live_blocks;
                    let global_block = block_claim % num_live_blocks;
                    let Some(unit) = units.iter().copied().find(|u| {
                        let start = u.global_block_start();
                        global_block >= start && global_block - start < u.num_live_blocks()
                    }) else {
                        return Ok(E::zero());
                    };
                    let witness_column = unit.t_index(
                        num_claims,
                        n_a,
                        depth_commit,
                        claim,
                        global_block,
                        a_row,
                        digit,
                    )?;
                    let lane_start = canonical_relation_lane_index(
                        context.opening_source_len,
                        context.opening_ring_dim,
                        inner_ring_dimension,
                        coeff_count,
                        witness_column,
                        outer_subcolumn * outer_lanes,
                    )?;
                    evaluate_lane_segment(equality_window, lane_start, outer_lane_alpha_powers)
                })
                .collect::<Result<Vec<E>, AkitaError>>()?;
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
        // Build the A-role column weights in parallel over the (position,
        // commit_digit) column index. Each column accumulates independently
        // over units and fold digits; the per-column sum is identical to the
        // serial `-=` accumulation (associative field addition). Columns
        // `[active_a_columns, a_columns)` stay zero (padding), matching the
        // original.
        let depth_witness = group.depth_witness;
        let num_positions = group.opening_a_evals.len();
        let fold_gadget = gadget_row_scalars::<F>(group.depth_fold, group.log_basis_open);
        let mut a_column_weights = cfg_into_iter!(0..active_a_columns)
            .map(|a_column| -> Result<E, AkitaError> {
                let position = a_column / depth_witness;
                let commit_digit = a_column % depth_witness;
                let mut column_weight = E::zero();
                for &unit in &units {
                    for (fold_digit, &fold_weight) in fold_gadget.iter().enumerate() {
                        let witness_column = unit.z_index(
                            num_positions,
                            depth_witness,
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
                        column_weight -= E::lift_base(fold_weight)
                            * evaluate_lane_segment(
                                equality_window,
                                lane_start,
                                inner_lane_alpha_powers,
                            )?;
                    }
                }
                Ok(column_weight)
            })
            .collect::<Result<Vec<E>, AkitaError>>()?;
        a_column_weights.resize(a_columns, E::zero());
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
                context.opening_source_len,
                context.opening_ring_dim,
                evaluator.relation_address_geometry.carrier_ring_dimension(),
                prepared_point.common_relation_witness_coeff_count(),
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

#[cfg(test)]
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
    let matrix = setup.shared_matrix.covering_at_dyn(
        row_count
            .checked_mul(column_weights.len())
            .ok_or(AkitaError::InvalidProof)?,
        ring_dimension,
    )?;
    let view = matrix.ring_view_dyn(row_count, column_weights.len(), ring_dimension)?;
    let rows = (0..row_count)
        .map(|row| view.row_flat(row))
        .collect::<Result<Vec<_>, _>>()?;
    // Parallelize over the **column** axis rather than the row axis. The
    // full sum `Σ_row Σ_col row_weight·col_weight·⟨ring[row][col], α⟩` is
    // associative/commutative, so summing column-major is numerically
    // identical, but non-uniform levels (e.g. the D role with `n_d = 1`) have
    // only a handful of rows and hundreds of thousands of columns — folding
    // over rows leaves that inner loop serial. This is a mixed-path-only
    // helper, so uniform schedules are unaffected and the value is unchanged.
    cfg_fold_reduce!(
        0..column_weights.len(),
        || Ok(E::zero()),
        |acc: Result<E, AkitaError>, column| {
            let column_weight = *column_weights.get(column).ok_or(AkitaError::InvalidProof)?;
            if column_weight.is_zero() {
                return acc;
            }
            let start = column
                .checked_mul(ring_dimension)
                .ok_or_else(|| AkitaError::InvalidSetup("setup column overflow".into()))?;
            let end = start
                .checked_add(ring_dimension)
                .ok_or_else(|| AkitaError::InvalidSetup("setup column overflow".into()))?;
            let mut column_evaluation = E::zero();
            for row in 0..row_count {
                let coefficients = *rows.get(row).ok_or(AkitaError::InvalidProof)?;
                let row_weight = *row_weights.get(row).ok_or(AkitaError::InvalidProof)?;
                let ring = coefficients
                    .get(start..end)
                    .ok_or(AkitaError::InvalidProof)?;
                column_evaluation +=
                    row_weight * eval_flat_ring_at_pows_fast::<F, E>(ring, alpha_powers);
            }
            Ok(acc? + column_weight * column_evaluation)
        },
        |left: Result<E, AkitaError>, right: Result<E, AkitaError>| Ok(left? + right?)
    )
}
