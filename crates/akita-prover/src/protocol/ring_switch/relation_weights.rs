//! Semantic relation-weight events and their canonical consumers.

#[path = "relation_weights/setup_columns.rs"]
mod setup_columns;

use std::ops::Range;

use akita_algebra::eq_poly::SplitEqEvals;
use akita_algebra::ring::{eval_flat_ring_at_pows_fast, scalar_powers};
use akita_error::AkitaError;
use akita_types::{
    gadget_row_scalars, prepare_coefficient_packing_batch_semantics, r_decomp_levels,
    AkitaExpandedSetup, CoefficientPackingBatchSemanticInputs, CoefficientPackingBatchSemantics,
    CommittedGroupParams, FpExtEncoding, OpeningClaimsLayout, OpeningFamily, OpeningMethod,
    PreparedSubringCoefficientPackingPoint, RelationAddressGeometry, RelationRangeImagePlan,
    RelationRowFamily, RelationWitnessGeometry, RingRelationInstance, SetupProjectionGeometry,
};
pub use akita_types::{RelationWeightContribution, RelationWeightEvent};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use setup_columns::{evaluate_setup_columns, SetupRows};

/// Source of setup-matrix relation weights for this evaluation.
#[derive(Clone, Copy)]
pub enum RelationSetupSource<'a, F: Field> {
    /// Emit setup events directly from the expanded setup matrix.
    Matrix(&'a AkitaExpandedSetup<F>),
    /// Omit setup events because their complete evaluation is supplied separately.
    DeferredClaim,
}

/// Inputs to the one semantic relation-event builder.
pub struct RelationWeightEventInputs<'a, F: Field, E: Field> {
    pub setup: RelationSetupSource<'a, F>,
    pub instance: &'a RingRelationInstance<F>,
    pub alpha: E,
    pub level_params: &'a CommittedGroupParams,
    pub relation_row_point: &'a [E],
    pub claim_coefficients: &'a [E],
    pub opening_source_len: usize,
    pub opening_ring_dim: usize,
    pub relation_plan: &'a RelationRangeImagePlan,
    /// Method-typed prepared points for the current fold.
    pub opening_points:
        OpeningFamily<(), &'a [(usize, &'a PreparedSubringCoefficientPackingPoint<E>)]>,
}

mod events;
pub use events::{RelationWeightEvents, RelationWeightFactorization};

fn relation_d_group_width(
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    relation_geometry: &RelationWitnessGeometry,
    group_index: usize,
) -> Result<usize, AkitaError> {
    let group_lp = lp.group_params_geometry(opening_batch, group_index)?;
    let group_dims = lp.group_role_dims_geometry(opening_batch, group_index)?;
    let opening_width = relation_geometry
        .group_opening_geometry(group_index)?
        .physical_coefficient_width();
    let d_subcolumns = opening_width
        .checked_div(group_dims.d_d())
        .filter(|count| *count > 0 && opening_width.is_multiple_of(group_dims.d_d()))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("opening width does not factor the D role".into())
        })?;
    let num_claims = opening_batch.group_layout(group_index)?.num_polynomials();
    num_claims
        .checked_mul(group_lp.num_live_blocks())
        .and_then(|n| n.checked_mul(group_lp.num_digits_open()))
        .and_then(|n| n.checked_mul(d_subcolumns))
        .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".to_string()))
}

fn relation_d_column_ranges(
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    relation_geometry: &RelationWitnessGeometry,
) -> Result<Vec<Range<usize>>, AkitaError> {
    let mut cursor = 0usize;
    let mut seen = vec![false; opening_batch.num_groups()];
    let mut ranges = vec![0..0; opening_batch.num_groups()];
    for group_id in opening_batch.root_group_order()? {
        let slot = seen
            .get_mut(group_id)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D group id out of range".into()))?;
        if std::mem::replace(slot, true) {
            return Err(AkitaError::InvalidSetup(
                "setup D group id appears more than once".into(),
            ));
        }
        let width = relation_d_group_width(lp, opening_batch, relation_geometry, group_id)?;
        let end = cursor
            .checked_add(width)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))?;
        ranges[group_id] = cursor..end;
        cursor = end;
    }
    if seen.iter().any(|present| !present) {
        return Err(AkitaError::InvalidSetup(
            "setup D group ids are not contiguous".into(),
        ));
    }
    Ok(ranges)
}

fn matching_row_range(
    row_families: &[RelationRowFamily],
    mut matches: impl FnMut(&RelationRowFamily) -> bool,
) -> Result<Range<usize>, AkitaError> {
    let mut matched = row_families
        .iter()
        .enumerate()
        .filter_map(|(row, family)| matches(family).then_some(row));
    let start = matched.next().ok_or(AkitaError::InvalidProof)?;
    let mut end = start + 1;
    for row in matched {
        if row != end {
            return Err(AkitaError::InvalidSetup(
                "relation row family is not contiguous".into(),
            ));
        }
        end += 1;
    }
    Ok(start..end)
}

/// Emit the complete checked relation semantics for one fold.
pub(super) type RelationWeightBuild<E> = (
    RelationWeightEvents<E>,
    OpeningFamily<(), CoefficientPackingBatchSemantics<E>>,
);

#[tracing::instrument(skip_all, name = "build_relation_weight_events")]
pub fn build_relation_weight_events<F, E>(
    inputs: RelationWeightEventInputs<'_, F, E>,
) -> Result<RelationWeightBuild<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
{
    let RelationWeightEventInputs {
        setup,
        instance,
        alpha,
        level_params: lp,
        relation_row_point: tau1,
        claim_coefficients: gamma,
        opening_source_len,
        opening_ring_dim,
        relation_plan,
        opening_points,
    } = inputs;
    let opening_batch = instance.opening_batch();
    lp.witness_chunk.validate()?;
    if gamma.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }
    let role_dims = instance.role_dims();
    if role_dims != lp.role_dims() {
        return Err(AkitaError::InvalidSetup(
            "relation instance and level role dimensions disagree".into(),
        ));
    }
    let d_a = role_dims.d_a();
    let d_b = role_dims.d_b();
    let d_d = role_dims.d_d();
    let alpha_pows_b = scalar_powers(alpha, d_b);
    let alpha_pows_d = scalar_powers(alpha, d_d);
    let relation_geometry =
        RelationWitnessGeometry::for_level(lp, opening_batch, instance.extension_degree())?;
    let packing_required = matches!(
        relation_geometry.group_opening_method(0)?,
        OpeningMethod::SubringCoefficientPacking { .. }
    );
    if packing_required != matches!(opening_points, OpeningFamily::SubringCoefficientPacking(_)) {
        return Err(AkitaError::InvalidSetup(
            "relation opening family disagrees with prepared points".into(),
        ));
    }
    let relation_rhs_layout = relation_geometry.rhs_layout();
    let row_families = relation_rhs_layout.row_families()?;
    let quotient_row_dims = row_families
        .iter()
        .map(|row| row.geometry().polynomial_modulus_dimension())
        .collect::<Vec<_>>();
    let rows = quotient_row_dims.len();
    if rows == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let mut additional_quotient_alpha_powers = Vec::new();
    for &row_dim in &quotient_row_dims {
        if row_dim != d_a
            && row_dim != d_b
            && row_dim != d_d
            && additional_quotient_alpha_powers
                .iter()
                .all(|(dimension, _): &(usize, Vec<E>)| *dimension != row_dim)
        {
            additional_quotient_alpha_powers.push((row_dim, scalar_powers(alpha, row_dim)));
        }
    }
    let eq_tau1 = SplitEqEvals::new(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }
    let n_d_active = lp.open().matrix.output_rank();
    let levels = r_decomp_levels::<F>(lp.open().digits.log_basis);
    let witness_layout = instance.segment_layout(lp, None)?;
    if witness_layout.r_rows().len() != rows || witness_layout.quotient_depth() != levels {
        return Err(AkitaError::InvalidSetup(
            "relation matrix dimensions disagree with witness layout".to_string(),
        ));
    }
    for (row, family) in witness_layout.r_rows().iter().zip(&row_families) {
        if row.geometry() != family.geometry() {
            return Err(AkitaError::InvalidSetup(
                "relation quotient dimensions disagree with witness layout".into(),
            ));
        }
    }
    let live_witness_coeff_len = witness_layout.live_coeff_len();
    let physical_field_len = opening_source_len
        .checked_mul(opening_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("opening field length overflow".into()))?;
    if live_witness_coeff_len > physical_field_len {
        return Err(AkitaError::InvalidSize {
            expected: physical_field_len,
            actual: live_witness_coeff_len,
        });
    }
    let setup_matrix = match setup {
        RelationSetupSource::Matrix(setup) => Some(setup),
        RelationSetupSource::DeferredClaim => None,
    };
    let setup_is_deferred = setup_matrix.is_none();
    let d_column_ranges = if setup_matrix.is_some() {
        relation_d_column_ranges(lp, opening_batch, &relation_geometry)?
    } else {
        Vec::new()
    };
    let relation_coefficient_block_len = RelationAddressGeometry::for_relation(
        &relation_geometry,
        opening_ring_dim,
        live_witness_coeff_len,
    )?
    .relation_coefficient_block_len();
    if relation_plan.relation_witness_geometry() != &relation_geometry
        || relation_plan.witness_layout() != &witness_layout
        || relation_plan
            .relation_address_geometry()
            .relation_coefficient_block_len()
            != relation_coefficient_block_len
    {
        return Err(AkitaError::InvalidSetup(
            "relation plan disagrees with the current ring switch".into(),
        ));
    }
    let (coefficient_packing_events, opening_semantics) = match opening_points {
        OpeningFamily::SubringCoefficientPacking(prepared_points) => {
            let (events, batch) = prepare_coefficient_packing_batch_semantics(
                CoefficientPackingBatchSemanticInputs {
                    level_params: lp,
                    opening_batch,
                    relation_plan,
                    relation: instance,
                    prepared_points,
                    alpha,
                    tau1,
                    claim_coefficients: gamma,
                },
            )?;
            (events, OpeningFamily::SubringCoefficientPacking(batch))
        }
        OpeningFamily::EvaluationTrace(()) => (Vec::new(), OpeningFamily::EvaluationTrace(())),
    };
    let coefficient_packing_groups = match &opening_semantics {
        OpeningFamily::EvaluationTrace(()) => &[][..],
        OpeningFamily::SubringCoefficientPacking(batch) => batch.groups(),
    };
    let mut relation_events = RelationWeightEvents {
        events: Vec::new(),
        alpha_powers: scalar_powers(
            alpha,
            quotient_row_dims
                .iter()
                .copied()
                .max()
                .ok_or(AkitaError::InvalidProof)?,
        ),
        relation_coefficient_block_len,
        physical_field_len,
        setup_is_deferred,
    };
    let mut packing_semantics_by_group = vec![None; opening_batch.num_groups()];
    for semantics in coefficient_packing_groups {
        let group_index = semantics.group_index();
        let slot = packing_semantics_by_group
            .get_mut(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if slot.replace(semantics).is_some() {
            return Err(AkitaError::InvalidSetup(
                "packing relation group appears more than once".into(),
            ));
        }
        if semantics.stage2_terms().physical_field_len() != live_witness_coeff_len {
            return Err(AkitaError::InvalidSetup(
                "packing relation live domain disagrees with the current ring switch".into(),
            ));
        }
        if semantics.stage2_terms().relation_coefficient_block_len()
            != relation_coefficient_block_len
        {
            return Err(AkitaError::InvalidSetup(
                "packing relation coefficient block disagrees with the current ring switch".into(),
            ));
        }
    }
    relation_events.extend_events(coefficient_packing_events)?;
    let d_view = if let Some(setup) = setup_matrix {
        let d_physical_columns = d_column_ranges
            .iter()
            .map(|range| range.end)
            .max()
            .unwrap_or(0);
        let rank = lp.open().matrix.output_rank();
        Some((&setup.shared_matrix, rank, d_physical_columns))
    } else {
        None
    };
    let d_family = match &d_view {
        Some((matrix, rows, cols)) => {
            let view = matrix.ring_view_dyn(*rows, *cols, d_d)?;
            Some(SetupRows {
                rows: (0..*rows)
                    .map(|row| view.row_flat(row))
                    .collect::<Result<Vec<_>, _>>()?,
                ring_d: d_d,
            })
        }
        None => None,
    };
    let d_start = row_families
        .iter()
        .position(|row| matches!(row, akita_types::RelationRowFamily::Opening { .. }))
        .ok_or(AkitaError::InvalidProof)?;
    for (group_index, &packing_semantics) in packing_semantics_by_group.iter().enumerate() {
        let e_setup_offset = if setup_matrix.is_some() {
            d_column_ranges
                .get(group_index)
                .ok_or(AkitaError::InvalidProof)?
                .start
        } else {
            0
        };
        let group_lp = lp.group_params_geometry(opening_batch, group_index)?;
        let group_dims = lp.group_role_dims_geometry(opening_batch, group_index)?;
        let group_d_a = group_dims.d_a();
        let group_d_b = group_dims.d_b();
        let group_d_d = group_dims.d_d();
        let (b_ratio, _) = SetupProjectionGeometry::native_role_subcolumn_counts(group_dims)?;
        let opening_width = relation_geometry
            .group_opening_geometry(group_index)?
            .physical_coefficient_width();
        let d_ratio = opening_width
            .checked_div(group_d_d)
            .filter(|count| *count > 0 && opening_width.is_multiple_of(group_d_d))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("opening width does not factor the D role".into())
            })?;
        let group_alpha_pows_a = scalar_powers(alpha, group_d_a);
        let group_alpha_pows_b = scalar_powers(alpha, group_d_b);
        let group_alpha_pows_d = scalar_powers(alpha, group_d_d);
        let group_layout = opening_batch.group_layout(group_index)?;
        let group_id = group_index;
        let units = witness_layout.units_for_group(group_id)?;
        let k_g = group_layout.num_polynomials();
        let opening_method = relation_geometry.group_opening_method(group_index)?;
        match (opening_method, packing_semantics) {
            (OpeningMethod::EvaluationTrace, None) => {}
            (OpeningMethod::SubringCoefficientPacking { .. }, Some(semantics))
                if semantics.geometry().a_ring_dimension() == group_d_a => {}
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "packing semantic groups do not match scheduled opening methods".into(),
                ));
            }
        }
        let ring_multiplier_point = matches!(opening_method, OpeningMethod::EvaluationTrace)
            .then(|| instance.group_ring_multiplier_point(group_index))
            .transpose()?;
        let challenges = instance.group_ambient_a_challenges(group_index)?;
        if ring_multiplier_point.is_some_and(|point| {
            point.position_len() != group_lp.num_positions_per_block()
                || point.fold_len() != group_lp.num_live_blocks()
        }) {
            return Err(AkitaError::InvalidInput(
                "relation matrix col eval multiplier layout mismatch".to_string(),
            ));
        }
        let total_blocks = k_g
            .checked_mul(group_lp.num_live_blocks())
            .ok_or(AkitaError::InvalidProof)?;
        if challenges.len() != total_blocks {
            return Err(AkitaError::InvalidProof);
        }
        let depth_witness = group_lp.num_digits_inner();
        let depth_commit = group_lp.num_digits_outer();
        let depth_open = group_lp.num_digits_open();
        let depth_fold = group_lp.num_digits_fold();
        let log_basis_inner = group_lp.log_basis_inner();
        let log_basis_outer = group_lp.log_basis_outer();
        let log_basis_open = group_lp.log_basis_open();
        let n_a = group_lp.a_rows_len();
        let physical_n_b = group_lp.b_rows_len();
        let n_b = group_lp.logical_b_rows_len()?;
        let inner_width = group_lp.a_col_len();
        // Hoist per-group geometry into `Copy` locals so the parallel closures
        // below capture scalars instead of the `!Sync` `&dyn LevelParamsLike`.
        let num_live_blocks_g = group_lp.num_live_blocks();
        let num_positions_per_block_g = group_lp.num_positions_per_block();
        let slice_geometry = akita_types::CommitmentSliceGeometry::try_new(
            group_lp.outer_slice_count(),
            num_live_blocks_g,
            k_g,
            n_a,
            depth_commit,
            group_d_a,
            group_d_b,
        )?;
        let b_width = slice_geometry.physical_input_width();
        let (setup_a_family, b_family) = if let Some(setup) = setup_matrix {
            let a_view = setup
                .shared_matrix
                .ring_view_dyn(n_a, inner_width, group_d_a)?;
            let a_family = SetupRows {
                rows: (0..n_a)
                    .map(|row| a_view.row_flat(row))
                    .collect::<Result<Vec<_>, _>>()?,
                ring_d: group_d_a,
            };
            let b_view = setup
                .shared_matrix
                .ring_view_dyn(physical_n_b, b_width, group_d_b)?;
            let b_family = SetupRows {
                rows: (0..physical_n_b)
                    .map(|row| b_view.row_flat(row))
                    .collect::<Result<Vec<_>, _>>()?,
                ring_d: group_d_b,
            };
            (Some(a_family), Some(b_family))
        } else {
            (None, None)
        };
        let a_range = matching_row_range(
            &row_families,
            |family| matches!(family, RelationRowFamily::Inner { group_index: group, .. } if *group == group_index),
        )?;
        let b_range = matching_row_range(
            &row_families,
            |family| matches!(family, RelationRowFamily::Outer { group_index: group, .. } if *group == group_index),
        )?;
        let consistency_row = row_families
            .iter()
            .position(|family| {
                matches!(family, RelationRowFamily::Consistency { group_index: group, .. } if *group == group_index)
            })
            .ok_or(AkitaError::InvalidProof)?;
        let consistency_weight = eq_tau1.eval_at(consistency_row)?;
        if a_range.end > eq_tau1.len() || b_range.end > eq_tau1.len() || b_range.len() != n_b {
            return Err(AkitaError::InvalidProof);
        }
        let g_open: Vec<E> = gadget_row_scalars::<F>(depth_open, log_basis_open)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let t_commit_gadget: Vec<E> = gadget_row_scalars::<F>(depth_commit, log_basis_outer)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let witness_gadget: Vec<E> = gadget_row_scalars::<F>(depth_witness, log_basis_inner)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let fold_gadget: Vec<E> = gadget_row_scalars::<F>(depth_fold, log_basis_open)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let d_setup_start = e_setup_offset;
        let d_setup_len = total_blocks
            .checked_mul(d_ratio)
            .and_then(|len| len.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".to_string()))?;
        let d_setup_end = d_setup_start
            .checked_add(d_setup_len)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D extent overflow".to_string()))?;
        let d_setup_accs = if let Some(d_family) = &d_family {
            let _span = tracing::info_span!("relation_weight_d_setup_columns").entered();
            let row_weights = (0..n_d_active)
                .map(|row| Ok((row, vec![eq_tau1.eval_at(d_start + row)?])))
                .filter_map(|result| match result {
                    Ok((_, weights)) if weights[0].is_zero() => None,
                    other => Some(other),
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            Some(evaluate_setup_columns(
                d_family,
                d_setup_start..d_setup_end,
                &row_weights,
                1,
                &group_alpha_pows_d,
            )?)
        } else {
            None
        };
        let b_setup_accs = if let Some(b_family) = &b_family {
            let _span = tracing::info_span!("relation_weight_b_setup_columns").entered();
            let slice_count = group_lp.outer_slice_count().get();
            let row_weights = (0..physical_n_b)
                .map(|row| {
                    let weights = (0..slice_count)
                        .map(|slice_index| {
                            let logical_row = slice_geometry
                                .logical_row_index(slice_index, row, physical_n_b)?
                                .checked_add(b_range.start)
                                .ok_or(AkitaError::InvalidProof)?;
                            eq_tau1.eval_at(logical_row)
                        })
                        .collect::<Result<Vec<_>, AkitaError>>()?;
                    Ok((row, weights))
                })
                .filter_map(|result| match result {
                    Ok((_, ref weights)) if weights.iter().all(|weight| weight.is_zero()) => None,
                    other => Some(other),
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            Some(evaluate_setup_columns(
                b_family,
                0..b_width,
                &row_weights,
                slice_count,
                &group_alpha_pows_b,
            )?)
        } else {
            None
        };

        for claim in 0..k_g {
            for global_block in 0..num_live_blocks_g {
                let unit = witness_layout.unit_for_block(group_id, global_block)?;
                let challenge_index = claim
                    .checked_mul(num_live_blocks_g)
                    .and_then(|base| base.checked_add(global_block))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("relation challenge index overflow".into())
                    })?;
                let challenge_alpha =
                    challenges.eval_at_pows::<F, E>(challenge_index, &group_alpha_pows_a)?;
                let (slice_index, slice_block) = slice_geometry.block_coordinates(global_block)?;
                for (digit, &opening_gadget) in g_open.iter().enumerate() {
                    for role_subcol in 0..d_ratio {
                        let physical_start = unit.e_coefficient_index(
                            group_d_d,
                            k_g,
                            depth_open,
                            claim,
                            global_block,
                            role_subcol,
                            digit,
                            0,
                        )?;
                        let logical_block = claim * num_live_blocks_g + global_block;
                        let d_phys_col = logical_block
                            .checked_mul(d_ratio)
                            .and_then(|base| base.checked_add(role_subcol))
                            .and_then(|base| base.checked_mul(depth_open))
                            .and_then(|base| base.checked_add(digit))
                            .and_then(|local| e_setup_offset.checked_add(local))
                            .ok_or(AkitaError::InvalidProof)?;
                        let consistency_acc = consistency_weight * challenge_alpha * opening_gadget;
                        let setup_acc = if let Some(weights) = d_setup_accs.as_ref() {
                            let local_col = d_phys_col
                                .checked_sub(d_setup_start)
                                .ok_or(AkitaError::InvalidProof)?;
                            weights.get(0, local_col)?
                        } else {
                            E::zero()
                        };
                        if matches!(opening_method, OpeningMethod::EvaluationTrace) {
                            relation_events.push(
                                physical_start,
                                group_d_d,
                                role_subcol * group_d_d,
                                consistency_acc,
                                RelationWeightContribution::Constraint,
                            )?;
                        }
                        if d_setup_accs.is_some() {
                            relation_events.push(
                                physical_start,
                                group_d_d,
                                0,
                                setup_acc,
                                RelationWeightContribution::SetupMatrix,
                            )?;
                        }
                    }
                }
                for a_idx in 0..n_a {
                    let a_row_weight = eq_tau1.eval_at(a_range.start + a_idx)?;
                    for (digit, &opening_gadget) in t_commit_gadget.iter().enumerate() {
                        let block_claim = slice_geometry
                            .max_blocks_per_slice()
                            .checked_mul(claim)
                            .and_then(|base| base.checked_add(slice_block))
                            .ok_or(AkitaError::InvalidProof)?;
                        let row_block_claim = n_a
                            .checked_mul(block_claim)
                            .and_then(|base| base.checked_add(a_idx))
                            .ok_or(AkitaError::InvalidProof)?;
                        for role_subcol in 0..b_ratio {
                            let local_col = row_block_claim
                                .checked_mul(b_ratio)
                                .and_then(|base| base.checked_add(role_subcol))
                                .and_then(|base| base.checked_mul(depth_commit))
                                .and_then(|base| base.checked_add(digit))
                                .ok_or(AkitaError::InvalidProof)?;
                            let physical_start = unit.t_coefficient_index(
                                group_d_a,
                                group_d_b,
                                k_g,
                                n_a,
                                depth_commit,
                                claim,
                                global_block,
                                a_idx,
                                role_subcol,
                                digit,
                                0,
                            )?;
                            let a_acc = a_row_weight * challenge_alpha * opening_gadget;
                            let b_acc = if let Some(slice_weights) = b_setup_accs.as_ref() {
                                slice_weights.get(slice_index, local_col)?
                            } else {
                                E::zero()
                            };
                            relation_events.push(
                                physical_start,
                                group_d_b,
                                role_subcol * group_d_b,
                                a_acc,
                                RelationWeightContribution::Constraint,
                            )?;
                            if b_setup_accs.is_some() {
                                relation_events.push(
                                    physical_start,
                                    group_d_b,
                                    0,
                                    b_acc,
                                    RelationWeightContribution::SetupMatrix,
                                )?;
                            }
                        }
                    }
                }
            }
        }
        // These setup-column accumulators can be large and are not used by
        // the z-hat phase below. Release them at the named phase boundary.
        drop(d_setup_accs);
        drop(b_setup_accs);

        // For z_hat[blk, dc, df], the column value is:
        //
        // -G_fold[df] * (
        //     tau_consistency * a_alpha[blk] * G_commit[dc]
        //     + sum_r tau_A[r] * A_alpha[r, blk, dc]
        //   ).
        //
        // The first term is the opening row. The second term is the A-row setup
        // contribution. A is already digit-domain, so the A-row setup term does
        // not multiply by G_commit.
        let z_bases = cfg_into_iter!(0..inner_width)
            .map(|k| {
                let block_idx = k / depth_witness;
                let digit_idx = k % depth_witness;
                let constraint = if let Some(point) = ring_multiplier_point {
                    consistency_weight
                        * point.eval_position_at::<E>(block_idx, &group_alpha_pows_a)?
                        * witness_gadget[digit_idx]
                } else {
                    E::zero()
                };
                let mut setup = E::zero();
                if let Some(setup_a_family) = &setup_a_family {
                    for a_idx in 0..n_a {
                        let eq_i = eq_tau1.eval_at(a_range.start + a_idx)?;
                        if !eq_i.is_zero() {
                            setup += eq_i
                                * eval_flat_ring_at_pows_fast(
                                    setup_a_family.ring_slice(a_idx, k)?,
                                    &group_alpha_pows_a,
                                );
                        }
                    }
                }
                Ok((constraint, setup))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        for unit in units {
            for position in 0..num_positions_per_block_g {
                for commit_digit in 0..depth_witness {
                    for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
                        let phys_k = position * depth_witness + commit_digit;
                        let physical_start = unit.z_coefficient_index(
                            group_d_a,
                            num_positions_per_block_g,
                            depth_witness,
                            depth_fold,
                            position,
                            commit_digit,
                            fold_digit,
                            0,
                        )?;
                        if matches!(opening_method, OpeningMethod::EvaluationTrace) {
                            relation_events.push_native_ring(
                                physical_start,
                                group_d_a,
                                -(z_bases[phys_k].0 * fold),
                                RelationWeightContribution::Constraint,
                            )?;
                        }
                        if setup_matrix.is_some() {
                            relation_events.push_native_ring(
                                physical_start,
                                group_d_a,
                                -(z_bases[phys_k].1 * fold),
                                RelationWeightContribution::SetupMatrix,
                            )?;
                        }
                    }
                }
            }
        }
    }
    let r_gadget: Vec<E> = gadget_row_scalars::<F>(levels, lp.open().digits.log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    for (row, &row_dim) in quotient_row_dims.iter().enumerate() {
        if matches!(
            row_families[row],
            RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
        ) {
            continue;
        }
        if matches!(
            row_families[row],
            RelationRowFamily::Consistency {
                opening_method: OpeningMethod::SubringCoefficientPacking { .. },
                ..
            }
        ) {
            continue;
        }
        let eq_weight = eq_tau1.eval_at(row)?;
        let row_alpha_pows = if row_dim == d_a {
            relation_events.alpha_powers.as_slice()
        } else if row_dim == d_b {
            alpha_pows_b.as_slice()
        } else if row_dim == d_d {
            alpha_pows_d.as_slice()
        } else {
            additional_quotient_alpha_powers
                .iter()
                .find_map(|(dimension, powers)| {
                    (*dimension == row_dim).then_some(powers.as_slice())
                })
                .ok_or(AkitaError::InvalidProof)?
        };
        let row_denom = row_alpha_pows[row_dim - 1] * alpha + E::one();
        for (digit, gadget) in r_gadget.iter().enumerate() {
            let physical_start = witness_layout.r_coefficient_index(row, digit, 0, 0)?;
            relation_events.push_native_ring(
                physical_start,
                row_dim,
                -(eq_weight * row_denom * *gadget),
                RelationWeightContribution::Constraint,
            )?;
        }
    }
    Ok((relation_events, opening_semantics))
}

#[cfg(test)]
#[path = "relation_weights_tests.rs"]
mod tests;
