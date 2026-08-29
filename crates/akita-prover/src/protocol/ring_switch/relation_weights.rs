//! Semantic relation-weight events and their canonical consumers.

#[path = "relation_weights/compiler.rs"]
mod compiler;
#[path = "relation_weights/reduced_dense.rs"]
mod reduced_dense;
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
use compiler::{
    compile_group_et_addresses, compile_group_z_addresses, EAddress, RelationWeightCompilationPlan,
    RelationWeightSink, TAddress, ZAddress,
};
use setup_columns::{evaluate_setup_columns, SetupColumnEvaluations, SetupRows};

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
pub(super) use reduced_dense::build_reduced_dense_relation_weights;

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

struct LiftedGroupSink<'a, E: Field> {
    events: &'a mut RelationWeightEvents<E>,
    plan: &'a compiler::RelationWeightGroupPlan<E>,
    challenge_evaluations: &'a [E],
    opening_evaluations: &'a [E],
    d_setup: Option<&'a SetupColumnEvaluations<E>>,
    b_setup: Option<&'a SetupColumnEvaluations<E>>,
    a_setup: Option<&'a [E]>,
}

impl<E: Field> RelationWeightSink<E> for LiftedGroupSink<'_, E> {
    fn add_e(&mut self, address: EAddress<E>) -> Result<(), AkitaError> {
        if matches!(self.plan.opening_method, OpeningMethod::EvaluationTrace) {
            self.events.push(
                address.physical_start,
                self.plan.group_d_d,
                address.role_subcolumn * self.plan.group_d_d,
                self.challenge_evaluations
                    .get(address.challenge_index)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?
                    * address.constraint_scale,
                RelationWeightContribution::Constraint,
            )?;
        }
        if let Some(setup) = self.d_setup {
            self.events.push(
                address.physical_start,
                self.plan.group_d_d,
                0,
                setup.get(0, address.setup_column)?,
                RelationWeightContribution::SetupMatrix,
            )?;
        }
        Ok(())
    }

    fn add_t(&mut self, address: TAddress<E>) -> Result<(), AkitaError> {
        self.events.push(
            address.physical_start,
            self.plan.group_d_b,
            address.role_subcolumn * self.plan.group_d_b,
            self.challenge_evaluations
                .get(address.challenge_index)
                .copied()
                .ok_or(AkitaError::InvalidProof)?
                * address.constraint_scale,
            RelationWeightContribution::Constraint,
        )?;
        if let Some(setup) = self.b_setup {
            self.events.push(
                address.physical_start,
                self.plan.group_d_b,
                0,
                setup.get(address.slice_index, address.setup_column)?,
                RelationWeightContribution::SetupMatrix,
            )?;
        }
        Ok(())
    }

    fn add_z(&mut self, address: ZAddress<E>) -> Result<(), AkitaError> {
        if matches!(self.plan.opening_method, OpeningMethod::EvaluationTrace) {
            self.events.push_native_ring(
                address.physical_start,
                self.plan.group_d_a,
                self.opening_evaluations
                    .get(address.position)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?
                    * address.constraint_scale,
                RelationWeightContribution::Constraint,
            )?;
        }
        if let Some(setup) = self.a_setup {
            self.events.push_native_ring(
                address.physical_start,
                self.plan.group_d_a,
                setup
                    .get(address.setup_column)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?
                    * address.setup_scale,
                RelationWeightContribution::SetupMatrix,
            )?;
        }
        Ok(())
    }
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
    let row_weights = (0..rows)
        .map(|row| eq_tau1.eval_at(row))
        .collect::<Result<Vec<_>, _>>()?;
    let n_d_active = lp.open().matrix.output_rank();
    let levels = r_decomp_levels::<F>(lp.open().digits.log_basis);
    let witness_layout = instance.segment_layout(lp, None)?;
    if witness_layout.r_rows().len() != rows || witness_layout.quotient_depth() != Some(levels) {
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
    let compilation = RelationWeightCompilationPlan::new::<F>(
        lp,
        opening_batch,
        relation_plan,
        &row_families,
        &row_weights,
    )?;
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
    for group_plan in &compilation.groups {
        let group_index = group_plan.group_index;
        let packing_semantics = *packing_semantics_by_group
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        let e_setup_offset = if setup_matrix.is_some() {
            d_column_ranges
                .get(group_index)
                .ok_or(AkitaError::InvalidProof)?
                .start
        } else {
            0
        };
        let group_lp = lp.group_params_geometry(opening_batch, group_index)?;
        let group_d_a = group_plan.group_d_a;
        let group_d_b = group_plan.group_d_b;
        let group_d_d = group_plan.group_d_d;
        let d_ratio = group_plan.d_ratio;
        let group_alpha_pows_a = scalar_powers(alpha, group_d_a);
        let group_alpha_pows_b = scalar_powers(alpha, group_d_b);
        let group_alpha_pows_d = scalar_powers(alpha, group_d_d);
        let k_g = group_plan.num_claims;
        let opening_method = group_plan.opening_method;
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
        let challenge_evaluations = (0..total_blocks)
            .map(|index| challenges.eval_at_pows::<F, E>(index, &group_alpha_pows_a))
            .collect::<Result<Vec<_>, _>>()?;
        let depth_open = group_plan.depth_open;
        let n_a = group_plan.n_a;
        let physical_n_b = group_lp.b_rows_len();
        let inner_width = group_plan.inner_width;
        // Hoist per-group geometry into `Copy` locals so the parallel closures
        // below capture scalars instead of the `!Sync` `&dyn LevelParamsLike`.
        let slice_geometry = &group_plan.slice_geometry;
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
                            let logical_row =
                                slice_geometry.logical_row_index(slice_index, row, physical_n_b)?;
                            group_plan
                                .b_row_weights
                                .get(logical_row)
                                .copied()
                                .ok_or(AkitaError::InvalidProof)
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

        {
            let mut et_sink = LiftedGroupSink {
                events: &mut relation_events,
                plan: group_plan,
                challenge_evaluations: &challenge_evaluations,
                opening_evaluations: &[],
                d_setup: d_setup_accs.as_ref(),
                b_setup: b_setup_accs.as_ref(),
                a_setup: None,
            };
            compile_group_et_addresses(group_plan, &witness_layout, &mut et_sink)?;
        }
        // These setup-column accumulators can be large and are not used by
        // the z-hat phase below. Release them at the named phase boundary.
        drop(challenge_evaluations);
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
        let opening_evaluations = if let Some(point) = ring_multiplier_point {
            (0..group_plan.num_positions)
                .map(|position| point.eval_position_at::<E>(position, &group_alpha_pows_a))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![E::zero(); group_plan.num_positions]
        };
        let a_setup = setup_a_family
            .as_ref()
            .map(|setup_a_family| {
                cfg_into_iter!(0..inner_width)
                    .map(|k| {
                        let mut setup = E::zero();
                        for (a_idx, &row_weight) in group_plan.a_row_weights.iter().enumerate() {
                            if !row_weight.is_zero() {
                                setup += row_weight
                                    * eval_flat_ring_at_pows_fast(
                                        setup_a_family.ring_slice(a_idx, k)?,
                                        &group_alpha_pows_a,
                                    );
                            }
                        }
                        Ok(setup)
                    })
                    .collect::<Result<Vec<_>, AkitaError>>()
            })
            .transpose()?;
        let mut z_sink = LiftedGroupSink {
            events: &mut relation_events,
            plan: group_plan,
            challenge_evaluations: &[],
            opening_evaluations: &opening_evaluations,
            d_setup: None,
            b_setup: None,
            a_setup: a_setup.as_deref(),
        };
        compile_group_z_addresses(group_plan, &witness_layout, &mut z_sink)?;
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
