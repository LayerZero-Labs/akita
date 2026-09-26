//! Relation-weight compilation over the canonical witness addresses.

#[path = "relation_weights/compiler.rs"]
mod compiler;
mod lifted_sinks;
mod schedule;
mod sinks;
use lifted_sinks::{scatter_et, scatter_z, split_disjoint_mut};
#[path = "relation_weights/reduced_dense.rs"]
mod reduced_dense;
#[path = "relation_weights/setup_columns.rs"]
mod setup_columns;
mod stage2;
pub(crate) use stage2::compile_stage2_weights;

pub(crate) enum RelationWeightDescription<E: Field> {
    QuotientFactored(RelationWeightFactorization<E>),
    ReducedEvaluations {
        evaluations: Vec<E>,
        live_len: usize,
    },
}

use std::ops::Range;

use akita_algebra::eq_poly::SplitEqEvals;
use akita_algebra::ring::{eval_flat_ring_at_pows_fast, scalar_powers};
use akita_error::AkitaError;
use akita_types::RelationWeightContribution;
use akita_types::{
    gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup, CoefficientPackingBatchSemantics,
    CommittedGroupParams, FpExtEncoding, OpeningClaimsLayout, OpeningFamily, OpeningMethod,
    PreparedSubringCoefficientPackingPoint, RelationAddressGeometry, RelationRangeImagePlan,
    RelationRowFamily, RelationWitnessGeometry, RingRelationInstance, SetupProjectionGeometry,
};
use compiler::RelationWeightCompilation;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use setup_columns::{
    contract_setup_columns, contract_setup_residue_columns, SetupColumnValues, SetupRows,
};
use sinks::{compile_et_block_range, compile_z_position_range, EtWeightSink, ZWeightSink};

/// Source of setup-matrix relation weights for this evaluation.
#[derive(Clone, Copy)]
pub(crate) enum RelationSetupSource<'a, F: Field> {
    /// Add setup contributions directly from the expanded setup matrix.
    Matrix(&'a AkitaExpandedSetup<F>),
    #[cfg(test)]
    DeferredClaim,
}

/// Inputs to the one quotient-lift relation-weight builder.
pub(crate) struct RelationLaneWeightInputs<'a, F: Field, E: Field> {
    pub setup: RelationSetupSource<'a, F>,
    pub instance: &'a RingRelationInstance<F>,
    pub alpha: E,
    pub level_params: &'a CommittedGroupParams,
    pub relation_row_point: &'a [E],
    pub claim_coefficients: &'a [E],
    pub opening_source_len: usize,
    pub opening_ring_dim: usize,
    pub relation_plan: &'a RelationRangeImagePlan,
    pub packing_semantics: Option<&'a CoefficientPackingBatchSemantics<E>>,
    /// Method-typed prepared points for the current fold.
    pub opening_points:
        OpeningFamily<(), &'a [(usize, &'a PreparedSubringCoefficientPackingPoint<E>)]>,
}

mod lane_weights;
use lane_weights::LaneWindow;
pub(crate) use lane_weights::{RelationLaneWeights, RelationWeightFactorization};
pub(crate) use reduced_dense::build_reduced_dense_relation_weights;

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

fn validate_relation_inputs<'a, F, E>(
    inputs: &RelationLaneWeightInputs<'a, F, E>,
) -> Result<RelationWeightCompilation<'a, F, E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field + ExtField<F>,
{
    if inputs.claim_coefficients.len() != inputs.instance.opening_batch().num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }
    let setup_matrix = match inputs.setup {
        RelationSetupSource::Matrix(setup) => Some(setup),
        #[cfg(test)]
        RelationSetupSource::DeferredClaim => None,
    };
    RelationWeightCompilation::new(
        setup_matrix,
        inputs.instance,
        inputs.level_params,
        inputs.relation_row_point,
        inputs.opening_source_len,
        inputs.opening_ring_dim,
        inputs.relation_plan,
    )
}

fn pack_relation_events<E: Field>(
    weights: &mut RelationLaneWeights<E>,
    num_groups: usize,
    packing_required: bool,
    packing_semantics: Option<&CoefficientPackingBatchSemantics<E>>,
    live_coeff_len: usize,
    relation_coefficient_block_len: usize,
) -> Result<Vec<Option<usize>>, AkitaError> {
    let mut packing_a_ring_dims = vec![None; num_groups];
    if packing_required != packing_semantics.is_some() {
        return Err(AkitaError::InvalidProof);
    }
    if let Some(batch) = packing_semantics {
        for group in batch.groups() {
            let terms = group.stage2_terms();
            if terms.physical_field_len() != live_coeff_len
                || terms.relation_coefficient_block_len() != relation_coefficient_block_len
            {
                return Err(AkitaError::InvalidSetup(
                    "packing semantics disagree with the current ring switch".into(),
                ));
            }
            let slot = packing_a_ring_dims
                .get_mut(group.group_index())
                .ok_or(AkitaError::InvalidProof)?;
            if slot.replace(group.geometry().a_ring_dimension()).is_some() {
                return Err(AkitaError::InvalidSetup(
                    "packing relation group appears more than once".into(),
                ));
            }
            weights.extend_events(group.relation_weight_events().iter().cloned())?;
        }
    }
    Ok(packing_a_ring_dims)
}

/// The complete checked relation weights for one fold.
#[tracing::instrument(skip_all, name = "build_relation_lane_weights")]
pub(crate) fn build_relation_lane_weights<F, E>(
    inputs: RelationLaneWeightInputs<'_, F, E>,
) -> Result<RelationLaneWeights<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
{
    let compilation = validate_relation_inputs(&inputs)?;
    let RelationLaneWeightInputs {
        instance,
        alpha,
        level_params: lp,
        opening_points,
        packing_semantics,
        ..
    } = inputs;
    let opening_batch = instance.opening_batch();
    let role_dims = instance.role_dims();
    let d_a = role_dims.d_a();
    let d_b = role_dims.d_b();
    let d_d = role_dims.d_d();
    let alpha_pows_b = scalar_powers(alpha, d_b);
    let alpha_pows_d = scalar_powers(alpha, d_d);
    let packing_required = matches!(
        compilation.relation_geometry.group_opening_method(0)?,
        OpeningMethod::SubringCoefficientPacking { .. }
    );
    if packing_required != matches!(opening_points, OpeningFamily::SubringCoefficientPacking(_)) {
        return Err(AkitaError::InvalidSetup(
            "relation opening family disagrees with prepared points".into(),
        ));
    }
    let quotient_row_dims = compilation
        .row_families
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
    let levels = r_decomp_levels::<F>(lp.open().digits.log_basis);
    let setup_is_deferred = compilation.setup_sources.is_none();
    let relation_coefficient_block_len = compilation.relation_coefficient_block_len;
    let physical_field_len = compilation.physical_field_len;
    let mut weights = RelationLaneWeights::new(
        scalar_powers(
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
    )?;
    let lane_alpha_powers = weights.lane_alpha_powers();
    let packing_a_ring_dims = pack_relation_events(
        &mut weights,
        opening_batch.num_groups(),
        packing_required,
        packing_semantics,
        compilation.witness_layout.live_coeff_len(),
        relation_coefficient_block_len,
    )?;
    for group_plan in &compilation.plan.groups {
        let group_index = group_plan.group_index;
        let group_source = compilation.group_source(group_index)?;
        let group_setup = compilation
            .setup_sources
            .as_ref()
            .map(|sources| sources.group(group_index))
            .transpose()?;
        let packing_a_ring_dim = *packing_a_ring_dims
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        let group_d_a = group_plan.roles.d_a;
        let group_d_b = group_plan.roles.d_b;
        let group_d_d = group_plan.roles.d_d;
        let group_alpha_pows_a = scalar_powers(alpha, group_d_a);
        let group_alpha_pows_b = scalar_powers(alpha, group_d_b);
        let group_alpha_pows_d = scalar_powers(alpha, group_d_d);
        let opening_method = group_plan.opening_method;
        match (opening_method, packing_a_ring_dim) {
            (OpeningMethod::EvaluationTrace, None) => {}
            (OpeningMethod::SubringCoefficientPacking { .. }, Some(a_ring_dim))
                if a_ring_dim == group_d_a => {}
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "packing semantic groups do not match scheduled opening methods".into(),
                ));
            }
        }
        let ring_multiplier_point = match group_source.opening {
            OpeningFamily::EvaluationTrace(point) => Some(point),
            OpeningFamily::SubringCoefficientPacking(()) => None,
        };
        let challenges = group_source.challenges;
        let total_blocks = challenges.len();
        let challenge_lanes = {
            let lane_count = akita_error::checked::product([total_blocks, lane_alpha_powers.len()])
                .ok_or(AkitaError::InvalidProof)?;
            let mut lanes = Vec::with_capacity(lane_count);
            for index in 0..total_blocks {
                let evaluation = challenges.eval_at_pows::<F, E>(index, &group_alpha_pows_a)?;
                lanes.extend(lane_alpha_powers.iter().map(|&power| evaluation * power));
            }
            lanes
        };
        let d_setup_accs = if let Some(setup) = compilation.setup_sources.as_ref() {
            let _span = tracing::info_span!("relation_weight_d_setup_columns").entered();
            Some(contract_setup_columns(
                &setup.d,
                group_plan.rows.d_setup_range.clone(),
                &compilation.plan.d_row_weights,
                1,
                1,
                |coefficients| {
                    Ok(vec![eval_flat_ring_at_pows_fast(
                        coefficients,
                        &group_alpha_pows_d,
                    )])
                },
            )?)
        } else {
            None
        };
        let b_setup_accs = if let Some(group_setup) = group_setup {
            let _span = tracing::info_span!("relation_weight_b_setup_columns").entered();
            Some(contract_setup_columns(
                &group_setup.b,
                0..group_plan.witness.b_width,
                &group_plan.rows.b_setup_row_weights,
                group_plan.witness.slice_count,
                1,
                |coefficients| {
                    Ok(vec![eval_flat_ring_at_pows_fast(
                        coefficients,
                        &group_alpha_pows_b,
                    )])
                },
            )?)
        } else {
            None
        };

        scatter_et(
            group_plan,
            &compilation.witness_layout,
            weights.lanes_mut(),
            relation_coefficient_block_len,
            &challenge_lanes,
            &lane_alpha_powers,
            d_setup_accs.as_ref(),
            b_setup_accs.as_ref(),
        )?;
        // These setup-column accumulators can be large and are not used by
        // the z-hat phase below. Release them at the named phase boundary.
        drop(challenge_lanes);
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
            (0..group_plan.witness.num_positions)
                .map(|position| point.eval_position_at::<E>(position, &group_alpha_pows_a))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![E::zero(); group_plan.witness.num_positions]
        };
        let a_setup = group_setup
            .map(|group_setup| {
                contract_setup_columns(
                    &group_setup.a,
                    0..group_plan.witness.inner_width,
                    &group_plan.rows.a_setup_row_weights,
                    1,
                    1,
                    |coefficients| {
                        Ok(vec![eval_flat_ring_at_pows_fast(
                            coefficients,
                            &group_alpha_pows_a,
                        )])
                    },
                )
            })
            .transpose()?;
        scatter_z(
            group_plan,
            &compilation.witness_layout,
            weights.lanes_mut(),
            relation_coefficient_block_len,
            &opening_evaluations,
            &lane_alpha_powers,
            a_setup.as_ref(),
        )?;
    }
    let r_gadget: Vec<E> = gadget_row_scalars::<F>(levels, lp.open().digits.log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    for (row, &row_dim) in quotient_row_dims.iter().enumerate() {
        if matches!(
            compilation.row_families[row],
            RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
        ) {
            continue;
        }
        if matches!(
            compilation.row_families[row],
            RelationRowFamily::Consistency {
                opening_method: OpeningMethod::SubringCoefficientPacking { .. },
                ..
            }
        ) {
            continue;
        }
        let eq_weight = compilation.row_weights[row];
        let row_alpha_pows = if row_dim == d_a {
            weights.alpha_powers()
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
            let physical_start = compilation
                .witness_layout
                .r_coefficient_index(row, digit, 0, 0)?;
            weights.push(
                physical_start,
                row_dim,
                0,
                -(eq_weight * row_denom * *gadget),
                RelationWeightContribution::Constraint,
            )?;
        }
    }
    Ok(weights)
}

#[cfg(test)]
#[path = "relation_weights_tests.rs"]
mod tests;
