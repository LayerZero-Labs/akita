//! Relation-weight compilation over the canonical witness addresses.

#[path = "relation_weights/compiler.rs"]
mod compiler;
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
    coefficient_packing_relation_events, gadget_row_scalars, r_decomp_levels,
    validate_coefficient_packing_batch_groups, AkitaExpandedSetup,
    CoefficientPackingBatchSemanticInputs, CommittedGroupParams, FpExtEncoding,
    OpeningClaimsLayout, OpeningFamily, OpeningMethod, PreparedSubringCoefficientPackingPoint,
    RelationAddressGeometry, RelationRangeImagePlan, RelationRowFamily, RelationWitnessGeometry,
    RingRelationInstance, SetupProjectionGeometry,
};
use compiler::{
    compile_et_block_range, compile_z_position_range, EtWeightSink, RelationWeightCompilation,
    ZWeightSink,
};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use setup_columns::{
    contract_setup_columns, contract_setup_residue_columns, SetupColumnValues, SetupRows,
};

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

#[derive(Clone, Copy)]
enum LiftedEtSetup<'a, E: Field> {
    Matrix {
        d: &'a SetupColumnValues<E>,
        b: &'a SetupColumnValues<E>,
    },
    Deferred,
}

/// Blocks whose E/T addresses one task compiles.
const ET_BLOCKS_PER_TASK: usize = 8;

/// Positions whose Z addresses one task compiles.
const Z_POSITIONS_PER_TASK: usize = 64;

/// Disjoint mutable subslices of `values` at `ranges`, in the order of
/// `ranges`. An empty range has an empty subslice.
fn split_disjoint_mut<'a, T>(
    values: &'a mut [T],
    ranges: &[Range<usize>],
) -> Result<Vec<&'a mut [T]>, AkitaError> {
    let mut order = (0..ranges.len())
        .filter(|&index| !ranges[index].is_empty())
        .collect::<Vec<_>>();
    order.sort_unstable_by_key(|&index| ranges[index].start);
    let mut windows = ranges.iter().map(|_| None).collect::<Vec<_>>();
    let mut rest = values;
    let mut consumed = 0;
    for index in order {
        let range = &ranges[index];
        let window = range
            .start
            .checked_sub(consumed)
            .and_then(|gap| std::mem::take(&mut rest).split_at_mut_checked(gap))
            .and_then(|(_, tail)| tail.split_at_mut_checked(range.len()))
            .map(|(window, tail)| {
                rest = tail;
                window
            })
            .ok_or_else(|| {
                AkitaError::InvalidSetup("relation windows overlap or exceed their domain".into())
            })?;
        windows[index] = Some(window);
        consumed = range.end;
    }
    Ok(windows.into_iter().map(Option::unwrap_or_default).collect())
}

/// Add `scale * lane_values[j] + native * alpha^(j L)` to lane `j` of one
/// address, where `lane_alpha_powers[j] = alpha^(j L)`.
fn add_address_lanes<E: Field>(
    lanes: &mut [E],
    lane_values: Option<(&[E], E)>,
    native: E,
    lane_alpha_powers: &[E],
) -> Result<(), AkitaError> {
    if let Some((values, scale)) = lane_values {
        if values.len() != lanes.len() {
            return Err(AkitaError::InvalidProof);
        }
        for (lane, &value) in lanes.iter_mut().zip(values) {
            *lane += value * scale;
        }
    }
    let powers = lane_alpha_powers
        .get(1..lanes.len())
        .ok_or(AkitaError::InvalidProof)?;
    let (first, rest) = lanes.split_first_mut().ok_or(AkitaError::InvalidProof)?;
    *first += native;
    for (lane, &power) in rest.iter_mut().zip(powers) {
        *lane += native * power;
    }
    Ok(())
}

struct LiftedEtSink<'w, 'a, E: Field> {
    e_lanes: LaneWindow<'w, E>,
    t_lanes: LaneWindow<'w, E>,
    plan: &'a compiler::RelationWeightGroupPlan<E>,
    relation_coefficient_block_len: usize,
    /// `c_k(alpha) alpha^(m L)` for challenge `k` and every lane-aligned
    /// exponent `m L` of the relation alpha powers, challenge-major.
    challenge_lanes: &'a [E],
    lane_alpha_powers: &'a [E],
    setup: LiftedEtSetup<'a, E>,
}

impl<'a, E: Field> LiftedEtSink<'_, 'a, E> {
    /// Challenge lanes under subcolumn `role_subcolumn` of a `role_ring_dim`
    /// address.
    fn challenge_lanes(
        &self,
        challenge_index: usize,
        role_subcolumn: usize,
        role_ring_dim: usize,
    ) -> Result<&'a [E], AkitaError> {
        let width = self.lane_alpha_powers.len();
        let lanes = role_ring_dim / self.relation_coefficient_block_len;
        let offset = role_subcolumn
            .checked_mul(lanes)
            .filter(|offset| offset.checked_add(lanes).is_some_and(|end| end <= width))
            .ok_or(AkitaError::InvalidProof)?;
        let start = challenge_index
            .checked_mul(width)
            .and_then(|base| base.checked_add(offset))
            .ok_or(AkitaError::InvalidProof)?;
        self.challenge_lanes
            .get(start..start + lanes)
            .ok_or(AkitaError::InvalidProof)
    }
}

impl<E: Field> EtWeightSink<E> for LiftedEtSink<'_, '_, E> {
    fn add_e(
        &mut self,
        physical_start: usize,
        challenge_index: usize,
        role_subcolumn: usize,
        setup_column: usize,
        constraint_scale: E,
    ) -> Result<(), AkitaError> {
        let d_d = self.plan.roles.d_d;
        let setup = match self.setup {
            LiftedEtSetup::Matrix { d, .. } => d.get_scalar(0, setup_column)?,
            LiftedEtSetup::Deferred => E::zero(),
        };
        let challenge = match self.plan.opening_method {
            OpeningMethod::EvaluationTrace => Some((
                self.challenge_lanes(challenge_index, role_subcolumn, d_d)?,
                constraint_scale,
            )),
            OpeningMethod::SubringCoefficientPacking { .. } => None,
        };
        add_address_lanes(
            self.e_lanes.lanes_mut(physical_start, d_d)?,
            challenge,
            setup,
            self.lane_alpha_powers,
        )
    }

    fn add_t(
        &mut self,
        physical_start: usize,
        challenge_index: usize,
        role_subcolumn: usize,
        slice_index: usize,
        setup_column: usize,
        constraint_scale: E,
    ) -> Result<(), AkitaError> {
        let d_b = self.plan.roles.d_b;
        let setup = match self.setup {
            LiftedEtSetup::Matrix { b, .. } => b.get_scalar(slice_index, setup_column)?,
            LiftedEtSetup::Deferred => E::zero(),
        };
        let challenge = self.challenge_lanes(challenge_index, role_subcolumn, d_b)?;
        add_address_lanes(
            self.t_lanes.lanes_mut(physical_start, d_b)?,
            Some((challenge, constraint_scale)),
            setup,
            self.lane_alpha_powers,
        )
    }
}

#[derive(Clone, Copy)]
enum LiftedZSetup<'a, E: Field> {
    Matrix(&'a SetupColumnValues<E>),
    Deferred,
}

struct LiftedZSink<'w, 'a, E: Field> {
    lanes: LaneWindow<'w, E>,
    plan: &'a compiler::RelationWeightGroupPlan<E>,
    opening_evaluations: &'a [E],
    lane_alpha_powers: &'a [E],
    setup: LiftedZSetup<'a, E>,
}

impl<E: Field> ZWeightSink<E> for LiftedZSink<'_, '_, E> {
    fn add_z(
        &mut self,
        physical_start: usize,
        position: usize,
        setup_column: usize,
        constraint_scale: E,
        setup_scale: E,
    ) -> Result<(), AkitaError> {
        let mut native = match self.setup {
            LiftedZSetup::Matrix(setup) => setup.get_scalar(0, setup_column)? * setup_scale,
            LiftedZSetup::Deferred => E::zero(),
        };
        if matches!(self.plan.opening_method, OpeningMethod::EvaluationTrace) {
            native += self
                .opening_evaluations
                .get(position)
                .copied()
                .ok_or(AkitaError::InvalidProof)?
                * constraint_scale;
        }
        let lanes = self.lanes.lanes_mut(physical_start, self.plan.roles.d_a)?;
        add_address_lanes(lanes, None, native, self.lane_alpha_powers)
    }
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
    let RelationLaneWeightInputs {
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
    if gamma.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }
    let setup_matrix = match setup {
        RelationSetupSource::Matrix(setup) => Some(setup),
        #[cfg(test)]
        RelationSetupSource::DeferredClaim => None,
    };
    let compilation = RelationWeightCompilation::new(
        setup_matrix,
        instance,
        lp,
        tau1,
        opening_source_len,
        opening_ring_dim,
        relation_plan,
    )?;
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
    let setup_is_deferred = setup_matrix.is_none();
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
    // The A ring dimension of every packing group. Only the groups' relation
    // events enter these weights; their Stage 2 terms come from the ring
    // switch, which validated the same relation plan.
    let mut packing_a_ring_dims = vec![None; opening_batch.num_groups()];
    if let OpeningFamily::SubringCoefficientPacking(prepared_points) = opening_points {
        let groups = validate_coefficient_packing_batch_groups(
            &CoefficientPackingBatchSemanticInputs {
                level_params: lp,
                opening_batch,
                relation_plan,
                relation: instance,
                prepared_points,
                alpha,
                tau1,
                claim_coefficients: gamma,
            },
            |group| {
                Ok((
                    group.group_index(),
                    group.geometry().a_ring_dimension(),
                    coefficient_packing_relation_events(&group)?,
                ))
            },
        )?;
        for (group_index, a_ring_dimension, events) in groups {
            let slot = packing_a_ring_dims
                .get_mut(group_index)
                .ok_or(AkitaError::InvalidProof)?;
            if slot.replace(a_ring_dimension).is_some() {
                return Err(AkitaError::InvalidSetup(
                    "packing relation group appears more than once".into(),
                ));
            }
            weights.extend_events(events)?;
        }
    }
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

        {
            let setup = match (d_setup_accs.as_ref(), b_setup_accs.as_ref()) {
                (Some(d), Some(b)) => LiftedEtSetup::Matrix { d, b },
                (None, None) => LiftedEtSetup::Deferred,
                _ => {
                    return Err(AkitaError::InvalidSetup(
                        "lifted E/T setup phases disagree".into(),
                    ));
                }
            };
            let ranges =
                group_plan.et_block_ranges(&compilation.witness_layout, ET_BLOCKS_PER_TASK)?;
            let mut extents = Vec::with_capacity(2 * ranges.len());
            for range in &ranges {
                let [e, t] = group_plan.et_extents(range)?;
                extents.push(e);
                extents.push(t);
            }
            let mut windows = weights.windows_mut(&extents)?.into_iter();
            let tasks = ranges
                .into_iter()
                .map(|range| match (windows.next(), windows.next()) {
                    (Some(e_lanes), Some(t_lanes)) => Ok((range, e_lanes, t_lanes)),
                    _ => Err(AkitaError::InvalidProof),
                })
                .collect::<Result<Vec<_>, _>>()?;
            cfg_into_iter!(tasks).try_for_each(|(range, e_lanes, t_lanes)| {
                let mut sink = LiftedEtSink {
                    e_lanes,
                    t_lanes,
                    plan: group_plan,
                    relation_coefficient_block_len,
                    challenge_lanes: &challenge_lanes,
                    lane_alpha_powers: &lane_alpha_powers,
                    setup,
                };
                compile_et_block_range(group_plan, &range, &mut sink)
            })?;
        }
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
        let setup = match a_setup.as_ref() {
            Some(values) => LiftedZSetup::Matrix(values),
            None => LiftedZSetup::Deferred,
        };
        let ranges =
            group_plan.z_position_ranges(&compilation.witness_layout, Z_POSITIONS_PER_TASK)?;
        let extents = ranges
            .iter()
            .map(|range| group_plan.z_extent(range))
            .collect::<Result<Vec<_>, _>>()?;
        let windows = weights.windows_mut(&extents)?;
        cfg_into_iter!(ranges)
            .zip(windows)
            .try_for_each(|(range, lanes)| {
                let mut sink = LiftedZSink {
                    lanes,
                    plan: group_plan,
                    opening_evaluations: &opening_evaluations,
                    lane_alpha_powers: &lane_alpha_powers,
                    setup,
                };
                compile_z_position_range(group_plan, &range, &mut sink)
            })?;
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
