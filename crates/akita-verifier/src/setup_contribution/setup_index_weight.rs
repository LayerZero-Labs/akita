//! Stage-3 setup-index weight MLE evaluated by the verifier.
//!
//! The prover materializes the same weights densely with
//! [`akita_types::SetupContributionPlan::materialize_setup_index_weights`];
//! only the verifier needs the paired-equality tensor form built here.

use super::*;
use akita_algebra::offset_eq::{
    eval_boolean_pair_tensor_families, EqPairTensorAxis, EqPairTensorFamily,
};

struct ProjectedEqPairTensor<E: Field> {
    ratio: usize,
    families: Vec<EqPairTensorFamily<E>>,
    state: ProjectedEqPairTensorState,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProjectedEqPairTensorState {
    Native,
    RelationFactored,
}

/// Stage-3 setup-index weight polynomial in canonical paired-equality form.
///
/// Only the verifier builds this: it evaluates the packed setup-position
/// weight at one sumcheck point without materializing it. The prover
/// materializes the same weights densely with
/// [`akita_types::SetupContributionPlan::materialize_setup_index_weights`], so it never
/// pays for these tensors.
pub struct SetupIndexWeightMle<E: Field> {
    tensors: Vec<ProjectedEqPairTensor<E>>,
    setup_relation_point: Vec<E>,
    relation_base_bridge_point: Vec<E>,
    projection_geometry: SetupProjectionGeometry,
    relation_coefficient_block_len: usize,
}

impl<E: Field> SetupIndexWeightMle<E> {
    /// Build the setup-index weight tensors for `plan`.
    ///
    /// The B setup tensors are rebuilt here from the witness units the plan
    /// retained at preparation, instead of being stored on the plan.
    ///
    /// # Errors
    ///
    /// Returns an error if any relation tensor fails to align to the Stage-3
    /// setup base.
    pub fn new(plan: &SetupContributionPlan<E>) -> Result<Self, AkitaError> {
        let (relation_base_bridge_point, setup_relation_point) =
            plan.relation_base_bridge_split()?;
        let mut tensors = Vec::<ProjectedEqPairTensor<E>>::new();
        for group in plan.groups() {
            append_d_tensors(plan, group, &mut tensors)?;
            append_b_tensors(plan, group, &mut tensors)?;
            append_a_tensors(plan, group, &mut tensors)?;
        }
        for batch in &mut tensors {
            if batch.ratio > 1 && role_tensors_are_aligned(&batch.families, batch.ratio) {
                factor_aligned_role_tensors(&mut batch.families, batch.ratio)?;
                batch.state = ProjectedEqPairTensorState::RelationFactored;
            }
        }
        Ok(Self {
            tensors,
            setup_relation_point: setup_relation_point.to_vec(),
            relation_base_bridge_point: relation_base_bridge_point.to_vec(),
            projection_geometry: plan.projection_geometry(),
            relation_coefficient_block_len: plan
                .relation_address_geometry()
                .relation_coefficient_block_len(),
        })
    }

    /// Canonical common-base Stage-3 projection geometry of the source plan.
    #[must_use]
    pub const fn projection_geometry(&self) -> SetupProjectionGeometry {
        self.projection_geometry
    }

    /// Evaluate the packed setup-position weight polynomial from its canonical
    /// paired-equality tensors.
    ///
    /// For a role dimension `d_R`, let `q = d_R / base_ring_dim` and
    /// `beta = alpha^base_ring_dim`. The `q` setup subrings and `q` relation
    /// lanes are an explicit tensor axis carrying `beta^v` because
    /// their global compact addresses need not be `q`-aligned. Setup addresses
    /// are always `q`-aligned, so their `beta^u` factor is contracted from the
    /// low setup-point variables once. For `q = 1`, both factors are absent and
    /// no power vector or multiplication by one is performed.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSize`] if `rho_setup_idx` does not have
    /// one coordinate per setup-index variable.
    pub fn evaluate(&self, rho_setup_idx: &[E], alpha: E) -> Result<E, AkitaError> {
        let expected = self.projection_geometry.setup_index_len().trailing_zeros() as usize;
        if rho_setup_idx.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: rho_setup_idx.len(),
            });
        }
        let _span = tracing::info_span!("stage3_setup_index_weight_mle").entered();
        self.tensors
            .iter()
            .try_fold(E::zero(), |evaluation, batch| {
                let ratio = batch.ratio;
                let families = &batch.families;
                if ratio == 1 {
                    return Ok(evaluation
                        + eval_boolean_pair_tensor_families::<_, false, false>(
                            rho_setup_idx,
                            &self.setup_relation_point,
                            families,
                        )?);
                }
                let low_variable_count = ratio.trailing_zeros() as usize;
                let setup_low_point =
                    rho_setup_idx
                        .get(..low_variable_count)
                        .ok_or(AkitaError::InvalidSize {
                            expected: low_variable_count,
                            actual: rho_setup_idx.len(),
                        })?;
                let setup_high_point =
                    rho_setup_idx
                        .get(low_variable_count..)
                        .ok_or(AkitaError::InvalidSize {
                            expected: low_variable_count,
                            actual: rho_setup_idx.len(),
                        })?;
                let setup_projection = role_projection_evaluation(
                    alpha,
                    self.projection_geometry.base_ring_dim(),
                    setup_low_point,
                )?;
                let relation_point = &self.setup_relation_point;
                let contraction = match batch.state {
                    ProjectedEqPairTensorState::RelationFactored => {
                        let relation_low_point = relation_point
                            .get(..low_variable_count)
                            .ok_or(AkitaError::InvalidProof)?;
                        let relation_high_point = relation_point
                            .get(low_variable_count..)
                            .ok_or(AkitaError::InvalidProof)?;
                        let relation_projection = role_projection_evaluation(
                            alpha,
                            self.projection_geometry.base_ring_dim(),
                            relation_low_point,
                        )?;
                        relation_projection
                            * eval_boolean_pair_tensor_families::<_, false, false>(
                                setup_high_point,
                                relation_high_point,
                                families,
                            )?
                    }
                    ProjectedEqPairTensorState::Native => {
                        let projected = project_role_tensors(
                            families,
                            ratio,
                            alpha,
                            self.projection_geometry.base_ring_dim(),
                        )?;
                        eval_boolean_pair_tensor_families::<_, false, false>(
                            setup_high_point,
                            relation_point,
                            &projected,
                        )?
                    }
                };
                Ok(evaluation + setup_projection * contraction)
            })
            .and_then(|evaluation| {
                Ok(evaluation
                    * role_projection_evaluation(
                        alpha,
                        self.relation_coefficient_block_len,
                        &self.relation_base_bridge_point,
                    )?)
            })
    }
}

fn append_d_tensors<E: Field>(
    plan: &SetupContributionPlan<E>,
    group: &SetupContributionGroupPlan<E>,
    batches: &mut Vec<ProjectedEqPairTensor<E>>,
) -> Result<(), AkitaError> {
    if plan.d_rows() == 0 || plan.d_physical_cols() == 0 {
        return Ok(());
    }
    let lifted = group
        .d_tensors()
        .iter()
        .map(|tensor| {
            lift_role_tensor(
                tensor,
                group.d_col_range().start,
                plan.d_physical_cols(),
                plan.d_weights(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for tensor in compact_affine_unit_families(lifted, group.num_claims())? {
        push_projected_tensor(
            batches,
            group.d_ratio(),
            rebase_relation_tensor(&tensor, plan.relation_base_bridge_ratio()?)?,
        )?;
    }
    Ok(())
}

fn append_b_tensors<E: Field>(
    plan: &SetupContributionPlan<E>,
    group: &SetupContributionGroupPlan<E>,
    batches: &mut Vec<ProjectedEqPairTensor<E>>,
) -> Result<(), AkitaError> {
    if group.physical_b().physical_rows() == 0 {
        return Ok(());
    }
    let setup_tensors = build_group_b_setup_tensors(plan.relation_address_geometry(), group)?;
    let tensors = if group.physical_b().geometry().slice_count().is_sliced() {
        setup_tensors
    } else {
        compact_affine_unit_families(setup_tensors, group.num_claims())?
    };
    for tensor in tensors {
        push_projected_tensor(
            batches,
            group.b_ratio(),
            rebase_relation_tensor(&tensor, plan.relation_base_bridge_ratio()?)?,
        )?;
    }
    Ok(())
}

fn append_a_tensors<E: Field>(
    plan: &SetupContributionPlan<E>,
    group: &SetupContributionGroupPlan<E>,
    batches: &mut Vec<ProjectedEqPairTensor<E>>,
) -> Result<(), AkitaError> {
    if group.n_a() == 0 {
        return Ok(());
    }
    let lifted = group
        .a_tensors()
        .iter()
        .map(|tensor| lift_role_tensor(tensor, 0, group.z_cols(), group.a_row_weights()))
        .collect::<Result<Vec<_>, _>>()?;
    for tensor in compact_affine_unit_families(lifted, 1)? {
        push_projected_tensor(
            batches,
            group.a_ratio(),
            rebase_relation_tensor(&tensor, plan.relation_base_bridge_ratio()?)?,
        )?;
    }
    Ok(())
}

fn build_group_b_setup_tensors<E: Field>(
    relation_geometry: RelationAddressGeometry,
    group: &SetupContributionGroupPlan<E>,
) -> Result<Vec<EqPairTensorFamily<E>>, AkitaError> {
    let physical_b = &group.physical_b();
    let geometry = physical_b.geometry();
    let (b_subcolumns, _) =
        SetupProjectionGeometry::native_role_subcolumn_counts(group.role_dims())?;
    let source_lanes = group.a_relation_ratio();
    let a_row_setup_stride = checked::product([group.depth_commit(), b_subcolumns])
        .ok_or_else(|| AkitaError::InvalidSetup("setup B A-row stride overflow".into()))?;
    let block_setup_stride = checked::product([group.n_a(), a_row_setup_stride])
        .ok_or_else(|| AkitaError::InvalidSetup("setup B block stride overflow".into()))?;
    let a_row_relation_stride = checked::product([group.depth_commit(), source_lanes])
        .ok_or_else(|| AkitaError::InvalidSetup("setup B relation A-row stride overflow".into()))?;
    let subcolumn_relation_stride =
        checked::product([group.depth_commit(), group.b_relation_ratio()]).ok_or_else(|| {
            AkitaError::InvalidSetup("setup B subcolumn relation stride overflow".into())
        })?;
    let block_relation_stride = checked::product([group.n_a(), a_row_relation_stride])
        .ok_or_else(|| AkitaError::InvalidSetup("setup B relation block stride overflow".into()))?;
    let claim_setup_stride =
        checked::product([geometry.max_blocks_per_slice(), block_setup_stride])
            .ok_or_else(|| AkitaError::InvalidSetup("setup B claim stride overflow".into()))?;
    let slice_row_weights = (0..geometry.slice_count().get())
        .map(|slice_index| {
            let row_start =
                geometry.logical_row_index(slice_index, 0, physical_b.physical_rows())?;
            Ok::<std::sync::Arc<[E]>, AkitaError>(
                checked_slice(
                    physical_b.logical_row_weights(),
                    row_start,
                    physical_b.physical_rows(),
                    "B slice row weights",
                )?
                .to_vec()
                .into(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut tensors = Vec::new();
    for unit in group.active_units().iter() {
        let unit_start = unit.global_block_start();
        let unit_end = unit_start
            .checked_add(unit.num_live_blocks())
            .ok_or_else(|| AkitaError::InvalidSetup("setup B unit extent overflow".into()))?;
        for (slice_index, slice) in geometry.block_ranges().iter().enumerate() {
            let intersection_start = unit_start.max(slice.start);
            let intersection_end = unit_end.min(slice.end);
            if intersection_start >= intersection_end {
                continue;
            }
            let intersection_len = intersection_end - intersection_start;
            let local_block_start = intersection_start - slice.start;
            for claim in 0..group.num_claims() {
                let setup_column = claim
                    .checked_mul(claim_setup_stride)
                    .and_then(|base| {
                        local_block_start
                            .checked_mul(block_setup_stride)
                            .and_then(|offset| base.checked_add(offset))
                    })
                    .ok_or_else(|| AkitaError::InvalidSetup("setup B address overflow".into()))?;
                let witness_coefficient = unit.t_coefficient_index(
                    group.role_dims().d_a(),
                    group.role_dims().d_b(),
                    group.num_claims(),
                    group.n_a(),
                    group.depth_commit(),
                    claim,
                    intersection_start,
                    0,
                    0,
                    0,
                    0,
                )?;
                let relation_lane_start = checked::exact_div(
                    witness_coefficient,
                    relation_geometry.relation_coefficient_block_len(),
                )
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "setup B coefficient address is not relation-block aligned".into(),
                    )
                })?;
                let row_weights = slice_row_weights.get(slice_index).ok_or_else(|| {
                    AkitaError::InvalidSetup("B slice row weights are missing".into())
                })?;
                tensors.push(EqPairTensorFamily::new(
                    setup_column,
                    relation_lane_start,
                    E::one(),
                    vec![
                        EqPairTensorAxis::unit(group.depth_commit(), 1, group.b_relation_ratio()),
                        EqPairTensorAxis::unit(
                            b_subcolumns,
                            group.depth_commit(),
                            subcolumn_relation_stride,
                        ),
                        EqPairTensorAxis::unit(
                            group.n_a(),
                            a_row_setup_stride,
                            a_row_relation_stride,
                        ),
                        EqPairTensorAxis::unit(
                            intersection_len,
                            block_setup_stride,
                            block_relation_stride,
                        ),
                        EqPairTensorAxis::dense(
                            physical_b.physical_input_width(),
                            0,
                            row_weights.clone(),
                        ),
                    ],
                )?);
            }
        }
    }
    Ok(tensors)
}

fn rebase_relation_tensor<E: Field>(
    tensor: &EqPairTensorFamily<E>,
    bridge_ratio: usize,
) -> Result<EqPairTensorFamily<E>, AkitaError> {
    if bridge_ratio == 0
        || !bridge_ratio.is_power_of_two()
        || !tensor.right_offset.is_multiple_of(bridge_ratio)
        || tensor
            .axes
            .iter()
            .any(|axis| !axis.right_stride.is_multiple_of(bridge_ratio))
    {
        return Err(AkitaError::InvalidSetup(
            "relation tensor does not align to the Stage 3 setup base".into(),
        ));
    }
    let axes = tensor
        .axes
        .iter()
        .cloned()
        .map(|mut axis| {
            axis.right_stride /= bridge_ratio;
            axis
        })
        .collect();
    EqPairTensorFamily::new(
        tensor.left_offset,
        tensor.right_offset / bridge_ratio,
        tensor.scalar,
        axes,
    )
}

fn lift_role_tensor<E: Field>(
    tensor: &EqPairTensorFamily<E>,
    left_offset: usize,
    row_stride: usize,
    row_weights: &[E],
) -> Result<EqPairTensorFamily<E>, AkitaError> {
    let mut axes = tensor.axes.clone();
    axes.push(EqPairTensorAxis::dense(row_stride, 0, row_weights.to_vec()));
    EqPairTensorFamily::new(
        tensor
            .left_offset
            .checked_add(left_offset)
            .ok_or_else(|| AkitaError::InvalidSetup("setup tensor address overflow".into()))?,
        tensor.right_offset,
        tensor.scalar,
        axes,
    )
}

/// Collapse equal-width unit families into one explicit affine unit axis.
///
/// Families are chunk-major with `families_per_unit` semantic lanes inside
/// each chunk. Unequal or non-affine layouts retain their original families.
fn compact_affine_unit_families<E: Field>(
    families: Vec<EqPairTensorFamily<E>>,
    families_per_unit: usize,
) -> Result<Vec<EqPairTensorFamily<E>>, AkitaError> {
    if families_per_unit == 0 || !families.len().is_multiple_of(families_per_unit) {
        return Err(AkitaError::InvalidSetup(
            "setup tensor families disagree with unit lanes".into(),
        ));
    }
    let unit_count = families.len() / families_per_unit;
    if unit_count <= 1 {
        return Ok(families);
    }

    let mut compact = Vec::with_capacity(families_per_unit);
    for lane in 0..families_per_unit {
        let first = families.get(lane).ok_or(AkitaError::InvalidProof)?;
        let second = families
            .get(families_per_unit + lane)
            .ok_or(AkitaError::InvalidProof)?;
        let Some(left_stride) = second.left_offset.checked_sub(first.left_offset) else {
            return Ok(families);
        };
        let Some(right_stride) = second.right_offset.checked_sub(first.right_offset) else {
            return Ok(families);
        };
        for unit in 1..unit_count {
            let family_index = unit
                .checked_mul(families_per_unit)
                .and_then(|index| index.checked_add(lane))
                .ok_or_else(|| AkitaError::InvalidSetup("setup unit index overflow".into()))?;
            let family = families.get(family_index).ok_or(AkitaError::InvalidProof)?;
            let expected_left = first
                .left_offset
                .checked_add(left_stride.checked_mul(unit).ok_or_else(|| {
                    AkitaError::InvalidSetup("setup unit left stride overflow".into())
                })?)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup unit left offset overflow".into())
                })?;
            let expected_right = first
                .right_offset
                .checked_add(right_stride.checked_mul(unit).ok_or_else(|| {
                    AkitaError::InvalidSetup("setup unit right stride overflow".into())
                })?)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup unit right offset overflow".into())
                })?;
            if family.left_offset != expected_left
                || family.right_offset != expected_right
                || family.scalar != first.scalar
                || family.axes != first.axes
            {
                return Ok(families);
            }
        }
        let mut axes = first.axes.clone();
        axes.push(EqPairTensorAxis::unit(
            unit_count,
            left_stride,
            right_stride,
        ));
        compact.push(EqPairTensorFamily::new(
            first.left_offset,
            first.right_offset,
            first.scalar,
            axes,
        )?);
    }
    Ok(compact)
}

fn push_projected_tensor<E: Field>(
    batches: &mut Vec<ProjectedEqPairTensor<E>>,
    ratio: usize,
    family: EqPairTensorFamily<E>,
) -> Result<(), AkitaError> {
    match batches.iter_mut().find(|batch| batch.ratio == ratio) {
        Some(batch) if batch.state == ProjectedEqPairTensorState::Native => {
            batch.families.push(family);
        }
        Some(_) => {
            return Err(AkitaError::InvalidSetup(
                "projected tensors pushed after relation factoring".into(),
            ));
        }
        None => batches.push(ProjectedEqPairTensor {
            ratio,
            families: vec![family],
            state: ProjectedEqPairTensorState::Native,
        }),
    }
    Ok(())
}
