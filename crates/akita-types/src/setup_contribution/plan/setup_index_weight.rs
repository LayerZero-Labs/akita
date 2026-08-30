use super::physical_b::build_group_b_setup_tensors;
use super::types::{ProjectedEqPairTensor, ProjectedEqPairTensorState};
use super::*;
use akita_algebra::fft::field_pow;
use akita_algebra::{
    offset_eq::{
        eval_boolean_pair_tensor_families, materialize_eq_tensor_left, EqPairTensorAxis,
        EqPairTensorFamily, OffsetEqWindow,
    },
    ring::{evaluate_power_sequence_mle, scalar_powers_with_stride},
};

struct GroupSetupIndexWeights<E> {
    projection_scales: [Option<Vec<E>>; 3],
    column_weights: [Vec<E>; 3],
    physical_b_weights: Vec<E>,
}

impl<E: Field> SetupContributionPlan<E> {
    pub(super) fn materialize_role_tensor_weights(
        &self,
        ratio: usize,
        tensors: &[EqPairTensorFamily<E>],
        output_len: usize,
        alpha: E,
    ) -> Result<Vec<E>, AkitaError> {
        if ratio == 1 {
            return materialize_eq_tensor_left(
                self.relation_address.equality_window(),
                tensors,
                output_len,
            );
        }
        if role_tensors_are_aligned(tensors, ratio) {
            let low_variable_count = ratio.trailing_zeros() as usize;
            let point = self.relation_address.point();
            let low_point = point
                .get(..low_variable_count)
                .ok_or(AkitaError::InvalidProof)?;
            let high_point = point
                .get(low_variable_count..)
                .ok_or(AkitaError::InvalidProof)?;
            let mut factored = tensors.to_vec();
            factor_aligned_role_tensors(&mut factored, ratio)?;
            let equality = OffsetEqWindow::new(high_point)?;
            let mut weights = materialize_eq_tensor_left(&equality, &factored, output_len)?;
            let projection = role_projection_evaluation(
                alpha,
                self.relation_address_geometry
                    .relation_coefficient_block_len(),
                low_point,
            )?;
            if projection != E::one() {
                const PARALLEL_THRESHOLD: usize = 1 << 14;
                if weights.len() >= PARALLEL_THRESHOLD {
                    cfg_iter_mut!(weights).for_each(|weight| *weight *= projection);
                } else {
                    weights.iter_mut().for_each(|weight| *weight *= projection);
                }
            }
            return Ok(weights);
        }
        let projected = project_role_tensors(
            tensors,
            ratio,
            alpha,
            self.relation_address_geometry
                .relation_coefficient_block_len(),
        )?;
        materialize_eq_tensor_left(
            self.relation_address.equality_window(),
            &projected,
            output_len,
        )
    }

    /// Materialize the dense packed setup-position weight vector.
    pub fn materialize_setup_index_weights(&self, alpha: E) -> Result<Vec<E>, AkitaError> {
        // Both power families depend only on the group. Hoist their allocation
        // and exponentiation out of the potentially million-element setup loop.
        let group_weights = self
            .groups
            .iter()
            .map(|group| -> Result<_, AkitaError> {
                let column_weights = [
                    self.materialize_role_tensor_weights(
                        group.a_relation_ratio,
                        &group.a_tensors,
                        group.z_cols,
                        alpha,
                    )?,
                    self.materialize_role_tensor_weights(
                        group.b_relation_ratio,
                        &group.physical_b.relation_tensors,
                        group.physical_b.logical_input_width(),
                        alpha,
                    )?,
                    self.materialize_role_tensor_weights(
                        group.d_relation_ratio,
                        &group.d_tensors,
                        group.d_col_range.len(),
                        alpha,
                    )?,
                ];
                let physical_b_weights = group
                    .physical_b
                    .contract_logical_column_weights(&column_weights[1])?;
                Ok(GroupSetupIndexWeights {
                    projection_scales: self.group_projection_scales(group, alpha)?,
                    column_weights,
                    physical_b_weights,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        (0..self.required())
            .map(|setup_idx| self.setup_index_weight_at(setup_idx, &group_weights))
            .collect()
    }

    /// Evaluate the packed setup-position weight polynomial from its canonical
    /// paired-equality tensors.
    ///
    /// For a role dimension `d_R`, let `q = d_R / base_ring_dim` and
    /// `beta = alpha^base_ring_dim`. The `q` setup subrings and `q` relation
    /// relation lanes are an explicit tensor axis carrying `beta^v` because
    /// their global compact addresses need not be `q`-aligned. Setup addresses
    /// are always `q`-aligned, so their `beta^u` factor is contracted from the
    /// low setup-point variables once. For `q = 1`, both factors are absent and
    /// no power vector or multiplication by one is performed.
    pub fn evaluate_setup_index_weight_mle(
        &self,
        rho_setup_idx: &[E],
        alpha: E,
    ) -> Result<E, AkitaError> {
        let expected = self.projection_geometry.setup_index_len().trailing_zeros() as usize;
        if rho_setup_idx.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: rho_setup_idx.len(),
            });
        }
        let _span = tracing::info_span!("stage3_setup_index_weight_mle").entered();
        self.setup_index_tensors
            .iter()
            .try_fold(E::zero(), |evaluation, batch| {
                let ratio = batch.ratio;
                let families = &batch.families;
                if ratio == 1 {
                    return Ok(evaluation
                        + eval_boolean_pair_tensor_families::<_, false, false>(
                            rho_setup_idx,
                            self.setup_relation_address.point(),
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
                let relation_point = self.setup_relation_address.point();
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
                        self.relation_address_geometry
                            .relation_coefficient_block_len(),
                        &self.relation_base_bridge_point,
                    )?)
            })
    }

    pub(crate) fn prepare_setup_index_tensors(
        &mut self,
        witness_layout: &WitnessLayout,
    ) -> Result<Vec<ProjectedEqPairTensor<E>>, AkitaError> {
        let relation_geometry = self.relation_address_geometry;
        for group in &mut self.groups {
            let [d_tensors, b_tensors, a_tensors] =
                build_group_role_tensors(relation_geometry, group, witness_layout)?;
            group.d_tensors = d_tensors;
            group.physical_b.relation_tensors = b_tensors;
            group.a_tensors = a_tensors;
            group.physical_b.setup_tensors =
                build_group_b_setup_tensors(relation_geometry, group, witness_layout)?;
        }
        let mut batches = Vec::<ProjectedEqPairTensor<E>>::new();
        for group in &self.groups {
            self.append_d_tensors(group, &mut batches)?;
            self.append_b_tensors(group, &mut batches)?;
            self.append_a_tensors(group, &mut batches)?;
        }
        for batch in &mut batches {
            if batch.ratio > 1 && role_tensors_are_aligned(&batch.families, batch.ratio) {
                factor_aligned_role_tensors(&mut batch.families, batch.ratio)?;
                batch.state = ProjectedEqPairTensorState::RelationFactored;
            }
        }
        Ok(batches)
    }

    fn group_projection_scales(
        &self,
        group: &SetupContributionGroupPlan<E>,
        alpha: E,
    ) -> Result<[Option<Vec<E>>; 3], AkitaError> {
        let base_ring_dim = self.projection_geometry.base_ring_dim();
        let role_scales = |role_dim: usize| {
            let ratio = role_dim
                .checked_div(base_ring_dim)
                .filter(|count| *count != 0 && role_dim.is_multiple_of(base_ring_dim))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "setup role dimension does not decompose over its base ring".into(),
                    )
                })?;
            if ratio == 1 {
                Ok(None)
            } else {
                scalar_powers_with_stride(alpha, base_ring_dim, ratio).map(Some)
            }
        };
        Ok([
            role_scales(group.role_dims.d_a())?,
            role_scales(group.role_dims.d_b())?,
            role_scales(group.role_dims.d_d())?,
        ])
    }

    fn setup_index_weight_at(
        &self,
        setup_idx: usize,
        group_weights: &[GroupSetupIndexWeights<E>],
    ) -> Result<E, AkitaError> {
        let geometry = self.projection_geometry;
        if setup_idx >= geometry.required() {
            return Err(AkitaError::InvalidSize {
                expected: geometry.required(),
                actual: setup_idx,
            });
        }
        let mut weight = E::zero();
        for (group, weights) in self.groups.iter().zip(group_weights) {
            let scales = &weights.projection_scales;
            let [z_eq, _t_eq, e_eq] = &weights.column_weights;
            let d_idx = setup_idx / group.d_ratio;
            let d_footprint = self
                .d_rows
                .checked_mul(self.d_physical_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
            if d_idx < d_footprint {
                let d_col = d_idx % self.d_physical_cols;
                let d_row = d_idx / self.d_physical_cols;
                if group.d_col_range.contains(&d_col) {
                    let term = self.d_weights[d_row]
                        * *e_eq
                            .get(d_col - group.d_col_range.start)
                            .ok_or(AkitaError::InvalidProof)?;
                    weight += scales[2]
                        .as_ref()
                        .map_or(term, |scale| scale[setup_idx % group.d_ratio] * term);
                }
            }

            let b_idx = setup_idx / group.b_ratio;
            let b_footprint = group.physical_b.physical_footprint()?;
            if b_idx < b_footprint {
                let term = *weights
                    .physical_b_weights
                    .get(b_idx)
                    .ok_or(AkitaError::InvalidProof)?;
                weight += scales[1]
                    .as_ref()
                    .map_or(term, |scale| scale[setup_idx % group.b_ratio] * term);
            }

            let a_idx = setup_idx / group.a_ratio;
            let a_footprint = group
                .n_a
                .checked_mul(group.z_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("setup A footprint overflow".into()))?;
            if a_idx < a_footprint {
                let a_col = a_idx % group.z_cols;
                let a_row = a_idx / group.z_cols;
                let term = group.a_row_weights[a_row]
                    * *z_eq.get(a_col).ok_or(AkitaError::InvalidProof)?;
                weight += scales[0]
                    .as_ref()
                    .map_or(term, |scale| scale[setup_idx % group.a_ratio] * term);
            }
        }
        Ok(weight)
    }

    fn append_d_tensors(
        &self,
        group: &SetupContributionGroupPlan<E>,
        batches: &mut Vec<ProjectedEqPairTensor<E>>,
    ) -> Result<(), AkitaError> {
        if self.d_rows == 0 || self.d_physical_cols == 0 {
            return Ok(());
        }
        let lifted = group
            .d_tensors
            .iter()
            .map(|tensor| {
                lift_role_tensor(
                    tensor,
                    group.d_col_range.start,
                    self.d_physical_cols,
                    &self.d_weights,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        for tensor in compact_affine_unit_families(lifted, group.num_claims)? {
            push_projected_tensor(
                batches,
                group.d_ratio,
                rebase_relation_tensor(&tensor, self.relation_base_bridge_ratio()?)?,
            )?;
        }
        Ok(())
    }

    fn append_b_tensors(
        &self,
        group: &SetupContributionGroupPlan<E>,
        batches: &mut Vec<ProjectedEqPairTensor<E>>,
    ) -> Result<(), AkitaError> {
        if group.physical_b.physical_rows == 0 {
            return Ok(());
        }
        let tensors = if group.physical_b.geometry().slice_count().is_sliced() {
            group.physical_b.setup_tensors.clone()
        } else {
            compact_affine_unit_families(group.physical_b.setup_tensors.clone(), group.num_claims)?
        };
        for tensor in tensors {
            push_projected_tensor(
                batches,
                group.b_ratio,
                rebase_relation_tensor(&tensor, self.relation_base_bridge_ratio()?)?,
            )?;
        }
        Ok(())
    }

    fn append_a_tensors(
        &self,
        group: &SetupContributionGroupPlan<E>,
        batches: &mut Vec<ProjectedEqPairTensor<E>>,
    ) -> Result<(), AkitaError> {
        if group.n_a == 0 {
            return Ok(());
        }
        let lifted = group
            .a_tensors
            .iter()
            .map(|tensor| lift_role_tensor(tensor, 0, group.z_cols, &group.a_row_weights))
            .collect::<Result<Vec<_>, _>>()?;
        for tensor in compact_affine_unit_families(lifted, 1)? {
            push_projected_tensor(
                batches,
                group.a_ratio,
                rebase_relation_tensor(&tensor, self.relation_base_bridge_ratio()?)?,
            )?;
        }
        Ok(())
    }

    fn relation_base_bridge_ratio(&self) -> Result<usize, AkitaError> {
        let relation_base = self
            .relation_address_geometry
            .relation_coefficient_block_len();
        self.projection_geometry
            .base_ring_dim()
            .checked_div(relation_base)
            .filter(|ratio| {
                relation_base != 0
                    && self
                        .projection_geometry
                        .base_ring_dim()
                        .is_multiple_of(relation_base)
                    && ratio.is_power_of_two()
            })
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "setup base does not decompose over the relation base".into(),
                )
            })
    }
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

fn build_group_role_tensors<E: Field>(
    relation_geometry: RelationAddressGeometry,
    group: &SetupContributionGroupPlan<E>,
    witness_layout: &WitnessLayout,
) -> Result<[Vec<EqPairTensorFamily<E>>; 3], AkitaError> {
    let (b_subcolumns, _) = SetupProjectionGeometry::native_role_subcolumn_counts(group.role_dims)?;
    let d_subcolumns = group.opening_subcolumns;
    let source_lanes = group.a_relation_ratio;
    let a_relation_ring_stride = source_lanes;

    let d_block_setup_stride = checked::product([d_subcolumns, group.depth_open])
        .ok_or_else(|| AkitaError::InvalidSetup("setup D block stride overflow".into()))?;
    let opening_relation_lanes = checked::product([d_subcolumns, group.d_relation_ratio])
        .ok_or_else(|| AkitaError::InvalidSetup("setup D opening lane count overflow".into()))?;
    let d_block_relation_stride = checked::product([group.depth_open, opening_relation_lanes])
        .ok_or_else(|| AkitaError::InvalidSetup("setup D relation stride overflow".into()))?;
    let d_subcolumn_relation_stride = checked::product([group.depth_open, group.d_relation_ratio])
        .ok_or_else(|| {
            AkitaError::InvalidSetup("setup D subcolumn relation stride overflow".into())
        })?;
    let b_a_row_setup_stride = checked::product([group.depth_commit, b_subcolumns])
        .ok_or_else(|| AkitaError::InvalidSetup("setup B A-row stride overflow".into()))?;
    let b_block_setup_stride = checked::product([group.n_a, b_a_row_setup_stride])
        .ok_or_else(|| AkitaError::InvalidSetup("setup B block stride overflow".into()))?;
    let b_a_row_relation_stride = checked::product([group.depth_commit, source_lanes])
        .ok_or_else(|| AkitaError::InvalidSetup("setup B relation A-row stride overflow".into()))?;
    let b_subcolumn_relation_stride =
        checked::product([group.depth_commit, group.b_relation_ratio]).ok_or_else(|| {
            AkitaError::InvalidSetup("setup B subcolumn relation stride overflow".into())
        })?;
    let b_block_relation_stride = checked::product([group.n_a, b_a_row_relation_stride])
        .ok_or_else(|| AkitaError::InvalidSetup("setup B relation block stride overflow".into()))?;
    let a_relation_column_stride =
        checked::product([group.fold_gadget.len(), a_relation_ring_stride]).ok_or_else(|| {
            AkitaError::InvalidSetup("setup A relation column stride overflow".into())
        })?;
    let fold_weights = group
        .fold_gadget
        .iter()
        .copied()
        .map(std::ops::Neg::neg)
        .collect::<Vec<_>>();

    let mut d_tensors = Vec::new();
    let mut b_tensors = Vec::new();
    let mut a_tensors = Vec::new();
    // Emission is unit-major with claim as the inner index. Structured replay
    // consumes D/B tensors at `unit * num_claims + claim`; changing this order
    // requires changing that index contract at the same time.
    for unit in witness_layout.units_for_group(group.group_id)? {
        for claim in 0..group.num_claims {
            if unit.num_live_blocks() == 0 {
                continue;
            }
            let d_setup_column = claim
                .checked_mul(group.num_live_blocks)
                .and_then(|base| base.checked_add(unit.global_block_start()))
                .and_then(|base| base.checked_mul(d_block_setup_stride))
                .ok_or_else(|| AkitaError::InvalidSetup("setup D address overflow".into()))?;
            let d_witness_coefficient = unit.e_coefficient_index(
                group.role_dims.d_d(),
                group.num_claims,
                group.depth_open,
                claim,
                unit.global_block_start(),
                0,
                0,
                0,
            )?;
            let d_relation_lane_start = divide_aligned(
                d_witness_coefficient,
                relation_geometry.relation_coefficient_block_len(),
                "setup D coefficient address is not relation-block aligned",
            )?;
            d_tensors.push(EqPairTensorFamily::new(
                d_setup_column,
                d_relation_lane_start,
                E::one(),
                vec![
                    EqPairTensorAxis::unit(group.depth_open, 1, group.d_relation_ratio),
                    EqPairTensorAxis::unit(
                        d_subcolumns,
                        group.depth_open,
                        d_subcolumn_relation_stride,
                    ),
                    EqPairTensorAxis::unit(
                        unit.num_live_blocks(),
                        d_block_setup_stride,
                        d_block_relation_stride,
                    ),
                ],
            )?);

            if group.physical_b.logical_rows()? != 0 {
                let b_setup_column = claim
                    .checked_mul(group.num_live_blocks)
                    .and_then(|base| base.checked_add(unit.global_block_start()))
                    .and_then(|base| base.checked_mul(b_block_setup_stride))
                    .ok_or_else(|| AkitaError::InvalidSetup("setup B address overflow".into()))?;
                let b_witness_coefficient = unit.t_coefficient_index(
                    group.role_dims.d_a(),
                    group.role_dims.d_b(),
                    group.num_claims,
                    group.n_a,
                    group.depth_commit,
                    claim,
                    unit.global_block_start(),
                    0,
                    0,
                    0,
                    0,
                )?;
                let b_relation_lane_start = divide_aligned(
                    b_witness_coefficient,
                    relation_geometry.relation_coefficient_block_len(),
                    "setup B coefficient address is not relation-block aligned",
                )?;
                b_tensors.push(EqPairTensorFamily::new(
                    b_setup_column,
                    b_relation_lane_start,
                    E::one(),
                    vec![
                        EqPairTensorAxis::unit(group.depth_commit, 1, group.b_relation_ratio),
                        EqPairTensorAxis::unit(
                            b_subcolumns,
                            group.depth_commit,
                            b_subcolumn_relation_stride,
                        ),
                        EqPairTensorAxis::unit(
                            group.n_a,
                            b_a_row_setup_stride,
                            b_a_row_relation_stride,
                        ),
                        EqPairTensorAxis::unit(
                            unit.num_live_blocks(),
                            b_block_setup_stride,
                            b_block_relation_stride,
                        ),
                    ],
                )?);
            }
        }

        if group.n_a != 0 {
            let a_witness_coefficient = unit.z_coefficient_index(
                group.role_dims.d_a(),
                group.num_positions_per_block,
                group.depth_witness,
                group.fold_gadget.len(),
                0,
                0,
                0,
                0,
            )?;
            let a_relation_lane_start = divide_aligned(
                a_witness_coefficient,
                relation_geometry.relation_coefficient_block_len(),
                "setup A coefficient address is not relation-block aligned",
            )?;
            a_tensors.push(EqPairTensorFamily::new(
                0,
                a_relation_lane_start,
                E::one(),
                vec![
                    EqPairTensorAxis::unit(group.z_cols, 1, a_relation_column_stride),
                    EqPairTensorAxis::dense(0, a_relation_ring_stride, fold_weights.clone()),
                ],
            )?);
        }
    }
    Ok([d_tensors, b_tensors, a_tensors])
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

pub(super) fn role_tensors_are_aligned<E: Field>(
    tensors: &[EqPairTensorFamily<E>],
    ratio: usize,
) -> bool {
    ratio.is_power_of_two()
        && tensors.iter().all(|tensor| {
            tensor.right_offset.is_multiple_of(ratio)
                && tensor
                    .axes
                    .iter()
                    .all(|axis| axis.right_stride.is_multiple_of(ratio))
        })
}

pub(super) fn factor_aligned_role_tensors<E: Field>(
    tensors: &mut [EqPairTensorFamily<E>],
    ratio: usize,
) -> Result<(), AkitaError> {
    if ratio <= 1 || !role_tensors_are_aligned(tensors, ratio) {
        return Err(AkitaError::InvalidSetup(
            "setup role tensors are not aligned to their native lane count".into(),
        ));
    }
    for tensor in tensors {
        tensor.right_offset /= ratio;
        for axis in &mut tensor.axes {
            axis.right_stride /= ratio;
        }
    }
    Ok(())
}

fn role_projection_evaluation<E: Field>(
    alpha: E,
    base_ring_dim: usize,
    low_point: &[E],
) -> Result<E, AkitaError> {
    let base_ring_dim = u64::try_from(base_ring_dim).map_err(|_| {
        AkitaError::InvalidSetup("setup base ring dimension does not fit u64".into())
    })?;
    Ok(evaluate_power_sequence_mle(
        field_pow(alpha, base_ring_dim),
        low_point,
    ))
}

fn project_role_tensors<E: Field>(
    tensors: &[EqPairTensorFamily<E>],
    ratio: usize,
    alpha: E,
    base_ring_dim: usize,
) -> Result<Vec<EqPairTensorFamily<E>>, AkitaError> {
    if ratio <= 1 || !ratio.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "setup projection ratio must be a power of two greater than one".into(),
        ));
    }
    let alpha_powers = scalar_powers_with_stride(alpha, base_ring_dim, ratio)?;
    tensors
        .iter()
        .map(|tensor| {
            let mut axes = tensor.axes.clone();
            axes.push(EqPairTensorAxis::dense(0, 1, alpha_powers.clone()));
            EqPairTensorFamily::new(tensor.left_offset, tensor.right_offset, tensor.scalar, axes)
        })
        .collect()
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

#[cfg(test)]
mod projection_tests {
    use super::*;
    use akita_algebra::offset_eq::{eq_eval_at_index, OffsetEqWindow};
    use jolt_field::{One, Prime128OffsetA7F7, Ring};

    type F = Prime128OffsetA7F7;

    #[test]
    fn role_projection_preserves_unaligned_global_relation_lanes() {
        let alpha = F::from_u64(7);
        let ratio = 4;
        let base_ring_dim = 32;
        let relation_point = (0..4)
            .map(|index| F::from_u64(11 + index as u64))
            .collect::<Vec<_>>();
        let setup_point = (0..3)
            .map(|index| F::from_u64(21 + index as u64))
            .collect::<Vec<_>>();
        let family =
            EqPairTensorFamily::new(0, 2, F::one(), vec![EqPairTensorAxis::unit(2, 1, ratio)])
                .unwrap();
        let powers = scalar_powers_with_stride(alpha, base_ring_dim, ratio).unwrap();

        let relation_projected =
            project_role_tensors(std::slice::from_ref(&family), ratio, alpha, base_ring_dim)
                .unwrap();
        let equality = OffsetEqWindow::new(&relation_point).unwrap();
        let materialized = materialize_eq_tensor_left(&equality, &relation_projected, 2).unwrap();
        for (column, &actual) in materialized.iter().enumerate().take(2) {
            let expected = (0..ratio)
                .map(|lane| {
                    powers[lane] * eq_eval_at_index(&relation_point, 2 + ratio * column + lane)
                })
                .sum::<F>();
            assert_eq!(actual, expected);
        }

        let projected = project_role_tensors(&[family], ratio, alpha, base_ring_dim).unwrap();
        let setup_projection = evaluate_power_sequence_mle(
            field_pow(alpha, base_ring_dim as u64),
            &setup_point[..ratio.trailing_zeros() as usize],
        );
        let got = setup_projection
            * eval_boolean_pair_tensor_families::<_, false, false>(
                &setup_point[ratio.trailing_zeros() as usize..],
                &relation_point,
                &projected,
            )
            .unwrap();
        let expected = (0..2)
            .flat_map(|column| {
                let powers = &powers;
                let setup_point = &setup_point;
                let relation_point = &relation_point;
                (0..ratio).flat_map(move |setup_lane| {
                    (0..ratio).map(move |relation_lane| {
                        powers[setup_lane]
                            * powers[relation_lane]
                            * eq_eval_at_index(setup_point, ratio * column + setup_lane)
                            * eq_eval_at_index(relation_point, 2 + ratio * column + relation_lane)
                    })
                })
            })
            .sum::<F>();
        assert_eq!(got, expected);
    }
}
