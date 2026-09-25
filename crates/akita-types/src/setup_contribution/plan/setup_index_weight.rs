use super::*;
use akita_algebra::fft::field_pow;
use akita_algebra::{
    offset_eq::{materialize_eq_tensor_left, EqPairTensorAxis, EqPairTensorFamily, OffsetEqWindow},
    ring::{evaluate_power_sequence_mle, scalar_powers_with_stride},
};

struct GroupSetupIndexWeights<E> {
    projection_scales: [Option<Vec<E>>; 3],
    column_weights: [Vec<E>; 3],
    physical_b_weights: Vec<E>,
}

impl<E: Field> SetupContributionPlan<E> {
    /// Materialize one role's relation-column weights at `alpha` from its
    /// canonical relation tensors.
    pub fn materialize_role_tensor_weights(
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

    /// Split the relation address into its low bridge coordinates (relation
    /// base to setup base) and the setup-base relation point.
    pub fn relation_base_bridge_split(&self) -> Result<(&[E], &[E]), AkitaError> {
        let bridge_bits = self.relation_base_bridge_ratio()?.trailing_zeros() as usize;
        self.relation_address
            .point()
            .split_at_checked(bridge_bits)
            .ok_or(AkitaError::InvalidProof)
    }

    /// Number of relation-base coefficient blocks in one setup base ring.
    pub fn relation_base_bridge_ratio(&self) -> Result<usize, AkitaError> {
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

pub(super) fn build_group_role_tensors<E: Field>(
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
            let d_relation_lane_start = checked::exact_div(
                d_witness_coefficient,
                relation_geometry.relation_coefficient_block_len(),
            )
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "setup D coefficient address is not relation-block aligned".into(),
                )
            })?;
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
                let b_relation_lane_start = checked::exact_div(
                    b_witness_coefficient,
                    relation_geometry.relation_coefficient_block_len(),
                )
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "setup B coefficient address is not relation-block aligned".into(),
                    )
                })?;
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
            let a_relation_lane_start = checked::exact_div(
                a_witness_coefficient,
                relation_geometry.relation_coefficient_block_len(),
            )
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "setup A coefficient address is not relation-block aligned".into(),
                )
            })?;
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

/// Whether every relation offset and stride of `tensors` is `ratio`-aligned.
pub fn role_tensors_are_aligned<E: Field>(tensors: &[EqPairTensorFamily<E>], ratio: usize) -> bool {
    ratio.is_power_of_two()
        && tensors.iter().all(|tensor| {
            tensor.right_offset.is_multiple_of(ratio)
                && tensor
                    .axes
                    .iter()
                    .all(|axis| axis.right_stride.is_multiple_of(ratio))
        })
}

/// Divide the relation offsets and strides of `ratio`-aligned tensors by
/// `ratio`.
pub fn factor_aligned_role_tensors<E: Field>(
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

/// Multilinear extension of the `alpha^base_ring_dim` power sequence at
/// `low_point`.
pub fn role_projection_evaluation<E: Field>(
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

/// Append the `alpha^base_ring_dim` lane-projection axis to every tensor.
pub fn project_role_tensors<E: Field>(
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

#[cfg(test)]
mod projection_tests {
    use super::*;
    use akita_algebra::offset_eq::{
        eq_eval_at_index, eval_boolean_pair_tensor_families, OffsetEqWindow,
    };
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
