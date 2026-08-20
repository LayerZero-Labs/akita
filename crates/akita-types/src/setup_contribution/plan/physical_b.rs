use super::*;
use akita_algebra::offset_eq::{EqPairTensorAxis, EqPairTensorFamily};

pub(super) fn build_physical_b_weight_segments<E: FieldCore>(
    geometry: &crate::CommitmentSliceGeometry,
    physical_rows: usize,
    logical_row_weights: &[E],
) -> Result<Vec<PhysicalBWeightSegment<E>>, AkitaError> {
    let logical_rows = geometry.logical_output_rows(physical_rows)?;
    if logical_row_weights.len() != logical_rows {
        return Err(AkitaError::InvalidSize {
            expected: logical_rows,
            actual: logical_row_weights.len(),
        });
    }
    let block_width = geometry.ring_elements_per_block_per_polynomial();
    let physical_polynomial_stride = geometry
        .max_blocks_per_slice()
        .checked_mul(block_width)
        .ok_or_else(|| AkitaError::InvalidSetup("physical B stride overflow".into()))?;
    let mut width_boundaries = geometry
        .block_ranges()
        .iter()
        .map(|range| {
            range
                .len()
                .checked_mul(block_width)
                .ok_or_else(|| AkitaError::InvalidSetup("B slice width overflow".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    width_boundaries.push(0);
    width_boundaries.sort_unstable();
    width_boundaries.dedup();

    let mut segments = Vec::new();
    for physical_row in 0..physical_rows {
        let physical_row_start = physical_row
            .checked_mul(geometry.physical_input_width())
            .ok_or_else(|| AkitaError::InvalidSetup("physical B row overflow".into()))?;
        for polynomial in 0..geometry.num_polynomials() {
            let physical_polynomial_start = physical_row_start
                .checked_add(
                    polynomial
                        .checked_mul(physical_polynomial_stride)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("physical B polynomial offset overflow".into())
                        })?,
                )
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("physical B polynomial extent overflow".into())
                })?;
            for boundary in width_boundaries.windows(2) {
                let [start, end] = boundary else {
                    return Err(AkitaError::InvalidSetup(
                        "physical B width boundary is malformed".into(),
                    ));
                };
                if start == end {
                    continue;
                }
                let mut terms = Vec::new();
                for (slice_index, block_range) in geometry.block_ranges().iter().enumerate() {
                    let slice_width = block_range
                        .len()
                        .checked_mul(block_width)
                        .ok_or_else(|| AkitaError::InvalidSetup("B slice width overflow".into()))?;
                    if slice_width < *end {
                        continue;
                    }
                    let logical_row =
                        geometry.logical_row_index(slice_index, physical_row, physical_rows)?;
                    let row_weight = *logical_row_weights
                        .get(logical_row)
                        .ok_or(AkitaError::InvalidProof)?;
                    let logical_start = polynomial
                        .checked_mul(geometry.num_live_blocks())
                        .and_then(|base| base.checked_add(block_range.start))
                        .and_then(|block| block.checked_mul(block_width))
                        .and_then(|base| base.checked_add(*start))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("logical B column offset overflow".into())
                        })?;
                    terms.push(PhysicalBWeightTerm {
                        logical_start,
                        row_weight,
                    });
                }
                if terms.is_empty() {
                    return Err(AkitaError::InvalidSetup(
                        "physical B segment has no logical source".into(),
                    ));
                }
                segments.push(PhysicalBWeightSegment {
                    physical_start: physical_polynomial_start.checked_add(*start).ok_or_else(
                        || AkitaError::InvalidSetup("physical B segment offset overflow".into()),
                    )?,
                    len: end - start,
                    terms: terms.into(),
                });
            }
        }
    }
    Ok(segments)
}

impl<E: FieldCore> PhysicalBSetupPlan<E> {
    /// Contract logical slice-major B row and column weights onto the one
    /// physical B matrix. The logical columns remain polynomial-major, while
    /// each slice is padded independently to the physical matrix width.
    pub(crate) fn contract_logical_column_weights(
        &self,
        logical_column_weights: &[E],
    ) -> Result<Vec<E>, AkitaError> {
        if logical_column_weights.len() != self.geometry().logical_input_width() {
            return Err(AkitaError::InvalidSize {
                expected: self.geometry().logical_input_width(),
                actual: logical_column_weights.len(),
            });
        }
        let mut physical = vec![E::zero(); self.physical_footprint()?];
        for segment in self.weight_segments() {
            let end = segment
                .physical_start
                .checked_add(segment.len)
                .ok_or_else(|| AkitaError::InvalidSetup("physical B segment overflow".into()))?;
            let target = physical
                .get_mut(segment.physical_start..end)
                .ok_or(AkitaError::InvalidProof)?;
            for (offset, target) in target.iter_mut().enumerate() {
                for term in segment.terms.iter() {
                    let logical = term
                        .logical_start
                        .checked_add(offset)
                        .and_then(|index| logical_column_weights.get(index))
                        .copied()
                        .ok_or(AkitaError::InvalidProof)?;
                    *target += term.row_weight * logical;
                }
            }
        }
        Ok(physical)
    }
}

pub(super) fn build_group_b_setup_tensors<E: FieldCore>(
    relation_geometry: RelationAddressGeometry,
    group: &SetupContributionGroupPlan<E>,
    witness_layout: &WitnessLayout,
) -> Result<Vec<EqPairTensorFamily<E>>, AkitaError> {
    let physical_b = &group.physical_b;
    let geometry = physical_b.geometry();
    let (b_subcolumns, _) = SetupProjectionGeometry::native_role_subcolumn_counts(group.role_dims)?;
    let source_lanes = group.a_relation_ratio;
    let a_row_setup_stride = checked::product([group.depth_commit, b_subcolumns])
        .ok_or_else(|| AkitaError::InvalidSetup("setup B A-row stride overflow".into()))?;
    let block_setup_stride = checked::product([group.n_a, a_row_setup_stride])
        .ok_or_else(|| AkitaError::InvalidSetup("setup B block stride overflow".into()))?;
    let a_row_relation_stride = checked::product([group.depth_commit, source_lanes])
        .ok_or_else(|| AkitaError::InvalidSetup("setup B relation A-row stride overflow".into()))?;
    let subcolumn_relation_stride = checked::product([group.depth_commit, group.b_relation_ratio])
        .ok_or_else(|| {
            AkitaError::InvalidSetup("setup B subcolumn relation stride overflow".into())
        })?;
    let block_relation_stride = checked::product([group.n_a, a_row_relation_stride])
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
    for unit in witness_layout.units_for_group(group.group_id)? {
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
            for claim in 0..group.num_claims {
                let setup_column = claim
                    .checked_mul(claim_setup_stride)
                    .and_then(|base| {
                        local_block_start
                            .checked_mul(block_setup_stride)
                            .and_then(|offset| base.checked_add(offset))
                    })
                    .ok_or_else(|| AkitaError::InvalidSetup("setup B address overflow".into()))?;
                let witness_coefficient = unit.t_coefficient_index(
                    group.role_dims.d_a(),
                    group.role_dims.d_b(),
                    group.num_claims,
                    group.n_a,
                    group.depth_commit,
                    claim,
                    intersection_start,
                    0,
                    0,
                    0,
                    0,
                )?;
                let relation_lane_start = divide_aligned(
                    witness_coefficient,
                    relation_geometry.relation_coefficient_block_len(),
                    "setup B coefficient address is not relation-block aligned",
                )?;
                let row_weights = slice_row_weights.get(slice_index).ok_or_else(|| {
                    AkitaError::InvalidSetup("B slice row weights are missing".into())
                })?;
                tensors.push(EqPairTensorFamily::new(
                    setup_column,
                    relation_lane_start,
                    E::one(),
                    vec![
                        EqPairTensorAxis::unit(group.depth_commit, 1, group.b_relation_ratio),
                        EqPairTensorAxis::unit(
                            b_subcolumns,
                            group.depth_commit,
                            subcolumn_relation_stride,
                        ),
                        EqPairTensorAxis::unit(
                            group.n_a,
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
