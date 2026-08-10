use super::*;
use akita_algebra::offset_eq::{EqPairTensorAxis, EqPairTensorFamily};

pub(super) fn build_group_b_setup_tensors<E: FieldCore>(
    relation_geometry: RelationAddressGeometry,
    group: &SetupContributionGroupPlan<E>,
    witness_layout: &WitnessLayout,
) -> Result<Vec<EqPairTensorFamily<E>>, AkitaError> {
    let physical_b = &group.physical_b;
    let geometry = physical_b.geometry();
    let (b_subcolumns, _) = SetupProjectionGeometry::native_role_subcolumn_counts(group.role_dims)?;
    let source_lanes = group.a_ratio;
    let a_row_setup_stride = checked_mul(
        group.depth_commit,
        b_subcolumns,
        "setup B A-row stride overflow",
    )?;
    let block_setup_stride = checked_mul(
        group.n_a,
        a_row_setup_stride,
        "setup B block stride overflow",
    )?;
    let a_row_relation_stride = checked_mul(
        group.depth_commit,
        source_lanes,
        "setup B relation A-row stride overflow",
    )?;
    let subcolumn_relation_stride = checked_mul(
        group.depth_commit,
        group.b_ratio,
        "setup B subcolumn relation stride overflow",
    )?;
    let block_relation_stride = checked_mul(
        group.n_a,
        a_row_relation_stride,
        "setup B relation block stride overflow",
    )?;
    let claim_setup_stride = checked_mul(
        geometry.max_blocks_per_slice(),
        block_setup_stride,
        "setup B claim stride overflow",
    )?;
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
                        EqPairTensorAxis::unit(group.depth_commit, 1, group.b_ratio),
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
