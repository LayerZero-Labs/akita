//! The chunk-major witness body, shared by materialized layouts and measured
//! lengths.

use std::ops::Range;

use akita_error::AkitaError;
use jolt_algebra::checked;

use super::{
    tail, witness_range, witness_unit_lengths, RelationQuotientPlan, WitnessLayout,
    WitnessUnitLayout, MAX_WITNESS_CHUNKS,
};
use crate::{CommittedGroupParams, OpeningClaimsLayout, RelationWitnessGeometry};

impl WitnessLayout {
    /// The exact live length of [`Self::new`]'s layout of a level, computed
    /// without materializing its address ranges.
    ///
    /// The planner's counterpart of [`Self::new`] for its hot paths. The
    /// caller supplies the extension degree, not a prebuilt geometry: this
    /// function owns its one construction and validation of the level's
    /// relation layout.
    pub fn measure_live_coeff_len(
        lp: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        extension_degree: usize,
        num_chunks: usize,
        quotient_plan: RelationQuotientPlan,
    ) -> Result<usize, AkitaError> {
        check_layout_shape(opening_batch, num_chunks)?;
        let relation_geometry =
            RelationWitnessGeometry::for_level(lp, opening_batch, extension_degree)?;
        let body = resolve(
            lp,
            opening_batch,
            &relation_geometry,
            num_chunks,
            |_| {},
            |_| {},
        )?;
        tail::measure(
            lp,
            &relation_geometry,
            opening_batch.num_groups(),
            body.successor_a_alignment,
            body.end,
            quotient_plan,
        )
    }
}

/// Where a level's chunk-major body ends, and the alignment its tail suffix
/// keeps.
pub(super) struct ResolvedBody {
    pub(super) end: usize,
    pub(super) successor_a_alignment: usize,
}

/// Walk a level's chunk-major body, handing each unit and alignment gap to
/// the caller: the one body layout behind [`WitnessLayout::new`] and
/// [`WitnessLayout::measure_live_coeff_len`]. `relation_geometry` must be
/// `RelationWitnessGeometry::for_level` of `lp` and `opening_batch`.
pub(super) fn resolve(
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    relation_geometry: &RelationWitnessGeometry,
    num_chunks: usize,
    mut place_unit: impl FnMut(WitnessUnitLayout),
    mut place_alignment: impl FnMut(Range<usize>),
) -> Result<ResolvedBody, AkitaError> {
    let num_groups = opening_batch.num_groups();
    let relation_group_order = opening_batch.root_group_order()?;
    let final_group_index = opening_batch.root_final_group_index()?;

    let mut cursor = 0usize;
    let group_geometry = relation_group_order
        .iter()
        .map(|&group_index| {
            // The caller's relation geometry is `for_level` of this batch,
            // which validated it completely. Resolve per-group views
            // directly.
            let (params, role_dims) = if group_index == final_group_index {
                (lp.own_group(), lp.role_dims())
            } else {
                let params = lp.preceding_group_params(group_index).ok_or_else(|| {
                    AkitaError::Internal(
                        "validated witness layout is missing a preceding group".into(),
                    )
                })?;
                (params, params.role_dims(lp.open().matrix.ring_dimension()))
            };
            role_dims.validate_role_projection()?;
            let group = opening_batch.group_layout(group_index)?;
            let opening_geometry = relation_geometry.group_opening_geometry(group_index)?;
            let num_claims = group.num_polynomials();
            let depth_witness = params.num_digits_inner();
            let depth_commit = params.num_digits_outer();
            let depth_open = params.num_digits_open();
            let depth_fold = params.num_digits_fold();
            if num_claims == 0
                || params.num_live_blocks() == 0
                || params.num_positions_per_block() == 0
                || depth_open == 0
                || depth_witness == 0
                || depth_commit == 0
                || depth_fold == 0
                || params.a_rows_len() == 0
            {
                return Err(AkitaError::InvalidSetup(
                    "witness group has malformed dimensions".into(),
                ));
            }
            let chunk_block_ranges = lp.witness_block_ranges(group_index, num_chunks)?;
            Ok((
                group_index,
                params,
                role_dims,
                opening_geometry,
                num_claims,
                chunk_block_ranges,
            ))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    // Chunk is the outer physical key. Within each chunk, groups are laid
    // out in relation order, and each unit is `[Z | E | T]`.
    for chunk_index in 0..num_chunks {
        for &(
            group_index,
            params,
            role_dims,
            opening_geometry,
            num_claims,
            ref chunk_block_ranges,
        ) in &group_geometry
        {
            let global_block_range = chunk_block_ranges
                .get(chunk_index)
                .ok_or_else(|| AkitaError::InvalidSetup("witness chunk is missing".into()))?
                .clone();
            let global_block_start = global_block_range.start;
            let chunk_num_live_blocks = global_block_range.len();
            let (z_len, e_len, t_len) = witness_unit_lengths(
                params,
                role_dims,
                opening_geometry,
                num_claims,
                chunk_num_live_blocks,
            )?;
            let z_range = witness_range(cursor, z_len, "witness Z range overflow")?;
            let e_range = witness_range(z_range.end, e_len, "witness E range overflow")?;
            let t_range = witness_range(e_range.end, t_len, "witness T range overflow")?;
            cursor = t_range.end;
            place_unit(WitnessUnitLayout {
                group_index,
                chunk_index,
                global_block_start,
                num_live_blocks: chunk_num_live_blocks,
                z_range,
                e_range,
                e_geometry: opening_geometry,
                t_range,
            });
        }
        let aligned = checked::align_up(
            cursor,
            if num_chunks > 1 {
                lp.successor_block_len.unwrap_or(1)
            } else {
                1
            },
        )
        .ok_or_else(|| AkitaError::InvalidSetup("witness chunk alignment overflow".into()))?;
        if aligned != cursor {
            place_alignment(cursor..aligned);
        }
        cursor = aligned;
    }
    let successor_a_alignment = if num_groups > 1 {
        group_geometry
            .iter()
            .map(|(_, _, role_dims, _, _, _)| role_dims.d_a())
            .max()
            .ok_or_else(|| AkitaError::InvalidSetup("witness groups are empty".into()))?
    } else {
        relation_geometry
            .rhs_layout()
            .relation_coefficient_block_len()?
    };
    Ok(ResolvedBody {
        end: cursor,
        successor_a_alignment,
    })
}

/// The group and chunk counts every witness layout requires.
pub(super) fn check_layout_shape(
    opening_batch: &OpeningClaimsLayout,
    num_chunks: usize,
) -> Result<(), AkitaError> {
    if opening_batch.num_groups() == 0 || num_chunks == 0 {
        return Err(AkitaError::InvalidSetup(
            "witness layout requires non-empty groups and chunks".into(),
        ));
    }
    if num_chunks > MAX_WITNESS_CHUNKS {
        return Err(AkitaError::InvalidSetup(
            "witness chunk count exceeds verifier cap".into(),
        ));
    }
    Ok(())
}
