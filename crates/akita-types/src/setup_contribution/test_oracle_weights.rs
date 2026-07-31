//! Legacy column-weight formulas retained only as independent test oracles.

use akita_algebra::offset_eq::OffsetEqWindow;
use akita_field::parallel::*;
use akita_field::{AkitaError, FieldCore, MulBase};

use crate::WitnessLayout;

/// Per-role relation lane geometry for building canonical setup column weights.
///
/// The relation lays the flat witness out in coefficient blocks of width
/// `base_ring_dim`. One inner (`d_a`) ring element therefore spans
/// `a_ratio = d_a / base_ring_dim` consecutive relation-lane addresses. A role
/// with dimension `d_role` covers those `a_ratio` lanes with
/// `role_subcolumns = a_ratio / (d_role / base_ring_dim)` distinct physical
/// columns, each carrying `role_lanes = d_role / base_ring_dim` α-weighted lanes
/// (`role_lane_alpha[l] = α^{base_ring_dim · l}`). The canonical relation-lane
/// address of column `witness_column`, subcolumn `s`, lane `l` is
/// `witness_column · a_ratio + s · role_lanes + l`.
///
/// The uniform-role case is `a_ratio = role_subcolumns = role_lanes = 1`; then
/// the address is just `witness_column` and the weight is a single `eq`
/// evaluation, byte-identical to the pre-existing fill-interval fast path.
#[derive(Clone, Copy)]
pub(crate) struct RoleLaneSpec<'a, E> {
    /// Total relation lanes per inner ring element (`d_a / base_ring_dim`).
    pub a_ratio: usize,
    /// Distinct physical subcolumns for this role (`a_ratio / (d_role/base)`).
    pub role_subcolumns: usize,
    /// α-weighted lanes carried by one physical column (`d_role / base`).
    pub role_lanes: usize,
    /// `role_lane_alpha[l] = α^{base_ring_dim · l}`, length `role_lanes`.
    pub role_lane_alpha: &'a [E],
}

impl<E: FieldCore> RoleLaneSpec<'_, E> {
    /// Uniform roles: no lane expansion, so the fast fill-interval path applies.
    fn is_uniform(&self) -> bool {
        self.a_ratio == 1
    }
}

/// Canonical relation-lane weight for one physical setup column.
///
/// `Σ_{l<role_lanes} eq(first_lane + l) ·
/// role_lane_alpha[l]`.
#[inline]
fn canonical_lane_weight<E: FieldCore>(
    eq_window: &OffsetEqWindow<E>,
    opening_source_len: usize,
    first_lane: usize,
    spec: &RoleLaneSpec<'_, E>,
) -> Result<E, AkitaError> {
    let lane_bound = opening_source_len
        .checked_mul(spec.a_ratio)
        .ok_or_else(|| AkitaError::InvalidSetup("relation lane bound overflow".into()))?;
    if first_lane >= lane_bound {
        return Err(AkitaError::InvalidInput(
            "relation lane address exceeds opening source".into(),
        ));
    }
    let mut weight = E::zero();
    for (lane, &alpha) in spec.role_lane_alpha.iter().enumerate() {
        let index = first_lane
            .checked_add(lane)
            .ok_or_else(|| AkitaError::InvalidSetup("relation lane address overflow".into()))?;
        weight += eq_window.eval(index) * alpha;
    }
    Ok(weight)
}

/// Canonical D-role column weights in `(claim, block, subcolumn, opening_digit)`
/// order (subcolumn present only for genuinely per-role commitments).
#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_e_col_weights<E: FieldCore>(
    layout: &WitnessLayout,
    opening_source_len: usize,
    group_id: usize,
    num_live_blocks: usize,
    num_claims: usize,
    depth_open: usize,
    eq_window: &OffsetEqWindow<E>,
    spec: &RoleLaneSpec<'_, E>,
) -> Result<Vec<E>, AkitaError> {
    let e_cols = checked_mul4(
        num_claims,
        num_live_blocks,
        spec.role_subcolumns,
        depth_open,
        "setup D columns overflow",
    )?;
    let units = layout.units_for_group(group_id)?;
    if spec.is_uniform() {
        // Uniform-role fast path: contiguous `eq` fill (unchanged).
        let mut weights = vec![E::zero(); e_cols];
        for claim in 0..num_claims {
            for unit in &units {
                let unit_width =
                    unit.num_live_blocks()
                        .checked_mul(depth_open)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("witness E unit width overflow".into())
                        })?;
                let expected = num_claims
                    .checked_mul(unit_width)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness E shape overflow".into()))?;
                let source_range = unit.e_range();
                if source_range.len() != expected {
                    return Err(AkitaError::InvalidSetup(
                        "witness E shape disagrees with resolved range".into(),
                    ));
                }
                let source_start = source_range
                    .start
                    .checked_add(claim.checked_mul(unit_width).ok_or_else(|| {
                        AkitaError::InvalidSetup("witness E claim offset overflow".into())
                    })?)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness E source overflow".into()))?;
                let source_end = source_start
                    .checked_add(unit_width)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness E source overflow".into()))?;
                if source_end > source_range.end || source_end > opening_source_len {
                    return Err(AkitaError::InvalidInput(
                        "physical opening interval out of range".into(),
                    ));
                }
                let block_claim = claim
                    .checked_mul(num_live_blocks)
                    .and_then(|base| base.checked_add(unit.global_block_start()))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("setup D destination overflow".into())
                    })?;
                let destination_start = block_claim.checked_mul(depth_open).ok_or_else(|| {
                    AkitaError::InvalidSetup("setup D destination overflow".into())
                })?;
                let destination_end =
                    destination_start.checked_add(unit_width).ok_or_else(|| {
                        AkitaError::InvalidSetup("setup D destination overflow".into())
                    })?;
                let destination = weights
                    .get_mut(destination_start..destination_end)
                    .ok_or(AkitaError::InvalidProof)?;
                eq_window.fill_interval(source_start, destination)?;
            }
        }
        return Ok(weights);
    }
    // Per-role: parallel map over the expanded column index. Column order is
    // [claim][block][subcolumn][digit]; each column reads one canonical
    // lane-summed `eq` (subcolumn selects which relation lane the D column fills).
    cfg_into_iter!(0..e_cols)
        .map(|col| -> Result<E, AkitaError> {
            let digit = col % depth_open;
            let t = col / depth_open;
            let subcolumn = t % spec.role_subcolumns;
            let block_claim = t / spec.role_subcolumns;
            let global_block = block_claim % num_live_blocks;
            let claim = block_claim / num_live_blocks;
            let Some(unit) = units.iter().copied().find(|u| {
                let start = u.global_block_start();
                global_block >= start && global_block - start < u.num_live_blocks()
            }) else {
                return Ok(E::zero());
            };
            let block_in_unit = global_block - unit.global_block_start();
            let semantic = claim * unit.num_live_blocks() + block_in_unit;
            let carrier_subcolumns = spec.a_ratio / spec.role_lanes;
            let first_lane = unit
                .e_range()
                .start
                .checked_mul(spec.a_ratio)
                .and_then(|base| {
                    (semantic * carrier_subcolumns + subcolumn)
                        .checked_mul(depth_open)
                        .and_then(|index| index.checked_add(digit))
                        .and_then(|index| index.checked_mul(spec.role_lanes))
                        .and_then(|offset| base.checked_add(offset))
                })
                .ok_or_else(|| AkitaError::InvalidSetup("setup D source overflow".into()))?;
            canonical_lane_weight(eq_window, opening_source_len, first_lane, spec)
        })
        .collect()
}

/// Canonical B-role column weights in `(claim, block, A_row, subcolumn,
/// opening_digit)` order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_t_col_weights<E: FieldCore>(
    layout: &WitnessLayout,
    opening_source_len: usize,
    group_id: usize,
    num_live_blocks: usize,
    depth_open: usize,
    n_a: usize,
    num_claims: usize,
    eq_window: &OffsetEqWindow<E>,
    spec: &RoleLaneSpec<'_, E>,
) -> Result<Vec<E>, AkitaError> {
    let vector_width = checked_mul3(
        num_live_blocks,
        n_a,
        depth_open,
        "setup B columns per vector overflow",
    )?;
    let expanded_vector_width = vector_width
        .checked_mul(spec.role_subcolumns)
        .ok_or_else(|| AkitaError::InvalidSetup("setup B subcolumn width overflow".into()))?;
    let num_t_columns = num_claims
        .checked_mul(expanded_vector_width)
        .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".into()))?;
    let units = layout.units_for_group(group_id)?;
    if spec.is_uniform() {
        let mut weights = vec![E::zero(); num_t_columns];
        for claim in 0..num_claims {
            for unit in &units {
                let unit_width = unit
                    .num_live_blocks()
                    .checked_mul(n_a)
                    .and_then(|width| width.checked_mul(depth_open))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("witness T unit width overflow".into())
                    })?;
                let expected = num_claims
                    .checked_mul(unit_width)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness T shape overflow".into()))?;
                let source_range = unit.t_range();
                if source_range.len() != expected {
                    return Err(AkitaError::InvalidSetup(
                        "witness T shape disagrees with resolved range".into(),
                    ));
                }
                let source_start = source_range
                    .start
                    .checked_add(claim.checked_mul(unit_width).ok_or_else(|| {
                        AkitaError::InvalidSetup("witness T claim offset overflow".into())
                    })?)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness T source overflow".into()))?;
                let source_end = source_start
                    .checked_add(unit_width)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness T source overflow".into()))?;
                if source_end > source_range.end || source_end > opening_source_len {
                    return Err(AkitaError::InvalidInput(
                        "physical opening interval out of range".into(),
                    ));
                }
                let block_claim = claim
                    .checked_mul(num_live_blocks)
                    .and_then(|base| base.checked_add(unit.global_block_start()))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("setup B destination overflow".into())
                    })?;
                let destination_start = block_claim
                    .checked_mul(n_a)
                    .and_then(|base| base.checked_mul(depth_open))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("setup B destination overflow".into())
                    })?;
                let destination_end =
                    destination_start.checked_add(unit_width).ok_or_else(|| {
                        AkitaError::InvalidSetup("setup B destination overflow".into())
                    })?;
                let destination = weights
                    .get_mut(destination_start..destination_end)
                    .ok_or(AkitaError::InvalidProof)?;
                eq_window.fill_interval(source_start, destination)?;
            }
        }
        return Ok(weights);
    }
    // Per-role: parallel map over `[claim][block][A row][subcolumn][digit]`.
    cfg_into_iter!(0..num_t_columns)
        .map(|col| -> Result<E, AkitaError> {
            let digit = col % depth_open;
            let s1 = col / depth_open;
            let subcolumn = s1 % spec.role_subcolumns;
            let s2 = s1 / spec.role_subcolumns;
            let a_row = s2 % n_a;
            let s3 = s2 / n_a;
            let global_block = s3 % num_live_blocks;
            let claim = s3 / num_live_blocks;
            let Some(unit) = units.iter().copied().find(|u| {
                let start = u.global_block_start();
                global_block >= start && global_block - start < u.num_live_blocks()
            }) else {
                return Ok(E::zero());
            };
            let block_in_unit = global_block - unit.global_block_start();
            let semantic = a_row + n_a * (block_in_unit + unit.num_live_blocks() * claim);
            let carrier_subcolumns = spec.a_ratio / spec.role_lanes;
            let first_lane = unit
                .t_range()
                .start
                .checked_mul(spec.a_ratio)
                .and_then(|base| {
                    (semantic * carrier_subcolumns + subcolumn)
                        .checked_mul(depth_open)
                        .and_then(|index| index.checked_add(digit))
                        .and_then(|index| index.checked_mul(spec.role_lanes))
                        .and_then(|offset| base.checked_add(offset))
                })
                .ok_or_else(|| AkitaError::InvalidSetup("setup B source overflow".into()))?;
            canonical_lane_weight(eq_window, opening_source_len, first_lane, spec)
        })
        .collect()
}

/// Canonical A-role column weights in `(position, commit_digit)` order.
///
/// The A role has `role_subcolumns = 1`; its `a_ratio` relation lanes are all
/// α-summed inside [`canonical_lane_weight`], so the column count is unchanged
/// and only the per-column value gains the lane sum.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_z_col_weights<F, E>(
    layout: &WitnessLayout,
    opening_source_len: usize,
    group_id: usize,
    num_positions_per_block: usize,
    depth_commit: usize,
    depth_fold: usize,
    eq_window: &OffsetEqWindow<E>,
    fold_gadget: &[F],
    spec: &RoleLaneSpec<'_, E>,
    z_weights: &mut [E],
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: MulBase<F>,
{
    let units = layout.units_for_group(group_id)?;
    if fold_gadget.len() < depth_fold {
        return Err(AkitaError::InvalidSetup(
            "setup A weights have malformed ownership or block geometry".into(),
        ));
    }
    let z_cols = num_positions_per_block
        .checked_mul(depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("setup A width overflow".into()))?;
    if z_weights.len() != z_cols {
        return Err(AkitaError::InvalidSize {
            expected: z_cols,
            actual: z_weights.len(),
        });
    }
    cfg_iter_mut!(z_weights)
        .enumerate()
        .try_for_each(|(column, dst)| {
            let position = column / depth_commit;
            let commit_digit = column % depth_commit;
            let mut weight = E::zero();
            for unit in &units {
                for (fold_digit, &fold) in fold_gadget.iter().enumerate().take(depth_fold) {
                    let witness_index = unit.z_index(
                        num_positions_per_block,
                        depth_commit,
                        depth_fold,
                        position,
                        commit_digit,
                        fold_digit,
                    )?;
                    if spec.is_uniform() {
                        let opening_index =
                            crate::checked_opening_source_index(opening_source_len, witness_index)?;
                        weight -= eq_window.eval(opening_index).mul_base(fold);
                    } else {
                        // Per-role: α-lane-summed canonical `eq` (subcolumn 0).
                        let lane_weight = canonical_lane_weight(
                            eq_window,
                            opening_source_len,
                            witness_index * spec.a_ratio,
                            spec,
                        )?;
                        weight -= lane_weight.mul_base(fold);
                    }
                }
            }
            *dst += weight;
            Ok(())
        })
}

fn checked_mul3(a: usize, b: usize, c: usize, context: &str) -> Result<usize, AkitaError> {
    a.checked_mul(b)
        .and_then(|n| n.checked_mul(c))
        .ok_or_else(|| AkitaError::InvalidSetup(context.into()))
}

fn checked_mul4(
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    context: &str,
) -> Result<usize, AkitaError> {
    a.checked_mul(b)
        .and_then(|n| n.checked_mul(c))
        .and_then(|n| n.checked_mul(d))
        .ok_or_else(|| AkitaError::InvalidSetup(context.into()))
}
