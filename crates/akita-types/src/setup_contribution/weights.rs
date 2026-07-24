#![allow(dead_code)]

use akita_algebra::offset_eq::OffsetEqWindow;
use akita_field::parallel::*;
use akita_field::{AkitaError, FieldCore, MulBase};

use crate::WitnessLayout;

/// Canonical D-role column weights in `(claim, block, opening_digit)` order.
pub(crate) fn setup_e_col_weights<E: FieldCore>(
    layout: &WitnessLayout,
    opening_source_len: usize,
    group_id: usize,
    num_live_blocks: usize,
    num_claims: usize,
    depth_open: usize,
    eq_window: &OffsetEqWindow<E>,
) -> Result<Vec<E>, AkitaError> {
    let e_cols = checked_mul3(
        num_claims,
        num_live_blocks,
        depth_open,
        "setup D columns overflow",
    )?;
    let units = layout.units_for_group(group_id)?;
    let mut weights = vec![E::zero(); e_cols];
    for claim in 0..num_claims {
        for unit in &units {
            let unit_width = unit
                .num_live_blocks()
                .checked_mul(depth_open)
                .ok_or_else(|| AkitaError::InvalidSetup("witness E unit width overflow".into()))?;
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
                .ok_or_else(|| AkitaError::InvalidSetup("setup D destination overflow".into()))?;
            let destination_start = block_claim
                .checked_mul(depth_open)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D destination overflow".into()))?;
            let destination_end = destination_start
                .checked_add(unit_width)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D destination overflow".into()))?;
            let destination = weights
                .get_mut(destination_start..destination_end)
                .ok_or(AkitaError::InvalidProof)?;
            eq_window.fill_interval(source_start, destination)?;
        }
    }
    Ok(weights)
}

/// Canonical B-role column weights in `(claim, block, A_row, opening_digit)` order.
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
) -> Result<Vec<E>, AkitaError> {
    let vector_width = checked_mul3(
        num_live_blocks,
        n_a,
        depth_open,
        "setup B columns per vector overflow",
    )?;
    let num_t_columns = num_claims
        .checked_mul(vector_width)
        .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".into()))?;
    let units = layout.units_for_group(group_id)?;
    let mut weights = vec![E::zero(); num_t_columns];
    for claim in 0..num_claims {
        for unit in &units {
            let unit_width = unit
                .num_live_blocks()
                .checked_mul(n_a)
                .and_then(|width| width.checked_mul(depth_open))
                .ok_or_else(|| AkitaError::InvalidSetup("witness T unit width overflow".into()))?;
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
                .ok_or_else(|| AkitaError::InvalidSetup("setup B destination overflow".into()))?;
            let destination_start = block_claim
                .checked_mul(n_a)
                .and_then(|base| base.checked_mul(depth_open))
                .ok_or_else(|| AkitaError::InvalidSetup("setup B destination overflow".into()))?;
            let destination_end = destination_start
                .checked_add(unit_width)
                .ok_or_else(|| AkitaError::InvalidSetup("setup B destination overflow".into()))?;
            let destination = weights
                .get_mut(destination_start..destination_end)
                .ok_or(AkitaError::InvalidProof)?;
            eq_window.fill_interval(source_start, destination)?;
        }
    }
    Ok(weights)
}

/// Canonical A-role column weights in `(position, commit_digit)` order.
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
                    let opening_index =
                        crate::checked_opening_source_index(opening_source_len, witness_index)?;
                    weight -= eq_window.eval(opening_index).mul_base(fold);
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
