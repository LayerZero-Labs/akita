use akita_algebra::offset_eq::OffsetEqWindow;
use akita_field::parallel::*;
use akita_field::{AkitaError, FieldCore, MulBase};

use crate::WitnessLayout;

/// Canonical D-role column weights in `(claim, block, opening_digit)` order.
pub(crate) fn setup_e_col_weights<E: FieldCore>(
    layout: &WitnessLayout,
    opening_source_len: usize,
    group_id: usize,
    live_fold_count: usize,
    num_claims: usize,
    depth_open: usize,
    eq_window: &OffsetEqWindow<E>,
) -> Result<Vec<E>, AkitaError> {
    let e_cols = checked_mul3(
        num_claims,
        live_fold_count,
        depth_open,
        "setup D columns overflow",
    )?;
    cfg_into_iter!(0..e_cols)
        .map(|local_col| {
            let digit = local_col % depth_open;
            let block_claim = local_col / depth_open;
            let block = block_claim % live_fold_count;
            let claim = block_claim / live_fold_count;
            let unit = layout.unit_for_fold(group_id, block)?;
            let witness_index =
                layout.e_index(unit, num_claims, depth_open, claim, block, digit)?;
            let opening_index =
                crate::checked_opening_source_index(opening_source_len, witness_index)?;
            Ok(eq_window.eval(opening_index))
        })
        .collect()
}

/// Canonical B-role column weights in `(claim, block, A_row, opening_digit)` order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_t_col_weights<E: FieldCore>(
    layout: &WitnessLayout,
    opening_source_len: usize,
    group_id: usize,
    live_fold_count: usize,
    depth_open: usize,
    n_a: usize,
    cols_per_vector: usize,
    num_vectors: usize,
    active_vectors: usize,
    vector_base: usize,
    eq_window: &OffsetEqWindow<E>,
) -> Result<Vec<E>, AkitaError> {
    let expected_cols_per_vector = checked_mul3(
        live_fold_count,
        n_a,
        depth_open,
        "setup B columns per vector overflow",
    )?;
    if cols_per_vector != expected_cols_per_vector {
        return Err(AkitaError::InvalidSize {
            expected: expected_cols_per_vector,
            actual: cols_per_vector,
        });
    }
    let t_cols = num_vectors
        .checked_mul(cols_per_vector)
        .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".into()))?;
    cfg_into_iter!(0..t_cols)
        .map(|local_col| {
            let vector = local_col / cols_per_vector;
            if vector >= active_vectors {
                return Ok(E::zero());
            }
            let digit = local_col % depth_open;
            let rest = local_col / depth_open;
            let a_row = rest % n_a;
            let block = (rest / n_a) % live_fold_count;
            let claim = vector_base
                .checked_add(vector)
                .ok_or_else(|| AkitaError::InvalidSetup("setup B claim index overflow".into()))?;
            let unit = layout.unit_for_fold(group_id, block)?;
            let witness_index = layout.t_index(
                unit,
                num_vectors,
                n_a,
                depth_open,
                claim,
                block,
                a_row,
                digit,
            )?;
            let opening_index =
                crate::checked_opening_source_index(opening_source_len, witness_index)?;
            Ok(eq_window.eval(opening_index))
        })
        .collect()
}

/// Canonical A-role column weights in `(position, commit_digit)` order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_z_col_weights<F, E>(
    layout: &WitnessLayout,
    opening_source_len: usize,
    group_id: usize,
    fold_position_count: usize,
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
            "setup A weights have malformed ownership or fold geometry".into(),
        ));
    }
    let z_cols = fold_position_count
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
                    let witness_index = layout.z_index(
                        unit,
                        fold_position_count,
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
