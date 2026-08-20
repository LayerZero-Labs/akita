use std::ops::Range;

use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_error::AkitaError;
use akita_field::parallel::*;
use akita_field::{FieldCore, MulBaseUnreduced};

/// One family of setup-matrix rows read as per-column ring slices borrowed
/// from the materialized store.
pub(super) struct SetupRows<'a, F: FieldCore> {
    pub(super) rows: Vec<&'a [F]>,
    pub(super) ring_d: usize,
}

impl<F: FieldCore> SetupRows<'_, F> {
    pub(super) fn ring_slice(&self, row: usize, col: usize) -> Result<&[F], AkitaError> {
        self.rows
            .get(row)
            .and_then(|row| row.get(col * self.ring_d..(col + 1) * self.ring_d))
            .ok_or(AkitaError::InvalidProof)
    }
}

pub(super) fn evaluate_setup_columns<F, E>(
    family: &SetupRows<'_, F>,
    columns: Range<usize>,
    row_weights: &[(usize, Vec<E>)],
    batch_count: usize,
    alpha_powers: &[E],
) -> Result<SetupColumnEvaluations<E>, AkitaError>
where
    F: FieldCore,
    E: FieldCore + MulBaseUnreduced<F>,
{
    if batch_count == 0
        || row_weights
            .iter()
            .any(|(_, weights)| weights.len() != batch_count)
    {
        return Err(AkitaError::InvalidSetup(
            "setup column weight batches are malformed".into(),
        ));
    }
    let column_count = columns.len();
    let output_len = column_count
        .checked_mul(batch_count)
        .ok_or_else(|| AkitaError::InvalidSetup("setup column batch size overflow".into()))?;
    let mut values = vec![E::zero(); output_len];
    cfg_chunks_mut!(&mut values, batch_count)
        .enumerate()
        .try_for_each(|(column_offset, output)| -> Result<(), AkitaError> {
            let column = columns
                .start
                .checked_add(column_offset)
                .ok_or_else(|| AkitaError::InvalidSetup("setup column offset overflow".into()))?;
            for (row, weights) in row_weights {
                let evaluation =
                    eval_flat_ring_at_pows_fast(family.ring_slice(*row, column)?, alpha_powers);
                for (accumulator, &weight) in output.iter_mut().zip(weights) {
                    if !weight.is_zero() {
                        *accumulator += weight * evaluation;
                    }
                }
            }
            Ok(())
        })?;
    Ok(SetupColumnEvaluations {
        batch_count,
        column_count,
        values,
    })
}

pub(super) struct SetupColumnEvaluations<E> {
    batch_count: usize,
    column_count: usize,
    /// Column-major batches: `values[column * batch_count + batch]`.
    values: Vec<E>,
}

impl<E: Copy> SetupColumnEvaluations<E> {
    pub(super) fn get(&self, batch: usize, column: usize) -> Result<E, AkitaError> {
        if batch >= self.batch_count || column >= self.column_count {
            return Err(AkitaError::InvalidProof);
        }
        let index = column
            .checked_mul(self.batch_count)
            .and_then(|offset| offset.checked_add(batch))
            .ok_or(AkitaError::InvalidProof)?;
        self.values
            .get(index)
            .copied()
            .ok_or(AkitaError::InvalidProof)
    }
}
