use std::ops::Range;

use akita_error::AkitaError;
use jolt_field::solinas::parallel::*;
use jolt_field::Field;

/// One family of setup-matrix rows read as per-column ring slices borrowed
/// from the materialized store.
pub(super) struct SetupRows<'a, F: Field> {
    pub(super) rows: Vec<&'a [F]>,
    pub(super) ring_d: usize,
}

impl<F: Field> SetupRows<'_, F> {
    pub(super) fn ring_slice(&self, row: usize, col: usize) -> Result<&[F], AkitaError> {
        self.rows
            .get(row)
            .and_then(|row| row.get(col * self.ring_d..(col + 1) * self.ring_d))
            .ok_or(AkitaError::InvalidProof)
    }
}

pub(super) fn contract_setup_columns<F, E>(
    family: &SetupRows<'_, F>,
    columns: Range<usize>,
    row_weights: &[(usize, Vec<E>)],
    batch_count: usize,
    value_width: usize,
    contract: impl Fn(&[F]) -> Result<Vec<E>, AkitaError> + Sync,
) -> Result<SetupColumnValues<E>, AkitaError>
where
    F: Field,
    E: Field,
{
    if batch_count == 0
        || value_width == 0
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
        .and_then(|len| len.checked_mul(value_width))
        .ok_or_else(|| AkitaError::InvalidSetup("setup column batch size overflow".into()))?;
    let mut values = vec![E::zero(); output_len];
    cfg_chunks_mut!(&mut values, batch_count * value_width)
        .enumerate()
        .try_for_each(|(column_offset, output)| -> Result<(), AkitaError> {
            let column = columns
                .start
                .checked_add(column_offset)
                .ok_or_else(|| AkitaError::InvalidSetup("setup column offset overflow".into()))?;
            for (row, weights) in row_weights {
                let contracted = contract(family.ring_slice(*row, column)?)?;
                if contracted.len() != value_width {
                    return Err(AkitaError::InvalidSetup(
                        "setup column contraction width mismatch".into(),
                    ));
                }
                for (batch, &weight) in weights.iter().enumerate() {
                    if weight.is_zero() {
                        continue;
                    }
                    let destination = output
                        .get_mut(batch * value_width..(batch + 1) * value_width)
                        .ok_or(AkitaError::InvalidProof)?;
                    for (accumulator, &value) in destination.iter_mut().zip(&contracted) {
                        *accumulator += weight * value;
                    }
                }
            }
            Ok(())
        })?;
    Ok(SetupColumnValues {
        batch_count,
        column_count,
        value_width,
        values,
    })
}

pub(super) struct SetupColumnValues<E> {
    batch_count: usize,
    column_count: usize,
    value_width: usize,
    /// Column-major batches, each with one mode-owned contraction value.
    values: Vec<E>,
}

impl<E> SetupColumnValues<E> {
    pub(super) fn get(&self, batch: usize, column: usize) -> Result<&[E], AkitaError> {
        if batch >= self.batch_count || column >= self.column_count {
            return Err(AkitaError::InvalidProof);
        }
        let start = column
            .checked_mul(self.batch_count)
            .and_then(|offset| offset.checked_add(batch))
            .and_then(|index| index.checked_mul(self.value_width))
            .ok_or(AkitaError::InvalidProof)?;
        let end = start
            .checked_add(self.value_width)
            .ok_or(AkitaError::InvalidProof)?;
        self.values.get(start..end).ok_or(AkitaError::InvalidProof)
    }

    pub(super) fn get_scalar(&self, batch: usize, column: usize) -> Result<E, AkitaError>
    where
        E: Copy,
    {
        let [value] = self.get(batch, column)? else {
            return Err(AkitaError::InvalidProof);
        };
        Ok(*value)
    }
}
