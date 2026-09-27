use std::ops::Range;

use akita_algebra::ring::ResidueKernelPoint;
use akita_error::AkitaError;
use jolt_field::solinas::parallel::*;
use jolt_field::{ExtField, Field, MulBaseUnreduced, Unreduced, Zero};

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

trait BaseProductSum<F: Field, E: MulBaseUnreduced<F>> {
    fn zero() -> Self;
    fn add_product(&mut self, weight: E, coefficient: F);
    fn finish(self) -> E;
}

struct DelayedBaseProductSum<E: Unreduced>(E::Product);

impl<F, E> BaseProductSum<F, E> for DelayedBaseProductSum<E>
where
    F: Field,
    E: MulBaseUnreduced<F>,
{
    #[inline(always)]
    fn zero() -> Self {
        Self(E::Product::zero())
    }

    #[inline(always)]
    fn add_product(&mut self, weight: E, coefficient: F) {
        self.0 += weight.mul_base_unreduced(coefficient);
    }

    #[inline(always)]
    fn finish(self) -> E {
        E::reduce_product(self.0)
    }
}

struct CanonicalBaseProductSum<E>(E);

impl<F, E> BaseProductSum<F, E> for CanonicalBaseProductSum<E>
where
    F: Field,
    E: MulBaseUnreduced<F>,
{
    #[inline(always)]
    fn zero() -> Self {
        Self(E::zero())
    }

    #[inline(always)]
    fn add_product(&mut self, weight: E, coefficient: F) {
        self.0 += weight.mul_base(coefficient);
    }

    #[inline(always)]
    fn finish(self) -> E {
        self.0
    }
}

/// Contract setup rows before applying the linear negacyclic residue map.
///
/// For row weights `w[r, batch]` and setup rings `M[r, column]`, this computes
/// one residue kernel for `sum_r w[r, batch] M[r, column]`. Linearity makes it
/// identical to contracting one residue kernel per row, while avoiding those
/// repeated recurrences and their temporary vectors.
pub(super) fn contract_setup_residue_columns<F, E>(
    family: &SetupRows<'_, F>,
    columns: Range<usize>,
    row_weights: &[(usize, Vec<E>)],
    batch_count: usize,
    point: &ResidueKernelPoint<E>,
) -> Result<SetupColumnValues<E>, AkitaError>
where
    F: Field,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    if E::SUM_IS_EXACT {
        contract_setup_residue_columns_with::<F, E, DelayedBaseProductSum<E>>(
            family,
            columns,
            row_weights,
            batch_count,
            point,
        )
    } else {
        contract_setup_residue_columns_with::<F, E, CanonicalBaseProductSum<E>>(
            family,
            columns,
            row_weights,
            batch_count,
            point,
        )
    }
}

fn contract_setup_residue_columns_with<F, E, A>(
    family: &SetupRows<'_, F>,
    columns: Range<usize>,
    row_weights: &[(usize, Vec<E>)],
    batch_count: usize,
    point: &ResidueKernelPoint<E>,
) -> Result<SetupColumnValues<E>, AkitaError>
where
    F: Field,
    E: ExtField<F> + MulBaseUnreduced<F>,
    A: BaseProductSum<F, E>,
{
    if batch_count == 0
        || family.ring_d != point.dimension()
        || row_weights
            .iter()
            .any(|(_, weights)| weights.len() != batch_count)
    {
        return Err(AkitaError::InvalidSetup(
            "setup residue-column weight batches are malformed".into(),
        ));
    }
    let column_count = columns.len();
    let column_width = batch_count
        .checked_mul(family.ring_d)
        .ok_or_else(|| AkitaError::InvalidSetup("setup column batch size overflow".into()))?;
    let output_len = column_count
        .checked_mul(column_width)
        .ok_or_else(|| AkitaError::InvalidSetup("setup column batch size overflow".into()))?;
    let mut values = vec![E::zero(); output_len];
    cfg_chunks_mut!(&mut values, column_width)
        .enumerate()
        .try_for_each(|(column_offset, output)| -> Result<(), AkitaError> {
            let column = columns
                .start
                .checked_add(column_offset)
                .ok_or_else(|| AkitaError::InvalidSetup("setup column offset overflow".into()))?;
            let mut coefficient_sums = (0..column_width).map(|_| A::zero()).collect::<Vec<_>>();
            for (row, weights) in row_weights {
                let coefficients = family.ring_slice(*row, column)?;
                for (batch, &weight) in weights.iter().enumerate() {
                    if weight.is_zero() {
                        continue;
                    }
                    let destination = coefficient_sums
                        .get_mut(batch * family.ring_d..(batch + 1) * family.ring_d)
                        .ok_or(AkitaError::InvalidProof)?;
                    for (accumulator, &coefficient) in destination.iter_mut().zip(coefficients) {
                        accumulator.add_product(weight, coefficient);
                    }
                }
            }
            let contracted = coefficient_sums
                .into_iter()
                .map(A::finish)
                .collect::<Vec<_>>();
            for (coefficients, destination) in contracted
                .chunks_exact(family.ring_d)
                .zip(output.chunks_exact_mut(family.ring_d))
            {
                point.field_kernel_into(coefficients, destination)?;
            }
            Ok(())
        })?;
    Ok(SetupColumnValues {
        batch_count,
        column_count,
        value_width: family.ring_d,
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

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::ring::residue_kernel;
    use jolt_field::{FpExt4, Prime32Offset99, Ring};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    fn extension(seed: u64) -> E {
        E::from_base_fn(|coordinate| F::from_u64(seed + 13 * coordinate as u64))
    }

    #[test]
    fn row_first_residue_contraction_matches_kernel_first_oracle() {
        const D: usize = 4;
        let rows = (0..3)
            .map(|row| {
                (0..2 * D)
                    .map(|index| F::from_u64(17 + 11 * row + 7 * index as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let family = SetupRows {
            rows: rows.iter().map(Vec::as_slice).collect(),
            ring_d: D,
        };
        let row_weights = vec![
            (0, vec![extension(3), extension(5)]),
            (1, vec![extension(7), E::zero()]),
            (2, vec![extension(11), extension(13)]),
        ];
        let alpha = extension(19);
        let oracle = contract_setup_columns(&family, 0..2, &row_weights, 2, D, |coefficients| {
            residue_kernel::<F, E>(coefficients, alpha)
        })
        .unwrap();
        let point = ResidueKernelPoint::new(alpha, D).unwrap();
        let contracted =
            contract_setup_residue_columns(&family, 0..2, &row_weights, 2, &point).unwrap();

        assert_eq!(contracted.values, oracle.values);
    }
}
