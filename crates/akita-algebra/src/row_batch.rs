//! Bounded batching for jolt-field's shared-powers row kernels.
//!
//! The row kernels take slices of row references, but callers produce rows
//! one at a time from setup matrices and relation rows. Staging every row
//! reference before one kernel call makes verifier memory grow with the input.
//! These helpers stage at most [`ROW_BATCH`] rows in a fixed stack buffer and
//! flush them through the canonical kernel, so field arithmetic stays in
//! jolt-field while staging memory stays constant.

use akita_error::AkitaError;
use jolt_field::{Field, MulBaseUnreduced};

/// Rows staged per call into a jolt-field row kernel; also the term bound
/// for other batched verifier dot products.
pub const ROW_BATCH: usize = 32;

/// `Σ_i weights[i] · Σ_j rows[i][j]·pows[j]` over rows pushed one at a time,
/// flushed through [`MulBaseUnreduced::weighted_dot_base_rows`] every
/// [`ROW_BATCH`] rows.
///
/// Every pushed row must be as long as `pows`; the kernel panics otherwise.
pub struct WeightedRowBatch<'a, F, E> {
    pows: &'a [E],
    rows: [&'a [F]; ROW_BATCH],
    weights: [E; ROW_BATCH],
    /// Staged rows; strictly below [`ROW_BATCH`] between calls.
    len: usize,
    sum: E,
}

impl<'a, F, E> WeightedRowBatch<'a, F, E>
where
    F: Field,
    E: MulBaseUnreduced<F>,
{
    /// An empty sum over rows evaluated against `pows`.
    pub fn new(pows: &'a [E]) -> Self {
        Self {
            pows,
            rows: [&[]; ROW_BATCH],
            weights: [E::zero(); ROW_BATCH],
            len: 0,
            sum: E::zero(),
        }
    }

    /// Add `weight · Σ_j row[j]·pows[j]` to the sum.
    pub fn push(&mut self, row: &'a [F], weight: E) {
        self.rows[self.len] = row;
        self.weights[self.len] = weight;
        self.len += 1;
        if self.len == ROW_BATCH {
            self.flush();
        }
    }

    /// The weighted sum over every pushed row.
    pub fn finish(mut self) -> E {
        self.flush();
        self.sum
    }

    fn flush(&mut self) {
        if self.len > 0 {
            self.sum += E::weighted_dot_base_rows(
                &self.rows[..self.len],
                &self.weights[..self.len],
                self.pows,
            );
            self.len = 0;
        }
    }
}

/// `out[i] = Σ_j pows[j]·rows[i][j]`, fetching and evaluating [`ROW_BATCH`]
/// rows per [`MulBaseUnreduced::dot_base_rows`] call.
///
/// Every row must be as long as `pows`; the kernel panics otherwise.
///
/// # Errors
///
/// Propagates the first error produced by `rows`.
pub fn dot_base_rows_batched<'a, F, E>(
    pows: &[E],
    rows: impl IntoIterator<Item = Result<&'a [F], AkitaError>>,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + 'a,
    E: MulBaseUnreduced<F>,
{
    let mut rows = rows.into_iter();
    let mut out = Vec::with_capacity(rows.size_hint().0);
    let mut batch: [&[F]; ROW_BATCH] = [&[]; ROW_BATCH];
    let mut values = [E::zero(); ROW_BATCH];
    loop {
        let mut len = 0;
        for (slot, row) in batch.iter_mut().zip(rows.by_ref()) {
            *slot = row?;
            len += 1;
        }
        if len == 0 {
            return Ok(out);
        }
        E::dot_base_rows(pows, &batch[..len], &mut values[..len]);
        out.extend_from_slice(&values[..len]);
        if len < ROW_BATCH {
            return Ok(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{ExtField, FpExt4, Prime32Offset99, Ring, Zero};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    fn rows(count: usize, width: usize) -> Vec<Vec<F>> {
        (0..count)
            .map(|row| {
                (0..width)
                    .map(|col| F::from_u64((row * 131 + col * 17 + 5) as u64))
                    .collect()
            })
            .collect()
    }

    fn per_row_dot<E: MulBaseUnreduced<F>>(pows: &[E], row: &[F]) -> E {
        row.iter().zip(pows).fold(E::zero(), |acc, (&coeff, &pow)| {
            acc + pow * E::lift_base(coeff)
        })
    }

    /// Row counts around the batch size, checked against per-row field
    /// arithmetic: a dropped, repeated, or stale buffered row changes the sum.
    #[test]
    fn batching_matches_per_row_arithmetic_at_batch_boundaries() {
        let width = 8;
        let pows: Vec<E> = (0..width)
            .map(|index| E::from_u64(3 * index as u64 + 1) + E::lift_base(F::from_u64(7)))
            .collect();
        for count in [
            0,
            1,
            ROW_BATCH - 1,
            ROW_BATCH,
            ROW_BATCH + 1,
            2 * ROW_BATCH,
            2 * ROW_BATCH + 3,
        ] {
            let rows = rows(count, width);
            let weights: Vec<E> = (0..count).map(|i| E::from_u64(5 * i as u64 + 2)).collect();

            let mut batch = WeightedRowBatch::new(&pows);
            for (row, &weight) in rows.iter().zip(&weights) {
                batch.push(row, weight);
            }
            let expected = rows
                .iter()
                .zip(&weights)
                .fold(E::zero(), |acc, (row, &weight)| {
                    acc + weight * per_row_dot(&pows, row)
                });
            assert_eq!(batch.finish(), expected, "weighted, {count} rows");

            let values =
                dot_base_rows_batched(&pows, rows.iter().map(|row| Ok(row.as_slice()))).unwrap();
            let expected: Vec<E> = rows.iter().map(|row| per_row_dot(&pows, row)).collect();
            assert_eq!(values, expected, "per-row, {count} rows");
        }
    }

    #[test]
    fn dot_base_rows_batched_propagates_row_errors() {
        let pows = vec![F::from_u64(1); 4];
        let rows = rows(ROW_BATCH + 2, 4);
        let fetched = rows.iter().enumerate().map(|(index, row)| {
            if index == ROW_BATCH + 1 {
                Err(AkitaError::InvalidProof)
            } else {
                Ok(row.as_slice())
            }
        });
        assert!(dot_base_rows_batched::<F, F>(&pows, fetched).is_err());
    }
}
