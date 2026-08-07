use super::*;
use core::ops::Range;

/// Sparse transformed-witness evaluations for extension-opening reduction.
#[derive(Debug, Clone)]
pub struct SparseExtensionOpeningWitness<F: FieldCore, E: ExtField<F>> {
    pub(super) table_len: usize,
    pub(super) indices: Vec<usize>,
    pub(super) values: EvaluationTable<F, E>,
    /// Number of upcoming folds guaranteed to leave at most one entry per pair
    /// (no merges). While positive, the merge-free fast path is exact: the round
    /// message has a closed form and the witness folds in place without
    /// reallocating. Derived once at construction from the entry spacing; see
    /// [`Self::leading_merge_free_rounds`].
    pub(super) merge_free_rounds_left: usize,
}

#[cfg(feature = "parallel")]
const SPARSE_PARALLEL_ENTRY_THRESHOLD: usize = 1 << 14;
#[cfg(feature = "parallel")]
const SPARSE_PARALLEL_CHUNKS_PER_THREAD: usize = 4;

impl<F: FieldCore, E: ExtField<F>> SparseExtensionOpeningWitness<F, E> {
    /// Construct a sparse witness table from `(index, value)` entries.
    ///
    /// Duplicate indices are combined, and zero entries are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if `table_len` is not a nonzero power of two or if an
    /// entry index is out of range.
    pub fn new(table_len: usize, mut entries: Vec<(usize, E)>) -> Result<Self, AkitaError> {
        let _span = tracing::debug_span!(
            "SparseExtensionOpeningWitness::new",
            table_len,
            entries_len = entries.len()
        )
        .entered();
        entries.sort_unstable_by_key(|(idx, _)| *idx);
        Self::from_sorted_entries(table_len, entries)
    }

    /// Construct a sparse witness table from entries already sorted by index.
    ///
    /// Duplicate indices are combined, and zero entries are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if `table_len` is not a nonzero power of two, if an
    /// entry index is out of range, or if entries are not sorted by index.
    pub fn from_sorted_entries(
        table_len: usize,
        entries: Vec<(usize, E)>,
    ) -> Result<Self, AkitaError> {
        let _span = tracing::debug_span!(
            "SparseExtensionOpeningWitness::from_sorted_entries",
            table_len,
            entries_len = entries.len()
        )
        .entered();
        Self::validate_table_len(table_len)?;

        let mut combined: Vec<(usize, E)> = Vec::with_capacity(entries.len());
        let mut previous_idx = None;
        for (idx, value) in entries {
            Self::validate_index(table_len, idx)?;
            if previous_idx.is_some_and(|previous| idx < previous) {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness sorted constructor received unsorted entries"
                        .to_string(),
                ));
            }
            previous_idx = Some(idx);
            if value == E::zero() {
                continue;
            }
            if let Some((last_idx, last_value)) = combined.last_mut() {
                if *last_idx == idx {
                    *last_value += value;
                    if *last_value == E::zero() {
                        combined.pop();
                    }
                    continue;
                }
            }
            combined.push((idx, value));
        }
        Ok(Self::from_normalized_entries(table_len, combined))
    }

    /// Construct a sparse witness table from entries already normalized as
    /// strictly sorted, unique, nonzero `(index, value)` pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if `table_len` is not a nonzero power of two, if an
    /// entry index is out of range, if an entry is zero, or if entries are not
    /// strictly sorted by index.
    pub fn from_sorted_unique_entries(
        table_len: usize,
        entries: Vec<(usize, E)>,
    ) -> Result<Self, AkitaError> {
        let _span = tracing::debug_span!(
            "SparseExtensionOpeningWitness::from_sorted_unique_entries",
            table_len,
            entries_len = entries.len()
        )
        .entered();
        Self::validate_table_len(table_len)?;

        let mut previous_idx = None;
        for &(idx, value) in &entries {
            Self::validate_index(table_len, idx)?;
            if previous_idx.is_some_and(|previous| idx <= previous) {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness unique constructor received duplicate or unsorted entries"
                        .to_string(),
                ));
            }
            if value == E::zero() {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness unique constructor received a zero entry"
                        .to_string(),
                ));
            }
            previous_idx = Some(idx);
        }
        Ok(Self::from_normalized_entries(table_len, entries))
    }

    fn validate_table_len(table_len: usize) -> Result<(), AkitaError> {
        if table_len == 0 || !table_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "sparse extension-opening witness length must be a nonzero power of two"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn validate_index(table_len: usize, index: usize) -> Result<(), AkitaError> {
        if index >= table_len {
            return Err(AkitaError::InvalidInput(
                "sparse extension-opening witness index out of range".to_string(),
            ));
        }
        Ok(())
    }

    fn from_normalized_entries(table_len: usize, entries: Vec<(usize, E)>) -> Self {
        let mut indices = Vec::with_capacity(entries.len());
        let mut evaluations = Vec::with_capacity(entries.len());
        for (index, value) in entries {
            indices.push(index);
            evaluations.push(value);
        }
        let merge_free_rounds_left = Self::leading_merge_free_rounds(table_len, &indices);
        Self {
            table_len,
            indices,
            values: EvaluationTable::from_evaluations(&evaluations),
            merge_free_rounds_left,
        }
    }

    /// Number of leading folds guaranteed to be merge-free.
    pub(super) fn leading_merge_free_rounds(table_len: usize, indices: &[usize]) -> usize {
        let total = table_len.trailing_zeros() as usize;
        if indices.len() < 2 {
            return total;
        }
        let first_merge = indices
            .windows(2)
            .map(|window| usize::BITS - (window[0] ^ window[1]).leading_zeros())
            .min()
            .unwrap_or(usize::BITS);
        (first_merge as usize).saturating_sub(1).min(total)
    }

    /// Dense table length represented by this sparse witness.
    pub fn table_len(&self) -> usize {
        self.table_len
    }

    /// Sorted logical indices of the nonzero sparse rows.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Number of stored sparse rows.
    pub fn num_entries(&self) -> usize {
        self.indices.len()
    }

    /// Value belonging to one stored sparse row.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.num_entries()`.
    pub fn value(&self, row: usize) -> E {
        self.values.evaluation(row)
    }

    /// Combine sparse witnesses over the same table domain.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no terms or if the sparse witnesses have
    /// different table lengths.
    pub fn linear_combination<'a, I>(terms: I) -> Result<Self, AkitaError>
    where
        I: IntoIterator<Item = (E, &'a Self)>,
        F: 'a,
        E: 'a,
    {
        let _span =
            tracing::debug_span!("SparseExtensionOpeningWitness::linear_combination").entered();
        let mut table_len = None;
        let mut entries = Vec::new();
        {
            let _span = tracing::debug_span!("sparse_extension_witness_lc_collect").entered();
            for (coeff, witness) in terms {
                match table_len {
                    Some(len) if len != witness.table_len() => {
                        return Err(AkitaError::InvalidSize {
                            expected: len,
                            actual: witness.table_len(),
                        });
                    }
                    None => table_len = Some(witness.table_len()),
                    Some(_) => {}
                }
                entries.extend(
                    witness
                        .indices
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(row, index)| (index, witness.values.evaluation(row) * coeff)),
                );
            }
        }
        let table_len = table_len.ok_or_else(|| {
            AkitaError::InvalidInput(
                "sparse extension-opening witness combination requires at least one term"
                    .to_string(),
            )
        })?;
        let _span = tracing::debug_span!(
            "sparse_extension_witness_lc_normalize",
            table_len,
            entries_len = entries.len()
        )
        .entered();
        Self::new(table_len, entries)
    }

    pub(super) fn claim_with_factor(&self, factor_evals: &[E]) -> Result<E, AkitaError> {
        if factor_evals.len() != self.table_len {
            return Err(AkitaError::InvalidSize {
                expected: self.table_len,
                actual: factor_evals.len(),
            });
        }
        Ok(self
            .indices
            .iter()
            .copied()
            .enumerate()
            .fold(E::zero(), |acc, (row, index)| {
                acc + self.values.evaluation(row) * factor_evals[index]
            }))
    }

    pub(super) fn claim_with_factor_fn<P>(&self, factor_at: P) -> E
    where
        P: Fn(usize) -> E,
    {
        self.indices
            .iter()
            .copied()
            .enumerate()
            .fold(E::zero(), |acc, (row, index)| {
                acc + self.values.evaluation(row) * factor_at(index)
            })
    }

    pub(super) fn final_eval(&self) -> Option<E> {
        if self.table_len != 1 {
            return None;
        }
        Some(if self.values.is_empty() {
            E::zero()
        } else {
            self.values.evaluation(0)
        })
    }

    #[cfg(feature = "parallel")]
    pub(super) fn parallel_chunk_size(len: usize) -> usize {
        let target_chunks = rayon::current_num_threads() * SPARSE_PARALLEL_CHUNKS_PER_THREAD;
        len.div_ceil(target_chunks)
            .max(SPARSE_PARALLEL_ENTRY_THRESHOLD)
    }

    #[cfg(feature = "parallel")]
    pub(super) fn pair_aligned_ranges(&self) -> Vec<Range<usize>> {
        let len = self.indices.len();
        let chunk_size = Self::parallel_chunk_size(len);
        let mut ranges = Vec::with_capacity(len.div_ceil(chunk_size));
        let mut start = 0;
        while start < len {
            let mut end = (start + chunk_size).min(len);
            if end < len {
                let split_pair = self.indices[end] / 2;
                while end < len && self.indices[end] / 2 == split_pair {
                    end += 1;
                }
            }
            ranges.push(start..end);
            start = end;
        }
        ranges
    }
}

impl<F, E> SparseExtensionOpeningWitness<F, E>
where
    F: FieldCore,
    E: ExtField<F> + HasUnreducedOps,
{
    fn accumulate_range_with_factor<P>(
        &self,
        rows: Range<usize>,
        coeff: E,
        merge_free: bool,
        factor_pair: &P,
    ) -> (E, E)
    where
        P: Fn(usize) -> (E, E) + Sync,
    {
        let (constant, quadratic) = match (E::DELAYED_PRODUCT_SUM_IS_EXACT, merge_free) {
            (true, false) => self
                .accumulate_range_with_factor_using::<DelayedProductRoundAccumulator<E>, P>(
                    rows,
                    factor_pair,
                ),
            (false, false) => self
                .accumulate_range_with_factor_using::<DirectProductRoundAccumulator<E>, P>(
                    rows,
                    factor_pair,
                ),
            (true, true) => self
                .accumulate_merge_free_range_using::<DelayedProductRoundAccumulator<E>, P>(
                    rows,
                    factor_pair,
                ),
            (false, true) => self
                .accumulate_merge_free_range_using::<DirectProductRoundAccumulator<E>, P>(
                    rows,
                    factor_pair,
                ),
        };
        (coeff * constant, coeff * quadratic)
    }

    fn accumulate_range_with_factor_using<A, P>(
        &self,
        rows: Range<usize>,
        factor_pair: &P,
    ) -> (E, E)
    where
        A: ProductRoundAccumulator<E>,
        P: Fn(usize) -> (E, E) + Sync,
    {
        let mut acc = A::zero();
        let mut row = rows.start;
        while row < rows.end {
            let pair = self.indices[row] / 2;
            let mut w0 = E::zero();
            let mut w1 = E::zero();
            while row < rows.end && self.indices[row] / 2 == pair {
                let index = self.indices[row];
                let value = self.values.evaluation(row);
                if index & 1 == 0 {
                    w0 += value;
                } else {
                    w1 += value;
                }
                row += 1;
            }

            let (a0, a1) = factor_pair(pair);
            let da = a1 - a0;
            if w0 == E::zero() {
                acc.add_quadratic_product(w1, da);
            } else {
                acc.add_constant_product(w0, a0);
                acc.add_quadratic_product(w1 - w0, da);
            }
        }
        acc.finish()
    }

    fn accumulate_merge_free_range_using<A, P>(&self, rows: Range<usize>, factor_pair: &P) -> (E, E)
    where
        A: ProductRoundAccumulator<E>,
        P: Fn(usize) -> (E, E) + Sync,
    {
        let mut acc = A::zero();
        for row in rows {
            let index = self.indices[row];
            let value = self.values.evaluation(row);
            let (a0, a1) = factor_pair(index >> 1);
            let da = a1 - a0;
            if index & 1 == 0 {
                acc.add_constant_product(value, a0);
                acc.add_quadratic_product(E::zero() - value, da);
            } else {
                acc.add_quadratic_product(value, da);
            }
        }
        acc.finish()
    }

    fn accumulate_range(
        &self,
        rows: Range<usize>,
        factor_evals: &[E],
        coeff: E,
        merge_free: bool,
    ) -> (E, E) {
        self.accumulate_range_with_factor(rows, coeff, merge_free, &|pair| {
            (factor_evals[2 * pair], factor_evals[2 * pair + 1])
        })
    }

    pub(super) fn accumulate_round(
        &self,
        factor_evals: &[E],
        coeff: E,
        constant: &mut E,
        quadratic: &mut E,
    ) {
        let _span = tracing::trace_span!(
            "SparseExtensionOpeningWitness::accumulate_round",
            table_len = self.table_len,
            entries_len = self.indices.len()
        )
        .entered();
        let merge_free = self.merge_free_rounds_left > 0;
        #[cfg(feature = "parallel")]
        let (round_constant, round_quadratic) =
            if self.indices.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
                if merge_free {
                    let chunk_size = Self::parallel_chunk_size(self.indices.len());
                    (0..self.indices.len())
                        .into_par_iter()
                        .step_by(chunk_size)
                        .map(|start| {
                            self.accumulate_range(
                                start..(start + chunk_size).min(self.indices.len()),
                                factor_evals,
                                coeff,
                                true,
                            )
                        })
                        .reduce(
                            || (E::zero(), E::zero()),
                            |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                        )
                } else {
                    self.pair_aligned_ranges()
                        .into_par_iter()
                        .map(|rows| self.accumulate_range(rows, factor_evals, coeff, false))
                        .reduce(
                            || (E::zero(), E::zero()),
                            |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                        )
                }
            } else {
                self.accumulate_range(0..self.indices.len(), factor_evals, coeff, merge_free)
            };
        #[cfg(not(feature = "parallel"))]
        let (round_constant, round_quadratic) =
            self.accumulate_range(0..self.indices.len(), factor_evals, coeff, merge_free);
        *constant += round_constant;
        *quadratic += round_quadratic;
    }

    pub(super) fn accumulate_round_with_factor<P>(
        &self,
        coeff: E,
        constant: &mut E,
        quadratic: &mut E,
        factor_pair: P,
    ) where
        P: Fn(usize) -> (E, E) + Sync,
    {
        let _span = tracing::trace_span!(
            "SparseExtensionOpeningWitness::accumulate_round_with_factor",
            table_len = self.table_len,
            entries_len = self.indices.len()
        )
        .entered();
        let merge_free = self.merge_free_rounds_left > 0;
        #[cfg(feature = "parallel")]
        let (round_constant, round_quadratic) = if self.indices.len()
            >= SPARSE_PARALLEL_ENTRY_THRESHOLD
        {
            if merge_free {
                let chunk_size = Self::parallel_chunk_size(self.indices.len());
                (0..self.indices.len())
                    .into_par_iter()
                    .step_by(chunk_size)
                    .map(|start| {
                        self.accumulate_range_with_factor(
                            start..(start + chunk_size).min(self.indices.len()),
                            coeff,
                            true,
                            &factor_pair,
                        )
                    })
                    .reduce(
                        || (E::zero(), E::zero()),
                        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                    )
            } else {
                self.pair_aligned_ranges()
                    .into_par_iter()
                    .map(|rows| self.accumulate_range_with_factor(rows, coeff, false, &factor_pair))
                    .reduce(
                        || (E::zero(), E::zero()),
                        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                    )
            }
        } else {
            self.accumulate_range_with_factor(
                0..self.indices.len(),
                coeff,
                merge_free,
                &factor_pair,
            )
        };
        #[cfg(not(feature = "parallel"))]
        let (round_constant, round_quadratic) = self.accumulate_range_with_factor(
            0..self.indices.len(),
            coeff,
            merge_free,
            &factor_pair,
        );
        *constant += round_constant;
        *quadratic += round_quadratic;
    }

    pub(super) fn accumulate_grouped_tensor_round(
        &self,
        factor: &TensorEqualityFactor<E>,
        coeff: E,
        constant: &mut E,
        quadratic: &mut E,
    ) {
        let _span = tracing::trace_span!(
            "SparseExtensionOpeningWitness::accumulate_grouped_tensor_round",
            table_len = self.table_len,
            entries_len = self.indices.len()
        )
        .entered();
        debug_assert!(factor.supports_grouped_rounds());

        #[cfg(feature = "parallel")]
        let (round_constant, round_quadratic) =
            if self.indices.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
                self.pair_aligned_ranges()
                    .into_par_iter()
                    .map(|rows| factor.compute_grouped_round(self, rows))
                    .reduce(
                        || (E::zero(), E::zero()),
                        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                    )
            } else {
                factor.compute_grouped_round(self, 0..self.indices.len())
            };
        #[cfg(not(feature = "parallel"))]
        let (round_constant, round_quadratic) =
            factor.compute_grouped_round(self, 0..self.indices.len());
        *constant += coeff * round_constant;
        *quadratic += coeff * round_quadratic;
    }

    /// Fold one merge-free round and accumulate the next round in one sweep.
    pub(super) fn fused_fold_accumulate_merge_free<P>(
        &mut self,
        r_round: E,
        next_factor_pair: &P,
    ) -> (E, E)
    where
        P: Fn(usize) -> (E, E) + Sync,
    {
        let round = if E::DELAYED_PRODUCT_SUM_IS_EXACT {
            self.fused_fold_accumulate_merge_free_using::<DelayedProductRoundAccumulator<E>, P>(
                r_round,
                next_factor_pair,
            )
        } else {
            self.fused_fold_accumulate_merge_free_using::<DirectProductRoundAccumulator<E>, P>(
                r_round,
                next_factor_pair,
            )
        };
        self.table_len /= 2;
        self.merge_free_rounds_left -= 1;
        round
    }

    fn fused_fold_accumulate_merge_free_using<A, P>(
        &mut self,
        r_round: E,
        next_factor_pair: &P,
    ) -> (E, E)
    where
        A: ProductRoundAccumulator<E>,
        P: Fn(usize) -> (E, E) + Sync,
    {
        let one_minus = E::one() - r_round;
        let mut accumulator = A::zero();
        for row in 0..self.indices.len() {
            let index = self.indices[row];
            let value = self.values.evaluation(row);
            let folded = if index & 1 == 0 {
                value * one_minus
            } else {
                value * r_round
            };
            let folded_index = index >> 1;
            self.indices[row] = folded_index;
            self.values.set_evaluation(row, folded);

            let (a0, a1) = next_factor_pair(folded_index >> 1);
            let da = a1 - a0;
            if folded_index & 1 == 0 {
                accumulator.add_constant_product(folded, a0);
                accumulator.add_quadratic_product(E::zero() - folded, da);
            } else {
                accumulator.add_quadratic_product(folded, da);
            }
        }
        accumulator.finish()
    }
}

impl<F: FieldCore, E: ExtField<F>> SparseExtensionOpeningWitness<F, E> {
    pub(super) fn fold_in_place(&mut self, r_round: E) {
        let _span = tracing::trace_span!(
            "SparseExtensionOpeningWitness::fold_in_place",
            table_len = self.table_len,
            entries_len = self.indices.len()
        )
        .entered();
        if self.table_len <= 1 {
            return;
        }
        if self.merge_free_rounds_left > 0 {
            self.fold_in_place_merge_free(r_round);
            self.table_len /= 2;
            self.merge_free_rounds_left -= 1;
            return;
        }

        let one_minus = E::one() - r_round;
        let mut input_row = 0;
        let mut output_row = 0;
        while input_row < self.indices.len() {
            let pair = self.indices[input_row] / 2;
            let mut value = E::zero();
            while input_row < self.indices.len() && self.indices[input_row] / 2 == pair {
                let index = self.indices[input_row];
                let entry = self.values.evaluation(input_row);
                value += if index & 1 == 0 {
                    entry * one_minus
                } else {
                    entry * r_round
                };
                input_row += 1;
            }
            if value != E::zero() {
                self.indices[output_row] = pair;
                self.values.set_evaluation(output_row, value);
                output_row += 1;
            }
        }
        self.table_len /= 2;
        self.indices.truncate(output_row);
        self.values.truncate(output_row);
    }

    /// Allocation-free in-place fold for the merge-free regime.
    pub(super) fn fold_in_place_merge_free(&mut self, r_round: E) {
        let one_minus = E::one() - r_round;
        for row in 0..self.indices.len() {
            let index = self.indices[row];
            let value = self.values.evaluation(row);
            let folded = if index & 1 == 0 {
                value * one_minus
            } else {
                value * r_round
            };
            self.indices[row] = index >> 1;
            self.values.set_evaluation(row, folded);
        }
    }
}
