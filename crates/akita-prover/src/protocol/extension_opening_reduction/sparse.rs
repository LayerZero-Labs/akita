use super::*;

/// One term in an extension-opening reduction sumcheck.
///
/// A single dense term is the degenerate `1`-term case; the prover treats the
/// dense and batched paths uniformly.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionTerm<E: FieldCore> {
    pub(super) tables: ExtensionOpeningTables<E>,
    pub(super) coeff: E,
    /// `coeff`-scaled `(constant, quadratic)` for the next round, pre-computed
    /// by the fused fold in [`Self::ingest_challenge`] for the dense path.
    pub(super) cached_accumulate: Option<(E, E)>,
}

/// Sparse transformed-witness evaluations for extension-opening reduction.
#[derive(Debug, Clone)]
pub struct SparseExtensionOpeningWitness<E: FieldCore> {
    table_len: usize,
    entries: Vec<(usize, E)>,
    /// Number of upcoming folds guaranteed to leave at most one entry per pair
    /// (no merges). While positive, the merge-free fast path is exact: the round
    /// message has a closed form and the witness folds in place without
    /// reallocating. Derived once at construction from the entry spacing; see
    /// [`Self::leading_merge_free_rounds`].
    merge_free_rounds_left: usize,
}

#[cfg(feature = "parallel")]
const SPARSE_PARALLEL_ENTRY_THRESHOLD: usize = 1 << 14;
#[cfg(feature = "parallel")]
const SPARSE_PARALLEL_CHUNKS_PER_THREAD: usize = 4;

impl<E: FieldCore> SparseExtensionOpeningWitness<E> {
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
        if table_len == 0 || !table_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "sparse extension-opening witness length must be a nonzero power of two"
                    .to_string(),
            ));
        }
        let mut combined: Vec<(usize, E)> = Vec::with_capacity(entries.len());
        let mut previous_idx = None;
        for (idx, value) in entries {
            if idx >= table_len {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness index out of range".to_string(),
                ));
            }
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
        let merge_free_rounds_left = Self::leading_merge_free_rounds(table_len, &combined);
        Ok(Self {
            table_len,
            entries: combined,
            merge_free_rounds_left,
        })
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
        if table_len == 0 || !table_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "sparse extension-opening witness length must be a nonzero power of two"
                    .to_string(),
            ));
        }
        let mut previous_idx = None;
        for &(idx, value) in &entries {
            if idx >= table_len {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness index out of range".to_string(),
                ));
            }
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
        let merge_free_rounds_left = Self::leading_merge_free_rounds(table_len, &entries);
        Ok(Self {
            table_len,
            entries,
            merge_free_rounds_left,
        })
    }

    /// Number of leading folds guaranteed to be merge-free (every pair keeps at
    /// most one entry).
    ///
    /// Two adjacent entries `tᵢ < tⱼ` first land in the same folded index at fold
    /// `bit_length(tᵢ ⊕ tⱼ)`, so the minimum over neighbors is the first merging
    /// fold and the guaranteed merge-free run is that minus one. With fewer than
    /// two entries nothing ever merges, so the whole reduction stays merge-free.
    fn leading_merge_free_rounds(table_len: usize, entries: &[(usize, E)]) -> usize {
        let total = table_len.trailing_zeros() as usize;
        if entries.len() < 2 {
            return total;
        }
        let mut first_merge = usize::BITS;
        for window in entries.windows(2) {
            let diff = window[0].0 ^ window[1].0;
            let bit_length = usize::BITS - diff.leading_zeros();
            if bit_length < first_merge {
                first_merge = bit_length;
            }
        }
        (first_merge as usize).saturating_sub(1).min(total)
    }

    /// Dense table length represented by this sparse witness.
    pub fn table_len(&self) -> usize {
        self.table_len
    }

    /// Nonzero sparse entries, sorted by table index.
    pub fn entries(&self) -> &[(usize, E)] {
        &self.entries
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
                        .entries()
                        .iter()
                        .map(|&(idx, value)| (idx, value * coeff)),
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

    fn claim_with_factor(&self, factor_evals: &[E]) -> Result<E, AkitaError> {
        if factor_evals.len() != self.table_len {
            return Err(AkitaError::InvalidSize {
                expected: self.table_len,
                actual: factor_evals.len(),
            });
        }
        Ok(self.entries.iter().fold(E::zero(), |acc, &(idx, value)| {
            acc + value * factor_evals[idx]
        }))
    }

    fn claim_with_factor_fn<P>(&self, factor_at: P) -> E
    where
        P: Fn(usize) -> E,
    {
        self.entries
            .iter()
            .fold(E::zero(), |acc, &(idx, value)| acc + value * factor_at(idx))
    }

    fn final_eval(&self) -> Option<E> {
        if self.table_len != 1 {
            return None;
        }
        Some(
            self.entries
                .first()
                .map(|(_, value)| *value)
                .unwrap_or(E::zero()),
        )
    }

    fn fold_entries(entries: &[(usize, E)], r_round: E) -> Vec<(usize, E)> {
        let one_minus = E::one() - r_round;
        let mut folded = Vec::with_capacity(entries.len());
        let mut i = 0;
        while i < entries.len() {
            let pair = entries[i].0 / 2;
            let mut value = E::zero();
            while i < entries.len() && entries[i].0 / 2 == pair {
                let (idx, entry_value) = entries[i];
                value += if idx & 1 == 0 {
                    entry_value * one_minus
                } else {
                    entry_value * r_round
                };
                i += 1;
            }
            if value != E::zero() {
                folded.push((pair, value));
            }
        }
        folded
    }

    #[cfg(feature = "parallel")]
    fn parallel_chunk_size(len: usize) -> usize {
        let target_chunks = rayon::current_num_threads() * SPARSE_PARALLEL_CHUNKS_PER_THREAD;
        len.div_ceil(target_chunks)
            .max(SPARSE_PARALLEL_ENTRY_THRESHOLD)
    }

    #[cfg(feature = "parallel")]
    fn pair_aligned_ranges(&self) -> Vec<(usize, usize)> {
        let len = self.entries.len();
        let chunk_size = Self::parallel_chunk_size(len);
        let mut ranges = Vec::with_capacity(len.div_ceil(chunk_size));
        let mut start = 0;
        while start < len {
            let mut end = (start + chunk_size).min(len);
            if end < len {
                let split_pair = self.entries[end].0 / 2;
                while end < len && self.entries[end].0 / 2 == split_pair {
                    end += 1;
                }
            }
            ranges.push((start, end));
            start = end;
        }
        ranges
    }
}

impl<E: FieldCore + HasUnreducedOps> SparseExtensionOpeningWitness<E> {
    fn accumulate_entries_with_factor<P>(
        entries: &[(usize, E)],
        coeff: E,
        merge_free: bool,
        factor_pair: &P,
    ) -> (E, E)
    where
        P: Fn(usize) -> (E, E) + Sync,
    {
        // Honor `DELAYED_PRODUCT_SUM_IS_EXACT`: only sum wide products and reduce
        // once for fields whose accumulator is proven exact; otherwise reduce per
        // term so the coefficients stay byte-identical to `Mul`, matching the
        // dense round and `TensorEqualityFactor::factor_pair`.
        //
        // `merge_free` selects the closed-form pass valid while every pair has at
        // most one entry; it accumulates the identical products in the identical
        // order, so both paths agree bit-for-bit.
        let (constant, quadratic) = match (E::DELAYED_PRODUCT_SUM_IS_EXACT, merge_free) {
            (true, false) => Self::accumulate_entries_with_factor_using::<DelayedDeg2<E>, P>(
                entries,
                factor_pair,
            ),
            (false, false) => {
                Self::accumulate_entries_with_factor_using::<DirectDeg2<E>, P>(entries, factor_pair)
            }
            (true, true) => {
                Self::accumulate_entries_merge_free_using::<DelayedDeg2<E>, P>(entries, factor_pair)
            }
            (false, true) => {
                Self::accumulate_entries_merge_free_using::<DirectDeg2<E>, P>(entries, factor_pair)
            }
        };
        (coeff * constant, coeff * quadratic)
    }

    fn accumulate_entries_with_factor_using<A, P>(entries: &[(usize, E)], factor_pair: &P) -> (E, E)
    where
        A: Deg2RoundAccum<E>,
        P: Fn(usize) -> (E, E) + Sync,
    {
        let mut acc = A::zero();
        let mut i = 0;
        while i < entries.len() {
            let pair = entries[i].0 / 2;
            let mut w0 = E::zero();
            let mut w1 = E::zero();
            while i < entries.len() && entries[i].0 / 2 == pair {
                let (idx, value) = entries[i];
                if idx & 1 == 0 {
                    w0 += value;
                } else {
                    w1 += value;
                }
                i += 1;
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

    /// Closed-form merge-free specialization of
    /// [`Self::accumulate_entries_with_factor_using`].
    ///
    /// Valid only while every pair holds at most one entry (the leading
    /// `merge_free_rounds_left` rounds). It is the grouped loop with the inner
    /// pair-grouping `while` removed: each pair contributes exactly its single
    /// child, placed at `w0` or `w1` by parity, so the products, their order, and
    /// the `Deg2RoundAccum` calls are byte-identical to the general path.
    fn accumulate_entries_merge_free_using<A, P>(entries: &[(usize, E)], factor_pair: &P) -> (E, E)
    where
        A: Deg2RoundAccum<E>,
        P: Fn(usize) -> (E, E) + Sync,
    {
        let mut acc = A::zero();
        for &(idx, value) in entries {
            let (a0, a1) = factor_pair(idx >> 1);
            let da = a1 - a0;
            if idx & 1 == 0 {
                // even child: w0 = value, w1 = 0.
                acc.add_constant_product(value, a0);
                acc.add_quadratic_product(E::zero() - value, da);
            } else {
                // odd child: w0 = 0, w1 = value.
                acc.add_quadratic_product(value, da);
            }
        }
        acc.finish()
    }

    fn accumulate_entries(
        entries: &[(usize, E)],
        factor_evals: &[E],
        coeff: E,
        merge_free: bool,
    ) -> (E, E) {
        Self::accumulate_entries_with_factor(entries, coeff, merge_free, &|pair| {
            (factor_evals[2 * pair], factor_evals[2 * pair + 1])
        })
    }

    fn accumulate_round(&self, factor_evals: &[E], coeff: E, constant: &mut E, quadratic: &mut E) {
        let _span = tracing::trace_span!(
            "SparseExtensionOpeningWitness::accumulate_round",
            table_len = self.table_len,
            entries_len = self.entries.len()
        )
        .entered();
        let merge_free = self.merge_free_rounds_left > 0;
        #[cfg(feature = "parallel")]
        let (round_constant, round_quadratic) =
            if self.entries.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
                if merge_free {
                    let chunk_size = Self::parallel_chunk_size(self.entries.len());
                    self.entries
                        .par_chunks(chunk_size)
                        .map(|entries| Self::accumulate_entries(entries, factor_evals, coeff, true))
                        .reduce(
                            || (E::zero(), E::zero()),
                            |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                        )
                } else {
                    self.pair_aligned_ranges()
                        .into_par_iter()
                        .map(|(start, end)| {
                            Self::accumulate_entries(
                                &self.entries[start..end],
                                factor_evals,
                                coeff,
                                false,
                            )
                        })
                        .reduce(
                            || (E::zero(), E::zero()),
                            |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                        )
                }
            } else {
                Self::accumulate_entries(&self.entries, factor_evals, coeff, merge_free)
            };
        #[cfg(not(feature = "parallel"))]
        let (round_constant, round_quadratic) =
            Self::accumulate_entries(&self.entries, factor_evals, coeff, merge_free);
        *constant += round_constant;
        *quadratic += round_quadratic;
    }

    fn accumulate_round_with_factor<P>(
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
            entries_len = self.entries.len()
        )
        .entered();
        let merge_free = self.merge_free_rounds_left > 0;
        #[cfg(feature = "parallel")]
        let (round_constant, round_quadratic) =
            if self.entries.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
                if merge_free {
                    let chunk_size = Self::parallel_chunk_size(self.entries.len());
                    self.entries
                        .par_chunks(chunk_size)
                        .map(|entries| {
                            Self::accumulate_entries_with_factor(entries, coeff, true, &factor_pair)
                        })
                        .reduce(
                            || (E::zero(), E::zero()),
                            |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                        )
                } else {
                    self.pair_aligned_ranges()
                        .into_par_iter()
                        .map(|(start, end)| {
                            Self::accumulate_entries_with_factor(
                                &self.entries[start..end],
                                coeff,
                                false,
                                &factor_pair,
                            )
                        })
                        .reduce(
                            || (E::zero(), E::zero()),
                            |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                        )
                }
            } else {
                Self::accumulate_entries_with_factor(&self.entries, coeff, merge_free, &factor_pair)
            };
        #[cfg(not(feature = "parallel"))]
        let (round_constant, round_quadratic) =
            Self::accumulate_entries_with_factor(&self.entries, coeff, merge_free, &factor_pair);
        *constant += round_constant;
        *quadratic += round_quadratic;
    }
}

impl<E: FieldCore> SparseExtensionOpeningWitness<E> {
    fn fold_in_place(&mut self, r_round: E) {
        let _span = tracing::trace_span!(
            "SparseExtensionOpeningWitness::fold_in_place",
            table_len = self.table_len,
            entries_len = self.entries.len()
        )
        .entered();
        if self.table_len <= 1 {
            return;
        }
        // Merge-free regime: no pair merges this fold, so each entry just drops
        // its low tail bit and scales by the matching challenge weight. Fold in
        // place — no reallocation, no dedup, no pair-range scan.
        if self.merge_free_rounds_left > 0 {
            self.fold_in_place_merge_free(r_round);
            self.table_len /= 2;
            self.merge_free_rounds_left -= 1;
            return;
        }
        #[cfg(feature = "parallel")]
        let folded = if self.entries.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
            let chunks = self
                .pair_aligned_ranges()
                .into_par_iter()
                .map(|(start, end)| Self::fold_entries(&self.entries[start..end], r_round))
                .collect::<Vec<_>>();
            let len = chunks.iter().map(Vec::len).sum();
            let mut folded = Vec::with_capacity(len);
            for chunk in chunks {
                folded.extend(chunk);
            }
            folded
        } else {
            Self::fold_entries(&self.entries, r_round)
        };
        #[cfg(not(feature = "parallel"))]
        let folded = Self::fold_entries(&self.entries, r_round);
        self.table_len /= 2;
        self.entries = folded;
    }

    /// Alloc-free in-place fold for the merge-free regime.
    ///
    /// No pair has two entries, so folding never combines values: each entry
    /// `(idx, value)` becomes `(idx >> 1, value · weight)` with `weight` the
    /// even/odd challenge factor. Byte-identical to [`Self::fold_entries`] when
    /// every occupied pair holds one entry, and trivially parallel (no cross-entry
    /// dependency, no merges).
    fn fold_in_place_merge_free(&mut self, r_round: E) {
        let one_minus = E::one() - r_round;
        let fold_one = |entry: &mut (usize, E)| {
            let (idx, value) = *entry;
            let folded = if idx & 1 == 0 {
                value * one_minus
            } else {
                value * r_round
            };
            *entry = (idx >> 1, folded);
        };
        #[cfg(feature = "parallel")]
        {
            let len = self.entries.len();
            if len >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
                let chunk_size = Self::parallel_chunk_size(len);
                self.entries
                    .par_chunks_mut(chunk_size)
                    .for_each(|chunk| chunk.iter_mut().for_each(fold_one));
                return;
            }
        }
        self.entries.iter_mut().for_each(fold_one);
    }
}

#[derive(Debug, Clone)]
struct TensorFactorTransition<E: FieldCore> {
    zero: Vec<Vec<E>>,
    one: Vec<Vec<E>>,
}

/// Lazy transparent tensor factor for sparse extension-opening terms.
///
/// This stores the exact multilinear folding state for
/// `A_eta(w) = sum_u eq(u, eta) * coord_u(eq(r_tail, w))` without relying on
/// `coord_u` being extension-linear. Once the sparse low block has been folded,
/// it materializes into the ordinary dense factor table and rejoins the shared
/// reduction path.
#[derive(Debug, Clone)]
pub(super) struct TensorEqualityFactor<E: FieldCore> {
    table_vars: usize,
    round: usize,
    materialize_at: usize,
    prefix_state: Vec<E>,
    transitions: Vec<TensorFactorTransition<E>>,
    suffix_tables: Vec<Vec<E>>,
    low_states: Vec<Vec<E>>,
}

impl<E: FieldCore> TensorEqualityFactor<E> {
    fn new<F>(tail_point: Vec<E>, eta: Vec<E>, materialize_at: usize) -> Result<Self, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let (split_bits, width) = tensor_opening_split::<F, E>()?;
        if eta.len() != split_bits {
            return Err(AkitaError::InvalidSize {
                expected: split_bits,
                actual: eta.len(),
            });
        }
        if materialize_at > tail_point.len() {
            return Err(AkitaError::InvalidSize {
                expected: tail_point.len(),
                actual: materialize_at,
            });
        }
        checked_table_len(tail_point.len())?;
        checked_table_len(tail_point.len() - materialize_at)?;

        let eta_weights = EqPolynomial::evals(&eta)?;
        let basis = (0..width)
            .map(|idx| {
                let mut coords = vec![F::zero(); width];
                coords[idx] = F::one();
                E::from_base_slice(&coords)
            })
            .collect::<Vec<_>>();
        let one_coords = E::one().to_base_vec();
        if one_coords.len() != width {
            return Err(AkitaError::InvalidSize {
                expected: width,
                actual: one_coords.len(),
            });
        }
        let prefix_state = one_coords.into_iter().map(E::lift_base).collect::<Vec<_>>();

        let transitions = tail_point[..materialize_at]
            .iter()
            .copied()
            .map(|tail| Self::transition::<F>(&basis, tail, width))
            .collect::<Result<Vec<_>, _>>()?;
        let suffix_eq = EqPolynomial::evals(&tail_point[materialize_at..])?;
        let suffix_tables = basis
            .iter()
            .map(|&basis_elem| {
                suffix_eq
                    .iter()
                    .copied()
                    .map(|suffix| {
                        project_tensor_factor_value::<F, E>(
                            basis_elem * suffix,
                            &eta_weights,
                            width,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut factor = Self {
            table_vars: tail_point.len(),
            round: 0,
            materialize_at,
            prefix_state,
            transitions,
            suffix_tables,
            low_states: Vec::new(),
        };
        factor.rebuild_low_states();
        Ok(factor)
    }

    fn transition<F>(
        basis: &[E],
        tail: E,
        width: usize,
    ) -> Result<TensorFactorTransition<E>, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let tail_zero = E::one() - tail;
        let tail_one = tail;
        let mut zero = vec![vec![E::zero(); width]; width];
        let mut one = vec![vec![E::zero(); width]; width];
        for (src_idx, &basis_elem) in basis.iter().enumerate() {
            let zero_coords = (basis_elem * tail_zero).to_base_vec();
            let one_coords = (basis_elem * tail_one).to_base_vec();
            if zero_coords.len() != width || one_coords.len() != width {
                return Err(AkitaError::InvalidSize {
                    expected: width,
                    actual: zero_coords.len().max(one_coords.len()),
                });
            }
            for dst_idx in 0..width {
                zero[src_idx][dst_idx] = E::lift_base(zero_coords[dst_idx]);
                one[src_idx][dst_idx] = E::lift_base(one_coords[dst_idx]);
            }
        }
        Ok(TensorFactorTransition { zero, one })
    }

    fn len(&self) -> usize {
        1usize << (self.table_vars - self.round)
    }

    fn is_ready_to_materialize(&self) -> bool {
        self.round >= self.materialize_at
    }

    fn apply_transition(
        state: &[E],
        transition: &TensorFactorTransition<E>,
        challenge: E,
    ) -> Vec<E> {
        let width = state.len();
        let one_minus = E::one() - challenge;
        let mut next = vec![E::zero(); width];
        for (src_idx, &src) in state.iter().enumerate() {
            if src == E::zero() {
                continue;
            }
            for (dst_idx, dst) in next.iter_mut().enumerate() {
                let step = transition.zero[src_idx][dst_idx] * one_minus
                    + transition.one[src_idx][dst_idx] * challenge;
                *dst += src * step;
            }
        }
        next
    }

    fn apply_boolean_transition(
        state: &[E],
        transition: &TensorFactorTransition<E>,
        bit: usize,
    ) -> Vec<E> {
        let width = state.len();
        let matrix = if bit == 0 {
            &transition.zero
        } else {
            &transition.one
        };
        let mut next = vec![E::zero(); width];
        for (src_idx, &src) in state.iter().enumerate() {
            if src == E::zero() {
                continue;
            }
            for (dst_idx, dst) in next.iter_mut().enumerate() {
                *dst += src * matrix[src_idx][dst_idx];
            }
        }
        next
    }

    fn rebuild_low_states(&mut self) {
        let low_bits = self.materialize_at.saturating_sub(self.round);
        if low_bits == 0 {
            self.low_states.clear();
            return;
        }
        let count = 1usize << low_bits;
        let mut low_states = Vec::with_capacity(count);
        for low in 0..count {
            let mut state = self.prefix_state.clone();
            for bit_idx in 0..low_bits {
                let bit = (low >> bit_idx) & 1;
                state = Self::apply_boolean_transition(
                    &state,
                    &self.transitions[self.round + bit_idx],
                    bit,
                );
            }
            low_states.push(state);
        }
        self.low_states = low_states;
    }

    fn eval_state_at_suffix(&self, state: &[E], suffix_index: usize) -> E {
        self.suffix_tables
            .iter()
            .zip(state.iter().copied())
            .fold(E::zero(), |acc, (table, coeff)| {
                acc + coeff * table[suffix_index]
            })
    }

    fn factor_at_index(&self, index: usize) -> E {
        let low_bits = self.materialize_at.saturating_sub(self.round);
        if low_bits == 0 {
            return self.eval_state_at_suffix(&self.prefix_state, index);
        }
        let low_mask = (1usize << low_bits) - 1;
        let low = index & low_mask;
        let suffix_index = index >> low_bits;
        self.eval_state_at_suffix(&self.low_states[low], suffix_index)
    }

    fn fold_in_place(&mut self, r_round: E) {
        if self.len() <= 1 {
            return;
        }
        debug_assert!(self.round < self.materialize_at);
        self.prefix_state =
            Self::apply_transition(&self.prefix_state, &self.transitions[self.round], r_round);
        self.round += 1;
        self.rebuild_low_states();
    }

    fn materialize_dense(&self) -> Vec<E> {
        debug_assert!(self.is_ready_to_materialize());
        let suffix_len = self.suffix_tables.first().map(Vec::len).unwrap_or(0);
        let _span = tracing::debug_span!(
            "TensorEqualityFactor::materialize_dense",
            suffix_len,
            width = self.prefix_state.len()
        )
        .entered();
        #[cfg(feature = "parallel")]
        {
            (0..suffix_len)
                .into_par_iter()
                .map(|idx| self.eval_state_at_suffix(&self.prefix_state, idx))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..suffix_len)
                .map(|idx| self.eval_state_at_suffix(&self.prefix_state, idx))
                .collect()
        }
    }
}

impl<E: FieldCore + HasUnreducedOps> TensorEqualityFactor<E> {
    /// Factor inner product `sum_i state[i] * suffix_tables[i][suffix_index]`,
    /// reducing once at the end when the field's product accumulator is exact
    /// w.r.t. `Mul`, and otherwise falling back to the per-term
    /// [`Self::eval_state_at_suffix`].
    ///
    /// On the exact path (e.g. the fp32 `RingSubfieldFp4<Fp32>` campaign field)
    /// each product is widened into `E::ProductAccum` and the
    /// `state.len() == E::EXT_DEGREE` terms are summed before a single
    /// `reduce_product_accum`. The per-coefficient reduction is additive over
    /// the accumulator and the wide sum cannot overflow (`EXT_DEGREE` is a small
    /// power of two — 4 here — far below the accumulator's >= 2^63 headroom), so
    /// the result is byte-identical to `eval_state_at_suffix`.
    ///
    /// Fields whose wide accumulator is lossy versus `Mul` leave
    /// `DELAYED_PRODUCT_SUM_IS_EXACT` at `false` and take the per-term path, so
    /// the emitted factor, and the proof, stay unchanged. `Fp2<Fp64>` opts into
    /// the exact path only because its accumulator keeps the carry above bit
    /// 128 explicitly.
    ///
    /// The two factor values `a0` (state `low_zero`) and `a1` (state `low_one`)
    /// always share the same `suffix_index`, so both inner products read the
    /// same `suffix_tables[j][suffix_index]` column. They are fused into one
    /// pass over `j` that loads each column entry once and feeds it into both
    /// delayed accumulators, halving the column loads and tightening the loop
    /// without changing the accumulation order, so the result is byte-identical
    /// to two independent evaluations.
    fn factor_pair(&self, pair: usize) -> (E, E) {
        let low_bits = self.materialize_at - self.round;
        debug_assert!(low_bits > 0);
        let rest_low_bits = low_bits - 1;
        let low_mask = (1usize << rest_low_bits).saturating_sub(1);
        let low_rest = pair & low_mask;
        let suffix_index = pair >> rest_low_bits;
        let low_zero = low_rest << 1;
        let low_one = low_zero | 1;
        let state_zero = &self.low_states[low_zero];
        let state_one = &self.low_states[low_one];

        if !E::DELAYED_PRODUCT_SUM_IS_EXACT {
            return (
                self.eval_state_at_suffix(state_zero, suffix_index),
                self.eval_state_at_suffix(state_one, suffix_index),
            );
        }

        let mut accum_zero = E::ProductAccum::zero();
        let mut accum_one = E::ProductAccum::zero();
        for ((table, &coeff_zero), &coeff_one) in self
            .suffix_tables
            .iter()
            .zip(state_zero.iter())
            .zip(state_one.iter())
        {
            let column = table[suffix_index];
            accum_zero += coeff_zero.mul_to_product_accum(column);
            accum_one += coeff_one.mul_to_product_accum(column);
        }
        (
            E::reduce_product_accum(accum_zero),
            E::reduce_product_accum(accum_one),
        )
    }
}

/// Transparent factor for a sparse-witness term.
///
/// The lazy [`TensorEqualityFactor`] is only ever paired with a sparse witness,
/// so it lives inside the sparse case rather than as a standalone factor. This
/// is what makes the `(dense witness, tensor factor)` combination unrepresentable.
#[derive(Debug, Clone)]
pub(super) enum SparseFactor<E: FieldCore> {
    Dense(Vec<E>),
    Tensor(TensorEqualityFactor<E>),
}

/// Paired witness/factor tables for one extension-opening reduction term.
///
/// Only the three valid combinations are representable: a dense witness always
/// pairs with a dense factor, and the lazy tensor factor only ever pairs with a
/// sparse witness (via [`SparseFactor`]). A `(dense witness, tensor factor)`
/// pair cannot be constructed, so the prover never needs to reject or
/// `unreachable!` on it.
#[derive(Debug, Clone)]
pub(super) enum ExtensionOpeningTables<E: FieldCore> {
    Dense {
        witness: Vec<E>,
        factor: Vec<E>,
    },
    Sparse {
        witness: SparseExtensionOpeningWitness<E>,
        factor: SparseFactor<E>,
    },
}

impl<E: FieldCore> ExtensionOpeningTables<E> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Dense { witness, .. } => witness.len(),
            Self::Sparse { witness, .. } => witness.table_len(),
        }
    }

    pub(super) fn claim(&self) -> Result<E, AkitaError> {
        match self {
            Self::Dense { witness, factor } => extension_opening_reduction_claim(witness, factor),
            Self::Sparse { witness, factor } => match factor {
                SparseFactor::Dense(factor_evals) => witness.claim_with_factor(factor_evals),
                SparseFactor::Tensor(factor) => {
                    if witness.table_len() != factor.len() {
                        return Err(AkitaError::InvalidSize {
                            expected: witness.table_len(),
                            actual: factor.len(),
                        });
                    }
                    Ok(witness.claim_with_factor_fn(|idx| factor.factor_at_index(idx)))
                }
            },
        }
    }

    fn final_witness_and_factor_evals(&self) -> Option<(E, E)> {
        match self {
            Self::Dense { witness, factor } => {
                (factor.len() == 1 && witness.len() == 1).then(|| (witness[0], factor[0]))
            }
            Self::Sparse { witness, factor } => match factor {
                SparseFactor::Dense(factor_evals) => (factor_evals.len() == 1)
                    .then(|| witness.final_eval())
                    .flatten()
                    .map(|witness| (witness, factor_evals[0])),
                SparseFactor::Tensor(_) => None,
            },
        }
    }
}

impl<E: FieldCore + HasUnreducedOps> ExtensionOpeningTables<E> {
    fn accumulate_round(&self, coeff: E, constant: &mut E, quadratic: &mut E) {
        match self {
            Self::Dense { witness, factor } => {
                let (round_constant, round_quadratic) =
                    accumulate_dense_round(witness, factor, coeff);
                *constant += round_constant;
                *quadratic += round_quadratic;
            }
            Self::Sparse { witness, factor } => match factor {
                SparseFactor::Dense(factor_evals) => {
                    witness.accumulate_round(factor_evals, coeff, constant, quadratic);
                }
                SparseFactor::Tensor(factor) => {
                    witness.accumulate_round_with_factor(coeff, constant, quadratic, |pair| {
                        factor.factor_pair(pair)
                    });
                }
            },
        }
    }
}

impl<E: FieldCore + HasUnreducedOps + HasOptimizedFold> ExtensionOpeningTables<E> {
    fn fold_in_place(&mut self, r_round: E) {
        match self {
            Self::Dense { witness, factor } => {
                fold_dense_reduction_tables_in_place(witness, factor, r_round);
            }
            Self::Sparse { witness, factor } => {
                witness.fold_in_place(r_round);
                match factor {
                    SparseFactor::Dense(factor_evals) => {
                        fold_evals_in_place(factor_evals, r_round);
                    }
                    SparseFactor::Tensor(tensor_factor) => {
                        tensor_factor.fold_in_place(r_round);
                        if tensor_factor.is_ready_to_materialize() {
                            *factor = SparseFactor::Dense(tensor_factor.materialize_dense());
                        }
                    }
                }
            }
        }
    }
}

impl<E: FieldCore> ExtensionOpeningReductionTerm<E> {
    /// Construct one term `coeff * sum_x witness(x) * factor(x)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness/factor tables are malformed.
    pub fn new(witness_evals: Vec<E>, factor_evals: Vec<E>, coeff: E) -> Result<Self, AkitaError> {
        validate_reduction_tables(&witness_evals, &factor_evals)?;
        Ok(Self {
            tables: ExtensionOpeningTables::Dense {
                witness: witness_evals,
                factor: factor_evals,
            },
            coeff,
            cached_accumulate: None,
        })
    }

    /// Construct one sparse-witness term `coeff * sum_x witness(x) * factor(x)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the sparse witness and factor table shapes differ.
    pub fn new_sparse(
        witness_evals: SparseExtensionOpeningWitness<E>,
        factor_evals: Vec<E>,
        coeff: E,
    ) -> Result<Self, AkitaError> {
        if witness_evals.table_len() != factor_evals.len() {
            return Err(AkitaError::InvalidSize {
                expected: witness_evals.table_len(),
                actual: factor_evals.len(),
            });
        }
        Ok(Self {
            tables: ExtensionOpeningTables::Sparse {
                witness: witness_evals,
                factor: SparseFactor::Dense(factor_evals),
            },
            coeff,
            cached_accumulate: None,
        })
    }

    /// Construct one sparse-witness term with a lazy transparent tensor factor.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor factor shape and sparse witness domain
    /// differ, or if the tensor opening parameters are malformed.
    pub fn new_sparse_tensor_factor<F>(
        witness_evals: SparseExtensionOpeningWitness<E>,
        tail_point: Vec<E>,
        eta: Vec<E>,
        coeff: E,
        materialize_at: usize,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let factor = TensorEqualityFactor::new::<F>(tail_point, eta, materialize_at)?;
        if witness_evals.table_len() != factor.len() {
            return Err(AkitaError::InvalidSize {
                expected: witness_evals.table_len(),
                actual: factor.len(),
            });
        }
        let factor = if factor.is_ready_to_materialize() {
            SparseFactor::Dense(factor.materialize_dense())
        } else {
            SparseFactor::Tensor(factor)
        };
        Ok(Self {
            tables: ExtensionOpeningTables::Sparse {
                witness: witness_evals,
                factor,
            },
            coeff,
            cached_accumulate: None,
        })
    }

    /// Batching coefficient multiplying this term.
    pub fn coeff(&self) -> E {
        self.coeff
    }

    /// Return final folded witness/factor evaluations after all challenges.
    pub fn final_witness_and_factor_evals(&self) -> Option<(E, E)> {
        self.tables.final_witness_and_factor_evals()
    }
}

impl<E: FieldCore + HasUnreducedOps + HasOptimizedFold> ExtensionOpeningReductionTerm<E> {
    /// Add this term's `coeff`-scaled `(constant, quadratic)` round
    /// contribution into the shared accumulators.
    ///
    /// Consumes the cache filled by the previous round's fused fold when
    /// present; otherwise accumulates directly from the current tables (the
    /// first round, and every round of the sparse/tensor paths).
    pub(super) fn accumulate_into(&mut self, constant: &mut E, quadratic: &mut E) {
        match self.cached_accumulate.take() {
            Some((cached_constant, cached_quadratic)) => {
                *constant += cached_constant;
                *quadratic += cached_quadratic;
            }
            None => {
                self.tables
                    .accumulate_round(self.coeff, constant, quadratic);
            }
        }
    }

    /// Fold this term's tables by one sumcheck challenge.
    ///
    /// For a dense witness and dense factor with at least four entries, fold
    /// and pre-compute the next round's `(constant, quadratic)` in one pass and
    /// cache the `coeff`-scaled result. Every other shape folds in place and
    /// clears the cache.
    pub(super) fn ingest_challenge(&mut self, r_round: E) {
        if self.tables.len() <= 1 {
            return;
        }
        let fused = match &mut self.tables {
            ExtensionOpeningTables::Dense { witness, factor } if witness.len() >= 4 => {
                Some(fused_fold_and_accumulate(witness, factor, r_round))
            }
            _ => None,
        };
        match fused {
            Some((constant, quadratic)) => {
                self.cached_accumulate = Some((self.coeff * constant, self.coeff * quadratic));
            }
            None => {
                self.tables.fold_in_place(r_round);
                self.cached_accumulate = None;
            }
        }
    }
}

#[cfg(test)]
mod merge_free_fast_path_tests {
    use super::*;
    use akita_field::fields::{Prime24Offset3, TowerBasisFp4, TwoNr, UnitNr};
    use akita_field::RandomSampling;
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    type F = Prime24Offset3;
    type E = TowerBasisFp4<F, TwoNr, UnitNr>;

    /// One entry per stride-window at a random within-window offset — the real
    /// `np = 1` EOR witness shape (`stride = onehot_k / width = 2^s`).
    fn build_np1_witness(
        log_chunks: usize,
        s: usize,
        rng: &mut StdRng,
    ) -> SparseExtensionOpeningWitness<E> {
        let stride = 1usize << s;
        let num_chunks = 1usize << log_chunks;
        let table_len = num_chunks * stride;
        let mut entries = Vec::with_capacity(num_chunks);
        for chunk in 0..num_chunks {
            let off = (rng.next_u32() as usize) & (stride - 1);
            entries.push((chunk * stride + off, E::random(rng)));
        }
        SparseExtensionOpeningWitness::new(table_len, entries).unwrap()
    }

    /// Standard multilinear fold of a dense factor table: `A'[p] = (1-r)·A[2p] +
    /// r·A[2p+1]`. Keeps the factor in lock-step with the folding witness.
    fn fold_dense(factor: &mut Vec<E>, r: E) {
        let one_minus = E::one() - r;
        let half = factor.len() / 2;
        for p in 0..half {
            factor[p] = factor[2 * p] * one_minus + factor[2 * p + 1] * r;
        }
        factor.truncate(half);
    }

    /// The closed-form merge-free path must produce byte-identical round messages
    /// and byte-identical folded entries vs the general grouped/realloc path, in
    /// both the sequential and the parallel (`>= SPARSE_PARALLEL_ENTRY_THRESHOLD`,
    /// 2^14 entries) regimes.
    #[test]
    fn merge_free_matches_general_round_by_round() {
        for (log_chunks, s) in [(8usize, 4usize), (14usize, 4usize)] {
            let mut rng = StdRng::seed_from_u64(0x1234_5678 ^ ((log_chunks as u64) << 8));
            let coeff = E::random(&mut rng);

            let mut fast = build_np1_witness(log_chunks, s, &mut rng);
            // Reference: the identical witness with the fast path disabled.
            let mut reference = fast.clone();
            reference.merge_free_rounds_left = 0;

            // The guaranteed plateau is exactly log2(stride) = s, independent of
            // the random within-window offsets.
            assert_eq!(fast.merge_free_rounds_left, s, "unexpected plateau length");

            let mut factor: Vec<E> = (0..fast.table_len()).map(|_| E::random(&mut rng)).collect();

            let rounds = fast.table_len().trailing_zeros() as usize;
            for round in 0..rounds {
                let (mut c_fast, mut q_fast) = (E::zero(), E::zero());
                fast.accumulate_round(&factor, coeff, &mut c_fast, &mut q_fast);
                let (mut c_ref, mut q_ref) = (E::zero(), E::zero());
                reference.accumulate_round(&factor, coeff, &mut c_ref, &mut q_ref);
                assert_eq!(
                    c_fast, c_ref,
                    "constant mismatch (log_chunks={log_chunks}, round={round})"
                );
                assert_eq!(
                    q_fast, q_ref,
                    "quadratic mismatch (log_chunks={log_chunks}, round={round})"
                );

                if round < s {
                    assert!(
                        fast.merge_free_rounds_left > 0,
                        "fast path disengaged during plateau (round={round})"
                    );
                }

                let r = E::random(&mut rng);
                fast.fold_in_place(r);
                reference.fold_in_place(r);
                assert_eq!(fast.table_len(), reference.table_len());
                assert_eq!(
                    fast.entries(),
                    reference.entries(),
                    "folded entries mismatch (log_chunks={log_chunks}, round={round})"
                );
                fold_dense(&mut factor, r);
            }
            assert_eq!(fast.table_len(), 1);
            assert_eq!(fast.merge_free_rounds_left, 0);
        }
    }
}
