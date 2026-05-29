use super::*;

/// One term in a batched extension-opening reduction sumcheck.
#[derive(Debug, Clone)]
pub struct BatchedExtensionOpeningReductionTerm<E: FieldCore> {
    pub(super) current_witness_evals: BatchedExtensionOpeningWitness<E>,
    pub(super) current_factor: BatchedExtensionOpeningFactor<E>,
    pub(super) coeff: E,
}

/// Sparse transformed-witness evaluations for extension-opening reduction.
#[derive(Debug, Clone)]
pub struct SparseExtensionOpeningWitness<E: FieldCore> {
    table_len: usize,
    entries: Vec<(usize, E)>,
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
        Ok(Self {
            table_len,
            entries: combined,
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
        Ok(Self { table_len, entries })
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
    fn pair_aligned_ranges(&self) -> Vec<(usize, usize)> {
        let len = self.entries.len();
        let target_chunks = rayon::current_num_threads() * SPARSE_PARALLEL_CHUNKS_PER_THREAD;
        let chunk_size = len
            .div_ceil(target_chunks)
            .max(SPARSE_PARALLEL_ENTRY_THRESHOLD);
        let mut ranges = Vec::with_capacity(target_chunks.min(len.div_ceil(chunk_size)));
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
        factor_pair: &P,
    ) -> (E, E)
    where
        P: Fn(usize) -> (E, E) + Sync,
    {
        let mut const_accum = E::ProductAccum::zero();
        let mut quad_accum = E::ProductAccum::zero();
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
                quad_accum += w1.mul_to_product_accum(da);
            } else {
                const_accum += w0.mul_to_product_accum(a0);
                quad_accum += (w1 - w0).mul_to_product_accum(da);
            }
        }

        let constant = E::reduce_product_accum(const_accum);
        let quadratic = E::reduce_product_accum(quad_accum);
        (coeff * constant, coeff * quadratic)
    }

    fn accumulate_entries(entries: &[(usize, E)], factor_evals: &[E], coeff: E) -> (E, E) {
        Self::accumulate_entries_with_factor(entries, coeff, &|pair| {
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
        #[cfg(feature = "parallel")]
        let (round_constant, round_quadratic) =
            if self.entries.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
                self.pair_aligned_ranges()
                    .into_par_iter()
                    .map(|(start, end)| {
                        Self::accumulate_entries(&self.entries[start..end], factor_evals, coeff)
                    })
                    .reduce(
                        || (E::zero(), E::zero()),
                        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                    )
            } else {
                Self::accumulate_entries(&self.entries, factor_evals, coeff)
            };
        #[cfg(not(feature = "parallel"))]
        let (round_constant, round_quadratic) =
            Self::accumulate_entries(&self.entries, factor_evals, coeff);
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
        #[cfg(feature = "parallel")]
        let (round_constant, round_quadratic) =
            if self.entries.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
                self.pair_aligned_ranges()
                    .into_par_iter()
                    .map(|(start, end)| {
                        Self::accumulate_entries_with_factor(
                            &self.entries[start..end],
                            coeff,
                            &factor_pair,
                        )
                    })
                    .reduce(
                        || (E::zero(), E::zero()),
                        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                    )
            } else {
                Self::accumulate_entries_with_factor(&self.entries, coeff, &factor_pair)
            };
        #[cfg(not(feature = "parallel"))]
        let (round_constant, round_quadratic) =
            Self::accumulate_entries_with_factor(&self.entries, coeff, &factor_pair);
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
    /// Fields whose wide accumulator is lossy versus `Mul` (e.g. the
    /// `Fp2<Fp64>` schoolbook product, which wraps mod 2^128) leave
    /// `DELAYED_PRODUCT_SUM_IS_EXACT` at `false` and take the per-term path, so
    /// the emitted factor — and the proof — stays unchanged.
    #[inline]
    fn eval_state_at_suffix_fast(&self, state: &[E], suffix_index: usize) -> E {
        if !E::DELAYED_PRODUCT_SUM_IS_EXACT {
            return self.eval_state_at_suffix(state, suffix_index);
        }
        let mut accum = E::ProductAccum::zero();
        for (table, coeff) in self.suffix_tables.iter().zip(state.iter().copied()) {
            accum += coeff.mul_to_product_accum(table[suffix_index]);
        }
        E::reduce_product_accum(accum)
    }

    fn factor_pair(&self, pair: usize) -> (E, E) {
        let low_bits = self.materialize_at - self.round;
        debug_assert!(low_bits > 0);
        let rest_low_bits = low_bits - 1;
        let low_mask = (1usize << rest_low_bits).saturating_sub(1);
        let low_rest = pair & low_mask;
        let suffix_index = pair >> rest_low_bits;
        let low_zero = low_rest << 1;
        let low_one = low_zero | 1;
        (
            self.eval_state_at_suffix_fast(&self.low_states[low_zero], suffix_index),
            self.eval_state_at_suffix_fast(&self.low_states[low_one], suffix_index),
        )
    }
}

#[derive(Debug, Clone)]
pub(super) enum BatchedExtensionOpeningWitness<E: FieldCore> {
    Dense(Vec<E>),
    Sparse(SparseExtensionOpeningWitness<E>),
}

#[derive(Debug, Clone)]
pub(super) enum BatchedExtensionOpeningFactor<E: FieldCore> {
    Dense(Vec<E>),
    Tensor(TensorEqualityFactor<E>),
}

impl<E: FieldCore> BatchedExtensionOpeningFactor<E> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Dense(evals) => evals.len(),
            Self::Tensor(factor) => factor.len(),
        }
    }
}

impl<E: FieldCore> BatchedExtensionOpeningWitness<E> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Dense(evals) => evals.len(),
            Self::Sparse(witness) => witness.table_len(),
        }
    }

    pub(super) fn claim_with_factor(
        &self,
        factor: &BatchedExtensionOpeningFactor<E>,
    ) -> Result<E, AkitaError> {
        match self {
            Self::Dense(evals) => match factor {
                BatchedExtensionOpeningFactor::Dense(factor_evals) => {
                    extension_opening_reduction_claim(evals, factor_evals)
                }
                BatchedExtensionOpeningFactor::Tensor(_) => Err(AkitaError::InvalidInput(
                    "lazy tensor extension-opening factor requires a sparse witness".to_string(),
                )),
            },
            Self::Sparse(witness) => match factor {
                BatchedExtensionOpeningFactor::Dense(factor_evals) => {
                    witness.claim_with_factor(factor_evals)
                }
                BatchedExtensionOpeningFactor::Tensor(factor) => {
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

    pub(super) fn final_eval(&self) -> Option<E> {
        match self {
            Self::Dense(evals) => (evals.len() == 1).then_some(evals[0]),
            Self::Sparse(witness) => witness.final_eval(),
        }
    }
}

impl<E: FieldCore + HasUnreducedOps> BatchedExtensionOpeningWitness<E> {
    pub(super) fn accumulate_round(
        &self,
        factor: &BatchedExtensionOpeningFactor<E>,
        coeff: E,
        constant: &mut E,
        quadratic: &mut E,
    ) {
        match (self, factor) {
            (Self::Dense(witness_evals), BatchedExtensionOpeningFactor::Dense(factor_evals)) => {
                let (round_constant, round_quadratic) =
                    accumulate_dense_round(witness_evals, factor_evals, coeff);
                *constant += round_constant;
                *quadratic += round_quadratic;
            }
            (Self::Sparse(witness), BatchedExtensionOpeningFactor::Dense(factor_evals)) => {
                witness.accumulate_round(factor_evals, coeff, constant, quadratic);
            }
            (Self::Sparse(witness), BatchedExtensionOpeningFactor::Tensor(factor)) => {
                witness.accumulate_round_with_factor(coeff, constant, quadratic, |pair| {
                    factor.factor_pair(pair)
                });
            }
            (Self::Dense(_), BatchedExtensionOpeningFactor::Tensor(_)) => {
                unreachable!("lazy tensor factor is only constructed for sparse witnesses")
            }
        }
    }
}

impl<E: FieldCore + HasUnreducedOps + HasOptimizedFold> BatchedExtensionOpeningWitness<E> {
    pub(super) fn fold_with_factor_in_place(
        &mut self,
        factor: &mut BatchedExtensionOpeningFactor<E>,
        r_round: E,
    ) {
        match self {
            Self::Dense(witness_evals) => match factor {
                BatchedExtensionOpeningFactor::Dense(factor_evals) => {
                    fold_dense_reduction_tables_in_place(witness_evals, factor_evals, r_round);
                }
                BatchedExtensionOpeningFactor::Tensor(_) => {
                    unreachable!("lazy tensor factor is only constructed for sparse witnesses")
                }
            },
            Self::Sparse(witness) => {
                witness.fold_in_place(r_round);
                match factor {
                    BatchedExtensionOpeningFactor::Dense(factor_evals) => {
                        fold_evals_in_place(factor_evals, r_round);
                    }
                    BatchedExtensionOpeningFactor::Tensor(tensor_factor) => {
                        tensor_factor.fold_in_place(r_round);
                        if tensor_factor.is_ready_to_materialize() {
                            let dense = tensor_factor.materialize_dense();
                            *factor = BatchedExtensionOpeningFactor::Dense(dense);
                        }
                    }
                }
            }
        }
    }
}

impl<E: FieldCore> BatchedExtensionOpeningReductionTerm<E> {
    /// Construct one term `coeff * sum_x witness(x) * factor(x)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness/factor tables are malformed.
    pub fn new(witness_evals: Vec<E>, factor_evals: Vec<E>, coeff: E) -> Result<Self, AkitaError> {
        validate_reduction_tables(&witness_evals, &factor_evals)?;
        Ok(Self {
            current_witness_evals: BatchedExtensionOpeningWitness::Dense(witness_evals),
            current_factor: BatchedExtensionOpeningFactor::Dense(factor_evals),
            coeff,
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
            current_witness_evals: BatchedExtensionOpeningWitness::Sparse(witness_evals),
            current_factor: BatchedExtensionOpeningFactor::Dense(factor_evals),
            coeff,
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
        let current_factor = if factor.is_ready_to_materialize() {
            BatchedExtensionOpeningFactor::Dense(factor.materialize_dense())
        } else {
            BatchedExtensionOpeningFactor::Tensor(factor)
        };
        Ok(Self {
            current_witness_evals: BatchedExtensionOpeningWitness::Sparse(witness_evals),
            current_factor,
            coeff,
        })
    }

    /// Batching coefficient multiplying this term.
    pub fn coeff(&self) -> E {
        self.coeff
    }

    /// Return final folded witness/factor evaluations after all challenges.
    pub fn final_witness_and_factor_evals(&self) -> Option<(E, E)> {
        match &self.current_factor {
            BatchedExtensionOpeningFactor::Dense(factor_evals) => (factor_evals.len() == 1)
                .then(|| self.current_witness_evals.final_eval())
                .flatten()
                .map(|witness| (witness, factor_evals[0])),
            BatchedExtensionOpeningFactor::Tensor(_) => None,
        }
    }
}
