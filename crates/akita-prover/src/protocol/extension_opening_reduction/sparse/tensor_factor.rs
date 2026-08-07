use super::*;
use core::ops::Range;

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
pub(in crate::protocol::extension_opening_reduction) struct TensorEqualityFactor<E: FieldCore> {
    table_vars: usize,
    round: usize,
    materialize_at: usize,
    prefix_state: Vec<E>,
    transitions: Vec<TensorFactorTransition<E>>,
    suffix_tables: Vec<Vec<E>>,
    low_pair_states: Vec<E>,
}

impl<E: FieldCore> TensorEqualityFactor<E> {
    pub(super) fn new<F>(
        tail_point: Vec<E>,
        eta: Vec<E>,
        materialize_at: usize,
    ) -> Result<Self, AkitaError>
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
            low_pair_states: Vec::new(),
        };
        factor.rebuild_low_pairs();
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

    pub(super) fn len(&self) -> usize {
        1usize << (self.table_vars - self.round)
    }

    pub(super) fn is_ready_to_materialize(&self) -> bool {
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

    fn rebuild_low_pairs(&mut self) {
        let low_bits = self.materialize_at.saturating_sub(self.round);
        if low_bits == 0 {
            self.low_pair_states.clear();
            return;
        }
        let count = 1usize << low_bits;
        let width = self.prefix_state.len();
        let mut low_pair_states = Vec::with_capacity(count * width);
        for pair in 0..count / 2 {
            let zero = self.boolean_state(pair << 1, low_bits);
            let one = self.boolean_state((pair << 1) | 1, low_bits);
            low_pair_states.extend_from_slice(&zero);
            low_pair_states.extend(one.iter().zip(zero.iter()).map(|(&one, &zero)| one - zero));
        }
        self.low_pair_states = low_pair_states;
    }

    fn boolean_state(&self, low: usize, low_bits: usize) -> Vec<E> {
        let mut state = self.prefix_state.clone();
        for bit_idx in 0..low_bits {
            let bit = (low >> bit_idx) & 1;
            state = Self::apply_boolean_transition(
                &state,
                &self.transitions[self.round + bit_idx],
                bit,
            );
        }
        state
    }

    fn low_pair(&self, pair: usize) -> (&[E], &[E]) {
        let width = self.prefix_state.len();
        let start = pair * 2 * width;
        let zero = &self.low_pair_states[start..start + width];
        let delta = &self.low_pair_states[start + width..start + 2 * width];
        (zero, delta)
    }

    fn eval_state_at_suffix(&self, state: &[E], suffix_index: usize) -> E {
        self.suffix_tables
            .iter()
            .zip(state.iter().copied())
            .fold(E::zero(), |acc, (table, coeff)| {
                acc + coeff * table[suffix_index]
            })
    }

    pub(super) fn factor_at_index(&self, index: usize) -> E {
        let low_bits = self.materialize_at.saturating_sub(self.round);
        if low_bits == 0 {
            return self.eval_state_at_suffix(&self.prefix_state, index);
        }
        let low_mask = (1usize << low_bits) - 1;
        let low = index & low_mask;
        let suffix_index = index >> low_bits;
        let (state_zero, state_delta) = self.low_pair(low >> 1);
        if low & 1 == 0 {
            self.eval_state_at_suffix(state_zero, suffix_index)
        } else {
            self.suffix_tables
                .iter()
                .zip(state_zero.iter().zip(state_delta.iter()))
                .fold(E::zero(), |acc, (table, (&zero, &delta))| {
                    acc + (zero + delta) * table[suffix_index]
                })
        }
    }

    pub(super) fn fold_in_place(&mut self, r_round: E) {
        if self.len() <= 1 {
            return;
        }
        debug_assert!(self.round < self.materialize_at);
        self.prefix_state =
            Self::apply_transition(&self.prefix_state, &self.transitions[self.round], r_round);
        self.round += 1;
        self.rebuild_low_pairs();
    }

    pub(super) fn materialize_dense(&self) -> Vec<E> {
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

struct GroupedRoundAccumulator<E: FieldCore + HasUnreducedOps, const N: usize> {
    suffix_index: Option<usize>,
    constant: [E::ProductAccum; N],
    quadratic: [E::ProductAccum; N],
    round_constant: E::ProductAccum,
    round_quadratic: E::ProductAccum,
}

impl<E: FieldCore + HasUnreducedOps, const N: usize> GroupedRoundAccumulator<E, N> {
    fn new() -> Self {
        Self {
            suffix_index: None,
            constant: [E::ProductAccum::zero(); N],
            quadratic: [E::ProductAccum::zero(); N],
            round_constant: E::ProductAccum::zero(),
            round_quadratic: E::ProductAccum::zero(),
        }
    }

    fn add_pair(
        &mut self,
        factor: &TensorEqualityFactor<E>,
        pair: usize,
        witness_zero: E,
        witness_one: E,
    ) {
        let rest_low_bits = factor.materialize_at - factor.round - 1;
        let suffix_index = pair >> rest_low_bits;
        if self.suffix_index != Some(suffix_index) {
            self.flush(factor);
            self.suffix_index = Some(suffix_index);
        }

        let low_mask = (1usize << rest_low_bits).saturating_sub(1);
        let (state_zero, state_delta) = factor.low_pair(pair & low_mask);
        let witness_delta = witness_one - witness_zero;
        if witness_zero != E::zero() {
            for (column, &state_zero) in state_zero.iter().enumerate() {
                self.constant[column] += witness_zero.mul_to_product_accum(state_zero);
            }
        }
        for (column, &state_delta) in state_delta.iter().enumerate() {
            self.quadratic[column] += witness_delta.mul_to_product_accum(state_delta);
        }
    }

    fn flush(&mut self, factor: &TensorEqualityFactor<E>) {
        let Some(suffix_index) = self.suffix_index else {
            return;
        };
        for column in 0..N {
            let suffix = factor.suffix_tables[column][suffix_index];
            self.round_constant +=
                E::reduce_product_accum(self.constant[column]).mul_to_product_accum(suffix);
            self.round_quadratic +=
                E::reduce_product_accum(self.quadratic[column]).mul_to_product_accum(suffix);
        }
        self.constant = [E::ProductAccum::zero(); N];
        self.quadratic = [E::ProductAccum::zero(); N];
    }

    fn finish(mut self, factor: &TensorEqualityFactor<E>) -> (E, E) {
        self.flush(factor);
        (
            E::reduce_product_accum(self.round_constant),
            E::reduce_product_accum(self.round_quadratic),
        )
    }
}

impl<E: FieldCore + HasUnreducedOps> TensorEqualityFactor<E> {
    pub(super) fn supports_grouped_rounds(&self) -> bool {
        E::DELAYED_PRODUCT_SUM_IS_EXACT && matches!(self.prefix_state.len(), 1 | 2 | 4 | 8)
    }

    pub(super) fn supports_grouped_round_after_fold(&self) -> bool {
        self.supports_grouped_rounds() && self.round + 1 < self.materialize_at
    }

    /// Compute a sparse round by grouping witness pairs that share one suffix.
    ///
    /// The ordinary row loop constructs and reduces both factor children for
    /// every sparse pair. This operation first sums the witness-weighted low
    /// states for a shared suffix, then applies that suffix once. The number of
    /// factor reductions therefore scales with suffix groups instead of sparse
    /// rows.
    pub(super) fn compute_grouped_round<F>(
        &self,
        witness: &SparseExtensionOpeningWitness<F, E>,
        rows: Range<usize>,
    ) -> (E, E)
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        debug_assert!(self.supports_grouped_rounds());
        match self.prefix_state.len() {
            1 => self.compute_grouped_round_with_width::<F, 1>(witness, rows),
            2 => self.compute_grouped_round_with_width::<F, 2>(witness, rows),
            4 => self.compute_grouped_round_with_width::<F, 4>(witness, rows),
            8 => self.compute_grouped_round_with_width::<F, 8>(witness, rows),
            _ => unreachable!("grouped tensor round requires a supported extension width"),
        }
    }

    fn compute_grouped_round_with_width<F, const N: usize>(
        &self,
        witness: &SparseExtensionOpeningWitness<F, E>,
        rows: Range<usize>,
    ) -> (E, E)
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        debug_assert_eq!(self.prefix_state.len(), N);
        let mut round = GroupedRoundAccumulator::<E, N>::new();
        let mut row = rows.start;
        while row < rows.end {
            let pair = witness.indices[row] >> 1;
            let mut witness_zero = E::zero();
            let mut witness_one = E::zero();
            while row < rows.end && witness.indices[row] >> 1 == pair {
                let value = witness.values.evaluation(row);
                if witness.indices[row] & 1 == 0 {
                    witness_zero += value;
                } else {
                    witness_one += value;
                }
                row += 1;
            }
            round.add_pair(self, pair, witness_zero, witness_one);
        }
        round.finish(self)
    }

    /// Fold and compact a sparse witness while computing its next grouped
    /// tensor round in the same row pass.
    pub(super) fn fold_and_compute_grouped_round<F>(
        &self,
        witness: &mut SparseExtensionOpeningWitness<F, E>,
        challenge: E,
    ) -> (E, E)
    where
        F: FieldCore,
        E: ExtField<F> + HasOptimizedFold,
    {
        debug_assert!(self.supports_grouped_rounds());
        let round = match self.prefix_state.len() {
            1 => self.fold_and_compute_grouped_round_with_width::<F, 1>(witness, challenge),
            2 => self.fold_and_compute_grouped_round_with_width::<F, 2>(witness, challenge),
            4 => self.fold_and_compute_grouped_round_with_width::<F, 4>(witness, challenge),
            8 => self.fold_and_compute_grouped_round_with_width::<F, 8>(witness, challenge),
            _ => unreachable!("grouped tensor round requires a supported extension width"),
        };
        witness.table_len /= 2;
        witness.merge_free_rounds_left = witness.merge_free_rounds_left.saturating_sub(1);
        round
    }

    fn fold_and_compute_grouped_round_with_width<F, const N: usize>(
        &self,
        witness: &mut SparseExtensionOpeningWitness<F, E>,
        challenge: E,
    ) -> (E, E)
    where
        F: FieldCore,
        E: ExtField<F> + HasOptimizedFold,
    {
        debug_assert_eq!(self.prefix_state.len(), N);
        let fold = E::precompute_fold(challenge);
        let mut round = GroupedRoundAccumulator::<E, N>::new();
        let mut input_row = 0;
        let mut output_row = 0;
        let mut next_pair = None;
        let mut next_zero = E::zero();
        let mut next_one = E::zero();

        while input_row < witness.indices.len() {
            let pair = witness.indices[input_row] >> 1;
            let mut witness_zero = E::zero();
            let mut witness_one = E::zero();
            while input_row < witness.indices.len() && witness.indices[input_row] >> 1 == pair {
                let value = witness.values.evaluation(input_row);
                if witness.indices[input_row] & 1 == 0 {
                    witness_zero += value;
                } else {
                    witness_one += value;
                }
                input_row += 1;
            }

            let folded = E::fold_one(&fold, witness_zero, witness_one);
            if folded == E::zero() {
                continue;
            }
            witness.indices[output_row] = pair;
            witness.values.set_evaluation(output_row, folded);
            output_row += 1;

            let pair_for_next_round = pair >> 1;
            if next_pair != Some(pair_for_next_round) {
                if let Some(previous_pair) = next_pair {
                    round.add_pair(self, previous_pair, next_zero, next_one);
                }
                next_pair = Some(pair_for_next_round);
                next_zero = E::zero();
                next_one = E::zero();
            }
            if pair & 1 == 0 {
                next_zero = folded;
            } else {
                next_one = folded;
            }
        }

        if let Some(previous_pair) = next_pair {
            round.add_pair(self, previous_pair, next_zero, next_one);
        }
        witness.indices.truncate(output_row);
        witness.values.truncate(output_row);
        round.finish(self)
    }

    /// Factor inner product `sum_i state[i] * suffix_tables[i][suffix_index]`,
    /// reducing once at the end when the field's product accumulator is exact
    /// w.r.t. `Mul`, and otherwise falling back to the per-term
    /// [`Self::eval_state_at_suffix`].
    ///
    /// On the exact path (e.g. the fp32 `FpExt4<Fp32>` campaign field)
    /// each product is widened into `E::ProductAccum` and the
    /// `state.len() == E::EXT_DEGREE` terms are summed before a single
    /// `reduce_product_accum`. The per-coefficient reduction is additive over
    /// the accumulator and the wide sum cannot overflow (`EXT_DEGREE` is a small
    /// power of two — 4 here — far below the accumulator's >= 2^63 headroom), so
    /// the result is byte-identical to `eval_state_at_suffix`.
    ///
    /// Fields whose wide accumulator is lossy versus `Mul` leave
    /// `DELAYED_PRODUCT_SUM_IS_EXACT` at `false` and take the per-term path, so
    /// the emitted factor, and the proof, stay unchanged. `FpExt2<Fp64>` opts into
    /// the exact path only because its accumulator keeps the carry above bit
    /// 128 explicitly.
    ///
    /// The stored low pair is `(state_zero, state_one - state_zero)`. Both
    /// inner products read the same `suffix_tables[j][suffix_index]` column.
    /// One pass computes `a0` and `a1 - a0`, then reconstructs `a1`. This is the
    /// same pair shape consumed by the grouped round polynomial.
    pub(super) fn factor_pair(&self, pair: usize) -> (E, E) {
        let low_bits = self.materialize_at - self.round;
        debug_assert!(low_bits > 0);
        let rest_low_bits = low_bits - 1;
        let low_mask = (1usize << rest_low_bits).saturating_sub(1);
        let low_rest = pair & low_mask;
        let suffix_index = pair >> rest_low_bits;
        let (state_zero, state_delta) = self.low_pair(low_rest);

        if !E::DELAYED_PRODUCT_SUM_IS_EXACT {
            let zero = self.eval_state_at_suffix(state_zero, suffix_index);
            let delta = self.eval_state_at_suffix(state_delta, suffix_index);
            return (zero, zero + delta);
        }

        let (accum_zero, accum_delta) = match state_zero.len() {
            1 => self.factor_pair_product_accumulators::<1>(state_zero, state_delta, suffix_index),
            2 => self.factor_pair_product_accumulators::<2>(state_zero, state_delta, suffix_index),
            4 => self.factor_pair_product_accumulators::<4>(state_zero, state_delta, suffix_index),
            8 => self.factor_pair_product_accumulators::<8>(state_zero, state_delta, suffix_index),
            _ => {
                self.factor_pair_product_accumulators_dynamic(state_zero, state_delta, suffix_index)
            }
        };
        let zero = E::reduce_product_accum(accum_zero);
        let delta = E::reduce_product_accum(accum_delta);
        (zero, zero + delta)
    }

    fn factor_pair_product_accumulators<const N: usize>(
        &self,
        state_zero: &[E],
        state_delta: &[E],
        suffix_index: usize,
    ) -> (E::ProductAccum, E::ProductAccum) {
        debug_assert_eq!(state_zero.len(), N);
        debug_assert_eq!(state_delta.len(), N);
        debug_assert_eq!(self.suffix_tables.len(), N);
        let mut accum_zero = E::ProductAccum::zero();
        let mut accum_delta = E::ProductAccum::zero();
        for index in 0..N {
            let column = self.suffix_tables[index][suffix_index];
            accum_zero += state_zero[index].mul_to_product_accum(column);
            accum_delta += state_delta[index].mul_to_product_accum(column);
        }
        (accum_zero, accum_delta)
    }

    fn factor_pair_product_accumulators_dynamic(
        &self,
        state_zero: &[E],
        state_delta: &[E],
        suffix_index: usize,
    ) -> (E::ProductAccum, E::ProductAccum) {
        let mut accum_zero = E::ProductAccum::zero();
        let mut accum_delta = E::ProductAccum::zero();
        for ((table, &coeff_zero), &coeff_delta) in self
            .suffix_tables
            .iter()
            .zip(state_zero.iter())
            .zip(state_delta.iter())
        {
            let column = table[suffix_index];
            accum_zero += coeff_zero.mul_to_product_accum(column);
            accum_delta += coeff_delta.mul_to_product_accum(column);
        }
        (accum_zero, accum_delta)
    }
}

/// Transparent factor for a sparse-witness term.
///
/// The lazy [`TensorEqualityFactor`] is only ever paired with a sparse witness,
/// so it lives inside the sparse case rather than as a standalone factor. This
/// is what makes the `(dense witness, tensor factor)` combination unrepresentable.
#[derive(Debug, Clone)]
pub(in crate::protocol::extension_opening_reduction) enum SparseFactor<E: FieldCore> {
    Dense(Vec<E>),
    Tensor(TensorEqualityFactor<E>),
}
