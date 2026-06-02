use super::*;

#[cfg(feature = "parallel")]
const DENSE_PARALLEL_PAIR_THRESHOLD: usize = 1 << 14;

pub(crate) fn accumulate_dense_round<E: FieldCore + HasUnreducedOps>(
    witness_evals: &[E],
    factor_evals: &[E],
    coeff: E,
) -> (E, E) {
    let _span = tracing::trace_span!(
        "dense_extension_reduction_accumulate_round",
        table_len = witness_evals.len()
    )
    .entered();
    debug_assert_eq!(witness_evals.len(), factor_evals.len());
    if coeff == E::zero() {
        return (E::zero(), E::zero());
    }

    // Sum the wide products in `E::ProductAccum` only when the field has proven
    // that delayed reduction is exact for these batch sizes; otherwise reduce
    // each product immediately so the coefficients stay byte-identical to
    // per-term `Mul` (the `DELAYED_PRODUCT_SUM_IS_EXACT` contract).
    let (constant, quadratic) = if E::DELAYED_PRODUCT_SUM_IS_EXACT {
        accumulate_dense_round_with::<E, DelayedDeg2<E>>(witness_evals, factor_evals)
    } else {
        accumulate_dense_round_with::<E, DirectDeg2<E>>(witness_evals, factor_evals)
    };
    (coeff * constant, coeff * quadratic)
}

fn accumulate_dense_round_with<E, A>(witness_evals: &[E], factor_evals: &[E]) -> (E, E)
where
    E: FieldCore + HasUnreducedOps,
    A: Deg2RoundAccum<E>,
{
    let half = witness_evals.len() / 2;

    #[cfg(feature = "parallel")]
    {
        if half >= DENSE_PARALLEL_PAIR_THRESHOLD {
            return (0..half)
                .into_par_iter()
                .fold(A::zero, |mut acc, i| {
                    let w0 = witness_evals[2 * i];
                    let w1 = witness_evals[2 * i + 1];
                    let a0 = factor_evals[2 * i];
                    let a1 = factor_evals[2 * i + 1];

                    acc.add_constant_product(w0, a0);
                    acc.add_quadratic_product(w1 - w0, a1 - a0);
                    acc
                })
                .reduce(A::zero, A::merge)
                .finish();
        }
    }

    let mut acc = A::zero();
    for i in 0..half {
        let w0 = witness_evals[2 * i];
        let w1 = witness_evals[2 * i + 1];
        let a0 = factor_evals[2 * i];
        let a1 = factor_evals[2 * i + 1];

        acc.add_constant_product(w0, a0);
        acc.add_quadratic_product(w1 - w0, a1 - a0);
    }
    acc.finish()
}

pub(crate) fn fold_dense_reduction_tables_in_place<E: HasUnreducedOps + HasOptimizedFold>(
    witness_evals: &mut Vec<E>,
    factor_evals: &mut Vec<E>,
    r_round: E,
) {
    let _span = tracing::trace_span!(
        "fold_dense_reduction_tables_in_place",
        table_len = witness_evals.len()
    )
    .entered();
    debug_assert_eq!(witness_evals.len(), factor_evals.len());
    fold_evals_in_place(witness_evals, r_round);
    fold_evals_in_place(factor_evals, r_round);
}

/// Fold both tables by one variable AND pre-compute the next round's
/// `(constant, quadratic)` accumulation in a single pass over the data.
fn fused_fold_and_accumulate<E: HasUnreducedOps + HasOptimizedFold>(
    witness_evals: &mut Vec<E>,
    factor_evals: &mut Vec<E>,
    r_round: E,
) -> (E, E) {
    let _span = tracing::trace_span!("fused_fold_and_accumulate", table_len = witness_evals.len())
        .entered();
    debug_assert_eq!(witness_evals.len(), factor_evals.len());
    debug_assert!(witness_evals.len().is_power_of_two());
    debug_assert!(witness_evals.len() >= 4);

    // The fold itself (`E::fold_one`) is always exact; only the product
    // accumulation respects `DELAYED_PRODUCT_SUM_IS_EXACT`, matching
    // `accumulate_dense_round`.
    if E::DELAYED_PRODUCT_SUM_IS_EXACT {
        fused_fold_and_accumulate_with::<E, DelayedDeg2<E>>(witness_evals, factor_evals, r_round)
    } else {
        fused_fold_and_accumulate_with::<E, DirectDeg2<E>>(witness_evals, factor_evals, r_round)
    }
}

fn fused_fold_and_accumulate_with<E, A>(
    witness_evals: &mut Vec<E>,
    factor_evals: &mut Vec<E>,
    r_round: E,
) -> (E, E)
where
    E: FieldCore + HasUnreducedOps + HasOptimizedFold,
    A: Deg2RoundAccum<E>,
{
    let half = witness_evals.len() / 2;
    let quarter = half / 2;
    let ctx = E::precompute_fold(r_round);

    #[cfg(feature = "parallel")]
    {
        if quarter >= DENSE_PARALLEL_PAIR_THRESHOLD {
            let mut folded_w = Vec::<E>::with_capacity(half);
            let mut folded_f = Vec::<E>::with_capacity(half);
            // SAFETY: both vectors are allocated with capacity `half`. `half` is
            // even (table length is a power of two >= 4), so the `par_chunks_mut(2)`
            // loop below yields exactly `quarter` chunks of length 2 and writes all
            // `half` slots before the first read (`*witness_evals = folded_w`).
            // `E: FieldCore` is `Copy` with a trivial drop, so overwriting the
            // uninitialized slots is sound.
            unsafe {
                folded_w.set_len(half);
                folded_f.set_len(half);
            }

            let acc = {
                let input_w: &[E] = witness_evals;
                let input_f: &[E] = factor_evals;

                folded_w
                    .par_chunks_mut(2)
                    .zip(folded_f.par_chunks_mut(2))
                    .enumerate()
                    .fold(A::zero, |mut acc, (i, (w_out, f_out))| {
                        let fw0 = E::fold_one(&ctx, input_w[4 * i], input_w[4 * i + 1]);
                        let fw1 = E::fold_one(&ctx, input_w[4 * i + 2], input_w[4 * i + 3]);
                        let fa0 = E::fold_one(&ctx, input_f[4 * i], input_f[4 * i + 1]);
                        let fa1 = E::fold_one(&ctx, input_f[4 * i + 2], input_f[4 * i + 3]);

                        acc.add_constant_product(fw0, fa0);
                        acc.add_quadratic_product(fw1 - fw0, fa1 - fa0);

                        w_out[0] = fw0;
                        w_out[1] = fw1;
                        f_out[0] = fa0;
                        f_out[1] = fa1;

                        acc
                    })
                    .reduce(A::zero, A::merge)
            };

            *witness_evals = folded_w;
            *factor_evals = folded_f;
            return acc.finish();
        }
    }

    let mut acc = A::zero();
    for i in 0..quarter {
        let fw0 = E::fold_one(&ctx, witness_evals[4 * i], witness_evals[4 * i + 1]);
        let fw1 = E::fold_one(&ctx, witness_evals[4 * i + 2], witness_evals[4 * i + 3]);
        let fa0 = E::fold_one(&ctx, factor_evals[4 * i], factor_evals[4 * i + 1]);
        let fa1 = E::fold_one(&ctx, factor_evals[4 * i + 2], factor_evals[4 * i + 3]);

        acc.add_constant_product(fw0, fa0);
        acc.add_quadratic_product(fw1 - fw0, fa1 - fa0);

        witness_evals[2 * i] = fw0;
        witness_evals[2 * i + 1] = fw1;
        factor_evals[2 * i] = fa0;
        factor_evals[2 * i + 1] = fa1;
    }
    witness_evals.truncate(half);
    factor_evals.truncate(half);
    acc.finish()
}

/// Prover state for the degree-two extension-opening reduction sumcheck.
///
/// Uses a fused fold+accumulate strategy: after each fold, the next round's
/// accumulation is pre-computed in the same pass, avoiding a redundant read
/// of the folded table.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionProver<E: FieldCore> {
    current_witness_evals: Vec<E>,
    current_factor_evals: Vec<E>,
    cached_accumulate: Option<(E, E)>,
    input_claim: E,
    num_rounds: usize,
}

impl<E: FieldCore> ExtensionOpeningReductionProver<E> {
    /// Construct a prover from transformed-witness and transparent-factor
    /// Boolean-hypercube evaluation tables.
    ///
    /// # Errors
    ///
    /// Returns an error if the tables do not have the same nonzero power-of-two
    /// length.
    pub fn new(witness_evals: Vec<E>, factor_evals: Vec<E>) -> Result<Self, AkitaError> {
        let input_claim = extension_opening_reduction_claim(&witness_evals, &factor_evals)?;
        let num_rounds = num_rounds_from_table_len(witness_evals.len())?;
        Ok(Self {
            current_witness_evals: witness_evals,
            current_factor_evals: factor_evals,
            cached_accumulate: None,
            input_claim,
            num_rounds,
        })
    }

    /// Number of sumcheck rounds for this prover instance.
    pub fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    /// Initial claim for this prover instance.
    pub fn input_claim(&self) -> E {
        self.input_claim
    }

    /// Return the final folded witness and factor evaluations after all
    /// challenges have been ingested.
    pub fn final_witness_and_factor_evals(&self) -> Option<(E, E)> {
        (self.current_witness_evals.len() == 1 && self.current_factor_evals.len() == 1)
            .then(|| (self.current_witness_evals[0], self.current_factor_evals[0]))
    }
}

impl<E: FieldCore + HasUnreducedOps + HasOptimizedFold> SumcheckInstanceProver<E>
    for ExtensionOpeningReductionProver<E>
{
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        EXTENSION_OPENING_REDUCTION_DEGREE
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, round: usize, previous_claim: E) -> UniPoly<E> {
        debug_assert_eq!(
            self.current_witness_evals.len(),
            1usize << (self.num_rounds - round)
        );
        debug_assert_eq!(
            self.current_factor_evals.len(),
            self.current_witness_evals.len()
        );

        let (constant, quadratic) = self.cached_accumulate.take().unwrap_or_else(|| {
            accumulate_dense_round(
                &self.current_witness_evals,
                &self.current_factor_evals,
                E::one(),
            )
        });
        let linear = previous_claim - constant - constant - quadratic;

        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: E) {
        let current_len = self.current_witness_evals.len();
        if current_len <= 1 {
            return;
        }
        if current_len >= 4 {
            let (constant, quadratic) = fused_fold_and_accumulate(
                &mut self.current_witness_evals,
                &mut self.current_factor_evals,
                r_round,
            );
            self.cached_accumulate = Some((constant, quadratic));
        } else {
            fold_dense_reduction_tables_in_place(
                &mut self.current_witness_evals,
                &mut self.current_factor_evals,
                r_round,
            );
        }
    }
}
