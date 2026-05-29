use super::*;

#[cfg(feature = "parallel")]
const DENSE_PARALLEL_PAIR_THRESHOLD: usize = 1 << 14;

pub(crate) fn accumulate_dense_round<E: FieldCore>(
    witness_evals: &[E],
    factor_evals: &[E],
    coeff: E,
) -> (E, E, E) {
    let _span = tracing::trace_span!(
        "dense_extension_reduction_accumulate_round",
        table_len = witness_evals.len()
    )
    .entered();
    debug_assert_eq!(witness_evals.len(), factor_evals.len());
    let half = witness_evals.len() / 2;
    if coeff == E::zero() {
        return (E::zero(), E::zero(), E::zero());
    }

    #[cfg(feature = "parallel")]
    {
        if half >= DENSE_PARALLEL_PAIR_THRESHOLD {
            let (constant, linear, quadratic) = (0..half)
                .into_par_iter()
                .fold(
                    || (E::zero(), E::zero(), E::zero()),
                    |(mut constant, mut linear, mut quadratic), i| {
                        let w0 = witness_evals[2 * i];
                        let w1 = witness_evals[2 * i + 1];
                        let a0 = factor_evals[2 * i];
                        let a1 = factor_evals[2 * i + 1];
                        let dw = w1 - w0;
                        let da = a1 - a0;

                        constant += w0 * a0;
                        linear += dw * a0 + w0 * da;
                        quadratic += dw * da;
                        (constant, linear, quadratic)
                    },
                )
                .reduce(
                    || (E::zero(), E::zero(), E::zero()),
                    |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2),
                );
            return (coeff * constant, coeff * linear, coeff * quadratic);
        }
    }

    let mut constant = E::zero();
    let mut linear = E::zero();
    let mut quadratic = E::zero();
    for i in 0..half {
        let w0 = witness_evals[2 * i];
        let w1 = witness_evals[2 * i + 1];
        let a0 = factor_evals[2 * i];
        let a1 = factor_evals[2 * i + 1];
        let dw = w1 - w0;
        let da = a1 - a0;

        constant += w0 * a0;
        linear += dw * a0 + w0 * da;
        quadratic += dw * da;
    }
    (coeff * constant, coeff * linear, coeff * quadratic)
}

pub(crate) fn fold_dense_reduction_tables_in_place<E: FieldCore>(
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
    debug_assert!(witness_evals.len().is_power_of_two());
    debug_assert!(witness_evals.len() >= 2);
    let half = witness_evals.len() / 2;
    #[cfg(feature = "parallel")]
    {
        if half >= DENSE_PARALLEL_PAIR_THRESHOLD {
            let fold_pair = |pair: &[E]| pair[0] + r_round * (pair[1] - pair[0]);
            let (folded_witness, folded_factor) = rayon::join(
                || witness_evals.par_chunks_exact(2).map(fold_pair).collect(),
                || factor_evals.par_chunks_exact(2).map(fold_pair).collect(),
            );
            *witness_evals = folded_witness;
            *factor_evals = folded_factor;
            return;
        }
    }
    for i in 0..half {
        witness_evals[i] =
            witness_evals[2 * i] + r_round * (witness_evals[2 * i + 1] - witness_evals[2 * i]);
        factor_evals[i] =
            factor_evals[2 * i] + r_round * (factor_evals[2 * i + 1] - factor_evals[2 * i]);
    }
    witness_evals.truncate(half);
    factor_evals.truncate(half);
}

/// Prover state for the degree-two extension-opening reduction sumcheck.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionProver<E: FieldCore> {
    current_witness_evals: Vec<E>,
    current_factor_evals: Vec<E>,
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
            input_claim,
            num_rounds,
        })
    }

    /// Return the final folded witness and factor evaluations after all
    /// challenges have been ingested.
    pub fn final_witness_and_factor_evals(&self) -> Option<(E, E)> {
        (self.current_witness_evals.len() == 1 && self.current_factor_evals.len() == 1)
            .then(|| (self.current_witness_evals[0], self.current_factor_evals[0]))
    }
}

impl<E: FieldCore> SumcheckInstanceProver<E> for ExtensionOpeningReductionProver<E> {
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

        let (constant, linear, quadratic) = accumulate_dense_round(
            &self.current_witness_evals,
            &self.current_factor_evals,
            E::one(),
        );

        let poly = UniPoly::from_coeffs(vec![constant, linear, quadratic]);
        debug_assert_eq!(
            poly.evaluate(&E::zero()) + poly.evaluate(&E::one()),
            previous_claim
        );
        poly
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: E) {
        if self.current_witness_evals.len() > 1 {
            fold_dense_reduction_tables_in_place(
                &mut self.current_witness_evals,
                &mut self.current_factor_evals,
                r_round,
            );
        }
    }
}
