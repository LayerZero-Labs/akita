use super::*;

/// Prover state for one batched degree-two extension-opening reduction.
#[derive(Debug, Clone)]
pub struct BatchedExtensionOpeningReductionProver<E: FieldCore> {
    terms: Vec<BatchedExtensionOpeningReductionTerm<E>>,
    input_claim: E,
    num_rounds: usize,
}

impl<E: FieldCore> BatchedExtensionOpeningReductionProver<E> {
    /// Construct a batched prover from terms sharing one Boolean domain.
    ///
    /// The caller supplies the claimed input sum. This avoids recomputing it
    /// in protocol paths that already derived the claim while preparing the
    /// transcript-bound reduction.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no terms or their table lengths differ.
    pub fn new(
        terms: Vec<BatchedExtensionOpeningReductionTerm<E>>,
        input_claim: E,
    ) -> Result<Self, AkitaError> {
        let first = terms.first().ok_or_else(|| {
            AkitaError::InvalidInput(
                "batched extension-opening reduction requires at least one term".to_string(),
            )
        })?;
        let table_len = first.current_witness_evals.len();
        let num_rounds = num_rounds_from_table_len(table_len)?;
        for term in &terms {
            if term.current_witness_evals.len() != table_len
                || term.current_factor.len() != table_len
            {
                return Err(AkitaError::InvalidSize {
                    expected: table_len,
                    actual: term
                        .current_witness_evals
                        .len()
                        .max(term.current_factor.len()),
                });
            }
        }
        Ok(Self {
            terms,
            input_claim,
            num_rounds,
        })
    }

    /// Compute the input sum represented by a set of batched terms.
    ///
    /// This is useful for tests and standalone callers that do not already
    /// have an independently derived input claim.
    ///
    /// # Errors
    ///
    /// Returns an error if any term has malformed witness/factor tables.
    pub fn input_claim_from_terms(
        terms: &[BatchedExtensionOpeningReductionTerm<E>],
    ) -> Result<E, AkitaError> {
        terms.iter().try_fold(E::zero(), |acc, term| {
            term.current_witness_evals
                .claim_with_factor(&term.current_factor)
                .map(|claim| acc + term.coeff * claim)
        })
    }

    /// Final folded `(coeff, witness(rho), factor(rho))` tuples.
    pub fn final_terms(&self) -> Option<Vec<(E, E, E)>> {
        self.terms
            .iter()
            .map(|term| {
                term.final_witness_and_factor_evals()
                    .map(|(witness, factor)| (term.coeff, witness, factor))
            })
            .collect()
    }
}

impl<E: FieldCore> SumcheckInstanceProver<E> for BatchedExtensionOpeningReductionProver<E> {
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
        let mut constant = E::zero();
        let mut linear = E::zero();
        let mut quadratic = E::zero();

        for term in &self.terms {
            debug_assert_eq!(
                term.current_witness_evals.len(),
                1usize << (self.num_rounds - round)
            );
            debug_assert_eq!(term.current_factor.len(), term.current_witness_evals.len());

            term.current_witness_evals.accumulate_round(
                &term.current_factor,
                term.coeff,
                &mut constant,
                &mut linear,
                &mut quadratic,
            );
        }

        let poly = UniPoly::from_coeffs(vec![constant, linear, quadratic]);
        debug_assert_eq!(
            poly.evaluate(&E::zero()) + poly.evaluate(&E::one()),
            previous_claim
        );
        poly
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: E) {
        for term in &mut self.terms {
            if term.current_witness_evals.len() > 1 {
                term.current_witness_evals
                    .fold_with_factor_in_place(&mut term.current_factor, r_round);
            }
        }
    }
}
