use super::*;

/// Prover state for a degree-two extension-opening reduction sumcheck.
///
/// Holds one or more groups
/// `sum_i coeff_i * sum_x witness_i(x) * factor_group(x)` sharing a common
/// Boolean domain and a single round challenge sequence. Each group folds its
/// transparent factor once per challenge, regardless of its member count.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionProver<E: Field> {
    groups: Vec<ExtensionOpeningReductionGroup<E>>,
    input_claim: E,
    num_rounds: usize,
}

impl<E: Field> ExtensionOpeningReductionProver<E> {
    /// Construct a prover from groups sharing one Boolean domain.
    ///
    /// The caller supplies the claimed input sum. This avoids recomputing it
    /// in protocol paths that already derived the claim while preparing the
    /// transcript-bound reduction.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no groups or their table lengths differ.
    pub fn new(
        groups: Vec<ExtensionOpeningReductionGroup<E>>,
        input_claim: E,
    ) -> Result<Self, AkitaError> {
        let first = groups.first().ok_or_else(|| {
            AkitaError::InvalidInput(
                "extension-opening reduction requires at least one group".to_string(),
            )
        })?;
        let table_len = first.domain_len();
        let num_rounds = num_rounds_from_table_len(table_len)?;
        for group in &groups {
            if group.domain_len() != table_len {
                return Err(AkitaError::InvalidSize {
                    expected: table_len,
                    actual: group.domain_len(),
                });
            }
        }
        Ok(Self {
            groups,
            input_claim,
            num_rounds,
        })
    }

    /// Construct a single-term prover from dense transformed-witness and
    /// transparent-factor Boolean-hypercube evaluation tables.
    ///
    /// # Errors
    ///
    /// Returns an error if the tables do not have the same nonzero power-of-two
    /// length.
    pub fn from_dense_tables(
        witness_evals: Vec<E>,
        factor_evals: Vec<E>,
    ) -> Result<Self, AkitaError> {
        let input_claim = extension_opening_reduction_claim(&witness_evals, &factor_evals)?;
        let term = ExtensionOpeningReductionTerm::new(witness_evals, E::one());
        let group = ExtensionOpeningReductionGroup::new(vec![term], factor_evals)?;
        Self::new(vec![group], input_claim)
    }

    /// Compute the input sum represented by a set of groups.
    ///
    /// This is useful for tests and standalone callers that do not already
    /// have an independently derived input claim.
    ///
    /// # Errors
    ///
    /// Returns an error if any group has malformed witness/factor tables.
    pub fn input_claim_from_groups(
        groups: &[ExtensionOpeningReductionGroup<E>],
    ) -> Result<E, AkitaError> {
        groups.iter().try_fold(E::zero(), |acc, group| {
            group.claim().map(|claim| acc + claim)
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

    /// Final folded `(coeff, witness(rho), factor(rho))` tuples.
    pub fn final_terms(&self) -> Option<Vec<(E, E, E)>> {
        self.groups.iter().try_fold(Vec::new(), |mut out, group| {
            out.extend(group.final_terms()?);
            Some(out)
        })
    }

    /// Final folded `(witness(rho), factor(rho))` for a single-term prover.
    ///
    /// Returns `None` for multi-term provers or before all challenges have been
    /// ingested.
    pub fn final_witness_and_factor_evals(&self) -> Option<(E, E)> {
        match self.groups.as_slice() {
            [group] => match group.final_terms()?.as_slice() {
                [(_, witness, factor)] => Some((*witness, *factor)),
                _ => None,
            },
            _ => None,
        }
    }
}

impl<E: Field + Unreduced + Fold> SumcheckInstanceProver<E> for ExtensionOpeningReductionProver<E> {
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
        let expected_len = 1usize << (self.num_rounds - round);
        let mut constant = E::zero();
        let mut quadratic = E::zero();

        for group in &mut self.groups {
            debug_assert_eq!(group.domain_len(), expected_len);
            group.accumulate_into(&mut constant, &mut quadratic);
        }

        let linear = previous_claim - constant - constant - quadratic;
        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: E) {
        for group in &mut self.groups {
            group.ingest_challenge(r_round);
        }
    }
}
