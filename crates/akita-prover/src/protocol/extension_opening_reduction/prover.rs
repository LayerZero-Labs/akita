use super::*;

/// Prover state for a degree-two extension-opening reduction sumcheck.
///
/// Holds one or more terms `coeff_i * sum_x witness_i(x) * factor_i(x)` sharing
/// a common Boolean domain and a single round challenge sequence. A single
/// dense opening is the degenerate one-term case.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionProver<F: FieldCore, E: ExtField<F>> {
    terms: Vec<ExtensionOpeningReductionTerm<F, E>>,
    input_claim: E,
    num_rounds: usize,
    plan: SumcheckKernelPlan,
}

impl<F: FieldCore, E: ExtField<F>> ExtensionOpeningReductionProver<F, E> {
    /// Construct a prover from terms sharing one Boolean domain.
    ///
    /// The caller supplies the claimed input sum. This avoids recomputing it
    /// in protocol paths that already derived the claim while preparing the
    /// transcript-bound reduction.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no terms or their table lengths differ.
    pub fn new(
        terms: Vec<ExtensionOpeningReductionTerm<F, E>>,
        input_claim: E,
    ) -> Result<Self, AkitaError> {
        let first = terms.first().ok_or_else(|| {
            AkitaError::InvalidInput(
                "extension-opening reduction requires at least one term".to_string(),
            )
        })?;
        let table_len = first.tables.len();
        let num_rounds = num_rounds_from_table_len(table_len)?;
        for term in &terms {
            // Each term's witness/factor lengths agree by construction (the
            // table pairing is built equal-length); only the cross-term domain
            // needs checking here.
            if term.tables.len() != table_len {
                return Err(AkitaError::InvalidSize {
                    expected: table_len,
                    actual: term.tables.len(),
                });
            }
        }
        Ok(Self {
            terms,
            input_claim,
            num_rounds,
            plan: SumcheckKernelPlan::detect(),
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
        let term = ExtensionOpeningReductionTerm::new(witness_evals, factor_evals, E::one())?;
        Self::new(vec![term], input_claim)
    }

    /// Recompute the input sum represented by a set of terms.
    ///
    /// This scans the owned tables. It is intended for tests and standalone
    /// callers that do not already have an independently derived input claim;
    /// production protocol paths should pass that claim directly to [`Self::new`].
    ///
    pub fn recompute_input_claim(terms: &[ExtensionOpeningReductionTerm<F, E>]) -> E {
        terms.iter().fold(E::zero(), |acc, term| {
            acc + term.coeff * term.tables.claim()
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
        self.terms
            .iter()
            .map(|term| {
                term.final_witness_and_factor_evals()
                    .map(|(witness, factor)| (term.coeff, witness, factor))
            })
            .collect()
    }

    /// Final folded `(witness(rho), factor(rho))` for a single-term prover.
    ///
    /// Returns `None` for multi-term provers or before all challenges have been
    /// ingested.
    pub fn final_witness_and_factor_evals(&self) -> Option<(E, E)> {
        match self.terms.as_slice() {
            [term] => term.final_witness_and_factor_evals(),
            _ => None,
        }
    }
}

impl<F, E> SumcheckInstanceProver<E> for ExtensionOpeningReductionProver<F, E>
where
    F: FieldCore,
    E: SumcheckTableOperations<F>,
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
        let expected_len = 1usize << (self.num_rounds - round);
        let mut constant = E::zero();
        let mut quadratic = E::zero();

        for term in &mut self.terms {
            debug_assert_eq!(term.tables.len(), expected_len);

            term.accumulate_into(self.plan, &mut constant, &mut quadratic);
        }

        let linear = previous_claim - constant - constant - quadratic;
        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: E) {
        for term in &mut self.terms {
            term.ingest_challenge(self.plan, r_round);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp32, FpExt4, FromPrimitiveInt};
    use akita_serialization::AkitaSerialize;
    use akita_sumcheck::SumcheckInstanceProverExt;
    use akita_transcript::labels::CHALLENGE_SUMCHECK_ROUND;
    use akita_transcript::{sample_ext_challenge, AkitaTranscript};

    type TestBase = Fp32<2147483647>;
    type TestExt = FpExt4<TestBase>;

    fn value(index: usize) -> TestExt {
        TestExt::from_u64(u64::try_from(index).expect("test index fits in u64") + 1)
    }

    fn proof_with_plan(
        plan: SumcheckKernelPlan,
    ) -> (
        Vec<u8>,
        Vec<TestExt>,
        TestExt,
        Vec<(TestExt, TestExt, TestExt)>,
        TestExt,
    ) {
        let len = 1usize << 13;
        let witness = (0..len).map(|row| value(row * 3 + 7)).collect();
        let factor = (0..len).map(|row| value(row * 5 + 11)).collect();
        let mut prover = SumcheckKernelPlan::with_test_override(plan, || {
            ExtensionOpeningReductionProver::<TestBase, TestExt>::from_dense_tables(witness, factor)
                .expect("valid dense reduction")
        });
        let mut transcript = AkitaTranscript::<TestBase>::new(b"test/eor-kernel-plan-equivalence");
        let (proof, point, final_claim) = prover
            .prove::<TestBase, _, _>(&mut transcript, |transcript| {
                sample_ext_challenge::<TestBase, TestExt, _>(transcript, CHALLENGE_SUMCHECK_ROUND)
            })
            .expect("valid sumcheck proof");
        let final_terms = prover.final_terms().expect("fully folded terms");
        let continuation =
            sample_ext_challenge::<TestBase, TestExt, _>(&mut transcript, CHALLENGE_SUMCHECK_ROUND);
        let mut proof_bytes = Vec::new();
        proof
            .serialize_uncompressed(&mut proof_bytes)
            .expect("sumcheck proof serializes");
        (proof_bytes, point, final_claim, final_terms, continuation)
    }

    #[test]
    fn detected_plan_preserves_proof_bytes_and_transcript() {
        let detected = SumcheckKernelPlan::detect();
        let scalar = proof_with_plan(SumcheckKernelPlan::SCALAR);
        let optimized = proof_with_plan(detected);
        assert_eq!(optimized, scalar);
    }
}
