//! Verifier for the Akita stage-2 fused sumcheck.

use crate::protocol::ring_switch::RelationMatrixEvaluator;
use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
    MulBaseUnreduced,
};
use akita_sumcheck::SumcheckInstanceVerifier;
use akita_types::{
    eval_dense_trace_table, eval_trace_terms_closed, AkitaExpandedSetup, FpExtEncoding,
    RingRelationInstance, TraceClaim,
};
use std::marker::PhantomData;

/// Verifier for the stage-2 fused virtual-claim and relation sumcheck.
pub(crate) struct AkitaStage2Verifier<'a, F: FieldCore, E: FieldCore, const D: usize> {
    batching_coeff: E,
    s_claim: E,
    witness_eval: E,
    stage1_point: Vec<E>,
    relation_matrix_evaluator: RelationMatrixEvaluator<E>,
    setup_claim: Option<E>,
    setup: &'a AkitaExpandedSetup<F>,
    relation_instance: &'a RingRelationInstance<F>,
    alpha: E,
    col_bits: usize,
    ring_bits: usize,
    relation_claim: E,
    trace: Option<TraceClaim<F, E, D>>,
    _marker: PhantomData<([F; D], E)>,
}

impl<'a, F, E, const D: usize> AkitaStage2Verifier<'a, F, E, D>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: ExtField<F> + FpExtEncoding<F> + FromPrimitiveInt + MulBaseUnreduced<F>,
{
    /// Construct a verifier from the shared stage-2 context and the witness
    /// oracle selected by the current proof level.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "AkitaStage2Verifier::new")]
    pub(crate) fn new(
        batching_coeff: E,
        s_claim: E,
        witness_eval: E,
        stage1_point: Vec<E>,
        relation_matrix_evaluator: RelationMatrixEvaluator<E>,
        setup: &'a AkitaExpandedSetup<F>,
        relation_instance: &'a RingRelationInstance<F>,
        alpha: E,
        setup_claim: Option<E>,
        relation_claim: E,
        col_bits: usize,
        ring_bits: usize,
        trace: Option<TraceClaim<F, E, D>>,
    ) -> Result<Self, AkitaError> {
        let num_rounds = col_bits.checked_add(ring_bits).ok_or_else(|| {
            AkitaError::InvalidSetup("stage-2 variable count overflow".to_string())
        })?;
        if stage1_point.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: stage1_point.len(),
            });
        }
        Ok(Self {
            batching_coeff,
            s_claim,
            witness_eval,
            stage1_point,
            relation_matrix_evaluator,
            setup_claim,
            setup,
            relation_instance,
            alpha,
            col_bits,
            ring_bits,
            relation_claim,
            trace,
            _marker: PhantomData,
        })
    }

    fn witness_eval(&self, _challenges: &[E]) -> Result<E, AkitaError> {
        Ok(self.witness_eval)
    }
}

impl<'a, F, E, const D: usize> SumcheckInstanceVerifier<E> for AkitaStage2Verifier<'a, F, E, D>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: ExtField<F> + FpExtEncoding<F> + FromPrimitiveInt + MulBaseUnreduced<F>,
{
    fn num_rounds(&self) -> usize {
        self.col_bits + self.ring_bits
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> E {
        let mut claim = self.batching_coeff * self.s_claim + self.relation_claim;
        if let Some(trace) = &self.trace {
            claim += trace.trace_opening_claim;
        }
        claim
    }

    #[tracing::instrument(skip_all, name = "stage2_expected_output_claim")]
    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        let w_eval = {
            let _span = tracing::info_span!("stage2_witness_eval").entered();
            self.witness_eval(challenges)?
        };

        let (y_challenges, x_challenges) = challenges.split_at(self.ring_bits);
        let relation_weight = self.relation_matrix_evaluator.eval_flat_at_point::<F, D>(
            challenges,
            self.setup,
            self.relation_instance,
            self.alpha,
            self.setup_claim,
        )?;
        let relation_oracle = w_eval * relation_weight;
        let trace_oracle = if let Some(trace) = &self.trace {
            // Scalar/recursive folds use one layout; multi-group roots use one
            // closed-form batch per group because their e-hat segments have
            // different geometry.
            let trace_weight = if let Some(dense_evals) = &trace.dense_evals {
                eval_dense_trace_table::<E>(dense_evals, y_challenges, x_challenges)?
            } else if !trace.trace_term_batches.is_empty() {
                let (trace_y, trace_x) = challenges.split_at(trace.layout.ring_bits);
                trace
                    .trace_term_batches
                    .iter()
                    .try_fold(E::zero(), |acc, batch| {
                        Ok::<E, AkitaError>(
                            acc + eval_trace_terms_closed::<F, E, D>(
                                &batch.layout,
                                trace_y,
                                trace_x,
                                &batch.terms,
                            )?,
                        )
                    })?
            } else {
                let (trace_y, trace_x) = challenges.split_at(trace.layout.ring_bits);
                eval_trace_terms_closed::<F, E, D>(
                    &trace.layout,
                    trace_y,
                    trace_x,
                    &trace.trace_terms,
                )?
            };
            trace.trace_coeff * w_eval * trace_weight
        } else {
            E::zero()
        };

        // Terminal levels run with `batching_coeff = 0`, which zeros the
        // virtual half regardless of `stage1_point` / `w_eval`. Skip the
        // EqPolynomial eval and the `w * (w + 1)` round in that case.
        if self.batching_coeff.is_zero() {
            return Ok(relation_oracle + trace_oracle);
        }
        let eq_val = EqPolynomial::mle(&self.stage1_point, challenges)?;
        let virtual_oracle = eq_val * w_eval * (w_eval + E::one());
        Ok(self.batching_coeff * virtual_oracle + relation_oracle + trace_oracle)
    }
}

#[cfg(any())]
mod tests {
    use super::{cleartext_source_eval, Stage2CleartextSource};
    use akita_field::{AkitaError, FieldCore};
    use akita_field::{FpExt2, NegOneNr, Prime128Offset275};
    use akita_sumcheck::multilinear_eval;

    type F = Prime128Offset275;
    type E = FpExt2<F, NegOneNr>;
    const D: usize = 4;

    fn build_w_evals<F: FieldCore>(
        w: &[F],
        d: usize,
    ) -> Result<(Vec<F>, usize, usize), AkitaError> {
        if !w.len().is_multiple_of(d) {
            return Err(AkitaError::InvalidSize {
                expected: d,
                actual: w.len(),
            });
        }
        let ring_bits = d.trailing_zeros() as usize;
        let num_ring_elems = w.len() / d;
        let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
        let x_len = 1usize << col_bits;
        let n = x_len << ring_bits;

        let mut out = vec![F::zero(); n];
        out[..w.len()].copy_from_slice(w);
        Ok((out, col_bits, ring_bits))
    }

    #[test]
    fn logical_digits_eval_matches_materialized_table() {
        let d = 4usize;
        let w_digits = vec![3, -1, 2, 0, -2, 1, 4, -3, 1, 0, -4, 2];
        let w_field: Vec<F> = w_digits
            .iter()
            .map(|&digit| F::from_i64(digit as i64))
            .collect();
        let (w_evals, col_bits, ring_bits) =
            build_w_evals(&w_field, d).expect("valid witness shape");
        let challenges = vec![
            F::from_u64(2),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
        ];
        let expected = multilinear_eval(&w_evals, &challenges).expect("matching table shape");
        let source = Stage2CleartextSource::LogicalDigits(std::borrow::Cow::Borrowed(&w_digits));
        let actual = cleartext_source_eval::<F, F, 4>(
            w_digits.len(),
            &source,
            &challenges,
            col_bits,
            ring_bits,
        )
        .expect("valid logical digits");
        assert_eq!(actual, expected);
    }

    #[test]
    fn field_witness_eval_lifts_base_witness_to_extension_challenges() {
        let field_witness = vec![
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
        ];
        let challenges = vec![
            E::new(F::from_u64(2), F::from_u64(3)),
            E::new(F::from_u64(5), F::from_u64(7)),
        ];

        let lifted_witness: Vec<E> = field_witness
            .iter()
            .copied()
            .map(|x| E::new(x, F::zero()))
            .collect();
        let expected =
            multilinear_eval(&lifted_witness, &challenges).expect("matching extension table shape");
        let source = Stage2CleartextSource::FieldElements(&field_witness);
        let actual =
            cleartext_source_eval::<F, E, D>(field_witness.len(), &source, &challenges, 1, 1)
                .expect("valid witness");

        assert_eq!(actual, expected);
    }

    #[test]
    fn logical_digits_eval_rejects_challenge_dimension_mismatch() {
        let w_digits = vec![1i8, -1, 0, 2];
        let source = Stage2CleartextSource::LogicalDigits(std::borrow::Cow::Borrowed(&w_digits));
        let err = cleartext_source_eval::<F, E, D>(1, &source, &[E::zero()], 1, 1)
            .expect_err("wrong arity");
        assert!(matches!(err, AkitaError::InvalidSize { .. }));
    }

    #[test]
    fn logical_digits_eval_rejects_length_mismatch() {
        let w_digits = vec![1i8, -1, 0, 2];
        let challenges = vec![E::zero(), E::zero()];
        let source = Stage2CleartextSource::LogicalDigits(std::borrow::Cow::Borrowed(&w_digits));
        let err = cleartext_source_eval::<F, E, D>(8, &source, &challenges, 1, 1)
            .expect_err("witness length mismatch");
        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
