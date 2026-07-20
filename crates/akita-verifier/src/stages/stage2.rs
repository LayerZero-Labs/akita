//! Verifier for the Akita stage-2 fused sumcheck.

use crate::protocol::ring_switch::RelationWeightEvaluator;
use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
    MulBaseUnreduced,
};
use akita_sumcheck::SumcheckInstanceVerifier;
use akita_types::{
    AkitaExpandedSetup, EvaluationTraceWeights, FpExtEncoding, RingRelationInstance,
};
use std::marker::PhantomData;

/// Verifier for the stage-2 fused virtual-claim and relation sumcheck.
pub(crate) struct AkitaStage2Verifier<'a, F: FieldCore, E: FieldCore, const D: usize> {
    batching_coeff: E,
    range_image_evaluation: E,
    witness_eval: E,
    stage1_point: Vec<E>,
    relation_weight_evaluator: &'a RelationWeightEvaluator<E>,
    setup_claim: Option<E>,
    setup: &'a AkitaExpandedSetup<F>,
    relation_instance: &'a RingRelationInstance<F>,
    alpha: E,
    col_bits: usize,
    ring_bits: usize,
    relation_claim: E,
    evaluation_trace_weights: EvaluationTraceWeights<E>,
    evaluation_trace_row_weight: E,
    evaluation_trace_opening_claim: E,
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
        range_image_evaluation: E,
        witness_eval: E,
        stage1_point: Vec<E>,
        relation_weight_evaluator: &'a RelationWeightEvaluator<E>,
        setup: &'a AkitaExpandedSetup<F>,
        relation_instance: &'a RingRelationInstance<F>,
        alpha: E,
        setup_claim: Option<E>,
        relation_claim: E,
        col_bits: usize,
        ring_bits: usize,
        evaluation_trace_weights: EvaluationTraceWeights<E>,
        evaluation_trace_row_weight: E,
        evaluation_trace_opening_claim: E,
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
            range_image_evaluation,
            witness_eval,
            stage1_point,
            relation_weight_evaluator,
            setup_claim,
            setup,
            relation_instance,
            alpha,
            col_bits,
            ring_bits,
            relation_claim,
            evaluation_trace_weights,
            evaluation_trace_row_weight,
            evaluation_trace_opening_claim,
            _marker: PhantomData,
        })
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
        self.batching_coeff * self.range_image_evaluation
            + self.relation_claim
            + self.evaluation_trace_opening_claim
    }

    #[tracing::instrument(skip_all, name = "stage2_expected_output_claim")]
    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        let w_eval = {
            let _span = tracing::info_span!("stage2_witness_eval").entered();
            self.witness_eval
        };

        let relation_weight = {
            let _span = tracing::info_span!("stage2_relation_weight").entered();
            self.relation_weight_evaluator.eval_flat_at_point::<F, D>(
                challenges,
                self.setup,
                self.relation_instance,
                self.alpha,
                self.setup_claim,
            )?
        };
        let relation_oracle = w_eval * relation_weight;
        let trace_oracle = {
            let _span = tracing::info_span!("stage2_trace_oracle").entered();
            self.evaluation_trace_row_weight
                * w_eval
                * self
                    .evaluation_trace_weights
                    .evaluate_at_point(challenges)?
        };

        // A zero batching challenge removes the virtual term. Avoid the
        // unnecessary EqPolynomial evaluation in that degenerate case.
        if self.batching_coeff.is_zero() {
            return Ok(relation_oracle + trace_oracle);
        }
        let virtual_oracle = {
            let _span = tracing::info_span!("stage2_virtual_oracle").entered();
            let eq_val = EqPolynomial::mle(&self.stage1_point, challenges)?;
            eq_val * w_eval * (w_eval + E::one())
        };
        Ok(self.batching_coeff * virtual_oracle + relation_oracle + trace_oracle)
    }
}
