//! Verifier-side prepared relation-weight polynomial evaluator.

use crate::protocol::ring_switch::RingSwitchDeferredRowEval;
use akita_algebra::ring::scalar_powers;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBase};
use akita_sumcheck::multilinear_eval;
use akita_types::{
    eval_trace_terms_closed, AkitaExpandedSetup, FpExtEncoding, RingMultiplierOpeningPoint,
    RingOpeningPoint, SetupContributionPlanInputs, TraceTerm, TraceWeightLayout,
};

/// EvaluationTrace row data for closed-form relation-weight evaluation.
#[derive(Debug, Clone)]
pub(crate) struct EvaluationTraceRowEval<F: FieldCore, E: FieldCore, const D: usize> {
    pub layout: TraceWeightLayout,
    pub terms: Vec<TraceTerm<F, E, D>>,
}

/// Verifier-side prepared evaluator for the unified relation-weight polynomial.
pub(crate) struct PreparedRelationWeightPolynomial<F: FieldCore, E: FieldCore, const D: usize> {
    pub(crate) deferred: RingSwitchDeferredRowEval<E>,
    alpha: E,
    pub(crate) col_bits: usize,
    pub(crate) ring_bits: usize,
    evaluation_trace: Option<EvaluationTraceRowEval<F, E, D>>,
}

impl<F, E, const D: usize> PreparedRelationWeightPolynomial<F, E, D>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FpExtEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    pub(crate) fn from_ring_switch(
        deferred: RingSwitchDeferredRowEval<E>,
        alpha: E,
        col_bits: usize,
        ring_bits: usize,
    ) -> Self {
        Self {
            deferred,
            alpha,
            ring_bits,
            col_bits,
            evaluation_trace: None,
        }
    }

    pub(crate) fn with_evaluation_trace(
        mut self,
        layout: TraceWeightLayout,
        terms: Vec<TraceTerm<F, E, D>>,
    ) -> Self {
        self.evaluation_trace = Some(EvaluationTraceRowEval { layout, terms });
        self
    }

    pub(crate) fn setup_contribution_inputs(&self) -> SetupContributionPlanInputs<E> {
        self.deferred.create_setup_contribution_inputs()
    }

    pub(crate) fn eval_at_point(
        &self,
        challenges: &[E],
        setup: &AkitaExpandedSetup<F>,
        opening_point: &RingOpeningPoint<F>,
        ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError> {
        let num_rounds = self
            .col_bits
            .checked_add(self.ring_bits)
            .ok_or(AkitaError::InvalidProof)?;
        if challenges.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: challenges.len(),
            });
        }
        let (y_challenges, x_challenges) = challenges.split_at(self.ring_bits);
        let y_len = 1usize
            .checked_shl(self.ring_bits as u32)
            .ok_or(AkitaError::InvalidProof)?;
        let alpha_evals_y = scalar_powers(self.alpha, y_len);
        let alpha_val = multilinear_eval(&alpha_evals_y, y_challenges)?;
        let row_val = self.deferred.eval_at_point::<F, D>(
            x_challenges,
            setup,
            opening_point,
            ring_multiplier_point,
            self.alpha,
            setup_claim,
        )?;
        let trace_val = if let Some(trace) = &self.evaluation_trace {
            let eq_trace = self
                .setup_contribution_inputs()
                .eq_tau1
                .first()
                .copied()
                .unwrap_or(E::zero());
            eval_trace_terms_closed::<F, E, D>(
                &trace.layout,
                y_challenges,
                x_challenges,
                &trace.terms,
            )? * eq_trace
        } else {
            E::zero()
        };
        Ok(alpha_val * row_val + trace_val)
    }
}
