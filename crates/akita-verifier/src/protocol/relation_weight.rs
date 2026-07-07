//! Verifier-side prepared relation-weight polynomial evaluator.

use crate::protocol::ring_switch::RingSwitchDeferredRowEval;
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
    witness_live_len: usize,
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
        witness_live_len: usize,
    ) -> Self {
        Self {
            deferred,
            alpha,
            ring_bits,
            col_bits,
            witness_live_len,
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

    fn validate_flat_point<'a>(
        &self,
        challenges: &'a [E],
    ) -> Result<(&'a [E], &'a [E]), AkitaError> {
        let witness_coeff_len = D;
        if witness_coeff_len == 0
            || !witness_coeff_len.is_power_of_two()
            || self.ring_bits != witness_coeff_len.trailing_zeros() as usize
        {
            return Err(AkitaError::InvalidProof);
        }
        if self.witness_live_len == 0 || !self.witness_live_len.is_multiple_of(witness_coeff_len) {
            return Err(AkitaError::InvalidSize {
                expected: witness_coeff_len,
                actual: self.witness_live_len,
            });
        }
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
        Ok((y_challenges, x_challenges))
    }

    /// Quotient-bearing row families via the deferred setup scan (A/B/D rows and
    /// `r_hat` tail), scaled by the witness coefficient axis `alpha^coeff`.
    fn eval_quotient_row_families_at_point(
        &self,
        y_challenges: &[E],
        x_challenges: &[E],
        setup: &AkitaExpandedSetup<F>,
        opening_point: &RingOpeningPoint<F>,
        ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError> {
        let alpha_evals_y = akita_algebra::ring::scalar_powers(self.alpha, D);
        let witness_axis_weight = multilinear_eval(&alpha_evals_y, y_challenges)?;
        let row_family_sum = self.deferred.eval_at_point::<F, D>(
            x_challenges,
            setup,
            opening_point,
            ring_multiplier_point,
            self.alpha,
            setup_claim,
        )?;
        Ok(witness_axis_weight * row_family_sum)
    }

    /// [`EvaluationTrace`](akita_types::RelationRowFamily::EvaluationTrace) row
    /// contribution, weighted by `eq(tau1, EvaluationTrace)`.
    fn eval_evaluation_trace_row_at_point(
        &self,
        y_challenges: &[E],
        x_challenges: &[E],
    ) -> Result<E, AkitaError> {
        let Some(trace) = &self.evaluation_trace else {
            return Ok(E::zero());
        };
        let eq_trace = self
            .setup_contribution_inputs()
            .eq_tau1
            .first()
            .copied()
            .unwrap_or(E::zero());
        Ok(eval_trace_terms_closed::<F, E, D>(
            &trace.layout,
            y_challenges,
            x_challenges,
            &trace.terms,
        )? * eq_trace)
    }

    /// Evaluate the prepared relation-weight polynomial at a flat stage-2 point.
    ///
    /// `challenges` has length `col_bits + ring_bits` and indexes the live flat
    /// witness hypercube. Coordinate splitting is internal to this evaluator.
    pub(crate) fn eval_at_point(
        &self,
        challenges: &[E],
        setup: &AkitaExpandedSetup<F>,
        opening_point: &RingOpeningPoint<F>,
        ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError> {
        let (y_challenges, x_challenges) = self.validate_flat_point(challenges)?;
        let quotient_rows = self.eval_quotient_row_families_at_point(
            y_challenges,
            x_challenges,
            setup,
            opening_point,
            ring_multiplier_point,
            setup_claim,
        )?;
        let evaluation_trace =
            self.eval_evaluation_trace_row_at_point(y_challenges, x_challenges)?;
        Ok(quotient_rows + evaluation_trace)
    }
}
