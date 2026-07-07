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

struct FlatRelationWeightPoint<'a, E: FieldCore> {
    coefficient_challenges: &'a [E],
    segment_challenges: &'a [E],
}

/// Verifier-side prepared evaluator for the unified relation-weight polynomial.
pub(crate) struct PreparedRelationWeightPolynomial<F: FieldCore, E: FieldCore, const D: usize> {
    pub(crate) deferred: RingSwitchDeferredRowEval<E>,
    alpha: E,
    pub(crate) num_vars: usize,
    witness_coeff_bits: usize,
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
        num_vars: usize,
        witness_coeff_bits: usize,
        witness_live_len: usize,
    ) -> Self {
        Self {
            deferred,
            alpha,
            num_vars,
            witness_coeff_bits,
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
    ) -> Result<FlatRelationWeightPoint<'a, E>, AkitaError> {
        let witness_coeff_len = D;
        if witness_coeff_len == 0
            || !witness_coeff_len.is_power_of_two()
            || self.witness_coeff_bits != witness_coeff_len.trailing_zeros() as usize
        {
            return Err(AkitaError::InvalidProof);
        }
        if self.witness_live_len == 0 || !self.witness_live_len.is_multiple_of(witness_coeff_len) {
            return Err(AkitaError::InvalidSize {
                expected: witness_coeff_len,
                actual: self.witness_live_len,
            });
        }
        if challenges.len() != self.num_vars {
            return Err(AkitaError::InvalidSize {
                expected: self.num_vars,
                actual: challenges.len(),
            });
        }
        let (coefficient_challenges, segment_challenges) =
            challenges.split_at(self.witness_coeff_bits);
        Ok(FlatRelationWeightPoint {
            coefficient_challenges,
            segment_challenges,
        })
    }

    /// Quotient-bearing row families via the deferred setup scan (A/B/D rows and
    /// `r_hat` tail), embedded into the flat witness coefficient slots.
    fn eval_quotient_row_families_at_point(
        &self,
        point: &FlatRelationWeightPoint<'_, E>,
        setup: &AkitaExpandedSetup<F>,
        opening_point: &RingOpeningPoint<F>,
        ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError> {
        let witness_coeff_powers = akita_algebra::ring::scalar_powers(self.alpha, D);
        let coefficient_embedding_weight =
            multilinear_eval(&witness_coeff_powers, point.coefficient_challenges)?;
        let row_family_sum = self.deferred.eval_at_point::<F, D>(
            point.segment_challenges,
            setup,
            opening_point,
            ring_multiplier_point,
            self.alpha,
            setup_claim,
        )?;
        Ok(coefficient_embedding_weight * row_family_sum)
    }

    /// [`EvaluationTrace`](akita_types::RelationRowFamily::EvaluationTrace) row
    /// contribution, weighted by `eq(tau1, EvaluationTrace)`.
    fn eval_evaluation_trace_row_at_point(
        &self,
        point: &FlatRelationWeightPoint<'_, E>,
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
            point.coefficient_challenges,
            point.segment_challenges,
            &trace.terms,
        )? * eq_trace)
    }

    /// Evaluate the prepared relation-weight polynomial at a flat stage-2 point.
    ///
    /// `challenges` indexes the live flat witness hypercube. The local
    /// coefficient embedding used by the current row-family evaluator is
    /// internal to this method.
    pub(crate) fn eval_at_point(
        &self,
        challenges: &[E],
        setup: &AkitaExpandedSetup<F>,
        opening_point: &RingOpeningPoint<F>,
        ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError> {
        let point = self.validate_flat_point(challenges)?;
        let quotient_rows = self.eval_quotient_row_families_at_point(
            &point,
            setup,
            opening_point,
            ring_multiplier_point,
            setup_claim,
        )?;
        let evaluation_trace = self.eval_evaluation_trace_row_at_point(&point)?;
        Ok(quotient_rows + evaluation_trace)
    }
}
