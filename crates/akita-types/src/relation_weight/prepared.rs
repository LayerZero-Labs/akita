//! Verifier-side succinct relation-weight evaluator shell.
//!
//! `eval_at_point` is implemented in `akita-verifier` where
//! [`RingSwitchDeferredRowEval`] lives.

use akita_field::{AkitaError, FieldCore};

/// Verifier-side prepared evaluator for `RelationWeightPolynomial`.
///
/// The verifier crate constructs this from ring-switch replay output and
/// EvaluationTrace row terms, then calls [`Self::eval_at_point`] during stage-2.
#[derive(Debug, Clone)]
pub struct PreparedRelationWeightPolynomial<E: FieldCore> {
    _marker: core::marker::PhantomData<E>,
}

impl<E: FieldCore> PreparedRelationWeightPolynomial<E> {
    /// Placeholder until the verifier wires ring-switch replay state.
    #[must_use]
    pub fn placeholder() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }

    /// Evaluate the relation-weight polynomial at the final sumcheck point.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] until the verifier attaches row state.
    pub fn eval_at_point(&self, _challenges: &[E]) -> Result<E, AkitaError> {
        Err(AkitaError::InvalidSetup(
            "PreparedRelationWeightPolynomial::eval_at_point is implemented in akita-verifier"
                .to_string(),
        ))
    }
}
