//! Verifier-side succinct relation-weight evaluator (filled in during Slice 4).

use akita_field::{AkitaError, FieldCore};

/// Verifier-side prepared evaluator for `RelationWeightPolynomial`.
///
/// Slice 0 defines the shell; `eval_at_point` is implemented when the verifier
/// cutover lands.
#[derive(Debug, Clone)]
pub struct PreparedRelationWeightPolynomial<E: FieldCore> {
    _marker: core::marker::PhantomData<E>,
}

impl<E: FieldCore> PreparedRelationWeightPolynomial<E> {
    /// Placeholder constructor until verifier wiring provides row-family state.
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
    /// Returns [`AkitaError::InvalidSetup`] until Slice 4 implements the evaluator.
    pub fn eval_at_point(&self, _challenges: &[E]) -> Result<E, AkitaError> {
        Err(AkitaError::InvalidSetup(
            "PreparedRelationWeightPolynomial::eval_at_point not yet wired".to_string(),
        ))
    }
}
