use crate::error::HachiError;
use crate::FieldCore;

/// Linear expression `constant + <coeffs, witness>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearExpression<F: FieldCore> {
    /// Coefficients for witness coordinates.
    pub coeffs: Vec<F>,
    /// Public constant term.
    pub constant: F,
}

impl<F: FieldCore> LinearExpression<F> {
    /// Evaluate the expression at `witness`.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness length does not match the coefficient count.
    pub fn evaluate(&self, witness: &[F]) -> Result<F, HachiError> {
        if self.coeffs.len() != witness.len() {
            return Err(HachiError::InvalidSize {
                expected: self.coeffs.len(),
                actual: witness.len(),
            });
        }
        let mut acc = self.constant;
        for (&coeff, &value) in self.coeffs.iter().zip(witness) {
            acc += coeff * value;
        }
        Ok(acc)
    }
}

/// Claimed linear relation `expression(witness) = target`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearRelation<F: FieldCore> {
    /// Linear expression to evaluate.
    pub expression: LinearExpression<F>,
    /// Public target value.
    pub target: F,
}

/// Claimed quadratic relation `left(w) * right(w) - output(w) = target`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuadraticRelation<F: FieldCore> {
    /// Left linear factor.
    pub left: LinearExpression<F>,
    /// Right linear factor.
    pub right: LinearExpression<F>,
    /// Linear output term subtracted from the product.
    pub output: LinearExpression<F>,
    /// Public target value.
    pub target: F,
}
