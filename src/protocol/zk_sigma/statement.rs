use super::commitment::{CommitmentBackend, MatrixCommitmentKey};
use super::relation::{LinearRelation, QuadraticRelation};
use crate::error::HachiError;
use crate::FieldCore;

/// Public statement for the standalone Sigma protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZkSigmaStatement<F: FieldCore> {
    /// Commitment backend for this standalone instance.
    pub commitment_key: MatrixCommitmentKey<F>,
    /// Public commitment to the witness.
    pub commitment: Vec<F>,
    /// Linear relations to prove.
    pub linear_relations: Vec<LinearRelation<F>>,
    /// Product-of-linear quadratic relations to prove.
    pub quadratic_relations: Vec<QuadraticRelation<F>>,
    /// Optional centered infinity-norm bound for abort checks on the response.
    pub response_linf_bound: Option<u128>,
}

impl<F: FieldCore> ZkSigmaStatement<F> {
    pub(super) fn check_shapes(&self) -> Result<(), HachiError> {
        self.commitment_key.check_shape()?;
        if self.commitment.len() != self.commitment_key.rows {
            return Err(HachiError::InvalidSize {
                expected: self.commitment_key.rows,
                actual: self.commitment.len(),
            });
        }
        let witness_len = self.commitment_key.witness_len();
        for relation in &self.linear_relations {
            if relation.expression.coeffs.len() != witness_len {
                return Err(HachiError::InvalidSize {
                    expected: witness_len,
                    actual: relation.expression.coeffs.len(),
                });
            }
        }
        for relation in &self.quadratic_relations {
            for len in [
                relation.left.coeffs.len(),
                relation.right.coeffs.len(),
                relation.output.coeffs.len(),
            ] {
                if len != witness_len {
                    return Err(HachiError::InvalidSize {
                        expected: witness_len,
                        actual: len,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Private witness for the standalone Sigma protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZkSigmaWitness<F: FieldCore> {
    /// Committed witness coordinates.
    pub values: Vec<F>,
}
