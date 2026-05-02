use super::commitment::CommitmentBackend;
use super::statement::ZkSigmaStatement;
use crate::error::HachiError;
use crate::FieldCore;

/// First-message masks for one quadratic relation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuadraticMask<F: FieldCore> {
    /// `left(mask)`.
    pub left: F,
    /// `right(mask)`.
    pub right: F,
    /// `output(mask)`.
    pub output: F,
}

/// Proof object for the standalone Sigma protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZkSigmaProof<F: FieldCore> {
    /// Successful rejection-sampling attempt index.
    pub attempt: u32,
    /// Commitment to the mask vector.
    pub mask_commitment: Vec<F>,
    /// Masked evaluations for each linear relation.
    pub linear_masks: Vec<F>,
    /// Masked evaluations for each quadratic relation.
    pub quadratic_masks: Vec<QuadraticMask<F>>,
    /// Response vector `z = c * witness + mask`.
    pub response: Vec<F>,
}

impl<F: FieldCore> ZkSigmaProof<F> {
    pub(super) fn check_against_statement(
        &self,
        statement: &ZkSigmaStatement<F>,
    ) -> Result<(), HachiError> {
        let witness_len = statement.commitment_key.witness_len();
        if self.response.len() != witness_len {
            return Err(HachiError::InvalidSize {
                expected: witness_len,
                actual: self.response.len(),
            });
        }
        if self.mask_commitment.len() != statement.commitment_key.rows {
            return Err(HachiError::InvalidSize {
                expected: statement.commitment_key.rows,
                actual: self.mask_commitment.len(),
            });
        }
        if self.linear_masks.len() != statement.linear_relations.len() {
            return Err(HachiError::InvalidSize {
                expected: statement.linear_relations.len(),
                actual: self.linear_masks.len(),
            });
        }
        if self.quadratic_masks.len() != statement.quadratic_relations.len() {
            return Err(HachiError::InvalidSize {
                expected: statement.quadratic_relations.len(),
                actual: self.quadratic_masks.len(),
            });
        }
        Ok(())
    }
}
