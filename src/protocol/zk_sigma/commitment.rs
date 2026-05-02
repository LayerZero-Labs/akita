use crate::error::HachiError;
use crate::primitives::serialization::HachiSerialize;
use crate::FieldCore;

/// Minimal commitment interface used by the Sigma protocol.
pub trait CommitmentBackend<F: FieldCore> {
    /// Commitment type produced by the backend.
    type Commitment: Clone + PartialEq + HachiSerialize;

    /// Number of field coordinates in the committed witness.
    fn witness_len(&self) -> usize;

    /// Commit to a witness or mask vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness length does not match this backend.
    fn commit(&self, witness: &[F]) -> Result<Self::Commitment, HachiError>;

    /// Compute `challenge * base + mask` in commitment space.
    ///
    /// # Errors
    ///
    /// Returns an error if either commitment has the wrong shape.
    fn combine_commitments(
        &self,
        challenge: F,
        base: &Self::Commitment,
        mask: &Self::Commitment,
    ) -> Result<Self::Commitment, HachiError>;
}

/// Dense field-matrix commitment key for standalone tests and adapters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixCommitmentKey<F: FieldCore> {
    /// Number of commitment coordinates.
    pub rows: usize,
    /// Number of witness coordinates.
    pub cols: usize,
    /// Row-major matrix entries.
    pub entries: Vec<F>,
}

impl<F: FieldCore> MatrixCommitmentKey<F> {
    /// Construct a dense commitment matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if `entries.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, entries: Vec<F>) -> Result<Self, HachiError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| HachiError::InvalidInput("matrix shape overflow".into()))?;
        if entries.len() != expected {
            return Err(HachiError::InvalidSize {
                expected,
                actual: entries.len(),
            });
        }
        Ok(Self {
            rows,
            cols,
            entries,
        })
    }

    pub(super) fn check_shape(&self) -> Result<(), HachiError> {
        let expected = self
            .rows
            .checked_mul(self.cols)
            .ok_or_else(|| HachiError::InvalidInput("matrix shape overflow".into()))?;
        if self.entries.len() != expected {
            return Err(HachiError::InvalidSize {
                expected,
                actual: self.entries.len(),
            });
        }
        Ok(())
    }
}

impl<F: FieldCore> CommitmentBackend<F> for MatrixCommitmentKey<F> {
    type Commitment = Vec<F>;

    fn witness_len(&self) -> usize {
        self.cols
    }

    fn commit(&self, witness: &[F]) -> Result<Self::Commitment, HachiError> {
        self.check_shape()?;
        if witness.len() != self.cols {
            return Err(HachiError::InvalidSize {
                expected: self.cols,
                actual: witness.len(),
            });
        }

        let mut out = vec![F::zero(); self.rows];
        for (row_idx, row) in self.entries.chunks_exact(self.cols).enumerate() {
            let mut acc = F::zero();
            for (&a, &x) in row.iter().zip(witness) {
                acc += a * x;
            }
            out[row_idx] = acc;
        }
        Ok(out)
    }

    fn combine_commitments(
        &self,
        challenge: F,
        base: &Self::Commitment,
        mask: &Self::Commitment,
    ) -> Result<Self::Commitment, HachiError> {
        if base.len() != self.rows {
            return Err(HachiError::InvalidSize {
                expected: self.rows,
                actual: base.len(),
            });
        }
        if mask.len() != self.rows {
            return Err(HachiError::InvalidSize {
                expected: self.rows,
                actual: mask.len(),
            });
        }
        Ok(base
            .iter()
            .zip(mask)
            .map(|(&u, &t)| challenge * u + t)
            .collect())
    }
}
