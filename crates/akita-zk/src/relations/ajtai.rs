//! Ajtai opening relations over `R_q = F_q[X] / (X^D + 1)`.

use crate::error::ZkResult;
use crate::norm::ring_vec_within_infinity_bound;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, PseudoMersenneField};

/// Public Ajtai opening relation `A s = t` over a cyclotomic ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AjtaiRelation<F: FieldCore, const D: usize> {
    /// Public matrix `A in R_q^{k x m}`.
    matrix: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Public commitment `t in R_q^k`.
    commitment: Vec<CyclotomicRing<F, D>>,
}

impl<F: FieldCore, const D: usize> AjtaiRelation<F, D> {
    /// Construct a relation and validate its rectangular shape.
    ///
    /// # Errors
    ///
    /// Returns an error if `A` is empty, non-rectangular, or incompatible with
    /// the commitment length.
    pub fn new(
        matrix: Vec<Vec<CyclotomicRing<F, D>>>,
        commitment: Vec<CyclotomicRing<F, D>>,
    ) -> ZkResult<Self> {
        if matrix.is_empty() {
            return Err(AkitaError::InvalidInput(
                "Ajtai matrix must have at least one row".to_string(),
            ));
        }
        let col_count = matrix[0].len();
        if col_count == 0 {
            return Err(AkitaError::InvalidInput(
                "Ajtai matrix must have at least one column".to_string(),
            ));
        }
        for row in &matrix {
            if row.len() != col_count {
                return Err(AkitaError::InvalidInput(
                    "Ajtai matrix rows must have equal length".to_string(),
                ));
            }
        }
        if commitment.len() != matrix.len() {
            return Err(AkitaError::InvalidInput(format!(
                "Ajtai commitment length {} does not match matrix row count {}",
                commitment.len(),
                matrix.len()
            )));
        }
        Ok(Self { matrix, commitment })
    }

    /// Number of output ring elements.
    pub fn row_count(&self) -> usize {
        self.matrix.len()
    }

    /// Number of witness ring elements.
    pub fn col_count(&self) -> usize {
        self.matrix[0].len()
    }

    /// Public matrix `A in R_q^{k x m}`.
    pub fn matrix(&self) -> &[Vec<CyclotomicRing<F, D>>] {
        &self.matrix
    }

    /// Public commitment `t in R_q^k`.
    pub fn commitment(&self) -> &[CyclotomicRing<F, D>] {
        &self.commitment
    }

    /// Compute `A * witness`.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness length does not match the relation.
    pub fn commit(&self, witness: &[CyclotomicRing<F, D>]) -> ZkResult<Vec<CyclotomicRing<F, D>>> {
        matrix_vector_mul(&self.matrix, witness)
    }

    /// Check whether `witness` opens the public commitment.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness length does not match the relation.
    pub fn check_opening(&self, witness: &[CyclotomicRing<F, D>]) -> ZkResult<bool> {
        Ok(self.commit(witness)? == self.commitment)
    }
}

impl<F: FieldCore + CanonicalField + PseudoMersenneField, const D: usize> AjtaiRelation<F, D> {
    /// Check that a witness is short and opens the commitment.
    ///
    /// # Errors
    ///
    /// Returns an error if field modulus metadata is unsupported or the witness
    /// length does not match the relation.
    pub fn check_short_opening(
        &self,
        witness: &[CyclotomicRing<F, D>],
        witness_bound: u128,
    ) -> ZkResult<bool> {
        if !ring_vec_within_infinity_bound(witness, witness_bound)? {
            return Ok(false);
        }
        self.check_opening(witness)
    }
}

/// Multiply a ring matrix by a ring vector.
///
/// # Errors
///
/// Returns an error if the matrix is empty, non-rectangular, or incompatible
/// with the vector length.
pub fn matrix_vector_mul<F: FieldCore, const D: usize>(
    matrix: &[Vec<CyclotomicRing<F, D>>],
    vector: &[CyclotomicRing<F, D>],
) -> ZkResult<Vec<CyclotomicRing<F, D>>> {
    if matrix.is_empty() {
        return Err(AkitaError::InvalidInput(
            "matrix must have at least one row".to_string(),
        ));
    }
    let col_count = matrix[0].len();
    if col_count != vector.len() {
        return Err(AkitaError::InvalidInput(format!(
            "matrix column count {col_count} does not match vector length {}",
            vector.len()
        )));
    }
    let mut out = Vec::with_capacity(matrix.len());
    for row in matrix {
        if row.len() != col_count {
            return Err(AkitaError::InvalidInput(
                "matrix rows must have equal length".to_string(),
            ));
        }
        let mut acc = CyclotomicRing::zero();
        for (entry, value) in row.iter().zip(vector.iter()) {
            entry.mul_accumulate_into(value, &mut acc);
        }
        out.push(acc);
    }
    Ok(out)
}
