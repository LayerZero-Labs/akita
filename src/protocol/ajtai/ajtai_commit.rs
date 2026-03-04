use crate::error::HachiError;
use crate::FieldCore;

/// A generic interface for the Ajtai commitment scheme.
///
/// This trait supports a two-tier commitment structure:
/// 1. Inner Tier: $t = A \cdot w$.
/// 2. Outer Tier: Decompose $t \to \hat{t}$, then commit $u = B \cdot \hat{t}$.
pub trait AjtaiCommitmentScheme<F: FieldCore, const D: usize> {
    /// The public matrix representation (e.g., derived from seed or precomputed NTT form).
    type PublicMatrix;

    /// The input witness type.
    type Witness: ?Sized;

    /// The result of the first stage (A * w), before decomposition (e.g., t).
    type InnerCommitment;

    /// The decomposed inner value (e.g., t_hat).
    type DecomposedInnerCommitment;

    /// The final outer commitment (e.g., u).
    type OuterCommitment;

    /// Configuration parameters required for the commitment process.
    type Params;

    /// Performs the first stage of commitment: $t = A \cdot w$.
    ///
    /// This generates the inner commitment.
    ///
    /// # Errors
    /// Returns `HachiError` on matrix shape mismatch or invalid inputs.
    fn commit_witness(
        matrix: &Self::PublicMatrix,
        witness: &Self::Witness,
        params: &Self::Params,
    ) -> Result<Self::InnerCommitment, HachiError>;

    /// Performs the second stage: Decompose $t \to \hat{t}$, then compute $u = B \cdot \hat{t}$.
    ///
    /// This logic computes the outer commitment.
    /// Returns both the decomposed inner value and the outer commitment.
    ///
    /// # Errors
    /// Returns `HachiError` on matrix shape mismatch or invalid inputs.
    fn commit_inner(
        matrix: &Self::PublicMatrix,
        inner_commitment: &Self::InnerCommitment,
        params: &Self::Params,
    ) -> Result<(Self::DecomposedInnerCommitment, Self::OuterCommitment), HachiError>;

    /// Performs the full two-tier commitment process: $w \to t \to \hat{t} \to u$.
    ///
    /// Returns the decomposed inner value and the outer commitment.
    ///
    /// # Errors
    /// Returns `HachiError` on matrix shape mismatch or invalid inputs.
    fn two_tier_commit(
        matrix_a: &Self::PublicMatrix,
        matrix_b: &Self::PublicMatrix,
        witness: &Self::Witness,
        params: &Self::Params,
    ) -> Result<(Self::DecomposedInnerCommitment, Self::OuterCommitment), HachiError> {
        let t = Self::commit_witness(matrix_a, witness, params)?;
        Self::commit_inner(matrix_b, &t, params)
    }
}
