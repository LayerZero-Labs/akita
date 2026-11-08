//! Protocol errors shared by Akita crates.

#![deny(missing_docs)]
#![warn(unreachable_pub)]

use jolt_algebra::AlgebraError;
use jolt_crypto::CryptoError;

/// Errors that can occur in Akita PCS operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AkitaError {
    /// Proof bytes failed a protocol check during verification.
    #[error("Invalid proof")]
    InvalidProof,

    /// A polynomial or protocol object has an invalid size.
    #[error("Invalid polynomial size: expected {expected}, got {actual}")]
    InvalidSize {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },

    /// An evaluation point has the wrong dimension.
    #[error("Invalid evaluation point dimension: expected {expected}, got {actual}")]
    InvalidPointDimension {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },

    /// Input parameters are invalid.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// The requested polynomial layout has no supported folded proof schedule.
    #[error("Unsupported proof schedule: {0}")]
    UnsupportedSchedule(String),

    /// Setup data is missing or invalid.
    #[error("Invalid or missing setup file: {0}")]
    InvalidSetup(String),

    /// The one-shot fold response at `level` fell outside its digit range.
    ///
    /// Extension seam: zero knowledge (akita#120). Only a fold-response run with
    /// `nonce_bits = 0` probes its response once, and `main` schedules none.
    /// The event has probability at most the fold's one-shot failure target;
    /// a caller that resamples the prover's randomness may rerun the proof.
    #[error("Fold response overflowed its digit range at level {level}")]
    FoldOverflow {
        /// Fold level of the rejected response.
        level: u32,
    },

    /// An invariant that Akita maintains itself failed after the inputs were
    /// admitted. This reports a bug in Akita or in a custom backend, not a
    /// rejected proof and not a caller mistake. The message names the failed
    /// invariant.
    ///
    /// The variant is not tied to the prover: setup and cache code report
    /// their own failed invariants the same way. It must never stand in for a
    /// rejection. A check on a proof, a commitment, or any other untrusted
    /// value returns [`AkitaError::InvalidProof`], even when only a bug in an
    /// honest prover could make it fail.
    /// Code shared by the prover and verifier also returns this variant for
    /// checks proof bytes cannot make fail, such as a lookup into a table built
    /// from the admitted schedule; checks a proof can reach still return
    /// [`AkitaError::InvalidProof`].
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Algebra errors keep their classification: every [`AlgebraError`] variant
/// reports a rejected argument, so each maps to the same-named variant.
impl From<AlgebraError> for AkitaError {
    fn from(error: AlgebraError) -> Self {
        match error {
            AlgebraError::InvalidInput(message) => Self::InvalidInput(message),
            AlgebraError::InvalidSize { expected, actual } => {
                Self::InvalidSize { expected, actual }
            }
            AlgebraError::InvalidSetup(message) => Self::InvalidSetup(message),
        }
    }
}

/// Lattice-primitive errors keep their classification: each [`CryptoError`]
/// variant maps to its namesake, and a wrapped [`AlgebraError`] maps as above.
impl From<CryptoError> for AkitaError {
    fn from(error: CryptoError) -> Self {
        match error {
            CryptoError::InvalidInput(message) => Self::InvalidInput(message),
            CryptoError::InvalidSize { expected, actual } => Self::InvalidSize { expected, actual },
            CryptoError::InvalidSetup(message) => Self::InvalidSetup(message),
            CryptoError::Internal(message) => Self::Internal(message),
            CryptoError::Algebra(error) => error.into(),
        }
    }
}
