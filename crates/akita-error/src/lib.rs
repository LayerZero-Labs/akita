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
    /// Only a zero-knowledge opening probes a fold response once. The event
    /// has probability at most the fold's failure target. A rerun can
    /// succeed only once the opening carries hiding randomness, which it
    /// then resamples; until then the same inputs fail the same way.
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

/// A masked channel's failure (`specs/whiteout-akita.md` §4.5). A transcript
/// or tape failure rejects the proof; a pad schedule or message field that
/// disagrees with the schedule is a setup error; a decoded value failing its
/// check is a failed prover invariant.
#[cfg(feature = "zk")]
impl From<jolt_zk::masking::ZkError> for AkitaError {
    fn from(error: jolt_zk::masking::ZkError) -> Self {
        use jolt_zk::masking::ZkError;
        match error {
            ZkError::TranscriptRejected
            | ZkError::ChallengeRejected
            | ZkError::MaskedTapeExhausted
            | ZkError::MaskedTapeTrailing => Self::InvalidProof,
            ZkError::PadsExhausted { .. }
            | ZkError::PadCountMismatch { .. }
            | ZkError::InvalidLayout { .. }
            | ZkError::OperandPadsMismatch
            | ZkError::MessageFieldMismatch
            | ZkError::RoundWidthMismatch { .. }
            | ZkError::RoundCountMismatch { .. } => Self::InvalidSetup(error.to_string()),
            ZkError::InvalidRelation { .. }
            | ZkError::ClaimMismatch(_)
            | ZkError::Sumcheck(_)
            | ZkError::BlindFoldShape(_)
            | ZkError::BlindFoldProver(_)
            | ZkError::BlindFoldRejected(_)
            | ZkError::Whiteout(_) => Self::Internal(error.to_string()),
            ZkError::FoldOverflow { level } => Self::FoldOverflow { level },
        }
    }
}

/// Akita's failures inside a zero-knowledge back end (the response prover,
/// the masked prefix under W6): a one-shot fold overflow stays
/// [`jolt_zk::masking::ZkError::FoldOverflow`], the failure a caller reruns
/// on; every other error travels as the back end's erased source.
#[cfg(feature = "zk")]
impl From<AkitaError> for jolt_zk::masking::ZkError {
    fn from(error: AkitaError) -> Self {
        match error {
            AkitaError::FoldOverflow { level } => Self::FoldOverflow { level },
            AkitaError::InvalidProof
            | AkitaError::InvalidSize { .. }
            | AkitaError::InvalidPointDimension { .. }
            | AkitaError::InvalidInput(_)
            | AkitaError::UnsupportedSchedule(_)
            | AkitaError::InvalidSetup(_)
            | AkitaError::Internal(_) => Self::Whiteout(Box::new(error)),
        }
    }
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

#[cfg(all(test, feature = "zk"))]
mod tests {
    use super::*;
    use jolt_zk::masking::ZkError;

    /// A one-shot fold overflow crosses into jolt-zk as its retryable
    /// variant and comes back as itself; other errors do not become it.
    #[test]
    fn fold_overflow_round_trips_through_zk_error() {
        let level = 1;
        let crossed = ZkError::from(AkitaError::FoldOverflow { level });
        assert!(matches!(crossed, ZkError::FoldOverflow { level: 1 }));
        assert_eq!(
            AkitaError::from(crossed),
            AkitaError::FoldOverflow { level }
        );
        assert!(matches!(
            ZkError::from(AkitaError::InvalidProof),
            ZkError::Whiteout(_)
        ));
    }
}
