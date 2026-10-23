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

/// A transcript failure on the verifier side rejects the proof: a verifier
/// transcript fails only on malformed, missing, or out-of-range proof bytes.
/// A clear message on a masked channel is a schedule that sends one at a
/// masked level, on either side.
///
/// Prover-only transcript failures (grinding exhaustion, an over-bound
/// message) are classified at their call sites instead, since they are input
/// or setup errors rather than rejections.
impl From<jolt_transcript::TranscriptError> for AkitaError {
    fn from(error: jolt_transcript::TranscriptError) -> Self {
        use jolt_transcript::TranscriptError;
        match error {
            TranscriptError::ClearMessageOnMaskedChannel => {
                Self::UnsupportedSchedule(error.to_string())
            }
            TranscriptError::Truncated
            | TranscriptError::NonCanonical
            | TranscriptError::OutOfBounds
            | TranscriptError::TrailingBytes
            | TranscriptError::CursorMismatch
            | TranscriptError::GrindingRejected
            | TranscriptError::UnsupportedGrinding
            | TranscriptError::GrindingExhausted
            | TranscriptError::Poisoned => Self::InvalidProof,
        }
    }
}

/// A failure of the shared single-instance sumcheck driver.
///
/// Transcript failures reject the proof as [`jolt_transcript::TranscriptError`]
/// does. A malformed public shape is a setup error. Every other variant,
/// including a failed prover self-check (a kernel bug), is a failed internal
/// invariant.
impl<F: jolt_algebra::Field> From<jolt_sumcheck::SumcheckError<F>> for AkitaError {
    fn from(error: jolt_sumcheck::SumcheckError<F>) -> Self {
        use jolt_sumcheck::SumcheckError;
        match error {
            SumcheckError::Transcript(error) => error.into(),
            // A masked channel's error crosses the driver boxed; it keeps the
            // masked channel's classification, so a truncated or malformed
            // masked round rejects the proof.
            #[cfg(feature = "zk")]
            SumcheckError::PrivateChannel(source) => {
                match source.downcast::<jolt_zk::masking::ZkError>() {
                    Ok(error) => (*error).into(),
                    Err(source) => Self::Internal(source.to_string()),
                }
            }
            SumcheckError::ZeroDegreeBound
            | SumcheckError::ShapeOverflow { .. }
            | SumcheckError::EqFactoredZeroRounds
            | SumcheckError::EqualityPointLengthMismatch { .. }
            | SumcheckError::RoundCountMismatch { .. } => Self::InvalidSetup(error.to_string()),
            SumcheckError::RoundCheckFailed { .. }
            | SumcheckError::DegreeBoundExceeded { .. }
            | SumcheckError::ZeroBatchDegree { .. }
            | SumcheckError::BatchMemberRoundsOutOfRange { .. }
            | SumcheckError::BatchPaddingExponentOutOfRange { .. }
            | SumcheckError::BatchMemberWindowOverflow { .. }
            | SumcheckError::DegreeOverflow { .. }
            | SumcheckError::RoundSumCoefficientCountMismatch { .. }
            | SumcheckError::InvalidIntegerDomain { .. }
            | SumcheckError::EmptyRoundCoefficients
            | SumcheckError::MissingEvaluationSource { .. }
            | SumcheckError::ZeroCommitmentCapacity
            | SumcheckError::RoundExceedsCommitmentCapacity { .. }
            | SumcheckError::BatchMemberCountMismatch { .. }
            | SumcheckError::BatchMemberRoundsMismatch { .. }
            | SumcheckError::BatchMemberWindowOutOfRange { .. }
            | SumcheckError::MissingRoundMessage { .. }
            | SumcheckError::RoundMemberIndexOutOfRange { .. }
            | SumcheckError::BatchedPointRangeOverflow { .. }
            | SumcheckError::BatchedPointOutOfRange { .. }
            | SumcheckError::IntegerDomainNotDistinct { .. }
            | SumcheckError::WrongNumberOfChallenges { .. } => Self::Internal(error.to_string()),
            // `SumcheckError` is `#[non_exhaustive]`. A variant added upstream
            // must be classified above when Akita bumps jolt-sumcheck.
            _ => Self::Internal(error.to_string()),
        }
    }
}

/// A masked channel's failure (`specs/whiteout-akita.md` §4.5). A transcript
/// failure rejects the proof as [`jolt_transcript::TranscriptError`] does; a
/// pad schedule or message field that disagrees with the schedule is a setup
/// error; a decoded value failing its check is a failed prover invariant.
#[cfg(feature = "zk")]
impl From<jolt_zk::masking::ZkError> for AkitaError {
    fn from(error: jolt_zk::masking::ZkError) -> Self {
        use jolt_zk::masking::ZkError;
        match error {
            ZkError::Transcript(error) => error.into(),
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
