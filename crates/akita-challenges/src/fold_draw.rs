//! Fold-challenge drawing: one challenge message per commitment group, over
//! the live channel or a prover-side preview.
//!
//! A draw absorbs nothing. The schedule bound in the instance descriptor fixes
//! the family, and the group's position in the transcript fixes its index and
//! counts, so the message continues the squeeze run before it.

use crate::{
    Challenges, OperatorNormRejection, SparseChallenge, SparseChallengeConfig,
    SparseChallengeStream,
};
use akita_error::AkitaError;
use akita_transcript::ProofChannel;

/// Algebraic domain of one fold-challenge draw.
///
/// Coefficient packing adds an explicit method domain and challenge-subring
/// dimension before the challenge message is squeezed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldChallengeDrawDomain {
    EvaluationTrace,
    SubringCoefficientPacking { challenge_subring_dimension: usize },
}

/// The sparse challenge family one fold group draws from: its algebraic
/// domain, ring dimension, shell and operator-norm policy.
///
/// Schedules build it once; the grinding plan prices the fold from it and
/// every draw decodes with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldChallengeFamily {
    domain: FoldChallengeDrawDomain,
    ring_d: usize,
    config: SparseChallengeConfig,
    rejection: Option<OperatorNormRejection>,
}

impl FoldChallengeFamily {
    /// Validate one family.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] when a coefficient-packing family
    /// disagrees with its subring dimension or carries an operator-norm
    /// policy, and the errors of [`SparseChallengeStream::new`] for an
    /// unsupported shell or policy.
    pub fn new(
        domain: FoldChallengeDrawDomain,
        ring_d: usize,
        config: SparseChallengeConfig,
        rejection: Option<OperatorNormRejection>,
    ) -> Result<Self, AkitaError> {
        if let FoldChallengeDrawDomain::SubringCoefficientPacking {
            challenge_subring_dimension,
        } = domain
        {
            if ring_d != challenge_subring_dimension {
                return Err(AkitaError::InvalidInput(
                    "coefficient-packing draw dimension mismatch".into(),
                ));
            }
            if rejection.is_some() {
                return Err(AkitaError::InvalidInput(
                    "coefficient-packing draws require the L-infinity security route".into(),
                ));
            }
        }
        let family = Self {
            domain,
            ring_d,
            config,
            rejection,
        };
        family.stream(1)?;
        Ok(family)
    }

    /// Message layout and decoder for `count` challenges of this family.
    ///
    /// # Errors
    ///
    /// As for [`SparseChallengeStream::new`].
    pub fn stream(&self, count: usize) -> Result<SparseChallengeStream, AkitaError> {
        SparseChallengeStream::new(self.ring_d, &self.config, self.rejection, count)
    }

    #[must_use]
    pub const fn domain(&self) -> FoldChallengeDrawDomain {
        self.domain
    }

    #[must_use]
    pub const fn ring_d(&self) -> usize {
        self.ring_d
    }

    #[must_use]
    pub const fn config(&self) -> SparseChallengeConfig {
        self.config
    }

    #[must_use]
    pub const fn rejection(&self) -> Option<OperatorNormRejection> {
        self.rejection
    }
}

pub trait FoldDraw {
    /// Squeeze one group's challenge message of exactly `len` bytes and
    /// decode it. `decode` returns `None` when an operator-norm pool runs out;
    /// the verifier adapter rejects then.
    fn draw_message(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<Vec<SparseChallenge>>, AkitaError>,
    ) -> Result<Option<Vec<SparseChallenge>>, AkitaError>;

    #[cfg(feature = "logging-transcript")]
    fn record_challenge_range(&mut self, _group_index: usize, _coordinate_count: usize) {}

    /// Draw one group's challenges: squeeze and decode one message of the
    /// family's fixed length.
    ///
    /// Returns `None` when the message's operator-norm pool runs out, which
    /// a prover treats as a failed fold-response candidate. The verifier
    /// adapter rejects instead, so it never returns `None`.
    fn draw_fold_challenges(
        &mut self,
        family: &FoldChallengeFamily,
        group_index: usize,
        num_live_blocks: usize,
        num_claims: usize,
    ) -> Result<Option<Challenges>, AkitaError> {
        if num_live_blocks == 0 || num_claims == 0 {
            return Err(AkitaError::InvalidInput(
                "fold challenges require positive num_live_blocks and claims".to_string(),
            ));
        }
        let total = num_live_blocks.checked_mul(num_claims).ok_or_else(|| {
            AkitaError::InvalidSetup("sparse challenge count overflow".to_string())
        })?;
        // The index is positional; only the diagnostic range records it.
        #[cfg(not(feature = "logging-transcript"))]
        let _ = group_index;
        let stream = family.stream(total)?;
        let challenges =
            self.draw_message(stream.message_len(), |message| stream.decode(message))?;
        #[cfg(feature = "logging-transcript")]
        self.record_challenge_range(group_index, total);
        challenges
            .map(|challenges| Challenges::from_sparse(challenges, num_live_blocks, num_claims))
            .transpose()
    }
}

fn fold_record(level: u32, group: u32, message_len: usize) -> akita_transcript::ProtocolContextRecord {
    akita_transcript::ProtocolContextRecord::new(
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_FOLD_CHALLENGE,
            level,
            group,
            ..akita_transcript::ProtocolSiteId::default()
        }
        .to_bytes(),
        akita_transcript::ProtocolMessageKind::Challenge as u32,
        0,
        0,
        message_len as u64,
    )
}

/// One group-local fold draw against a candidate public state.
#[cfg(feature = "prover")]
pub struct PreviewFoldDraw<'a> {
    preview: &'a mut akita_transcript::FoldPreview,
}

#[cfg(feature = "prover")]
impl<'a> PreviewFoldDraw<'a> {
    /// Bind this short-lived draw adapter to one schedule-derived fold group.
    #[must_use]
    pub const fn new(preview: &'a mut akita_transcript::FoldPreview) -> Self {
        Self { preview }
    }
}

#[cfg(feature = "prover")]
impl FoldDraw for PreviewFoldDraw<'_> {
    fn draw_message(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<Vec<SparseChallenge>>, AkitaError>,
    ) -> Result<Option<Vec<SparseChallenge>>, AkitaError> {
        self.preview.challenge_message(len, decode)
    }
}

/// One group-local fold draw against the live prover channel.
pub struct ProverFoldDraw<'a> {
    state: &'a mut akita_transcript::ProverChannel,
    level: u32,
    group: u32,
}

impl<'a> ProverFoldDraw<'a> {
    /// Bind this short-lived draw adapter to one schedule-derived fold group.
    #[must_use]
    pub const fn new(
        state: &'a mut akita_transcript::ProverChannel,
        level: u32,
        group: u32,
    ) -> Self {
        Self {
            state,
            level,
            group,
        }
    }
}

impl FoldDraw for ProverFoldDraw<'_> {
    fn draw_message(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<Vec<SparseChallenge>>, AkitaError>,
    ) -> Result<Option<Vec<SparseChallenge>>, AkitaError> {
        self.state.context(fold_record(self.level, self.group, len));
        self.state.challenge_message(len, decode)
    }
}

/// One group-local fold draw against the live verifier channel.
pub struct VerifierFoldDraw<'a, 'proof> {
    state: &'a mut akita_transcript::VerifierChannel<'proof>,
    level: u32,
    group: u32,
}

impl<'a, 'proof> VerifierFoldDraw<'a, 'proof> {
    /// Bind this short-lived draw adapter to one schedule-derived fold group.
    #[must_use]
    pub const fn new(
        state: &'a mut akita_transcript::VerifierChannel<'proof>,
        level: u32,
        group: u32,
    ) -> Self {
        Self {
            state,
            level,
            group,
        }
    }
}

impl FoldDraw for VerifierFoldDraw<'_, '_> {
    fn draw_message(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<Vec<SparseChallenge>>, AkitaError>,
    ) -> Result<Option<Vec<SparseChallenge>>, AkitaError> {
        self.state.context(fold_record(self.level, self.group, len));
        self.state.challenge_message(len, |message| {
            decode(message)?.map(Some).ok_or(AkitaError::InvalidProof)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct CapturingDraw {
        message_lens: Vec<usize>,
    }

    impl FoldDraw for CapturingDraw {
        fn draw_message(
            &mut self,
            len: usize,
            decode: impl FnOnce(&[u8]) -> Result<Option<Vec<SparseChallenge>>, AkitaError>,
        ) -> Result<Option<Vec<SparseChallenge>>, AkitaError> {
            self.message_lens.push(len);
            decode(&vec![7; len])
        }
    }

    fn family(
        domain: FoldChallengeDrawDomain,
        ring_d: usize,
        config: SparseChallengeConfig,
    ) -> Result<FoldChallengeFamily, AkitaError> {
        FoldChallengeFamily::new(domain, ring_d, config, None)
    }

    #[test]
    fn evaluation_trace_draw_squeezes_one_family_message() {
        let config = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let family = family(FoldChallengeDrawDomain::EvaluationTrace, 64, config).unwrap();
        let mut draw = CapturingDraw::default();
        let challenges = draw
            .draw_fold_challenges(&family, 2, 3, 4)
            .unwrap()
            .unwrap();
        assert_eq!(challenges.num_claims(), 4);
        assert_eq!(
            draw.message_lens,
            vec![family.stream(12).unwrap().message_len()]
        );
        assert!(draw.draw_fold_challenges(&family, 0, 0, 4).is_err());
        assert!(draw.draw_fold_challenges(&family, 0, 3, 0).is_err());
        assert_eq!(draw.message_lens.len(), 1);
    }

    #[test]
    fn packing_family_requires_matching_subring_dimension() {
        let packing = |d: usize| FoldChallengeDrawDomain::SubringCoefficientPacking {
            challenge_subring_dimension: d,
        };
        let config_64 = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let config_128 = SparseChallengeConfig::production_for_ring_dim(128).unwrap();
        let mut draw_64 = CapturingDraw::default();
        draw_64
            .draw_fold_challenges(&family(packing(64), 64, config_64).unwrap(), 0, 2, 1)
            .unwrap();
        let mut draw_128 = CapturingDraw::default();
        draw_128
            .draw_fold_challenges(&family(packing(128), 128, config_128).unwrap(), 0, 2, 1)
            .unwrap();
        assert_eq!(draw_64.message_lens.len(), 1);
        assert_eq!(draw_128.message_lens.len(), 1);
        assert!(family(packing(64), 128, config_128).is_err());
        assert!(FoldChallengeFamily::new(
            packing(64),
            64,
            crate::D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            Some(OperatorNormRejection::D64_SELECTIVE_L2),
        )
        .is_err());
    }
}
