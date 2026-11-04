//! Fold-challenge drawing: one challenge message per commitment group, over
//! the live channel or a prover-side preview.
//!
//! A draw absorbs nothing. The schedule bound in the instance descriptor fixes
//! the family, and the group's position in the transcript fixes its index and
//! counts, so the message continues the squeeze run before it.

use akita_error::AkitaError;
use akita_transcript::ProofChannel;
use jolt_crypto::challenge::Challenges;

pub use akita_params::{FoldChallengeDrawDomain, FoldChallengeFamily};

pub trait FoldDraw {
    /// Squeeze one group's challenge message of exactly `len` bytes and
    /// decode it. `decode` returns `None` when an operator-norm pool runs out;
    /// the verifier adapter rejects then.
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError>;

    /// Draw one group's challenges: squeeze and decode one message of the
    /// family's fixed length.
    ///
    /// Returns `None` when the message's operator-norm pool runs out, which
    /// a prover treats as a failed fold-response candidate. The verifier
    /// adapter rejects instead, so it never returns `None`.
    fn draw_fold_challenges(
        &mut self,
        family: &FoldChallengeFamily,
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
        let stream = family.stream(total)?;
        self.draw_message(stream.message_len(), |message| {
            stream
                .decode_challenges(message, num_live_blocks, num_claims)
                .map_err(AkitaError::from)
        })
    }
}

fn fold_record(
    level: u32,
    group: u32,
    message_len: usize,
) -> akita_transcript::ProtocolContextRecord {
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
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
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
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        self.state.context(fold_record(self.level, self.group, len));
        self.state.challenge_message(len, decode)
    }
}

/// One group-local fold draw against the live verifier channel, clear or
/// masked: fold challenges are public in both modes.
pub struct VerifierFoldDraw<'a, S> {
    state: &'a mut S,
    level: u32,
    group: u32,
}

impl<'a, S: akita_transcript::ProofChannel> VerifierFoldDraw<'a, S> {
    /// Bind this short-lived draw adapter to one schedule-derived fold group.
    #[must_use]
    pub const fn new(state: &'a mut S, level: u32, group: u32) -> Self {
        Self {
            state,
            level,
            group,
        }
    }
}

impl<S: akita_transcript::ProofChannel> FoldDraw for VerifierFoldDraw<'_, S> {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        self.state.context(fold_record(self.level, self.group, len));
        self.state.challenge_message(len, |message| {
            decode(message)?.map(Some).ok_or(AkitaError::InvalidProof)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_crypto::challenge::{OperatorNormRejection, SparseChallengeConfig};

    #[derive(Default)]
    struct CapturingDraw {
        message_lens: Vec<usize>,
    }

    impl FoldDraw for CapturingDraw {
        fn draw_message<T>(
            &mut self,
            len: usize,
            decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
        ) -> Result<Option<T>, AkitaError> {
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
        let challenges = draw.draw_fold_challenges(&family, 3, 4).unwrap().unwrap();
        assert_eq!(challenges.num_claims(), 4);
        assert_eq!(
            draw.message_lens,
            vec![family.stream(12).unwrap().message_len()]
        );
        assert!(draw.draw_fold_challenges(&family, 0, 4).is_err());
        assert!(draw.draw_fold_challenges(&family, 3, 0).is_err());
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
            .draw_fold_challenges(&family(packing(64), 64, config_64).unwrap(), 2, 1)
            .unwrap();
        let mut draw_128 = CapturingDraw::default();
        draw_128
            .draw_fold_challenges(&family(packing(128), 128, config_128).unwrap(), 2, 1)
            .unwrap();
        assert_eq!(draw_64.message_lens.len(), 1);
        assert_eq!(draw_128.message_lens.len(), 1);
        assert!(family(packing(64), 128, config_128).is_err());
        assert!(FoldChallengeFamily::new(
            packing(64),
            64,
            jolt_crypto::challenge::D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            Some(OperatorNormRejection::D64_SELECTIVE_L2),
        )
        .is_err());
    }
}
