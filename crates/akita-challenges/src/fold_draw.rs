//! Fold-challenge drawing: one public payload and one challenge message per
//! commitment group, over the live channel or a prover-side preview.

use crate::{
    Challenges, OperatorNormRejection, SparseChallenge, SparseChallengeConfig,
    SparseChallengeStream,
};
use akita_error::AkitaError;

const FOLD_CHALLENGE_ROUND_DOMAIN: &[u8] = b"akita/fold-challenge-round/v2";
/// Label suffix of the sparse-challenge Fiat–Shamir absorb buffer for one draw
/// batch. It is embedded in the public fold payload.
const CHALLENGE_WITNESS_FOLD: &[u8] = b"ak/c/wf";
const SUBRING_COEFFICIENT_PACKING_DRAW_DOMAIN: &[u8] =
    b"akita/subring-coefficient-packing-fold-challenge/v1";

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

/// Build the canonical transcript prefix for one group-local fold draw.
///
/// The prefix binds the group index, exact `num_live_blocks`, claim count, and
/// challenge count before the sparse challenge message is squeezed.
///
/// # Errors
///
/// Returns an error if a platform-sized count does not fit the canonical u64
/// encoding.
pub fn fold_challenge_sample_label(
    group_index: usize,
    num_live_blocks: usize,
    num_claims: usize,
) -> Result<Vec<u8>, AkitaError> {
    let group_index = u64::try_from(group_index)
        .map_err(|_| AkitaError::InvalidSetup("fold group index exceeds u64".to_string()))?;
    let num_live_blocks = u64::try_from(num_live_blocks)
        .map_err(|_| AkitaError::InvalidSetup("num_live_blocks exceeds u64".to_string()))?;
    let num_claims = u64::try_from(num_claims)
        .map_err(|_| AkitaError::InvalidSetup("fold claim count exceeds u64".to_string()))?;
    let base_label = CHALLENGE_WITNESS_FOLD;
    let mut label = Vec::with_capacity(FOLD_CHALLENGE_ROUND_DOMAIN.len() + base_label.len() + 24);
    label.extend_from_slice(FOLD_CHALLENGE_ROUND_DOMAIN);
    label.extend_from_slice(&group_index.to_le_bytes());
    label.extend_from_slice(&num_live_blocks.to_le_bytes());
    label.extend_from_slice(&num_claims.to_le_bytes());
    label.extend_from_slice(base_label);
    Ok(label)
}

pub trait FoldDraw {
    /// Absorb one group's public payload, squeeze its challenge message of
    /// exactly `len` bytes and decode it. `decode` returns `None` when an
    /// operator-norm pool runs out; the verifier adapter rejects then.
    fn draw_message(
        &mut self,
        payload: &[u8],
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<Vec<SparseChallenge>>, AkitaError>,
    ) -> Result<Option<Vec<SparseChallenge>>, AkitaError>;

    #[cfg(feature = "logging-transcript")]
    fn record_challenge_range(&mut self, _group_index: usize, _coordinate_count: usize) {}

    /// Draw one group's challenges: absorb its public payload, then squeeze
    /// and decode one message of the family's fixed length.
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
        let FoldChallengeFamily {
            domain,
            ring_d,
            config,
            rejection,
        } = *family;
        let total = num_live_blocks.checked_mul(num_claims).ok_or_else(|| {
            AkitaError::InvalidSetup("sparse challenge count overflow".to_string())
        })?;
        let sample_label = fold_challenge_sample_label(group_index, num_live_blocks, num_claims)?;
        let domain_sep = config.domain_separator_bytes();
        let mut absorb_buf = Vec::with_capacity(sample_label.len() + 8 + 8 + domain_sep.len());
        absorb_buf.extend_from_slice(&sample_label);
        absorb_buf.extend_from_slice(&(total as u64).to_le_bytes());
        absorb_buf.extend_from_slice(&(ring_d as u64).to_le_bytes());
        absorb_buf.extend_from_slice(&domain_sep);
        if matches!(
            domain,
            FoldChallengeDrawDomain::SubringCoefficientPacking { .. }
        ) {
            absorb_buf.extend_from_slice(SUBRING_COEFFICIENT_PACKING_DRAW_DOMAIN);
        }
        if let Some(rejection) = rejection {
            absorb_buf.extend_from_slice(&rejection.domain_separator_bytes());
        }
        let stream = family.stream(total)?;
        let challenges = self.draw_message(&absorb_buf, stream.message_len(), |message| {
            stream.decode(message)
        })?;
        #[cfg(feature = "logging-transcript")]
        self.record_challenge_range(group_index, total);
        challenges
            .map(|challenges| Challenges::from_sparse(challenges, num_live_blocks, num_claims))
            .transpose()
    }
}

fn fold_record(
    level: u32,
    group: u32,
    payload_len: usize,
    message_len: usize,
) -> akita_transcript::ProtocolContextRecord {
    let encoded_bytes = payload_len as u64;
    akita_transcript::ProtocolContextRecord::new(
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_FOLD_CHALLENGE,
            level,
            group,
            ..akita_transcript::ProtocolSiteId::default()
        }
        .to_bytes(),
        akita_transcript::ProtocolMessageKind::Challenge as u32,
        encoded_bytes,
        encoded_bytes,
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
        payload: &[u8],
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<Vec<SparseChallenge>>, AkitaError>,
    ) -> Result<Option<Vec<SparseChallenge>>, AkitaError> {
        self.preview.challenge_message(payload, len, decode)
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
        payload: &[u8],
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<Vec<SparseChallenge>>, AkitaError>,
    ) -> Result<Option<Vec<SparseChallenge>>, AkitaError> {
        akita_transcript::fold_challenge_message(
            self.state,
            fold_record(self.level, self.group, payload.len(), len),
            payload,
            len,
            decode,
        )
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
        payload: &[u8],
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<Vec<SparseChallenge>>, AkitaError>,
    ) -> Result<Option<Vec<SparseChallenge>>, AkitaError> {
        akita_transcript::fold_challenge_message(
            self.state,
            fold_record(self.level, self.group, payload.len(), len),
            payload,
            len,
            |message| decode(message)?.map(Some).ok_or(AkitaError::InvalidProof),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct CapturingDraw {
        payloads: Vec<Vec<u8>>,
    }

    impl FoldDraw for CapturingDraw {
        fn draw_message(
            &mut self,
            payload: &[u8],
            len: usize,
            decode: impl FnOnce(&[u8]) -> Result<Option<Vec<SparseChallenge>>, AkitaError>,
        ) -> Result<Option<Vec<SparseChallenge>>, AkitaError> {
            self.payloads.push(payload.to_vec());
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
    fn evaluation_trace_draw_uses_nonce_free_group_payload() {
        let config = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let family = family(FoldChallengeDrawDomain::EvaluationTrace, 64, config).unwrap();
        let mut draw = CapturingDraw::default();
        let challenges = draw
            .draw_fold_challenges(&family, 2, 3, 4)
            .unwrap()
            .unwrap();
        assert_eq!(challenges.num_claims(), 4);
        let label = fold_challenge_sample_label(2, 3, 4).unwrap();
        let mut expected = label;
        expected.extend_from_slice(&12u64.to_le_bytes());
        expected.extend_from_slice(&64u64.to_le_bytes());
        expected.extend_from_slice(&config.domain_separator_bytes());
        assert_eq!(draw.payloads, vec![expected]);
    }

    #[test]
    fn packing_draw_binds_method_and_subring_dimension() {
        let packing = |d: usize| FoldChallengeDrawDomain::SubringCoefficientPacking {
            challenge_subring_dimension: d,
        };
        let config_64 = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let config_128 = SparseChallengeConfig::production_for_ring_dim(128).unwrap();
        let mut draw_64 = CapturingDraw::default();
        draw_64
            .draw_fold_challenges(&family(packing(64), 64, config_64).unwrap(), 0, 2, 1)
            .unwrap();
        assert!(draw_64.payloads[0].ends_with(SUBRING_COEFFICIENT_PACKING_DRAW_DOMAIN));

        let mut draw_128 = CapturingDraw::default();
        draw_128
            .draw_fold_challenges(&family(packing(128), 128, config_128).unwrap(), 0, 2, 1)
            .unwrap();
        assert_ne!(draw_64.payloads, draw_128.payloads);
        assert!(family(packing(64), 128, config_128).is_err());
        assert!(FoldChallengeFamily::new(
            packing(64),
            64,
            crate::D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            Some(OperatorNormRejection::D64_SELECTIVE_L2),
        )
        .is_err());

        let mut evaluation_trace = CapturingDraw::default();
        evaluation_trace
            .draw_fold_challenges(
                &family(FoldChallengeDrawDomain::EvaluationTrace, 64, config_64).unwrap(),
                0,
                2,
                1,
            )
            .unwrap();
        assert_ne!(draw_64.payloads, evaluation_trace.payloads);
    }
}
