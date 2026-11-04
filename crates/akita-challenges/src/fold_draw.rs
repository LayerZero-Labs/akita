//! Fold-challenge drawing: one challenge message per commitment group, over
//! the live channel or a prover-side preview.
//!
//! A draw absorbs nothing. The schedule bound in the instance descriptor fixes
//! the family, and the group's position in the transcript fixes its index and
//! counts, so the message continues the squeeze run before it.

use crate::sampler::SparseChallengeShell;
use crate::{Challenges, FoldChallengeDistribution, OperatorNormRejection, SparseChallengeStream};
use akita_error::AkitaError;
use akita_transcript::ProofChannel;

/// Algebraic domain of one fold-challenge draw.
///
/// Coefficient packing draws in a challenge subring whose dimension must equal
/// the family's ring dimension. Like the family, the domain is bound by the
/// descriptor, not absorbed by the draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldChallengeDrawDomain {
    EvaluationTrace,
    SubringCoefficientPacking { challenge_subring_dimension: usize },
}

/// The sparse challenge family one fold group draws from: its algebraic
/// domain, ring dimension, distribution and operator-norm policy.
///
/// Schedules build it once; the grinding plan prices the fold from it and
/// every draw decodes with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldChallengeFamily {
    domain: FoldChallengeDrawDomain,
    shell: FamilyShell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FamilyShell {
    SignedSparse(SparseChallengeShell),
    /// Decoded from one 32-byte message by indexed SHAKE256 substreams.
    #[cfg(feature = "labinius-challenges")]
    BinaryRoot {
        ring_d: usize,
    },
}

impl FoldChallengeFamily {
    /// Validate one family.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] when a coefficient-packing family
    /// disagrees with its subring dimension or carries an operator-norm
    /// policy, and [`AkitaError::InvalidSetup`] for an unsupported shell,
    /// distribution or policy.
    pub fn new(
        domain: FoldChallengeDrawDomain,
        ring_d: usize,
        distribution: impl Into<FoldChallengeDistribution>,
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
        let shell = match distribution.into() {
            FoldChallengeDistribution::SignedSparse(config) => {
                FamilyShell::SignedSparse(SparseChallengeShell::new(ring_d, &config, rejection)?)
            }
            #[cfg(feature = "labinius-challenges")]
            FoldChallengeDistribution::BinaryRoot => {
                if rejection.is_some() {
                    return Err(AkitaError::InvalidSetup(
                        "binary root family has no operator-norm rejection policy".into(),
                    ));
                }
                crate::binary::root_family::validate(ring_d, 0)
                    .map_err(|error| AkitaError::InvalidSetup(error.into()))?;
                FamilyShell::BinaryRoot { ring_d }
            }
        };
        Ok(Self { domain, shell })
    }

    /// Message layout and decoder for `count` challenges of a signed-sparse
    /// family.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `count` is zero, the
    /// message exceeds the message limit, or the family is not signed-sparse.
    pub fn stream(&self, count: usize) -> Result<SparseChallengeStream, AkitaError> {
        match self.shell {
            FamilyShell::SignedSparse(shell) => shell.stream(count),
            #[cfg(feature = "labinius-challenges")]
            FamilyShell::BinaryRoot { .. } => Err(AkitaError::InvalidSetup(
                "binary root challenges have no sparse message stream".into(),
            )),
        }
    }

    /// Byte length of the one message that `count` challenges decode from.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `count` is zero or the
    /// message exceeds the message limit.
    pub fn message_len(&self, count: usize) -> Result<usize, AkitaError> {
        match self.shell {
            FamilyShell::SignedSparse(shell) => Ok(shell.stream(count)?.message_len()),
            #[cfg(feature = "labinius-challenges")]
            FamilyShell::BinaryRoot { .. } => {
                if count == 0 {
                    return Err(AkitaError::InvalidSetup(
                        "binary root challenge count must be positive".into(),
                    ));
                }
                Ok(crate::binary::BINARY_CHALLENGE_SEED_LEN)
            }
        }
    }

    /// Guaranteed `log2` size of the set each challenge is uniform on, for
    /// any challenge count: `K` for an unfiltered family, the certified
    /// exponent for an operator-norm one, and the floor of the exact support
    /// entropy for the binary root family.
    #[must_use]
    pub fn set_bits(&self) -> u32 {
        match self.shell {
            FamilyShell::SignedSparse(shell) => shell.set_bits(),
            #[cfg(feature = "labinius-challenges")]
            FamilyShell::BinaryRoot { ring_d } => {
                crate::binary::root_family::entropy_bits(ring_d).floor() as u32
            }
        }
    }

    #[must_use]
    pub const fn domain(&self) -> FoldChallengeDrawDomain {
        self.domain
    }

    #[must_use]
    pub const fn ring_d(&self) -> usize {
        match self.shell {
            FamilyShell::SignedSparse(shell) => shell.ring_d(),
            #[cfg(feature = "labinius-challenges")]
            FamilyShell::BinaryRoot { ring_d } => ring_d,
        }
    }

    #[must_use]
    pub const fn distribution(&self) -> FoldChallengeDistribution {
        match self.shell {
            FamilyShell::SignedSparse(shell) => {
                FoldChallengeDistribution::SignedSparse(shell.config())
            }
            #[cfg(feature = "labinius-challenges")]
            FamilyShell::BinaryRoot { .. } => FoldChallengeDistribution::BinaryRoot,
        }
    }

    #[must_use]
    pub const fn rejection(&self) -> Option<OperatorNormRejection> {
        match self.shell {
            FamilyShell::SignedSparse(shell) => shell.rejection(),
            #[cfg(feature = "labinius-challenges")]
            FamilyShell::BinaryRoot { .. } => None,
        }
    }
}

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
        match family.shell {
            FamilyShell::SignedSparse(shell) => {
                let stream = shell.stream(total)?;
                let challenges =
                    self.draw_message(stream.message_len(), |message| stream.decode(message))?;
                challenges
                    .map(|challenges| {
                        Challenges::from_sampled(
                            challenges,
                            num_live_blocks,
                            num_claims,
                            shell.ring_d(),
                            shell.config(),
                        )
                    })
                    .transpose()
            }
            #[cfg(feature = "labinius-challenges")]
            FamilyShell::BinaryRoot { .. } => {
                let challenges =
                    self.draw_message(crate::binary::BINARY_CHALLENGE_SEED_LEN, |message| {
                        let seed = message.try_into().map_err(|_| {
                            AkitaError::Internal("binary root message has the wrong width".into())
                        })?;
                        crate::binary::root_family::sample(seed, total).map(Some)
                    })?;
                challenges
                    .map(|challenges| {
                        Challenges::from_sparse(challenges, num_live_blocks, num_claims)
                    })
                    .transpose()
            }
        }
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
    use crate::SparseChallengeConfig;

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
            crate::D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            Some(OperatorNormRejection::D64_SELECTIVE_L2),
        )
        .is_err());
    }
}
