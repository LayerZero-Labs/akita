//! Fold-challenge drawing over a fold group's family: the draw adapters live
//! in `akita-challenges`, below the family in `akita-params`; this trait
//! decodes one group's message with that family.

use akita_error::AkitaError;
use jolt_crypto::challenge::Challenges;

#[cfg(feature = "prover")]
pub use akita_challenges::PreviewFoldDraw;
pub use akita_challenges::{FoldDraw, ProverFoldDraw, VerifierFoldDraw};
pub use akita_params::{FoldChallengeDrawDomain, FoldChallengeFamily};

/// Draws one fold group's challenges from its [`FoldChallengeFamily`].
pub trait FoldChallengeDraw: FoldDraw {
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
        #[cfg(feature = "labinius-trinomial")]
        if family.distribution() == akita_params::FoldChallengeDistribution::BinaryRoot {
            let challenges =
                self.draw_message(akita_challenges::BINARY_CHALLENGE_SEED_LEN, |message| {
                    let seed = message.try_into().map_err(|_| {
                        AkitaError::Internal("binary root message has the wrong width".into())
                    })?;
                    akita_challenges::root_family::sample(seed, total).map(Some)
                })?;
            return challenges
                .map(|challenges| {
                    Challenges::from_sparse(challenges, num_live_blocks, num_claims)
                        .map_err(AkitaError::from)
                })
                .transpose();
        }
        let stream = family.stream(total)?;
        self.draw_message(stream.message_len(), |message| {
            stream
                .decode_challenges(message, num_live_blocks, num_claims)
                .map_err(AkitaError::from)
        })
    }
}

impl<D: FoldDraw + ?Sized> FoldChallengeDraw for D {}

#[cfg(all(test, feature = "labinius-trinomial", feature = "prover"))]
mod binary_root_tests;

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

#[cfg(test)]
mod field_switch_tests;
