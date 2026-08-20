//! Fold-challenge preview drawing for prover-side Fiat–Shamir grinding.

use crate::sampler::MAX_STACK_RING_DIM;
use crate::{Challenges, OperatorNormRejection, SparseChallengeConfig};
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::{FoldChallengeSeedPreview, Transcript, FOLD_CHALLENGE_SEED_LEN};
use std::marker::PhantomData;

const FOLD_CHALLENGE_ROUND_DOMAIN: &[u8] = b"akita/fold-challenge-round/v1";
const SUBRING_COEFFICIENT_PACKING_DRAW_DOMAIN: &[u8] =
    b"akita/subring-coefficient-packing-fold-challenge/v1";

/// Algebraic domain of one fold-challenge draw.
///
/// The evaluation-trace variant preserves the historical transcript encoding.
/// Coefficient packing adds an explicit method domain and challenge-subring
/// dimension before the seed is squeezed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldChallengeDrawDomain {
    EvaluationTrace,
    SubringCoefficientPacking { challenge_subring_dimension: usize },
}

/// Build the canonical transcript prefix for one group-local fold draw.
///
/// The prefix binds the group index, exact `num_live_blocks`, claim count, and
/// challenge count before the sparse challenge seed is squeezed.
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
    let base_label = akita_transcript::labels::CHALLENGE_WITNESS_FOLD;
    let mut label = Vec::with_capacity(FOLD_CHALLENGE_ROUND_DOMAIN.len() + base_label.len() + 24);
    label.extend_from_slice(FOLD_CHALLENGE_ROUND_DOMAIN);
    label.extend_from_slice(&group_index.to_le_bytes());
    label.extend_from_slice(&num_live_blocks.to_le_bytes());
    label.extend_from_slice(&num_claims.to_le_bytes());
    label.extend_from_slice(base_label);
    Ok(label)
}

pub trait FoldDraw {
    fn absorb_and_squeeze(&mut self, label: &[u8], payload: &[u8]) -> Vec<u8>;

    fn draw_folding_challenges(
        &mut self,
        ring_d: usize,
        group_index: usize,
        num_live_blocks: usize,
        num_claims: usize,
        cfg: &SparseChallengeConfig,
        grind_nonce: u32,
    ) -> Result<Challenges, AkitaError> {
        self.draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            ring_d,
            group_index,
            num_live_blocks,
            num_claims,
            cfg,
            grind_nonce,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_folding_challenges_with_rejection(
        &mut self,
        domain: FoldChallengeDrawDomain,
        ring_d: usize,
        group_index: usize,
        num_live_blocks: usize,
        num_claims: usize,
        cfg: &SparseChallengeConfig,
        grind_nonce: u32,
        rejection: Option<OperatorNormRejection>,
    ) -> Result<Challenges, AkitaError> {
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
        if ring_d > MAX_STACK_RING_DIM {
            return Err(AkitaError::InvalidInput(format!(
                "ring dimension {ring_d} exceeds supported stack sampler limit ({MAX_STACK_RING_DIM})"
            )));
        }
        cfg.validate_dyn(ring_d).map_err(|e| {
            AkitaError::InvalidInput(format!("invalid sparse challenge config: {e}"))
        })?;
        if let Some(rejection) = rejection {
            rejection
                .validate(ring_d, cfg)
                .map_err(|error| AkitaError::InvalidInput(error.into()))?;
        }
        if num_live_blocks == 0 || num_claims == 0 {
            return Err(AkitaError::InvalidInput(
                "fold challenges require positive num_live_blocks and claims".to_string(),
            ));
        }

        let total = num_live_blocks.checked_mul(num_claims).ok_or_else(|| {
            AkitaError::InvalidSetup("sparse challenge count overflow".to_string())
        })?;
        let sample_label = fold_challenge_sample_label(group_index, num_live_blocks, num_claims)?;
        let domain_sep = cfg.domain_separator_bytes();
        let mut absorb_buf = Vec::with_capacity(sample_label.len() + 8 + 8 + domain_sep.len() + 4);
        absorb_buf.extend_from_slice(&sample_label);
        absorb_buf.extend_from_slice(&(total as u64).to_le_bytes());
        absorb_buf.extend_from_slice(&(ring_d as u64).to_le_bytes());
        absorb_buf.extend_from_slice(&domain_sep);
        absorb_buf.extend_from_slice(&grind_nonce.to_le_bytes());
        if matches!(
            domain,
            FoldChallengeDrawDomain::SubringCoefficientPacking { .. }
        ) {
            absorb_buf.extend_from_slice(SUBRING_COEFFICIENT_PACKING_DRAW_DOMAIN);
        }
        if let Some(rejection) = rejection {
            absorb_buf.extend_from_slice(&rejection.domain_separator_bytes());
        }
        let seed = self.absorb_and_squeeze(ABSORB_SPARSE_CHALLENGE, &absorb_buf);
        let challenges = crate::sampler::sample_indexed_challenges_from_seed(
            &seed, ring_d, total, cfg, rejection,
        )?;
        Challenges::from_sparse(challenges, num_live_blocks, num_claims)
    }
}

pub struct PreviewFoldDraw<'a> {
    preview: &'a dyn FoldChallengeSeedPreview,
    absorbs: Vec<Vec<u8>>,
}

impl<'a> PreviewFoldDraw<'a> {
    pub fn new(preview: &'a dyn FoldChallengeSeedPreview) -> Self {
        Self {
            preview,
            absorbs: Vec::new(),
        }
    }
}

impl FoldDraw for PreviewFoldDraw<'_> {
    fn absorb_and_squeeze(&mut self, _label: &[u8], payload: &[u8]) -> Vec<u8> {
        self.absorbs.push(payload.to_vec());
        let absorbs = self.absorbs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        self.preview.preview_fold_challenge_seed(&absorbs)
    }
}

pub struct LiveFoldDraw<'a, F, T> {
    transcript: &'a mut T,
    _field: PhantomData<F>,
}

impl<'a, F, T> LiveFoldDraw<'a, F, T> {
    pub fn new(transcript: &'a mut T) -> Self {
        Self {
            transcript,
            _field: PhantomData::<F>,
        }
    }
}

impl<F, T> FoldDraw for LiveFoldDraw<'_, F, T>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    fn absorb_and_squeeze(&mut self, label: &[u8], payload: &[u8]) -> Vec<u8> {
        self.transcript.append_bytes(label, payload);
        self.transcript
            .challenge_bytes(CHALLENGE_SPARSE_CHALLENGE, FOLD_CHALLENGE_SEED_LEN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp64;
    use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
    use akita_transcript::AkitaTranscript;
    use sha3::digest::{ExtendableOutput, Update, XofReader};
    use sha3::Shake256;

    type TestField = Fp64<4294967197>;

    #[derive(Default)]
    struct CapturingDraw {
        payloads: Vec<Vec<u8>>,
    }

    impl FoldDraw for CapturingDraw {
        fn absorb_and_squeeze(&mut self, _label: &[u8], payload: &[u8]) -> Vec<u8> {
            self.payloads.push(payload.to_vec());
            vec![7; FOLD_CHALLENGE_SEED_LEN]
        }
    }

    fn challenge_fingerprint(challenges: &Challenges) -> [u8; 32] {
        let mut xof = Shake256::default();
        xof.update(b"akita/indexed-fold-challenge-test-vector/v1");
        for challenge in challenges.as_slice() {
            xof.update(&(challenge.positions.len() as u64).to_le_bytes());
            for &position in challenge.positions.iter() {
                xof.update(&position.to_le_bytes());
            }
            xof.update(&(challenge.coeffs.len() as u64).to_le_bytes());
            for &coefficient in challenge.coeffs.iter() {
                xof.update(&coefficient.to_le_bytes());
            }
        }
        let mut fingerprint = [0u8; 32];
        xof.finalize_xof().read(&mut fingerprint);
        fingerprint
    }

    fn draw_live(
        domain: FoldChallengeDrawDomain,
        ring_d: usize,
        cfg: &SparseChallengeConfig,
        rejection: Option<OperatorNormRejection>,
    ) -> Challenges {
        let mut transcript = AkitaTranscript::<TestField>::new(DOMAIN_AKITA_PROTOCOL);
        transcript.append_field(b"indexed-fold-test-seed", &TestField::from_u64(0x417));
        LiveFoldDraw::<TestField, _>::new(&mut transcript)
            .draw_folding_challenges_with_rejection(domain, ring_d, 2, 3, 2, cfg, 5, rejection)
            .unwrap()
    }

    #[test]
    fn evaluation_trace_draw_preserves_legacy_payload() {
        let config = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let mut draw = CapturingDraw::default();
        draw.draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            64,
            2,
            3,
            4,
            &config,
            5,
            None,
        )
        .unwrap();
        let label = fold_challenge_sample_label(2, 3, 4).unwrap();
        let mut expected = label;
        expected.extend_from_slice(&12u64.to_le_bytes());
        expected.extend_from_slice(&64u64.to_le_bytes());
        expected.extend_from_slice(&config.domain_separator_bytes());
        expected.extend_from_slice(&5u32.to_le_bytes());
        assert_eq!(draw.payloads, vec![expected]);
    }

    #[test]
    fn packing_draw_binds_method_and_subring_dimension() {
        let mut draw_64 = CapturingDraw::default();
        let config_64 = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        draw_64
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::SubringCoefficientPacking {
                    challenge_subring_dimension: 64,
                },
                64,
                0,
                2,
                1,
                &config_64,
                0,
                None,
            )
            .unwrap();
        assert!(draw_64.payloads[0].ends_with(SUBRING_COEFFICIENT_PACKING_DRAW_DOMAIN));

        let mut draw_128 = CapturingDraw::default();
        let config_128 = SparseChallengeConfig::production_for_ring_dim(128).unwrap();
        draw_128
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::SubringCoefficientPacking {
                    challenge_subring_dimension: 128,
                },
                128,
                0,
                2,
                1,
                &config_128,
                0,
                None,
            )
            .unwrap();
        assert_ne!(draw_64.payloads, draw_128.payloads);
        assert!(CapturingDraw::default()
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::SubringCoefficientPacking {
                    challenge_subring_dimension: 64,
                },
                128,
                0,
                2,
                1,
                &config_128,
                0,
                None,
            )
            .is_err());

        let mut evaluation_trace = CapturingDraw::default();
        evaluation_trace
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::EvaluationTrace,
                64,
                0,
                2,
                1,
                &config_64,
                0,
                None,
            )
            .unwrap();
        assert_ne!(draw_64.payloads, evaluation_trace.payloads);
    }

    #[test]
    fn preview_and_live_indexed_draws_match_across_groups() {
        let cfg = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let mut transcript = AkitaTranscript::<TestField>::new(DOMAIN_AKITA_PROTOCOL);
        transcript.append_field(b"indexed-fold-test-seed", &TestField::from_u64(0x417));
        let (preview_first, preview_second) = {
            let mut preview = PreviewFoldDraw::new(&transcript);
            let first = preview
                .draw_folding_challenges(64, 0, 3, 2, &cfg, 7)
                .unwrap();
            let second = preview
                .draw_folding_challenges(64, 1, 2, 2, &cfg, 7)
                .unwrap();
            (first, second)
        };
        let mut live = LiveFoldDraw::<TestField, _>::new(&mut transcript);
        let live_first = live.draw_folding_challenges(64, 0, 3, 2, &cfg, 7).unwrap();
        let live_second = live.draw_folding_challenges(64, 1, 2, 2, &cfg, 7).unwrap();
        assert_eq!(preview_first, live_first);
        assert_eq!(preview_second, live_second);
    }

    #[test]
    fn indexed_fold_challenge_golden_vectors() {
        let evaluation_cfg = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let evaluation = draw_live(
            FoldChallengeDrawDomain::EvaluationTrace,
            64,
            &evaluation_cfg,
            None,
        );
        let packing = draw_live(
            FoldChallengeDrawDomain::SubringCoefficientPacking {
                challenge_subring_dimension: 64,
            },
            64,
            &evaluation_cfg,
            None,
        );
        let rejected_d64 = draw_live(
            FoldChallengeDrawDomain::EvaluationTrace,
            64,
            &crate::D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            Some(OperatorNormRejection::D64_SELECTIVE_L2),
        );
        let rejected_d128 = draw_live(
            FoldChallengeDrawDomain::EvaluationTrace,
            128,
            &crate::D128_SELECTIVE_L2_CHALLENGE_CONFIG,
            Some(OperatorNormRejection::D128_SELECTIVE_L2),
        );

        assert_eq!(
            challenge_fingerprint(&evaluation),
            [
                225, 68, 222, 77, 115, 62, 167, 225, 108, 115, 143, 222, 246, 232, 154, 117, 54,
                149, 6, 181, 16, 86, 107, 181, 8, 132, 152, 50, 215, 177, 72, 148,
            ]
        );
        assert_eq!(
            challenge_fingerprint(&packing),
            [
                79, 224, 242, 67, 178, 245, 55, 135, 65, 197, 201, 217, 138, 252, 160, 116, 6, 208,
                97, 54, 48, 103, 113, 124, 92, 111, 34, 166, 132, 56, 241, 157,
            ]
        );
        assert_eq!(
            challenge_fingerprint(&rejected_d64),
            [
                211, 56, 140, 154, 235, 102, 212, 222, 171, 51, 243, 146, 217, 134, 16, 136, 179,
                148, 119, 92, 243, 228, 174, 3, 210, 173, 246, 7, 37, 194, 53, 62,
            ]
        );
        assert_eq!(
            challenge_fingerprint(&rejected_d128),
            [
                58, 50, 193, 231, 187, 62, 62, 138, 58, 0, 63, 68, 192, 83, 34, 15, 190, 215, 248,
                28, 11, 250, 234, 69, 227, 254, 209, 7, 129, 128, 179, 116,
            ]
        );
    }
}
