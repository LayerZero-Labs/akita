//! Fold-challenge preview drawing for prover-side Fiat–Shamir grinding.

use crate::sampler::MAX_STACK_RING_DIM;
use crate::{Challenges, OperatorNormRejection, SparseChallengeConfig};
use akita_error::AkitaError;
use akita_transcript::FOLD_CHALLENGE_SEED_LEN;

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
    fn absorb_and_squeeze(&mut self, payload: &[u8]) -> [u8; FOLD_CHALLENGE_SEED_LEN];

    #[cfg(feature = "logging-transcript")]
    fn record_challenge_range(&mut self, _group_index: usize, _coordinate_count: usize) {}

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
        let seed = self.absorb_and_squeeze(&absorb_buf);
        let challenges = crate::sampler::sample_indexed_challenges_from_seed(
            &seed, ring_d, total, cfg, rejection,
        )?;
        #[cfg(feature = "logging-transcript")]
        self.record_challenge_range(group_index, total);
        Challenges::from_sparse(challenges, num_live_blocks, num_claims)
    }
}

fn native_fold_record(
    level: u32,
    group: u32,
    payload_len: usize,
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
        FOLD_CHALLENGE_SEED_LEN as u64,
    )
}

/// One group-local fold-root draw against a candidate native public state.
pub struct NativePreviewFoldDraw<'a> {
    preview: &'a mut akita_transcript::NativeFoldPreview,
}

impl<'a> NativePreviewFoldDraw<'a> {
    /// Bind this short-lived draw adapter to one schedule-derived fold group.
    #[must_use]
    pub const fn new(preview: &'a mut akita_transcript::NativeFoldPreview) -> Self {
        Self { preview }
    }
}

impl FoldDraw for NativePreviewFoldDraw<'_> {
    fn absorb_and_squeeze(&mut self, payload: &[u8]) -> [u8; FOLD_CHALLENGE_SEED_LEN] {
        self.preview.fold_root(payload)
    }
}

/// One group-local fold-root draw against the live native prover state.
pub struct NativeProverFoldDraw<'a> {
    state: &'a mut akita_transcript::NativeProverState,
    level: u32,
    group: u32,
}

impl<'a> NativeProverFoldDraw<'a> {
    /// Bind this short-lived draw adapter to one schedule-derived fold group.
    #[must_use]
    pub const fn new(
        state: &'a mut akita_transcript::NativeProverState,
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

impl FoldDraw for NativeProverFoldDraw<'_> {
    fn absorb_and_squeeze(&mut self, payload: &[u8]) -> [u8; FOLD_CHALLENGE_SEED_LEN] {
        akita_transcript::native_prover_fold_root(
            self.state,
            native_fold_record(self.level, self.group, payload.len()),
            payload,
        )
    }
}

/// One group-local fold-root draw against the live native verifier state.
pub struct NativeVerifierFoldDraw<'a, 'proof> {
    state: &'a mut akita_transcript::NativeVerifierState<'proof>,
    level: u32,
    group: u32,
}

impl<'a, 'proof> NativeVerifierFoldDraw<'a, 'proof> {
    /// Bind this short-lived draw adapter to one schedule-derived fold group.
    #[must_use]
    pub const fn new(
        state: &'a mut akita_transcript::NativeVerifierState<'proof>,
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

impl FoldDraw for NativeVerifierFoldDraw<'_, '_> {
    fn absorb_and_squeeze(&mut self, payload: &[u8]) -> [u8; FOLD_CHALLENGE_SEED_LEN] {
        akita_transcript::native_verifier_fold_root(
            self.state,
            native_fold_record(self.level, self.group, payload.len()),
            payload,
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
        fn absorb_and_squeeze(&mut self, payload: &[u8]) -> [u8; FOLD_CHALLENGE_SEED_LEN] {
            self.payloads.push(payload.to_vec());
            [7; FOLD_CHALLENGE_SEED_LEN]
        }
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
}
