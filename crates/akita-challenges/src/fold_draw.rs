//! Fold-challenge preview drawing for prover-side Fiat–Shamir grinding.

use crate::sampler::{SignedSparseScratch, XofCursor, MAX_STACK_RING_DIM};
use crate::{
    fold_high_digest, ChallengeLabels, ChallengeShape, Challenges, SparseChallenge,
    SparseChallengeConfig, TensorChallenges,
};
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::{FoldChallengeSeedPreview, Transcript};
use std::marker::PhantomData;

const SPARSE_CHALLENGE_SEED_LEN: usize = 32;
const FOLD_CHALLENGE_ROUND_DOMAIN: &[u8] = b"akita/fold-challenge-round/v1";

/// Build the canonical transcript prefix for one group-local fold draw.
///
/// The prefix binds the group index, exact `num_live_blocks`, claim count, and
/// tensor shape before the sparse challenge seed is squeezed.
///
/// # Errors
///
/// Returns an error if a platform-sized count does not fit the canonical u64
/// encoding.
pub fn fold_challenge_sample_label(
    base_label: &[u8],
    group_index: usize,
    num_live_blocks: usize,
    num_claims: usize,
    shape: ChallengeShape,
) -> Result<Vec<u8>, AkitaError> {
    let group_index = u64::try_from(group_index)
        .map_err(|_| AkitaError::InvalidSetup("fold group index exceeds u64".to_string()))?;
    let num_live_blocks = u64::try_from(num_live_blocks)
        .map_err(|_| AkitaError::InvalidSetup("num_live_blocks exceeds u64".to_string()))?;
    let num_claims = u64::try_from(num_claims)
        .map_err(|_| AkitaError::InvalidSetup("fold claim count exceeds u64".to_string()))?;
    let mut label = Vec::with_capacity(FOLD_CHALLENGE_ROUND_DOMAIN.len() + base_label.len() + 33);
    label.extend_from_slice(FOLD_CHALLENGE_ROUND_DOMAIN);
    label.extend_from_slice(&group_index.to_le_bytes());
    label.extend_from_slice(&num_live_blocks.to_le_bytes());
    label.extend_from_slice(&num_claims.to_le_bytes());
    match shape {
        ChallengeShape::Flat => {
            label.push(0);
            label.extend_from_slice(&0u64.to_le_bytes());
        }
        ChallengeShape::Tensor { fold_low_len } => {
            let fold_low_len = u64::try_from(fold_low_len).map_err(|_| {
                AkitaError::InvalidSetup("tensor fold-low length exceeds u64".to_string())
            })?;
            label.push(1);
            label.extend_from_slice(&fold_low_len.to_le_bytes());
        }
    }
    label.extend_from_slice(base_label);
    Ok(label)
}

pub trait FoldDraw {
    fn absorb(&mut self, label: &[u8], payload: &[u8]);

    fn absorb_and_squeeze(&mut self, label: &[u8], payload: &[u8]) -> Vec<u8>;

    fn draw_sparse_challenges(
        &mut self,
        label: &[u8],
        ring_d: usize,
        n: usize,
        cfg: &SparseChallengeConfig,
        grind_nonce: u32,
    ) -> Vec<SparseChallenge> {
        let domain_sep = cfg.domain_separator_bytes();
        let mut absorb_buf = Vec::with_capacity(label.len() + 8 + 8 + domain_sep.len() + 4);
        absorb_buf.extend_from_slice(label);
        absorb_buf.extend_from_slice(&(n as u64).to_le_bytes());
        absorb_buf.extend_from_slice(&(ring_d as u64).to_le_bytes());
        absorb_buf.extend_from_slice(&domain_sep);
        absorb_buf.extend_from_slice(&grind_nonce.to_le_bytes());

        let seed = self.absorb_and_squeeze(ABSORB_SPARSE_CHALLENGE, &absorb_buf);
        let mut cursor = XofCursor::from_seed(&seed);
        SignedSparseScratch::sample_challenges(&mut cursor, ring_d, n, cfg)
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_folding_challenges(
        &mut self,
        ring_d: usize,
        group_index: usize,
        num_live_blocks: usize,
        num_claims: usize,
        cfg: &SparseChallengeConfig,
        shape: &ChallengeShape,
        labels: ChallengeLabels<'_>,
        grind_nonce: u32,
    ) -> Result<Challenges, AkitaError> {
        if ring_d > MAX_STACK_RING_DIM {
            return Err(AkitaError::InvalidInput(format!(
                "ring dimension {ring_d} exceeds supported stack sampler limit ({MAX_STACK_RING_DIM})"
            )));
        }
        cfg.validate_dyn(ring_d).map_err(|e| {
            AkitaError::InvalidInput(format!("invalid sparse challenge config: {e}"))
        })?;
        if num_live_blocks == 0 || num_claims == 0 {
            return Err(AkitaError::InvalidInput(
                "fold challenges require positive num_live_blocks and claims".to_string(),
            ));
        }

        match shape {
            ChallengeShape::Flat => {
                let total = challenge_count(num_live_blocks, num_claims, "sparse")?;
                let sample_label = fold_challenge_sample_label(
                    labels.flat,
                    group_index,
                    num_live_blocks,
                    num_claims,
                    *shape,
                )?;
                let challenges =
                    self.draw_sparse_challenges(&sample_label, ring_d, total, cfg, grind_nonce);
                Challenges::from_sparse(challenges, num_live_blocks, num_claims)
            }
            ChallengeShape::Tensor { fold_low_len } => {
                if num_live_blocks == 0 || !fold_low_len.is_power_of_two() {
                    return Err(AkitaError::InvalidInput(
                        "tensor challenges require positive num_live_blocks and a power-of-two low length"
                            .to_string(),
                    ));
                }
                let fold_high_len = num_live_blocks.div_ceil(*fold_low_len);
                let high_total = challenge_count(fold_high_len, num_claims, "fold-high")?;
                let low_total = challenge_count(*fold_low_len, num_claims, "fold-low")?;
                let high_label = fold_challenge_sample_label(
                    labels.fold_high,
                    group_index,
                    num_live_blocks,
                    num_claims,
                    *shape,
                )?;
                let fold_high =
                    self.draw_sparse_challenges(&high_label, ring_d, high_total, cfg, grind_nonce);
                let high_digest = fold_high_digest(&fold_high, fold_high_len, num_claims, ring_d)?;
                self.absorb(labels.fold_high_digest, &high_digest);
                let low_label = fold_challenge_sample_label(
                    labels.fold_low,
                    group_index,
                    num_live_blocks,
                    num_claims,
                    *shape,
                )?;
                let fold_low =
                    self.draw_sparse_challenges(&low_label, ring_d, low_total, cfg, grind_nonce);
                Challenges::from_tensor_dyn(
                    TensorChallenges {
                        fold_high,
                        fold_low,
                        num_live_blocks_per_claim: num_live_blocks,
                        fold_low_len: *fold_low_len,
                        num_claims,
                    },
                    ring_d,
                )
            }
        }
    }
}

pub struct PreviewFoldDraw<'a> {
    preview: &'a dyn FoldChallengeSeedPreview,
    absorbs: Vec<Vec<u8>>,
    squeeze_lens: Vec<usize>,
}

impl<'a> PreviewFoldDraw<'a> {
    pub fn new(preview: &'a dyn FoldChallengeSeedPreview) -> Self {
        Self {
            preview,
            absorbs: Vec::new(),
            squeeze_lens: Vec::new(),
        }
    }
}

impl FoldDraw for PreviewFoldDraw<'_> {
    fn absorb(&mut self, _label: &[u8], payload: &[u8]) {
        self.absorbs.push(payload.to_vec());
        self.squeeze_lens.push(0);
    }

    fn absorb_and_squeeze(&mut self, _label: &[u8], payload: &[u8]) -> Vec<u8> {
        self.absorbs.push(payload.to_vec());
        self.squeeze_lens.push(SPARSE_CHALLENGE_SEED_LEN);
        let absorbs = self.absorbs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        self.preview
            .preview_challenge_bytes_after_absorb_chain(&absorbs, &self.squeeze_lens)
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
    fn absorb(&mut self, label: &[u8], payload: &[u8]) {
        self.transcript.append_bytes(label, payload);
    }

    fn absorb_and_squeeze(&mut self, label: &[u8], payload: &[u8]) -> Vec<u8> {
        self.absorb(label, payload);
        self.transcript
            .challenge_bytes(CHALLENGE_SPARSE_CHALLENGE, SPARSE_CHALLENGE_SEED_LEN)
    }
}

fn challenge_count(lhs: usize, rhs: usize, label: &str) -> Result<usize, AkitaError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{label} challenge count overflow")))
}
