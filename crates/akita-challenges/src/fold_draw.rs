//! Fold-challenge preview drawing for prover-side Fiat–Shamir grinding.

use crate::sampler::{SignedSparseScratch, XofCursor, MAX_STACK_RING_DIM};
use crate::{
    tensor_left_digest, tensor_split, ChallengeLabels, ChallengeShape, Challenges, SparseChallenge,
    SparseChallengeConfig, TensorChallenges,
};
use akita_error::AkitaError;
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::{FoldChallengeSeedPreview, Transcript};
use jolt_field::{CanonicalField, FieldCore};
use std::marker::PhantomData;

const SPARSE_CHALLENGE_SEED_LEN: usize = 32;

pub trait FoldDraw {
    fn absorb(&mut self, payload: &[u8]);

    fn absorb_and_squeeze(&mut self, payload: &[u8]) -> Vec<u8>;

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

        let seed = self.absorb_and_squeeze(&absorb_buf);
        let mut cursor = XofCursor::from_seed(&seed);
        SignedSparseScratch::sample_challenges(&mut cursor, ring_d, n, cfg)
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_folding_challenges(
        &mut self,
        ring_d: usize,
        num_blocks: usize,
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

        match shape {
            ChallengeShape::Flat => {
                let total = challenge_count(num_blocks, num_claims, "sparse")?;
                let challenges =
                    self.draw_sparse_challenges(labels.flat, ring_d, total, cfg, grind_nonce);
                Challenges::from_sparse(challenges, num_blocks, num_claims)
            }
            ChallengeShape::Tensor => {
                let (left_len, right_len) = tensor_split(num_blocks)?;
                let left_total = challenge_count(left_len, num_claims, "tensor-left")?;
                let right_total = challenge_count(right_len, num_claims, "tensor-right")?;
                let left = self.draw_sparse_challenges(
                    labels.tensor_left,
                    ring_d,
                    left_total,
                    cfg,
                    grind_nonce,
                );
                let left_digest = tensor_left_digest(&left, left_len, num_claims, ring_d)?;
                self.absorb(&left_digest);
                let right = self.draw_sparse_challenges(
                    labels.tensor_right,
                    ring_d,
                    right_total,
                    cfg,
                    grind_nonce,
                );
                Challenges::from_tensor_dyn(
                    TensorChallenges {
                        left,
                        right,
                        left_len,
                        right_len,
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
    fn absorb(&mut self, payload: &[u8]) {
        self.absorbs.push(payload.to_vec());
        self.squeeze_lens.push(0);
    }

    fn absorb_and_squeeze(&mut self, payload: &[u8]) -> Vec<u8> {
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
    fn absorb(&mut self, payload: &[u8]) {
        self.transcript
            .append_bytes(ABSORB_SPARSE_CHALLENGE, payload);
    }

    fn absorb_and_squeeze(&mut self, payload: &[u8]) -> Vec<u8> {
        self.absorb(payload);
        self.transcript
            .challenge_bytes(CHALLENGE_SPARSE_CHALLENGE, SPARSE_CHALLENGE_SEED_LEN)
    }
}

fn challenge_count(lhs: usize, rhs: usize, label: &str) -> Result<usize, AkitaError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{label} challenge count overflow")))
}
