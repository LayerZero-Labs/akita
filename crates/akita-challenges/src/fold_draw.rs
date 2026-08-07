//! Fold-challenge preview drawing for prover-side Fiat–Shamir grinding.

use crate::sampler::{SignedSparseScratch, XofCursor, MAX_STACK_RING_DIM};
use crate::{Challenges, SparseChallengeConfig};
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::{FoldChallengeSeedPreview, Transcript, FOLD_CHALLENGE_SEED_LEN};
use std::marker::PhantomData;

const FOLD_CHALLENGE_ROUND_DOMAIN: &[u8] = b"akita/fold-challenge-round/v1";

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
        let seed = self.absorb_and_squeeze(ABSORB_SPARSE_CHALLENGE, &absorb_buf);
        let mut cursor = XofCursor::from_seed(&seed);
        let challenges = SignedSparseScratch::sample_challenges(&mut cursor, ring_d, total, cfg);
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
