//! Fold-challenge preview drawing for prover-side Fiat–Shamir grinding.

use crate::sampler::{
    sparse_challenge_absorb_buf, sparse_challenges_from_seed, sparse_challenges_from_xof_cursor,
    validate_sparse_challenge_draw, XofCursor,
};
use crate::{
    tensor_left_digest, tensor_split, ChallengeLabels, ChallengeShape, Challenges, SparseChallenge,
    SparseChallengeConfig, TensorChallenges,
};
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::{FoldChallengeSeedPreview, Transcript};

fn preview_sparse_challenges(
    preview: &dyn FoldChallengeSeedPreview,
    label: &[u8],
    ring_d: usize,
    n: usize,
    cfg: &SparseChallengeConfig,
    grind_nonce: u32,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    let absorb_buf = sparse_challenge_absorb_buf(label, n as u64, ring_d, cfg, grind_nonce);
    let seed = preview.preview_challenge_bytes_after_absorb(&absorb_buf, 32);
    sparse_challenges_from_seed(ring_d, &seed, n, cfg)
}

fn derive_live_sparse_seed<F, T>(transcript: &mut T, absorb_buf: &[u8]) -> Vec<u8>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, absorb_buf);
    transcript.challenge_bytes(CHALLENGE_SPARSE_CHALLENGE, 32)
}

fn sparse_challenges_from_live_seed<F, T>(
    transcript: &mut T,
    label: &[u8],
    ring_d: usize,
    n: usize,
    cfg: &SparseChallengeConfig,
    grind_nonce: u32,
) -> Result<Vec<SparseChallenge>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let absorb_buf = sparse_challenge_absorb_buf(label, n as u64, ring_d, cfg, grind_nonce);
    let seed = derive_live_sparse_seed::<F, T>(transcript, &absorb_buf);
    let mut cursor = XofCursor::from_seed(&seed);
    sparse_challenges_from_xof_cursor(&mut cursor, ring_d, n, cfg)
}

/// Preview folding challenges for grind probing without advancing the transcript.
///
/// # Errors
///
/// Returns an error if count arithmetic overflows, tensor splitting is invalid,
/// or sparse challenge expansion fails.
#[allow(clippy::too_many_arguments)]
pub fn preview_folding_challenges(
    preview: &dyn FoldChallengeSeedPreview,
    ring_d: usize,
    num_blocks: usize,
    num_claims: usize,
    cfg: &SparseChallengeConfig,
    shape: &ChallengeShape,
    labels: ChallengeLabels<'_>,
    grind_nonce: u32,
) -> Result<Challenges, AkitaError> {
    validate_sparse_challenge_draw(ring_d, cfg)?;
    match shape {
        ChallengeShape::Flat => {
            let total = num_blocks.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("sparse challenge count overflow".to_string())
            })?;
            let challenges =
                preview_sparse_challenges(preview, labels.flat, ring_d, total, cfg, grind_nonce)?;
            Challenges::from_sparse(challenges, num_blocks, num_claims)
        }
        ChallengeShape::Tensor => {
            let (left_len, right_len) = tensor_split(num_blocks)?;
            let left_total = left_len.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("tensor-left challenge count overflow".to_string())
            })?;
            let right_total = right_len.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("tensor-right challenge count overflow".to_string())
            })?;
            let left_buf = sparse_challenge_absorb_buf(
                labels.tensor_left,
                left_total as u64,
                ring_d,
                cfg,
                grind_nonce,
            );
            let left = preview_sparse_challenges(
                preview,
                labels.tensor_left,
                ring_d,
                left_total,
                cfg,
                grind_nonce,
            )?;
            let left_digest = tensor_left_digest(&left, left_len, num_claims, ring_d)?;
            let right_buf = sparse_challenge_absorb_buf(
                labels.tensor_right,
                right_total as u64,
                ring_d,
                cfg,
                grind_nonce,
            );
            let right_seed = preview.preview_challenge_bytes_after_absorb_chain(
                &[&left_buf, &left_digest, &right_buf],
                &[32, 0, 32],
            );
            let right = sparse_challenges_from_seed(ring_d, &right_seed, right_total, cfg)?;
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

/// Sample folding challenges using the configured shape (live transcript advance).
///
/// # Errors
///
/// Returns an error if count arithmetic overflows, if tensor splitting is
/// invalid, or if sparse challenge sampling fails.
#[allow(clippy::too_many_arguments)]
pub fn sample_folding_challenges<F, T>(
    transcript: &mut T,
    ring_d: usize,
    num_blocks: usize,
    num_claims: usize,
    cfg: &SparseChallengeConfig,
    shape: &ChallengeShape,
    labels: ChallengeLabels<'_>,
    grind_nonce: u32,
) -> Result<Challenges, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    validate_sparse_challenge_draw(ring_d, cfg)?;
    match shape {
        ChallengeShape::Flat => {
            let total = num_blocks.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("sparse challenge count overflow".to_string())
            })?;
            let challenges = sparse_challenges_from_live_seed::<F, T>(
                transcript,
                labels.flat,
                ring_d,
                total,
                cfg,
                grind_nonce,
            )?;
            Challenges::from_sparse(challenges, num_blocks, num_claims)
        }
        ChallengeShape::Tensor => {
            let (left_len, right_len) = tensor_split(num_blocks)?;
            let left_total = left_len.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("tensor-left challenge count overflow".to_string())
            })?;
            let right_total = right_len.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("tensor-right challenge count overflow".to_string())
            })?;
            let left = sparse_challenges_from_live_seed::<F, T>(
                transcript,
                labels.tensor_left,
                ring_d,
                left_total,
                cfg,
                grind_nonce,
            )?;
            let left_digest = tensor_left_digest(&left, left_len, num_claims, ring_d)?;
            transcript.append_bytes(labels.tensor_left_digest, &left_digest);
            let right = sparse_challenges_from_live_seed::<F, T>(
                transcript,
                labels.tensor_right,
                ring_d,
                right_total,
                cfg,
                grind_nonce,
            )?;
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
