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

fn preview_sparse_challenges<const D: usize>(
    preview: &dyn FoldChallengeSeedPreview,
    label: &[u8],
    n: usize,
    cfg: &SparseChallengeConfig,
    grind_nonce: u32,
    op_norm_rejection: bool,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    let absorb_buf = sparse_challenge_absorb_buf::<D>(label, n as u64, cfg, grind_nonce);
    let seed = preview.preview_challenge_bytes_after_absorb(&absorb_buf, 32);
    sparse_challenges_from_seed::<D>(&seed, n, cfg, op_norm_rejection)
}

fn derive_live_sparse_seed<F, T>(transcript: &mut T, absorb_buf: &[u8]) -> Vec<u8>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, absorb_buf);
    transcript.challenge_bytes(CHALLENGE_SPARSE_CHALLENGE, 32)
}

fn sparse_challenges_from_live_seed<F, T, const D: usize>(
    transcript: &mut T,
    label: &[u8],
    n: usize,
    cfg: &SparseChallengeConfig,
    grind_nonce: u32,
    op_norm_rejection: bool,
) -> Result<Vec<SparseChallenge>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let absorb_buf = sparse_challenge_absorb_buf::<D>(label, n as u64, cfg, grind_nonce);
    let seed = derive_live_sparse_seed::<F, T>(transcript, &absorb_buf);
    let mut cursor = XofCursor::from_seed(&seed);
    sparse_challenges_from_xof_cursor::<D>(&mut cursor, n, cfg, op_norm_rejection)
}

/// Preview folding challenges for grind probing without advancing the transcript.
///
/// `op_norm_rejection` is the per-level layout decision. When false, exact-shell
/// factors are sampled from the full shell even if their configured threshold is
/// binding; downstream sizing must then use the L1 mass, not the Gamma cap.
///
/// # Errors
///
/// Returns an error if count arithmetic overflows, tensor splitting is invalid,
/// or sparse challenge expansion fails.
#[allow(clippy::too_many_arguments)]
pub fn preview_folding_challenges<const D: usize>(
    preview: &dyn FoldChallengeSeedPreview,
    num_blocks: usize,
    num_claims: usize,
    cfg: &SparseChallengeConfig,
    shape: &ChallengeShape,
    labels: ChallengeLabels<'_>,
    grind_nonce: u32,
    op_norm_rejection: bool,
) -> Result<Challenges, AkitaError> {
    validate_sparse_challenge_draw::<D>(cfg)?;
    match shape {
        ChallengeShape::Flat => {
            let total = num_blocks.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("sparse challenge count overflow".to_string())
            })?;
            let challenges = preview_sparse_challenges::<D>(
                preview,
                labels.flat,
                total,
                cfg,
                grind_nonce,
                op_norm_rejection,
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
            let left_buf = sparse_challenge_absorb_buf::<D>(
                labels.tensor_left,
                left_total as u64,
                cfg,
                grind_nonce,
            );
            let left = preview_sparse_challenges::<D>(
                preview,
                labels.tensor_left,
                left_total,
                cfg,
                grind_nonce,
                op_norm_rejection,
            )?;
            let left_digest = tensor_left_digest::<D>(&left, left_len, num_claims)?;
            let right_buf = sparse_challenge_absorb_buf::<D>(
                labels.tensor_right,
                right_total as u64,
                cfg,
                grind_nonce,
            );
            let right_seed = preview.preview_challenge_bytes_after_absorb_chain(
                &[&left_buf, &left_digest, &right_buf],
                &[32, 0, 32],
            );
            let right =
                sparse_challenges_from_seed::<D>(&right_seed, right_total, cfg, op_norm_rejection)?;
            Challenges::from_tensor::<D>(TensorChallenges {
                left,
                right,
                left_len,
                right_len,
                num_claims,
            })
        }
    }
}

/// Sample folding challenges using the configured shape (live transcript advance).
///
/// `op_norm_rejection` is the per-level layout decision. It must match the
/// pricing used by the level parameters so prover and verifier sample the same
/// challenge support that the SIS bounds assume.
///
/// # Errors
///
/// Returns an error if count arithmetic overflows, if tensor splitting is
/// invalid, or if sparse challenge sampling fails.
#[allow(clippy::too_many_arguments)]
pub fn sample_folding_challenges<F, T, const D: usize>(
    transcript: &mut T,
    num_blocks: usize,
    num_claims: usize,
    cfg: &SparseChallengeConfig,
    shape: &ChallengeShape,
    labels: ChallengeLabels<'_>,
    grind_nonce: u32,
    op_norm_rejection: bool,
) -> Result<Challenges, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    validate_sparse_challenge_draw::<D>(cfg)?;
    match shape {
        ChallengeShape::Flat => {
            let total = num_blocks.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("sparse challenge count overflow".to_string())
            })?;
            let challenges = sparse_challenges_from_live_seed::<F, T, D>(
                transcript,
                labels.flat,
                total,
                cfg,
                grind_nonce,
                op_norm_rejection,
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
            let left = sparse_challenges_from_live_seed::<F, T, D>(
                transcript,
                labels.tensor_left,
                left_total,
                cfg,
                grind_nonce,
                op_norm_rejection,
            )?;
            let left_digest = tensor_left_digest::<D>(&left, left_len, num_claims)?;
            transcript.append_bytes(labels.tensor_left_digest, &left_digest);
            let right = sparse_challenges_from_live_seed::<F, T, D>(
                transcript,
                labels.tensor_right,
                right_total,
                cfg,
                grind_nonce,
                op_norm_rejection,
            )?;
            Challenges::from_tensor::<D>(TensorChallenges {
                left,
                right,
                left_len,
                right_len,
                num_claims,
            })
        }
    }
}
