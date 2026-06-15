//! Fold-l∞ Fiat–Shamir grind: preview off-sponge clones, commit the winning nonce.

use crate::{AkitaPolyOps, DecomposeFoldWitness};
use akita_challenges::{
    sample_folding_challenges, sparse_challenge_absorb_buf, sparse_challenges_from_seed,
    stage1_fold_challenge_labels, tensor_left_digest, tensor_split, ChallengeLabels,
    ChallengeShape, Challenges, SparseChallengeConfig,
};
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::{AkitaTranscript, Transcript, TranscriptSponge};
use akita_types::{
    sis::{FoldLinfGrindContract, FoldLinfThresholdPolicy},
    FoldLinfProtocolBinding, LevelParams,
};

use super::ring_relation::build_point_decompose_fold_witness;

/// Preview-only transcript access for prover-side fold grinding.
///
/// Implemented only for production prover transcripts; grinding stays confined
/// to this module instead of infecting the public [`Transcript`] trait surface.
pub trait ProverTranscriptGrind<F>: Transcript<F>
where
    F: FieldCore + CanonicalField,
{
    /// Derive challenge bytes after a hypothetical absorb, without mutating replay state.
    fn preview_challenge_bytes_after_absorb(&self, absorb_payload: &[u8], len: usize) -> Vec<u8>;

    /// Derive challenge bytes after a hypothetical absorb/squeeze chain on a sponge clone.
    fn preview_challenge_bytes_after_absorb_chain(
        &self,
        absorbs: &[&[u8]],
        squeeze_lens: &[usize],
    ) -> Vec<u8>;
}

impl<F> ProverTranscriptGrind<F> for AkitaTranscript<F, TranscriptSponge>
where
    F: FieldCore + CanonicalField + akita_field::CanonicalBytes + akita_field::TranscriptChallenge,
{
    fn preview_challenge_bytes_after_absorb(&self, absorb_payload: &[u8], len: usize) -> Vec<u8> {
        AkitaTranscript::preview_challenge_bytes_after_absorb(self, absorb_payload, len)
    }

    fn preview_challenge_bytes_after_absorb_chain(
        &self,
        absorbs: &[&[u8]],
        squeeze_lens: &[usize],
    ) -> Vec<u8> {
        AkitaTranscript::preview_challenge_bytes_after_absorb_chain(self, absorbs, squeeze_lens)
    }
}

#[cfg(feature = "logging-transcript")]
impl<F, T> ProverTranscriptGrind<F> for akita_transcript::LoggingTranscript<T>
where
    F: FieldCore + CanonicalField,
    T: ProverTranscriptGrind<F>,
{
    fn preview_challenge_bytes_after_absorb(&self, absorb_payload: &[u8], len: usize) -> Vec<u8> {
        self.inner
            .preview_challenge_bytes_after_absorb(absorb_payload, len)
    }

    fn preview_challenge_bytes_after_absorb_chain(
        &self,
        absorbs: &[&[u8]],
        squeeze_lens: &[usize],
    ) -> Vec<u8> {
        self.inner
            .preview_challenge_bytes_after_absorb_chain(absorbs, squeeze_lens)
    }
}

fn preview_sparse_challenges<F, T, const D: usize>(
    transcript: &T,
    label: &[u8],
    n: usize,
    cfg: &SparseChallengeConfig,
    grind_nonce: u32,
) -> Result<Vec<akita_challenges::SparseChallenge>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: ProverTranscriptGrind<F>,
{
    cfg.validate::<D>()
        .map_err(|e| AkitaError::InvalidInput(format!("invalid sparse challenge config: {e}")))?;
    let absorb_buf = sparse_challenge_absorb_buf::<D>(label, n as u64, cfg, grind_nonce);
    let seed = transcript.preview_challenge_bytes_after_absorb(&absorb_buf, 32);
    sparse_challenges_from_seed::<D>(&seed, n, cfg)
}

fn preview_folding_challenges<F, T, const D: usize>(
    transcript: &T,
    num_blocks: usize,
    num_claims: usize,
    cfg: &SparseChallengeConfig,
    shape: &ChallengeShape,
    labels: ChallengeLabels<'_>,
    grind_nonce: u32,
) -> Result<Challenges, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: ProverTranscriptGrind<F>,
{
    match shape {
        ChallengeShape::Flat => {
            let total = num_blocks.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("sparse challenge count overflow".to_string())
            })?;
            let challenges = preview_sparse_challenges::<F, T, D>(
                transcript,
                labels.flat,
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
            let left_buf = sparse_challenge_absorb_buf::<D>(
                labels.tensor_left,
                left_total as u64,
                cfg,
                grind_nonce,
            );
            let left = preview_sparse_challenges::<F, T, D>(
                transcript,
                labels.tensor_left,
                left_total,
                cfg,
                grind_nonce,
            )?;
            let left_digest = tensor_left_digest::<D>(&left, left_len, num_claims)?;
            let right_buf = sparse_challenge_absorb_buf::<D>(
                labels.tensor_right,
                right_total as u64,
                cfg,
                grind_nonce,
            );
            let right_seed = transcript.preview_challenge_bytes_after_absorb_chain(
                &[&left_buf, &left_digest, &right_buf],
                &[32, 0, 32],
            );
            let right = sparse_challenges_from_seed::<D>(&right_seed, right_total, cfg)?;
            Challenges::from_tensor::<D>(akita_challenges::TensorChallenges {
                left,
                right,
                left_len,
                right_len,
                num_claims,
            })
        }
    }
}

fn accepts_witness(contract: &FoldLinfGrindContract, centered_inf_norm: u32) -> bool {
    contract.policy == FoldLinfThresholdPolicy::DeterministicBetaInf
        || u128::from(centered_inf_norm) <= contract.inf_threshold
}

/// Probe fold challenges off-sponge, accept the first witness under `t*`, then commit.
pub(crate) fn sample_fold_decompose_witness<F, P, T, const D: usize>(
    transcript: &mut T,
    polys: &[&P],
    lp: &LevelParams,
    num_claims: usize,
) -> Result<(DecomposeFoldWitness<F, D>, Challenges, u32), AkitaError>
where
    F: FieldCore + CanonicalField,
    P: AkitaPolyOps<F, D>,
    T: Transcript<F> + ProverTranscriptGrind<F>,
{
    let contract = lp.fold_linf_grind_contract(
        num_claims,
        FoldLinfProtocolBinding::CURRENT.max_grind_attempts,
    )?;
    let point_indices = (0..polys.len()).collect::<Vec<_>>();
    let labels = stage1_fold_challenge_labels();

    for nonce in 0..contract.max_nonce_exclusive {
        let challenges = preview_folding_challenges::<F, T, D>(
            transcript,
            lp.num_blocks,
            num_claims,
            &lp.stage1_config,
            &lp.fold_challenge_shape,
            labels,
            nonce,
        )?;
        let witness =
            build_point_decompose_fold_witness::<F, P, D>(&challenges, polys, &point_indices, lp)?;
        if accepts_witness(&contract, witness.centered_inf_norm) {
            let challenges = sample_folding_challenges::<F, T, D>(
                transcript,
                lp.num_blocks,
                num_claims,
                &lp.stage1_config,
                &lp.fold_challenge_shape,
                labels,
                nonce,
            )?;
            return Ok((witness, challenges, nonce));
        }
    }

    Err(AkitaError::InvalidInput(format!(
        "fold grind exceeded {} attempts (threshold={})",
        contract.max_nonce_exclusive, contract.inf_threshold
    )))
}
