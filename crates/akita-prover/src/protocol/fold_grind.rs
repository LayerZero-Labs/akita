//! Fold-l∞ Fiat–Shamir grind: preview off-sponge clones, commit the winning nonce.

use crate::{AkitaPolyOps, DecomposeFoldWitness};
use akita_challenges::Challenges;
use akita_challenges::{
    preview_folding_challenges, sample_folding_challenges, stage1_fold_challenge_labels,
};
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::{AkitaTranscript, FoldChallengeSeedPreview, Transcript, TranscriptSponge};
use akita_types::{
    sis::{FoldLinfGrindContract, FoldLinfThresholdPolicy},
    FoldLinfProtocolBinding, LevelParams,
};

use super::ring_relation::build_point_decompose_fold_witness;

/// Preview-only transcript access for prover-side fold grinding.
///
/// Implemented only for production prover transcripts; grinding stays confined
/// to this module instead of infecting the public [`Transcript`] trait surface.
pub trait ProverTranscriptGrind<F>: Transcript<F> + FoldChallengeSeedPreview
where
    F: FieldCore + CanonicalField,
{
}

impl<F> ProverTranscriptGrind<F> for AkitaTranscript<F, TranscriptSponge> where
    F: FieldCore + CanonicalField + akita_field::CanonicalBytes + akita_field::TranscriptChallenge
{
}

#[cfg(feature = "logging-transcript")]
impl<F, T> ProverTranscriptGrind<F> for akita_transcript::LoggingTranscript<T>
where
    F: FieldCore + CanonicalField,
    T: ProverTranscriptGrind<F>,
{
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
        let challenges = preview_folding_challenges::<D>(
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
