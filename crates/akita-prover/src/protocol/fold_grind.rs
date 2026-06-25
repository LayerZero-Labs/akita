//! Fold-l∞ Fiat–Shamir grind: preview off-sponge clones, commit the winning nonce.

use crate::compute::{OpeningBatchKernel, OpeningFoldKernel, RootOpeningSource};
use crate::DecomposeFoldWitness;
use akita_challenges::{
    grind_probe_permutation, preview_folding_challenges, sample_folding_challenges,
    stage1_fold_challenge_labels, Challenges,
};
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::{AkitaTranscript, FoldChallengeSeedPreview, Transcript, TranscriptSponge};
use akita_types::{
    golomb_rice_rows_admit_terminal_wire,
    sis::{FoldWitnessGrindContract, FoldWitnessLinfCapPolicy},
    FoldLinfProtocolBinding, LevelParams, FOLD_GRIND_PROBE_ORDER_SEQUENTIAL_MIN,
    FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE,
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
    F: FieldCore + CanonicalField + akita_field::CanonicalBytes + akita_field::TranscriptChallenge,
    T: ProverTranscriptGrind<F>,
{
}

fn accepts_fold_witness<const D: usize>(
    contract: &FoldWitnessGrindContract,
    witness: &DecomposeFoldWitness<impl CanonicalField, D>,
    witness_linf_cap: u128,
    tail_t_vectors: Option<usize>,
) -> bool {
    if contract.policy != FoldWitnessLinfCapPolicy::WorstCaseBetaOnly
        && u128::from(witness.centered_inf_norm) > witness_linf_cap
    {
        return false;
    }
    if tail_t_vectors.is_some()
        && golomb_rice_rows_admit_terminal_wire(&witness.centered_coeffs, witness_linf_cap).is_err()
    {
        return false;
    }
    true
}

fn witness_linf_cap_for_grind(
    lp: &LevelParams,
    contract: &FoldWitnessGrindContract,
    tail_t_vectors: Option<usize>,
) -> Result<u128, AkitaError> {
    match tail_t_vectors {
        Some(num_t_vectors) => lp.fold_witness_linf_cap_for_claims(num_t_vectors),
        None => Ok(contract.witness_linf_cap),
    }
}

fn grind_probe_nonces(
    contract: &FoldWitnessGrindContract,
    binding: &FoldLinfProtocolBinding,
    transcript: &dyn FoldChallengeSeedPreview,
    lp: &LevelParams,
    num_claims: usize,
) -> Result<Vec<u32>, AkitaError> {
    let cap = contract.max_nonce_exclusive;
    match binding.grind_probe_order {
        FOLD_GRIND_PROBE_ORDER_SEQUENTIAL_MIN => Ok((0..cap).collect()),
        FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE
            if contract.policy == FoldWitnessLinfCapPolicy::TailBoundWithGrind =>
        {
            let absorb_buf = lp.fold_grind_probe_order_absorb_buf(num_claims);
            let seed = transcript.preview_challenge_bytes_after_absorb(&absorb_buf, 32);
            Ok(grind_probe_permutation(&seed, cap))
        }
        FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE => Ok(vec![0]),
        other => Err(AkitaError::InvalidSetup(format!(
            "unsupported fold grind probe order tag {other}"
        ))),
    }
}

/// Probe fold challenges off-sponge, accept the first witness under `t*`, then commit.
///
/// Plain presets probe `nonce = 0, 1, …` (minimum accepting nonce). ZK presets
/// with tail-bound grind use a transcript-seeded uniform permutation of the same
/// range; see `specs/fold-linf-rejection.md` (*ZK: grind probe order*).
///
/// When `tail_t_vectors` is set, presets reject witnesses whose centered coefficients do not
/// fit the terminal Golomb planner budget at wire low bits (including `WorstCaseBetaOnly`
/// presets that do not reroll on linf cap).
pub(crate) fn sample_fold_decompose_witness<F, P, B, T, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup<D>>,
    transcript: &mut T,
    polys: &[&P],
    lp: &LevelParams,
    num_claims: usize,
    tail_t_vectors: Option<usize>,
) -> Result<(DecomposeFoldWitness<F, D>, Challenges, u32), AkitaError>
where
    F: FieldCore + CanonicalField,
    P: RootOpeningSource<F, D>,
    B: crate::compute::ComputeBackendSetup<F>
        + for<'a> OpeningBatchKernel<P::OpeningBatchView<'a>, F, D>
        + for<'a> OpeningFoldKernel<P::OpeningView<'a>, F, D>,
    T: Transcript<F> + ProverTranscriptGrind<F>,
{
    let binding = FoldLinfProtocolBinding::CURRENT;
    let contract = lp.fold_witness_grind_contract(num_claims, binding.max_grind_attempts)?;
    let witness_linf_cap = witness_linf_cap_for_grind(lp, &contract, tail_t_vectors)?;
    let point_indices = (0..polys.len()).collect::<Vec<_>>();
    let labels = stage1_fold_challenge_labels();
    let probe_nonces = grind_probe_nonces(&contract, &binding, transcript, lp, num_claims)?;

    let mut grind_probe_count = 0u32;
    for nonce in probe_nonces {
        grind_probe_count = grind_probe_count.saturating_add(1);
        let challenges = preview_folding_challenges::<D>(
            transcript,
            lp.num_blocks,
            num_claims,
            &lp.stage1_config,
            &lp.fold_challenge_shape,
            labels,
            nonce,
            lp.op_norm_rejection,
        )?;
        let witness = build_point_decompose_fold_witness::<F, P, B, D>(
            backend,
            prepared,
            &challenges,
            polys,
            &point_indices,
            lp,
        )?;
        if !accepts_fold_witness(&contract, &witness, witness_linf_cap, tail_t_vectors) {
            continue;
        }
        super::fold_grind_observer::record_fold_grind_acceptance(nonce, grind_probe_count);
        let challenges = sample_folding_challenges::<F, T, D>(
            transcript,
            lp.num_blocks,
            num_claims,
            &lp.stage1_config,
            &lp.fold_challenge_shape,
            labels,
            nonce,
            lp.op_norm_rejection,
        )?;
        return Ok((witness, challenges, nonce));
    }

    Err(AkitaError::InvalidInput(format!(
        "fold grind exceeded {} attempts (threshold={})",
        contract.max_nonce_exclusive, contract.witness_linf_cap
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_transcript::AkitaTranscript;
    use akita_types::sis::{FoldWitnessGrindContract, FoldWitnessLinfCapPolicy};
    use akita_types::SisModulusFamily;

    type F = akita_field::Prime128Offset275;

    fn sample_level() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            64,
            3,
            2,
            4,
            3,
            SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
            },
        )
    }

    #[test]
    fn transcript_shuffle_order_differs_from_sequential() {
        let lp = sample_level();
        let contract = FoldWitnessGrindContract {
            policy: FoldWitnessLinfCapPolicy::TailBoundWithGrind,
            witness_linf_cap: 1_000,
            max_nonce_exclusive: 64,
        };
        let transcript = AkitaTranscript::<F>::prover(b"grind/order", b"instance");
        let mut binding = FoldLinfProtocolBinding::CURRENT;
        binding.grind_probe_order = FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE;
        let shuffled =
            grind_probe_nonces(&contract, &binding, &transcript, &lp, 1).expect("shuffle order");
        let sequential = (0..contract.max_nonce_exclusive).collect::<Vec<_>>();
        assert_ne!(shuffled, sequential);
    }

    #[test]
    fn worst_case_beta_only_still_rejects_golomb_inadmissible_terminal_tail() {
        const D: usize = 4;
        let cap = 1008u128;
        let contract = FoldWitnessGrindContract {
            policy: FoldWitnessLinfCapPolicy::WorstCaseBetaOnly,
            witness_linf_cap: cap,
            max_nonce_exclusive: 1,
        };
        let witness = DecomposeFoldWitness::<F, D> {
            z_folded_rings: Vec::new(),
            centered_coeffs: vec![[cap as i32; D]],
            centered_inf_norm: cap as u32,
        };
        assert!(!accepts_fold_witness(&contract, &witness, cap, Some(1),));
    }
}
