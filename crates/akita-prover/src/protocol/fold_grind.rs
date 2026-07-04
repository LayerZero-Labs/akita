//! Fold-l∞ Fiat–Shamir grind: preview off-sponge clones, commit the winning nonce.

use crate::compute::{
    OpeningBatchKernel, OpeningFoldKernel, RootOpeningSource, RuntimeOpeningProveBackendFor,
};
use akita_challenges::{
    grind_probe_permutation, preview_folding_challenges, sample_folding_challenges,
    stage1_fold_challenge_labels, Challenges, SparseChallengeConfig, TensorChallengeShape,
};
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_transcript::{AkitaTranscript, FoldChallengeSeedPreview, Transcript, TranscriptSponge};
use akita_types::{
    golomb_rice_rows_admit_terminal_wire,
    sis::{
        fold_witness_beta, fold_witness_verifier_linf_bound, FoldWitnessGrindContract,
        FoldWitnessLinfCapPolicy,
    },
    FoldLinfProtocolBinding, LevelParams, FOLD_GRIND_PROBE_ORDER_SEQUENTIAL_MIN,
    FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE,
};

use super::fold_grind_observer::{self, FoldGrindLevelMeta, FoldGrindObservation};
use super::ring_relation::{
    aggregate_decompose_fold_witnesses, build_point_decompose_fold_witness,
    window_sparse_challenges, PointFoldShape,
};
use crate::DecomposeFoldWitness;

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

struct FoldGrindAcceptanceCtx {
    check_grind_cap: bool,
    check_golomb: bool,
    witness_linf_cap: u128,
    digit_negative_abs_bound: u128,
    digit_positive_bound: u128,
}

fn fold_grind_acceptance_ctx(
    contract: &FoldWitnessGrindContract,
    witness_linf_cap: u128,
    digit_negative_abs_bound: u128,
    digit_positive_bound: u128,
    tail_t_vectors: Option<usize>,
) -> FoldGrindAcceptanceCtx {
    FoldGrindAcceptanceCtx {
        check_grind_cap: contract.policy != FoldWitnessLinfCapPolicy::WorstCaseBetaOnly,
        check_golomb: tail_t_vectors.is_some(),
        witness_linf_cap,
        digit_negative_abs_bound,
        digit_positive_bound,
    }
}

fn coeff_within_digit_bounds(coeff: i32, ctx: &FoldGrindAcceptanceCtx) -> bool {
    if coeff < 0 {
        u128::from(coeff.unsigned_abs()) <= ctx.digit_negative_abs_bound
    } else {
        (coeff as u128) <= ctx.digit_positive_bound
    }
}

fn accepts_fold_witness<F: CanonicalField, const D: usize>(
    contract: &FoldWitnessGrindContract,
    witness: &DecomposeFoldWitness<F>,
    z_folded_centered_per_chunk: &[Vec<[i32; D]>],
    witness_linf_cap: u128,
    digit_negative_abs_bound: u128,
    digit_positive_bound: u128,
    tail_t_vectors: Option<usize>,
) -> bool {
    let ctx = fold_grind_acceptance_ctx(
        contract,
        witness_linf_cap,
        digit_negative_abs_bound,
        digit_positive_bound,
        tail_t_vectors,
    );
    for coeff in z_folded_centered_per_chunk
        .iter()
        .flat_map(|chunk| chunk.iter())
        .flat_map(|coeffs| coeffs.iter())
    {
        if !coeff_within_digit_bounds(*coeff, &ctx) {
            return false;
        }
        if ctx.check_grind_cap && u128::from(coeff.unsigned_abs()) > ctx.witness_linf_cap {
            return false;
        }
    }
    if ctx.check_grind_cap && u128::from(witness.centered_inf_norm) > ctx.witness_linf_cap {
        return false;
    }
    if ctx.check_golomb
        && golomb_rice_rows_admit_terminal_wire(
            witness.centered_coeffs_trusted::<D>(),
            ctx.witness_linf_cap,
        )
        .is_err()
    {
        return false;
    }
    if ctx.check_golomb
        && z_folded_centered_per_chunk
            .iter()
            .any(|chunk| golomb_rice_rows_admit_terminal_wire(chunk, ctx.witness_linf_cap).is_err())
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
        FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE if contract.policy.allows_grind() => {
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

/// Extracted fold-probe geometry from one schedule level.
///
/// Callers extract these numbers from [`LevelParams`] before entering the
/// const-`D` kernel; the kernel must not read schedule types.
#[derive(Debug, Clone, Copy)]
struct FoldProbeParams {
    point_fold: PointFoldShape,
    num_chunks: usize,
    blocks_per_chunk: usize,
}

impl FoldProbeParams {
    fn from_level(lp: &LevelParams) -> Self {
        let num_chunks = lp.witness_chunk.num_chunks;
        Self {
            point_fold: PointFoldShape::from_level(lp),
            num_chunks,
            blocks_per_chunk: lp.num_blocks / num_chunks.max(1),
        }
    }
}

/// One fold probe: returns the global folded witness and the per-window centered
/// responses `z_i` under the given (preview) challenges.
///
/// For `num_chunks <= 1` this is the legacy single global fold and the sole
/// window equals the global centered response (byte-identical to the
/// pre-chunking path). For `num_chunks > 1` the fold is computed per block
/// window (`window_sparse_challenges`) and the global witness is the exact
/// coefficient-wise sum of the windows (`Σ_i z_i = z`), so grind acceptance on
/// the global L∞ is identical to a standalone global fold over all blocks.
#[allow(clippy::type_complexity)]
fn fold_probe_witness_kernel<F, P, B, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    challenges: &Challenges,
    polys: &[&P],
    point_indices: &[usize],
    params: FoldProbeParams,
) -> Result<(DecomposeFoldWitness<F>, Vec<Vec<[i32; D]>>), AkitaError>
where
    F: FieldCore + CanonicalField,
    P: RootOpeningSource<F, D>,
    B: crate::compute::ComputeBackendSetup<F>
        + for<'a> OpeningBatchKernel<P::OpeningBatchView<'a>, F, D>
        + for<'a> OpeningFoldKernel<P::OpeningView<'a>, F, D>,
{
    let FoldProbeParams {
        point_fold: shape,
        num_chunks,
        blocks_per_chunk,
    } = params;
    if num_chunks <= 1 {
        let witness = build_point_decompose_fold_witness::<F, P, B, D>(
            backend,
            prepared,
            challenges,
            polys,
            point_indices,
            shape,
        )?;
        let per_chunk = vec![witness.centered_coeffs_owned::<D>()];
        return Ok((witness, per_chunk));
    }

    let windows = (0..num_chunks)
        .map(|chunk| {
            let windowed = window_sparse_challenges(challenges, chunk, blocks_per_chunk)?;
            build_point_decompose_fold_witness::<F, P, B, D>(
                backend,
                prepared,
                &windowed,
                polys,
                point_indices,
                shape,
            )
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let per_chunk = windows
        .iter()
        .map(|w| w.centered_coeffs_owned::<D>())
        .collect();
    let global = aggregate_decompose_fold_witnesses::<F, D>(windows)?;
    Ok((global, per_chunk))
}

/// Grind loop for one compile-time ring dimension: one dispatch arm, no per-nonce rematch.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn sample_fold_decompose_witness_at_dim<F, P, B, T, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    transcript: &mut T,
    polys: &[&P],
    num_blocks: usize,
    stage1_config: &SparseChallengeConfig,
    fold_challenge_shape: TensorChallengeShape,
    num_claims: usize,
    tail_t_vectors: Option<usize>,
    ring_d: usize,
    contract: &FoldWitnessGrindContract,
    witness_linf_cap: u128,
    digit_negative_abs_bound: u128,
    digit_positive_bound: u128,
    level_meta: FoldGrindLevelMeta,
    fold_probe_params: FoldProbeParams,
    probe_nonces: &[u32],
) -> Result<(DecomposeFoldWitness<F>, Vec<Vec<Vec<i32>>>, Challenges, u32), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootOpeningSource<F, D>,
    B: crate::compute::ComputeBackendSetup<F>
        + for<'a> OpeningBatchKernel<P::OpeningBatchView<'a>, F, D>
        + for<'a> OpeningFoldKernel<P::OpeningView<'a>, F, D>,
    T: Transcript<F> + ProverTranscriptGrind<F>,
{
    let point_indices = (0..polys.len()).collect::<Vec<_>>();
    let labels = stage1_fold_challenge_labels();
    let mut grind_probe_count = 0u32;
    for &nonce in probe_nonces {
        grind_probe_count = grind_probe_count.saturating_add(1);
        let challenges = preview_folding_challenges(
            transcript,
            ring_d,
            num_blocks,
            num_claims,
            stage1_config,
            &fold_challenge_shape,
            labels,
            nonce,
        )?;
        let (witness, z_per_chunk) = fold_probe_witness_kernel::<F, P, B, D>(
            backend,
            prepared,
            &challenges,
            polys,
            &point_indices,
            fold_probe_params,
        )?;
        if !accepts_fold_witness::<F, D>(
            contract,
            &witness,
            &z_per_chunk,
            witness_linf_cap,
            digit_negative_abs_bound,
            digit_positive_bound,
            tail_t_vectors,
        ) {
            continue;
        }
        let z_folded_centered_per_chunk: Vec<Vec<Vec<i32>>> = z_per_chunk
            .into_iter()
            .map(|chunk| chunk.into_iter().map(|row| row.to_vec()).collect())
            .collect();
        fold_grind_observer::record_fold_grind_acceptance(FoldGrindObservation {
            level_index: fold_grind_observer::next_fold_grind_level_index(),
            grind_nonce: nonce,
            grind_probe_count,
            observed_linf: witness.centered_inf_norm,
            level_meta,
        });
        let challenges = sample_folding_challenges::<F, T>(
            transcript,
            ring_d,
            num_blocks,
            num_claims,
            stage1_config,
            &fold_challenge_shape,
            labels,
            nonce,
        )?;
        return Ok((witness, z_folded_centered_per_chunk, challenges, nonce));
    }

    Err(AkitaError::InvalidInput(format!(
        "fold grind exceeded {} attempts (threshold={})",
        contract.max_nonce_exclusive, contract.witness_linf_cap
    )))
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
#[allow(clippy::type_complexity)]
pub(crate) fn sample_fold_decompose_witness<F, P, B, T>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    transcript: &mut T,
    polys: &[&P],
    lp: &LevelParams,
    num_claims: usize,
    tail_t_vectors: Option<usize>,
) -> Result<(DecomposeFoldWitness<F>, Vec<Vec<Vec<i32>>>, Challenges, u32), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootOpeningSource<F, 32>
        + RootOpeningSource<F, 64>
        + RootOpeningSource<F, 128>
        + RootOpeningSource<F, 256>,
    B: crate::compute::ComputeBackendSetup<F> + RuntimeOpeningProveBackendFor<F, P>,
    T: Transcript<F> + ProverTranscriptGrind<F>,
{
    // A-role fold dimension; per-role split attaches here (mixed-row spec).
    let ring_d = lp.role_dims().d_a();
    let binding = FoldLinfProtocolBinding::CURRENT;
    let contract = lp.fold_witness_grind_contract(num_claims, binding.max_grind_attempts)?;
    // Tail Golomb grinding sizes caps from `tail_t_vectors`; keep observer metrics on the
    // same claim count so snap hints do not mix β/t*/δ from one count with cap from another.
    let sizing_claims = tail_t_vectors.unwrap_or(num_claims);
    let witness_linf_cap = witness_linf_cap_for_grind(lp, &contract, tail_t_vectors)?;
    let digit_plan = lp.fold_witness_linf_digit_plan_for_claims(sizing_claims)?;
    let delta_fold = digit_plan.delta_fold;
    let (digit_negative_abs_bound, digit_positive_bound) =
        akita_types::sis::fold_witness_representable_linf_bounds(lp.log_basis, delta_fold);
    let verifier_linf_bound = fold_witness_verifier_linf_bound(lp.log_basis, delta_fold);
    let challenge =
        akita_types::sis::fold_challenge_norms(&lp.stage1_config, lp.fold_challenge_shape);
    let witness_norms = lp.fold_witness_norms();
    let beta_inf = fold_witness_beta(lp.r_vars, sizing_claims, challenge, witness_norms)?;
    let level_meta = FoldGrindLevelMeta {
        beta_inf,
        t_star: digit_plan.t_star,
        honest_cap: witness_linf_cap,
        delta_fold,
        verifier_linf_bound,
        policy: contract.policy,
        log_basis: lp.log_basis,
        r_vars: u32::try_from(lp.r_vars).unwrap_or(u32::MAX),
        num_claims: u32::try_from(sizing_claims).unwrap_or(u32::MAX),
    };
    let fold_probe_params = FoldProbeParams::from_level(lp);
    let probe_nonces = grind_probe_nonces(&contract, &binding, transcript, lp, num_claims)?;

    let num_blocks = lp.num_blocks;
    let stage1_config = &lp.stage1_config;
    let fold_challenge_shape = lp.fold_challenge_shape;

    akita_types::dispatch_ring_dim_result!(ring_d, |D| {
        sample_fold_decompose_witness_at_dim::<F, P, B, T, D>(
            backend,
            prepared,
            transcript,
            polys,
            num_blocks,
            stage1_config,
            fold_challenge_shape,
            num_claims,
            tail_t_vectors,
            ring_d,
            &contract,
            witness_linf_cap,
            digit_negative_abs_bound,
            digit_positive_bound,
            level_meta,
            fold_probe_params,
            &probe_nonces,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::CyclotomicRing;
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
        let witness = DecomposeFoldWitness::from_parts::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            vec![[cap as i32; D]],
            cap as u32,
        );
        let chunks = vec![witness.centered_coeffs_owned::<D>()];
        let (neg_bound, pos_bound) = akita_types::sis::fold_witness_representable_linf_bounds(4, 2);
        assert!(!accepts_fold_witness::<F, D>(
            &contract,
            &witness,
            &chunks,
            cap,
            neg_bound,
            pos_bound,
            Some(1),
        ));
    }

    #[test]
    fn grind_rejects_chunk_payload_outside_snapped_cap() {
        const D: usize = 4;
        let cap = 32u128;
        let contract = FoldWitnessGrindContract {
            policy: FoldWitnessLinfCapPolicy::TailBoundWithGrind,
            witness_linf_cap: cap,
            max_nonce_exclusive: 8,
        };
        let witness = DecomposeFoldWitness::from_parts::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            vec![[12; D]],
            12,
        );
        let chunks = vec![vec![[33, 0, 0, 0]], vec![[-12; D]]];
        let (neg_bound, pos_bound) = akita_types::sis::fold_witness_representable_linf_bounds(4, 2);
        assert!(!accepts_fold_witness::<F, D>(
            &contract, &witness, &chunks, cap, neg_bound, pos_bound, None,
        ));
    }

    #[test]
    fn grind_rejects_positive_coefficients_past_balanced_digit_reach() {
        const D: usize = 4;
        let contract = FoldWitnessGrindContract {
            policy: FoldWitnessLinfCapPolicy::TailBoundWithGrind,
            witness_linf_cap: 2080,
            max_nonce_exclusive: 8,
        };
        let witness = DecomposeFoldWitness::from_parts::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            vec![[2022, 0, 0, 0]],
            2022,
        );
        let chunks = vec![witness.centered_coeffs_owned::<D>()];
        let (neg_bound, pos_bound) = akita_types::sis::fold_witness_representable_linf_bounds(6, 2);
        assert_eq!(neg_bound, 2080);
        assert_eq!(pos_bound, 2015);
        assert!(!accepts_fold_witness::<F, D>(
            &contract, &witness, &chunks, 2080, neg_bound, pos_bound, None,
        ));
    }
}
