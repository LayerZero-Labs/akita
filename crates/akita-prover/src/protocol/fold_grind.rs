//! Fold-l∞ Fiat–Shamir grind: preview off-sponge clones, commit the winning nonce.

use crate::compute::{
    OpeningBatchKernel, OpeningFoldKernel, RootOpeningSource, RuntimeOpeningProveBackendFor,
};
use akita_challenges::{
    grind_probe_permutation, witness_fold_challenge_labels, Challenges, FoldDraw, LiveFoldDraw,
    PreviewFoldDraw,
};
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_transcript::{AkitaTranscript, FoldChallengeSeedPreview, Transcript, TranscriptSponge};
use akita_types::{
    golomb_rice_rows_admit_terminal_wire,
    sis::{FoldWitnessGrindBatchContract, FoldWitnessGrindContract, FoldWitnessLinfCapPolicy},
    FoldLinfProtocolBinding, LevelParams, LevelParamsLike, OpeningClaimsLayout,
    FOLD_GRIND_PROBE_ORDER_SEQUENTIAL_MIN, FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE,
};

use super::ring_relation::{
    aggregate_decompose_fold_witnesses, build_point_decompose_fold_witness,
    window_sparse_challenges,
};
use crate::DecomposeFoldWitness;
use akita_types::dispatch_for_field;

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
    ctx: &FoldGrindAcceptanceCtx,
    witness: &DecomposeFoldWitness<F>,
    z_folded_centered_per_chunk: &[Vec<[i32; D]>],
) -> bool {
    for coeff in z_folded_centered_per_chunk
        .iter()
        .flat_map(|chunk| chunk.iter())
        .flat_map(|coeffs| coeffs.iter())
    {
        if !coeff_within_digit_bounds(*coeff, ctx) {
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

pub(crate) fn grind_probe_nonces(
    contract: &FoldWitnessGrindBatchContract,
    binding: &FoldLinfProtocolBinding,
    transcript: &dyn FoldChallengeSeedPreview,
    root_lp: &LevelParams,
    groups: &[(&dyn LevelParamsLike, usize)],
) -> Result<Vec<u32>, AkitaError> {
    let cap = contract.max_nonce_exclusive();
    match binding.grind_probe_order {
        FOLD_GRIND_PROBE_ORDER_SEQUENTIAL_MIN => Ok((0..cap).collect()),
        FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE if contract.allows_grind() => {
            let absorb_buf = fold_grind_probe_order_absorb_buf(root_lp, groups)?;
            let seed = transcript.preview_challenge_bytes_after_absorb(&absorb_buf, 32);
            Ok(grind_probe_permutation(&seed, cap))
        }
        FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE => Ok(vec![0]),
        other => Err(AkitaError::InvalidSetup(format!(
            "unsupported fold grind probe order tag {other}"
        ))),
    }
}

fn fold_grind_probe_order_absorb_buf(
    root_lp: &LevelParams,
    groups: &[(&dyn LevelParamsLike, usize)],
) -> Result<Vec<u8>, AkitaError> {
    fn push_usize(buf: &mut Vec<u8>, value: usize, name: &str) -> Result<(), AkitaError> {
        let value = u64::try_from(value)
            .map_err(|_| AkitaError::InvalidSetup(format!("{name} exceeds u64")))?;
        buf.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    let mut buf = Vec::with_capacity(24 + 72 * groups.len());
    buf.extend_from_slice(akita_types::sis::FOLD_GRIND_PROBE_ORDER_ABSORB);
    push_usize(&mut buf, root_lp.d_a(), "fold grind ring dimension")?;
    push_usize(&mut buf, groups.len(), "fold grind group count")?;
    for (group_index, (params, num_claims)) in groups.iter().copied().enumerate() {
        push_usize(&mut buf, group_index, "fold grind group index")?;
        buf.extend_from_slice(&params.log_basis_open().to_le_bytes());
        push_usize(
            &mut buf,
            params.num_live_ring_elements_per_claim(),
            "fold grind source length",
        )?;
        push_usize(
            &mut buf,
            params.num_positions_per_block(),
            "fold grind position count",
        )?;
        push_usize(
            &mut buf,
            params.num_live_blocks(),
            "fold grind num_live_blocks",
        )?;
        push_usize(&mut buf, params.a_col_len(), "fold grind A width")?;
        push_usize(&mut buf, num_claims, "fold grind claim count")?;
        match params.fold_challenge_shape() {
            akita_challenges::TensorChallengeShape::Flat => {
                buf.push(0);
                push_usize(&mut buf, 0, "fold grind flat low length")?;
            }
            akita_challenges::TensorChallengeShape::Tensor { fold_low_len } => {
                buf.push(1);
                push_usize(&mut buf, fold_low_len, "fold grind tensor low length")?;
            }
        }
    }
    Ok(buf)
}

pub(crate) struct FoldGrindGroup<'params, 'poly, P> {
    pub(crate) group_index: usize,
    pub(crate) polys: &'poly [&'poly P],
    pub(crate) params: &'params dyn LevelParamsLike,
}

impl<P> Copy for FoldGrindGroup<'_, '_, P> {}

impl<P> Clone for FoldGrindGroup<'_, '_, P> {
    fn clone(&self) -> Self {
        *self
    }
}

pub(crate) struct FoldGrindGroupOutput<F: FieldCore> {
    pub(crate) witness: DecomposeFoldWitness<F>,
    pub(crate) centered_per_chunk: Vec<Vec<Vec<i32>>>,
    pub(crate) challenges: Challenges,
}

struct PreparedFoldGrindGroup<'params, 'poly, P> {
    input: FoldGrindGroup<'params, 'poly, P>,
    acceptance: FoldGrindAcceptanceCtx,
    point_indices: Vec<usize>,
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
    root_lp: &LevelParams,
    params: &(impl LevelParamsLike + ?Sized),
) -> Result<(DecomposeFoldWitness<F>, Vec<Vec<[i32; D]>>), AkitaError>
where
    F: FieldCore + CanonicalField,
    P: RootOpeningSource<F, D>,
    B: crate::compute::ComputeBackendSetup<F>
        + for<'a> OpeningBatchKernel<P::OpeningBatchView<'a>, F, D>
        + for<'a> OpeningFoldKernel<P::OpeningView<'a>, F, D>,
{
    let num_chunks = root_lp.witness_chunk.num_chunks;
    if num_chunks <= 1 {
        let witness = build_point_decompose_fold_witness::<F, P, B, D>(
            backend,
            prepared,
            challenges,
            polys,
            point_indices,
            params,
        )?;
        let per_chunk = vec![witness.centered_coeffs_owned::<D>()];
        return Ok((witness, per_chunk));
    }

    let chunk_block_ranges = akita_types::WitnessLayout::resolve_chunk_block_ranges(
        params.num_live_blocks(),
        num_chunks,
    )?;
    let windows = chunk_block_ranges
        .into_iter()
        .map(|fold_range| {
            let windowed = window_sparse_challenges(challenges, fold_range)?;
            build_point_decompose_fold_witness::<F, P, B, D>(
                backend,
                prepared,
                &windowed,
                polys,
                point_indices,
                params,
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

fn first_jointly_accepted_nonce<T>(
    probe_nonces: &[u32],
    mut probe: impl FnMut(u32) -> Result<Option<T>, AkitaError>,
) -> Result<(u32, T), AkitaError> {
    for &nonce in probe_nonces {
        if let Some(value) = probe(nonce)? {
            return Ok((nonce, value));
        }
    }
    Err(AkitaError::InvalidInput(format!(
        "fold grind exceeded {} joint attempts",
        probe_nonces.len()
    )))
}

/// Probe every group as one transcript transaction for each candidate nonce.
#[allow(clippy::too_many_arguments)]
fn sample_multi_group_fold_decompose_witnesses_at_dim<F, P, B, T, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    transcript: &mut T,
    root_lp: &LevelParams,
    groups: &[PreparedFoldGrindGroup<'_, '_, P>],
    probe_nonces: &[u32],
) -> Result<(Vec<FoldGrindGroupOutput<F>>, u32), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootOpeningSource<F, D>,
    B: crate::compute::ComputeBackendSetup<F>
        + for<'a> OpeningBatchKernel<P::OpeningBatchView<'a>, F, D>
        + for<'a> OpeningFoldKernel<P::OpeningView<'a>, F, D>,
    T: Transcript<F> + ProverTranscriptGrind<F>,
{
    if groups.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "fold grind batch has no groups".to_string(),
        ));
    }
    let ring_d = root_lp.role_dims().d_a();
    let labels = witness_fold_challenge_labels();
    let (nonce, mut candidate_outputs) = first_jointly_accepted_nonce(probe_nonces, |nonce| {
        let mut candidate_outputs = Vec::with_capacity(groups.len());
        {
            let mut preview = PreviewFoldDraw::new(transcript);
            for prepared_group in groups {
                let group = &prepared_group.input;
                let challenges = preview.draw_folding_challenges(
                    ring_d,
                    group.group_index,
                    group.params.num_live_blocks(),
                    group.polys.len(),
                    &root_lp.fold_challenge_config,
                    &group.params.fold_challenge_shape(),
                    labels,
                    nonce,
                )?;
                let (witness, z_per_chunk) = fold_probe_witness_kernel::<F, P, B, D>(
                    backend,
                    prepared,
                    &challenges,
                    group.polys,
                    &prepared_group.point_indices,
                    root_lp,
                    group.params,
                )?;
                if !accepts_fold_witness::<F, D>(&prepared_group.acceptance, &witness, &z_per_chunk)
                {
                    return Ok(None);
                }
                let centered_per_chunk = z_per_chunk
                    .into_iter()
                    .map(|chunk| chunk.into_iter().map(|row| row.to_vec()).collect())
                    .collect();
                candidate_outputs.push(FoldGrindGroupOutput {
                    witness,
                    centered_per_chunk,
                    challenges,
                });
            }
        }
        Ok(Some(candidate_outputs))
    })?;

    let mut live = LiveFoldDraw::<F, T>::new(transcript);
    for (prepared_group, output) in groups.iter().zip(candidate_outputs.iter_mut()) {
        let group = &prepared_group.input;
        let challenges = live.draw_folding_challenges(
            ring_d,
            group.group_index,
            group.params.num_live_blocks(),
            group.polys.len(),
            &root_lp.fold_challenge_config,
            &group.params.fold_challenge_shape(),
            labels,
            nonce,
        )?;
        if challenges != output.challenges {
            return Err(AkitaError::InvalidInput(
                "fold grind preview did not match live transcript replay".to_string(),
            ));
        }
    }
    Ok((candidate_outputs, nonce))
}

/// Probe all root groups off-sponge and commit the first jointly accepted nonce.
///
/// Plain presets probe `nonce = 0, 1, …` (minimum accepting nonce). ZK presets
/// with tail-bound grind use a transcript-seeded uniform permutation of the same
/// range; see `specs/fold-linf-rejection.md` (*ZK: grind probe order*).
///
/// When `tail_t_vectors` is set, presets reject witnesses whose centered coefficients do not
/// fit the terminal Golomb planner budget at wire low bits (including `WorstCaseBetaOnly`
/// presets that do not reroll on linf cap).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_multi_group_fold_decompose_witnesses<F, P, B, T>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    transcript: &mut T,
    root_lp: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    groups: &[FoldGrindGroup<'_, '_, P>],
    tail_t_vectors: Option<usize>,
) -> Result<(Vec<FoldGrindGroupOutput<F>>, u32), AkitaError>
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
    let ring_d = root_lp.role_dims().d_a();
    let binding = FoldLinfProtocolBinding::CURRENT;
    let contract =
        root_lp.fold_witness_grind_batch_contract(opening_batch, binding.max_grind_attempts)?;
    if groups.len() != contract.group_contracts().len() {
        return Err(AkitaError::InvalidSetup(
            "fold grind groups do not match the batch contract".to_string(),
        ));
    }
    let mut prepared_groups = Vec::with_capacity(groups.len());
    for (expected_group_index, (group, group_contract)) in
        groups.iter().zip(contract.group_contracts()).enumerate()
    {
        let expected_claims = opening_batch
            .group_layout(expected_group_index)?
            .num_polynomials();
        if group.group_index != expected_group_index
            || group.polys.is_empty()
            || group.polys.len() != expected_claims
        {
            return Err(AkitaError::InvalidSetup(
                "fold grind group descriptor is malformed".to_string(),
            ));
        }
        let challenge = akita_types::sis::FoldChallengeNorms::new(
            &root_lp.fold_challenge_config,
            group.params.fold_challenge_shape(),
        );
        let cap_config = root_lp.fold_witness_linf_cap_config_for_params(group.params)?;
        let witness_norms = root_lp.fold_witness_norms_for_params(group.params);
        let sizing_claims = tail_t_vectors.unwrap_or(group.polys.len());
        let (delta_fold, witness_linf_cap) = akita_types::sis::fold_witness_digit_plan(
            group.params.num_live_blocks(),
            sizing_claims,
            root_lp.field_bits_for_cache(),
            group.params.log_basis_open(),
            challenge,
            witness_norms,
            &cap_config,
        )?;
        let (digit_negative_abs_bound, digit_positive_bound) =
            akita_types::sis::fold_witness_representable_linf_bounds(
                group.params.log_basis_open(),
                delta_fold,
            );
        prepared_groups.push(PreparedFoldGrindGroup {
            input: *group,
            acceptance: fold_grind_acceptance_ctx(
                group_contract,
                witness_linf_cap,
                digit_negative_abs_bound,
                digit_positive_bound,
                tail_t_vectors,
            ),
            point_indices: (0..group.polys.len()).collect(),
        });
    }
    let group_geometries = prepared_groups
        .iter()
        .map(|group| (group.input.params, group.input.polys.len()))
        .collect::<Vec<_>>();
    let probe_nonces =
        grind_probe_nonces(&contract, &binding, transcript, root_lp, &group_geometries)?;

    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        ring_d,
        |D| {
            sample_multi_group_fold_decompose_witnesses_at_dim::<F, P, B, T, D>(
                backend,
                prepared,
                transcript,
                root_lp,
                &prepared_groups,
                &probe_nonces,
            )
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::CyclotomicRing;
    use akita_challenges::SparseChallengeConfig;
    use akita_transcript::AkitaTranscript;
    use akita_types::sis::{FoldWitnessGrindContract, FoldWitnessLinfCapPolicy};
    use akita_types::SisModulusProfileId;

    type F = akita_field::Prime128Offset275;

    fn sample_level() -> LevelParams {
        LevelParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            3,
            2,
            4,
            3,
            SparseChallengeConfig::pm1_only(3),
        )
    }

    #[test]
    fn transcript_shuffle_order_differs_from_sequential() {
        let lp = sample_level();
        let group_contract = FoldWitnessGrindContract {
            policy: FoldWitnessLinfCapPolicy::TailBoundWithGrind,
            witness_linf_cap: 1_000,
        };
        let contract = FoldWitnessGrindBatchContract::new(vec![group_contract], 64).unwrap();
        let transcript = AkitaTranscript::<F>::prover(b"grind/order", b"instance");
        let mut binding = FoldLinfProtocolBinding::CURRENT;
        binding.grind_probe_order = FOLD_GRIND_PROBE_ORDER_TRANSCRIPT_SHUFFLE;
        let groups = [(&lp as &dyn LevelParamsLike, 1)];
        let shuffled = grind_probe_nonces(&contract, &binding, &transcript, &lp, &groups)
            .expect("shuffle order");
        let sequential = (0..contract.max_nonce_exclusive()).collect::<Vec<_>>();
        assert_ne!(shuffled, sequential);
    }

    #[test]
    fn joint_grind_skips_different_group_first_nonces() {
        let group_accepts = [[0, 2], [1, 2]];
        let mut probed = Vec::new();
        let (nonce, ()) = first_jointly_accepted_nonce(&[0, 1, 2, 3], |nonce| {
            probed.push(nonce);
            Ok(group_accepts
                .iter()
                .all(|accepted| accepted.contains(&nonce))
                .then_some(()))
        })
        .unwrap();

        assert_eq!(nonce, 2);
        assert_eq!(probed, vec![0, 1, 2]);
    }

    #[test]
    fn worst_case_beta_only_still_rejects_golomb_inadmissible_terminal_tail() {
        const D: usize = 4;
        let cap = 1008u128;
        let contract = FoldWitnessGrindContract {
            policy: FoldWitnessLinfCapPolicy::WorstCaseBetaOnly,
            witness_linf_cap: cap,
        };
        let witness = DecomposeFoldWitness::from_parts::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            vec![[cap as i32; D]],
            cap as u32,
        );
        let chunks = vec![witness.centered_coeffs_owned::<D>()];
        let (neg_bound, pos_bound) = akita_types::sis::fold_witness_representable_linf_bounds(4, 2);
        let acceptance = fold_grind_acceptance_ctx(&contract, cap, neg_bound, pos_bound, Some(1));
        assert!(!accepts_fold_witness::<F, D>(
            &acceptance,
            &witness,
            &chunks,
        ));
    }

    #[test]
    fn grind_rejects_chunk_payload_outside_snapped_cap() {
        const D: usize = 4;
        let cap = 32u128;
        let contract = FoldWitnessGrindContract {
            policy: FoldWitnessLinfCapPolicy::TailBoundWithGrind,
            witness_linf_cap: cap,
        };
        let witness = DecomposeFoldWitness::from_parts::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            vec![[12; D]],
            12,
        );
        let chunks = vec![vec![[33, 0, 0, 0]], vec![[-12; D]]];
        let (neg_bound, pos_bound) = akita_types::sis::fold_witness_representable_linf_bounds(4, 2);
        let acceptance = fold_grind_acceptance_ctx(&contract, cap, neg_bound, pos_bound, None);
        assert!(!accepts_fold_witness::<F, D>(
            &acceptance,
            &witness,
            &chunks
        ));
    }

    #[test]
    fn grind_rejects_positive_coefficients_past_balanced_digit_reach() {
        const D: usize = 4;
        let contract = FoldWitnessGrindContract {
            policy: FoldWitnessLinfCapPolicy::TailBoundWithGrind,
            witness_linf_cap: 2080,
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
        let acceptance = fold_grind_acceptance_ctx(&contract, 2080, neg_bound, pos_bound, None);
        assert!(!accepts_fold_witness::<F, D>(
            &acceptance,
            &witness,
            &chunks
        ));
    }
}
