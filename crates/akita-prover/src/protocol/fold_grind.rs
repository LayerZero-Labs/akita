//! Fold-l∞ Fiat–Shamir grind: preview off-sponge clones, commit the winning nonce.

use crate::compute::{
    OpeningBatchKernel, OpeningFoldKernel, RootOpeningSource, RuntimeOpeningProveBackendFor,
};
use akita_challenges::{
    witness_fold_challenge_labels, Challenges, FoldDraw, LiveFoldDraw, PreviewFoldDraw,
};
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_transcript::{AkitaTranscript, FoldChallengeSeedPreview, Transcript, TranscriptSponge};
use akita_types::{
    dyadic_block_ranges, golomb_rice_total_wire_bits, golomb_rice_values_within_cap,
    golomb_rice_zigzag_width, CommittedGroupParams, FoldLinfProtocolBinding, LevelParamsLike,
    OpeningClaimsLayout, TerminalCommittedGroupParams, TerminalResponseShape,
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
    digit_negative_abs_bound: u128,
    digit_positive_bound: u128,
}

fn fold_grind_acceptance_ctx(
    digit_negative_abs_bound: u128,
    digit_positive_bound: u128,
) -> FoldGrindAcceptanceCtx {
    FoldGrindAcceptanceCtx {
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

#[cfg(test)]
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
    }
    let _ = witness;
    true
}

fn accepts_fold_witness_flat<F: CanonicalField>(
    ctx: &FoldGrindAcceptanceCtx,
    witness: &DecomposeFoldWitness<F>,
    centered_per_chunk: &[Vec<Vec<i32>>],
) -> bool {
    let coefficients = centered_per_chunk
        .iter()
        .flat_map(|chunk| chunk.iter())
        .flat_map(|row| row.iter());
    for &coefficient in coefficients {
        if !coeff_within_digit_bounds(coefficient, ctx) {
            return false;
        }
    }
    let _ = witness;
    true
}

pub(crate) struct FoldGrindGroup<'params, 'group, G> {
    pub(crate) group_index: usize,
    pub(crate) group: &'group G,
    pub(crate) params: &'params dyn LevelParamsLike,
}

impl<G> Copy for FoldGrindGroup<'_, '_, G> {}

impl<G> Clone for FoldGrindGroup<'_, '_, G> {
    fn clone(&self) -> Self {
        *self
    }
}

pub(crate) struct FoldGrindGroupOutput<F: FieldCore> {
    pub(crate) witness: DecomposeFoldWitness<F>,
    pub(crate) centered_per_chunk: Vec<Vec<Vec<i32>>>,
    pub(crate) challenges: Challenges,
}

pub(crate) struct TerminalFoldGrindOutput<F: FieldCore> {
    pub(crate) witness: DecomposeFoldWitness<F>,
    pub(crate) nonce: u32,
}

/// Sample the flat scalar terminal fold against its capacity-based response
/// cap. The returned witness retains centered `z` coefficients only; terminal
/// `e` and `t` are never gadget decomposed.
pub(crate) fn sample_terminal_fold_response<F, P, B, T>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    transcript: &mut T,
    params: &TerminalCommittedGroupParams,
    sparse: &akita_challenges::SparseChallengeConfig,
    poly: &P,
    shape: &TerminalResponseShape,
) -> Result<TerminalFoldGrindOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootOpeningSource<F, 32>
        + RootOpeningSource<F, 64>
        + RootOpeningSource<F, 128>
        + RootOpeningSource<F, 256>
        + RootOpeningSource<F, 512>,
    B: crate::compute::ComputeBackendSetup<F> + RuntimeOpeningProveBackendFor<F, P>,
    T: Transcript<F> + ProverTranscriptGrind<F>,
{
    let expected_group =
        shape.layout.groups.first().ok_or_else(|| {
            AkitaError::InvalidSetup("terminal response shape has no group".into())
        })?;
    if shape.layout.groups.len() != 1
        || expected_group.z_coords
            != params
                .inner_width()
                .checked_mul(params.d_a())
                .ok_or_else(|| AkitaError::InvalidSetup("terminal z width overflow".into()))?
    {
        return Err(AkitaError::InvalidSetup(
            "terminal response shape does not match terminal A width".into(),
        ));
    }
    let admission_cap = expected_group.z_admission_linf_cap;
    if admission_cap > params.certified_response_linf_cap(sparse)? {
        return Err(AkitaError::InvalidSetup(
            "terminal response cap exceeds its fixed matrix capacity".into(),
        ));
    }
    let binding = FoldLinfProtocolBinding::CURRENT;
    let labels = witness_fold_challenge_labels();
    let polys = [poly];
    let point_indices = [0usize];
    let (nonce, (witness, challenges)) =
        first_jointly_accepted_nonce(binding.max_grind_attempts, |nonce| {
            let mut preview = PreviewFoldDraw::new(transcript);
            let challenges = preview.draw_folding_challenges(
                params.d_a(),
                0,
                params.num_live_blocks,
                1,
                sparse,
                &akita_challenges::TensorChallengeShape::Flat,
                labels,
                nonce,
            )?;
            let witness = dispatch_for_field!(
                akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
                F,
                params.d_a(),
                |D| {
                    build_point_decompose_fold_witness::<F, P, B, D>(
                        backend,
                        prepared,
                        &challenges,
                        &polys,
                        &point_indices,
                        params.num_positions_per_block,
                        params.num_digits_inner,
                        params.log_basis_inner,
                    )
                }
            )?;
            let centered = witness
                .centered_coeffs_flat()
                .iter()
                .map(|&value| i64::from(value))
                .collect::<Vec<_>>();
            if golomb_rice_values_within_cap(&centered, admission_cap).is_err() {
                return Ok(None);
            }
            let zigzag_width = golomb_rice_zigzag_width(admission_cap);
            let wire_bits = golomb_rice_total_wire_bits(
                &centered,
                expected_group.z_rice_low_bits,
                zigzag_width,
            )?;
            if wire_bits > expected_group.z_payload_bytes.saturating_mul(8) {
                return Ok(None);
            }
            Ok(Some((witness, challenges)))
        })?;
    let mut live = LiveFoldDraw::<F, T>::new(transcript);
    let live_challenges = live.draw_folding_challenges(
        params.d_a(),
        0,
        params.num_live_blocks,
        1,
        sparse,
        &akita_challenges::TensorChallengeShape::Flat,
        labels,
        nonce,
    )?;
    if live_challenges != challenges {
        return Err(AkitaError::InvalidInput(
            "terminal grind preview did not match live transcript replay".into(),
        ));
    }
    Ok(TerminalFoldGrindOutput { witness, nonce })
}

struct PreparedFoldGrindGroup<'params, 'group, G> {
    input: FoldGrindGroup<'params, 'group, G>,
    acceptance: FoldGrindAcceptanceCtx,
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
pub(in crate::protocol) fn fold_probe_witness_kernel<F, P, B, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    challenges: &Challenges,
    polys: &[&P],
    point_indices: &[usize],
    root_lp: &CommittedGroupParams,
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
            params.num_positions_per_block(),
            params.num_digits_inner(),
            params.log_basis_inner(),
        )?;
        let per_chunk = vec![witness.centered_coeffs_owned::<D>()];
        return Ok((witness, per_chunk));
    }

    let chunk_block_ranges = dyadic_block_ranges(params.num_live_blocks(), num_chunks)?;
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
                params.num_positions_per_block(),
                params.num_digits_inner(),
                params.log_basis_inner(),
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
    max_grind_attempts: u32,
    mut probe: impl FnMut(u32) -> Result<Option<T>, AkitaError>,
) -> Result<(u32, T), AkitaError> {
    for nonce in 0..max_grind_attempts {
        if let Some(value) = probe(nonce)? {
            return Ok((nonce, value));
        }
    }
    Err(AkitaError::InvalidInput(format!(
        "fold grind exceeded {} joint attempts",
        max_grind_attempts
    )))
}

/// Probe every group at its native A dimension as one transcript transaction
/// for each candidate nonce.
#[allow(clippy::too_many_arguments)]
fn sample_multi_group_fold_decompose_witnesses_native<F, E, G, B, T>(
    opening_ctx: &crate::compute::OperationCtx<'_, F, B>,
    transcript: &mut T,
    root_lp: &CommittedGroupParams,
    groups: &[PreparedFoldGrindGroup<'_, '_, G>],
    max_grind_attempts: u32,
) -> Result<(Vec<FoldGrindGroupOutput<F>>, u32), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: akita_types::FpExtEncoding<F>
        + akita_field::ExtField<F>
        + akita_serialization::AkitaSerialize,
    G: crate::protocol::core::RootProverGroupOpening<F, E, B>,
    B: crate::compute::ComputeBackendSetup<F> + crate::DigitRowsComputeBackend<F>,
    T: Transcript<F> + ProverTranscriptGrind<F>,
{
    if groups.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "fold grind batch has no groups".to_string(),
        ));
    }
    let labels = witness_fold_challenge_labels();
    let (nonce, mut candidate_outputs) =
        first_jointly_accepted_nonce(max_grind_attempts, |nonce| {
            let mut candidate_outputs = Vec::with_capacity(groups.len());
            {
                let mut preview = PreviewFoldDraw::new(transcript);
                for prepared_group in groups {
                    let group = &prepared_group.input;
                    let ring_d = group.params.inner_commit_matrix_params().ring_dimension();
                    let challenges = preview.draw_folding_challenges(
                        ring_d,
                        group.group_index,
                        group.params.num_live_blocks(),
                        group.group.num_polynomials(),
                        &group.params.fold_challenge_config(),
                        &group.params.fold_challenge_shape(),
                        labels,
                        nonce,
                    )?;
                    let output =
                        group
                            .group
                            .probe_fold(opening_ctx, &challenges, root_lp, group.params)?;
                    let candidate = accepts_fold_witness_flat(
                        &prepared_group.acceptance,
                        &output.witness,
                        &output.centered_per_chunk,
                    )
                    .then_some(output);
                    let Some(candidate) = candidate else {
                        return Ok(None);
                    };
                    candidate_outputs.push(candidate);
                }
            }
            Ok(Some(candidate_outputs))
        })?;

    let mut live = LiveFoldDraw::<F, T>::new(transcript);
    for (prepared_group, output) in groups.iter().zip(candidate_outputs.iter_mut()) {
        let group = &prepared_group.input;
        let ring_d = group.params.inner_commit_matrix_params().ring_dimension();
        let challenges = live.draw_folding_challenges(
            ring_d,
            group.group_index,
            group.params.num_live_blocks(),
            group.group.num_polynomials(),
            &group.params.fold_challenge_config(),
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
/// Every preset probes `nonce = 0, 1, …` and commits the minimum accepting nonce.
/// When `tail_t_vectors` is set, the terminal response must fit the exact cap
/// and Golomb-Rice byte budget carried by its scheduled response shape.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_multi_group_fold_decompose_witnesses<F, E, G, B, T>(
    opening_ctx: &crate::compute::OperationCtx<'_, F, B>,
    transcript: &mut T,
    root_lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    groups: &[FoldGrindGroup<'_, '_, G>],
    _tail_t_vectors: Option<usize>,
) -> Result<(Vec<FoldGrindGroupOutput<F>>, u32), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: akita_types::FpExtEncoding<F>
        + akita_field::ExtField<F>
        + akita_serialization::AkitaSerialize,
    G: crate::protocol::core::RootProverGroupOpening<F, E, B>,
    B: crate::compute::ComputeBackendSetup<F> + crate::DigitRowsComputeBackend<F>,
    T: Transcript<F> + ProverTranscriptGrind<F>,
{
    let binding = FoldLinfProtocolBinding::CURRENT;
    if groups.len() != opening_batch.num_groups() {
        return Err(AkitaError::InvalidSetup(
            "fold grind groups do not match the opening batch".to_string(),
        ));
    }
    let mut prepared_groups = Vec::with_capacity(groups.len());
    for (expected_group_index, group) in groups.iter().enumerate() {
        let expected_claims = opening_batch
            .group_layout(expected_group_index)?
            .num_polynomials();
        if group.group_index != expected_group_index
            || group.group.num_polynomials() == 0
            || group.group.num_polynomials() != expected_claims
        {
            return Err(AkitaError::InvalidSetup(
                "fold grind group descriptor is malformed".to_string(),
            ));
        }
        let delta_fold = group.params.num_digits_fold();
        let (digit_negative_abs_bound, digit_positive_bound) =
            akita_types::sis::fold_witness_representable_linf_bounds(
                group.params.log_basis_open(),
                delta_fold,
            );
        prepared_groups.push(PreparedFoldGrindGroup {
            input: *group,
            acceptance: fold_grind_acceptance_ctx(digit_negative_abs_bound, digit_positive_bound),
        });
    }
    sample_multi_group_fold_decompose_witnesses_native::<F, E, G, B, T>(
        opening_ctx,
        transcript,
        root_lp,
        &prepared_groups,
        binding.max_grind_attempts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::CyclotomicRing;

    type F = akita_field::Prime128Offset275;

    #[test]
    fn joint_grind_skips_different_group_first_nonces() {
        let group_accepts = [[0, 2], [1, 2]];
        let mut probed = Vec::new();
        let (nonce, ()) = first_jointly_accepted_nonce(4, |nonce| {
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
    fn grind_rejects_chunk_payload_outside_digit_interval() {
        const D: usize = 4;
        let witness = DecomposeFoldWitness::from_parts::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            vec![[12; D]],
            12,
        );
        let chunks = vec![vec![[129, 0, 0, 0]], vec![[-12; D]]];
        let (neg_bound, pos_bound) = akita_types::sis::fold_witness_representable_linf_bounds(4, 2);
        let acceptance = fold_grind_acceptance_ctx(neg_bound, pos_bound);
        assert!(!accepts_fold_witness::<F, D>(
            &acceptance,
            &witness,
            &chunks
        ));
    }

    #[test]
    fn grind_rejects_positive_coefficients_past_balanced_digit_reach() {
        const D: usize = 4;
        let witness = DecomposeFoldWitness::from_parts::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            vec![[2022, 0, 0, 0]],
            2022,
        );
        let chunks = vec![witness.centered_coeffs_owned::<D>()];
        let (neg_bound, pos_bound) = akita_types::sis::fold_witness_representable_linf_bounds(6, 2);
        assert_eq!(neg_bound, 2080);
        assert_eq!(pos_bound, 2015);
        let acceptance = fold_grind_acceptance_ctx(neg_bound, pos_bound);
        assert!(!accepts_fold_witness::<F, D>(
            &acceptance,
            &witness,
            &chunks
        ));
    }

    #[test]
    fn digit_interval_accepts_both_endpoints_and_rejects_neighbors() {
        let (negative_abs, positive) =
            akita_types::sis::fold_witness_representable_linf_bounds(4, 2);
        let acceptance = fold_grind_acceptance_ctx(negative_abs, positive);
        let negative_abs = i32::try_from(negative_abs).unwrap();
        let positive = i32::try_from(positive).unwrap();

        assert!(coeff_within_digit_bounds(-negative_abs, &acceptance));
        assert!(coeff_within_digit_bounds(positive, &acceptance));
        assert!(!coeff_within_digit_bounds(-negative_abs - 1, &acceptance));
        assert!(!coeff_within_digit_bounds(positive + 1, &acceptance));
    }
}
