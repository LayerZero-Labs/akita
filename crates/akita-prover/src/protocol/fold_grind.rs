//! Fold-l∞ Fiat–Shamir grind: preview off-sponge clones, commit the winning nonce.

use crate::compute::{
    OpeningBatchKernel, OpeningFoldKernel, RootOpeningSource, RuntimeOpeningProveBackendFor,
    RuntimeOpeningSource,
};
use akita_challenges::{Challenges, FoldDraw, LiveFoldDraw, PreviewFoldDraw};
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_transcript::{AkitaTranscript, FoldChallengeSeedPreview, Transcript, TranscriptSponge};
use akita_types::{
    dyadic_block_ranges, golomb_rice_total_wire_bits, golomb_rice_values_within_cap,
    golomb_rice_zigzag_width, CommittedGroupParams, FoldLinfProtocolBinding,
    InnerCommitSecurityRoute, LevelParamsLike, OpeningClaimsLayout, TerminalCommittedGroupParams,
    TerminalResponseShape,
};

use super::ring_relation::{
    aggregate_decompose_fold_witnesses, build_point_decompose_fold_witness,
    window_sparse_challenges,
};
use super::ring_relation_witness::{CenteredFoldChunk, FoldChunkCoefficients};
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
    response_l2_sq_cap: Option<u128>,
}

#[inline]
fn response_model_diagnostics_enabled() -> bool {
    #[cfg(feature = "response-model-diagnostics")]
    {
        tracing::enabled!(
            target: "akita_prover::protocol::fold_response_model",
            tracing::Level::INFO
        )
    }
    #[cfg(not(feature = "response-model-diagnostics"))]
    {
        false
    }
}

fn fold_grind_acceptance_ctx(
    digit_negative_abs_bound: u128,
    digit_positive_bound: u128,
    response_l2_sq_cap: Option<u128>,
) -> FoldGrindAcceptanceCtx {
    FoldGrindAcceptanceCtx {
        digit_negative_abs_bound,
        digit_positive_bound,
        response_l2_sq_cap,
    }
}

fn coeff_within_digit_bounds(coeff: i32, ctx: &FoldGrindAcceptanceCtx) -> bool {
    if coeff < 0 {
        u128::from(coeff.unsigned_abs()) <= ctx.digit_negative_abs_bound
    } else {
        (coeff as u128) <= ctx.digit_positive_bound
    }
}

fn accepts_fold_witness_flat<F: CanonicalField>(
    ctx: &FoldGrindAcceptanceCtx,
    witness: &DecomposeFoldWitness<F>,
    coefficients: &FoldChunkCoefficients,
) -> Option<Option<u128>> {
    if !coefficients.all_extrema_within(witness, |min, max| {
        coeff_within_digit_bounds(min, ctx) && coeff_within_digit_bounds(max, ctx)
    }) {
        return None;
    }
    let measure_l2 = ctx.response_l2_sq_cap.is_some() || response_model_diagnostics_enabled();
    if !measure_l2 {
        return Some(None);
    }
    let mut response_l2_sq = 0u128;
    coefficients
        .try_for_each(
            witness.centered_coeffs_flat(),
            coefficients.num_chunks(),
            |chunk| {
                for &coefficient in chunk {
                    let magnitude = u128::from(coefficient.unsigned_abs());
                    response_l2_sq = magnitude
                        .checked_mul(magnitude)
                        .and_then(|square| response_l2_sq.checked_add(square))
                        .ok_or_else(|| {
                            AkitaError::InvalidInput("fold response L2 norm overflow".into())
                        })?;
                }
                Ok(())
            },
        )
        .ok()?;
    ctx.response_l2_sq_cap
        .is_none_or(|cap| response_l2_sq <= cap)
        .then_some(Some(response_l2_sq))
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

pub(crate) struct FoldProbeOutput<F: FieldCore> {
    pub(crate) witness: DecomposeFoldWitness<F>,
    pub(crate) coefficients: FoldChunkCoefficients,
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
    P: RuntimeOpeningSource<F> + crate::compute::RootPolyMeta<F>,
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
    let linf_cap = expected_group.z_linf_cap;
    params.validate_terminal_linf_cap(sparse, linf_cap)?;
    let response_l2_sq_cap = params.response_l2_sq_cap();
    let operator_rejection = if response_l2_sq_cap.is_some() {
        Some(
            akita_challenges::selective_l2_operator_norm_rejection(params.d_a(), sparse)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("unsupported terminal L2 challenge policy".into())
                })?,
        )
    } else {
        None
    };
    let binding = FoldLinfProtocolBinding::CURRENT;
    let polys = [poly];
    let point_indices = [0usize];
    let (nonce, (witness, challenges)) =
        first_jointly_accepted_nonce(binding.max_grind_attempts, |nonce| {
            let mut preview = PreviewFoldDraw::new(transcript);
            let challenges = preview.draw_folding_challenges_with_rejection(
                params.d_a(),
                0,
                params.num_live_blocks,
                1,
                sparse,
                nonce,
                operator_rejection,
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
            let centered = witness.centered_coeffs_flat();
            if let Some(cap) = linf_cap {
                if golomb_rice_values_within_cap(centered, cap).is_err() {
                    return Ok(None);
                }
            } else if centered.iter().any(|&value| i16::try_from(value).is_err()) {
                return Ok(None);
            }
            if response_l2_sq_cap.is_some_and(|cap| {
                akita_types::sis::checked_centered_l2_sq(centered).is_none_or(|norm| norm > cap)
            }) {
                return Ok(None);
            }
            let zigzag_width = golomb_rice_zigzag_width(linf_cap.unwrap_or(i16::MAX as u128));
            let wire_bits = golomb_rice_total_wire_bits(
                centered,
                expected_group.z_rice_low_bits,
                zigzag_width,
            )?;
            if wire_bits > expected_group.z_payload_bytes.saturating_mul(8) {
                return Ok(None);
            }
            Ok(Some((witness, challenges)))
        })?;
    let mut live = LiveFoldDraw::<F, T>::new(transcript);
    let live_challenges = live.draw_folding_challenges_with_rejection(
        params.d_a(),
        0,
        params.num_live_blocks,
        1,
        sparse,
        nonce,
        operator_rejection,
    )?;
    if live_challenges != challenges {
        return Err(AkitaError::InvalidInput(
            "terminal grind preview did not match live transcript replay".into(),
        ));
    }
    #[cfg(feature = "response-model-diagnostics")]
    if response_model_diagnostics_enabled() {
        let source_l2_sq = crate::compute::RootPolyMeta::exact_integer_coeff_l2_sq(poly);
        let conditional_mean_l2_sq =
            source_l2_sq.and_then(|energy| energy.checked_mul(sparse.challenge_l2_sq_max()));
        let response_l2_sq =
            akita_types::sis::checked_centered_l2_sq(witness.centered_coeffs_flat());
        tracing::info!(
            target: "akita_prover::protocol::fold_response_model",
            terminal = true,
            nonce,
            attempts = nonce + 1,
            ring_dimension = params.d_a(),
            num_live_blocks = params.num_live_blocks,
            num_positions_per_block = params.num_positions_per_block,
            response_coeffs = witness.centered_coeffs_flat().len(),
            log_basis_inner = params.log_basis_inner,
            num_digits_inner = params.num_digits_inner,
            challenge_weight = sparse.weight(),
            challenge_l1 = sparse.l1_norm(),
            challenge_l2_sq = sparse.challenge_l2_sq_max(),
            challenge_linf = sparse.infinity_norm(),
            source_l2_sq = ?source_l2_sq,
            conditional_mean_l2_sq = ?conditional_mean_l2_sq,
            response_l2_sq = ?response_l2_sq,
            response_l2_sq_cap = ?response_l2_sq_cap,
            "terminal fold response model sample"
        );
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
) -> Result<(DecomposeFoldWitness<F>, FoldChunkCoefficients), AkitaError>
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
        return Ok((witness, FoldChunkCoefficients::single()));
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
        .map(CenteredFoldChunk::from_witness)
        .collect();
    let global = aggregate_decompose_fold_witnesses::<F, D>(windows)?;
    Ok((global, FoldChunkCoefficients::chunked(per_chunk)?))
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
) -> Result<(Vec<FoldProbeOutput<F>>, u32), AkitaError>
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
    let (nonce, mut candidate_outputs) =
        first_jointly_accepted_nonce(max_grind_attempts, |nonce| {
            let mut candidate_outputs = Vec::with_capacity(groups.len());
            {
                let mut preview = PreviewFoldDraw::new(transcript);
                for prepared_group in groups {
                    let group = &prepared_group.input;
                    let ring_d = group.params.inner_commit_matrix_params().ring_dimension();
                    let challenge_config = group.params.fold_challenge_config();
                    let rejection = matches!(
                        group.params.inner_commit_matrix_params().security_route(),
                        InnerCommitSecurityRoute::L2 { .. }
                    )
                    .then(|| {
                        akita_challenges::selective_l2_operator_norm_rejection(
                            ring_d,
                            &challenge_config,
                        )
                    })
                    .flatten();
                    let challenges = preview.draw_folding_challenges_with_rejection(
                        ring_d,
                        group.group_index,
                        group.params.num_live_blocks(),
                        group.group.num_polynomials(),
                        &challenge_config,
                        nonce,
                        rejection,
                    )?;
                    let output =
                        group
                            .group
                            .probe_fold(opening_ctx, &challenges, root_lp, group.params)?;
                    let observed_l2_sq = {
                        let _span = tracing::info_span!("fold_grind_acceptance_check").entered();
                        accepts_fold_witness_flat(
                            &prepared_group.acceptance,
                            &output.witness,
                            &output.coefficients,
                        )
                    };
                    let Some(observed_l2_sq) = observed_l2_sq else {
                        return Ok(None);
                    };
                    candidate_outputs.push((output, observed_l2_sq));
                }
            }
            Ok(Some(candidate_outputs))
        })?;

    {
        let _span = tracing::info_span!("fold_grind_live_replay").entered();
        let mut live = LiveFoldDraw::<F, T>::new(transcript);
        for (prepared_group, (output, observed_l2_sq)) in
            groups.iter().zip(candidate_outputs.iter_mut())
        {
            let group = &prepared_group.input;
            let ring_d = group.params.inner_commit_matrix_params().ring_dimension();
            let challenge_config = group.params.fold_challenge_config();
            let rejection = matches!(
                group.params.inner_commit_matrix_params().security_route(),
                InnerCommitSecurityRoute::L2 { .. }
            )
            .then(|| {
                akita_challenges::selective_l2_operator_norm_rejection(ring_d, &challenge_config)
            })
            .flatten();
            let challenges = live.draw_folding_challenges_with_rejection(
                ring_d,
                group.group_index,
                group.params.num_live_blocks(),
                group.group.num_polynomials(),
                &challenge_config,
                nonce,
                rejection,
            )?;
            if challenges != output.challenges {
                return Err(AkitaError::InvalidInput(
                    "fold grind preview did not match live transcript replay".to_string(),
                ));
            }
            tracing::info!(
                group_index = group.group_index,
                nonce,
                attempts = nonce + 1,
                response_l2_sq = ?observed_l2_sq,
                response_l2_sq_cap = ?prepared_group.acceptance.response_l2_sq_cap,
                "selected physical fold response"
            );
            #[cfg(feature = "response-model-diagnostics")]
            if response_model_diagnostics_enabled() {
                if let Some(response_l2_sq) = *observed_l2_sq {
                    let source_l2_sq = group.group.exact_integer_coeff_l2_sq();
                    let conditional_mean_l2_sq = source_l2_sq.and_then(|energy| {
                        energy.checked_mul(challenge_config.challenge_l2_sq_max())
                    });
                    tracing::info!(
                        target: "akita_prover::protocol::fold_response_model",
                        group_index = group.group_index,
                        nonce,
                        attempts = nonce + 1,
                        ring_dimension = ring_d,
                        num_polynomials = group.group.num_polynomials(),
                        num_live_blocks = group.params.num_live_blocks(),
                        num_positions_per_block = group.params.num_positions_per_block(),
                        num_chunks = output.coefficients.num_chunks(),
                        response_coeffs = output.coefficients.coefficient_count(
                            output.witness.centered_coeffs_flat()
                        ),
                        log_basis_inner = group.params.log_basis_inner(),
                        num_digits_inner = group.params.num_digits_inner(),
                        log_basis_response = group.params.log_basis_open(),
                        num_digits_response = group.params.num_digits_fold(),
                        challenge_weight = challenge_config.weight(),
                        challenge_l1 = challenge_config.l1_norm(),
                        challenge_l2_sq = challenge_config.challenge_l2_sq_max(),
                        challenge_linf = challenge_config.infinity_norm(),
                        source_l2_sq = ?source_l2_sq,
                        conditional_mean_l2_sq = ?conditional_mean_l2_sq,
                        response_l2_sq,
                        response_l2_sq_cap = ?prepared_group.acceptance.response_l2_sq_cap,
                        "fold response model sample"
                    );
                }
            }
        }
    }
    Ok((
        candidate_outputs
            .into_iter()
            .map(|(output, _)| output)
            .collect(),
        nonce,
    ))
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
) -> Result<(Vec<FoldProbeOutput<F>>, u32), AkitaError>
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
            akita_types::sis::balanced_digit_representable_bounds(
                group.params.log_basis_open(),
                delta_fold,
            );
        let response_l2_sq_cap = match group.params.inner_commit_matrix_params().security_route() {
            InnerCommitSecurityRoute::Linf(_) => None,
            InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap, ..
            } => {
                if groups.len() != 1 || group.group_index != 0 {
                    return Err(AkitaError::InvalidSetup(
                        "L2 fold grinding requires one scalar group".into(),
                    ));
                }
                Some(response_l2_sq_cap)
            }
        };
        prepared_groups.push(PreparedFoldGrindGroup {
            input: *group,
            acceptance: fold_grind_acceptance_ctx(
                digit_negative_abs_bound,
                digit_positive_bound,
                response_l2_sq_cap,
            ),
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
    use akita_challenges::SparseChallenge;

    type F = akita_field::Prime128Offset275;

    #[test]
    fn empty_chunk_window_has_zero_fold_challenges() {
        let challenges = Challenges::from_sparse(
            vec![
                SparseChallenge {
                    positions: vec![0].into(),
                    coeffs: vec![1].into(),
                };
                4
            ],
            4,
            1,
        )
        .expect("challenges");
        let empty = window_sparse_challenges(&challenges, 2..2).expect("empty window");

        assert!(empty
            .as_slice()
            .iter()
            .all(|challenge| challenge.positions.is_empty() && challenge.coeffs.is_empty()));
    }

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
        );
        let rejected_chunk = DecomposeFoldWitness::from_parts::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            vec![[129, 0, 0, 0]],
        );
        let accepted_chunk = DecomposeFoldWitness::from_parts::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            vec![[-12; D]],
        );
        let chunks = vec![
            CenteredFoldChunk::from_witness(&rejected_chunk),
            CenteredFoldChunk::from_witness(&accepted_chunk),
        ];
        let (neg_bound, pos_bound) = akita_types::sis::balanced_digit_representable_bounds(4, 2);
        let acceptance = fold_grind_acceptance_ctx(neg_bound, pos_bound, None);
        assert!(accepts_fold_witness_flat(
            &acceptance,
            &witness,
            &FoldChunkCoefficients::chunked(chunks).unwrap()
        )
        .is_none());
    }

    #[test]
    fn distributed_fold_chunk_state_rejects_empty_and_singleton_sets() {
        const D: usize = 4;
        let witness = DecomposeFoldWitness::from_parts::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            vec![[0; D]],
        );

        assert!(FoldChunkCoefficients::chunked(Vec::new()).is_err());
        assert!(
            FoldChunkCoefficients::chunked(vec![CenteredFoldChunk::from_witness(&witness)])
                .is_err()
        );
    }

    #[test]
    fn grind_rejects_positive_coefficients_past_balanced_digit_reach() {
        const D: usize = 4;
        let witness = DecomposeFoldWitness::from_parts::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            vec![[2022, 0, 0, 0]],
        );
        let (neg_bound, pos_bound) = akita_types::sis::balanced_digit_representable_bounds(6, 2);
        assert_eq!(neg_bound, 2080);
        assert_eq!(pos_bound, 2015);
        let acceptance = fold_grind_acceptance_ctx(neg_bound, pos_bound, None);
        assert!(
            accepts_fold_witness_flat(&acceptance, &witness, &FoldChunkCoefficients::single())
                .is_none()
        );
    }

    #[test]
    fn digit_interval_accepts_both_endpoints_and_rejects_neighbors() {
        let (negative_abs, positive) = akita_types::sis::balanced_digit_representable_bounds(4, 2);
        let acceptance = fold_grind_acceptance_ctx(negative_abs, positive, None);
        let negative_abs = i32::try_from(negative_abs).unwrap();
        let positive = i32::try_from(positive).unwrap();

        assert!(coeff_within_digit_bounds(-negative_abs, &acceptance));
        assert!(coeff_within_digit_bounds(positive, &acceptance));
        assert!(!coeff_within_digit_bounds(-negative_abs - 1, &acceptance));
        assert!(!coeff_within_digit_bounds(positive + 1, &acceptance));
    }

    #[test]
    fn fold_witness_records_signed_extrema_for_constant_time_acceptance() {
        const D: usize = 4;
        let witness = DecomposeFoldWitness::from_parts::<D>(
            vec![CyclotomicRing::<F, D>::zero(); 2],
            vec![[-2_080, 17, 0, 2_015], [-3, 4, 9, -11]],
        );

        assert_eq!(witness.centered_signed_extrema(), (-2_080, 2_015));
        assert_eq!(witness.centered_inf_norm(), 2_080);

        let acceptance = fold_grind_acceptance_ctx(2_080, 2_015, None);
        assert!(
            accepts_fold_witness_flat(&acceptance, &witness, &FoldChunkCoefficients::single())
                .is_some()
        );
    }

    #[test]
    fn l2_acceptance_checks_complete_retained_response() {
        const D: usize = 4;
        let witness = DecomposeFoldWitness::from_parts::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            vec![[3, 4, 0, 0]],
        );
        let coefficients = FoldChunkCoefficients::single();
        let acceptance = fold_grind_acceptance_ctx(8, 7, Some(25));
        assert_eq!(
            accepts_fold_witness_flat(&acceptance, &witness, &coefficients),
            Some(Some(25))
        );

        let too_small = fold_grind_acceptance_ctx(8, 7, Some(24));
        assert!(accepts_fold_witness_flat(&too_small, &witness, &coefficients).is_none());
    }
}
