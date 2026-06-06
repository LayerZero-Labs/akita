//! Root and recursive level verifier replay for Akita proofs.
//!
//! This module owns the transcript and algebra checks for an already selected
//! root or fold level. Schedule/config dispatch stays with the scheme crate
//! until the verifier-facing config boundary is extracted.

use super::validate_level_dispatch;
#[cfg(not(feature = "zk"))]
mod extension_opening_reduction;
#[cfg(feature = "zk")]
mod zk;
use crate::protocol::ring_switch::{
    ring_switch_verifier, ring_switch_verifier_terminal, RingSwitchReplay,
};
use crate::stages::stage1::{derive_stage1_challenges, AkitaStage1Verifier};
use crate::stages::stage2::{AkitaStage2Verifier, Stage2RowEvalSource};
use crate::stages::SetupSumcheckVerifier;
use akita_algebra::CyclotomicRing;
#[cfg(feature = "zk")]
use akita_algebra::EqPolynomial;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    PseudoMersenneField, RandomSampling,
};
#[cfg(feature = "zk")]
use akita_r1cs::{
    lift_hiding_witness, zk_ext_mask_lc, zk_ext_mask_lc_at, zk_masked_compressed_round_claim_mask,
    zk_push_linear_zero, zk_row_masks_from_column_masks, ZkR1csLinearCombination,
    ZkRelationAccumulator,
};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::SumcheckInstanceVerifier;
#[cfg(not(feature = "zk"))]
use akita_sumcheck::SumcheckInstanceVerifierExt;
#[cfg(feature = "zk")]
use akita_sumcheck::ZkSumcheckInstanceVerifierExt;
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_STAGE2_NEXT_W_EVAL,
    ABSORB_SUMCHECK_S_CLAIM, ABSORB_TERMINAL_E_HAT, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_ROUND, CHALLENGE_TRACE_BATCH,
};
#[cfg(feature = "zk")]
use akita_transcript::labels::{ABSORB_SUMCHECK_CLAIM, ABSORB_ZK_HIDING_COMMITMENT};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
#[cfg(not(feature = "zk"))]
use akita_types::check_tensor_extension_opening_claim;
#[cfg(feature = "zk")]
use akita_types::EXTENSION_OPENING_REDUCTION_DEGREE;
use akita_types::{
    append_batched_commitments_to_transcript, append_claim_incidence_shape_to_transcript,
    append_claim_points_to_transcript, append_claim_values_to_transcript,
    flatten_batched_commitment_rows, generate_y, prepare_recursive_opening_point_ext,
    prepare_root_opening_point_ext, relation_claim_from_rows_extension, reorder_stage1_coords,
    ring_subfield_packed_extension_opening_point, root_extension_opening_partials,
    sample_public_row_coefficients, schedule_num_fold_levels, tensor_equality_factor_eval_at_point,
    tensor_reduction_claim_from_rows, tensor_row_partials_from_columns,
    terminal_witness_segment_layout, w_ring_element_count_with_counts, AkitaBatchedProof,
    AkitaLevelProof, AkitaProofStep, AkitaStage1Proof, AkitaStage2Proof, AkitaVerifierSetup,
    BasisMode, BlockOrder, ClaimIncidenceSummary, CleartextWitnessProof, CommitmentRouting,
    ExtensionOpeningReductionProof, FlatRingVec, LevelParams, MRowLayout, RingCommitment,
    RingOpeningPoint, RingRelationInstance, RingSubfieldEncoding, Schedule, SetupContributionMode,
    SetupSumcheckProof, Step, TerminalLevelProof, TerminalWitnessSegmentLayout,
    TerminalWitnessTranscriptParts,
};
use akita_types::{
    trace_input_claim, trace_stage2_enabled, trace_stage2_opening_owned_root_terms,
    trace_weight_layout_from_segment, TraceStage2Wire,
};
#[cfg(not(feature = "zk"))]
use extension_opening_reduction::verify_extension_opening_reduction_sumcheck;
#[cfg(feature = "zk")]
use zk::verify_zk_hiding_commitment;

mod recursive;

pub(crate) use recursive::verify_fold_batched_proof;

fn stage3_sumcheck_proof_for_mode<L: FieldCore>(
    mode: SetupContributionMode,
    stage3_sumcheck_proof: Option<&SetupSumcheckProof<L>>,
) -> Result<Option<&SetupSumcheckProof<L>>, AkitaError> {
    match (mode, stage3_sumcheck_proof) {
        (SetupContributionMode::Direct, None) => Ok(None),
        (SetupContributionMode::Direct, Some(_)) => Err(AkitaError::InvalidSetup(
            "direct setup-contribution mode received stage3_sumcheck_proof".to_string(),
        )),
        (SetupContributionMode::Recursive, Some(proof)) => Ok(Some(proof)),
        (SetupContributionMode::Recursive, None) => Err(AkitaError::InvalidSetup(
            "recursive setup-contribution mode is missing stage3_sumcheck_proof".to_string(),
        )),
    }
}

/// Verifier state carried between recursive fold levels.
pub(crate) struct RecursiveVerifierState<'a, F: FieldCore, L: FieldCore> {
    /// Current opening point for the committed recursive witness.
    pub opening_point: Vec<L>,
    /// Claimed opening value for the current commitment.
    pub opening: L,
    /// Hidden mask added to `opening` in the public proof.
    #[cfg(feature = "zk")]
    pub opening_mask: ZkR1csLinearCombination<L>,
    /// Current recursive witness commitment.
    pub commitment: &'a FlatRingVec<F>,
    /// Basis used to interpret the current opening point.
    pub basis: BasisMode,
    /// Current recursive witness length in field elements.
    pub w_len: usize,
    /// Current digit basis, as `log2(b)`.
    pub log_basis: u32,
}

#[cfg(feature = "zk")]
struct ZkMaskedClaim<E: FieldCore> {
    public: E,
    mask: ZkR1csLinearCombination<E>,
}

struct TerminalWitnessReplay {
    parts: TerminalWitnessTranscriptParts,
}

#[cfg(feature = "zk")]
#[allow(clippy::too_many_arguments)]
fn verify_zk_extension_opening_reduction_sumcheck<F, E, T, S>(
    input_claim: E,
    num_rounds: usize,
    proof: &akita_sumcheck::SumcheckProofMasked<E>,
    input_claim_mask: ZkR1csLinearCombination<E>,
    transcript: &mut T,
    mut sample_challenge: S,
    hiding_cursor: &mut usize,
) -> Result<(ZkMaskedClaim<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
    S: FnMut(&mut T) -> E,
{
    if proof.masked_round_polys.len() != num_rounds {
        return Err(AkitaError::InvalidSize {
            expected: num_rounds,
            actual: proof.masked_round_polys.len(),
        });
    }

    transcript.append_serde(ABSORB_SUMCHECK_CLAIM, &input_claim);
    let mut masked_claim = input_claim;
    let mut claim_mask = input_claim_mask;
    let mut challenges = Vec::with_capacity(num_rounds);
    for masked_poly in &proof.masked_round_polys {
        if masked_poly.degree() > EXTENSION_OPENING_REDUCTION_DEGREE {
            return Err(AkitaError::InvalidInput(format!(
                "extension-opening reduction round poly exceeds degree bound {}",
                EXTENSION_OPENING_REDUCTION_DEGREE
            )));
        }
        transcript.append_serde(akita_transcript::labels::ABSORB_SUMCHECK_ROUND, masked_poly);
        let r_i = sample_challenge(transcript);
        challenges.push(r_i);
        let next_claim_mask = zk_masked_compressed_round_claim_mask::<F, E>(
            &claim_mask,
            &masked_poly.coeffs_except_linear_term,
            r_i,
            hiding_cursor,
        );
        masked_claim = masked_poly.eval_from_hint(&masked_claim, &r_i);
        claim_mask = next_claim_mask;
    }

    Ok((
        ZkMaskedClaim {
            public: masked_claim,
            mask: claim_mask,
        },
        challenges,
    ))
}

fn prepare_terminal_witness_replay<F, T>(
    transcript: &mut T,
    final_witness: &CleartextWitnessProof<F>,
    final_w_len: usize,
    layout: TerminalWitnessSegmentLayout,
) -> Result<TerminalWitnessReplay, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    if final_witness.num_elems() != final_w_len {
        return Err(AkitaError::InvalidProof);
    }
    let parts = final_witness.terminal_transcript_parts(layout)?;
    transcript.record_wire_bytes(ABSORB_TERMINAL_E_HAT, &parts.e_hat);
    transcript.append_bytes(ABSORB_TERMINAL_E_HAT, &parts.e_hat);
    Ok(TerminalWitnessReplay { parts })
}

/// Verify the intermediate-root proof payload for batched proofs whose root
/// is followed by additional recursive levels.
///
/// This replays the canonical root transcript layout: batch-shape header,
/// commitments, padded opening points, per-claim field openings, one gamma
/// challenge per claim, and gamma-combined per-point y-rings, then runs the
/// stage-1 norm-check sumcheck and the stage-2 fused sumcheck, threading
/// `next_w_commitment` through `ABSORB_NEXT_LEVEL_WITNESS_BINDING`.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, any public trace check
/// fails, ring-switch replay fails, or either sumcheck verifier rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn verify_intermediate_root_level<F, E, C, T, const D: usize>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<C>>,
    v_flat: &FlatRingVec<F>,
    stage1: &AkitaStage1Proof<C>,
    stage2: &AkitaStage2Proof<F, C>,
    stage3_sumcheck_proof: Option<&SetupSumcheckProof<C>>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claim_points: &[&[E]],
    openings: &[E],
    commitments: &[RingCommitment<F, D>],
    incidence_summary: &ClaimIncidenceSummary,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<C>,
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
) -> Result<Vec<C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
{
    let stage3_sumcheck_proof =
        stage3_sumcheck_proof_for_mode(setup_contribution_mode, stage3_sumcheck_proof)?;
    verify_root_level_inner::<F, E, C, T, D>(
        extension_opening_reduction,
        Some(v_flat),
        Some(stage1),
        #[cfg(not(feature = "zk"))]
        &stage2.sumcheck_proof,
        #[cfg(feature = "zk")]
        &stage2.sumcheck_proof_masked,
        stage3_sumcheck_proof,
        setup,
        transcript,
        claim_points,
        openings,
        commitments,
        incidence_summary,
        basis,
        root_lp,
        batched_lp,
        RootStageInput::Intermediate {
            next_w_commitment: &stage2.next_w_commitment,
            next_w_eval: stage2.next_w_eval(),
        },
        #[cfg(feature = "zk")]
        zk_hiding_cursor,
        #[cfg(feature = "zk")]
        zk_relations,
    )
}

/// Verify the terminal-root proof payload — used when the schedule contains a
/// single fold level (the root is itself the terminal fold and ships
/// `final_witness` in cleartext).
///
/// Mirrors [`verify_intermediate_root_level`] up through the ring-switch
/// preamble; at the terminal, [`ABSORB_NEXT_LEVEL_WITNESS_BINDING`] absorbs `final_witness`
/// instead of a next-witness commitment, stage-1 is skipped entirely, and
/// stage-2 runs in relation-only mode (`batching_coeff = 0`, `s_claim = 0`).
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, any public trace
/// check fails, ring-switch replay fails, or the stage-2 sumcheck rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn verify_terminal_root_level<F, E, C, T, const D: usize>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<C>>,
    #[cfg(not(feature = "zk"))] stage2_sumcheck: &akita_sumcheck::SumcheckProof<C>,
    #[cfg(feature = "zk")] stage2_sumcheck_masked: &akita_sumcheck::SumcheckProofMasked<C>,
    final_witness: &CleartextWitnessProof<F>,
    final_w_len: usize,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claim_points: &[&[E]],
    openings: &[E],
    commitments: &[RingCommitment<F, D>],
    incidence_summary: &ClaimIncidenceSummary,
    basis: BasisMode,
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<C>,
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
) -> Result<Vec<C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
{
    verify_root_level_inner::<F, E, C, T, D>(
        extension_opening_reduction,
        None,
        None,
        #[cfg(not(feature = "zk"))]
        stage2_sumcheck,
        #[cfg(feature = "zk")]
        stage2_sumcheck_masked,
        None,
        setup,
        transcript,
        claim_points,
        openings,
        commitments,
        incidence_summary,
        basis,
        root_lp,
        batched_lp,
        RootStageInput::Terminal {
            final_witness,
            final_w_len,
        },
        #[cfg(feature = "zk")]
        zk_hiding_cursor,
        #[cfg(feature = "zk")]
        zk_relations,
    )
}

enum RootStageInput<'a, F: FieldCore, C: FieldCore> {
    Intermediate {
        next_w_commitment: &'a FlatRingVec<F>,
        next_w_eval: C,
    },
    Terminal {
        final_witness: &'a CleartextWitnessProof<F>,
        final_w_len: usize,
    },
}

#[allow(clippy::too_many_arguments)]
fn verify_root_level_inner<F, E, C, T, const D: usize>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<C>>,
    v_flat: Option<&FlatRingVec<F>>,
    stage1: Option<&AkitaStage1Proof<C>>,
    #[cfg(not(feature = "zk"))] stage2_sumcheck: &akita_sumcheck::SumcheckProof<C>,
    #[cfg(feature = "zk")] stage2_sumcheck_masked: &akita_sumcheck::SumcheckProofMasked<C>,
    stage3_sumcheck_proof: Option<&SetupSumcheckProof<C>>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claim_points: &[&[E]],
    openings: &[E],
    commitments: &[RingCommitment<F, D>],
    incidence_summary: &ClaimIncidenceSummary,
    basis: BasisMode,
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
    stage_input: RootStageInput<'_, F, C>,
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<C>,
) -> Result<Vec<C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
{
    validate_level_dispatch::<D>(root_lp)?;
    let is_terminal = matches!(stage_input, RootStageInput::Terminal { .. });
    let final_w_len_opt = match &stage_input {
        RootStageInput::Terminal { final_w_len, .. } => Some(*final_w_len),
        RootStageInput::Intermediate { .. } => None,
    };
    let v_typed_owned: Vec<CyclotomicRing<F, D>>;
    let v_typed: &[CyclotomicRing<F, D>] = match v_flat {
        Some(v_flat) => v_flat.as_ring_slice::<D>()?,
        None => {
            v_typed_owned = Vec::new();
            &v_typed_owned
        }
    };
    let num_claims = incidence_summary.num_claims();
    let num_points = incidence_summary.num_points();
    if num_points == 0
        || num_points != incidence_summary.num_points()
        || claim_points.len() != incidence_summary.num_points()
        || openings.len() != num_claims
        || commitments.len() != incidence_summary.num_points()
        || incidence_summary.claim_to_point().len() != num_claims
        || incidence_summary.claim_poly_indices().len() != num_claims
    {
        return Err(AkitaError::InvalidProof);
    }
    if incidence_summary
        .claim_to_point()
        .iter()
        .any(|&point_idx| point_idx >= num_points)
    {
        return Err(AkitaError::InvalidProof);
    }
    if commitments
        .iter()
        .any(|commitment| commitment.u.len() != root_lp.b_key.row_len())
    {
        return Err(AkitaError::InvalidProof);
    }
    // Mirror the prover's commitment-rows optimization: avoid a clone when
    // there is only a single commitment.
    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    append_claim_incidence_shape_to_transcript::<F, T>(incidence_summary, transcript)?;
    append_batched_commitments_to_transcript(commitments, transcript);
    append_claim_points_to_transcript::<F, E, T>(claim_points, transcript);
    append_claim_values_to_transcript::<F, E, T>(openings, transcript);
    let row_coefficients =
        sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;

    let alpha_bits = root_lp.ring_dimension.trailing_zeros() as usize;
    #[cfg(feature = "zk")]
    let mut zk_eor_final: Option<(ZkMaskedClaim<C>, Vec<C>)> = None;
    #[cfg(not(feature = "zk"))]
    let mut eor_trace_final: Option<(C, Vec<C>)> = None;
    let reduction_check = if let Some(reduction) = extension_opening_reduction {
        if <C as ExtField<F>>::EXT_DEGREE == 1 {
            return Err(AkitaError::InvalidProof);
        }
        if <C as ExtField<F>>::EXT_DEGREE != <E as ExtField<F>>::EXT_DEGREE {
            return Err(AkitaError::InvalidProof);
        }
        let (split_bits, width) = {
            let width = <C as ExtField<F>>::EXT_DEGREE;
            if width == 0 || !width.is_power_of_two() {
                return Err(AkitaError::InvalidProof);
            }
            (width.trailing_zeros() as usize, width)
        };
        let expected_partials =
            root_extension_opening_partials(width, incidence_summary.num_claims());
        if split_bits > incidence_summary.num_vars()
            || reduction.partials.len() != expected_partials
        {
            return Err(AkitaError::InvalidProof);
        }
        let padded_points = claim_points
            .iter()
            .map(|point| {
                if point.len() > incidence_summary.num_vars() {
                    return Err(AkitaError::InvalidProof);
                }
                let mut lifted = point.iter().copied().map(C::lift_base).collect::<Vec<_>>();
                lifted.resize(incidence_summary.num_vars(), C::zero());
                Ok(lifted)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut input_claim = C::zero();
        #[cfg(feature = "zk")]
        let mut input_claim_mask = ZkR1csLinearCombination::zero();
        for (claim_idx, opening) in openings
            .iter()
            .copied()
            .enumerate()
            .take(incidence_summary.num_claims())
        {
            let point_idx = incidence_summary.claim_to_point()[claim_idx];
            let partial_start = claim_idx * width;
            let partial_end = partial_start + width;
            let partials = &reduction.partials[partial_start..partial_end];
            #[cfg(feature = "zk")]
            let partial_masks = (0..width)
                .map(|_| zk_ext_mask_lc::<F, C>(zk_hiding_cursor))
                .collect::<Vec<_>>();
            #[cfg(not(feature = "zk"))]
            check_tensor_extension_opening_claim::<F, C>(
                &padded_points[point_idx],
                C::lift_base(opening),
                partials,
            )?;
            #[cfg(feature = "zk")]
            {
                let head_weights =
                    EqPolynomial::<C>::evals(&padded_points[point_idx][..split_bits])?;
                let mut residual = ZkR1csLinearCombination::constant(-C::lift_base(opening));
                for ((&partial, mask), weight) in
                    partials.iter().zip(partial_masks.iter()).zip(head_weights)
                {
                    let true_partial = ZkRelationAccumulator::unmask_lc(partial, mask);
                    residual.add_scaled(weight, &true_partial);
                }
                zk_relations.push_r1cs(
                    "root extension-opening partial claim",
                    residual,
                    ZkR1csLinearCombination::one(),
                    ZkR1csLinearCombination::zero(),
                )?;
            }
            for partial in partials {
                append_ext_field::<F, C, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
            }
        }
        let eta = (0..split_bits)
            .map(|_| sample_ext_challenge::<F, C, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
            .collect::<Vec<_>>();
        for (claim_idx, &row_coefficient) in row_coefficients
            .iter()
            .enumerate()
            .take(incidence_summary.num_claims())
        {
            let partial_start = claim_idx * width;
            let partial_end = partial_start + width;
            let row_partials = tensor_row_partials_from_columns::<F, C>(
                &reduction.partials[partial_start..partial_end],
            )?;
            let claim = tensor_reduction_claim_from_rows::<F, C>(&row_partials, &eta)?;
            input_claim += row_coefficient * claim;
            #[cfg(feature = "zk")]
            {
                let mut partial_masks = Vec::with_capacity(width);
                for offset in 0..width {
                    let mask_start = partial_start + offset;
                    let mask = zk_ext_mask_lc_at::<F, C>(
                        *zk_hiding_cursor
                            - reduction.partials.len() * <C as ExtField<F>>::EXT_DEGREE
                            + mask_start * <C as ExtField<F>>::EXT_DEGREE,
                    );
                    partial_masks.push(mask);
                }
                let row_masks = zk_row_masks_from_column_masks::<F, C>(&partial_masks)?;
                for (weight, row_mask) in EqPolynomial::<C>::evals(&eta)?.into_iter().zip(row_masks)
                {
                    input_claim_mask.add_scaled(row_coefficient * weight, &row_mask);
                }
            }
        }
        #[cfg(not(feature = "zk"))]
        {
            let (final_claim, rho) = verify_extension_opening_reduction_sumcheck::<F, T, C, _>(
                input_claim,
                incidence_summary.num_vars() - split_bits,
                &reduction.sumcheck,
                transcript,
                |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
            )?;
            let factors_by_point = padded_points
                .iter()
                .map(|point| {
                    tensor_equality_factor_eval_at_point::<F, C>(&point[split_bits..], &eta, &rho)
                })
                .collect::<Result<Vec<_>, _>>()?;
            eor_trace_final = Some((final_claim, factors_by_point));
            Some(rho)
        }
        #[cfg(feature = "zk")]
        {
            let (final_claim_lc, challenges) =
                verify_zk_extension_opening_reduction_sumcheck::<F, C, T, _>(
                    input_claim,
                    incidence_summary.num_vars() - split_bits,
                    &reduction.sumcheck_proof_masked,
                    input_claim_mask,
                    transcript,
                    |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                    zk_hiding_cursor,
                )?;
            let factors_by_point = padded_points
                .iter()
                .map(|point| {
                    tensor_equality_factor_eval_at_point::<F, C>(
                        &point[split_bits..],
                        &eta,
                        &challenges,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            zk_eor_final = Some((final_claim_lc, factors_by_point));
            Some(challenges)
        }
    } else {
        None
    };

    let prepared_points = if let Some(rho) = &reduction_check {
        let protocol_point =
            ring_subfield_packed_extension_opening_point::<F, C, D>(rho.len(), rho)?;
        let prepared = prepare_root_opening_point_ext::<F, C, C, D>(
            &protocol_point,
            basis,
            root_lp,
            alpha_bits,
        )?;
        vec![prepared; incidence_summary.num_points()]
    } else {
        claim_points
            .iter()
            .map(|opening_point| {
                prepare_root_opening_point_ext::<F, E, C, D>(
                    opening_point,
                    basis,
                    root_lp,
                    alpha_bits,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let gamma_tr: C = sample_ext_challenge::<F, C, T>(transcript, CHALLENGE_TRACE_BATCH);
    let mut batched_openings_per_row: Vec<C> = vec![C::zero(); incidence_summary.num_public_rows()];
    for (row_idx, row) in incidence_summary.public_rows().iter().enumerate() {
        if row.point_idx() >= prepared_points.len() || row.claim_indices().is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        for &claim_idx in row.claim_indices() {
            if claim_idx >= openings.len()
                || incidence_summary.claim_to_point()[claim_idx] != row.point_idx()
            {
                return Err(AkitaError::InvalidProof);
            }
            batched_openings_per_row[row_idx] +=
                row_coefficients[claim_idx] * C::lift_base(openings[claim_idx]);
        }
    }
    let w_len = if is_terminal {
        final_w_len_opt.ok_or(AkitaError::InvalidProof)?
    } else {
        w_ring_element_count_with_counts::<F>(
            batched_lp,
            incidence_summary.num_polys_per_point().len(),
            incidence_summary.num_polys_per_point().iter().sum(),
            num_claims,
            incidence_summary.num_public_rows(),
        )?
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("next witness length overflow".to_string()))?
    };
    let terminal_replay = if let RootStageInput::Terminal { final_witness, .. } = &stage_input {
        let layout = terminal_witness_segment_layout(
            batched_lp,
            num_claims,
            incidence_summary.num_public_rows(),
            F::modulus_bits(),
        )?;
        Some(prepare_terminal_witness_replay::<F, T>(
            transcript,
            final_witness,
            w_len,
            layout,
        )?)
    } else {
        None
    };

    let stage1_challenges = derive_stage1_challenges::<F, T, D>(
        transcript,
        v_typed,
        root_lp.num_blocks,
        num_claims,
        batched_lp,
        if is_terminal {
            MRowLayout::WithoutDBlock
        } else {
            MRowLayout::WithDBlock
        },
    )?;

    let ring_opening_points: Vec<RingOpeningPoint<F>> = incidence_summary
        .public_rows()
        .iter()
        .map(|row| prepared_points[row.point_idx()].ring_opening_point.clone())
        .collect();
    let ring_multiplier_points: Vec<_> = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points[row.point_idx()]
                .ring_multiplier_point
                .clone()
        })
        .collect();
    let m_row_layout = if is_terminal {
        MRowLayout::WithoutDBlock
    } else {
        MRowLayout::WithDBlock
    };
    let commitment_routing = CommitmentRouting::from_root_incidence(incidence_summary)?;
    let (gamma, row_coefficient_rings) =
        RingRelationInstance::<F, D>::gamma_and_row_rings_from_coefficients::<C>(
            &row_coefficients,
        )?;
    let (y_v_slice, n_d_active) = match m_row_layout {
        MRowLayout::WithDBlock => (v_typed, batched_lp.d_key.row_len()),
        MRowLayout::WithoutDBlock => (&[][..], 0usize),
    };
    let relation_y = generate_y::<F, D>(
        y_v_slice,
        commitment_rows,
        n_d_active,
        batched_lp.b_key.row_len(),
        batched_lp.a_key.row_len(),
    )?;
    let relation_instance = RingRelationInstance::new(
        m_row_layout,
        stage1_challenges.clone(),
        ring_opening_points.clone(),
        ring_multiplier_points.clone(),
        incidence_summary.clone(),
        commitment_routing,
        gamma,
        row_coefficient_rings,
        relation_y,
        v_typed.to_vec(),
    )?;
    relation_instance.check_v_shape_for_level(batched_lp)?;
    let ring_switch_replay = RingSwitchReplay {
        relation: &relation_instance,
        row_coefficients: &row_coefficients,
        lp: batched_lp,
    };
    let rs = match &stage_input {
        RootStageInput::Intermediate {
            next_w_commitment, ..
        } => ring_switch_verifier::<F, C, T, D>(
            &ring_switch_replay,
            w_len,
            next_w_commitment,
            transcript,
        )?,
        RootStageInput::Terminal { .. } => {
            let replay = terminal_replay.as_ref().ok_or(AkitaError::InvalidProof)?;
            ring_switch_verifier_terminal::<F, C, T, D>(
                &ring_switch_replay,
                w_len,
                transcript,
                &replay.parts,
            )?
        }
    };
    let relation_claim = relation_claim_from_rows_extension::<F, C, D>(
        &rs.tau1,
        rs.alpha,
        v_typed,
        commitment_rows,
    )?;
    #[cfg(feature = "zk")]
    let relation_claim_mask = ZkR1csLinearCombination::<C>::zero();
    #[cfg(feature = "zk")]
    let mut s_claim_mask = ZkR1csLinearCombination::<C>::zero();
    let (batching_coeff, s_claim, stage1_point) = match (&stage_input, stage1) {
        (RootStageInput::Intermediate { .. }, Some(stage1_proof)) => {
            let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
            let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
            #[cfg(not(feature = "zk"))]
            let stage1_point = {
                let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
                stage1_verifier.verify::<F, T>(stage1_proof, transcript)?
            };
            #[cfg(feature = "zk")]
            let stage1_point = {
                let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
                let (r, mask) = stage1_verifier.verify::<F, T>(
                    stage1_proof,
                    transcript,
                    zk_relations,
                    zk_hiding_cursor,
                )?;
                s_claim_mask = mask;
                r
            };
            transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1_proof.s_claim);
            let batching_coeff: C =
                sample_ext_challenge::<F, C, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
            (batching_coeff, stage1_proof.s_claim, stage1_point)
        }
        (RootStageInput::Terminal { .. }, None) => {
            // Relation-only stage-2: skip stage-1 entirely. Dummy zeros for
            // stage1_point + batching_coeff zero out the virtual half.
            let stage1_point = vec![C::zero(); rs.col_bits + rs.ring_bits];
            (C::zero(), C::zero(), stage1_point)
        }
        _ => return Err(AkitaError::InvalidProof),
    };
    let trace_wire = if !trace_stage2_enabled(
        batched_lp,
        <C as ExtField<F>>::EXT_DEGREE,
        reduction_check.is_some(),
    ) {
        None
    } else {
        let segment = relation_instance.segment_layout(batched_lp)?;
        let num_trace_blocks = incidence_summary
            .num_claims()
            .checked_mul(batched_lp.num_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("trace block count overflow".to_string()))?;
        let layout = trace_weight_layout_from_segment(
            batched_lp,
            &segment,
            rs.col_bits,
            rs.ring_bits,
            num_trace_blocks,
        )?;
        let ordinary_trace_opening = || {
            batched_openings_per_row
                .iter()
                .fold(C::zero(), |acc, opening| acc + *opening)
        };
        #[cfg(not(feature = "zk"))]
        let trace_opening = eor_trace_final
            .as_ref()
            .map(|(final_claim, _)| *final_claim)
            .unwrap_or_else(ordinary_trace_opening);
        #[cfg(feature = "zk")]
        let trace_opening = zk_eor_final
            .as_ref()
            .map(|(final_claim, _)| final_claim.public)
            .unwrap_or_else(ordinary_trace_opening);
        #[cfg(not(feature = "zk"))]
        let trace_claim_scales = eor_trace_final
            .as_ref()
            .map(|(_, factors_by_point)| {
                incidence_summary
                    .claim_to_point()
                    .iter()
                    .map(|&point_idx| {
                        factors_by_point
                            .get(point_idx)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        #[cfg(feature = "zk")]
        let trace_claim_scales = zk_eor_final
            .as_ref()
            .map(|(_, factors_by_point)| {
                incidence_summary
                    .claim_to_point()
                    .iter()
                    .map(|&point_idx| {
                        factors_by_point
                            .get(point_idx)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        Some(TraceStage2Wire {
            layout,
            gamma_tr,
            trace_opening_claim: trace_input_claim(gamma_tr, trace_opening),
            opening: trace_stage2_opening_owned_root_terms(
                batched_lp,
                incidence_summary,
                &prepared_points,
                &row_coefficients,
                trace_claim_scales.as_deref(),
            )?,
        })
    };
    #[cfg(feature = "zk")]
    let mut trace_claim_mask = ZkR1csLinearCombination::<C>::zero();
    #[cfg(feature = "zk")]
    if trace_wire.is_some() {
        if let Some((final_claim, _)) = &zk_eor_final {
            trace_claim_mask.add_scaled(gamma_tr, &final_claim.mask);
        }
    }
    let mut stage2_input_claim = batching_coeff * s_claim + relation_claim;
    if let Some(trace) = &trace_wire {
        stage2_input_claim += trace.trace_opening_claim;
    }
    let setup_prepared_row_eval = stage3_sumcheck_proof.map(|_| rs.prepared_row_eval.clone());
    let row_eval_source = if let Some(stage3_sumcheck_proof) = stage3_sumcheck_proof {
        Stage2RowEvalSource::new_with_setup_claim(rs.prepared_row_eval, stage3_sumcheck_proof.claim)
    } else {
        Stage2RowEvalSource::new(rs.prepared_row_eval)
    };
    #[cfg(feature = "zk")]
    let stage2_next_w_eval_mask_cursor =
        *zk_hiding_cursor + (rs.col_bits + rs.ring_bits) * 3 * <C as ExtField<F>>::EXT_DEGREE;
    let stage2_verifier = match &stage_input {
        RootStageInput::Terminal { final_witness, .. } => {
            AkitaStage2Verifier::new_with_cleartext_witness(
                batching_coeff,
                s_claim,
                #[cfg(feature = "zk")]
                s_claim_mask,
                #[cfg(feature = "zk")]
                relation_claim_mask,
                #[cfg(feature = "zk")]
                trace_claim_mask,
                final_witness,
                w_len,
                stage1_point,
                rs.alpha_evals_y,
                row_eval_source,
                &setup.expanded,
                &ring_opening_points,
                &ring_multiplier_points,
                &rs.tau1,
                v_typed,
                commitment_rows,
                Some(relation_claim),
                rs.alpha,
                rs.col_bits,
                rs.ring_bits,
                trace_wire,
            )?
        }
        RootStageInput::Intermediate { next_w_eval, .. } => {
            AkitaStage2Verifier::new_with_claimed_w_eval(
                batching_coeff,
                s_claim,
                #[cfg(feature = "zk")]
                s_claim_mask,
                #[cfg(feature = "zk")]
                relation_claim_mask,
                #[cfg(feature = "zk")]
                trace_claim_mask,
                *next_w_eval,
                #[cfg(feature = "zk")]
                zk_ext_mask_lc_at::<F, C>(stage2_next_w_eval_mask_cursor),
                stage1_point,
                rs.alpha_evals_y,
                row_eval_source,
                &setup.expanded,
                &ring_opening_points,
                &ring_multiplier_points,
                &rs.tau1,
                v_typed,
                commitment_rows,
                Some(relation_claim),
                rs.alpha,
                rs.col_bits,
                rs.ring_bits,
                trace_wire,
            )?
        }
    };
    if stage2_input_claim != SumcheckInstanceVerifier::input_claim(&stage2_verifier) {
        return Err(AkitaError::InvalidProof);
    }
    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        #[cfg(not(feature = "zk"))]
        {
            stage2_verifier.verify::<F, T, _>(stage2_sumcheck, transcript, |tr| {
                sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?
        }
        #[cfg(feature = "zk")]
        {
            let challenges = stage2_verifier.verify_zk::<F, T, _>(
                stage2_sumcheck_masked,
                transcript,
                zk_relations,
                zk_hiding_cursor,
                |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
            )?;
            if matches!(stage_input, RootStageInput::Intermediate { .. }) {
                *zk_hiding_cursor += <C as ExtField<F>>::EXT_DEGREE;
            }
            challenges
        }
    };
    if let RootStageInput::Intermediate { next_w_eval, .. } = &stage_input {
        transcript.record_wire_serde(ABSORB_STAGE2_NEXT_W_EVAL, next_w_eval);
        transcript.append_serde(ABSORB_STAGE2_NEXT_W_EVAL, next_w_eval);
    }
    if let Some(stage3_sumcheck_proof) = stage3_sumcheck_proof {
        let setup_prepared_row_eval = setup_prepared_row_eval
            .as_ref()
            .ok_or(AkitaError::InvalidProof)?;
        let verifier = SetupSumcheckVerifier::new::<F, D>(
            setup_prepared_row_eval,
            &sumcheck_challenges[rs.ring_bits..],
            rs.alpha,
        )?;
        verifier.verify::<F, T, D>(&setup.expanded, stage3_sumcheck_proof, transcript)?;
    }
    Ok(sumcheck_challenges)
}
