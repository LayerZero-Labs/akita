//! Root and recursive level verifier replay for Akita proofs.
//!
//! This module owns the transcript and algebra checks for an already selected
//! root or fold level. Schedule/config dispatch stays with the scheme crate
//! until the verifier-facing config boundary is extracted.

use super::validate_level_dispatch;
#[cfg(feature = "zk")]
use crate::protocol::batched::{
    append_direct_blinding, direct_decomposed_inner_rows, field_evals_to_rings,
    mat_vec_mul_i8_plain,
};
use crate::protocol::ring_switch::{ring_switch_verifier, ring_switch_verifier_terminal};
use crate::stages::stage1::{derive_stage1_challenges, AkitaStage1Verifier};
use crate::stages::stage2::{AkitaStage2Verifier, Stage2RowEvalSource};
use akita_algebra::CyclotomicRing;
#[cfg(feature = "zk")]
use akita_algebra::EqPolynomial;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    PseudoMersenneField, RandomSampling,
};
#[cfg(feature = "zk")]
use akita_r1cs::{
    lift_hiding_witness, zk_base_mask_lcs, zk_ext_mask_lc, zk_ext_mask_lc_at,
    zk_masked_linear_value_lc, zk_push_linear_zero, zk_relation_claim_mask_from_y_masks,
    zk_row_masks_from_column_masks, ZkR1csLinearCombination, ZkRelationAccumulator,
};
use akita_serialization::AkitaSerialize;
#[cfg(feature = "zk")]
use akita_sumcheck::ZkSumcheckInstanceVerifierExt;
#[cfg(not(feature = "zk"))]
use akita_sumcheck::{
    check_extension_opening_reduction_output, check_tensor_extension_opening_claim,
    SumcheckInstanceVerifierExt,
};
use akita_sumcheck::{
    tensor_equality_factor_eval_at_point, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, ExtensionOpeningReductionSumcheck, SumcheckInstanceVerifier,
};
#[cfg(feature = "zk")]
use akita_transcript::labels::ABSORB_ZK_HIDING_COMMITMENT;
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_STAGE2_NEXT_W_EVAL,
    ABSORB_SUMCHECK_S_CLAIM, ABSORB_TERMINAL_W_HAT, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
#[cfg(not(feature = "zk"))]
use akita_types::dispatch_trace_inner_product_check;
#[cfg(feature = "zk")]
use akita_types::ZkHidingProof;
use akita_types::{
    append_batched_commitments_to_transcript, append_claim_incidence_shape_to_transcript,
    append_claim_points_to_transcript, append_claim_values_to_transcript,
    flatten_batched_commitment_rows, prepare_recursive_opening_point_ext,
    prepare_root_opening_point_ext, recover_ring_subfield_inner_product,
    relation_claim_from_rows_extension, reorder_stage1_coords,
    ring_subfield_packed_extension_opening_point, root_extension_opening_partials,
    sample_public_row_coefficients, schedule_num_fold_levels, terminal_witness_segment_layout,
    w_ring_element_count_with_counts, AkitaBatchedProof, AkitaLevelProof, AkitaProofStep,
    AkitaStage1Proof, AkitaStage2Proof, AkitaVerifierSetup, BasisMode, BlockOrder,
    ClaimIncidenceSummary, DirectWitnessProof, ExtensionOpeningReductionProof, FlatRingVec,
    LevelParams, MRowLayout, RingCommitment, RingOpeningPoint, RingSubfieldEncoding, Schedule,
    Step, TerminalLevelProof, TerminalWitnessSegmentLayout, TerminalWitnessTranscriptParts,
};

mod recursive;

pub(crate) use recursive::verify_fold_batched_proof;

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

struct TerminalWitnessReplay {
    parts: TerminalWitnessTranscriptParts,
}

fn prepare_terminal_witness_replay<F, T>(
    transcript: &mut T,
    final_witness: &DirectWitnessProof<F>,
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
    transcript.record_wire_bytes(ABSORB_TERMINAL_W_HAT, &parts.w_hat);
    transcript.append_bytes(ABSORB_TERMINAL_W_HAT, &parts.w_hat);
    Ok(TerminalWitnessReplay { parts })
}

#[cfg(feature = "zk")]
fn zk_recovered_y_ring_lc<F, E, const D: usize>(
    y_ring: &CyclotomicRing<F, D>,
    y_masks: &[ZkR1csLinearCombination<E>],
    inner_reduction: &CyclotomicRing<F, D>,
) -> Result<ZkR1csLinearCombination<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F>,
{
    if y_masks.len() != D {
        return Err(AkitaError::InvalidProof);
    }
    let masked_opening = recover_ring_subfield_inner_product::<F, E, D>(y_ring, inner_reduction)?;
    let mut mask_coeffs = Vec::with_capacity(D);
    for coeff_idx in 0..D {
        let mut basis_y = CyclotomicRing::<F, D>::zero();
        basis_y.coeffs[coeff_idx] = F::one();
        mask_coeffs.push(recover_ring_subfield_inner_product::<F, E, D>(
            &basis_y,
            inner_reduction,
        )?);
    }
    zk_masked_linear_value_lc(masked_opening, y_masks, &mask_coeffs)
}

#[cfg(feature = "zk")]
fn verify_zk_hiding_commitment<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    root_params: &LevelParams,
    proof: &ZkHidingProof<F>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if D == 0 || proof.u_blind.is_empty() || proof.hiding_witness.is_empty() {
        return Err(AkitaError::InvalidProof);
    }

    let num_ring = proof
        .hiding_witness
        .len()
        .div_ceil(D)
        .max(1)
        .checked_next_power_of_two()
        .ok_or(AkitaError::InvalidProof)?;
    let eval_len = num_ring
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK hiding witness length overflow".to_string()))?;
    let mut evals = vec![F::zero(); eval_len];
    let live_evals = evals
        .get_mut(..proof.hiding_witness.len())
        .ok_or(AkitaError::InvalidProof)?;
    live_evals.copy_from_slice(&proof.hiding_witness);

    let hiding_params = root_params.with_decomp(
        num_ring.trailing_zeros() as usize,
        0,
        root_params.num_digits_commit,
        root_params.num_digits_open,
        root_params.num_digits_fold,
        num_ring,
    )?;
    let witness_rings = field_evals_to_rings::<F, D>(&evals)?;
    let mut b_input_digits = direct_decomposed_inner_rows(&witness_rings, setup, &hiding_params)?;
    append_direct_blinding::<F, D>(
        &mut b_input_digits,
        &proof.b_blinding_digits,
        &hiding_params,
    )?;
    let expanded = setup.expanded.as_ref();
    let seed = expanded.seed();
    let shared_matrix = expanded.shared_matrix();
    if b_input_digits.len() > seed.max_stride {
        return Err(AkitaError::InvalidSetup(
            "ZK hiding commitment exceeds shared matrix stride".to_string(),
        ));
    }

    let b_matrix = shared_matrix.ring_view::<D>(hiding_params.b_key.row_len(), seed.max_stride)?;
    let b_rows: Vec<_> = b_matrix.rows().collect();
    let expected_u_blind_rings = mat_vec_mul_i8_plain::<F, D>(&b_rows, &b_input_digits);
    let expected_len = expected_u_blind_rings
        .len()
        .checked_mul(D)
        .ok_or(AkitaError::InvalidProof)?;
    if proof.u_blind.len() != expected_len {
        return Err(AkitaError::InvalidProof);
    }
    let expected_u_blind = expected_u_blind_rings
        .iter()
        .flat_map(|ring| ring.coeffs.iter().copied())
        .collect::<Vec<_>>();
    if proof.u_blind.as_slice() != expected_u_blind.as_slice() {
        return Err(AkitaError::InvalidProof);
    }

    Ok(())
}

/// Verify the intermediate-root proof payload for batched proofs whose root
/// is followed by additional recursive levels.
///
/// This replays the canonical root transcript layout: batch-shape header,
/// commitments, padded opening points, per-claim field openings, one gamma
/// challenge per claim, and gamma-combined per-point y-rings, then runs the
/// stage-1 norm-check sumcheck and the stage-2 fused sumcheck, threading
/// `next_w_commitment` through `ABSORB_SUMCHECK_W`.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, any public trace check
/// fails, ring-switch replay fails, or either sumcheck verifier rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn verify_intermediate_root_level<F, E, C, T, const D: usize>(
    y_rings_flat: &FlatRingVec<F>,
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<C>>,
    v_flat: &FlatRingVec<F>,
    stage1: &AkitaStage1Proof<C>,
    stage2: &AkitaStage2Proof<F, C>,
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
        y_rings_flat,
        extension_opening_reduction,
        Some(v_flat),
        Some(stage1),
        #[cfg(not(feature = "zk"))]
        &stage2.sumcheck_proof,
        #[cfg(feature = "zk")]
        &stage2.sumcheck_proof_masked,
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
/// preamble; at the terminal, [`ABSORB_SUMCHECK_W`] absorbs `final_witness`
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
    y_rings_flat: &FlatRingVec<F>,
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<C>>,
    #[cfg(not(feature = "zk"))] stage2_sumcheck: &akita_sumcheck::SumcheckProof<C>,
    #[cfg(feature = "zk")] stage2_sumcheck_masked: &akita_sumcheck::SumcheckProofMasked<C>,
    final_witness: &DirectWitnessProof<F>,
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
        y_rings_flat,
        extension_opening_reduction,
        None,
        None,
        #[cfg(not(feature = "zk"))]
        stage2_sumcheck,
        #[cfg(feature = "zk")]
        stage2_sumcheck_masked,
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
        final_witness: &'a DirectWitnessProof<F>,
        final_w_len: usize,
    },
}

#[allow(clippy::too_many_arguments)]
fn verify_root_level_inner<F, E, C, T, const D: usize>(
    y_rings_flat: &FlatRingVec<F>,
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<C>>,
    v_flat: Option<&FlatRingVec<F>>,
    stage1: Option<&AkitaStage1Proof<C>>,
    #[cfg(not(feature = "zk"))] stage2_sumcheck: &akita_sumcheck::SumcheckProof<C>,
    #[cfg(feature = "zk")] stage2_sumcheck_masked: &akita_sumcheck::SumcheckProofMasked<C>,
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
    let y_rings = y_rings_flat.as_ring_slice::<D>()?;
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
        || y_rings.len() != incidence_summary.num_public_rows()
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
        let result = ExtensionOpeningReductionSumcheck::new(
            input_claim,
            incidence_summary.num_vars() - split_bits,
        )
        .verify::<F, _, _>(&reduction.sumcheck, transcript, |tr| {
            sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?;
        #[cfg(feature = "zk")]
        let (final_claim_lc, challenges) = ExtensionOpeningReductionSumcheck::new(
            input_claim,
            incidence_summary.num_vars() - split_bits,
        )
        .verify_zk::<F, _, _>(
            &reduction.sumcheck_proof_masked,
            input_claim_mask,
            transcript,
            |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
            zk_relations,
            zk_hiding_cursor,
        )?;
        #[cfg(feature = "zk")]
        let result_challenges = challenges;
        #[cfg(not(feature = "zk"))]
        let result_challenges = result.challenges;
        let factors_by_point = padded_points
            .iter()
            .map(|point| {
                tensor_equality_factor_eval_at_point::<F, C>(
                    &point[split_bits..],
                    &eta,
                    &result_challenges,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(feature = "zk")]
        {
            Some((final_claim_lc, result_challenges, factors_by_point))
        }
        #[cfg(not(feature = "zk"))]
        {
            Some((result.final_claim, result_challenges, factors_by_point))
        }
    } else {
        None
    };

    let prepared_points = if let Some((_final_claim, rho, _factors_by_point)) = &reduction_check {
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
    for y_ring in y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }
    #[cfg(feature = "zk")]
    let y_masks = zk_base_mask_lcs::<C>(y_rings.len() * D, zk_hiding_cursor);

    // Per-row trace check: for each public row `r`, verify the ring-subfield
    // trace inner-product identity
    // `trace_h(Y_r · σ_{-1}(v_{point(r)})) == (D / K) · embed_subfield(opening_r)`
    // in R_q, where `opening_r = Σ_{c in row(r)} γ_{r,c} · opening_c`.
    if reduction_check.is_none() {
        let mut batched_openings_per_row: Vec<C> =
            vec![C::zero(); incidence_summary.num_public_rows()];
        for (row_idx, row) in incidence_summary.public_rows().iter().enumerate() {
            if row.point_idx() >= prepared_points.len() || row.claim_indices().is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            for &claim_idx in row.claim_indices() {
                if claim_idx >= openings.len()
                    || incidence_summary.claim_to_point()[claim_idx] != row_idx
                    || incidence_summary.claim_to_point()[claim_idx] != row.point_idx()
                {
                    return Err(AkitaError::InvalidProof);
                }
                batched_openings_per_row[row_idx] +=
                    row_coefficients[claim_idx] * C::lift_base(openings[claim_idx]);
            }
        }
        #[cfg(feature = "zk")]
        {
            for (row_idx, row) in incidence_summary.public_rows().iter().enumerate() {
                let y_mask_start = row_idx.checked_mul(D).ok_or(AkitaError::InvalidProof)?;
                let y_mask_end = y_mask_start
                    .checked_add(D)
                    .ok_or(AkitaError::InvalidProof)?;
                let y_opening = zk_recovered_y_ring_lc::<F, C, D>(
                    &y_rings[row_idx],
                    y_masks
                        .get(y_mask_start..y_mask_end)
                        .ok_or(AkitaError::InvalidProof)?,
                    &prepared_points[row.point_idx()].inner_reduction,
                )?;
                let mut residual = y_opening;
                residual.constant -= batched_openings_per_row[row_idx];
                zk_push_linear_zero(zk_relations, "root y-ring opening relation", residual)?;
            }
        }
        #[cfg(not(feature = "zk"))]
        {
            for (row, (y_ring, batched_opening)) in incidence_summary
                .public_rows()
                .iter()
                .zip(y_rings.iter().zip(batched_openings_per_row.iter()))
            {
                let v = &prepared_points[row.point_idx()].inner_reduction;
                let trace_input = *y_ring * v.sigma_m1();
                let coords = batched_opening.to_ring_subfield_coords();
                if !dispatch_trace_inner_product_check::<F, { D }>(
                    &trace_input,
                    &coords,
                    AkitaError::InvalidProof,
                )? {
                    return Err(AkitaError::InvalidProof);
                }
            }
        }
    } else if let Some((final_claim, _rho, factors_by_point)) = &reduction_check {
        #[cfg(feature = "zk")]
        {
            let mut final_opening = ZkR1csLinearCombination::zero();
            for (row_idx, row) in incidence_summary.public_rows().iter().enumerate() {
                if row.point_idx() >= factors_by_point.len()
                    || row.point_idx() >= prepared_points.len()
                {
                    return Err(AkitaError::InvalidProof);
                }
                let y_mask_start = row_idx.checked_mul(D).ok_or(AkitaError::InvalidProof)?;
                let y_mask_end = y_mask_start
                    .checked_add(D)
                    .ok_or(AkitaError::InvalidProof)?;
                let y_opening = zk_recovered_y_ring_lc::<F, C, D>(
                    &y_rings[row_idx],
                    y_masks
                        .get(y_mask_start..y_mask_end)
                        .ok_or(AkitaError::InvalidProof)?,
                    &prepared_points[row.point_idx()].inner_reduction,
                )?;
                final_opening.add_scaled(factors_by_point[row.point_idx()], &y_opening);
            }
            let mut residual = final_claim.clone();
            residual.add_scaled(-C::one(), &final_opening);
            zk_push_linear_zero(
                zk_relations,
                "root extension-opening reduction output",
                residual,
            )?;
        }
        #[cfg(not(feature = "zk"))]
        {
            let internal_claims = y_rings
                .iter()
                .zip(incidence_summary.public_rows().iter())
                .map(|(y_ring, row)| {
                    recover_ring_subfield_inner_product::<F, C, D>(
                        y_ring,
                        &prepared_points[row.point_idx()].inner_reduction,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let final_opening = internal_claims
                .iter()
                .zip(incidence_summary.public_rows().iter())
                .fold(C::zero(), |acc, (&opening, row)| {
                    acc + opening * factors_by_point[row.point_idx()]
                });
            check_extension_opening_reduction_output(*final_claim, final_opening, C::one())?;
        }
    } else {
        return Err(AkitaError::InvalidProof);
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
            MRowLayout::Terminal
        } else {
            MRowLayout::Intermediate
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
    let rs = match &stage_input {
        RootStageInput::Intermediate {
            next_w_commitment, ..
        } => ring_switch_verifier::<F, C, T, { D }>(
            &ring_opening_points,
            &ring_multiplier_points,
            incidence_summary.claim_to_point(),
            &stage1_challenges,
            w_len,
            next_w_commitment,
            transcript,
            batched_lp,
            incidence_summary.num_polys_per_point(),
            incidence_summary.claim_to_point(),
            incidence_summary.claim_poly_indices(),
            &row_coefficients,
            incidence_summary.num_public_rows(),
        )?,
        RootStageInput::Terminal { .. } => {
            let replay = terminal_replay.as_ref().ok_or(AkitaError::InvalidProof)?;
            ring_switch_verifier_terminal::<F, C, T, { D }>(
                &ring_opening_points,
                &ring_multiplier_points,
                incidence_summary.claim_to_point(),
                &stage1_challenges,
                w_len,
                transcript,
                &replay.parts,
                batched_lp,
                incidence_summary.num_polys_per_point(),
                incidence_summary.claim_to_point(),
                incidence_summary.claim_poly_indices(),
                &row_coefficients,
                incidence_summary.num_public_rows(),
            )?
        }
    };
    let relation_claim = relation_claim_from_rows_extension::<F, C, D>(
        &rs.tau1,
        rs.alpha,
        v_typed,
        commitment_rows,
        y_rings,
    )?;
    #[cfg(feature = "zk")]
    let relation_claim_mask =
        zk_relation_claim_mask_from_y_masks::<C, D>(&rs.tau1, rs.alpha, y_rings.len(), &y_masks)?;
    #[cfg(feature = "zk")]
    let mut s_claim_mask = ZkR1csLinearCombination::<C>::zero();
    let (batching_coeff, s_claim, r_stage1) = match (&stage_input, stage1) {
        (RootStageInput::Intermediate { .. }, Some(stage1_proof)) => {
            let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
            let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
            #[cfg(not(feature = "zk"))]
            let r_stage1 = {
                let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
                stage1_verifier.verify::<F, T>(stage1_proof, transcript)?
            };
            #[cfg(feature = "zk")]
            let r_stage1 = {
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
            (batching_coeff, stage1_proof.s_claim, r_stage1)
        }
        (RootStageInput::Terminal { .. }, None) => {
            // Relation-only stage-2: skip stage-1 entirely. Dummy zeros for
            // r_stage1 + batching_coeff zero out the virtual half.
            let r_stage1 = vec![C::zero(); rs.col_bits + rs.ring_bits];
            (C::zero(), C::zero(), r_stage1)
        }
        _ => return Err(AkitaError::InvalidProof),
    };
    let stage2_input_claim = batching_coeff * s_claim + relation_claim;
    let row_eval_source = Stage2RowEvalSource::new(rs.prepared_row_eval);
    #[cfg(feature = "zk")]
    let stage2_next_w_eval_mask_cursor =
        *zk_hiding_cursor + (rs.col_bits + rs.ring_bits) * 3 * <C as ExtField<F>>::EXT_DEGREE;
    let stage2_verifier = match &stage_input {
        RootStageInput::Terminal { final_witness, .. } => {
            AkitaStage2Verifier::new_with_direct_witness(
                batching_coeff,
                s_claim,
                #[cfg(feature = "zk")]
                s_claim_mask,
                #[cfg(feature = "zk")]
                relation_claim_mask,
                final_witness,
                w_len,
                r_stage1,
                rs.alpha_evals_y,
                row_eval_source,
                &setup.expanded,
                &ring_opening_points,
                &ring_multiplier_points,
                &rs.tau1,
                v_typed,
                commitment_rows,
                y_rings,
                Some(relation_claim),
                rs.alpha,
                rs.col_bits,
                rs.ring_bits,
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
                *next_w_eval,
                #[cfg(feature = "zk")]
                zk_ext_mask_lc_at::<F, C>(stage2_next_w_eval_mask_cursor),
                r_stage1,
                rs.alpha_evals_y,
                row_eval_source,
                &setup.expanded,
                &ring_opening_points,
                &ring_multiplier_points,
                &rs.tau1,
                v_typed,
                commitment_rows,
                y_rings,
                Some(relation_claim),
                rs.alpha,
                rs.col_bits,
                rs.ring_bits,
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
    Ok(sumcheck_challenges)
}
