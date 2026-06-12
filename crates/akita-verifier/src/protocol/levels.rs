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
    ring_switch_verifier, ring_switch_verifier_terminal, RingSwitchReplay, RingSwitchVerifyOutput,
};
use crate::stages::stage1::{derive_stage1_challenges, AkitaStage1Verifier};
use crate::stages::stage2::{AkitaStage2Verifier, Stage2WitnessOracle};
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
    lift_hiding_witness, zk_base_mask_lcs, zk_ext_mask_lc, zk_ext_mask_lc_at,
    zk_masked_compressed_round_claim_mask, zk_push_linear_zero,
    zk_relation_claim_mask_from_y_masks, zk_row_masks_from_column_masks, ZkR1csLinearCombination,
    ZkRelationAccumulator,
};
use akita_serialization::AkitaSerialize;
#[cfg(not(feature = "zk"))]
use akita_sumcheck::SumcheckInstanceVerifierExt;
#[cfg(feature = "zk")]
use akita_sumcheck::ZkSumcheckInstanceVerifierExt;
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_STAGE2_NEXT_W_EVAL,
    ABSORB_SUMCHECK_S_CLAIM, ABSORB_TERMINAL_E_HAT, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_ROUND,
};
#[cfg(feature = "zk")]
use akita_transcript::labels::{ABSORB_SUMCHECK_CLAIM, ABSORB_ZK_HIDING_COMMITMENT};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
#[cfg(not(feature = "zk"))]
use akita_types::check_tensor_extension_opening_claim;
#[cfg(not(feature = "zk"))]
use akita_types::dispatch_trace_inner_product_check;
use akita_types::{
    append_batched_commitments_to_transcript, append_claim_incidence_shape_to_transcript,
    append_claim_points_to_transcript, append_claim_values_to_transcript,
    flatten_batched_commitment_rows, prepare_recursive_opening_point_ext,
    prepare_root_opening_point_ext, relation_claim_from_rows_extension, reorder_stage1_coords,
    ring_subfield_packed_extension_opening_point, sample_public_row_coefficients,
    schedule_num_fold_levels, scheduled_next_level_params, terminal_witness_segment_layout,
    w_ring_element_count_with_counts, AkitaBatchedProof, AkitaLevelProof, AkitaProofStep,
    AkitaStage1Proof, AkitaStage2Proof, AkitaVerifierSetup, BasisMode, BlockOrder,
    ClaimIncidenceSummary, CleartextWitnessProof, CommitmentRouting,
    ExtensionOpeningReductionProof, FlatRingVec, LevelParams, MRowLayout, PreparedRootOpeningPoint,
    RingCommitment, RingMultiplierOpeningPoint, RingOpeningPoint, RingRelationInstance,
    RingSubfieldEncoding, Schedule, SetupContributionMode, SetupSumcheckProof, Step,
    TerminalLevelProof, TerminalWitnessSegmentLayout, TerminalWitnessTranscriptParts,
};
#[cfg(feature = "zk")]
use akita_types::{tensor_equality_factor_eval_at_point, EXTENSION_OPENING_REDUCTION_DEGREE};
use akita_types::{
    tensor_opening_split, tensor_reduction_claim_from_rows, tensor_row_partials_from_columns,
};
#[cfg(not(feature = "zk"))]
use extension_opening_reduction::ExtensionOpeningReductionVerifier;
#[cfg(feature = "zk")]
use zk::{verify_zk_hiding_commitment, zk_recovered_y_ring_lc};

mod recursive;
pub(crate) use recursive::verify_fold_batched_proof;

/// Verifier state carried between recursive fold levels.
struct RecursiveVerifierState<'a, F: FieldCore, L: FieldCore> {
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

fn prepare_terminal_witness_replay<F, T>(
    transcript: &mut T,
    final_witness: &CleartextWitnessProof<F>,
    final_w_len: usize,
    layout: TerminalWitnessSegmentLayout,
) -> Result<TerminalWitnessTranscriptParts, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    if final_witness.num_elems() != final_w_len {
        return Err(AkitaError::InvalidProof);
    }
    let parts = final_witness.terminal_transcript_parts(layout)?;
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_E_HAT, &parts.e_hat);
    Ok(parts)
}

enum RootLevelProofView<'a, F: FieldCore, C: FieldCore> {
    Intermediate {
        y_rings_flat: &'a FlatRingVec<F>,
        extension_opening_reduction: Option<&'a ExtensionOpeningReductionProof<C>>,
        v_flat: &'a FlatRingVec<F>,
        stage1: &'a AkitaStage1Proof<C>,
        stage2: &'a AkitaStage2Proof<F, C>,
        stage3_sumcheck_proof: Option<&'a SetupSumcheckProof<C>>,
        setup_contribution_mode: SetupContributionMode,
        next_fold_level_params: &'a LevelParams,
    },
    Terminal {
        y_rings_flat: &'a FlatRingVec<F>,
        extension_opening_reduction: Option<&'a ExtensionOpeningReductionProof<C>>,
        #[cfg(not(feature = "zk"))]
        stage2_sumcheck: &'a akita_sumcheck::SumcheckProof<C>,
        #[cfg(feature = "zk")]
        stage2_sumcheck_masked: &'a akita_sumcheck::SumcheckProofMasked<C>,
        final_witness: &'a CleartextWitnessProof<F>,
        final_w_len: usize,
    },
}

struct RootEorReplay<F: FieldCore, C: FieldCore, const D: usize> {
    prepared_points: Vec<PreparedRootOpeningPoint<F, D>>,
    reduction_challenges: Option<Vec<C>>,
    #[cfg(feature = "zk")]
    final_relation: Option<(ZkR1csLinearCombination<C>, Vec<C>)>,
}

#[derive(Clone, Copy)]
struct EorReductionShape {
    split_bits: usize,
    width: usize,
    num_rounds: usize,
}

fn eor_reduction_shape<F, C>(
    opening_num_vars: usize,
    partials_len: usize,
    num_claims: usize,
) -> Result<EorReductionShape, AkitaError>
where
    F: FieldCore,
    C: ExtField<F>,
{
    let (split_bits, width) =
        tensor_opening_split::<F, C>().map_err(|_| AkitaError::InvalidProof)?;
    let num_rounds = opening_num_vars
        .checked_sub(split_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let expected_partials = width
        .checked_mul(num_claims)
        .ok_or(AkitaError::InvalidProof)?;
    if width == 1 || partials_len != expected_partials {
        return Err(AkitaError::InvalidProof);
    }
    Ok(EorReductionShape {
        split_bits,
        width,
        num_rounds,
    })
}

fn eor_input_claim_from_partials<F, C>(
    partials: &[C],
    shape: EorReductionShape,
    eta: &[C],
    row_coefficients: &[C],
) -> Result<C, AkitaError>
where
    F: FieldCore,
    C: ExtField<F>,
{
    if shape.width == 0
        || !partials.len().is_multiple_of(shape.width)
        || row_coefficients.len() != partials.len() / shape.width
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut input_claim = C::zero();
    for (&row_coefficient, partials) in row_coefficients
        .iter()
        .zip(partials.chunks_exact(shape.width))
    {
        let row_partials = tensor_row_partials_from_columns::<F, C>(partials)?;
        let claim = tensor_reduction_claim_from_rows::<F, C>(&row_partials, eta)?;
        input_claim += row_coefficient * claim;
    }
    Ok(input_claim)
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
    relations: &mut ZkRelationAccumulator<E>,
    hiding_cursor: &mut usize,
) -> Result<(ZkR1csLinearCombination<E>, Vec<E>), AkitaError>
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
        relations.push_masked_claim_relation(
            "extension-opening reduction final claim",
            masked_claim,
            &claim_mask,
        ),
        challenges,
    ))
}

#[allow(clippy::too_many_arguments)]
fn verify_root_eor_and_prepare_points<F, E, C, T, const D: usize>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<C>>,
    y_rings: &[CyclotomicRing<F, D>],
    claim_points: &[&[E]],
    openings: &[E],
    row_coefficients: &[C],
    incidence_summary: &ClaimIncidenceSummary,
    basis: BasisMode,
    root_lp: &LevelParams,
    transcript: &mut T,
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<C>,
) -> Result<RootEorReplay<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
{
    #[cfg(feature = "zk")]
    let _ = y_rings;
    let alpha_bits = root_lp.ring_dimension.trailing_zeros() as usize;
    // The zk EOR final relation consumes the y-ring opening masks, which are a
    // shared resource with the downstream ring-switch binding, so it stays in
    // this outer flow rather than inside the sumcheck driver. These extras carry
    // `(final_claim_lc, factors_by_point)` for that deferred relation.
    #[cfg(feature = "zk")]
    let mut zk_eor_final: Option<(ZkR1csLinearCombination<C>, Vec<C>)> = None;
    let reduction_check = if let Some(reduction) = extension_opening_reduction {
        if <C as ExtField<F>>::EXT_DEGREE == 1 {
            return Err(AkitaError::InvalidProof);
        }
        if <C as ExtField<F>>::EXT_DEGREE != <E as ExtField<F>>::EXT_DEGREE {
            return Err(AkitaError::InvalidProof);
        }
        let shape = eor_reduction_shape::<F, C>(
            incidence_summary.num_vars(),
            reduction.partials.len(),
            incidence_summary.num_claims(),
        )?;
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
        #[cfg(feature = "zk")]
        let mut input_claim_mask = ZkR1csLinearCombination::zero();
        for (claim_idx, opening) in openings
            .iter()
            .copied()
            .enumerate()
            .take(incidence_summary.num_claims())
        {
            let point_idx = incidence_summary.claim_to_point()[claim_idx];
            let partial_start = claim_idx * shape.width;
            let partial_end = partial_start + shape.width;
            let partials = &reduction.partials[partial_start..partial_end];
            #[cfg(feature = "zk")]
            let partial_masks = (0..shape.width)
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
                    EqPolynomial::<C>::evals(&padded_points[point_idx][..shape.split_bits])?;
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
        let eta = (0..shape.split_bits)
            .map(|_| sample_ext_challenge::<F, C, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
            .collect::<Vec<_>>();
        let input_claim = eor_input_claim_from_partials::<F, C>(
            &reduction.partials,
            shape,
            &eta,
            row_coefficients,
        )?;
        #[cfg(feature = "zk")]
        for (claim_idx, &row_coefficient) in row_coefficients
            .iter()
            .enumerate()
            .take(incidence_summary.num_claims())
        {
            let partial_start = claim_idx * shape.width;
            let mut partial_masks = Vec::with_capacity(shape.width);
            for offset in 0..shape.width {
                let mask_start = partial_start + offset;
                let mask = zk_ext_mask_lc_at::<F, C>(
                    *zk_hiding_cursor - reduction.partials.len() * <C as ExtField<F>>::EXT_DEGREE
                        + mask_start * <C as ExtField<F>>::EXT_DEGREE,
                );
                partial_masks.push(mask);
            }
            let row_masks = zk_row_masks_from_column_masks::<F, C>(&partial_masks)?;
            for (weight, row_mask) in EqPolynomial::<C>::evals(&eta)?.into_iter().zip(row_masks) {
                input_claim_mask.add_scaled(row_coefficient * weight, &row_mask);
            }
        }
        #[cfg(not(feature = "zk"))]
        {
            let rows = incidence_summary
                .public_rows()
                .iter()
                .enumerate()
                .map(|(row_idx, row)| {
                    let y_ring = y_rings.get(row_idx).ok_or(AkitaError::InvalidProof)?;
                    let point = padded_points
                        .get(row.point_idx())
                        .ok_or(AkitaError::InvalidProof)?;
                    let point_tail = point
                        .get(shape.split_bits..)
                        .ok_or(AkitaError::InvalidProof)?
                        .to_vec();
                    Ok((y_ring, point_tail))
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            let eor_verifier = ExtensionOpeningReductionVerifier::<F, C, D>::new(
                shape.num_rounds,
                input_claim,
                eta,
                rows,
                Box::new(
                    move |rho: &[C]| -> Result<CyclotomicRing<F, D>, AkitaError> {
                        let protocol_point = ring_subfield_packed_extension_opening_point::<F, C, D>(
                            rho.len(),
                            rho,
                        )?;
                        Ok(prepare_root_opening_point_ext::<F, C, C, D>(
                            &protocol_point,
                            basis,
                            root_lp,
                            alpha_bits,
                        )?
                        .inner_reduction)
                    },
                ),
            );
            let rho = eor_verifier.verify::<F, T, _>(&reduction.sumcheck, transcript, |tr| {
                sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;
            Some(rho)
        }
        #[cfg(feature = "zk")]
        {
            let (final_claim_lc, challenges) =
                verify_zk_extension_opening_reduction_sumcheck::<F, C, T, _>(
                    input_claim,
                    shape.num_rounds,
                    &reduction.sumcheck_proof_masked,
                    input_claim_mask,
                    transcript,
                    |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                    zk_relations,
                    zk_hiding_cursor,
                )?;
            let factors_by_point = padded_points
                .iter()
                .map(|point| {
                    tensor_equality_factor_eval_at_point::<F, C>(
                        &point[shape.split_bits..],
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
    Ok(RootEorReplay {
        prepared_points,
        reduction_challenges: reduction_check,
        #[cfg(feature = "zk")]
        final_relation: zk_eor_final,
    })
}

enum Stage2ProofReplay<'a, F: FieldCore, E: FieldCore> {
    Intermediate {
        next_w_eval: E,
        #[cfg(not(feature = "zk"))]
        sumcheck: &'a akita_sumcheck::SumcheckProof<E>,
        #[cfg(feature = "zk")]
        sumcheck_masked: &'a akita_sumcheck::SumcheckProofMasked<E>,
    },
    Terminal {
        final_witness: &'a CleartextWitnessProof<F>,
        physical_w_len: usize,
        #[cfg(not(feature = "zk"))]
        sumcheck: &'a akita_sumcheck::SumcheckProof<E>,
        #[cfg(feature = "zk")]
        sumcheck_masked: &'a akita_sumcheck::SumcheckProofMasked<E>,
    },
}

struct Stage2ReplayInput<'a, F: FieldCore, E: FieldCore, const D: usize> {
    setup: &'a AkitaVerifierSetup<F>,
    stage2: Stage2ProofReplay<'a, F, E>,
    stage1: Stage1Replay<E>,
    rs: RingSwitchVerifyOutput<E>,
    relation_claim: E,
    #[cfg(feature = "zk")]
    relation_claim_mask: ZkR1csLinearCombination<E>,
    setup_replay: Option<SetupReplay<'a, E>>,
    ring_multiplier_points: &'a [RingMultiplierOpeningPoint<F, D>],
}

struct SetupReplay<'a, E: FieldCore> {
    proof: &'a SetupSumcheckProof<E>,
    next_fold_level_params: &'a LevelParams,
}

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

fn verify_stage2_and_setup_replay<F, E, T, const D: usize>(
    transcript: &mut T,
    input: Stage2ReplayInput<'_, F, E, D>,
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<E>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let Stage2ReplayInput {
        setup,
        stage2,
        stage1,
        rs,
        relation_claim,
        #[cfg(feature = "zk")]
        relation_claim_mask,
        setup_replay,
        ring_multiplier_points,
    } = input;
    let Stage1Replay {
        batching_coeff,
        s_claim,
        stage1_point,
        #[cfg(feature = "zk")]
        s_claim_mask,
    } = stage1;
    let setup_claim = setup_replay.as_ref().map(|replay| replay.proof.claim);
    #[cfg(feature = "zk")]
    let stage2_next_w_eval_mask_cursor =
        *zk_hiding_cursor + (rs.col_bits + rs.ring_bits) * 3 * <E as ExtField<F>>::EXT_DEGREE;
    let witness_oracle = match &stage2 {
        Stage2ProofReplay::Terminal {
            final_witness,
            physical_w_len,
            ..
        } => Stage2WitnessOracle::Cleartext {
            witness: final_witness,
            physical_w_len: *physical_w_len,
        },
        Stage2ProofReplay::Intermediate { next_w_eval, .. } => Stage2WitnessOracle::ClaimedEval {
            eval: *next_w_eval,
            #[cfg(feature = "zk")]
            mask: zk_ext_mask_lc_at::<F, E>(stage2_next_w_eval_mask_cursor),
        },
    };
    let stage2_verifier = AkitaStage2Verifier::new(
        batching_coeff,
        s_claim,
        #[cfg(feature = "zk")]
        s_claim_mask,
        #[cfg(feature = "zk")]
        relation_claim_mask,
        witness_oracle,
        stage1_point,
        &rs.alpha_evals_y,
        &rs.prepared_row_eval,
        setup_claim,
        &setup.expanded,
        ring_multiplier_points,
        relation_claim,
        rs.alpha,
        rs.col_bits,
        rs.ring_bits,
    )?;

    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        #[cfg(not(feature = "zk"))]
        {
            let stage2_sumcheck = match &stage2 {
                Stage2ProofReplay::Intermediate { sumcheck, .. }
                | Stage2ProofReplay::Terminal { sumcheck, .. } => *sumcheck,
            };
            stage2_verifier.verify::<F, T, _>(stage2_sumcheck, transcript, |tr| {
                sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?
        }
        #[cfg(feature = "zk")]
        {
            let stage2_sumcheck_masked = match &stage2 {
                Stage2ProofReplay::Intermediate {
                    sumcheck_masked, ..
                }
                | Stage2ProofReplay::Terminal {
                    sumcheck_masked, ..
                } => *sumcheck_masked,
            };
            let challenges = stage2_verifier.verify_zk::<F, T, _>(
                stage2_sumcheck_masked,
                transcript,
                zk_relations,
                zk_hiding_cursor,
                |tr| sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND),
            )?;
            if matches!(stage2, Stage2ProofReplay::Intermediate { .. }) {
                *zk_hiding_cursor += <E as ExtField<F>>::EXT_DEGREE;
            }
            challenges
        }
    };
    if let Stage2ProofReplay::Intermediate { next_w_eval, .. } = stage2 {
        transcript.absorb_and_record_serde(ABSORB_STAGE2_NEXT_W_EVAL, &next_w_eval);
    }
    if let Some(setup_replay) = setup_replay {
        let verifier = SetupSumcheckVerifier::new::<F, D>(
            &rs.prepared_row_eval,
            &sumcheck_challenges[rs.ring_bits..],
            rs.alpha,
        )?;
        verifier.verify::<F, T, D>(
            setup,
            setup_replay.next_fold_level_params,
            setup_replay.proof,
            transcript,
        )?;
    }
    Ok(sumcheck_challenges)
}

struct Stage1Replay<E: FieldCore> {
    batching_coeff: E,
    s_claim: E,
    stage1_point: Vec<E>,
    #[cfg(feature = "zk")]
    s_claim_mask: ZkR1csLinearCombination<E>,
}

fn verify_stage1_or_terminal<F, E, T>(
    proof: Option<&AkitaStage1Proof<E>>,
    rs: &RingSwitchVerifyOutput<E>,
    transcript: &mut T,
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<E>,
) -> Result<Stage1Replay<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let num_rounds = rs
        .col_bits
        .checked_add(rs.ring_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("stage-1 variable count overflow".to_string()))?;
    let stage1 = match (proof, rs.stage1_tau0.as_deref()) {
        (Some(proof), Some(tau0)) => Some((proof, tau0)),
        (None, None) => None,
        _ => return Err(AkitaError::InvalidProof),
    };
    if let Some((proof, tau0)) = stage1 {
        if tau0.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: tau0.len(),
            });
        }
        let tau0_reordered = reorder_stage1_coords(tau0, rs.col_bits, rs.ring_bits);
        let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
        #[cfg(not(feature = "zk"))]
        let stage1_point = {
            let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
            stage1_verifier.verify::<F, T>(proof, transcript)?
        };
        #[cfg(feature = "zk")]
        let (stage1_point, s_claim_mask) = {
            let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
            stage1_verifier.verify::<F, T>(proof, transcript, zk_relations, zk_hiding_cursor)?
        };
        transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &proof.s_claim);
        let batching_coeff: E =
            sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
        return Ok(Stage1Replay {
            batching_coeff,
            s_claim: proof.s_claim,
            stage1_point,
            #[cfg(feature = "zk")]
            s_claim_mask,
        });
    }

    Ok(Stage1Replay {
        batching_coeff: E::zero(),
        s_claim: E::zero(),
        // Relation-only stage-2: skip stage-1 entirely. Dummy zeros for
        // stage1_point + batching_coeff zero out the virtual half.
        stage1_point: vec![E::zero(); num_rounds],
        #[cfg(feature = "zk")]
        s_claim_mask: ZkR1csLinearCombination::zero(),
    })
}

/// Verify the folded-root proof payload for either an intermediate root or the
/// 1-fold terminal root.
///
/// This replays the canonical root transcript layout: batch-shape header,
/// commitments, padded opening points, per-claim field openings, row
/// coefficients, EOR if present, y-rings, ring switch, stage-1 when present,
/// stage-2, and setup replay when required by the intermediate branch.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, any public trace check
/// fails, ring-switch replay fails, or a sumcheck verifier rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn verify_root_level<F, E, C, T, const D: usize>(
    proof: RootLevelProofView<'_, F, C>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claim_points: &[&[E]],
    openings: &[E],
    commitments: &[RingCommitment<F, D>],
    incidence_summary: &ClaimIncidenceSummary,
    basis: BasisMode,
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
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
    let (m_row_layout, y_rings_flat, extension_opening_reduction) = match &proof {
        RootLevelProofView::Intermediate {
            y_rings_flat,
            extension_opening_reduction,
            ..
        } => (
            MRowLayout::WithDBlock,
            *y_rings_flat,
            *extension_opening_reduction,
        ),
        RootLevelProofView::Terminal {
            y_rings_flat,
            extension_opening_reduction,
            ..
        } => (
            MRowLayout::WithoutDBlock,
            *y_rings_flat,
            *extension_opening_reduction,
        ),
    };
    let y_rings = y_rings_flat.as_ring_slice::<D>()?;
    let v_typed: &[CyclotomicRing<F, D>] = match &proof {
        RootLevelProofView::Intermediate { v_flat, .. } => v_flat.as_ring_slice::<D>()?,
        RootLevelProofView::Terminal { .. } => &[],
    };
    let next_fold_level_params = match &proof {
        RootLevelProofView::Intermediate {
            next_fold_level_params,
            ..
        } => *next_fold_level_params,
        RootLevelProofView::Terminal { .. } => root_lp,
    };
    let stage3_sumcheck_proof = match &proof {
        RootLevelProofView::Intermediate {
            stage3_sumcheck_proof,
            setup_contribution_mode,
            ..
        } => stage3_sumcheck_proof_for_mode(*setup_contribution_mode, *stage3_sumcheck_proof)?,
        RootLevelProofView::Terminal { .. } => None,
    };
    let num_claims = incidence_summary.num_claims();
    let num_points = incidence_summary.num_points();
    if num_points == 0
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
        .any(|commitment| commitment.u.len() != root_lp.effective_commit_rows())
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

    let root_eor = verify_root_eor_and_prepare_points::<F, E, C, T, D>(
        extension_opening_reduction,
        y_rings,
        claim_points,
        openings,
        &row_coefficients,
        incidence_summary,
        basis,
        root_lp,
        transcript,
        #[cfg(feature = "zk")]
        zk_hiding_cursor,
        #[cfg(feature = "zk")]
        zk_relations,
    )?;
    let reduction_check = root_eor.reduction_challenges;
    let prepared_points = root_eor.prepared_points;
    #[cfg(feature = "zk")]
    let zk_eor_final = root_eor.final_relation;
    for row in incidence_summary.public_rows() {
        if row.point_idx() >= prepared_points.len() || row.claim_indices().is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        for &claim_idx in row.claim_indices() {
            if claim_idx >= openings.len()
                || incidence_summary.claim_to_point()[claim_idx] != row.point_idx()
            {
                return Err(AkitaError::InvalidProof);
            }
        }
    }

    // `y_ring` is standalone wire data pinned at the EOR output point rho; this
    // absorb binds it before downstream relation-sumcheck challenges are sampled.
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
            for &claim_idx in row.claim_indices() {
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
                if !dispatch_trace_inner_product_check::<F, D>(
                    &trace_input,
                    &coords,
                    AkitaError::InvalidProof,
                )? {
                    return Err(AkitaError::InvalidProof);
                }
            }
        }
    }
    // The non-zk EOR final relation is enforced inside the sumcheck driver via
    // `ExtensionOpeningReductionVerifier::expected_output_claim`. In zk mode the
    // final relation consumes the shared y-ring opening masks, so it stays here.
    #[cfg(feature = "zk")]
    if let Some((final_claim, factors_by_point)) = &zk_eor_final {
        let mut final_opening = ZkR1csLinearCombination::zero();
        for (row_idx, row) in incidence_summary.public_rows().iter().enumerate() {
            if row.point_idx() >= factors_by_point.len() {
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

    let w_len = match &proof {
        RootLevelProofView::Terminal { final_w_len, .. } => *final_w_len,
        RootLevelProofView::Intermediate { .. } => w_ring_element_count_with_counts::<F>(
            batched_lp,
            incidence_summary.num_polys_per_point().len(),
            incidence_summary.num_polys_per_point().iter().sum(),
            num_claims,
            incidence_summary.num_public_rows(),
        )?
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("next witness length overflow".to_string()))?,
    };
    let terminal_replay = match &proof {
        RootLevelProofView::Terminal { final_witness, .. } => {
            let layout = terminal_witness_segment_layout(
                batched_lp,
                num_claims,
                incidence_summary.num_public_rows(),
                F::modulus_bits(),
            )?;
            Some(prepare_terminal_witness_replay::<F, T>(
                transcript,
                *final_witness,
                w_len,
                layout,
            )?)
        }
        RootLevelProofView::Intermediate { .. } => None,
    };

    let stage1_challenges = derive_stage1_challenges::<F, T, D>(
        transcript,
        v_typed,
        root_lp.num_blocks,
        num_claims,
        batched_lp,
        m_row_layout,
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
    let commitment_routing = CommitmentRouting::from_root_incidence(incidence_summary)?;
    let (gamma, row_coefficient_rings) =
        RingRelationInstance::<F, D>::gamma_and_row_rings_from_coefficients::<C>(
            &row_coefficients,
        )?;
    let relation_instance = RingRelationInstance::new(
        m_row_layout,
        stage1_challenges,
        ring_opening_points,
        ring_multiplier_points,
        incidence_summary.clone(),
        commitment_routing,
        gamma,
        row_coefficient_rings,
        y_rings.to_vec(),
        v_typed.to_vec(),
    )?;
    let ring_switch_replay = RingSwitchReplay {
        relation: &relation_instance,
        row_coefficients: &row_coefficients,
        lp: batched_lp,
    };
    let rs = match &proof {
        RootLevelProofView::Intermediate { stage2, .. } => ring_switch_verifier::<F, C, T, D>(
            &ring_switch_replay,
            w_len,
            &stage2.next_w_commitment,
            transcript,
        )?,
        RootLevelProofView::Terminal { .. } => {
            let replay = terminal_replay.as_ref().ok_or(AkitaError::InvalidProof)?;
            ring_switch_verifier_terminal::<F, C, T, D>(
                &ring_switch_replay,
                w_len,
                transcript,
                replay,
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
    let stage1_proof = match &proof {
        RootLevelProofView::Intermediate { stage1, .. } => Some(*stage1),
        RootLevelProofView::Terminal { .. } => None,
    };
    let stage1_replay = verify_stage1_or_terminal::<F, C, T>(
        stage1_proof,
        &rs,
        transcript,
        #[cfg(feature = "zk")]
        zk_hiding_cursor,
        #[cfg(feature = "zk")]
        zk_relations,
    )?;
    let stage2_replay = match &proof {
        RootLevelProofView::Intermediate { stage2, .. } => Stage2ProofReplay::Intermediate {
            next_w_eval: stage2.next_w_eval(),
            #[cfg(not(feature = "zk"))]
            sumcheck: &stage2.sumcheck_proof,
            #[cfg(feature = "zk")]
            sumcheck_masked: &stage2.sumcheck_proof_masked,
        },
        RootLevelProofView::Terminal {
            final_witness,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck,
            #[cfg(feature = "zk")]
            stage2_sumcheck_masked,
            ..
        } => Stage2ProofReplay::Terminal {
            final_witness: *final_witness,
            physical_w_len: w_len,
            #[cfg(not(feature = "zk"))]
            sumcheck: *stage2_sumcheck,
            #[cfg(feature = "zk")]
            sumcheck_masked: *stage2_sumcheck_masked,
        },
    };
    let stage2_input = Stage2ReplayInput {
        setup,
        stage2: stage2_replay,
        stage1: stage1_replay,
        rs,
        relation_claim,
        #[cfg(feature = "zk")]
        relation_claim_mask,
        setup_replay: stage3_sumcheck_proof.map(|proof| SetupReplay {
            proof,
            next_fold_level_params,
        }),
        ring_multiplier_points: relation_instance.ring_multiplier_points(),
    };
    verify_stage2_and_setup_replay::<F, C, T, D>(
        transcript,
        stage2_input,
        #[cfg(feature = "zk")]
        zk_hiding_cursor,
        #[cfg(feature = "zk")]
        zk_relations,
    )
}
