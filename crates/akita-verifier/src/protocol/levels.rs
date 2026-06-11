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
use akita_algebra::CyclotomicRing;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    PseudoMersenneField, RandomSampling,
};
#[cfg(feature = "zk")]
use akita_r1cs::{
    lift_hiding_witness, zk_base_mask_lcs, zk_push_linear_zero,
    zk_relation_claim_mask_from_y_masks, ZkR1csLinearCombination, ZkRelationAccumulator,
};
use akita_serialization::AkitaSerialize;
#[cfg(not(feature = "zk"))]
use akita_sumcheck::SumcheckInstanceVerifierExt;
#[cfg(feature = "zk")]
use akita_transcript::labels::ABSORB_ZK_HIDING_COMMITMENT;
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_SUMCHECK_S_CLAIM, ABSORB_TERMINAL_E_HAT,
    CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
#[cfg(not(feature = "zk"))]
use akita_types::dispatch_trace_inner_product_check;
use akita_types::{
    append_batched_commitments_to_transcript, append_claim_incidence_shape_to_transcript,
    append_claim_points_to_transcript, append_claim_values_to_transcript,
    flatten_batched_commitment_rows, prepare_recursive_opening_point_ext,
    relation_claim_from_rows_extension, reorder_stage1_coords,
    ring_subfield_packed_extension_opening_point, sample_public_row_coefficients,
    schedule_num_fold_levels, terminal_witness_segment_layout, w_ring_element_count_with_counts,
    AkitaBatchedProof, AkitaLevelProof, AkitaProofStep, AkitaStage1Proof, AkitaStage2Proof,
    AkitaVerifierSetup, BasisMode, BlockOrder, ClaimIncidenceSummary, CleartextWitnessProof,
    CommitmentRouting, ExtensionOpeningReductionProof, FlatRingVec, LevelParams, MRowLayout,
    RingCommitment, RingOpeningPoint, RingRelationInstance, RingSubfieldEncoding, Schedule,
    SetupContributionMode, SetupSumcheckProof, Step, TerminalLevelProof,
    TerminalWitnessSegmentLayout, TerminalWitnessTranscriptParts,
};
#[cfg(feature = "zk")]
use zk::{verify_zk_hiding_commitment, zk_recovered_y_ring_lc};

mod recursive;
mod root_eor;
mod stage2_replay;

pub(crate) use recursive::verify_fold_batched_proof;
use root_eor::verify_root_eor_and_prepare_points;
#[cfg(feature = "zk")]
use root_eor::verify_zk_extension_opening_reduction_sumcheck;
use stage2_replay::{
    stage3_sumcheck_proof_for_mode, verify_stage2_and_setup_replay, Stage2ProofReplay,
    Stage2ReplayInput,
};

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
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_E_HAT, &parts.e_hat);
    Ok(TerminalWitnessReplay { parts })
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

pub(super) struct Stage1Replay<E: FieldCore> {
    pub(super) batching_coeff: E,
    pub(super) s_claim: E,
    pub(super) stage1_point: Vec<E>,
    #[cfg(feature = "zk")]
    pub(super) s_claim_mask: ZkR1csLinearCombination<E>,
}

pub(super) struct Stage1ReplayInput<'a, E: FieldCore> {
    pub(super) proof: Option<&'a AkitaStage1Proof<E>>,
    pub(super) tau0: &'a [E],
    pub(super) col_bits: usize,
    pub(super) ring_bits: usize,
    pub(super) b: usize,
}

pub(super) fn verify_stage1_or_terminal<F, E, T>(
    input: Stage1ReplayInput<'_, E>,
    transcript: &mut T,
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<E>,
) -> Result<Stage1Replay<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let Stage1ReplayInput {
        proof,
        tau0,
        col_bits,
        ring_bits,
        b,
    } = input;
    let num_rounds = col_bits
        .checked_add(ring_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("stage-1 variable count overflow".to_string()))?;
    if let Some(stage1_proof) = proof {
        let tau0_reordered = reorder_stage1_coords(tau0, col_bits, ring_bits);
        let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, b);
        #[cfg(not(feature = "zk"))]
        let stage1_point = {
            let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
            stage1_verifier.verify::<F, T>(stage1_proof, transcript)?
        };
        #[cfg(feature = "zk")]
        let (stage1_point, s_claim_mask) = {
            let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
            stage1_verifier.verify::<F, T>(
                stage1_proof,
                transcript,
                zk_relations,
                zk_hiding_cursor,
            )?
        };
        transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1_proof.s_claim);
        let batching_coeff: E =
            sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
        return Ok(Stage1Replay {
            batching_coeff,
            s_claim: stage1_proof.s_claim,
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
    let v_typed_owned: Vec<CyclotomicRing<F, D>>;
    let v_typed: &[CyclotomicRing<F, D>] = match &proof {
        RootLevelProofView::Intermediate { v_flat, .. } => v_flat.as_ring_slice::<D>()?,
        RootLevelProofView::Terminal { .. } => {
            v_typed_owned = Vec::new();
            &v_typed_owned
        }
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

    // `y_ring` is standalone wire data pinned at the EOR output point ρ (see
    // `ExtensionOpeningReductionVerifier::expected_output_claim`). A future
    // hardening could absorb `y_rings` before the EOR sumcheck so EOR is
    // transcript-self-contained; that is a breaking prover/verifier reorder and
    // must ship with coordinated prover edits plus a full round-trip test. Today
    // the downstream relation sumcheck challenges are sampled after this absorb,
    // so the current order is not exploitable.
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
            if row.point_idx() >= factors_by_point.len() || row.point_idx() >= prepared_points.len()
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
        stage1_challenges.clone(),
        ring_opening_points.clone(),
        ring_multiplier_points.clone(),
        incidence_summary.clone(),
        commitment_routing,
        gamma,
        row_coefficient_rings,
        y_rings.to_vec(),
        v_typed.to_vec(),
    )?;
    relation_instance.check_v_shape_for_level(batched_lp)?;
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
                &replay.parts,
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
        Stage1ReplayInput {
            proof: stage1_proof,
            tau0: &rs.tau0,
            col_bits: rs.col_bits,
            ring_bits: rs.ring_bits,
            b: rs.b,
        },
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
        setup_sumcheck_proof: stage3_sumcheck_proof,
        next_fold_level_params,
        ring_multiplier_points: &ring_multiplier_points,
        v: v_typed,
        u: commitment_rows,
        y_rings,
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
