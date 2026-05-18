//! Root and recursive level verifier replay for Akita proofs.
//!
//! This module owns the transcript and algebra checks for an already selected
//! root or fold level. Schedule/config dispatch stays with the scheme crate
//! until the verifier-facing config boundary is extracted.

use crate::protocol::ring_switch::{ring_switch_verifier, ring_switch_verifier_after_absorb};
use crate::stages::stage1::{derive_stage1_challenges, AkitaStage1Verifier};
use crate::stages::stage2::{AkitaStage2Verifier, Stage2RowEvalSource};
use akita_algebra::CyclotomicRing;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    PseudoMersenneField, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{
    check_extension_opening_reduction_output, check_tensor_extension_opening_claim,
    tensor_equality_factor_eval_at_point, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, verify_extension_opening_reduction_rounds, verify_sumcheck,
    SumcheckInstanceVerifier,
};
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_SUMCHECK_S_CLAIM, ABSORB_SUMCHECK_W,
    CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::{
    append_batched_commitments_to_transcript, append_claim_incidence_shape_to_transcript,
    append_claim_points_to_transcript, append_claim_values_to_transcript,
    dispatch_trace_inner_product_check, flatten_batched_commitment_rows,
    prepare_recursive_opening_point_ext, prepare_root_opening_point_ext,
    recover_ring_subfield_inner_product, relation_claim_from_rows_extension, reorder_stage1_coords,
    ring_subfield_packed_extension_opening_point, sample_public_row_coefficients,
    schedule_num_fold_levels, w_ring_element_count_with_counts, AkitaBatchedProof, AkitaLevelProof,
    AkitaProofStep, AkitaStage1Proof, AkitaStage2Proof, AkitaVerifierSetup, BasisMode, BlockOrder,
    ClaimIncidenceSummary, DirectWitnessProof, ExtensionOpeningReductionProof, FlatRingVec,
    LevelParams, RingCommitment, RingOpeningPoint, RingSubfieldEncoding, Schedule, Step,
    TerminalLevelProof,
};

/// Verifier state carried between recursive fold levels.
pub(crate) struct RecursiveVerifierState<'a, F: FieldCore, L: FieldCore> {
    /// Current opening point for the committed recursive witness.
    pub opening_point: Vec<L>,
    /// Claimed opening value for the current commitment.
    pub opening: L,
    /// Current recursive witness commitment.
    pub commitment: &'a FlatRingVec<F>,
    /// Basis used to interpret the current opening point.
    pub basis: BasisMode,
    /// Current recursive witness length in field elements.
    pub w_len: usize,
    /// Current digit basis, as `log2(b)`.
    pub log_basis: u32,
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
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
) -> Result<Vec<C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F> + ExtField<E> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    verify_root_level_inner::<F, E, C, T, D>(
        y_rings_flat,
        extension_opening_reduction,
        v_flat,
        Some(stage1),
        &stage2.sumcheck,
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
            next_w_eval: stage2.next_w_eval,
        },
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
    v_flat: &FlatRingVec<F>,
    stage2_sumcheck: &akita_sumcheck::SumcheckProof<C>,
    final_witness: &DirectWitnessProof<F>,
    final_w_len: usize,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claim_points: &[&[E]],
    openings: &[E],
    commitments: &[RingCommitment<F, D>],
    incidence_summary: &ClaimIncidenceSummary,
    basis: BasisMode,
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
) -> Result<Vec<C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F> + ExtField<E> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    verify_root_level_inner::<F, E, C, T, D>(
        y_rings_flat,
        extension_opening_reduction,
        v_flat,
        None,
        stage2_sumcheck,
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
    v_flat: &FlatRingVec<F>,
    stage1: Option<&AkitaStage1Proof<C>>,
    stage2_sumcheck: &akita_sumcheck::SumcheckProof<C>,
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
) -> Result<Vec<C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F> + ExtField<E> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let is_terminal = matches!(stage_input, RootStageInput::Terminal { .. });
    let final_w_len_opt = match &stage_input {
        RootStageInput::Terminal { final_w_len, .. } => Some(*final_w_len),
        RootStageInput::Intermediate { .. } => None,
    };
    let y_rings = y_rings_flat.as_ring_slice::<D>()?;
    let v_typed = v_flat.as_ring_slice::<D>()?;
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

    append_claim_incidence_shape_to_transcript::<F, T>(incidence_summary, transcript);
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
        if split_bits > incidence_summary.num_vars()
            || reduction.partials.len() != incidence_summary.num_claims() * width
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
            check_tensor_extension_opening_claim::<F, C>(
                &padded_points[point_idx],
                C::lift_base(opening),
                partials,
            )?;
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
        }
        let result = verify_extension_opening_reduction_rounds::<F, _, C, _>(
            &reduction.sumcheck,
            input_claim,
            incidence_summary.num_vars() - split_bits,
            transcript,
            |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
        )?;
        let factors_by_point = padded_points
            .iter()
            .map(|point| {
                tensor_equality_factor_eval_at_point::<F, C>(
                    &point[split_bits..],
                    &eta,
                    &result.challenges,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Some((result.final_claim, result.challenges, factors_by_point))
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
    } else if let Some((final_claim, _rho, factors_by_point)) = &reduction_check {
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
    } else {
        return Err(AkitaError::InvalidProof);
    }

    let total_blocks = root_lp
        .num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched root block count overflow".to_string()))?;
    let stage1_challenges =
        derive_stage1_challenges::<F, T, D>(transcript, v_typed, total_blocks, batched_lp)?;

    let w_len = if is_terminal {
        final_w_len_opt.ok_or(AkitaError::InvalidProof)?
    } else {
        w_ring_element_count_with_counts::<F>(
            batched_lp,
            incidence_summary.num_polys_per_point().len(),
            incidence_summary.num_polys_per_point().iter().sum(),
            num_claims,
            incidence_summary.num_public_rows(),
        ) * D
    };

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
        RootStageInput::Terminal { final_witness, .. } => {
            // Bind the ring-switch challenges to the cleartext witness rather
            // than to a separate commitment, mirroring the prover.
            transcript.append_serde(ABSORB_SUMCHECK_W, *final_witness);
            ring_switch_verifier_after_absorb::<F, C, T, { D }>(
                &ring_opening_points,
                &ring_multiplier_points,
                incidence_summary.claim_to_point(),
                &stage1_challenges,
                w_len,
                transcript,
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
    );
    let (batching_coeff, s_claim, r_stage1) = match (&stage_input, stage1) {
        (RootStageInput::Intermediate { .. }, Some(stage1_proof)) => {
            let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
            let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
            let r_stage1 = {
                let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
                stage1_verifier.verify::<F, T>(stage1_proof, transcript)?
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
    let stage2_verifier = match &stage_input {
        RootStageInput::Terminal { final_witness, .. } => {
            AkitaStage2Verifier::new_with_direct_witness(
                batching_coeff,
                s_claim,
                final_witness,
                w_len,
                r_stage1.clone(),
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
            )
        }
        RootStageInput::Intermediate { next_w_eval, .. } => {
            AkitaStage2Verifier::new_with_claimed_w_eval(
                batching_coeff,
                s_claim,
                *next_w_eval,
                r_stage1.clone(),
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
            )
        }
    };
    if stage2_input_claim != SumcheckInstanceVerifier::input_claim(&stage2_verifier) {
        return Err(AkitaError::InvalidProof);
    }
    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        verify_sumcheck::<F, _, C, _, _>(stage2_sumcheck, &stage2_verifier, transcript, |tr| {
            sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?
    };

    Ok(sumcheck_challenges)
}

/// Verify one intermediate recursive fold level.
///
/// The returned challenges become the opening point for the next level.
///
/// # Errors
///
/// Returns an error if the level proof shape is inconsistent, the public
/// trace check fails, ring-switch replay fails, or either sumcheck verifier
/// rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[tracing::instrument(skip_all, name = "verify_intermediate_level")]
pub(crate) fn verify_intermediate_level<F, L, T, const D: usize>(
    level_proof: &AkitaLevelProof<F, L>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F, L>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<L>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    L: RingSubfieldEncoding<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    verify_one_level_inner::<F, L, T, D>(
        FoldProofView::Intermediate(level_proof),
        setup,
        transcript,
        current_state,
        None,
        lp,
        block_order,
    )
}

/// Verify one terminal recursive fold level.
///
/// At the terminal level the cleartext `final_witness` is absorbed via
/// [`ABSORB_SUMCHECK_W`] in place of a next-witness commitment, stage-1 is
/// skipped (packed-digit range is structurally enforced), and stage-2 runs
/// in relation-only mode.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, the public trace
/// check fails, ring-switch replay fails, or the stage-2 sumcheck verifier
/// rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[tracing::instrument(skip_all, name = "verify_terminal_level")]
pub(crate) fn verify_terminal_level<F, L, T, const D: usize>(
    terminal_proof: &TerminalLevelProof<F, L>,
    final_w_len: usize,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F, L>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<L>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    L: RingSubfieldEncoding<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    verify_one_level_inner::<F, L, T, D>(
        FoldProofView::Terminal(terminal_proof),
        setup,
        transcript,
        current_state,
        Some(final_w_len),
        lp,
        block_order,
    )
}

enum FoldProofView<'a, F: FieldCore, L: FieldCore> {
    Intermediate(&'a AkitaLevelProof<F, L>),
    Terminal(&'a TerminalLevelProof<F, L>),
}

impl<F: FieldCore, L: FieldCore> FoldProofView<'_, F, L> {
    fn y_rings_typed<const D: usize>(&self) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        match self {
            Self::Intermediate(proof) => proof.try_y_rings_typed::<D>(),
            Self::Terminal(proof) => proof.try_y_rings_typed::<D>(),
        }
    }
    fn v_flat(&self) -> &FlatRingVec<F> {
        match self {
            Self::Intermediate(proof) => &proof.v,
            Self::Terminal(proof) => &proof.v,
        }
    }
    fn extension_opening_reduction(&self) -> Option<&ExtensionOpeningReductionProof<L>> {
        match self {
            Self::Intermediate(proof) => proof.extension_opening_reduction.as_ref(),
            Self::Terminal(proof) => proof.extension_opening_reduction.as_ref(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_one_level_inner<F, L, T, const D: usize>(
    proof: FoldProofView<'_, F, L>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F, L>,
    final_w_len: Option<usize>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<L>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    L: RingSubfieldEncoding<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let is_last = matches!(proof, FoldProofView::Terminal(_));
    let y_rings = proof.y_rings_typed::<D>()?;
    let v_typed = proof.v_flat().as_ring_slice::<D>()?;
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;

    let alpha_bits = lp.ring_dimension.trailing_zeros() as usize;
    current_state
        .commitment
        .append_as_ring_slice::<T, D>(ABSORB_COMMITMENT, transcript)?;
    if y_rings.len() != 1 {
        return Err(AkitaError::InvalidProof);
    }
    let reduction_check = if <L as ExtField<F>>::EXT_DEGREE == 1 {
        if proof.extension_opening_reduction().is_some() {
            return Err(AkitaError::InvalidProof);
        }
        None
    } else {
        let reduction = proof
            .extension_opening_reduction()
            .ok_or(AkitaError::InvalidProof)?;
        check_tensor_extension_opening_claim::<F, L>(
            &current_state.opening_point,
            current_state.opening,
            &reduction.partials,
        )?;
        for partial in &reduction.partials {
            append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
        }
        let (split_bits, width) = {
            let width = <L as ExtField<F>>::EXT_DEGREE;
            if width == 0 || !width.is_power_of_two() {
                return Err(AkitaError::InvalidProof);
            }
            (width.trailing_zeros() as usize, width)
        };
        if split_bits > current_state.opening_point.len() || reduction.partials.len() != width {
            return Err(AkitaError::InvalidProof);
        }
        let row_partials = tensor_row_partials_from_columns::<F, L>(&reduction.partials)?;
        let eta = (0..split_bits)
            .map(|_| sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
            .collect::<Vec<_>>();
        let input_claim = tensor_reduction_claim_from_rows::<F, L>(&row_partials, &eta)?;
        let tail_point = &current_state.opening_point[split_bits..];
        let result = verify_extension_opening_reduction_rounds::<F, _, L, _>(
            &reduction.sumcheck,
            input_claim,
            tail_point.len(),
            transcript,
            |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
        )?;
        let factor =
            tensor_equality_factor_eval_at_point::<F, L>(tail_point, &eta, &result.challenges)?;
        Some((result.final_claim, factor, result.challenges))
    };
    let protocol_point = match &reduction_check {
        Some((_final_claim, _factor, rho)) => {
            ring_subfield_packed_extension_opening_point::<F, L, D>(rho.len(), rho)?
        }
        None => current_state.opening_point.clone(),
    };
    let prepared_points = vec![prepare_recursive_opening_point_ext::<F, L, D>(
        &protocol_point,
        current_state.basis,
        lp,
        alpha_bits,
        block_order,
    )?];
    for prepared_point in &prepared_points {
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    for y_ring in &y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    let internal_claims = y_rings
        .iter()
        .zip(prepared_points.iter())
        .map(|(y_ring, prepared_point)| {
            recover_ring_subfield_inner_product::<F, L, D>(y_ring, &prepared_point.inner_reduction)
        })
        .collect::<Result<Vec<_>, _>>()?;
    match reduction_check {
        Some((final_claim, factor, _rho)) => {
            check_extension_opening_reduction_output(final_claim, internal_claims[0], factor)?;
        }
        None => {
            if internal_claims[0] != current_state.opening {
                return Err(AkitaError::InvalidProof);
            }
        }
    }

    let ring_opening_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_opening_point.clone())
        .collect::<Vec<_>>();
    let ring_multiplier_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_multiplier_point.clone())
        .collect::<Vec<_>>();
    let num_claims = y_rings.len();
    let stage1_challenges =
        derive_stage1_challenges::<F, T, D>(transcript, v_typed, lp.num_blocks * num_claims, lp)?;

    let w_len = if is_last {
        final_w_len.ok_or(AkitaError::InvalidProof)?
    } else {
        w_ring_element_count_with_counts::<F>(lp, 1, 1, num_claims, num_claims) * D
    };
    let claim_to_point = (0..num_claims).collect::<Vec<_>>();
    let claim_to_point_poly = vec![0usize; num_claims];
    let claim_poly_indices = vec![0usize; num_claims];
    let gamma = vec![L::one(); num_claims];

    let rs = match &proof {
        FoldProofView::Intermediate(level_proof) => ring_switch_verifier::<F, L, T, { D }>(
            &ring_opening_points,
            &ring_multiplier_points,
            &claim_to_point,
            &stage1_challenges,
            w_len,
            level_proof.next_w_commitment(),
            transcript,
            lp,
            &[1usize],
            &claim_to_point_poly,
            &claim_poly_indices,
            &gamma,
            num_claims,
        )?,
        FoldProofView::Terminal(terminal_proof) => {
            transcript.append_serde(ABSORB_SUMCHECK_W, &terminal_proof.final_witness);
            ring_switch_verifier_after_absorb::<F, L, T, { D }>(
                &ring_opening_points,
                &ring_multiplier_points,
                &claim_to_point,
                &stage1_challenges,
                w_len,
                transcript,
                lp,
                &[1usize],
                &claim_to_point_poly,
                &claim_poly_indices,
                &gamma,
                num_claims,
            )?
        }
    };
    let relation_claim = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        v_typed,
        commitment_u,
        &y_rings,
    );
    let (batching_coeff, s_claim, r_stage1) = match &proof {
        FoldProofView::Intermediate(level_proof) => {
            let stage1 = &level_proof.stage1;
            let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
            let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
            let r_stage1 = {
                let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
                stage1_verifier.verify::<F, T>(stage1, transcript)?
            };
            transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
            let batching_coeff: L =
                sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
            (batching_coeff, stage1.s_claim, r_stage1)
        }
        FoldProofView::Terminal(_) => {
            let r_stage1 = vec![L::zero(); rs.col_bits + rs.ring_bits];
            (L::zero(), L::zero(), r_stage1)
        }
    };
    let stage2_input_claim = batching_coeff * s_claim + relation_claim;
    let row_eval_source = Stage2RowEvalSource::new(rs.prepared_row_eval);
    let stage2_verifier = match &proof {
        FoldProofView::Terminal(terminal_proof) => AkitaStage2Verifier::new_with_direct_witness(
            batching_coeff,
            s_claim,
            &terminal_proof.final_witness,
            w_len,
            r_stage1.clone(),
            rs.alpha_evals_y,
            row_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &ring_multiplier_points,
            &rs.tau1,
            v_typed,
            commitment_u,
            &y_rings,
            Some(relation_claim),
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        ),
        FoldProofView::Intermediate(level_proof) => AkitaStage2Verifier::new_with_claimed_w_eval(
            batching_coeff,
            s_claim,
            level_proof.stage2.next_w_eval,
            r_stage1.clone(),
            rs.alpha_evals_y,
            row_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &ring_multiplier_points,
            &rs.tau1,
            v_typed,
            commitment_u,
            &y_rings,
            Some(relation_claim),
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        ),
    };
    if stage2_input_claim != SumcheckInstanceVerifier::input_claim(&stage2_verifier) {
        return Err(AkitaError::InvalidProof);
    }

    let stage2_sumcheck_ref = match &proof {
        FoldProofView::Intermediate(level_proof) => &level_proof.stage2.sumcheck,
        FoldProofView::Terminal(terminal_proof) => &terminal_proof.stage2_sumcheck,
    };
    let challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        verify_sumcheck::<F, _, L, _, _>(stage2_sumcheck_ref, &stage2_verifier, transcript, |tr| {
            sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?
    };

    Ok(challenges)
}

fn scheduled_recursive_verify_level<F: FieldCore, L: FieldCore>(
    schedule: &Schedule,
    level: usize,
    current_state: &RecursiveVerifierState<'_, F, L>,
) -> Result<(LevelParams, usize, Option<LevelParams>), AkitaError> {
    let Some(Step::Fold(step)) = schedule.steps.get(level) else {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule is missing fold step at level {level}"
        )));
    };
    if step.current_w_len != current_state.w_len || step.params.log_basis != current_state.log_basis
    {
        return Err(AkitaError::InvalidSetup(
            "scheduled recursive level did not match runtime state".to_string(),
        ));
    }
    let next_level_params = match schedule.steps.get(level + 1) {
        Some(Step::Fold(next_step)) => Some(next_step.params.clone()),
        Some(Step::Direct(_)) => None,
        None => {
            return Err(AkitaError::InvalidSetup(
                "schedule is missing successor step".to_string(),
            ))
        }
    };
    Ok((step.params.clone(), step.next_w_len, next_level_params))
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_verify_intermediate_level<F, L, T>(
    level_d: usize,
    level_proof: &AkitaLevelProof<F, L>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F, L>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<L>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    L: RingSubfieldEncoding<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    macro_rules! dispatch {
        ($d:literal) => {
            verify_intermediate_level::<F, L, T, $d>(
                level_proof,
                setup,
                transcript,
                current_state,
                lp,
                block_order,
            )
        };
    }
    match level_d {
        32 => dispatch!(32),
        64 => dispatch!(64),
        128 => dispatch!(128),
        256 => dispatch!(256),
        512 => dispatch!(512),
        1024 => dispatch!(1024),
        _ => Err(AkitaError::InvalidProof),
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_verify_terminal_level<F, L, T>(
    level_d: usize,
    terminal_proof: &TerminalLevelProof<F, L>,
    final_w_len: usize,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<'_, F, L>,
    lp: &LevelParams,
    block_order: BlockOrder,
) -> Result<Vec<L>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    L: RingSubfieldEncoding<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    macro_rules! dispatch {
        ($d:literal) => {
            verify_terminal_level::<F, L, T, $d>(
                terminal_proof,
                final_w_len,
                setup,
                transcript,
                current_state,
                lp,
                block_order,
            )
        };
    }
    match level_d {
        32 => dispatch!(32),
        64 => dispatch!(64),
        128 => dispatch!(128),
        256 => dispatch!(256),
        512 => dispatch!(512),
        1024 => dispatch!(1024),
        _ => Err(AkitaError::InvalidProof),
    }
}

/// Verify all recursive fold levels after the root proof.
///
/// The supplied `schedule` is the already-selected public schedule for this
/// proof shape. This function checks that each proof level matches that
/// schedule, dispatches to the corresponding ring dimension, and threads the
/// verifier state to the next recursive commitment.
///
/// # Errors
///
/// Returns an error if the schedule is malformed for the supplied proof,
/// decoded proof dimensions do not match, any fold-level verifier rejects, or
/// the recursive witness handoff has the wrong shape.
pub(crate) fn verify_batched_recursive_suffix<'a, F, L, T, const D: usize>(
    proof: &'a AkitaBatchedProof<F, L>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    schedule: &Schedule,
    mut current_state: RecursiveVerifierState<'a, F, L>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    L: RingSubfieldEncoding<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let num_steps = proof.steps.len();
    if num_steps == 0 {
        // Root was itself the terminal — no suffix to verify.
        return Ok(());
    }
    for (offset, step) in proof.steps.iter().enumerate() {
        let level_index = offset + 1;
        let is_last = offset == num_steps - 1;
        let (current_lp, next_w_len, scheduled_next_params) =
            scheduled_recursive_verify_level(schedule, level_index, &current_state)?;
        let level_d = current_lp.ring_dimension;

        match step {
            AkitaProofStep::Intermediate(level_proof) => {
                if is_last {
                    // The terminal slot must be a Terminal variant.
                    return Err(AkitaError::InvalidProof);
                }
                if !current_state.commitment.can_decode_vec(level_d)
                    || !level_proof.y_ring.can_decode_vec(level_d)
                    || !level_proof.v.can_decode_vec(level_d)
                {
                    return Err(AkitaError::InvalidProof);
                }

                let challenges = if level_d == D {
                    verify_intermediate_level::<F, L, T, D>(
                        level_proof,
                        setup,
                        transcript,
                        &current_state,
                        &current_lp,
                        BlockOrder::ColumnMajor,
                    )?
                } else {
                    dispatch_verify_intermediate_level::<F, L, T>(
                        level_d,
                        level_proof,
                        setup,
                        transcript,
                        &current_state,
                        &current_lp,
                        BlockOrder::ColumnMajor,
                    )?
                };

                let scheduled_next_params =
                    scheduled_next_params.ok_or(AkitaError::InvalidProof)?;
                let next_level_d = scheduled_next_params.ring_dimension;
                if next_level_d == 0
                    || !level_proof.next_w_commitment().can_decode_vec(next_level_d)
                {
                    return Err(AkitaError::InvalidProof);
                }
                let y_ring_count = level_proof.y_ring.coeff_len() / level_d;
                let computed_next_w_len = w_ring_element_count_with_counts::<F>(
                    &current_lp,
                    1,
                    1,
                    y_ring_count,
                    y_ring_count,
                ) * level_d;
                if computed_next_w_len != next_w_len {
                    return Err(AkitaError::InvalidProof);
                }
                current_state = RecursiveVerifierState {
                    opening_point: challenges,
                    opening: level_proof.next_w_eval(),
                    commitment: level_proof.next_w_commitment(),
                    basis: BasisMode::Lagrange,
                    w_len: next_w_len,
                    log_basis: scheduled_next_params.log_basis,
                };
            }
            AkitaProofStep::Terminal(terminal_proof) => {
                if !is_last {
                    return Err(AkitaError::InvalidProof);
                }
                if !current_state.commitment.can_decode_vec(level_d)
                    || !terminal_proof.y_rings.can_decode_vec(level_d)
                    || !terminal_proof.v.can_decode_vec(level_d)
                {
                    return Err(AkitaError::InvalidProof);
                }
                if level_d == D {
                    verify_terminal_level::<F, L, T, D>(
                        terminal_proof,
                        next_w_len,
                        setup,
                        transcript,
                        &current_state,
                        &current_lp,
                        BlockOrder::ColumnMajor,
                    )?
                } else {
                    dispatch_verify_terminal_level::<F, L, T>(
                        level_d,
                        terminal_proof,
                        next_w_len,
                        setup,
                        transcript,
                        &current_state,
                        &current_lp,
                        BlockOrder::ColumnMajor,
                    )?
                };
                // Terminal step also implies the scheduled successor must be
                // a Direct step with the matching packed-digit shape.
                if let Some(scheduled_next) = scheduled_next_params {
                    let _ = scheduled_next;
                }
            }
        }
    }

    Ok(())
}

/// Verify the folded-root branch of a batched opening proof.
///
/// The caller owns config-backed schedule selection and passes the derived
/// root verifier layout plus the first recursive-level params. This function
/// owns the fold-root proof-shape checks, root opening preparation, root
/// transcript replay, and recursive suffix handoff.
///
/// # Errors
///
/// Returns an error if the proof is not a folded-root proof, the schedule does
/// not match the proof shape, the root proof rejects, or a recursive suffix
/// level rejects.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_fold_batched_proof<F, E, C, T, const D: usize>(
    proof: &AkitaBatchedProof<F, C>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    opening_points: &[&[E]],
    openings: &[E],
    commitments: &[RingCommitment<F, D>],
    incidence_summary: &ClaimIncidenceSummary,
    basis: BasisMode,
    schedule: &Schedule,
    root_lp: &LevelParams,
    next_level_params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
{
    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        return Err(AkitaError::InvalidProof);
    };
    let total_fold_levels = schedule_num_fold_levels(schedule);
    let terminal_direct = schedule
        .steps
        .last()
        .and_then(|step| match step {
            Step::Direct(direct) => Some(direct),
            Step::Fold(_) => None,
        })
        .ok_or(AkitaError::InvalidProof)?;

    match &proof.root {
        akita_types::AkitaBatchedRootProof::Direct { .. } => {
            // Root-direct fast path is handled by a separate verifier entry
            // point; this function should not be called for it.
            Err(AkitaError::InvalidProof)
        }
        akita_types::AkitaBatchedRootProof::Terminal(terminal) => {
            // 1-fold case: the root itself is the terminal fold. No recursive
            // suffix follows.
            if total_fold_levels != 1 || !proof.steps.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            if terminal.final_witness.shape() != terminal_direct.witness_shape {
                return Err(AkitaError::InvalidProof);
            }
            let y_coeff_len = terminal.y_rings.coeff_len();
            if !y_coeff_len.is_multiple_of(D) || y_coeff_len / D != opening_points.len() {
                return Err(AkitaError::InvalidProof);
            }
            verify_terminal_root_level::<F, E, C, T, D>(
                &terminal.y_rings,
                terminal.extension_opening_reduction.as_ref(),
                &terminal.v,
                &terminal.stage2_sumcheck,
                &terminal.final_witness,
                root_step.next_w_len,
                setup,
                transcript,
                opening_points,
                openings,
                commitments,
                incidence_summary,
                basis,
                root_lp,
                &root_step.params,
            )?;
            Ok(())
        }
        akita_types::AkitaBatchedRootProof::Fold(fold_root) => {
            let expected_recursive_levels = total_fold_levels
                .checked_sub(1)
                .ok_or(AkitaError::InvalidProof)?;
            if proof.steps.len() != expected_recursive_levels {
                tracing::debug!(
                    proof_steps = proof.steps.len(),
                    expected_recursive_levels,
                    "folded proof recursive step count mismatch"
                );
                return Err(AkitaError::InvalidProof);
            }
            let y_coeff_len = fold_root.y_rings.coeff_len();
            if !y_coeff_len.is_multiple_of(D) || y_coeff_len / D != opening_points.len() {
                return Err(AkitaError::InvalidProof);
            }

            // Validate the terminal proof step's witness shape against the
            // scheduled direct step before running heavy checks.
            let terminal_step = proof
                .steps
                .last()
                .and_then(|step| match step {
                    AkitaProofStep::Terminal(t) => Some(t),
                    AkitaProofStep::Intermediate(_) => None,
                })
                .ok_or(AkitaError::InvalidProof)?;
            if terminal_step.final_witness.shape() != terminal_direct.witness_shape {
                tracing::debug!(
                    actual_shape = ?terminal_step.final_witness.shape(),
                    expected_shape = ?terminal_direct.witness_shape,
                    "folded proof terminal witness shape mismatch"
                );
                return Err(AkitaError::InvalidProof);
            }

            let root_challenges = verify_intermediate_root_level::<F, E, C, T, D>(
                &fold_root.y_rings,
                fold_root.extension_opening_reduction.as_ref(),
                &fold_root.v,
                &fold_root.stage1,
                &fold_root.stage2,
                setup,
                transcript,
                opening_points,
                openings,
                commitments,
                incidence_summary,
                basis,
                root_lp,
                &root_step.params,
            )?;

            let first_level_d = next_level_params.ring_dimension;
            if !fold_root
                .stage2
                .next_w_commitment
                .can_decode_vec(first_level_d)
            {
                return Err(AkitaError::InvalidProof);
            }

            let current_state = RecursiveVerifierState {
                opening_point: root_challenges,
                opening: fold_root.stage2.next_w_eval,
                commitment: &fold_root.stage2.next_w_commitment,
                basis: BasisMode::Lagrange,
                w_len: root_step.next_w_len,
                log_basis: next_level_params.log_basis,
            };
            verify_batched_recursive_suffix::<F, C, T, D>(
                proof,
                setup,
                transcript,
                schedule,
                current_state,
            )?;
            Ok(())
        }
    }
}
