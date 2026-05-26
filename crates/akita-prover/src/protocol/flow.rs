//! Prover flow state shared by root orchestration during crate extraction.

use crate::kernels::crt_ntt::NttSlotCache;
use crate::protocol::ring_switch::{
    commit_next_w, ring_switch_build_w, ring_switch_finalize, ring_switch_finalize_terminal,
    ring_switch_finalize_terminal_with_gamma, ring_switch_finalize_with_gamma,
    NextWitnessCommitment, RingSwitchOutput, TerminalRingSwitchOutput,
};
use crate::protocol::sumcheck::{AkitaStage1Prover, AkitaStage2Prover};
use crate::{
    AkitaPolyOps, AkitaProverSetup, CommittedPolynomials, MultiDNttCaches, ProverClaims,
    QuadraticEquation, RecursiveCommitmentHintCache, RecursiveWitnessFlat, RecursiveWitnessView,
    RootTensorProjectionPoly,
};
use akita_algebra::CyclotomicRing;
use akita_config::{bind_transcript_instance_descriptor, CommitmentConfig};
use akita_field::fields::wide::HasWide;
use akita_field::fields::HasUnreducedOps;
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, Invertible, PseudoMersenneField, RandomSampling,
};
use akita_serialization::{AkitaSerialize, Valid};
use akita_sumcheck::{
    check_extension_opening_reduction_output, check_tensor_extension_opening_claim,
    prove_extension_opening_reduction, prove_sumcheck, tensor_equality_factor_eval_at_point,
    tensor_equality_factor_evals, tensor_logical_claim_from_partials, tensor_opening_split,
    tensor_packed_witness_evals, tensor_partials_from_base_evals, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, BatchedExtensionOpeningReductionProver,
    BatchedExtensionOpeningReductionTerm, ExtensionOpeningReductionProver, SumcheckInstanceProver,
    SumcheckProof, SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS,
};
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_SUMCHECK_S_CLAIM, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::{
    append_batched_commitments_to_transcript, append_claim_incidence_shape_to_transcript,
    append_claim_points_to_transcript, append_claim_values_to_transcript, basis_weights,
    embed_ring_subfield_scalar, embed_ring_subfield_vector, flatten_batched_commitment_rows,
    folded_root_supports_opening_shape, prepare_recursive_opening_point_ext,
    prepare_root_opening_point_ext, recover_ring_subfield_inner_product,
    relation_claim_from_rows_extension, reorder_stage1_coords,
    ring_subfield_packed_extension_opening_point, root_direct_schedule,
    root_extension_opening_partials, root_tensor_projection_enabled,
    sample_public_row_coefficients, schedule_is_root_direct, schedule_num_fold_levels,
    schedule_root_fold_step, schedule_terminal_direct_witness_shape, scheduled_fold_execution,
    scheduled_next_level_params, terminal_witness_segment_layout, validate_batched_inputs,
    AkitaBatchedProof, AkitaBatchedRootProof, AkitaCommitmentHint, AkitaExpandedSetup,
    AkitaLevelProof, AkitaProofStep, AkitaScheduleInputs, AkitaStage1Proof, BasisMode, BlockOrder,
    ClaimIncidence, ClaimIncidenceLimits, ClaimIncidenceSummary, DirectWitnessProof,
    DirectWitnessShape, ExtensionOpeningReductionProof, FlatRingVec, IncidenceClaim, LevelParams,
    MRowLayout, PackedDigits, PreparedRootOpeningPoint, RingCommitment, RingMultiplierOpeningPoint,
    RingSubfieldEncoding, Schedule, Step, TerminalLevelProof, TerminalWitnessSegmentLayout,
};

/// Runtime state carried between recursive prove levels.
pub struct RecursiveProverState<F: FieldCore, L: FieldCore> {
    /// Current committed recursive witness representation.
    pub w: RecursiveWitnessFlat,
    /// Logical recursive witness when it differs from the committed representation.
    pub logical_w: Option<RecursiveWitnessFlat>,
    /// Current recursive witness commitment.
    pub commitment: FlatRingVec<F>,
    /// D-erased recursive commitment hint cache.
    pub hint: RecursiveCommitmentHintCache<F>,
    /// Current digit basis, as `log2(b)`.
    pub log_basis: u32,
    /// Sumcheck challenges that become the next recursive opening point.
    pub sumcheck_challenges: Vec<L>,
    /// Claimed logical opening of `logical_w` at `sumcheck_challenges`.
    pub opening: L,
}

impl<F: FieldCore, L: FieldCore> RecursiveProverState<F, L> {
    /// Logical witness represented by the carried opening claim.
    #[inline]
    pub fn logical_w(&self) -> &RecursiveWitnessFlat {
        self.logical_w.as_ref().unwrap_or(&self.w)
    }
}

/// Output from a single prove level, used to extend proof wire data and state.
pub struct ProveLevelOutput<F: FieldCore, L: FieldCore> {
    /// Fold proof produced at this level.
    pub level_proof: AkitaLevelProof<F, L>,
    /// Recursive prover state for the next level.
    pub next_state: RecursiveProverState<F, L>,
}

fn terminal_segment_layout_for_quad<F, const D: usize>(
    quad_eq: &QuadraticEquation<F, D>,
    lp: &LevelParams,
) -> Result<TerminalWitnessSegmentLayout, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    terminal_witness_segment_layout(
        lp,
        quad_eq.claim_to_point().len(),
        quad_eq.num_public_rows(),
    )
}

fn pack_scheduled_terminal_witness<F: FieldCore>(
    logical_w: &RecursiveWitnessFlat,
    expected_shape: &DirectWitnessShape,
) -> Result<DirectWitnessProof<F>, AkitaError> {
    let DirectWitnessShape::PackedDigits((expected_len, bits_per_elem)) = expected_shape else {
        return Err(AkitaError::InvalidSetup(
            "terminal fold requires a packed-digit direct step".to_string(),
        ));
    };
    if logical_w.len() != *expected_len {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled terminal witness length did not match runtime witness: expected={expected_len}, actual={}",
            logical_w.len()
        )));
    }
    if PackedDigits::required_bits_per_elem(logical_w.as_i8_digits()) > *bits_per_elem {
        return Err(AkitaError::InvalidSetup(
            "scheduled terminal witness bit width is too small".to_string(),
        ));
    }
    let witness = DirectWitnessProof::PackedDigits(PackedDigits::from_i8_digits(
        logical_w.as_i8_digits(),
        *bits_per_elem,
    ));
    if witness.shape() != *expected_shape {
        return Err(AkitaError::InvalidSetup(
            "packed terminal witness shape does not match schedule".to_string(),
        ));
    }
    Ok(witness)
}

/// Raw pieces produced by the unified root-level prover.
///
/// Callers assemble either a singleton or batched root proof from these
/// components while sharing the same inner prover flow.
pub struct RootLevelRawOutput<F: FieldCore, L: FieldCore, const D: usize> {
    /// Gamma-combined public y-rings, one per opening point.
    pub y_rings: Vec<CyclotomicRing<F, D>>,
    /// Optional extension-opening reduction payload for folded root openings.
    /// `None` when the root proof uses ordinary degree-one openings.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    /// Public v rows for the root relation.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// Stage-1 sumcheck proof.
    pub stage1: AkitaStage1Proof<L>,
    /// Stage-2 sumcheck proof.
    pub stage2_sumcheck: SumcheckProof<L>,
    /// Recursive witness commitment carried in the proof.
    pub w_commitment_proof: FlatRingVec<F>,
    /// Claimed terminal evaluation of the recursive witness at this level.
    pub w_eval: L,
    /// Recursive prover state for the first suffix level.
    pub next_state: RecursiveProverState<F, L>,
}

/// Outcome of the recursive fold suffix after the root level.
pub struct RecursiveSuffixOutcome<F: FieldCore, L: FieldCore> {
    /// Per-level intermediate fold proofs, in order. Does not include the
    /// root proof or the terminal-level proof.
    pub intermediate_levels: Vec<AkitaLevelProof<F, L>>,
    /// Terminal fold proof shipping `final_witness` in cleartext.
    pub terminal: TerminalLevelProof<F, L>,
    /// Total fold-level count reached, including the root level and the
    /// terminal level.
    pub num_levels: usize,
}

fn root_claim_opening_from_y_ring<F, E, const D: usize>(
    y_ring: &CyclotomicRing<F, D>,
    prepared_point: &PreparedRootOpeningPoint<F, D>,
    inner_opening_point: &[E],
    basis: BasisMode,
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: RingSubfieldEncoding<F>,
{
    if <E as ExtField<F>>::EXT_DEGREE == 1 {
        return (*y_ring * prepared_point.inner_reduction.sigma_m1())
            .coefficients()
            .first()
            .copied()
            .map(E::lift_base)
            .ok_or_else(|| AkitaError::InvalidInput("empty root y-ring".to_string()));
    }
    if !D.is_multiple_of(<E as ExtField<F>>::EXT_DEGREE)
        || !(D / <E as ExtField<F>>::EXT_DEGREE).is_power_of_two()
    {
        return Err(AkitaError::InvalidInput(
            "claim-field degree must divide the ring dimension into power-of-two slots".to_string(),
        ));
    }
    let packed_slots = D / <E as ExtField<F>>::EXT_DEGREE;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if inner_opening_point.len() > packed_inner_bits
        && inner_opening_point[packed_inner_bits..]
            .iter()
            .any(|coord| !coord.is_zero())
    {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: inner_opening_point.len(),
        });
    }
    let mut point =
        inner_opening_point[..inner_opening_point.len().min(packed_inner_bits)].to_vec();
    point.resize(packed_inner_bits, E::zero());
    let weights = basis_weights(&point, basis)?;
    let inner_reduction = embed_ring_subfield_vector::<F, E, D>(
        &weights,
        AkitaError::InvalidInput(
            "root opening point does not encode in the ring-subfield basis".to_string(),
        ),
    )?;
    recover_ring_subfield_inner_product::<F, E, D>(y_ring, &inner_reduction)
}

fn row_coefficient_rings<F, L, const D: usize>(
    coefficients: &[L],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    L: RingSubfieldEncoding<F>,
{
    coefficients
        .iter()
        .copied()
        .map(|coefficient| {
            embed_ring_subfield_scalar::<F, L, D>(
                coefficient,
                AkitaError::InvalidInput(
                    "public-row coefficient does not encode in the ring-subfield basis".to_string(),
                ),
            )
        })
        .collect()
}

fn combine_root_y_rings<F, const D: usize>(
    per_claim_y_rings: &[CyclotomicRing<F, D>],
    incidence: &ClaimIncidenceSummary,
    row_coefficient_rings: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if per_claim_y_rings.len() != incidence.num_claims()
        || row_coefficient_rings.len() != incidence.num_claims()
        || incidence.claim_to_point().len() != incidence.num_claims()
    {
        return Err(AkitaError::InvalidInput(
            "root y-ring batching input lengths do not match".to_string(),
        ));
    }

    let mut y_rings = vec![CyclotomicRing::<F, D>::zero(); incidence.num_public_rows()];
    for (row_idx, row) in incidence.public_rows().iter().enumerate() {
        if row.claim_indices().is_empty() || row.point_idx() >= incidence.num_points() {
            return Err(AkitaError::InvalidInput(
                "root y-ring public-row incidence is invalid".to_string(),
            ));
        }
        for &claim_idx in row.claim_indices() {
            if claim_idx >= per_claim_y_rings.len()
                || incidence.claim_to_point()[claim_idx] != row.point_idx()
            {
                return Err(AkitaError::InvalidInput(
                    "root y-ring public-row term is inconsistent".to_string(),
                ));
            }
            y_rings[row_idx] += row_coefficient_rings[claim_idx] * per_claim_y_rings[claim_idx];
        }
    }
    Ok(y_rings)
}

/// Config-free flattened view of batched prover claims.
pub struct PreparedBatchedProveInputs<'a, F: FieldCore, E: FieldCore, P, const D: usize> {
    /// Distinct opening points in caller order.
    pub opening_points: Vec<&'a [E]>,
    /// Commitments flattened in point/group order.
    pub commitments_by_point: Vec<RingCommitment<F, D>>,
    /// Normalized incidence summary that owns canonical root claim routing.
    pub incidence_summary: ClaimIncidenceSummary,
    /// Polynomials flattened in claim order.
    pub flat_polys: Vec<&'a P>,
    /// Polynomials flattened in committed-group order.
    pub group_polys: Vec<&'a P>,
    /// Commitment hints flattened in claim-group order.
    pub flat_hints: Vec<AkitaCommitmentHint<F, D>>,
}

/// Assemble intermediate fold-level proofs followed by the terminal-level
/// proof.
///
/// The terminal proof already carries the cleartext `final_witness` (in
/// place of the prior `next_w_commitment`), so the recursive suffix is
/// `Intermediate(...) × N + Terminal(...)`.
pub fn build_final_proof_steps<F, L>(
    intermediate_levels: Vec<AkitaLevelProof<F, L>>,
    terminal: TerminalLevelProof<F, L>,
) -> Vec<AkitaProofStep<F, L>>
where
    F: FieldCore,
    L: ExtField<F>,
{
    let mut steps = intermediate_levels
        .into_iter()
        .map(AkitaProofStep::Intermediate)
        .collect::<Vec<_>>();
    steps.push(AkitaProofStep::Terminal(terminal));
    steps
}

struct ProverPreparedIncidence<'a, F: FieldCore, E: FieldCore, P, const D: usize> {
    points: Vec<&'a [E]>,
    point_payloads:
        Vec<CommittedPolynomials<'a, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>>,
    summary: ClaimIncidenceSummary,
}

fn prover_claims_to_incidence<'a, F, E, P, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    claims: ProverClaims<'a, E, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
) -> Result<ProverPreparedIncidence<'a, F, E, P, D>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
{
    let points: Vec<&'a [E]> = claims.iter().map(|(point, _)| *point).collect();
    let mut point_payloads: Vec<
        CommittedPolynomials<'a, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
    > = Vec::with_capacity(claims.len());
    let mut incidence_claims = Vec::new();

    for (point_idx, (_, payload)) in claims.into_iter().enumerate() {
        let poly_count = payload.poly_count();
        incidence_claims.extend((0..poly_count).map(|poly_idx| IncidenceClaim {
            point_idx,
            poly_idx,
            // Prover inputs do not contain claimed evaluations. The shared
            // incidence validator ignores this field, so zero is only a
            // structural placeholder.
            claimed_eval: E::zero(),
        }));
        point_payloads.push(payload);
    }

    let incidence = ClaimIncidence {
        points: points.clone(),
        claims: incidence_claims,
    };
    let summary = incidence.validate(ClaimIncidenceLimits {
        max_num_vars: expanded.seed().max_num_vars,
        max_num_points: expanded.seed().max_num_points,
        max_num_claims: expanded.seed().max_num_batched_polys,
    })?;

    Ok(ProverPreparedIncidence {
        points,
        point_payloads,
        summary,
    })
}

/// Validate and flatten batched prover claims into the root proof shape.
///
/// # Errors
///
/// Returns an error if the claim shape exceeds setup capacity, mixes
/// incompatible dimensions, or has malformed batch counts.
pub fn prepare_batched_prove_inputs<'a, F, E, P, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    claims: ProverClaims<'a, E, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
) -> Result<PreparedBatchedProveInputs<'a, F, E, P, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    validate_batched_inputs(expanded, &claims, |payload| payload.polynomials.len(), true)?;

    let prepared_incidence = prover_claims_to_incidence(expanded, claims)?;
    let opening_points = prepared_incidence.points;
    let commitments_by_point: Vec<RingCommitment<F, D>> = prepared_incidence
        .point_payloads
        .iter()
        .map(|payload| payload.commitment.clone())
        .collect();
    let incidence_summary = prepared_incidence.summary;
    let flat_polys: Vec<&P> = incidence_summary
        .claim_to_point()
        .iter()
        .zip(incidence_summary.claim_poly_indices().iter())
        .map(|(&point_idx, &poly_idx)| {
            &prepared_incidence.point_payloads[point_idx].polynomials[poly_idx]
        })
        .collect();
    let group_polys: Vec<&P> = prepared_incidence
        .point_payloads
        .iter()
        .flat_map(|payload| payload.polynomials.iter())
        .collect();
    let flat_hints: Vec<AkitaCommitmentHint<F, D>> = prepared_incidence
        .point_payloads
        .into_iter()
        .map(|payload| payload.hint)
        .collect();

    Ok(PreparedBatchedProveInputs {
        opening_points,
        commitments_by_point,
        incidence_summary,
        flat_polys,
        group_polys,
        flat_hints,
    })
}

/// Build a root-direct batched proof from flattened polynomial references and
/// their commitment-group hints.
///
/// # Errors
///
/// Returns an error if any polynomial cannot produce a direct root witness.
pub fn prove_root_direct<F, L, const D: usize, P>(
    polys: &[&P],
    hints: &[AkitaCommitmentHint<F, D>],
) -> Result<AkitaBatchedProof<F, L>, AkitaError>
where
    F: FieldCore,
    L: ExtField<F>,
    P: AkitaPolyOps<F, D>,
{
    let witnesses = polys
        .iter()
        .map(|poly| poly.direct_root_witness())
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(feature = "zk")]
    {
        let b_blinding_digits = hints
            .iter()
            .flat_map(|hint| hint.b_blinding_digits())
            .map(|digits| {
                let mut flat_digits = Vec::with_capacity(digits.flat_digits().len() * D);
                for plane in digits.flat_digits() {
                    flat_digits.extend_from_slice(plane);
                }
                flat_digits
            })
            .collect();
        Ok(AkitaBatchedProof {
            root: AkitaBatchedRootProof::new_direct(witnesses, b_blinding_digits),
            steps: Vec::new(),
        })
    }
    #[cfg(not(feature = "zk"))]
    {
        let _ = hints;
        Ok(AkitaBatchedProof {
            root: AkitaBatchedRootProof::new_direct(witnesses),
            steps: Vec::new(),
        })
    }
}

/// Build the recursive suffix from an intermediate-root handoff, then
/// assemble the final folded batched proof.
///
/// The caller owns suffix schedule/config policy inside `build_suffix`; this
/// helper owns the config-free handoff from root raw output into suffix
/// construction and final proof assembly.
///
/// # Errors
///
/// Returns an error if suffix construction fails.
pub fn build_folded_batched_proof_with_suffix<F, L, const D: usize, BuildSuffix>(
    raw: RootLevelRawOutput<F, L, D>,
    build_suffix: BuildSuffix,
) -> Result<(AkitaBatchedProof<F, L>, usize), AkitaError>
where
    F: FieldCore,
    L: ExtField<F>,
    BuildSuffix:
        FnOnce(RecursiveProverState<F, L>) -> Result<RecursiveSuffixOutcome<F, L>, AkitaError>,
{
    let RootLevelRawOutput {
        y_rings,
        extension_opening_reduction,
        v,
        stage1,
        stage2_sumcheck,
        w_commitment_proof,
        w_eval,
        next_state,
    } = raw;
    let suffix = build_suffix(next_state)?;
    let RecursiveSuffixOutcome {
        intermediate_levels,
        terminal,
        num_levels,
    } = suffix;
    let root = AkitaBatchedRootProof::new_two_stage_with_extension_opening_reduction::<D>(
        y_rings,
        extension_opening_reduction,
        v,
        stage1,
        stage2_sumcheck,
        w_commitment_proof,
        w_eval,
    );
    let steps = build_final_proof_steps::<F, L>(intermediate_levels, terminal);
    Ok((AkitaBatchedProof { root, steps }, num_levels))
}

/// Assemble the 1-fold batched proof when the root level is itself the
/// terminal fold (no recursive suffix follows).
pub fn build_terminal_root_batched_proof<F, L>(
    terminal: TerminalLevelProof<F, L>,
) -> AkitaBatchedProof<F, L>
where
    F: FieldCore,
    L: ExtField<F>,
{
    AkitaBatchedProof {
        root: AkitaBatchedRootProof::new_terminal(terminal),
        steps: Vec::new(),
    }
}

/// Prove one recursive fold level after the caller has built its quadratic
/// equation and selected the commitment policy for the next `w`.
///
/// The caller owns config/schedule decisions through `commit_w_for_next`; this
/// function owns the config-free prover mechanics: build `w`, commit it using
/// that closure, finish ring switching, run stage-1/stage-2 sumchecks, and
/// produce the next recursive state.
///
/// # Errors
///
/// Returns an error if ring switching, recursive commitment, or either
/// sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_fold_level_from_quadratic<F, L, T, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    commitment_u: &[CyclotomicRing<F, D>],
    level: usize,
    lp: &LevelParams,
    next_log_basis: u32,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: ExtField<F> + RingSubfieldEncoding<F> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
    let logical_w = ring_switch_build_w::<F, { D }>(&mut quad_eq, expanded, ntt_shared, lp)?;
    let next_commitment = {
        let _span = tracing::info_span!("commit_w_level", level).entered();
        commit_w_for_next(&logical_w)?
    };
    let NextWitnessCommitment {
        witness: packed_witness,
        commitment: committed_commitment,
        hint: committed_hint,
    } = next_commitment;
    let w_commitment_proof = committed_commitment.clone();
    let rs = ring_switch_finalize::<F, L, T, { D }>(
        &quad_eq,
        expanded,
        transcript,
        &logical_w,
        &w_commitment_proof,
        lp,
    )?;

    let relation_claim = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        &quad_eq.v,
        commitment_u,
        &y_rings,
    )?;
    let RingSwitchOutput {
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1: _,
        b,
        alpha: _,
    } = rs;
    let tau0_reordered = reorder_stage1_coords(&tau0, col_bits, ring_bits);
    let (stage1_proof, r_stage1, s_claim) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let stage1_prover = AkitaStage1Prover::new(
            &w_evals_compact,
            &tau0_reordered,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        )?;
        let (stage1_proof, r_stage1) = stage1_prover.prove(transcript)?;
        let s_claim = stage1_proof.s_claim;
        (stage1_proof, r_stage1, s_claim)
    };
    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &s_claim);
    let batching_coeff: L = sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
    let (stage2_sumcheck, sumcheck_challenges, _stage2_final_claim, w_eval) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        let stage2_prover_result = AkitaStage2Prover::new(
            batching_coeff,
            w_evals_compact,
            &r_stage1,
            s_claim,
            b,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        );
        let mut stage2_prover = stage2_prover_result?;
        let (stage2_sumcheck, sumcheck_challenges, stage2_final_claim) =
            prove_sumcheck::<F, _, L, _, _>(&mut stage2_prover, transcript, |tr| {
                sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;

        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level).entered();
            stage2_prover.final_w_eval()
        };
        (
            stage2_sumcheck,
            sumcheck_challenges,
            stage2_final_claim,
            w_eval,
        )
    };

    let (level_proof, sumcheck_challenges) = (
        AkitaLevelProof::new_two_stage_many_with_extension_opening_reduction::<D>(
            y_rings,
            extension_opening_reduction,
            quad_eq.v,
            stage1_proof,
            stage2_sumcheck,
            w_commitment_proof,
            w_eval,
        ),
        sumcheck_challenges,
    );

    let (committed_witness, logical_w) = match packed_witness {
        Some(packed_witness) => (packed_witness, Some(logical_w)),
        None => (logical_w, None),
    };

    Ok(ProveLevelOutput {
        level_proof,
        next_state: RecursiveProverState {
            w: committed_witness,
            logical_w,
            commitment: committed_commitment,
            hint: committed_hint,
            log_basis: next_log_basis,
            sumcheck_challenges,
            opening: w_eval,
        },
    })
}

/// Prove the terminal recursive fold level after the caller has built its
/// quadratic equation.
///
/// At the terminal level the next witness is shipped in cleartext as
/// [`PackedDigits`], so this function:
///
/// * builds `logical_w` via ring switching,
/// * packs it into the scheduled terminal [`DirectWitnessProof`] shape,
/// * absorbs the final-witness remainder before sampling any ring-switch
///   challenges (the terminal `w_hat` segment was already absorbed before
///   sparse-challenge sampling),
/// * skips the stage-1 sumcheck entirely (packed-digit range is structurally
///   enforced by the packing), and
/// * runs stage-2 in relation-only mode with `batching_coeff = 0`,
///   `s_claim = 0`, and dummy `r_stage1` zeros — these zero the virtual
///   sumcheck contribution leaving only the relation oracle.
///
/// # Errors
///
/// Returns an error if ring switching or the stage-2 sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_fold_level_from_quadratic<F, L, T, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    commitment_u: &[CyclotomicRing<F, D>],
    _level: usize,
    lp: &LevelParams,
    final_witness_shape: &DirectWitnessShape,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
) -> Result<TerminalLevelProof<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: ExtField<F> + RingSubfieldEncoding<F> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let terminal_layout = terminal_segment_layout_for_quad(&quad_eq, lp)?;
    let logical_w = ring_switch_build_w::<F, { D }>(&mut quad_eq, expanded, ntt_shared, lp)?;
    let final_witness = pack_scheduled_terminal_witness(&logical_w, final_witness_shape)?;
    let rs = ring_switch_finalize_terminal::<F, L, T, { D }>(
        &quad_eq,
        expanded,
        transcript,
        &logical_w,
        &final_witness,
        terminal_layout,
        lp,
    )?;

    // Terminal layout drops the D-block: the relation claim no longer sums
    // any `v` rows, so pass an empty slice for the v parameter.
    let relation_claim = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        &[],
        commitment_u,
        &y_rings,
    )?;
    let relation_only_inputs = rs.relation_only_stage2_inputs();
    let TerminalRingSwitchOutput {
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau1: _,
        b,
        alpha: _,
    } = rs;

    // Relation-only stage-2: batching_coeff = 0 zeros the virtual-claim
    // contribution to every round polynomial regardless of `r_stage1`, so
    // dummy zeros for `r_stage1` and `s_claim` are safe.
    let stage2_sumcheck = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck_terminal").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            relation_only_inputs.batching_coeff,
            w_evals_compact,
            &relation_only_inputs.r_stage1,
            relation_only_inputs.s_claim,
            b,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        )?;
        let (stage2_sumcheck, _sumcheck_challenges, _stage2_final_claim) =
            prove_sumcheck::<F, _, L, _, _>(&mut stage2_prover, transcript, |tr| {
                sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;
        stage2_sumcheck
    };

    Ok(
        TerminalLevelProof::new_with_extension_opening_reduction::<D>(
            y_rings,
            extension_opening_reduction,
            stage2_sumcheck,
            final_witness,
        ),
    )
}

struct RecursiveExtensionOpeningReduction<L: FieldCore> {
    proof: ExtensionOpeningReductionProof<L>,
    rho: Vec<L>,
    final_claim: L,
    final_factor: L,
}

fn recursive_witness_base_evals<F>(logical_w: &RecursiveWitnessFlat) -> Vec<F>
where
    F: FieldCore + FromPrimitiveInt,
{
    logical_w
        .as_i8_digits()
        .iter()
        .copied()
        .map(F::from_i8)
        .collect()
}

fn prove_recursive_extension_opening_reduction<F, L, T>(
    logical_w: &RecursiveWitnessFlat,
    opening_point: &[L],
    expected_opening: L,
    transcript: &mut T,
) -> Result<RecursiveExtensionOpeningReduction<L>, AkitaError>
where
    F: FieldCore + CanonicalField,
    L: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    let num_vars = opening_point.len();
    let padded_len = 1usize.checked_shl(num_vars as u32).ok_or_else(|| {
        AkitaError::InvalidInput("recursive opening point is too large".to_string())
    })?;
    let (split_bits, _width) = tensor_opening_split::<F, L>()?;
    if split_bits > num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: split_bits,
            actual: opening_point.len(),
        });
    }
    if logical_w.len() > padded_len {
        return Err(AkitaError::InvalidSize {
            expected: padded_len,
            actual: logical_w.len(),
        });
    }
    let mut base_evals = recursive_witness_base_evals::<F>(logical_w);
    base_evals.resize(padded_len, F::zero());
    let tensor = tensor_partials_from_base_evals::<F, L>(num_vars, &base_evals, opening_point)?;
    check_tensor_extension_opening_claim::<F, L>(
        opening_point,
        expected_opening,
        &tensor.column_partials,
    )?;
    for partial in &tensor.column_partials {
        append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
    }

    let eta = (0..split_bits)
        .map(|_| sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
        .collect::<Vec<_>>();
    let input_claim = tensor_reduction_claim_from_rows::<F, L>(&tensor.row_partials, &eta)?;
    let packed_witness = tensor_packed_witness_evals::<F, L>(num_vars, &base_evals)?;
    let tail_point = &opening_point[split_bits..];
    let factor_evals = tensor_equality_factor_evals::<F, L>(tail_point, &eta)?;
    let mut prover = ExtensionOpeningReductionProver::new(packed_witness, factor_evals)?;
    if prover.input_claim() != input_claim {
        return Err(AkitaError::InvalidInput(
            "extension-opening reduction input claim mismatch".to_string(),
        ));
    }
    let (sumcheck, result) =
        prove_extension_opening_reduction::<F, _, L, _>(&mut prover, transcript, |tr| {
            sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?;
    let (final_witness, final_factor_from_table) =
        prover.final_witness_and_factor_evals().ok_or_else(|| {
            AkitaError::InvalidInput(
                "extension-opening reduction has not reached a final point".to_string(),
            )
        })?;
    let final_factor =
        tensor_equality_factor_eval_at_point::<F, L>(tail_point, &eta, &result.challenges)?;
    if final_factor != final_factor_from_table {
        return Err(AkitaError::InvalidInput(
            "extension-opening reduction transparent factor mismatch".to_string(),
        ));
    }
    check_extension_opening_reduction_output(result.final_claim, final_witness, final_factor)?;
    Ok(RecursiveExtensionOpeningReduction {
        proof: ExtensionOpeningReductionProof {
            partials: tensor.column_partials,
            sumcheck,
        },
        rho: result.challenges,
        final_claim: result.final_claim,
        final_factor,
    })
}

/// Prove one recursive fold level using already-selected current and next
/// level parameters.
///
/// The caller owns schedule/config selection and passes the next-level
/// commitment policy as a closure. This function owns recursive opening-point
/// reduction, witness folding, public recursive transcript absorbs, recursive
/// quadratic-equation construction, and the folded-level prover mechanics.
///
/// # Errors
///
/// Returns an error if the recursive opening point has the wrong dimension,
/// witness folding or quadratic-equation construction fails, or the folded
/// prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_recursive_fold_with_params<F, L, T, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    witness: &RecursiveWitnessView<'_, F, D>,
    logical_w: &RecursiveWitnessFlat,
    opening_point: &[L],
    expected_opening: L,
    hint: AkitaCommitmentHint<F, D>,
    commitment: &FlatRingVec<F>,
    level: usize,
    level_params: &LevelParams,
    next_log_basis: u32,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level,
            "prove_recursive_fold_with_params"
        );
    }

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    let commitment_u = commitment.as_ring_slice::<D>()?;
    commitment.append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;

    let reduction = if <L as ExtField<F>>::EXT_DEGREE == 1 {
        None
    } else {
        Some(prove_recursive_extension_opening_reduction::<F, L, T>(
            logical_w,
            opening_point,
            expected_opening,
            transcript,
        )?)
    };
    let protocol_point = match &reduction {
        Some(reduction) => ring_subfield_packed_extension_opening_point::<F, L, D>(
            reduction.rho.len(),
            &reduction.rho,
        )?,
        None => opening_point.to_vec(),
    };
    let prepared_points = {
        let _span = tracing::info_span!("ring_opening_point", level).entered();
        vec![prepare_recursive_opening_point_ext::<F, L, D>(
            &protocol_point,
            BasisMode::Lagrange,
            level_params,
            alpha,
            BlockOrder::ColumnMajor,
        )?]
    };

    let (y_rings, w_folded_by_claim) = {
        let _span = tracing::info_span!(
            "evaluate_and_fold",
            level,
            num_ring_elems = witness.num_ring_elems(),
            num_points = prepared_points.len()
        )
        .entered();
        let mut y_rings = Vec::with_capacity(prepared_points.len());
        let mut folded = Vec::with_capacity(prepared_points.len());
        for prepared_point in &prepared_points {
            let (y_ring, w_folded) = evaluate_recursive_witness_at_multiplier_point(
                witness,
                &prepared_point.ring_multiplier_point,
                level_params.block_len,
                level_params.num_blocks,
            )?;
            y_rings.push(y_ring);
            folded.push(w_folded);
        }
        (y_rings, folded)
    };

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
    match &reduction {
        Some(reduction) => {
            check_extension_opening_reduction_output(
                reduction.final_claim,
                internal_claims[0],
                reduction.final_factor,
            )?;
        }
        None => {
            if internal_claims[0] != expected_opening {
                return Err(AkitaError::InvalidInput(
                    "recursive opening does not match carried claim".to_string(),
                ));
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
    let quad_eq = Box::new(
        QuadraticEquation::<F, { D }>::new_recursive_multipoint_prover(
            ntt_shared,
            ring_opening_points,
            ring_multiplier_points,
            witness,
            w_folded_by_claim,
            level_params.clone(),
            hint,
            transcript,
            commitment_u,
            &y_rings,
            expanded.seed().max_stride,
            MRowLayout::Intermediate,
        )?,
    );

    let extension_opening_reduction = reduction.map(|reduction| reduction.proof);
    prove_fold_level_from_quadratic::<F, L, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        commitment_u,
        level,
        level_params,
        next_log_basis,
        quad_eq,
        extension_opening_reduction,
        y_rings,
        commit_w_for_next,
    )
}

/// Mirror of [`prove_recursive_fold_with_params`] producing a
/// [`TerminalLevelProof`] instead of an intermediate
/// [`AkitaLevelProof`] + next-witness commitment pair.
///
/// All recursive-opening, witness folding, and quadratic-equation setup is
/// identical to the intermediate path. The two differ only inside the inner
/// fold proof (see [`prove_terminal_fold_level_from_quadratic`]).
///
/// # Errors
///
/// Returns an error if recursive-opening setup, witness folding, or the
/// inner terminal fold-level prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_recursive_fold_with_params<F, L, T, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    witness: &RecursiveWitnessView<'_, F, D>,
    logical_w: &RecursiveWitnessFlat,
    opening_point: &[L],
    expected_opening: L,
    hint: AkitaCommitmentHint<F, D>,
    commitment: &FlatRingVec<F>,
    level: usize,
    level_params: &LevelParams,
    final_witness_shape: &DirectWitnessShape,
) -> Result<TerminalLevelProof<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
{
    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level,
            "prove_terminal_recursive_fold_with_params"
        );
    }

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    let commitment_u = commitment.as_ring_slice::<D>()?;
    commitment.append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;

    let reduction = if <L as ExtField<F>>::EXT_DEGREE == 1 {
        None
    } else {
        Some(prove_recursive_extension_opening_reduction::<F, L, T>(
            logical_w,
            opening_point,
            expected_opening,
            transcript,
        )?)
    };
    let protocol_point = match &reduction {
        Some(reduction) => ring_subfield_packed_extension_opening_point::<F, L, D>(
            reduction.rho.len(),
            &reduction.rho,
        )?,
        None => opening_point.to_vec(),
    };
    let prepared_points = {
        let _span = tracing::info_span!("ring_opening_point", level).entered();
        vec![prepare_recursive_opening_point_ext::<F, L, D>(
            &protocol_point,
            BasisMode::Lagrange,
            level_params,
            alpha,
            BlockOrder::ColumnMajor,
        )?]
    };

    let (y_rings, w_folded_by_claim) = {
        let _span = tracing::info_span!(
            "evaluate_and_fold",
            level,
            num_ring_elems = witness.num_ring_elems(),
            num_points = prepared_points.len()
        )
        .entered();
        let mut y_rings = Vec::with_capacity(prepared_points.len());
        let mut folded = Vec::with_capacity(prepared_points.len());
        for prepared_point in &prepared_points {
            let (y_ring, w_folded) = evaluate_recursive_witness_at_multiplier_point(
                witness,
                &prepared_point.ring_multiplier_point,
                level_params.block_len,
                level_params.num_blocks,
            )?;
            y_rings.push(y_ring);
            folded.push(w_folded);
        }
        (y_rings, folded)
    };

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
    match &reduction {
        Some(reduction) => {
            check_extension_opening_reduction_output(
                reduction.final_claim,
                internal_claims[0],
                reduction.final_factor,
            )?;
        }
        None => {
            if internal_claims[0] != expected_opening {
                return Err(AkitaError::InvalidInput(
                    "recursive opening does not match carried claim".to_string(),
                ));
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
    let quad_eq = Box::new(
        QuadraticEquation::<F, { D }>::new_recursive_multipoint_prover(
            ntt_shared,
            ring_opening_points,
            ring_multiplier_points,
            witness,
            w_folded_by_claim,
            level_params.clone(),
            hint,
            transcript,
            commitment_u,
            &y_rings,
            expanded.seed().max_stride,
            MRowLayout::Terminal,
        )?,
    );

    let extension_opening_reduction = reduction.map(|reduction| reduction.proof);
    prove_terminal_fold_level_from_quadratic::<F, L, T, D>(
        expanded,
        ntt_shared,
        transcript,
        commitment_u,
        level,
        level_params,
        final_witness_shape,
        quad_eq,
        extension_opening_reduction,
        y_rings,
    )
}

/// Prove one recursive fold level from D-erased recursive state using
/// caller-supplied config policy.
///
/// The prover crate owns the state unpacking, typed recursive witness view,
/// typed hint conversion, opening-point handoff, and fold proof mechanics.
/// The caller supplies only the current-witness layout policy and the
/// next-level recursive commitment policy.
///
/// # Errors
///
/// Returns an error if the current witness cannot be viewed at `D`, the hint
/// cannot be typed at `D`, layout selection fails, or recursive proving fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_recursive_level<F, Cfg, T, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    commit_ntt_cache: &mut crate::MultiDNttCaches,
    transcript: &mut T,
    current_state: &RecursiveProverState<F, Cfg::ChallengeField>,
    level: usize,
    level_params: &LevelParams,
    next_params: &LevelParams,
) -> Result<ProveLevelOutput<F, Cfg::ChallengeField>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg: akita_config::CommitmentConfig<Field = F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
{
    let _setup_span = tracing::info_span!("inter_level_setup", level).entered();

    let current_w = &current_state.w;
    let w_lp = akita_types::recursive_level_layout_from_params(
        level_params,
        current_w.len(),
        Cfg::decomposition(),
    )?;
    let w_view = current_w.view::<F, D>()?;
    let typed_hint: AkitaCommitmentHint<F, D> = current_state.hint.to_typed::<D>()?;
    drop(_setup_span);

    prove_recursive_fold_with_params::<F, Cfg::ChallengeField, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        &w_view,
        current_state.logical_w(),
        &current_state.sumcheck_challenges,
        current_state.opening,
        typed_hint,
        &current_state.commitment,
        level,
        &w_lp,
        next_params.log_basis,
        |w| {
            use akita_config::CommitmentConfig as _;
            crate::commit_next_w::<F, Cfg, D>(
                next_params,
                ntt_shared,
                commit_ntt_cache,
                expanded,
                w,
                akita_config::WCommitmentConfig::<{ D }, Cfg>::decomposition(),
            )
        },
    )
}

/// Terminal-fold analogue of [`prove_recursive_level`].
///
/// The terminal fold ships `final_witness` in cleartext (packed digits)
/// instead of committing to a next witness.
///
/// # Errors
///
/// Returns an error if the recursive layout derivation fails or the
/// underlying terminal fold prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_recursive_level<F, Cfg, T, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    current_state: &RecursiveProverState<F, Cfg::ChallengeField>,
    level: usize,
    level_params: &LevelParams,
    final_witness_shape: &DirectWitnessShape,
) -> Result<TerminalLevelProof<F, Cfg::ChallengeField>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg: akita_config::CommitmentConfig<Field = F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
{
    let _setup_span = tracing::info_span!("inter_level_setup_terminal", level).entered();

    let current_w = &current_state.w;
    let w_lp = akita_types::recursive_level_layout_from_params(
        level_params,
        current_w.len(),
        Cfg::decomposition(),
    )?;
    let w_view = current_w.view::<F, D>()?;
    let typed_hint: AkitaCommitmentHint<F, D> = current_state.hint.to_typed::<D>()?;
    drop(_setup_span);

    prove_terminal_recursive_fold_with_params::<F, Cfg::ChallengeField, T, D>(
        expanded,
        ntt_shared,
        transcript,
        &w_view,
        current_state.logical_w(),
        &current_state.sumcheck_challenges,
        current_state.opening,
        typed_hint,
        &current_state.commitment,
        level,
        &w_lp,
        final_witness_shape,
    )
}

struct PreparedRootExtensionOpeningReduction<E: FieldCore, C: FieldCore> {
    openings: Vec<E>,
    partials: Vec<C>,
    row_partials_by_claim: Vec<Vec<C>>,
    padded_points: Vec<Vec<C>>,
    split_bits: usize,
}

struct RootExtensionOpeningReduction<C: FieldCore> {
    proof: ExtensionOpeningReductionProof<C>,
    rho: Vec<C>,
    final_claim: C,
    factors_by_point: Vec<C>,
}

fn lift_claim_point<E, C>(point: &[E], num_vars: usize) -> Result<Vec<C>, AkitaError>
where
    E: FieldCore,
    C: ExtField<E>,
{
    if point.len() > num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: num_vars,
            actual: point.len(),
        });
    }
    let mut lifted = point.iter().copied().map(C::lift_base).collect::<Vec<_>>();
    lifted.resize(num_vars, C::zero());
    Ok(lifted)
}

fn prepare_root_extension_opening_reduction<F, E, C, P, const D: usize>(
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
) -> Result<PreparedRootExtensionOpeningReduction<E, C>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F> + ExtField<E>,
    P: AkitaPolyOps<F, D>,
{
    let _span = tracing::info_span!(
        "prepare_root_extension_opening_reduction",
        num_claims = incidence_summary.num_claims(),
        num_points = incidence_summary.num_points(),
        num_vars = incidence_summary.num_vars()
    )
    .entered();
    if <C as ExtField<F>>::EXT_DEGREE != <E as ExtField<F>>::EXT_DEGREE {
        return Err(AkitaError::InvalidInput(
            "root extension-opening reduction currently requires claim and challenge fields to have the same base degree"
                .to_string(),
        ));
    }
    let num_vars = incidence_summary.num_vars();
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    if split_bits > num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: split_bits,
            actual: num_vars,
        });
    }
    if polys.len() != incidence_summary.num_claims()
        || claim_points.len() != incidence_summary.num_points()
    {
        return Err(AkitaError::InvalidInput(
            "root extension-opening reduction input lengths do not match".to_string(),
        ));
    }

    let padded_points_e = claim_points
        .iter()
        .map(|point| {
            if point.len() > num_vars {
                return Err(AkitaError::InvalidPointDimension {
                    expected: num_vars,
                    actual: point.len(),
                });
            }
            let mut padded = point.to_vec();
            padded.resize(num_vars, E::zero());
            Ok(padded)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let padded_points = claim_points
        .iter()
        .map(|point| lift_claim_point::<E, C>(point, num_vars))
        .collect::<Result<Vec<_>, _>>()?;

    let mut openings = Vec::with_capacity(incidence_summary.num_claims());
    let mut partials = Vec::with_capacity(root_extension_opening_partials(
        width,
        incidence_summary.num_claims(),
    ));
    let mut row_partials_by_claim = Vec::with_capacity(incidence_summary.num_claims());
    {
        let _span =
            tracing::info_span!("root_extension_prepare_partials", width, split_bits).entered();
        let mut column_partials_by_claim = (0..incidence_summary.num_claims())
            .map(|_| None)
            .collect::<Vec<_>>();
        for (point_idx, logical_point) in padded_points_e
            .iter()
            .enumerate()
            .take(incidence_summary.num_points())
        {
            let claim_indices = incidence_summary
                .claim_to_point()
                .iter()
                .enumerate()
                .filter_map(|(claim_idx, &claim_point_idx)| {
                    (claim_point_idx == point_idx).then_some(claim_idx)
                })
                .collect::<Vec<_>>();
            let point_polys = claim_indices
                .iter()
                .map(|&claim_idx| polys[claim_idx])
                .collect::<Vec<_>>();
            let point_partials = <P as AkitaPolyOps<F, D>>::tensor_extension_column_partials_batch::<
                E,
            >(&point_polys, logical_point)?;
            if point_partials.len() != claim_indices.len() {
                return Err(AkitaError::InvalidSize {
                    expected: claim_indices.len(),
                    actual: point_partials.len(),
                });
            }
            for (claim_idx, column_partials) in claim_indices.into_iter().zip(point_partials) {
                column_partials_by_claim[claim_idx] = Some(column_partials);
            }
        }
        for (claim_idx, column_partials) in column_partials_by_claim.into_iter().enumerate() {
            let point_idx = incidence_summary.claim_to_point()[claim_idx];
            let logical_point = &padded_points_e[point_idx];
            let column_partials = column_partials.ok_or_else(|| {
                AkitaError::InvalidInput(
                    "missing root extension-opening column partials for claim".to_string(),
                )
            })?;
            let opening =
                tensor_logical_claim_from_partials::<F, E>(logical_point, &column_partials)?;
            let row_partials = tensor_row_partials_from_columns::<F, E>(&column_partials)?
                .into_iter()
                .map(C::lift_base)
                .collect::<Vec<_>>();
            partials.extend(column_partials.into_iter().map(C::lift_base));
            openings.push(opening);
            row_partials_by_claim.push(row_partials);
        }
    }

    Ok(PreparedRootExtensionOpeningReduction {
        openings,
        partials,
        row_partials_by_claim,
        padded_points,
        split_bits,
    })
}

fn prove_prepared_root_extension_opening_reduction<F, E, C, T, P, const D: usize>(
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    _root_params: &LevelParams,
    _basis: BasisMode,
    row_coefficients: &[C],
    prepared: PreparedRootExtensionOpeningReduction<E, C>,
    transcript: &mut T,
) -> Result<RootExtensionOpeningReduction<C>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F> + ExtField<E> + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
{
    let _span = tracing::info_span!(
        "prove_prepared_root_extension_opening_reduction",
        num_claims = incidence_summary.num_claims(),
        num_points = incidence_summary.num_points()
    )
    .entered();
    let PreparedRootExtensionOpeningReduction {
        openings: _,
        partials,
        row_partials_by_claim,
        padded_points,
        split_bits,
    } = prepared;
    {
        let _span = tracing::debug_span!(
            "root_extension_absorb_partials",
            partials_len = partials.len()
        )
        .entered();
        for partial in &partials {
            append_ext_field::<F, C, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
        }
    }
    let eta = (0..split_bits)
        .map(|_| sample_ext_challenge::<F, C, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
        .collect::<Vec<_>>();
    let input_claim = {
        let _span = tracing::debug_span!("root_extension_input_claim").entered();
        row_partials_by_claim.iter().enumerate().try_fold(
            C::zero(),
            |acc, (claim_idx, row_partials)| {
                tensor_reduction_claim_from_rows::<F, C>(row_partials, &eta)
                    .map(|claim| acc + row_coefficients[claim_idx] * claim)
            },
        )?
    };

    let sparse_terms = {
        let _span = tracing::info_span!("root_extension_sparse_terms").entered();
        let mut terms = Vec::with_capacity(incidence_summary.num_points());
        let mut all_sparse = true;
        for (point_idx, padded_point) in padded_points
            .iter()
            .enumerate()
            .take(incidence_summary.num_points())
        {
            let tail_point = &padded_point[split_bits..];
            let mut point_polys = Vec::new();
            let mut point_coeffs = Vec::new();
            for (claim_idx, poly) in polys.iter().enumerate() {
                if incidence_summary.claim_to_point()[claim_idx] == point_idx {
                    point_polys.push(*poly);
                    point_coeffs.push(row_coefficients[claim_idx]);
                }
            }
            let witness_evals = {
                let _span = tracing::info_span!(
                    "root_extension_sparse_witnesses",
                    point_idx,
                    num_terms = point_polys.len()
                )
                .entered();
                <P as AkitaPolyOps<F, D>>::tensor_packed_extension_sparse_linear_combination::<C>(
                    &point_polys,
                    &point_coeffs,
                )?
            };
            let Some(witness_evals) = witness_evals else {
                all_sparse = false;
                break;
            };
            let lazy_rounds = tail_point.len().min(SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS);
            if lazy_rounds == 0 {
                let factor_evals = {
                    let _span = tracing::debug_span!(
                        "root_extension_factor_evals",
                        point_idx,
                        tail_vars = tail_point.len()
                    )
                    .entered();
                    tensor_equality_factor_evals::<F, C>(tail_point, &eta)?
                };
                terms.push(BatchedExtensionOpeningReductionTerm::new_sparse(
                    witness_evals,
                    factor_evals,
                    C::one(),
                )?);
            } else {
                let _span = tracing::debug_span!(
                    "root_extension_lazy_tensor_factor",
                    point_idx,
                    tail_vars = tail_point.len(),
                    lazy_rounds
                )
                .entered();
                terms.push(
                    BatchedExtensionOpeningReductionTerm::new_sparse_tensor_factor::<F>(
                        witness_evals,
                        tail_point.to_vec(),
                        eta.clone(),
                        C::one(),
                        lazy_rounds,
                    )?,
                );
            }
        }
        if all_sparse {
            Some(terms)
        } else {
            None
        }
    };
    let terms = if let Some(terms) = sparse_terms {
        terms
    } else {
        let mut terms = Vec::with_capacity(incidence_summary.num_claims());
        {
            let _span = tracing::info_span!("root_extension_dense_terms").entered();
            for (claim_idx, poly) in polys.iter().enumerate() {
                let point_idx = incidence_summary.claim_to_point()[claim_idx];
                let tail_point = &padded_points[point_idx][split_bits..];
                let factor_evals = tensor_equality_factor_evals::<F, C>(tail_point, &eta)?;
                let witness_evals = poly.tensor_packed_extension_evals::<C>()?;
                terms.push(BatchedExtensionOpeningReductionTerm::new(
                    witness_evals,
                    factor_evals,
                    row_coefficients[claim_idx],
                )?);
            }
        }
        terms
    };
    let mut prover = {
        let _span = tracing::info_span!("root_extension_reduction_prover_new").entered();
        BatchedExtensionOpeningReductionProver::new(terms, input_claim)?
    };
    let (sumcheck, rho, final_claim) =
        prove_sumcheck::<F, _, C, _, _>(&mut prover, transcript, |tr| {
            sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?;
    let final_terms = prover.final_terms().ok_or_else(|| {
        AkitaError::InvalidInput(
            "root extension-opening reduction has not reached a final point".to_string(),
        )
    })?;
    let expected_final = final_terms
        .into_iter()
        .fold(C::zero(), |acc, (coeff, witness, factor)| {
            acc + coeff * witness * factor
        });
    if final_claim != expected_final {
        return Err(AkitaError::InvalidInput(
            "root extension-opening reduction final oracle mismatch".to_string(),
        ));
    }

    let factors_by_point = {
        let _span = tracing::debug_span!("root_extension_final_factors").entered();
        padded_points
            .iter()
            .map(|point| {
                tensor_equality_factor_eval_at_point::<F, C>(&point[split_bits..], &eta, &rho)
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(RootExtensionOpeningReduction {
        proof: ExtensionOpeningReductionProof { partials, sumcheck },
        rho,
        final_claim,
        factors_by_point,
    })
}

type MultiplierWeightSlices<'a, F, const D: usize> =
    (&'a [CyclotomicRing<F, D>], &'a [CyclotomicRing<F, D>]);
type FoldedRings<F, const D: usize> = Vec<CyclotomicRing<F, D>>;
type RootClaimEvaluations<F, const D: usize> = (Vec<CyclotomicRing<F, D>>, Vec<FoldedRings<F, D>>);

fn evaluate_root_claims_at_prepared_points<F, P, const D: usize>(
    polys: &[&P],
    claim_to_point: &[usize],
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    block_len: usize,
) -> Result<RootClaimEvaluations<F, D>, AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    let mut per_claim_y_rings = Vec::with_capacity(polys.len());
    let mut w_folded_by_poly = Vec::with_capacity(polys.len());
    for (poly, &point_idx) in polys.iter().zip(claim_to_point.iter()) {
        let prepared_point = &prepared_points[point_idx];
        let (y_ring, w_folded) = evaluate_poly_at_multiplier_point(
            *poly,
            &prepared_point.ring_multiplier_point,
            block_len,
        )?;
        per_claim_y_rings.push(y_ring);
        w_folded_by_poly.push(w_folded);
    }
    Ok((per_claim_y_rings, w_folded_by_poly))
}

fn multiplier_ring_weights<F: FieldCore, const D: usize>(
    point: &RingMultiplierOpeningPoint<F, D>,
) -> Result<MultiplierWeightSlices<'_, F, D>, AkitaError> {
    let b = point.b_rings().ok_or_else(|| {
        AkitaError::InvalidInput("ring multiplier must carry ring b weights".to_string())
    })?;
    let a = point.a_rings().ok_or_else(|| {
        AkitaError::InvalidInput("ring multiplier must carry ring a weights".to_string())
    })?;
    Ok((b, a))
}

fn evaluate_poly_at_multiplier_point<F, P, const D: usize>(
    poly: &P,
    point: &RingMultiplierOpeningPoint<F, D>,
    block_len: usize,
) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    if let Some(base_point) = point.as_base() {
        return Ok(poly.evaluate_and_fold(&base_point.b, &base_point.a, block_len));
    }

    let (b, a) = multiplier_ring_weights(point)?;
    Ok(poly.evaluate_and_fold_ring(b, a, block_len))
}

fn evaluate_recursive_witness_at_multiplier_point<F, const D: usize>(
    witness: &RecursiveWitnessView<'_, F, D>,
    point: &RingMultiplierOpeningPoint<F, D>,
    block_len: usize,
    num_blocks: usize,
) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if let Some(base_point) = point.as_base() {
        return Ok(witness.evaluate_and_fold(&base_point.b, &base_point.a, block_len, num_blocks));
    }

    let (b, a) = multiplier_ring_weights(point)?;
    Ok(witness.evaluate_and_fold_ring(b, a, block_len, num_blocks))
}

#[allow(clippy::too_many_arguments)]
fn finish_root_fold_with_prepared_openings<F, C, T, P, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    expected_w_len: usize,
    next_log_basis: u32,
    commit_w_for_next: CommitW,
    prepared_points: Vec<PreparedRootOpeningPoint<F, D>>,
    w_folded_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    row_coefficients: Vec<C>,
    row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<C>>,
) -> Result<RootLevelRawOutput<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    C: ExtField<F> + RingSubfieldEncoding<F> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
    let ring_opening_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_opening_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ring_multiplier_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_multiplier_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let quad_eq = Box::new(QuadraticEquation::<F, { D }>::new_prover(
        ntt_shared,
        ring_opening_points,
        ring_multiplier_points,
        incidence_summary.claim_to_point().to_vec(),
        polys,
        w_folded_by_poly,
        incidence_summary,
        root_params.clone(),
        hints,
        transcript,
        commitments,
        &y_rings,
        row_coefficient_rings,
        expanded.seed().max_stride,
        MRowLayout::Intermediate,
    )?);

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    let mut raw = prove_root_fold_from_quadratic::<F, C, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        commitment_rows,
        root_params,
        expected_w_len,
        next_log_basis,
        quad_eq,
        y_rings,
        row_coefficients,
        commit_w_for_next,
    )?;
    raw.extension_opening_reduction = extension_opening_reduction;
    Ok(raw)
}

/// Prove the folded root level using already-selected root and next-level
/// parameters.
///
/// The caller owns schedule/config selection and passes the expected next
/// recursive witness length, next digit basis, and commitment policy for that
/// witness. This function owns root polynomial folding, public root transcript
/// setup, root quadratic-equation construction, and the folded-root prover
/// mechanics.
///
/// # Errors
///
/// Returns an error if root inputs are malformed, polynomial folding or
/// quadratic-equation construction fails, or the folded-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_root_fold_with_params<F, E, C, T, const D: usize, P, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    expected_w_len: usize,
    next_log_basis: u32,
    basis: BasisMode,
    commit_w_for_next: CommitW,
) -> Result<RootLevelRawOutput<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F> + ExtField<E> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
    let claim_to_point = incidence_summary.claim_to_point();
    let num_claims = incidence_summary.num_claims();

    if claim_points.is_empty()
        || claim_points.len() != incidence_summary.num_points()
        || claim_to_point.len() != num_claims
        || polys.len() != num_claims
        || commitments.len() != incidence_summary.num_points()
        || hints.len() != incidence_summary.num_points()
    {
        return Err(AkitaError::InvalidInput(
            "invalid root-level inputs".to_string(),
        ));
    }
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= claim_points.len())
    {
        return Err(AkitaError::InvalidInput(
            "root-level claim-to-point index out of range".to_string(),
        ));
    }

    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level = 0usize,
            num_claims,
            num_points = claim_points.len(),
            "prove_root_fold_with_params"
        );
    }

    append_claim_incidence_shape_to_transcript::<F, T>(incidence_summary, transcript)?;
    append_batched_commitments_to_transcript(commitments, transcript);
    append_claim_points_to_transcript::<F, E, T>(claim_points, transcript);

    let alpha_bits = root_params.ring_dimension.trailing_zeros() as usize;
    let needs_extension_reduction =
        root_tensor_projection_enabled::<F, E, C, D>(incidence_summary.num_vars());
    let extension_reduction_prepare = if !needs_extension_reduction {
        None
    } else {
        Some(prepare_root_extension_opening_reduction::<F, E, C, P, D>(
            polys,
            incidence_summary,
            claim_points,
        )?)
    };

    let openings: Vec<E>;
    let prepared_points: Vec<PreparedRootOpeningPoint<F, D>>;
    if let Some(prepared_reduction) = extension_reduction_prepare {
        openings = prepared_reduction.openings.clone();
        append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
        let row_coefficients =
            sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;
        let row_coefficient_rings = row_coefficient_rings::<F, C, D>(&row_coefficients)?;
        let reduction = prove_prepared_root_extension_opening_reduction::<F, E, C, T, P, D>(
            polys,
            incidence_summary,
            root_params,
            basis,
            &row_coefficients,
            prepared_reduction,
            transcript,
        )?;
        let protocol_point = ring_subfield_packed_extension_opening_point::<F, C, D>(
            reduction.rho.len(),
            &reduction.rho,
        )?;
        let prepared_protocol_point = prepare_root_opening_point_ext::<F, C, C, D>(
            &protocol_point,
            basis,
            root_params,
            alpha_bits,
        )?;
        prepared_points = vec![prepared_protocol_point; incidence_summary.num_points()];
        let transformed_polys = cfg_iter!(polys)
            .map(|poly| poly.tensor_packed_extension_root_poly::<C>())
            .collect::<Result<Vec<RootTensorProjectionPoly<F, D>>, _>>()?;
        let transformed_refs = transformed_polys.iter().collect::<Vec<_>>();

        let (per_claim_y_rings, w_folded_by_poly) = evaluate_root_claims_at_prepared_points(
            &transformed_refs,
            claim_to_point,
            &prepared_points,
            root_params.block_len,
        )?;
        let y_rings = combine_root_y_rings::<F, D>(
            &per_claim_y_rings,
            incidence_summary,
            &row_coefficient_rings,
        )?;
        for y_ring in &y_rings {
            transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
        }
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
                acc + opening * reduction.factors_by_point[row.point_idx()]
            });
        check_extension_opening_reduction_output(reduction.final_claim, final_opening, C::one())?;
        let extension_opening_reduction = Some(reduction.proof);

        return finish_root_fold_with_prepared_openings::<
            F,
            C,
            T,
            RootTensorProjectionPoly<F, D>,
            D,
            _,
        >(
            expanded,
            ntt_shared,
            transcript,
            &transformed_refs,
            incidence_summary,
            commitments,
            hints,
            root_params,
            expected_w_len,
            next_log_basis,
            commit_w_for_next,
            prepared_points,
            w_folded_by_poly,
            y_rings,
            row_coefficients,
            row_coefficient_rings,
            extension_opening_reduction,
        );
    }

    prepared_points = claim_points
        .iter()
        .map(|opening_point| {
            prepare_root_opening_point_ext::<F, E, C, D>(
                opening_point,
                basis,
                root_params,
                alpha_bits,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (per_claim_y_rings, w_folded_by_poly) = evaluate_root_claims_at_prepared_points(
        polys,
        claim_to_point,
        &prepared_points,
        root_params.block_len,
    )?;

    let target_num_vars = root_params
        .m_vars
        .checked_add(root_params.r_vars)
        .and_then(|n| n.checked_add(alpha_bits))
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    let inner_claim_points = claim_points
        .iter()
        .map(|point| {
            if point.len() > target_num_vars {
                return Err(AkitaError::InvalidPointDimension {
                    expected: target_num_vars,
                    actual: point.len(),
                });
            }
            Ok(point[..point.len().min(alpha_bits)].to_vec())
        })
        .collect::<Result<Vec<_>, _>>()?;

    openings = per_claim_y_rings
        .iter()
        .zip(claim_to_point.iter())
        .map(|(y_ring, &point_idx)| {
            root_claim_opening_from_y_ring::<F, E, D>(
                y_ring,
                &prepared_points[point_idx],
                &inner_claim_points[point_idx],
                basis,
            )
        })
        .collect::<Result<_, _>>()?;
    append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
    let row_coefficients =
        sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;
    let row_coefficient_rings = row_coefficient_rings::<F, C, D>(&row_coefficients)?;

    let y_rings = combine_root_y_rings::<F, D>(
        &per_claim_y_rings,
        incidence_summary,
        &row_coefficient_rings,
    )?;
    for y_ring in &y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    let ring_opening_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_opening_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ring_multiplier_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_multiplier_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let quad_eq = Box::new(QuadraticEquation::<F, { D }>::new_prover(
        ntt_shared,
        ring_opening_points,
        ring_multiplier_points,
        incidence_summary.claim_to_point().to_vec(),
        polys,
        w_folded_by_poly,
        incidence_summary,
        root_params.clone(),
        hints,
        transcript,
        commitments,
        &y_rings,
        row_coefficient_rings,
        expanded.seed().max_stride,
        MRowLayout::Intermediate,
    )?);

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    prove_root_fold_from_quadratic::<F, C, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        commitment_rows,
        root_params,
        expected_w_len,
        next_log_basis,
        quad_eq,
        y_rings,
        row_coefficients,
        commit_w_for_next,
    )
}

/// Terminal-root analogue of [`prove_root_fold_with_params`] used when the
/// schedule has exactly one fold level (the root is itself the terminal).
///
/// Mirrors the intermediate-root path through claim-incidence absorbs,
/// optional extension-opening reduction, and quadratic-equation setup, then
/// emits a [`TerminalLevelProof`] via
/// [`prove_terminal_root_fold_from_quadratic`] instead of a
/// [`RootLevelRawOutput`].
///
/// # Errors
///
/// Returns an error if claim-incidence/transcript setup fails, the
/// extension-opening reduction proof construction fails, or the inner
/// terminal-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_root_fold_with_params<F, E, C, T, const D: usize, P>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    expected_w_len: usize,
    final_witness_shape: &DirectWitnessShape,
    basis: BasisMode,
) -> Result<TerminalLevelProof<F, C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F> + ExtField<E> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    match final_witness_shape {
        DirectWitnessShape::PackedDigits((scheduled_len, _))
            if *scheduled_len == expected_w_len => {}
        DirectWitnessShape::PackedDigits((scheduled_len, _)) => {
            return Err(AkitaError::InvalidSetup(format!(
                "terminal root direct shape length does not match root next-w length: expected={expected_w_len}, actual={scheduled_len}"
            )));
        }
        DirectWitnessShape::FieldElements(_) => {
            return Err(AkitaError::InvalidSetup(
                "terminal root requires a packed-digit direct step".to_string(),
            ));
        }
    }

    let claim_to_point = incidence_summary.claim_to_point();
    let num_claims = incidence_summary.num_claims();

    if claim_points.is_empty()
        || claim_points.len() != incidence_summary.num_points()
        || claim_to_point.len() != num_claims
        || polys.len() != num_claims
        || commitments.len() != incidence_summary.num_points()
        || hints.len() != incidence_summary.num_points()
    {
        return Err(AkitaError::InvalidInput(
            "invalid root-level inputs".to_string(),
        ));
    }
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= claim_points.len())
    {
        return Err(AkitaError::InvalidInput(
            "root-level claim-to-point index out of range".to_string(),
        ));
    }

    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level = 0usize,
            num_claims,
            num_points = claim_points.len(),
            "prove_terminal_root_fold_with_params"
        );
    }

    append_claim_incidence_shape_to_transcript::<F, T>(incidence_summary, transcript)?;
    append_batched_commitments_to_transcript(commitments, transcript);
    append_claim_points_to_transcript::<F, E, T>(claim_points, transcript);

    let alpha_bits = root_params.ring_dimension.trailing_zeros() as usize;
    let needs_extension_reduction =
        root_tensor_projection_enabled::<F, E, C, D>(incidence_summary.num_vars());
    let extension_reduction_prepare = if !needs_extension_reduction {
        None
    } else {
        Some(prepare_root_extension_opening_reduction::<F, E, C, P, D>(
            polys,
            incidence_summary,
            claim_points,
        )?)
    };

    let openings: Vec<E>;
    let prepared_points: Vec<PreparedRootOpeningPoint<F, D>>;
    if let Some(prepared_reduction) = extension_reduction_prepare {
        openings = prepared_reduction.openings.clone();
        append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
        let row_coefficients =
            sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;
        let row_coefficient_rings = row_coefficient_rings::<F, C, D>(&row_coefficients)?;
        let reduction = prove_prepared_root_extension_opening_reduction::<F, E, C, T, P, D>(
            polys,
            incidence_summary,
            root_params,
            basis,
            &row_coefficients,
            prepared_reduction,
            transcript,
        )?;
        let protocol_point = ring_subfield_packed_extension_opening_point::<F, C, D>(
            reduction.rho.len(),
            &reduction.rho,
        )?;
        let prepared_protocol_point = prepare_root_opening_point_ext::<F, C, C, D>(
            &protocol_point,
            basis,
            root_params,
            alpha_bits,
        )?;
        prepared_points = vec![prepared_protocol_point; incidence_summary.num_points()];
        let transformed_polys = polys
            .iter()
            .map(|poly| poly.tensor_packed_extension_root_poly::<C>())
            .collect::<Result<Vec<RootTensorProjectionPoly<F, D>>, _>>()?;
        let transformed_refs = transformed_polys.iter().collect::<Vec<_>>();

        let (per_claim_y_rings, w_folded_by_poly) = evaluate_root_claims_at_prepared_points(
            &transformed_refs,
            claim_to_point,
            &prepared_points,
            root_params.block_len,
        )?;
        let y_rings = combine_root_y_rings::<F, D>(
            &per_claim_y_rings,
            incidence_summary,
            &row_coefficient_rings,
        )?;
        for y_ring in &y_rings {
            transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
        }
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
                acc + opening * reduction.factors_by_point[row.point_idx()]
            });
        check_extension_opening_reduction_output(reduction.final_claim, final_opening, C::one())?;
        let extension_opening_reduction = Some(reduction.proof);

        return finish_terminal_root_fold_with_prepared_openings::<
            F,
            C,
            T,
            RootTensorProjectionPoly<F, D>,
            D,
        >(
            expanded,
            ntt_shared,
            transcript,
            &transformed_refs,
            incidence_summary,
            commitments,
            hints,
            root_params,
            expected_w_len,
            final_witness_shape,
            prepared_points,
            w_folded_by_poly,
            y_rings,
            row_coefficients,
            row_coefficient_rings,
            extension_opening_reduction,
        );
    }

    prepared_points = claim_points
        .iter()
        .map(|opening_point| {
            prepare_root_opening_point_ext::<F, E, C, D>(
                opening_point,
                basis,
                root_params,
                alpha_bits,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (per_claim_y_rings, w_folded_by_poly) = evaluate_root_claims_at_prepared_points(
        polys,
        claim_to_point,
        &prepared_points,
        root_params.block_len,
    )?;

    let target_num_vars = root_params
        .m_vars
        .checked_add(root_params.r_vars)
        .and_then(|n| n.checked_add(alpha_bits))
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    let inner_claim_points = claim_points
        .iter()
        .map(|point| {
            if point.len() > target_num_vars {
                return Err(AkitaError::InvalidPointDimension {
                    expected: target_num_vars,
                    actual: point.len(),
                });
            }
            Ok(point[..point.len().min(alpha_bits)].to_vec())
        })
        .collect::<Result<Vec<_>, _>>()?;

    openings = per_claim_y_rings
        .iter()
        .zip(claim_to_point.iter())
        .map(|(y_ring, &point_idx)| {
            root_claim_opening_from_y_ring::<F, E, D>(
                y_ring,
                &prepared_points[point_idx],
                &inner_claim_points[point_idx],
                basis,
            )
        })
        .collect::<Result<_, _>>()?;
    append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
    let row_coefficients =
        sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)?;
    let row_coefficient_rings = row_coefficient_rings::<F, C, D>(&row_coefficients)?;

    let y_rings = combine_root_y_rings::<F, D>(
        &per_claim_y_rings,
        incidence_summary,
        &row_coefficient_rings,
    )?;
    for y_ring in &y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    let ring_opening_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_opening_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ring_multiplier_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_multiplier_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let quad_eq = Box::new(QuadraticEquation::<F, { D }>::new_prover(
        ntt_shared,
        ring_opening_points,
        ring_multiplier_points,
        incidence_summary.claim_to_point().to_vec(),
        polys,
        w_folded_by_poly,
        incidence_summary,
        root_params.clone(),
        hints,
        transcript,
        commitments,
        &y_rings,
        row_coefficient_rings,
        expanded.seed().max_stride,
        MRowLayout::Terminal,
    )?);

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    prove_terminal_root_fold_from_quadratic::<F, C, T, D>(
        expanded,
        ntt_shared,
        transcript,
        commitment_rows,
        root_params,
        expected_w_len,
        final_witness_shape,
        quad_eq,
        y_rings,
        row_coefficients,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_terminal_root_fold_with_prepared_openings<F, C, T, P, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    expected_w_len: usize,
    final_witness_shape: &DirectWitnessShape,
    prepared_points: Vec<PreparedRootOpeningPoint<F, D>>,
    w_folded_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    row_coefficients: Vec<C>,
    row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<C>>,
) -> Result<TerminalLevelProof<F, C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    C: ExtField<F> + RingSubfieldEncoding<F> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
{
    let ring_opening_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_opening_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ring_multiplier_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_multiplier_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let quad_eq = Box::new(QuadraticEquation::<F, { D }>::new_prover(
        ntt_shared,
        ring_opening_points,
        ring_multiplier_points,
        incidence_summary.claim_to_point().to_vec(),
        polys,
        w_folded_by_poly,
        incidence_summary,
        root_params.clone(),
        hints,
        transcript,
        commitments,
        &y_rings,
        row_coefficient_rings,
        expanded.seed().max_stride,
        MRowLayout::Terminal,
    )?);

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    let mut terminal = prove_terminal_root_fold_from_quadratic::<F, C, T, D>(
        expanded,
        ntt_shared,
        transcript,
        commitment_rows,
        root_params,
        expected_w_len,
        final_witness_shape,
        quad_eq,
        y_rings,
        row_coefficients,
    )?;
    terminal.extension_opening_reduction = extension_opening_reduction;
    Ok(terminal)
}

/// Prove the folded root level after root orchestration has built its
/// quadratic equation and selected the next recursive commitment policy.
///
/// The root caller owns transcript setup for public openings and gamma
/// batching, schedule selection, and the commitment-row view used by the root
/// relation. It also passes the already-validated challenge sampler used for
/// the remaining base-field stage proofs. This function owns the config-free
/// prover mechanics from `w` construction through the stage proofs and next
/// recursive state.
///
/// # Errors
///
/// Returns an error if ring switching, recursive commitment, or either
/// sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_root_fold_from_quadratic<F, C, T, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    commitment_rows: &[CyclotomicRing<F, D>],
    lp: &akita_types::LevelParams,
    expected_w_len: usize,
    next_log_basis: u32,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    row_coefficients: Vec<C>,
    commit_w_for_next: CommitW,
) -> Result<RootLevelRawOutput<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    C: ExtField<F> + RingSubfieldEncoding<F> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
    let logical_w = ring_switch_build_w::<F, { D }>(&mut quad_eq, expanded, ntt_shared, lp)?;
    if logical_w.len() != expected_w_len {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled root next-w length did not match runtime witness: expected={expected_w_len}, actual={}",
            logical_w.len()
        )));
    }
    let next_commitment = {
        let _span = tracing::info_span!("commit_w_level", level = 0usize).entered();
        commit_w_for_next(&logical_w)?
    };
    let NextWitnessCommitment {
        witness: packed_witness,
        commitment: committed_commitment,
        hint: committed_hint,
    } = next_commitment;
    let w_commitment_proof = committed_commitment.clone();

    let rs = ring_switch_finalize_with_gamma::<F, C, T, { D }>(
        &quad_eq,
        expanded,
        transcript,
        &logical_w,
        &w_commitment_proof,
        lp,
        &row_coefficients,
    )?;

    let relation_claim = relation_claim_from_rows_extension::<F, C, D>(
        &rs.tau1,
        rs.alpha,
        &quad_eq.v,
        commitment_rows,
        &y_rings,
    )?;

    let RingSwitchOutput {
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1: _,
        b,
        alpha: _,
    } = rs;
    let tau0_reordered = reorder_stage1_coords(&tau0, col_bits, ring_bits);
    let (stage1_proof, r_stage1, s_claim) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let stage1_prover = AkitaStage1Prover::new(
            &w_evals_compact,
            &tau0_reordered,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        )?;
        let (stage1_proof, r_stage1) = stage1_prover.prove(transcript)?;
        let s_claim = stage1_proof.s_claim;
        (stage1_proof, r_stage1, s_claim)
    };

    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &s_claim);
    let batching_coeff: C = sample_ext_challenge::<F, C, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
    let (stage2_sumcheck, sumcheck_challenges, _stage2_final_claim, w_eval) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        let stage2_prover_result = AkitaStage2Prover::new(
            batching_coeff,
            w_evals_compact,
            &r_stage1,
            s_claim,
            b,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        );
        let mut stage2_prover = stage2_prover_result?;
        let (stage2_sumcheck, sumcheck_challenges, stage2_final_claim) =
            prove_sumcheck::<F, _, C, _, _>(&mut stage2_prover, transcript, |tr| {
                sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;

        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level = 0usize).entered();
            stage2_prover.final_w_eval()
        };
        (
            stage2_sumcheck,
            sumcheck_challenges,
            stage2_final_claim,
            w_eval,
        )
    };

    let (committed_witness, logical_w) = match packed_witness {
        Some(packed_witness) => (packed_witness, Some(logical_w)),
        None => (logical_w, None),
    };

    Ok(RootLevelRawOutput {
        y_rings,
        extension_opening_reduction: None,
        v: quad_eq.v,
        stage1: stage1_proof,
        stage2_sumcheck,
        w_commitment_proof,
        w_eval,
        next_state: RecursiveProverState {
            w: committed_witness,
            logical_w,
            commitment: committed_commitment,
            hint: committed_hint,
            log_basis: next_log_basis,
            sumcheck_challenges,
            opening: w_eval,
        },
    })
}

/// Terminal-root analogue of [`prove_root_fold_from_quadratic`] used when the
/// schedule has exactly one fold level (the root is itself the terminal).
///
/// Produces a [`TerminalLevelProof`] with cleartext `final_witness` instead
/// of a `RootLevelRawOutput`. There is no recursive suffix and no
/// `next_state` to thread.
///
/// # Errors
///
/// Returns an error if witness reconstruction does not match the schedule's
/// expected length, ring-switch replay fails, or the stage-2 sumcheck prover
/// fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_root_fold_from_quadratic<F, C, T, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    commitment_rows: &[CyclotomicRing<F, D>],
    lp: &akita_types::LevelParams,
    expected_w_len: usize,
    final_witness_shape: &DirectWitnessShape,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    row_coefficients: Vec<C>,
) -> Result<TerminalLevelProof<F, C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    C: ExtField<F> + RingSubfieldEncoding<F> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let terminal_layout = terminal_segment_layout_for_quad(&quad_eq, lp)?;
    let logical_w = ring_switch_build_w::<F, { D }>(&mut quad_eq, expanded, ntt_shared, lp)?;
    if logical_w.len() != expected_w_len {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled root next-w length did not match runtime witness: expected={expected_w_len}, actual={}",
            logical_w.len()
        )));
    }
    let final_witness = pack_scheduled_terminal_witness(&logical_w, final_witness_shape)?;
    let rs = ring_switch_finalize_terminal_with_gamma::<F, C, T, { D }>(
        &quad_eq,
        expanded,
        transcript,
        &logical_w,
        &final_witness,
        terminal_layout,
        lp,
        &row_coefficients,
    )?;

    // Terminal layout: the D-block is omitted, so the relation claim sums no
    // `v` rows. `quad_eq.v` is constructed as an empty vector under
    // `MRowLayout::Terminal`; pass `&[]` here for symmetry with the verifier.
    let relation_claim = relation_claim_from_rows_extension::<F, C, D>(
        &rs.tau1,
        rs.alpha,
        &[],
        commitment_rows,
        &y_rings,
    )?;

    let relation_only_inputs = rs.relation_only_stage2_inputs();
    let TerminalRingSwitchOutput {
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau1: _,
        b,
        alpha: _,
    } = rs;

    let stage2_sumcheck = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck_terminal_root").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            relation_only_inputs.batching_coeff,
            w_evals_compact,
            &relation_only_inputs.r_stage1,
            relation_only_inputs.s_claim,
            b,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        )?;
        let (stage2_sumcheck, _sumcheck_challenges, _stage2_final_claim) =
            prove_sumcheck::<F, _, C, _, _>(&mut stage2_prover, transcript, |tr| {
                sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;
        stage2_sumcheck
    };

    Ok(
        TerminalLevelProof::new_with_extension_opening_reduction::<D>(
            y_rings,
            None,
            stage2_sumcheck,
            final_witness,
        ),
    )
}

// ===========================================================================
// `<Cfg>`-generic top-level prove orchestration
// ===========================================================================

/// Dispatch a prove-level operation to the correct ring dimension.
///
/// Handles the fast-path (`level_d == D`) and the dynamic dispatch path.
/// `#[inline(never)]` isolates the monomorphized match arms in their own
/// stack frame.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_prove_level<F, T, const D: usize, Cfg>(
    level_d: usize,
    ntt_cache: &mut MultiDNttCaches,
    expanded: &AkitaExpandedSetup<F>,
    setup_ntt_shared: &NttSlotCache<D>,
    commit_ntt_cache: &mut MultiDNttCaches,
    current_state: &RecursiveProverState<F, Cfg::ChallengeField>,
    transcript: &mut T,
    level: usize,
    level_params: &LevelParams,
    next_params: LevelParams,
) -> Result<ProveLevelOutput<F, Cfg::ChallengeField>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + PseudoMersenneField,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + HasUnreducedOps
        + AkitaSerialize,
{
    if level_d == D {
        prove_recursive_level::<F, Cfg, T, D>(
            expanded,
            setup_ntt_shared,
            commit_ntt_cache,
            transcript,
            current_state,
            level,
            level_params,
            &next_params,
        )
    } else {
        crate::dispatch_with_ntt!(level_d, ntt_cache, expanded, |D_LEVEL, ntt_shared| {
            prove_recursive_level::<F, Cfg, T, { D_LEVEL }>(
                expanded,
                ntt_shared,
                commit_ntt_cache,
                transcript,
                current_state,
                level,
                level_params,
                &next_params,
            )
        })
    }
}

/// Dispatch a terminal prove-level operation to the correct ring dimension.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_prove_terminal_level<F, T, const D: usize, Cfg>(
    level_d: usize,
    ntt_cache: &mut MultiDNttCaches,
    expanded: &AkitaExpandedSetup<F>,
    setup_ntt_shared: &NttSlotCache<D>,
    current_state: &RecursiveProverState<F, Cfg::ChallengeField>,
    transcript: &mut T,
    level: usize,
    level_params: &LevelParams,
    final_witness_shape: &DirectWitnessShape,
) -> Result<TerminalLevelProof<F, Cfg::ChallengeField>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + PseudoMersenneField,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + HasUnreducedOps
        + AkitaSerialize,
{
    if level_d == D {
        prove_terminal_recursive_level::<F, Cfg, T, D>(
            expanded,
            setup_ntt_shared,
            transcript,
            current_state,
            level,
            level_params,
            final_witness_shape,
        )
    } else {
        crate::dispatch_with_ntt!(level_d, ntt_cache, expanded, |D_LEVEL, ntt_shared| {
            prove_terminal_recursive_level::<F, Cfg, T, { D_LEVEL }>(
                expanded,
                ntt_shared,
                transcript,
                current_state,
                level,
                level_params,
                final_witness_shape,
            )
        })
    }
}

/// Drive the recursive fold levels (after the root) and resolve the scheduled
/// packed terminal direct-witness shape.
#[allow(clippy::too_many_arguments)]
fn prove_recursive_suffix<F, T, const D: usize, Cfg>(
    setup: &AkitaProverSetup<F, D>,
    ntt_cache: &mut MultiDNttCaches,
    commit_ntt_cache: &mut MultiDNttCaches,
    num_vars: usize,
    transcript: &mut T,
    initial_state: RecursiveProverState<F, Cfg::ChallengeField>,
    schedule: &Schedule,
) -> Result<RecursiveSuffixOutcome<F, Cfg::ChallengeField>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + PseudoMersenneField
        + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + HasUnreducedOps
        + AkitaSerialize,
{
    let planned_num_levels = schedule_num_fold_levels(schedule);
    if planned_num_levels < 2 {
        return Err(AkitaError::InvalidSetup(
            "prove_recursive_suffix expects a non-empty recursive suffix".to_string(),
        ));
    }
    let terminal_level = planned_num_levels - 1;

    let mut intermediate_levels = Vec::new();
    let mut current_state = initial_state;
    let mut level = 1usize;

    while level < terminal_level {
        let inputs = AkitaScheduleInputs {
            num_vars,
            level,
            current_w_len: current_state.w.len(),
        };
        let (level_params, next_params) =
            scheduled_fold_execution(schedule, level, inputs, current_state.log_basis)?;
        let out = dispatch_prove_level::<F, T, D, Cfg>(
            level_params.ring_dimension,
            ntt_cache,
            &setup.expanded,
            &setup.ntt_shared,
            commit_ntt_cache,
            &current_state,
            transcript,
            level,
            &level_params,
            next_params,
        )?;
        intermediate_levels.push(out.level_proof);
        current_state = out.next_state;
        level += 1;
    }

    debug_assert_eq!(level, terminal_level);
    let inputs = AkitaScheduleInputs {
        num_vars,
        level,
        current_w_len: current_state.w.len(),
    };
    let (level_params, next_params) =
        scheduled_fold_execution(schedule, level, inputs, current_state.log_basis)?;
    let final_witness_shape = schedule_terminal_direct_witness_shape(schedule)?;
    let terminal = dispatch_prove_terminal_level::<F, T, D, Cfg>(
        level_params.ring_dimension,
        ntt_cache,
        &setup.expanded,
        &setup.ntt_shared,
        &current_state,
        transcript,
        level,
        &level_params,
        final_witness_shape,
    )?;
    let _ = next_params;

    Ok(RecursiveSuffixOutcome {
        intermediate_levels,
        terminal,
        num_levels: planned_num_levels,
    })
}

/// Top-level `<Cfg>`-generic prove entry point.
///
/// Closure-free orchestration: schedule selection, root-direct fallback,
/// descriptor binding, root proving (folded or terminal), and recursive
/// suffix proving are all sourced directly from `Cfg`.
///
/// # Errors
///
/// Returns an error when prepared claims, schedule selection, descriptor
/// binding, root proving, or recursive suffix proving fails.
pub fn prove_batched<'a, F, Cfg, T, P, const D: usize>(
    setup: &AkitaProverSetup<F, D>,
    claims: ProverClaims<'a, Cfg::ClaimField, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
    transcript: &mut T,
    basis: BasisMode,
) -> Result<AkitaBatchedProof<F, Cfg::ChallengeField>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HasUnreducedOps
        + HalvingField
        + FromPrimitiveInt
        + PseudoMersenneField
        + Valid
        + AkitaSerialize,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + HasUnreducedOps
        + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    let prepared_claims =
        prepare_batched_prove_inputs::<F, Cfg::ClaimField, P, D>(&setup.expanded, claims)?;
    let num_vars = prepared_claims.incidence_summary.num_vars();

    let mut schedule = Cfg::get_params_for_prove(&prepared_claims.incidence_summary)?;
    if let Some(root_step) = schedule_root_fold_step(&schedule) {
        let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
        if !folded_root_supports_opening_shape::<F, Cfg::ClaimField, Cfg::ChallengeField, D>(
            &prepared_claims.opening_points,
            &root_step.params,
            alpha_bits,
        ) && !root_tensor_projection_enabled::<F, Cfg::ClaimField, Cfg::ChallengeField, D>(
            num_vars,
        ) {
            // Use the real incidence (not a synthetic singleton) so the
            // schedule's `commit_params` describe the actual layout the
            // root-direct check below will recompute via
            // `get_params_for_batched_commitment`. Otherwise the
            // descriptor binds singleton-sized params while the
            // verifier checks against batched ones.
            let commit_params =
                Cfg::get_params_for_batched_commitment(&prepared_claims.incidence_summary)?;
            schedule = root_direct_schedule(num_vars, commit_params)?;
        }
    }

    bind_transcript_instance_descriptor::<F, T, D, Cfg>(
        &setup.expanded,
        &prepared_claims.incidence_summary,
        &schedule,
        basis,
        transcript,
    )?;

    if schedule_is_root_direct(&schedule) {
        return prove_root_direct::<F, Cfg::ChallengeField, D, P>(
            &prepared_claims.group_polys,
            &prepared_claims.flat_hints,
        );
    }

    let Some(root_step) = schedule_root_fold_step(&schedule) else {
        return Err(AkitaError::InvalidSetup(
            "root schedule does not start with a fold".to_string(),
        ));
    };

    if prepared_claims
        .commitments_by_point
        .iter()
        .any(|commitment| commitment.u.len() != root_step.params.b_key.row_len())
    {
        return Err(AkitaError::InvalidInput(
            "batched_prove received a commitment with the wrong length".to_string(),
        ));
    }

    let root_next_params = scheduled_next_level_params(&schedule, 1)?;

    if schedule_num_fold_levels(&schedule) == 1 {
        // Root is the terminal fold; no recursive suffix.
        let direct_step = match schedule.steps.get(1) {
            Some(Step::Direct(direct_step)) => direct_step.clone(),
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "1-fold schedule must terminate in a direct step".to_string(),
                ));
            }
        };
        let final_witness_shape = &direct_step.witness_shape;
        let terminal = prove_terminal_root_fold_with_params::<
            F,
            Cfg::ClaimField,
            Cfg::ChallengeField,
            T,
            D,
            P,
        >(
            &setup.expanded,
            &setup.ntt_shared,
            transcript,
            &prepared_claims.flat_polys,
            &prepared_claims.incidence_summary,
            &prepared_claims.opening_points,
            &prepared_claims.commitments_by_point,
            prepared_claims.flat_hints,
            &root_step.params,
            root_step.next_w_len,
            final_witness_shape,
            basis,
        )?;
        return Ok(build_terminal_root_batched_proof::<F, Cfg::ChallengeField>(
            terminal,
        ));
    }

    let mut ntt_cache = MultiDNttCaches::new();
    let mut commit_ntt_cache = MultiDNttCaches::new();

    let raw = prove_root_fold_with_params::<F, Cfg::ClaimField, Cfg::ChallengeField, T, D, P, _>(
        &setup.expanded,
        &setup.ntt_shared,
        transcript,
        &prepared_claims.flat_polys,
        &prepared_claims.incidence_summary,
        &prepared_claims.opening_points,
        &prepared_claims.commitments_by_point,
        prepared_claims.flat_hints,
        &root_step.params,
        root_step.next_w_len,
        root_next_params.log_basis,
        basis,
        |w| {
            commit_next_w::<F, Cfg, D>(
                &root_next_params,
                &setup.ntt_shared,
                &mut commit_ntt_cache,
                &setup.expanded,
                w,
                Cfg::decomposition(),
            )
        },
    )?;

    build_folded_batched_proof_with_suffix::<F, Cfg::ChallengeField, D, _>(raw, |next_state| {
        prove_recursive_suffix::<F, T, D, Cfg>(
            setup,
            &mut ntt_cache,
            &mut commit_ntt_cache,
            num_vars,
            transcript,
            next_state,
            &schedule,
        )
    })
    .map(|(proof, _total_levels)| proof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp2, Fp32, LiftBase, NegOneNr};
    use akita_transcript::AkitaTranscript;
    #[cfg(feature = "zk")]
    use akita_types::FlatDigitBlocks;
    use akita_types::{AkitaSetupSeed, FlatMatrix};

    type F = Fp32<251>;
    type E = Fp2<F, NegOneNr>;

    fn setup() -> AkitaExpandedSetup<F> {
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupSeed {
                max_num_vars: 3,
                max_num_batched_polys: 4,
                max_num_points: 2,
                max_stride: 1,
                gen_ring_dim: 1,
                total_ring_elements: 1,
                public_matrix_seed: [0u8; 32],
            },
            FlatMatrix::from_flat_data(vec![F::zero()], 1),
        )
    }

    #[test]
    fn prover_claim_preparation_accepts_extension_points() {
        let point = [
            E::new(F::from_u64(1), F::from_u64(2)),
            E::new(F::from_u64(3), F::from_u64(4)),
        ];
        let polys = [10usize, 11usize];
        let commitment = RingCommitment::<F, 2>::default();
        #[cfg(feature = "zk")]
        let hint = AkitaCommitmentHint::with_recomposed_inner_rows(
            Vec::new(),
            Vec::new(),
            vec![FlatDigitBlocks::empty()],
        );
        #[cfg(not(feature = "zk"))]
        let hint = AkitaCommitmentHint::new(Vec::new());
        let claims = vec![(
            &point[..],
            crate::CommittedPolynomials {
                polynomials: &polys[..],
                commitment: &commitment,
                hint,
            },
        )];

        let prepared = prepare_batched_prove_inputs::<F, E, usize, 2>(&setup(), claims)
            .expect("extension-valued prover points should validate by shape");

        assert_eq!(prepared.opening_points, vec![&point[..]]);
        assert_eq!(prepared.incidence_summary.num_claims(), 2);
        assert_eq!(prepared.incidence_summary.num_points(), 1);
        assert_eq!(prepared.incidence_summary.num_public_rows(), 1);
        assert_eq!(prepared.incidence_summary.num_polys_per_point(), &[2]);
        assert_eq!(prepared.incidence_summary.claim_to_point(), &[0, 0]);
        assert_eq!(prepared.flat_polys, vec![&polys[0], &polys[1]]);
        assert_eq!(prepared.group_polys, vec![&polys[0], &polys[1]]);
    }

    #[test]
    fn recursive_extension_opening_reduction_pads_to_opening_cube() {
        let logical_w = RecursiveWitnessFlat::from_i8_digits(vec![1, -1, 2, 0, 3, -2]);
        let point = [
            E::new(F::from_u64(2), F::from_u64(3)),
            E::new(F::from_u64(5), F::from_u64(7)),
            E::new(F::from_u64(11), F::from_u64(13)),
        ];
        let mut base_evals = recursive_witness_base_evals::<F>(&logical_w);
        base_evals.resize(1usize << point.len(), F::zero());
        let expected_opening =
            base_evals
                .iter()
                .enumerate()
                .fold(E::zero(), |acc, (idx, &eval)| {
                    let weight = point
                        .iter()
                        .enumerate()
                        .fold(E::one(), |weight, (bit, &x)| {
                            if (idx >> bit) & 1 == 1 {
                                weight * x
                            } else {
                                weight * (E::one() - x)
                            }
                        });
                    acc + weight * E::lift_base(eval)
                });

        let mut transcript =
            AkitaTranscript::<F>::new(b"test/recursive-extension-opening-reduction-padding");
        let reduction = prove_recursive_extension_opening_reduction::<F, E, _>(
            &logical_w,
            &point,
            expected_opening,
            &mut transcript,
        )
        .expect("padded logical witnesses should reduce over the opening cube");

        assert_eq!(
            reduction.proof.partials.len(),
            <E as ExtField<F>>::EXT_DEGREE
        );
        assert_eq!(reduction.proof.num_rounds(), point.len() - 1);
    }
}
