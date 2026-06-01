//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
#[cfg(not(feature = "zk"))]
use akita_algebra::CyclotomicRing;
use akita_challenges::Challenges;
#[cfg(not(feature = "zk"))]
use akita_challenges::IntegerChallenge;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, RandomSampling,
};
use akita_transcript::labels::{
    ABSORB_NEXT_LEVEL_WITNESS_BINDING, ABSORB_TERMINAL_W_REMAINDER, CHALLENGE_RING_SWITCH,
    CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use akita_transcript::{sample_ext_challenge, Transcript};
#[cfg(feature = "zk")]
use akita_types::zk;
#[cfg(not(feature = "zk"))]
use akita_types::DirectWitnessProof;
use akita_types::{
    embed_ring_subfield_scalar, gadget_row_scalars, r_decomp_levels,
    validate_opening_points_for_claims, AkitaExpandedSetup, FlatRingVec, LevelParams, MRowLayout,
    RingMultiplierOpeningPoint, RingOpeningPoint, RingRelationInstance, RingRelationSegmentLayout,
    RingSubfieldEncoding, SetupContributionPlanInputs, TerminalWitnessTranscriptParts,
};

#[cfg(feature = "zk")]
use super::slice_mle::{compute_b_blinding_part, compute_d_blinding_part};
use super::slice_mle::{
    compute_r_contribution, EStructuredSlicesEvaluator, SetupEvaluation, SetupEvaluator,
    SetupEvaluatorMode, StructuredSliceMleEvaluator, TStructuredSlicesEvaluator,
    ZDenseSlicesEvaluator, ZStructuredPow2SlicesEvaluator,
};
use super::{validate_level_dispatch, validate_log_basis, validate_ring_dispatch};
pub(crate) use tensor_challenges::PreparedChallengeEvals;

mod tensor_challenges;
#[cfg(test)]
mod tests;

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline(always)]
fn jolt_cycle_marker(marker_id_str: &str, event_type: u32) {
    const JOLT_CYCLE_TRACK_CALL_ID: u32 = 0xC7C1E;
    let marker_id = marker_id_str.as_ptr() as usize as u32;
    let marker_len = marker_id_str.len() as u32;
    unsafe {
        core::arch::asm!(
            ".insn i 0x5B, 2, x0, x0, 0",
            in("x10") JOLT_CYCLE_TRACK_CALL_ID,
            in("x11") marker_id,
            in("x12") marker_len,
            in("x13") event_type,
            options(nostack, preserves_flags)
        );
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline(always)]
fn jolt_start_cycle_tracking(marker_id: &str) {
    jolt_cycle_marker(marker_id, 1);
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
#[inline(always)]
fn jolt_start_cycle_tracking(_marker_id: &str) {}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline(always)]
fn jolt_end_cycle_tracking(marker_id: &str) {
    jolt_cycle_marker(marker_id, 2);
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
#[inline(always)]
fn jolt_end_cycle_tracking(_marker_id: &str) {}

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub(crate) struct RingSwitchVerifyOutput<E: FieldCore> {
    /// Prepared data for deferred ring-switch row MLE evaluation.
    pub prepared_row_eval: RingSwitchDeferredRowEval<E>,
    /// Evaluation table of alpha powers over the ring-coordinate dimension.
    pub alpha_evals_y: Vec<E>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
    /// Challenge tau0 for the stage-1 sumcheck.
    pub tau0: Vec<E>,
    /// Challenge tau1 for the stage-2 M-row combination.
    pub tau1: Vec<E>,
    /// Basis size `b = 2^log_basis`.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: E,
}

struct RingSwitchVerifyCoreOutput<E: FieldCore> {
    prepared_row_eval: RingSwitchDeferredRowEval<E>,
    alpha_evals_y: Vec<E>,
    col_bits: usize,
    ring_bits: usize,
    tau0: Option<Vec<E>>,
    tau1: Vec<E>,
    b: usize,
    alpha: E,
}

impl<E: FieldCore> RingSwitchVerifyCoreOutput<E> {
    fn into_intermediate(self) -> Result<RingSwitchVerifyOutput<E>, AkitaError> {
        let tau0 = self.tau0.ok_or(AkitaError::InvalidProof)?;
        Ok(RingSwitchVerifyOutput {
            prepared_row_eval: self.prepared_row_eval,
            alpha_evals_y: self.alpha_evals_y,
            col_bits: self.col_bits,
            ring_bits: self.ring_bits,
            tau0,
            tau1: self.tau1,
            b: self.b,
            alpha: self.alpha,
        })
    }

    fn into_terminal_as_output(self) -> Result<RingSwitchVerifyOutput<E>, AkitaError> {
        if self.tau0.is_some() {
            return Err(AkitaError::InvalidProof);
        }
        Ok(RingSwitchVerifyOutput {
            prepared_row_eval: self.prepared_row_eval,
            alpha_evals_y: self.alpha_evals_y,
            col_bits: self.col_bits,
            ring_bits: self.ring_bits,
            tau0: Vec::new(),
            tau1: self.tau1,
            b: self.b,
            alpha: self.alpha,
        })
    }
}

/// Precomputed challenge-derived data for deferred ring-switch row MLE evaluation.
///
/// Stores only data that cannot be derived from context at evaluation time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else is passed by reference at evaluation time to avoid
/// duplicating setup matrix views, opening points, and gadget vectors.
#[derive(Clone)]
pub struct RingSwitchDeferredRowEval<F: FieldCore> {
    pub(crate) c_alphas: PreparedChallengeEvals<F>,
    pub(crate) eq_tau1: Vec<F>,
    pub(crate) num_t_vectors: usize,
    pub(crate) num_blocks: usize,
    pub(crate) num_claims: usize,
    pub(crate) depth_open: usize,
    pub(crate) depth_commit: usize,
    pub(crate) depth_fold: usize,
    #[cfg(feature = "zk")]
    pub(crate) d_blinding_segment_len: usize,
    #[cfg(feature = "zk")]
    pub(crate) b_blinding_digit_planes_per_point: usize,
    #[cfg(feature = "zk")]
    pub(crate) b_blinding_segment_len: usize,
    pub(crate) block_len: usize,
    pub(crate) inner_width: usize,
    pub(crate) log_basis: u32,
    pub(crate) n_a: usize,
    pub(crate) n_d: usize,
    pub(crate) m_row_layout: MRowLayout,
    pub(crate) n_b: usize,
    pub(crate) num_points: usize,
    pub(crate) rows: usize,
    pub(crate) claim_to_commitment_group_poly: Vec<(usize, usize)>,
    pub(crate) num_polys_per_commitment_group: Vec<usize>,
    pub(crate) num_public_rows: usize,
    pub(crate) gamma: Vec<F>,
    pub(crate) claim_to_point: Vec<usize>,
    pub(crate) witness_segment_layout: RingRelationSegmentLayout,
}

pub(crate) type RingSwitchSegmentLayout = RingRelationSegmentLayout;

/// Fixed public relation inputs for verifier ring-switch replay.
pub struct RingSwitchReplay<'a, F: FieldCore, E, const D: usize> {
    pub relation: &'a RingRelationInstance<F, D>,
    pub row_coefficients: &'a [E],
    pub lp: &'a LevelParams,
}

#[cfg(not(feature = "zk"))]
#[derive(Clone)]
struct TerminalDirectSegments<F: FieldCore, const D: usize> {
    w_folded: Vec<CyclotomicRing<F, D>>,
    t_digits: Vec<CyclotomicRing<F, D>>,
    t_recomposed_rows: Vec<Vec<CyclotomicRing<F, D>>>,
    z_pre: Vec<CyclotomicRing<F, D>>,
}

/// Replay the verifier half of ring switching.
///
/// This handles multiple opening points, arbitrary claim-to-point mapping, and
/// point-local polynomial bundles. The recursive/single-point path is the
/// `opening_points = [pt]`, `claim_to_point = [0]`,
/// `num_polys_per_commitment_group = [1]`, `num_public_rows = 1` specialization.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or ring-switch row-eval
/// preparation fails.
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub(crate) fn ring_switch_verifier<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    w_len: usize,
    w_commitment: &FlatRingVec<F>,
    transcript: &mut T,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    // `validate_ring_dispatch` is called inside `ring_switch_verifier_core`;
    // the outer wrapper just performs the witness absorb before delegating.
    transcript.record_wire_serde(ABSORB_NEXT_LEVEL_WITNESS_BINDING, w_commitment);
    transcript.append_serde(ABSORB_NEXT_LEVEL_WITNESS_BINDING, w_commitment);
    ring_switch_verifier_core::<F, E, T, D>(replay, w_len, transcript, MRowLayout::WithDBlock)?
        .into_intermediate()
}

/// Terminal variant of [`ring_switch_verifier`].
///
/// This owns the required terminal final-witness remainder absorb before
/// sampling ring-switch challenges.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or
/// ring-switch row-eval preparation fails.
#[tracing::instrument(skip_all, name = "ring_switch_verifier_terminal")]
#[inline(never)]
pub(crate) fn ring_switch_verifier_terminal<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    w_len: usize,
    transcript: &mut T,
    terminal_parts: &TerminalWitnessTranscriptParts,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    transcript.record_wire_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_parts.remainder);
    transcript.append_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_parts.remainder);
    ring_switch_verifier_core::<F, E, T, D>(replay, w_len, transcript, MRowLayout::WithoutDBlock)?
        .into_terminal_as_output()
}

/// Verify terminal direct ring relations without sampling ring-switch or
/// stage-2 challenges.
///
/// The caller must have already bound the descriptor-selected terminal
/// `w_hat` slice and sampled the stage-1 folding challenges. This function
/// binds the remaining final-witness bytes, decodes the transparent terminal
/// witness, and checks the reduced M-row equations directly.
#[cfg(not(feature = "zk"))]
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "verify_terminal_direct_ring_relations")]
pub(crate) fn verify_terminal_direct_ring_relations<F, E, T, const D: usize>(
    opening_points: &[RingOpeningPoint<F>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    challenges: &Challenges,
    final_w_len: usize,
    final_witness: &DirectWitnessProof<F>,
    transcript: &mut T,
    terminal_parts: &TerminalWitnessTranscriptParts,
    setup: &AkitaExpandedSetup<F>,
    lp: &LevelParams,
    num_polys_per_point: &[usize],
    claim_to_point_poly: &[usize],
    claim_poly_indices: &[usize],
    gamma: &[E],
    commitment_rows: &[CyclotomicRing<F, D>],
    y_rings: &[CyclotomicRing<F, D>],
    num_public_rows: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    transcript.record_wire_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_parts.remainder);
    transcript.append_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_parts.remainder);
    verify_terminal_direct_relation_rows::<F, E, D>(
        opening_points,
        ring_multiplier_points,
        claim_to_point,
        challenges,
        final_w_len,
        final_witness,
        setup,
        lp,
        num_polys_per_point,
        claim_to_point_poly,
        claim_poly_indices,
        gamma,
        commitment_rows,
        y_rings,
        num_public_rows,
    )
}

#[cfg(not(feature = "zk"))]
#[allow(clippy::too_many_arguments)]
fn verify_terminal_direct_relation_rows<F, E, const D: usize>(
    opening_points: &[RingOpeningPoint<F>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    challenges: &Challenges,
    final_w_len: usize,
    final_witness: &DirectWitnessProof<F>,
    setup: &AkitaExpandedSetup<F>,
    lp: &LevelParams,
    num_polys_per_point: &[usize],
    claim_to_point_poly: &[usize],
    claim_poly_indices: &[usize],
    gamma: &[E],
    commitment_rows: &[CyclotomicRing<F, D>],
    y_rings: &[CyclotomicRing<F, D>],
    num_public_rows: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
{
    validate_terminal_direct_shape::<F, D>(
        opening_points,
        ring_multiplier_points,
        claim_to_point,
        challenges,
        lp,
        num_polys_per_point,
        claim_to_point_poly,
        claim_poly_indices,
        commitment_rows,
        y_rings,
        num_public_rows,
    )?;
    if gamma.len() != claim_to_point.len() || final_witness.num_elems() != final_w_len {
        return Err(AkitaError::InvalidProof);
    }

    let challenge_rings = materialize_challenge_rings::<F, D>(challenges)?;
    let segments = decode_terminal_direct_segments::<F, D>(
        final_witness,
        final_w_len,
        lp,
        claim_to_point.len(),
        num_polys_per_point,
        num_public_rows,
    )?;
    let row_coefficient_rings = gamma
        .iter()
        .copied()
        .map(|coefficient| {
            embed_ring_subfield_scalar::<F, E, D>(coefficient, AkitaError::InvalidProof)
        })
        .collect::<Result<Vec<_>, _>>()?;

    check_terminal_direct_consistency_row::<F, D>(
        &segments.w_folded,
        &segments.z_pre,
        &challenge_rings,
        ring_multiplier_points,
        lp,
        num_public_rows,
    )?;
    check_terminal_direct_public_rows::<F, D>(
        &segments.w_folded,
        &row_coefficient_rings,
        ring_multiplier_points,
        claim_to_point,
        y_rings,
        lp.num_blocks,
    )?;
    check_terminal_direct_b_rows::<F, D>(
        setup,
        &segments.t_digits,
        lp,
        num_polys_per_point,
        commitment_rows,
    )?;
    check_terminal_direct_a_rows::<F, D>(
        setup,
        &segments.t_recomposed_rows,
        &segments.z_pre,
        &challenge_rings,
        lp,
        num_polys_per_point,
        claim_to_point_poly,
        claim_poly_indices,
        num_public_rows,
    )
}

#[cfg(not(feature = "zk"))]
#[allow(clippy::too_many_arguments)]
fn validate_terminal_direct_shape<F, const D: usize>(
    opening_points: &[RingOpeningPoint<F>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    challenges: &Challenges,
    lp: &LevelParams,
    num_polys_per_point: &[usize],
    claim_to_point_poly: &[usize],
    claim_poly_indices: &[usize],
    commitment_rows: &[CyclotomicRing<F, D>],
    y_rings: &[CyclotomicRing<F, D>],
    num_public_rows: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    validate_level_dispatch::<D>(lp)?;
    validate_log_basis(lp.log_basis)?;
    let num_claims = claim_to_point.len();
    let num_points = num_polys_per_point.len();
    if num_claims == 0
        || num_points == 0
        || num_public_rows != num_points
        || y_rings.len() != num_public_rows
        || num_polys_per_point.contains(&0)
    {
        return Err(AkitaError::InvalidProof);
    }
    validate_opening_points_for_claims(opening_points, claim_to_point, lp, num_claims)?;
    if opening_points.len() != num_points || ring_multiplier_points.len() != num_public_rows {
        return Err(AkitaError::InvalidProof);
    }
    if ring_multiplier_points
        .iter()
        .any(|point| point.a_len() < lp.block_len || point.b_len() != lp.num_blocks)
    {
        return Err(AkitaError::InvalidProof);
    }
    if claim_to_point_poly.len() != num_claims
        || claim_poly_indices.len() != num_claims
        || challenges.logical_len()
            != num_claims
                .checked_mul(lp.num_blocks)
                .ok_or(AkitaError::InvalidProof)?
    {
        return Err(AkitaError::InvalidProof);
    }
    for claim_idx in 0..num_claims {
        let point_idx = *claim_to_point_poly
            .get(claim_idx)
            .ok_or(AkitaError::InvalidProof)?;
        let poly_idx = *claim_poly_indices
            .get(claim_idx)
            .ok_or(AkitaError::InvalidProof)?;
        let point_poly_count = *num_polys_per_point
            .get(point_idx)
            .ok_or(AkitaError::InvalidProof)?;
        if poly_idx >= point_poly_count {
            return Err(AkitaError::InvalidProof);
        }
    }
    let expected_commitment_rows = lp
        .b_key
        .row_len()
        .checked_mul(num_points)
        .ok_or(AkitaError::InvalidProof)?;
    if commitment_rows.len() != expected_commitment_rows {
        return Err(AkitaError::InvalidProof);
    }
    if lp.num_blocks == 0
        || !lp.num_blocks.is_power_of_two()
        || lp.block_len == 0
        || lp.num_digits_open == 0
        || lp.num_digits_commit == 0
        || lp.num_digits_fold == 0
    {
        return Err(AkitaError::InvalidSetup(
            "terminal direct verifier layout has zero width".to_string(),
        ));
    }
    Ok(())
}

#[cfg(not(feature = "zk"))]
fn materialize_challenge_rings<F, const D: usize>(
    challenges: &Challenges,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    match challenges {
        Challenges::Sparse {
            challenges: sparse, ..
        } => sparse
            .iter()
            .map(sparse_challenge_to_ring::<F, D>)
            .collect(),
        Challenges::Tensor { factored } => factored
            .expand_integer::<D>()?
            .iter()
            .map(integer_challenge_to_ring::<F, D>)
            .collect(),
    }
}

#[cfg(not(feature = "zk"))]
fn sparse_challenge_to_ring<F, const D: usize>(
    challenge: &akita_challenges::SparseChallenge,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    challenge.validate::<D>()?;
    let mut coeffs = [F::zero(); D];
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let slot = coeffs
            .get_mut(pos as usize)
            .ok_or(AkitaError::InvalidProof)?;
        *slot += F::from_i64(i64::from(coeff));
    }
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

#[cfg(not(feature = "zk"))]
fn integer_challenge_to_ring<F, const D: usize>(
    challenge: &IntegerChallenge,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    if challenge.positions.len() != challenge.coeffs.len() {
        return Err(AkitaError::InvalidInput(
            "integer challenge positions/coeffs length mismatch".to_string(),
        ));
    }
    let mut coeffs = [F::zero(); D];
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        if coeff == 0 {
            return Err(AkitaError::InvalidInput(
                "integer challenge coefficients must be non-zero".to_string(),
            ));
        }
        let slot = coeffs.get_mut(pos as usize).ok_or_else(|| {
            AkitaError::InvalidInput(format!(
                "integer challenge position {pos} out of range for D={D}"
            ))
        })?;
        *slot += F::from_i64(i64::from(coeff));
    }
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

#[cfg(not(feature = "zk"))]
fn decode_terminal_direct_segments<F, const D: usize>(
    final_witness: &DirectWitnessProof<F>,
    final_w_len: usize,
    lp: &LevelParams,
    num_claims: usize,
    num_polys_per_point: &[usize],
    num_public_rows: usize,
) -> Result<TerminalDirectSegments<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    validate_log_basis(lp.log_basis)?;
    let digits = final_witness.packed_i8_digits()?;
    if digits.len() != final_w_len || !digits.len().is_multiple_of(D) {
        return Err(AkitaError::InvalidProof);
    }
    let planes = digits
        .chunks_exact(D)
        .map(|chunk| {
            let mut plane = [0i8; D];
            plane.copy_from_slice(chunk);
            plane
        })
        .collect::<Vec<_>>();
    let layout =
        terminal_direct_plane_layout(lp, num_claims, num_polys_per_point, num_public_rows)?;
    if planes.len() != layout.total_planes {
        return Err(AkitaError::InvalidProof);
    }
    let w_folded = decode_terminal_direct_w_folded::<F, D>(&planes, &layout, lp)?;
    let (t_digits, t_recomposed_rows) =
        decode_terminal_direct_t::<F, D>(&planes, &layout, lp, num_polys_per_point)?;
    let z_pre = decode_terminal_direct_z::<F, D>(&planes, &layout, lp, num_public_rows)?;
    Ok(TerminalDirectSegments {
        w_folded,
        t_digits,
        t_recomposed_rows,
        z_pre,
    })
}

#[cfg(not(feature = "zk"))]
#[derive(Clone, Copy)]
struct TerminalDirectPlaneLayout {
    offset_w: usize,
    offset_t: usize,
    offset_z: usize,
    total_blocks: usize,
    t_total_blocks: usize,
    total_planes: usize,
}

#[cfg(not(feature = "zk"))]
fn terminal_direct_plane_layout(
    lp: &LevelParams,
    num_claims: usize,
    num_polys_per_point: &[usize],
    num_public_rows: usize,
) -> Result<TerminalDirectPlaneLayout, AkitaError> {
    let total_blocks = lp
        .num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal direct W width overflow".to_string()))?;
    let total_poly_slots = num_polys_per_point.iter().try_fold(0usize, |acc, &count| {
        acc.checked_add(count)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal direct T width overflow".to_string()))
    })?;
    let t_total_blocks = lp.num_blocks.checked_mul(total_poly_slots).ok_or_else(|| {
        AkitaError::InvalidSetup("terminal direct T block count overflow".to_string())
    })?;
    let w_len = lp
        .num_digits_open
        .checked_mul(total_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal direct W width overflow".to_string()))?;
    let t_len = lp
        .num_digits_open
        .checked_mul(lp.a_key.row_len())
        .and_then(|len| len.checked_mul(t_total_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("terminal direct T width overflow".to_string()))?;
    let z_len = lp
        .num_digits_fold
        .checked_mul(lp.inner_width())
        .and_then(|len| len.checked_mul(num_public_rows))
        .ok_or_else(|| AkitaError::InvalidSetup("terminal direct Z width overflow".to_string()))?;
    let offset_z = if lp.m_vars >= lp.r_vars {
        0
    } else {
        w_len.checked_add(t_len).ok_or_else(|| {
            AkitaError::InvalidSetup("terminal direct Z offset overflow".to_string())
        })?
    };
    let offset_w = if lp.m_vars >= lp.r_vars { z_len } else { 0 };
    let offset_t = if lp.m_vars >= lp.r_vars {
        z_len.checked_add(w_len).ok_or_else(|| {
            AkitaError::InvalidSetup("terminal direct T offset overflow".to_string())
        })?
    } else {
        w_len
    };
    let total_planes = w_len
        .checked_add(t_len)
        .and_then(|len| len.checked_add(z_len))
        .ok_or_else(|| AkitaError::InvalidSetup("terminal direct width overflow".to_string()))?;
    Ok(TerminalDirectPlaneLayout {
        offset_w,
        offset_t,
        offset_z,
        total_blocks,
        t_total_blocks,
        total_planes,
    })
}

#[cfg(not(feature = "zk"))]
fn plane_at<const D: usize>(planes: &[[i8; D]], idx: usize) -> Result<[i8; D], AkitaError> {
    planes.get(idx).copied().ok_or(AkitaError::InvalidProof)
}

#[cfg(not(feature = "zk"))]
fn recompose_i8_planes<F, const D: usize>(
    planes: &[[i8; D]],
    log_basis: u32,
) -> CyclotomicRing<F, D>
where
    F: FieldCore + CanonicalField,
{
    CyclotomicRing::gadget_recompose_pow2_i8(planes, log_basis)
}

#[cfg(not(feature = "zk"))]
fn decode_terminal_direct_w_folded<F, const D: usize>(
    planes: &[[i8; D]],
    layout: &TerminalDirectPlaneLayout,
    lp: &LevelParams,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let mut out = Vec::with_capacity(layout.total_blocks);
    for block_idx in 0..layout.total_blocks {
        let digits = (0..lp.num_digits_open)
            .map(|digit_idx| {
                layout
                    .offset_w
                    .checked_add(
                        digit_idx
                            .checked_mul(layout.total_blocks)
                            .and_then(|offset| offset.checked_add(block_idx))
                            .ok_or(AkitaError::InvalidProof)?,
                    )
                    .ok_or(AkitaError::InvalidProof)
                    .and_then(|idx| plane_at(planes, idx))
            })
            .collect::<Result<Vec<_>, _>>()?;
        out.push(recompose_i8_planes::<F, D>(&digits, lp.log_basis));
    }
    Ok(out)
}

#[cfg(not(feature = "zk"))]
type TerminalDirectTDigits<F, const D: usize> =
    (Vec<CyclotomicRing<F, D>>, Vec<Vec<CyclotomicRing<F, D>>>);

#[cfg(not(feature = "zk"))]
fn decode_terminal_direct_t<F, const D: usize>(
    planes: &[[i8; D]],
    layout: &TerminalDirectPlaneLayout,
    lp: &LevelParams,
    num_polys_per_point: &[usize],
) -> Result<TerminalDirectTDigits<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let n_a = lp.a_key.row_len();
    let planes_per_block = n_a
        .checked_mul(lp.num_digits_open)
        .ok_or(AkitaError::InvalidProof)?;
    let t_digit_len = layout
        .t_total_blocks
        .checked_mul(planes_per_block)
        .ok_or(AkitaError::InvalidProof)?;
    let mut t_digits = vec![CyclotomicRing::<F, D>::zero(); t_digit_len];
    let mut recomposed = Vec::with_capacity(layout.t_total_blocks);
    for flat_block_idx in 0..layout.t_total_blocks {
        let mut rows = Vec::with_capacity(n_a);
        for a_idx in 0..n_a {
            let digits = (0..lp.num_digits_open)
                .map(|digit_idx| {
                    let compound_digit = a_idx
                        .checked_mul(lp.num_digits_open)
                        .and_then(|offset| offset.checked_add(digit_idx))
                        .ok_or(AkitaError::InvalidProof)?;
                    let source_idx = layout
                        .offset_t
                        .checked_add(
                            compound_digit
                                .checked_mul(layout.t_total_blocks)
                                .and_then(|offset| offset.checked_add(flat_block_idx))
                                .ok_or(AkitaError::InvalidProof)?,
                        )
                        .ok_or(AkitaError::InvalidProof)?;
                    plane_at(planes, source_idx)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let ring = recompose_i8_planes::<F, D>(&digits, lp.log_basis);
            rows.push(ring);
            for (digit_idx, plane) in digits.into_iter().enumerate() {
                let row_digit_idx = a_idx
                    .checked_mul(lp.num_digits_open)
                    .and_then(|idx| idx.checked_add(digit_idx))
                    .ok_or(AkitaError::InvalidProof)?;
                let target_idx = flat_block_idx
                    .checked_mul(planes_per_block)
                    .and_then(|idx| idx.checked_add(row_digit_idx))
                    .ok_or(AkitaError::InvalidProof)?;
                let slot = t_digits
                    .get_mut(target_idx)
                    .ok_or(AkitaError::InvalidProof)?;
                *slot = recompose_i8_planes::<F, D>(&[plane], lp.log_basis);
            }
        }
        recomposed.push(rows);
    }

    let expected_blocks = num_polys_per_point
        .iter()
        .try_fold(0usize, |acc, &count| {
            acc.checked_add(count).ok_or(AkitaError::InvalidProof)
        })?
        .checked_mul(lp.num_blocks)
        .ok_or(AkitaError::InvalidProof)?;
    if recomposed.len() != expected_blocks || expected_blocks != layout.t_total_blocks {
        return Err(AkitaError::InvalidProof);
    }
    Ok((t_digits, recomposed))
}

#[cfg(not(feature = "zk"))]
fn decode_terminal_direct_z<F, const D: usize>(
    planes: &[[i8; D]],
    layout: &TerminalDirectPlaneLayout,
    lp: &LevelParams,
    num_public_rows: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let inner_width = lp.inner_width();
    let expected = num_public_rows
        .checked_mul(inner_width)
        .ok_or(AkitaError::InvalidProof)?;
    let mut out = Vec::with_capacity(expected);
    for point_idx in 0..num_public_rows {
        for block_idx in 0..lp.block_len {
            for commit_digit_idx in 0..lp.num_digits_commit {
                let digits = (0..lp.num_digits_fold)
                    .map(|fold_digit_idx| {
                        let source_idx = layout
                            .offset_z
                            .checked_add(
                                commit_digit_idx
                                    .checked_mul(lp.num_digits_fold)
                                    .and_then(|idx| idx.checked_add(fold_digit_idx))
                                    .and_then(|idx| idx.checked_mul(num_public_rows))
                                    .and_then(|idx| idx.checked_mul(lp.block_len))
                                    .and_then(|idx| {
                                        point_idx
                                            .checked_mul(lp.block_len)
                                            .and_then(|offset| offset.checked_add(block_idx))
                                            .and_then(|offset| idx.checked_add(offset))
                                    })
                                    .ok_or(AkitaError::InvalidProof)?,
                            )
                            .ok_or(AkitaError::InvalidProof)?;
                        plane_at(planes, source_idx)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                out.push(recompose_i8_planes::<F, D>(&digits, lp.log_basis));
            }
        }
    }
    if out.len() != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(out)
}

#[cfg(not(feature = "zk"))]
fn ring_dot<F, const D: usize>(
    row: &[CyclotomicRing<F, D>],
    input: &[CyclotomicRing<F, D>],
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore,
{
    if input.len() > row.len() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(row
        .iter()
        .zip(input.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (lhs, rhs)| {
            acc + (*lhs * *rhs)
        }))
}

#[cfg(not(feature = "zk"))]
fn check_terminal_direct_consistency_row<F, const D: usize>(
    w_folded: &[CyclotomicRing<F, D>],
    z_pre: &[CyclotomicRing<F, D>],
    challenge_rings: &[CyclotomicRing<F, D>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    lp: &LevelParams,
    num_public_rows: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let g_commit = gadget_row_scalars::<F>(lp.num_digits_commit, lp.log_basis);
    let mut folded = CyclotomicRing::<F, D>::zero();
    for (challenge, w) in challenge_rings.iter().zip(w_folded.iter()) {
        folded += *challenge * *w;
    }
    let inner_width = lp.inner_width();
    let mut z_reduced = CyclotomicRing::<F, D>::zero();
    for point_idx in 0..num_public_rows {
        let point = ring_multiplier_points
            .get(point_idx)
            .ok_or(AkitaError::InvalidProof)?;
        for block_idx in 0..lp.block_len {
            let mut z_block = CyclotomicRing::<F, D>::zero();
            for digit_idx in 0..lp.num_digits_commit {
                let z_idx = point_idx
                    .checked_mul(inner_width)
                    .and_then(|idx| {
                        block_idx
                            .checked_mul(lp.num_digits_commit)
                            .and_then(|offset| offset.checked_add(digit_idx))
                            .and_then(|offset| idx.checked_add(offset))
                    })
                    .ok_or(AkitaError::InvalidProof)?;
                let z = z_pre.get(z_idx).ok_or(AkitaError::InvalidProof)?;
                let gadget = g_commit
                    .get(digit_idx)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?;
                z_block += z.scale(&gadget);
            }
            if let Some(scalar) = point.a_constant_coeff(block_idx) {
                z_reduced += z_block.scale(&scalar);
            } else {
                let multiplier = point
                    .a_rings()
                    .and_then(|rings| rings.get(block_idx))
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?;
                z_reduced += multiplier * z_block;
            }
        }
    }
    if folded != z_reduced {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

#[cfg(not(feature = "zk"))]
fn check_terminal_direct_public_rows<F, const D: usize>(
    w_folded: &[CyclotomicRing<F, D>],
    row_coefficient_rings: &[CyclotomicRing<F, D>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    y_rings: &[CyclotomicRing<F, D>],
    blocks_per_claim: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore,
{
    for (target_point_idx, expected) in y_rings.iter().enumerate() {
        let mut actual = CyclotomicRing::<F, D>::zero();
        for (claim_idx, &point_idx) in claim_to_point.iter().enumerate() {
            if point_idx != target_point_idx {
                continue;
            }
            let point = ring_multiplier_points
                .get(point_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let coefficient_ring = row_coefficient_rings
                .get(claim_idx)
                .ok_or(AkitaError::InvalidProof)?;
            for block_idx in 0..blocks_per_claim {
                let folded_idx = claim_idx
                    .checked_mul(blocks_per_claim)
                    .and_then(|idx| idx.checked_add(block_idx))
                    .ok_or(AkitaError::InvalidProof)?;
                let folded = w_folded.get(folded_idx).ok_or(AkitaError::InvalidProof)?;
                let weighted_multiplier = if let Some(scalar) = point.b_constant_coeff(block_idx) {
                    coefficient_ring.scale(&scalar)
                } else {
                    let b_ring = point
                        .b_rings()
                        .and_then(|rings| rings.get(block_idx))
                        .copied()
                        .ok_or(AkitaError::InvalidProof)?;
                    *coefficient_ring * b_ring
                };
                actual += weighted_multiplier * *folded;
            }
        }
        if &actual != expected {
            return Err(AkitaError::InvalidProof);
        }
    }
    Ok(())
}

#[cfg(not(feature = "zk"))]
fn check_terminal_direct_b_rows<F, const D: usize>(
    setup: &AkitaExpandedSetup<F>,
    t_digits: &[CyclotomicRing<F, D>],
    lp: &LevelParams,
    num_polys_per_point: &[usize],
    commitment_rows: &[CyclotomicRing<F, D>],
) -> Result<(), AkitaError>
where
    F: FieldCore,
{
    let n_b = lp.b_key.row_len();
    let n_a = lp.a_key.row_len();
    let per_poly_cols = lp
        .num_blocks
        .checked_mul(n_a)
        .and_then(|cols| cols.checked_mul(lp.num_digits_open))
        .ok_or(AkitaError::InvalidProof)?;
    let max_point_poly_count = num_polys_per_point.iter().copied().max().unwrap_or(0);
    let b_stride = max_point_poly_count
        .checked_mul(per_poly_cols)
        .ok_or(AkitaError::InvalidProof)?;
    let b_view = setup.shared_matrix().ring_view::<D>(n_b, b_stride)?;
    let mut group_offset = 0usize;
    for (point_idx, &group_size) in num_polys_per_point.iter().enumerate() {
        let group_len = group_size
            .checked_mul(per_poly_cols)
            .ok_or(AkitaError::InvalidProof)?;
        let start = group_offset
            .checked_mul(per_poly_cols)
            .ok_or(AkitaError::InvalidProof)?;
        let end = start
            .checked_add(group_len)
            .ok_or(AkitaError::InvalidProof)?;
        let group_t = t_digits.get(start..end).ok_or(AkitaError::InvalidProof)?;
        for b_idx in 0..n_b {
            let actual = ring_dot(b_view.row(b_idx)?, group_t)?;
            let commitment_idx = point_idx
                .checked_mul(n_b)
                .and_then(|idx| idx.checked_add(b_idx))
                .ok_or(AkitaError::InvalidProof)?;
            if commitment_rows
                .get(commitment_idx)
                .ok_or(AkitaError::InvalidProof)?
                != &actual
            {
                return Err(AkitaError::InvalidProof);
            }
        }
        group_offset = group_offset
            .checked_add(group_size)
            .ok_or(AkitaError::InvalidProof)?;
    }
    Ok(())
}

#[cfg(not(feature = "zk"))]
#[allow(clippy::too_many_arguments)]
fn check_terminal_direct_a_rows<F, const D: usize>(
    setup: &AkitaExpandedSetup<F>,
    t_recomposed_rows: &[Vec<CyclotomicRing<F, D>>],
    z_pre: &[CyclotomicRing<F, D>],
    challenge_rings: &[CyclotomicRing<F, D>],
    lp: &LevelParams,
    num_polys_per_point: &[usize],
    claim_to_point_poly: &[usize],
    claim_poly_indices: &[usize],
    num_public_rows: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore,
{
    let n_a = lp.a_key.row_len();
    let inner_width = lp.inner_width();
    let a_view = setup.shared_matrix().ring_view::<D>(n_a, inner_width)?;
    let mut group_offsets = Vec::with_capacity(num_polys_per_point.len());
    let mut offset = 0usize;
    for &count in num_polys_per_point {
        group_offsets.push(offset);
        offset = offset.checked_add(count).ok_or(AkitaError::InvalidProof)?;
    }
    for a_idx in 0..n_a {
        let mut lhs = CyclotomicRing::<F, D>::zero();
        for (challenge_idx, challenge) in challenge_rings.iter().enumerate() {
            let claim_idx = challenge_idx
                .checked_div(lp.num_blocks)
                .ok_or(AkitaError::InvalidProof)?;
            let block_idx = challenge_idx % lp.num_blocks;
            let point_idx = *claim_to_point_poly
                .get(claim_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let poly_idx = *claim_poly_indices
                .get(claim_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let poly_slot = group_offsets
                .get(point_idx)
                .and_then(|base| base.checked_add(poly_idx))
                .ok_or(AkitaError::InvalidProof)?;
            let inner_idx = poly_slot
                .checked_mul(lp.num_blocks)
                .and_then(|idx| idx.checked_add(block_idx))
                .ok_or(AkitaError::InvalidProof)?;
            let row_value = t_recomposed_rows
                .get(inner_idx)
                .and_then(|rows| rows.get(a_idx))
                .copied()
                .ok_or(AkitaError::InvalidProof)?;
            lhs += *challenge * row_value;
        }
        let mut rhs = CyclotomicRing::<F, D>::zero();
        let row = a_view.row(a_idx)?;
        for point_idx in 0..num_public_rows {
            let start = point_idx
                .checked_mul(inner_width)
                .ok_or(AkitaError::InvalidProof)?;
            let end = start
                .checked_add(inner_width)
                .ok_or(AkitaError::InvalidProof)?;
            let z_segment = z_pre.get(start..end).ok_or(AkitaError::InvalidProof)?;
            rhs += ring_dot(row, z_segment)?;
        }
        if lhs != rhs {
            return Err(AkitaError::InvalidProof);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "ring_switch_verifier_core")]
#[inline(never)]
fn ring_switch_verifier_core<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    w_len: usize,
    transcript: &mut T,
    m_row_layout: MRowLayout,
) -> Result<RingSwitchVerifyCoreOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_points = relation.opening_points();
    let ring_multiplier_points = relation.ring_multiplier_points();
    let claim_to_point = relation.claim_to_point();
    let routing = relation.commitment_routing();
    let num_polys_per_commitment_group = routing.num_polys_per_commitment_group();
    let claim_to_commitment_group = routing.claim_to_commitment_group();
    let claim_poly_in_commitment_group = routing.claim_poly_in_commitment_group();
    let num_public_rows = relation.num_public_rows();
    let gamma = replay.row_coefficients;

    let alpha: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH);

    let num_claims = claim_to_point.len();
    validate_opening_points_for_claims(opening_points, claim_to_point, lp, num_claims)?;
    if ring_multiplier_points.len() != opening_points.len()
        || ring_multiplier_points
            .iter()
            .any(|point| point.a_len() < lp.block_len || point.b_len() != lp.num_blocks)
    {
        return Err(AkitaError::InvalidProof);
    }
    if claim_to_commitment_group.len() != num_claims
        || claim_poly_in_commitment_group.len() != num_claims
    {
        return Err(AkitaError::InvalidProof);
    }
    let num_points = num_polys_per_commitment_group.len();
    for claim_idx in 0..num_claims {
        let group_idx = claim_to_commitment_group[claim_idx];
        if group_idx >= num_points
            || claim_poly_in_commitment_group[claim_idx]
                >= num_polys_per_commitment_group[group_idx]
        {
            return Err(AkitaError::InvalidProof);
        }
    }

    if w_len == 0 || !w_len.is_multiple_of(D) {
        return Err(AkitaError::InvalidProof);
    }
    let num_ring_elems = w_len / D;
    let col_bits = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch column count overflow".to_string()))?
        .trailing_zeros() as usize;
    let ring_bits = validate_ring_dispatch::<D>()?;
    let m_rows = lp.m_row_count_for(num_points, num_public_rows, m_row_layout)?;
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch row count overflow".to_string()))?
        .trailing_zeros() as usize;

    let tau0 = match m_row_layout {
        MRowLayout::WithDBlock => Some(
            (0..num_sc_vars)
                .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
                .collect(),
        ),
        MRowLayout::WithoutDBlock => None,
    };
    let tau1: Vec<E> = (0..num_i)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
        .collect();
    let alpha_evals_y = scalar_powers(alpha, D);
    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let prepared_row_eval = prepare_ring_switch_row_eval::<F, E, D>(replay, alpha, &tau1)?;

    Ok(RingSwitchVerifyCoreOutput {
        prepared_row_eval,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b: 1usize
            .checked_shl(lp.log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("basis size overflow".to_string()))?,
        alpha,
    })
}

/// Prepare deferred verifier ring-switch row evaluation data from a fixed
/// [`RingRelationInstance`] and transcript-sampled row coefficients.
///
/// # Errors
///
/// Returns an error if gamma/challenge lengths do not match the claim shape,
/// the expanded tau1 table is too short for the level layout, or sparse
/// challenge evaluation fails.
#[tracing::instrument(skip_all, name = "prepare_ring_switch_row_eval")]
pub fn prepare_ring_switch_row_eval<F, E, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    alpha: E,
    tau1: &[E],
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let witness_segment_layout = relation.segment_layout(lp)?;
    let routing = relation.commitment_routing();
    prepare_ring_switch_row_eval_inner::<F, E, D>(
        &relation.challenges,
        alpha,
        lp,
        tau1,
        routing.num_polys_per_commitment_group(),
        routing.claim_to_commitment_group(),
        routing.claim_poly_in_commitment_group(),
        replay.row_coefficients,
        relation.num_public_rows(),
        relation.m_row_layout(),
        relation.opening_points().len(),
        relation.ring_multiplier_points(),
        relation.claim_to_point(),
        witness_segment_layout,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_ring_switch_row_eval_inner<F, E, const D: usize>(
    challenges: &Challenges,
    alpha: E,
    lp: &LevelParams,
    tau1: &[E],
    num_polys_per_commitment_group: &[usize],
    claim_to_commitment_group: &[usize],
    claim_poly_in_commitment_group: &[usize],
    gamma: &[E],
    num_public_rows: usize,
    m_row_layout: MRowLayout,
    opening_points_len: usize,
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    witness_segment_layout: RingRelationSegmentLayout,
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    validate_level_dispatch::<D>(lp)?;
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = claim_to_point.len();
    if claim_to_commitment_group.len() != num_claims
        || claim_poly_in_commitment_group.len() != num_claims
    {
        return Err(AkitaError::InvalidProof);
    }
    let num_points = num_polys_per_commitment_group.len();
    for claim_idx in 0..num_claims {
        let group_idx = claim_to_commitment_group[claim_idx];
        if group_idx >= num_points
            || claim_poly_in_commitment_group[claim_idx]
                >= num_polys_per_commitment_group[group_idx]
        {
            return Err(AkitaError::InvalidProof);
        }
    }

    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidSize {
            expected: num_claims,
            actual: gamma.len(),
        });
    }

    let log_basis = lp.log_basis;
    validate_log_basis(log_basis)?;
    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let depth_fold = lp.num_digits_fold(num_claims, F::modulus_bits())?;
    let num_blocks = lp.num_blocks;
    if num_blocks == 0 || !num_blocks.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "num_blocks must be a non-zero power of two".to_string(),
        ));
    }
    if lp.block_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "block_len must be non-zero".to_string(),
        ));
    }
    if depth_commit == 0 || depth_open == 0 || depth_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "digit depths must be non-zero".to_string(),
        ));
    }
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    let num_t_vectors = num_polys_per_commitment_group
        .iter()
        .try_fold(0usize, |acc, &count| acc.checked_add(count))
        .ok_or_else(|| AkitaError::InvalidSetup("batched t-vector count overflow".to_string()))?;
    #[cfg(feature = "zk")]
    let d_blinding_segment_len = match m_row_layout {
        MRowLayout::WithDBlock => zk::blinding_digit_plane_count::<F>(n_d, D, log_basis),
        MRowLayout::WithoutDBlock => 0,
    };
    #[cfg(feature = "zk")]
    let b_blinding_digit_planes_per_point = zk::blinding_digit_plane_count::<F>(n_b, D, log_basis);
    #[cfg(feature = "zk")]
    let b_blinding_segment_len = num_points
        .checked_mul(b_blinding_digit_planes_per_point)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK blinding width overflow".to_string()))?;
    // Must match [`RingSwitchDeferredRowEval::total_blocks`] on the prepared value.
    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.logical_len() != total_blocks {
        return Err(AkitaError::InvalidSize {
            expected: total_blocks,
            actual: challenges.logical_len(),
        });
    }
    let block_len = lp.block_len;
    let inner_width = block_len
        .checked_mul(depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".to_string()))?;
    if lp.a_key.col_len() < inner_width {
        return Err(AkitaError::InvalidSetup(
            "A-key column width is too small for verifier layout".to_string(),
        ));
    }
    let _expected_d_width = depth_open
        .checked_mul(num_blocks)
        .and_then(|width| width.checked_mul(num_claims))
        .ok_or_else(|| AkitaError::InvalidSetup("D-matrix width overflow".to_string()))?;
    // TODO: re-enable (or gate on schedule shape) once root-direct
    // commit params no longer carry zero-width D-key placeholders.
    // The planner emits `d_key.col_len = 0` for root-direct schedules
    // since the relation fold (which is what consumes D) doesn't run.
    // if lp.d_key.col_len() < expected_d_width {
    //     return Err(AkitaError::InvalidSetup(
    //         "D-key column width is too small for verifier layout".to_string(),
    //     ));
    // }
    let max_point_poly_count = num_polys_per_commitment_group
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let expected_b_width = max_point_poly_count
        .checked_mul(lp.a_key.row_len())
        .and_then(|width| width.checked_mul(depth_open))
        .and_then(|width| width.checked_mul(num_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("B-matrix width overflow".to_string()))?;
    if lp.b_key.col_len() < expected_b_width {
        return Err(AkitaError::InvalidSetup(
            "B-key column width is too small for verifier layout".to_string(),
        ));
    }
    if opening_points_len != num_points {
        return Err(AkitaError::InvalidProof);
    }
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= num_points)
    {
        return Err(AkitaError::InvalidProof);
    }
    if ring_multiplier_points.len() != opening_points_len {
        return Err(AkitaError::InvalidProof);
    }
    let rows = lp.m_row_count_for(num_points, num_public_rows, m_row_layout)?;

    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let c_alphas: PreparedChallengeEvals<E> = match challenges {
        Challenges::Sparse {
            challenges: sparse, ..
        } => PreparedChallengeEvals::Flat(
            sparse
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E, D>(&alpha_pows))
                .collect::<Result<_, _>>()?,
        ),
        Challenges::Tensor { factored } => {
            if D < 2 {
                return Err(AkitaError::InvalidInput(
                    "tensor challenge factored evaluation requires D >= 2".to_string(),
                ));
            }
            factored.validate::<D>()?;
            if factored.num_claims != num_claims {
                return Err(AkitaError::InvalidSize {
                    expected: num_claims,
                    actual: factored.num_claims,
                });
            }
            let blocks_per_claim = factored.blocks_per_claim()?;
            if blocks_per_claim != lp.num_blocks {
                return Err(AkitaError::InvalidSize {
                    expected: lp.num_blocks,
                    actual: blocks_per_claim,
                });
            }
            PreparedChallengeEvals::Tensor {
                challenges: factored.clone(),
                alpha_pows: alpha_pows.clone(),
            }
        }
    };

    let claim_to_commitment_group_poly: Vec<(usize, usize)> = claim_to_commitment_group
        .iter()
        .zip(claim_poly_in_commitment_group.iter())
        .map(|(&group_idx, &poly_idx)| (group_idx, poly_idx))
        .collect();

    Ok(RingSwitchDeferredRowEval {
        c_alphas,
        eq_tau1,
        num_t_vectors,
        num_blocks,
        num_claims,
        depth_open,
        depth_commit,
        depth_fold,
        #[cfg(feature = "zk")]
        d_blinding_segment_len,
        #[cfg(feature = "zk")]
        b_blinding_digit_planes_per_point,
        #[cfg(feature = "zk")]
        b_blinding_segment_len,
        block_len,
        inner_width,
        log_basis,
        n_a: lp.a_key.row_len(),
        n_d,
        m_row_layout,
        n_b,
        num_points,
        rows,
        claim_to_commitment_group_poly,
        num_polys_per_commitment_group: num_polys_per_commitment_group.to_vec(),
        num_public_rows,
        gamma: gamma.to_vec(),
        claim_to_point: claim_to_point.to_vec(),
        witness_segment_layout,
    })
}

impl<E: FieldCore> RingSwitchDeferredRowEval<E> {
    /// `num_blocks * num_claims` (W/D challenge logical length).
    ///
    /// Prepare validates the product with checked arithmetic before building
    /// this struct; replay uses the unchecked product on those same fields.
    #[inline(always)]
    pub(crate) fn total_blocks(&self) -> usize {
        self.num_blocks * self.num_claims
    }

    /// Number of active D rows in the selected M-row layout.
    pub(crate) fn n_d_active(&self) -> usize {
        match self.m_row_layout {
            MRowLayout::WithDBlock => self.n_d,
            MRowLayout::WithoutDBlock => 0,
        }
    }

    pub(crate) fn segment_layout(&self) -> Result<RingSwitchSegmentLayout, AkitaError> {
        Ok(self.witness_segment_layout)
    }

    pub(crate) fn create_setup_contribution_inputs(&self) -> SetupContributionPlanInputs<E> {
        SetupContributionPlanInputs {
            eq_tau1: self.eq_tau1.clone(),
            num_t_vectors: self.num_t_vectors,
            num_blocks: self.num_blocks,
            num_claims: self.num_claims,
            depth_open: self.depth_open,
            depth_commit: self.depth_commit,
            depth_fold: self.depth_fold,
            block_len: self.block_len,
            inner_width: self.inner_width,
            n_a: self.n_a,
            n_d: self.n_d,
            m_row_layout: self.m_row_layout,
            n_b: self.n_b,
            num_points: self.num_points,
            rows: self.rows,
            num_polys_per_commitment_group: self.num_polys_per_commitment_group.clone(),
            num_public_rows: self.num_public_rows,
        }
    }

    /// Evaluate the prepared ring-switch row table at the supplied point.
    ///
    /// # Errors
    ///
    /// Returns an error if the setup matrix cannot be viewed at `D` or an
    /// internal offset-eq evaluation receives inconsistent dimensions.
    #[inline]
    pub fn eval_at_point<F, const D: usize>(
        &self,
        x_challenges: &[E],
        setup: &AkitaExpandedSetup<F>,
        opening_points: &[RingOpeningPoint<F>],
        ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    {
        let _ring_bits = validate_ring_dispatch::<D>()?;
        if ring_multiplier_points.len() != opening_points.len() {
            return Err(AkitaError::InvalidProof);
        }
        // ----- Witness-layout offsets ----------------------------------------
        let layout = self.segment_layout()?;
        validate_log_basis(self.log_basis)?;
        if opening_points.len() != self.num_points {
            return Err(AkitaError::InvalidSize {
                expected: self.num_points,
                actual: opening_points.len(),
            });
        }
        for opening_point in opening_points {
            if opening_point.b.len() != self.num_blocks || opening_point.a.len() < self.block_len {
                return Err(AkitaError::InvalidProof);
            }
        }
        for point in ring_multiplier_points {
            if point.b_len() != self.num_blocks || point.a_len() < self.block_len {
                return Err(AkitaError::InvalidProof);
            }
        }

        // ----- Shared precomputes --------------------------------------------
        let alpha_pows = scalar_powers(alpha, D);
        let g1_open = gadget_row_scalars::<F>(self.depth_open, self.log_basis);
        let fold_gadget = gadget_row_scalars::<F>(self.depth_fold, self.log_basis);

        // Eq table over the low `log₂(num_blocks)` bits, shared by e-hat/T
        // peeled summaries and by `SetupEvaluator` direct mode.
        let offset_low_bits = self.num_blocks.trailing_zeros() as usize;
        if offset_low_bits > x_challenges.len() {
            return Err(AkitaError::InvalidSize {
                expected: offset_low_bits,
                actual: x_challenges.len(),
            });
        }
        let eq_low = EqPolynomial::evals(&x_challenges[..offset_low_bits])?;
        let block_offset_low = layout.offset_e & (self.num_blocks - 1);
        debug_assert_eq!(block_offset_low, layout.offset_t & (self.num_blocks - 1));

        // `z` peels `block_len` (not `num_blocks`) and uses its own
        // low-bit eq table.
        let z_offset_low_bits = self.block_len.trailing_zeros() as usize;
        if z_offset_low_bits > x_challenges.len() {
            return Err(AkitaError::InvalidSize {
                expected: z_offset_low_bits,
                actual: x_challenges.len(),
            });
        }
        let z_block_low_eq = EqPolynomial::evals(&x_challenges[..z_offset_low_bits])?;

        let high_challenges = &x_challenges[offset_low_bits..];

        let x_low_challenges = &x_challenges[..offset_low_bits];
        let total_blocks = self.total_blocks();
        if let Some(c_alphas) = self.c_alphas.as_flat() {
            if c_alphas.len() != total_blocks {
                return Err(AkitaError::InvalidSize {
                    expected: total_blocks,
                    actual: c_alphas.len(),
                });
            }
        }
        let challenge_block_summaries: Vec<[E; 2]> =
            self.c_alphas.summarize_all_block_carries::<F, D>(
                self.num_claims,
                x_low_challenges,
                &eq_low,
                block_offset_low,
                self.num_blocks,
            )?;
        let mut challenge_block_summaries_by_t_vector =
            vec![[E::zero(), E::zero()]; self.num_t_vectors];
        // Per-commitment-group t-vector starting indices. Under the current
        // relation-routing contract these match opening-point groups.
        let t_vector_offsets: Vec<usize> = self
            .num_polys_per_commitment_group
            .iter()
            .scan(0usize, |acc, &count| {
                let offset = *acc;
                *acc += count;
                Some(offset)
            })
            .collect();
        for (claim_idx, &(group_idx, poly_idx)) in
            self.claim_to_commitment_group_poly.iter().enumerate()
        {
            let t_vector_idx = t_vector_offsets[group_idx] + poly_idx;
            let [carry0, carry1] = challenge_block_summaries[claim_idx];
            challenge_block_summaries_by_t_vector[t_vector_idx][0] += carry0;
            challenge_block_summaries_by_t_vector[t_vector_idx][1] += carry1;
        }

        // ----- E-hat ---------------------------------------------------------
        let e_structured_contribution = {
            let _span = tracing::info_span!("e_structured").entered();
            let uses_ring_multipliers = ring_multiplier_points
                .iter()
                .any(|point| point.as_base().is_none());
            let row_coefficient_rings = if uses_ring_multipliers {
                Some(
                    self.gamma
                        .iter()
                        .copied()
                        .map(|coefficient| {
                            embed_ring_subfield_scalar::<F, E, D>(
                                coefficient,
                                AkitaError::InvalidProof,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            } else {
                None
            };
            let public_block_summaries: Vec<[E; 2]> = (0..self.num_claims)
                .map(|claim_idx| {
                    let point_idx = self.claim_to_point[claim_idx];
                    if point_idx >= ring_multiplier_points.len() {
                        return Err(AkitaError::InvalidProof);
                    }
                    let point = &ring_multiplier_points[point_idx];
                    let coefficient_ring = row_coefficient_rings
                        .as_ref()
                        .map(|rings| &rings[claim_idx]);
                    summarize_pow2_multiplier_block_carries(
                        &eq_low,
                        block_offset_low,
                        point.b_len(),
                        |idx| {
                            point.eval_b_with_coefficient(
                                idx,
                                self.gamma[claim_idx],
                                coefficient_ring,
                                &alpha_pows,
                            )
                        },
                    )
                })
                .collect::<Result<_, _>>()?;
            let public_row_weights_by_claim: Vec<E> = self
                .claim_to_point
                .iter()
                .map(|&point_idx| {
                    point_idx
                        .checked_add(1)
                        .and_then(|idx| self.eq_tau1.get(idx))
                        .copied()
                        .ok_or(AkitaError::InvalidProof)
                })
                .collect::<Result<_, _>>()?;
            EStructuredSlicesEvaluator {
                high_challenges,
                offset_high: layout.offset_e >> offset_low_bits,
                gadget_vector: &g1_open,
                public_block_summaries: &public_block_summaries,
                challenge_block_summaries: &challenge_block_summaries,
                public_row_weights_by_claim: &public_row_weights_by_claim,
                challenge_weight: self.eq_tau1[0],
            }
            .evaluate()
        };

        // ----- T -------------------------------------------------------------
        let t_structured_contribution = {
            let _span = tracing::info_span!("t_structured").entered();
            let a_start = 1 + self.num_public_rows + self.n_d_active() + self.n_b * self.num_points;
            TStructuredSlicesEvaluator {
                high_challenges,
                offset_high: layout.offset_t >> offset_low_bits,
                gadget_vector: &g1_open,
                challenge_block_summaries: &challenge_block_summaries_by_t_vector,
                a_row_weights: &self.eq_tau1[a_start..self.rows],
            }
            .evaluate()
        };

        // ----- Fused D·ŵ + B·t̂ + A·ẑ ---------------------------------------
        let setup_contribution = {
            let _span = tracing::info_span!("setup_contribution").entered();
            jolt_start_cycle_tracking("setup_contribution");
            let result = if let Some(claim) = setup_claim {
                Ok(claim)
            } else {
                let setup_contribution_inputs = self.create_setup_contribution_inputs();
                let evaluator = SetupEvaluator::new(
                    &setup_contribution_inputs,
                    x_challenges,
                    Some(&eq_low),
                    Some(&z_block_low_eq),
                    &alpha_pows,
                    &fold_gadget,
                    layout.offset_e,
                    layout.offset_t,
                    layout.offset_z,
                );
                match evaluator.evaluate::<D>(SetupEvaluatorMode::Direct { setup })? {
                    SetupEvaluation::Direct(value) => Ok(value),
                    #[cfg(test)]
                    SetupEvaluation::Recursive(_) => Err(AkitaError::InvalidSetup(
                        "setup evaluator returned recursive output for direct mode".into(),
                    )),
                }
            };
            jolt_end_cycle_tracking("setup_contribution");
            result?
        };

        // ----- Z (consistency-row) ------------------------------------------
        let z_structured_contribution = {
            let _span = tracing::info_span!("z_structured").entered();
            let g1_commit = gadget_row_scalars::<F>(self.depth_commit, self.log_basis);
            if self.block_len.is_power_of_two() {
                let z_offset_low = layout.offset_z & (self.block_len - 1);
                let a_block_summary: Vec<[E; 2]> = ring_multiplier_points
                    .iter()
                    .map(|ring_multiplier_point| {
                        summarize_pow2_multiplier_block_carries(
                            &z_block_low_eq,
                            z_offset_low,
                            self.block_len,
                            |idx| ring_multiplier_point.eval_a_at::<E>(idx, &alpha_pows),
                        )
                    })
                    .collect::<Result<_, _>>()?;
                ZStructuredPow2SlicesEvaluator {
                    high_challenges: &x_challenges[z_offset_low_bits..],
                    offset_high: layout.offset_z >> z_offset_low_bits,
                    g1_commit: &g1_commit,
                    fold_gadget: &fold_gadget,
                    a_block_summary: &a_block_summary,
                    consistency_weight: self.eq_tau1[0],
                }
                .evaluate()
            } else {
                let a_evals_by_point: Vec<Vec<E>> = ring_multiplier_points
                    .iter()
                    .map(|ring_multiplier_point| {
                        (0..self.block_len)
                            .map(|idx| ring_multiplier_point.eval_a_at::<E>(idx, &alpha_pows))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<_, AkitaError>>()?;
                ZDenseSlicesEvaluator {
                    g1_commit: &g1_commit,
                    fold_gadget: &fold_gadget,
                    consistency_weight: self.eq_tau1[0],
                    a_evals_by_point: &a_evals_by_point,
                    full_vec_randomness: x_challenges,
                    offset_z: layout.offset_z,
                    block_len: self.block_len,
                }
                .evaluate()?
            }
        };

        // ----- r-tail --------------------------------------------------------
        let r_contribution = {
            let r_gadget =
                gadget_row_scalars::<F>(r_decomp_levels::<F>(self.log_basis), self.log_basis);
            let denom = alpha_pows[D - 1] * alpha + E::one();
            compute_r_contribution(self, x_challenges, layout.offset_r, denom, &r_gadget)?
        };

        #[allow(unused_mut)]
        let mut total = e_structured_contribution
            + t_structured_contribution
            + z_structured_contribution
            + setup_contribution
            + r_contribution;

        #[cfg(feature = "zk")]
        {
            let b_blinding = compute_b_blinding_part::<F, E, D>(self, x_challenges, setup, alpha)?;
            let d_blinding = compute_d_blinding_part::<F, E, D>(self, x_challenges, setup, alpha)?;
            total = total + b_blinding + d_blinding;
        }

        Ok(total)
    }
}

#[inline]
fn summarize_pow2_multiplier_block_carries<E, EvalAt>(
    eq_low: &[E],
    offset_low: usize,
    values_len: usize,
    mut eval_at: EvalAt,
) -> Result<[E; 2], AkitaError>
where
    E: FieldCore,
    EvalAt: FnMut(usize) -> Result<E, AkitaError>,
{
    if !values_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "peeled inner block length must be a power of two".to_string(),
        ));
    }
    if eq_low.len() != values_len {
        return Err(AkitaError::InvalidSize {
            expected: values_len,
            actual: eq_low.len(),
        });
    }
    if offset_low >= values_len {
        return Err(AkitaError::InvalidInput(
            "low offset must lie inside the peeled block".to_string(),
        ));
    }

    let inner_bits = values_len.trailing_zeros() as usize;
    let inner_mask = values_len - 1;
    let mut out = [E::zero(), E::zero()];

    for u in 0..values_len {
        let sum = offset_low + u;
        let carry = sum >> inner_bits;
        debug_assert!(
            carry < 2,
            "sum of two peeled indices must carry at most one bit"
        );
        let low_idx = sum & inner_mask;
        out[carry] += eq_low[low_idx] * eval_at(u)?;
    }

    Ok(out)
}

#[cfg(test)]
#[inline]
pub(crate) fn summarize_pow2_block_carries_base<F, E>(
    eq_low: &[E],
    offset_low: usize,
    values: &[F],
) -> Result<[E; 2], AkitaError>
where
    F: FieldCore,
    E: akita_field::ExtField<F>,
{
    if !values.len().is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "peeled inner block length must be a power of two".to_string(),
        ));
    }
    if eq_low.len() != values.len() {
        return Err(AkitaError::InvalidSize {
            expected: values.len(),
            actual: eq_low.len(),
        });
    }
    if offset_low >= values.len() {
        return Err(AkitaError::InvalidInput(
            "low offset must lie inside the peeled block".to_string(),
        ));
    }

    let inner_bits = values.len().trailing_zeros() as usize;
    let inner_mask = values.len() - 1;
    let mut out = [E::zero(), E::zero()];

    for (u, &value) in values.iter().enumerate() {
        let sum = offset_low + u;
        let carry = sum >> inner_bits;
        debug_assert!(
            carry < 2,
            "sum of two peeled indices must carry at most one bit"
        );
        let low_idx = sum & inner_mask;
        out[carry] += eq_low[low_idx].mul_base(value);
    }

    Ok(out)
}
