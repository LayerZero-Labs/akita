//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use akita_challenges::Challenges;
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
use akita_types::{
    validate_opening_points_for_claims, FlatRingVec, LevelParams, MRowLayout,
    RingMultiplierOpeningPoint, RingRelationInstance, RingRelationSegmentLayout,
    RingSubfieldEncoding, TerminalWitnessTranscriptParts,
};

use super::{validate_level_dispatch, validate_log_basis, validate_ring_dispatch};
#[cfg(test)]
pub(crate) use row_eval::summarize_pow2_block_carries_base;
#[cfg(test)]
use row_eval::summarize_pow2_multiplier_block_carries;
pub(crate) use tensor_challenges::PreparedChallengeEvals;

mod row_eval;
mod tensor_challenges;
#[cfg(test)]
mod tests;

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
    /// Tiered split factor `f` (`1` = single-tier).
    pub(crate) tier_split: usize,
    /// Second-tier `F` rank (`0` = single-tier); the sent-commitment length.
    pub(crate) n_f: usize,
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
    transcript.absorb_and_record_serde(ABSORB_NEXT_LEVEL_WITNESS_BINDING, w_commitment);
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
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_parts.remainder);
    ring_switch_verifier_core::<F, E, T, D>(replay, w_len, transcript, MRowLayout::WithoutDBlock)?
        .into_terminal_as_output()
}

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
    // Tiered: the stored first-tier `B'` is the full B width divided by the
    // reuse factor `tier_split`.
    let expected_stored_b_width = if lp.f_key.is_some() {
        expected_b_width.div_ceil(lp.tier_split.max(1))
    } else {
        expected_b_width
    };
    if lp.b_key.col_len() < expected_stored_b_width {
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
        tier_split: lp.tier_split,
        n_f: lp.f_key.as_ref().map_or(0, |fk| fk.row_len()),
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
