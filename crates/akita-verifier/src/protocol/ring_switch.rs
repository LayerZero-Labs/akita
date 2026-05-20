//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::summarize_pow2_block_carries;
use akita_algebra::ring::scalar_powers;
use akita_challenges::{TensorChallengeSet, TensorChallenges};
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, RandomSampling,
};
use akita_transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use akita_transcript::{sample_ext_challenge, Transcript};
#[cfg(feature = "zk")]
use akita_types::zk;
use akita_types::{
    embed_ring_subfield_scalar, gadget_row_scalars, r_decomp_levels,
    validate_opening_points_for_claims, AkitaExpandedSetup, FlatRingVec, LevelParams,
    RingMultiplierOpeningPoint, RingOpeningPoint, RingSubfieldEncoding,
};

#[cfg(feature = "zk")]
use super::slice_mle::{compute_b_blinding_part, compute_d_blinding_part};
use super::slice_mle::{
    compute_r_contribution, compute_setup_contribution, StructuredSliceMleEvaluator,
    TStructuredSlicesEvaluator, WStructuredSlicesEvaluator, ZDenseSlicesEvaluator,
    ZStructuredPow2SlicesEvaluator,
};
use super::{validate_level_dispatch, validate_log_basis, validate_ring_dispatch};

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

/// Carrier for the per-block challenge evaluations consumed by
/// [`RingSwitchDeferredRowEval`]'s deferred row-MLE replay.
///
/// `Flat` stores the materialised dense per-block evaluations exactly as
/// `TensorChallenges::Flat::evals_at_pows` would produce them: one entry per
/// `(claim, block)`, packed claim-major. `Tensor` keeps the factored
/// `(left, right)` challenge set together with the cached
/// `alpha_pows` / `alpha_pow_d_plus_one` so consumers can summarise carry
/// terms via the factored aggregate without ever materialising a length-
/// `num_claims * num_blocks` vector.
pub(crate) enum PreparedChallengeEvals<F: FieldCore> {
    /// Dense per-block evaluations packed as `[claim_0_block_0, ..., claim_0_block_{B-1}, claim_1_block_0, ...]`.
    Flat(Vec<F>),
    /// Factored tensor challenges plus the cached `alpha` powers needed for
    /// `eval_factored_aggregate_at_pows`. `alpha_pow_d_plus_one = α^D + 1`,
    /// pre-derived from `alpha_pows` because the consumer reuses it on every
    /// `(carry_q, final_carry)` combination.
    Tensor {
        challenges: TensorChallengeSet,
        alpha_pows: Vec<F>,
        alpha_pow_d_plus_one: F,
    },
}

impl<F: FieldCore> PreparedChallengeEvals<F> {
    /// Return the materialised flat per-block evaluations, or `None` if the
    /// challenges are stored in factored tensor form. Test-only helper used
    /// by the `slice_mle` regression tests that compare against an expanded
    /// reference; production code consumes through
    /// [`Self::summarize_all_block_carries`] instead.
    #[cfg(test)]
    pub(crate) fn as_flat(&self) -> Option<&[F]> {
        match self {
            Self::Flat(c_alphas) => Some(c_alphas),
            Self::Tensor { .. } => None,
        }
    }

    /// Summarise `(carry=0, carry=1)` block-carry pairs for every claim.
    ///
    /// Returns `Vec<[F; 2]>` of length `num_claims`. The shape matches what
    /// `(0..num_claims).map(|claim| summarize_pow2_block_carries(eq_low,
    /// offset_low, &flat_c_alphas[claim * num_blocks .. (claim + 1) *
    /// num_blocks]))` would produce, but the `Tensor` variant computes each
    /// `[F; 2]` in `O(√num_blocks · D)` work via
    /// `TensorChallengeSet::eval_factored_aggregate_at_pows` instead of
    /// touching every block index.
    ///
    /// `x_low_challenges` are the raw low-bit challenge scalars that produce
    /// `eq_low = EqPolynomial::evals(x_low_challenges)`. The `Flat` arm only
    /// consults `eq_low`; the `Tensor` arm splits `x_low_challenges` into the
    /// left/right halves matching the tensor factor sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_claims` is out of range for the underlying
    /// challenges, if `offset_low >= num_blocks`, or if the factored
    /// aggregate rejects the supplied weights/powers.
    pub(crate) fn summarize_all_block_carries<const D: usize>(
        &self,
        num_claims: usize,
        x_low_challenges: &[F],
        eq_low: &[F],
        offset_low: usize,
        num_blocks: usize,
    ) -> Result<Vec<[F; 2]>, AkitaError>
    where
        F: FromPrimitiveInt + MulBase<F>,
    {
        match self {
            Self::Flat(c_alphas) => (0..num_claims)
                .map(|claim_idx| {
                    let start = claim_idx.checked_mul(num_blocks).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "flat challenge summary offset overflow".to_string(),
                        )
                    })?;
                    let end = start.checked_add(num_blocks).ok_or_else(|| {
                        AkitaError::InvalidSetup("flat challenge summary end overflow".to_string())
                    })?;
                    let values = c_alphas.get(start..end).ok_or(AkitaError::InvalidSize {
                        expected: end,
                        actual: c_alphas.len(),
                    })?;
                    summarize_pow2_block_carries(eq_low, offset_low, values)
                })
                .collect(),
            Self::Tensor {
                challenges,
                alpha_pows,
                alpha_pow_d_plus_one,
            } => summarize_tensor_all_block_carries::<F, D>(
                challenges,
                num_claims,
                x_low_challenges,
                offset_low,
                num_blocks,
                alpha_pows,
                *alpha_pow_d_plus_one,
            ),
        }
    }
}

/// Factored-aggregate replacement for the `(0..num_claims)`-loop of
/// `summarize_pow2_block_carries` calls.
///
/// Each block index `u ∈ [0, num_blocks)` decomposes as
/// `u = p * right_len + q`. The weights `eq_low[(offset_low + u) mod
/// num_blocks]` factor across `(p, q)` via the two half-eq tables `eq_left`,
/// `eq_right` derived from the high and low halves of `x_low_challenges`.
/// For every `(carry_q ∈ {0, 1}, final_carry ∈ {0, 1})` combination we build
/// the matching `(u_weights, v_weights)` pair, call
/// `eval_factored_aggregate_at_pows` once per claim, and accumulate into the
/// per-claim `[carry_0, carry_1]` output.
///
/// Total work is `O((left_len + right_len) · D · num_claims)` versus the flat
/// path's `O(num_blocks · num_claims)` — the win grows with `num_blocks`.
#[allow(clippy::too_many_arguments)]
fn summarize_tensor_all_block_carries<F, const D: usize>(
    challenges: &TensorChallengeSet,
    num_claims: usize,
    x_low_challenges: &[F],
    offset_low: usize,
    num_blocks: usize,
    alpha_pows: &[F],
    alpha_pow_d_plus_one: F,
) -> Result<Vec<[F; 2]>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt + MulBase<F>,
{
    if num_claims > challenges.num_claims {
        return Err(AkitaError::InvalidSize {
            expected: challenges.num_claims,
            actual: num_claims,
        });
    }
    if challenges.left_len.checked_mul(challenges.right_len) != Some(num_blocks) {
        return Err(AkitaError::InvalidSize {
            expected: num_blocks,
            actual: challenges.left_len.saturating_mul(challenges.right_len),
        });
    }
    if !challenges.left_len.is_power_of_two() || !challenges.right_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "tensor challenge dimensions must be powers of two".to_string(),
        ));
    }
    if offset_low >= num_blocks {
        return Err(AkitaError::InvalidInput(format!(
            "low offset {offset_low} out of range for {num_blocks} blocks"
        )));
    }

    let right_bits = challenges.right_len.trailing_zeros() as usize;
    let left_bits = challenges.left_len.trailing_zeros() as usize;
    if x_low_challenges.len() != right_bits + left_bits {
        return Err(AkitaError::InvalidSize {
            expected: right_bits + left_bits,
            actual: x_low_challenges.len(),
        });
    }

    let eq_right = EqPolynomial::evals(&x_low_challenges[..right_bits])?;
    let eq_left = EqPolynomial::evals(&x_low_challenges[right_bits..])?;
    let right_mask = challenges.right_len - 1;
    let left_mask = challenges.left_len - 1;
    let offset_right = offset_low & right_mask;
    let offset_left = offset_low >> right_bits;

    let mut out = vec![[F::zero(), F::zero()]; num_claims];
    let mut v_weights = vec![F::zero(); challenges.right_len];
    let mut u_weights = vec![F::zero(); challenges.left_len];
    for carry_q in 0..=1 {
        v_weights.fill(F::zero());
        let mut has_v_weight = false;
        for (q, v_weight) in v_weights.iter_mut().enumerate() {
            let shifted = offset_right + q;
            if (shifted >> right_bits) == carry_q {
                *v_weight = eq_right[shifted & right_mask];
                has_v_weight |= !v_weight.is_zero();
            }
        }
        if !has_v_weight {
            continue;
        }

        for final_carry in 0..=1 {
            u_weights.fill(F::zero());
            let mut has_u_weight = false;
            for (p, u_weight) in u_weights.iter_mut().enumerate() {
                let shifted = offset_left + p + carry_q;
                if (shifted >> left_bits) == final_carry {
                    *u_weight = eq_left[shifted & left_mask];
                    has_u_weight |= !u_weight.is_zero();
                }
            }
            if !has_u_weight {
                continue;
            }
            for (claim_idx, out_terms) in out.iter_mut().enumerate() {
                out_terms[final_carry] += challenges.eval_factored_aggregate_at_pows::<F, F, D>(
                    claim_idx,
                    &u_weights,
                    &v_weights,
                    alpha_pows,
                    alpha_pow_d_plus_one,
                )?;
            }
        }
    }

    Ok(out)
}

/// Precomputed challenge-derived data for deferred ring-switch row MLE evaluation.
///
/// Stores only data that cannot be derived from context at evaluation time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else is passed by reference at evaluation time to avoid
/// duplicating setup matrix views, opening points, and gadget vectors.
pub struct RingSwitchDeferredRowEval<F: FieldCore> {
    pub(crate) c_alphas: PreparedChallengeEvals<F>,
    pub(crate) eq_tau1: Vec<F>,
    pub(crate) total_blocks: usize,
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
    pub(crate) n_b: usize,
    pub(crate) num_points: usize,
    pub(crate) rows: usize,
    pub(crate) z_first: bool,
    pub(crate) claim_to_point_poly: Vec<(usize, usize)>,
    pub(crate) num_polys_per_point: Vec<usize>,
    pub(crate) num_public_rows: usize,
    pub(crate) gamma: Vec<F>,
    pub(crate) claim_to_point: Vec<usize>,
}

pub(crate) struct RingSwitchSegmentLayout {
    #[cfg(feature = "zk")]
    pub(crate) w_len: usize,
    pub(crate) offset_w: usize,
    pub(crate) offset_t: usize,
    pub(crate) offset_z: usize,
    pub(crate) offset_r: usize,
    #[cfg(feature = "zk")]
    pub(crate) b_blinding_offset: usize,
    #[cfg(feature = "zk")]
    pub(crate) d_blinding_offset: usize,
}

/// Replay the verifier half of ring switching.
///
/// This handles multiple opening points, arbitrary claim-to-point mapping, and
/// arbitrary commitment grouping. The recursive/single-point path is the
/// `opening_points = [pt]`, `claim_to_point = [0]`,
/// `num_polys_per_point = [1]`, `num_public_rows = 1` specialization.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or ring-switch row-eval
/// preparation fails.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub(crate) fn ring_switch_verifier<F, E, T, const D: usize>(
    opening_points: &[RingOpeningPoint<F>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    challenges: &TensorChallenges,
    w_len: usize,
    w_commitment: &FlatRingVec<F>,
    transcript: &mut T,
    lp: &LevelParams,
    num_polys_per_point: &[usize],
    claim_to_point_poly: &[usize],
    claim_poly_indices: &[usize],
    gamma: &[E],
    num_public_rows: usize,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    let ring_bits = validate_ring_dispatch::<D>()?;
    transcript.append_serde(ABSORB_SUMCHECK_W, w_commitment);

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
    if claim_to_point_poly.len() != num_claims || claim_poly_indices.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let num_points = num_polys_per_point.len();
    for claim_idx in 0..num_claims {
        let point_idx = claim_to_point_poly[claim_idx];
        if point_idx >= num_points
            || claim_poly_indices[claim_idx] >= num_polys_per_point[point_idx]
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
    let m_rows = lp.m_row_count(num_points, num_public_rows)?;
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch row count overflow".to_string()))?
        .trailing_zeros() as usize;

    let tau0: Vec<E> = (0..num_sc_vars)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
        .collect();
    let tau1: Vec<E> = (0..num_i)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
        .collect();
    let alpha_evals_y = scalar_powers(alpha, D);
    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let prepared_row_eval = prepare_ring_switch_row_eval::<F, E, D>(
        challenges,
        alpha,
        lp,
        &tau1,
        num_polys_per_point,
        claim_to_point_poly,
        claim_poly_indices,
        gamma,
        num_public_rows,
        opening_points.len(),
        ring_multiplier_points,
        claim_to_point,
    )?;

    Ok(RingSwitchVerifyOutput {
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

/// Prepare deferred verifier ring-switch row evaluation data.
///
/// # Errors
///
/// Returns an error if gamma/challenge lengths do not match the claim shape,
/// the expanded tau1 table is too short for the level layout, or sparse
/// challenge evaluation fails.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "prepare_ring_switch_row_eval")]
pub fn prepare_ring_switch_row_eval<F, E, const D: usize>(
    challenges: &TensorChallenges,
    alpha: E,
    lp: &LevelParams,
    tau1: &[E],
    num_polys_per_point: &[usize],
    claim_to_point_poly: &[usize],
    claim_poly_indices: &[usize],
    gamma: &[E],
    num_public_rows: usize,
    opening_points_len: usize,
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    validate_level_dispatch::<D>(lp)?;
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = claim_to_point.len();
    if claim_to_point_poly.len() != num_claims || claim_poly_indices.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let num_points = num_polys_per_point.len();
    for claim_idx in 0..num_claims {
        let point_idx = claim_to_point_poly[claim_idx];
        if point_idx >= num_points
            || claim_poly_indices[claim_idx] >= num_polys_per_point[point_idx]
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

    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let depth_fold = lp.num_digits_fold;
    let log_basis = lp.log_basis;
    let num_blocks = lp.num_blocks;
    validate_log_basis(log_basis)?;
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
    let num_t_vectors = num_polys_per_point
        .iter()
        .try_fold(0usize, |acc, &count| acc.checked_add(count))
        .ok_or_else(|| AkitaError::InvalidSetup("batched t-vector count overflow".to_string()))?;
    #[cfg(feature = "zk")]
    let d_blinding_segment_len = zk::blinding_digit_plane_count::<F>(n_d, D, log_basis);
    #[cfg(feature = "zk")]
    let b_blinding_digit_planes_per_point = zk::blinding_digit_plane_count::<F>(n_b, D, log_basis);
    #[cfg(feature = "zk")]
    let b_blinding_segment_len = num_points
        .checked_mul(b_blinding_digit_planes_per_point)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK blinding width overflow".to_string()))?;
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
    let expected_d_width = depth_open
        .checked_mul(num_blocks)
        .and_then(|width| width.checked_mul(num_claims))
        .ok_or_else(|| AkitaError::InvalidSetup("D-matrix width overflow".to_string()))?;
    if lp.d_key.col_len() < expected_d_width {
        return Err(AkitaError::InvalidSetup(
            "D-key column width is too small for verifier layout".to_string(),
        ));
    }
    let max_point_poly_count = num_polys_per_point.iter().copied().max().unwrap_or(0);
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
    let rows = lp.m_row_count(num_points, num_public_rows)?;

    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    // Tensor challenges keep their factored form so the verifier's per-claim
    // carry summary can stay in `O(√num_blocks · D)` rather than expanding
    // back to a `num_claims * num_blocks` materialised vector.
    let c_alphas: PreparedChallengeEvals<E> = match challenges {
        TensorChallenges::Flat(_) => {
            PreparedChallengeEvals::Flat(challenges.evals_at_pows::<F, E, D>(&alpha_pows)?)
        }
        TensorChallenges::Tensor(set) => {
            if D < 2 {
                return Err(AkitaError::InvalidInput(
                    "tensor challenge factored evaluation requires D >= 2".to_string(),
                ));
            }
            let alpha_pow_d_plus_one = alpha_pows[D - 1] * alpha_pows[1] + E::one();
            PreparedChallengeEvals::Tensor {
                challenges: set.clone(),
                alpha_pows: alpha_pows.clone(),
                alpha_pow_d_plus_one,
            }
        }
    };

    let z_first = lp.m_vars >= lp.r_vars;

    let claim_to_point_poly: Vec<(usize, usize)> = claim_to_point_poly
        .iter()
        .zip(claim_poly_indices.iter())
        .map(|(&point_idx, &poly_idx)| (point_idx, poly_idx))
        .collect();

    Ok(RingSwitchDeferredRowEval {
        c_alphas,
        eq_tau1,
        total_blocks,
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
        n_b,
        num_points,
        rows,
        z_first,
        claim_to_point_poly,
        num_polys_per_point: num_polys_per_point.to_vec(),
        num_public_rows,
        gamma: gamma.to_vec(),
        claim_to_point: claim_to_point.to_vec(),
    })
}

impl<E: FieldCore> RingSwitchDeferredRowEval<E> {
    pub(crate) fn segment_layout(&self) -> Result<RingSwitchSegmentLayout, AkitaError> {
        if self.num_blocks == 0 || !self.num_blocks.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "num_blocks must be a non-zero power of two".to_string(),
            ));
        }
        if self.block_len == 0
            || self.depth_open == 0
            || self.depth_commit == 0
            || self.depth_fold == 0
        {
            return Err(AkitaError::InvalidSetup(
                "prepared ring-switch layout has zero width".to_string(),
            ));
        }

        let w_len = self
            .depth_open
            .checked_mul(self.total_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("W segment length overflow".to_string()))?;
        let t_total_blocks = self
            .num_blocks
            .checked_mul(self.num_t_vectors)
            .ok_or_else(|| AkitaError::InvalidSetup("T block count overflow".to_string()))?;
        let t_len = self
            .depth_open
            .checked_mul(self.n_a)
            .and_then(|len| len.checked_mul(t_total_blocks))
            .ok_or_else(|| AkitaError::InvalidSetup("T segment length overflow".to_string()))?;
        let z_len = self
            .depth_fold
            .checked_mul(self.depth_commit)
            .and_then(|len| len.checked_mul(self.num_points))
            .and_then(|len| len.checked_mul(self.block_len))
            .ok_or_else(|| AkitaError::InvalidSetup("Z segment length overflow".to_string()))?;
        #[cfg(feature = "zk")]
        let b_blinding_segment_len = self.b_blinding_segment_len;
        #[cfg(not(feature = "zk"))]
        let b_blinding_segment_len = 0usize;
        #[cfg(feature = "zk")]
        let d_blinding_segment_len = self.d_blinding_segment_len;
        #[cfg(not(feature = "zk"))]
        let d_blinding_segment_len = 0usize;

        let offset_z = if self.z_first {
            0
        } else {
            w_len
                .checked_add(t_len)
                .and_then(|offset| offset.checked_add(b_blinding_segment_len))
                .and_then(|offset| offset.checked_add(d_blinding_segment_len))
                .ok_or_else(|| AkitaError::InvalidSetup("Z offset overflow".to_string()))?
        };
        let offset_w = if self.z_first { z_len } else { 0 };
        let offset_t = if self.z_first {
            z_len
                .checked_add(w_len)
                .ok_or_else(|| AkitaError::InvalidSetup("T offset overflow".to_string()))?
        } else {
            w_len
        };
        let b_blinding_offset = offset_t
            .checked_add(t_len)
            .ok_or_else(|| AkitaError::InvalidSetup("B blinding offset overflow".to_string()))?;
        let d_blinding_offset = b_blinding_offset
            .checked_add(b_blinding_segment_len)
            .ok_or_else(|| AkitaError::InvalidSetup("D blinding offset overflow".to_string()))?;
        let offset_r_base = d_blinding_offset
            .checked_add(d_blinding_segment_len)
            .ok_or_else(|| AkitaError::InvalidSetup("r-tail offset overflow".to_string()))?;
        let offset_r = if self.z_first {
            offset_r_base
        } else {
            offset_r_base
                .checked_add(z_len)
                .ok_or_else(|| AkitaError::InvalidSetup("r-tail offset overflow".to_string()))?
        };

        Ok(RingSwitchSegmentLayout {
            #[cfg(feature = "zk")]
            w_len,
            offset_w,
            offset_t,
            offset_z,
            offset_r,
            #[cfg(feature = "zk")]
            b_blinding_offset,
            #[cfg(feature = "zk")]
            d_blinding_offset,
        })
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

        // Eq table over the low `log₂(num_blocks)` bits, shared by W/T
        // peeled summaries and by `compute_setup_contribution`.
        let offset_low_bits = self.num_blocks.trailing_zeros() as usize;
        if offset_low_bits > x_challenges.len() {
            return Err(AkitaError::InvalidSize {
                expected: offset_low_bits,
                actual: x_challenges.len(),
            });
        }
        let eq_low = EqPolynomial::evals(&x_challenges[..offset_low_bits])?;
        let block_offset_low = layout.offset_w & (self.num_blocks - 1);
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

        // Per-claim `c_alpha` carry summary — shared by W and T. For tensor
        // challenges this dispatches into `summarize_tensor_all_block_carries`
        // and never materialises a length-`num_claims * num_blocks` vector;
        // for flat challenges the behaviour matches the pre-PR4 loop.
        let x_low_challenges = &x_challenges[..offset_low_bits];
        let challenge_block_summaries: Vec<[E; 2]> =
            self.c_alphas.summarize_all_block_carries::<D>(
                self.num_claims,
                x_low_challenges,
                &eq_low,
                block_offset_low,
                self.num_blocks,
            )?;
        let mut challenge_block_summaries_by_t_vector =
            vec![[E::zero(), E::zero()]; self.num_t_vectors];
        // Per-point t-vector starting indices: `t_vector_offsets[p]` is the
        // running sum of bundle sizes for points `< p`. Lifted out of the
        // per-claim loop so the routing is O(num_points + num_claims) rather
        // than O(num_points * num_claims).
        let t_vector_offsets: Vec<usize> = self
            .num_polys_per_point
            .iter()
            .scan(0usize, |acc, &count| {
                let offset = *acc;
                *acc += count;
                Some(offset)
            })
            .collect();
        for (claim_idx, &(point_idx, poly_idx)) in self.claim_to_point_poly.iter().enumerate() {
            let t_vector_idx = t_vector_offsets[point_idx] + poly_idx;
            let [carry0, carry1] = challenge_block_summaries[claim_idx];
            challenge_block_summaries_by_t_vector[t_vector_idx][0] += carry0;
            challenge_block_summaries_by_t_vector[t_vector_idx][1] += carry1;
        }

        // ----- W -------------------------------------------------------------
        let w_structured_contribution = {
            let _span = tracing::info_span!("w_structured").entered();
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
            WStructuredSlicesEvaluator {
                high_challenges,
                offset_high: layout.offset_w >> offset_low_bits,
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
            let a_start = 1 + self.num_public_rows + self.n_d + self.n_b * self.num_points;
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
            compute_setup_contribution::<F, E, D>(
                self,
                x_challenges,
                setup,
                &eq_low,
                &z_block_low_eq,
                &alpha_pows,
                &fold_gadget,
                layout.offset_w,
                layout.offset_t,
                layout.offset_z,
            )?
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
        let mut total = w_structured_contribution
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

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp32;
    use akita_types::SisModulusFamily;

    type F = Fp32<251>;
    const D: usize = 32;

    fn stage1_config() -> SparseChallengeConfig {
        SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![1],
        }
    }

    fn empty_flat_challenges() -> TensorChallenges {
        TensorChallenges::Flat(Vec::new())
    }

    #[test]
    fn ring_switch_prepare_rejects_invalid_log_basis() {
        let lp = LevelParams::params_only(SisModulusFamily::Q32, D, 0, 1, 1, 1, stage1_config());
        let challenges = empty_flat_challenges();
        let err = match prepare_ring_switch_row_eval::<F, F, D>(
            &challenges,
            F::one(),
            &lp,
            &[],
            &[],
            &[],
            &[],
            &[],
            1,
            0,
            &[],
            &[],
        ) {
            Ok(_) => panic!("invalid log_basis should be rejected"),
            Err(err) => err,
        };
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn ring_switch_prepare_rejects_zero_num_blocks() {
        let lp = LevelParams::params_only(SisModulusFamily::Q32, D, 2, 1, 1, 1, stage1_config());
        let challenges = empty_flat_challenges();
        let err = match prepare_ring_switch_row_eval::<F, F, D>(
            &challenges,
            F::one(),
            &lp,
            &[],
            &[],
            &[],
            &[],
            &[],
            1,
            0,
            &[],
            &[],
        ) {
            Ok(_) => panic!("zero num_blocks should be rejected"),
            Err(err) => err,
        };
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn multiplier_block_summary_rejects_malformed_shapes() {
        let eq_low = vec![F::one(); 2];

        let err =
            summarize_pow2_multiplier_block_carries(&eq_low, 0, 3, |_| Ok(F::one())).unwrap_err();
        assert!(matches!(err, AkitaError::InvalidInput(_)));

        let err =
            summarize_pow2_multiplier_block_carries(&eq_low, 2, 2, |_| Ok(F::one())).unwrap_err();
        assert!(matches!(err, AkitaError::InvalidInput(_)));

        let err = summarize_pow2_multiplier_block_carries(&eq_low[..1], 0, 2, |_| Ok(F::one()))
            .unwrap_err();
        assert!(matches!(err, AkitaError::InvalidSize { .. }));
    }

    /// Build a small `TensorChallenges::Tensor` value and check that the
    /// factored-aggregate carry summary (`PreparedChallengeEvals::Tensor`)
    /// agrees with the materialised flat summary (`PreparedChallengeEvals::Flat`)
    /// at every legal `(offset_low, x_low)` combination. Guards the
    /// equivalence invariant the perf refactor depends on so future kernel
    /// changes that drift the two paths apart fail loudly instead of
    /// silently producing wrong c_alpha summaries on the tensor path.
    #[test]
    fn factored_carry_summary_matches_flat_for_tensor_challenges() {
        use akita_algebra::eq_poly::EqPolynomial;
        use akita_algebra::ring::scalar_powers;
        use akita_challenges::{SparseChallenge, TensorChallengeSet};

        type FF = Fp32<251>;
        const DD: usize = 32;

        let sparse = |positions: &[u32], coeffs: &[i8]| -> SparseChallenge {
            SparseChallenge {
                positions: positions.to_vec(),
                coeffs: coeffs.to_vec(),
            }
        };

        let num_claims = 2usize;
        let left_len = 4usize;
        let right_len = 4usize;
        let num_blocks = left_len * right_len; // = 16, power of two

        let left = vec![
            sparse(&[0, 6], &[1, -1]),
            sparse(&[1, 7], &[1, 1]),
            sparse(&[3, 12], &[-1, 1]),
            sparse(&[2, 9], &[1, -1]),
            sparse(&[5, 10], &[1, 1]),
            sparse(&[4, 8], &[-1, -1]),
            sparse(&[11, 13], &[1, 1]),
            sparse(&[15, 30], &[1, -1]),
        ];
        let right = vec![
            sparse(&[0], &[1]),
            sparse(&[2], &[-1]),
            sparse(&[4], &[1]),
            sparse(&[6], &[-1]),
            sparse(&[8], &[1]),
            sparse(&[10], &[-1]),
            sparse(&[12], &[1]),
            sparse(&[14], &[-1]),
        ];
        let set = TensorChallengeSet {
            left,
            right,
            left_len,
            right_len,
            num_claims,
        };
        let tensor_challenges = TensorChallenges::Tensor(set.clone());

        let alpha = FF::from_u64(11);
        let alpha_pows = scalar_powers(alpha, DD);
        let alpha_pow_d_plus_one = alpha_pows[DD - 1] * alpha_pows[1] + FF::one();

        let flat_evals = tensor_challenges
            .evals_at_pows::<FF, FF, DD>(&alpha_pows)
            .expect("flat tensor materialisation");
        assert_eq!(flat_evals.len(), num_claims * num_blocks);

        let flat = PreparedChallengeEvals::Flat(flat_evals.clone());
        let factored = PreparedChallengeEvals::Tensor {
            challenges: set,
            alpha_pows: alpha_pows.clone(),
            alpha_pow_d_plus_one,
        };

        // `right_bits + left_bits = log₂(num_blocks)`. With `num_blocks = 16`
        // we need a 4-element low-bit challenge vector.
        let block_bits = num_blocks.trailing_zeros() as usize;
        assert_eq!(block_bits, 4);
        let x_low_cases = [
            vec![FF::from_u64(2), FF::from_u64(3), FF::zero(), FF::one()],
            vec![
                FF::from_u64(7),
                -FF::from_u64(4),
                FF::from_u64(5),
                FF::from_u64(9),
            ],
            vec![FF::zero(), FF::one(), -FF::from_u64(2), FF::from_u64(3)],
        ];

        for x_low in x_low_cases {
            let eq_low = EqPolynomial::evals(&x_low).expect("eq_low evals");
            for offset_low in 0..num_blocks {
                let got_factored = factored
                    .summarize_all_block_carries::<DD>(
                        num_claims, &x_low, &eq_low, offset_low, num_blocks,
                    )
                    .expect("factored summary");
                let got_flat = flat
                    .summarize_all_block_carries::<DD>(
                        num_claims, &x_low, &eq_low, offset_low, num_blocks,
                    )
                    .expect("flat summary");
                assert_eq!(
                    got_factored, got_flat,
                    "factored summary mismatch for x_low={x_low:?}, offset_low={offset_low}"
                );
            }
        }
    }
}
