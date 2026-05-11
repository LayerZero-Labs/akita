//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBase, RandomSampling};
use akita_transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use akita_transcript::{sample_ext_challenge, Transcript};
#[cfg(feature = "zk")]
use akita_types::zk;
use akita_types::{
    validate_opening_points_for_claims, AkitaExpandedSetup, FlatRingVec, LevelParams,
    RingOpeningPoint,
};

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub struct RingSwitchVerifyOutput<E: FieldCore> {
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

/// Precomputed challenge-derived data for deferred ring-switch row MLE evaluation.
///
/// Stores only data that cannot be derived from context at evaluation time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else is passed by reference at evaluation time to avoid
/// duplicating setup matrix views, opening points, and gadget vectors.
pub struct RingSwitchDeferredRowEval<F: FieldCore> {
    pub(crate) c_alphas: Vec<F>,
    pub(crate) eq_tau1: Vec<F>,
    pub(crate) total_blocks: usize,
    pub(crate) num_blocks: usize,
    pub(crate) num_claims: usize,
    pub(crate) depth_open: usize,
    pub(crate) depth_commit: usize,
    pub(crate) depth_fold: usize,
    #[cfg(feature = "zk")]
    pub(crate) d_blinding_segment_len: usize,
    #[cfg(feature = "zk")]
    pub(crate) b_blinding_digit_planes_per_group: usize,
    #[cfg(feature = "zk")]
    pub(crate) b_blinding_segment_len: usize,
    pub(crate) block_len: usize,
    pub(crate) inner_width: usize,
    pub(crate) log_basis: u32,
    pub(crate) n_a: usize,
    pub(crate) n_d: usize,
    pub(crate) n_b: usize,
    pub(crate) num_commitment_groups: usize,
    pub(crate) rows: usize,
    pub(crate) z_first: bool,
    pub(crate) claim_to_group: Vec<(usize, usize)>,
    #[cfg(feature = "zk")]
    pub(crate) group_poly_counts: Vec<usize>,
    pub(crate) num_points: usize,
    pub(crate) num_public_eval_rows: usize,
    pub(crate) gamma: Vec<F>,
    pub(crate) claim_to_point: Vec<usize>,
}

/// Replay the verifier half of ring switching.
///
/// This handles multiple opening points, arbitrary claim-to-point mapping, and
/// arbitrary commitment grouping. The recursive/single-point path is the
/// `opening_points = [pt]`, `claim_to_point = [0]`,
/// `group_poly_counts = [1]`, `num_public_eval_rows = 1` specialization.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or ring-switch row-eval
/// preparation fails.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub fn ring_switch_verifier<F, E, T, const D: usize>(
    opening_points: &[RingOpeningPoint<F>],
    claim_to_point: &[usize],
    challenges: &[SparseChallenge],
    w_len: usize,
    w_commitment: &FlatRingVec<F>,
    transcript: &mut T,
    lp: &LevelParams,
    group_poly_counts: &[usize],
    claim_to_group: &[usize],
    claim_poly_indices: &[usize],
    gamma: &[F],
    num_public_eval_rows: usize,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: ExtField<F>,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_SUMCHECK_W, w_commitment);

    let alpha: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH);

    let num_claims = claim_to_point.len();
    validate_opening_points_for_claims(opening_points, claim_to_point, lp, num_claims)?;
    if claim_to_group.len() != num_claims || claim_poly_indices.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let num_commitment_groups = group_poly_counts.len();
    for claim_idx in 0..num_claims {
        let group_idx = claim_to_group[claim_idx];
        if group_idx >= num_commitment_groups
            || claim_poly_indices[claim_idx] >= group_poly_counts[group_idx]
        {
            return Err(AkitaError::InvalidProof);
        }
    }

    let num_ring_elems = w_len / D;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let ring_bits = D.trailing_zeros() as usize;
    let m_rows = lp.m_row_count(num_commitment_groups, num_public_eval_rows);
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;

    let tau0: Vec<E> = (0..num_sc_vars)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
        .collect();
    let tau1: Vec<E> = (0..num_i)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
        .collect();
    let alpha_evals_y = scalar_powers(alpha, D);
    let gamma_e: Vec<E> = gamma.iter().copied().map(E::lift_base).collect();
    let prepared_row_eval = prepare_ring_switch_row_eval::<F, E, D>(
        challenges,
        alpha,
        lp,
        &tau1,
        group_poly_counts,
        claim_to_group,
        claim_poly_indices,
        &gamma_e,
        num_public_eval_rows,
        opening_points.len(),
        claim_to_point,
    )?;

    Ok(RingSwitchVerifyOutput {
        prepared_row_eval,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b: 1usize << lp.log_basis,
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
    challenges: &[SparseChallenge],
    alpha: E,
    lp: &LevelParams,
    tau1: &[E],
    group_poly_counts: &[usize],
    claim_to_group: &[usize],
    claim_poly_indices: &[usize],
    gamma: &[E],
    num_public_eval_rows: usize,
    opening_points_len: usize,
    claim_to_point: &[usize],
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + MulBase<F>,
{
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = claim_to_point.len();
    if claim_to_group.len() != num_claims || claim_poly_indices.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let num_commitment_groups = group_poly_counts.len();
    for claim_idx in 0..num_claims {
        let group_idx = claim_to_group[claim_idx];
        if group_idx >= num_commitment_groups
            || claim_poly_indices[claim_idx] >= group_poly_counts[group_idx]
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
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    #[cfg(feature = "zk")]
    let d_blinding_segment_len = zk::blinding_digit_plane_count::<F>(n_d, D, log_basis);
    #[cfg(feature = "zk")]
    let b_blinding_digit_planes_per_group = zk::blinding_digit_plane_count::<F>(n_b, D, log_basis);
    #[cfg(feature = "zk")]
    let b_blinding_segment_len = num_commitment_groups
        .checked_mul(b_blinding_digit_planes_per_group)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK blinding width overflow".to_string()))?;
    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.len() != total_blocks {
        return Err(AkitaError::InvalidSize {
            expected: total_blocks,
            actual: challenges.len(),
        });
    }
    let block_len = lp.block_len;
    let inner_width = block_len * depth_commit;
    let num_points = opening_points_len.max(1);
    let rows = lp.m_row_count(num_commitment_groups, num_public_eval_rows);

    let eq_tau1 = EqPolynomial::evals(tau1);
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let c_alphas: Vec<E> = challenges
        .iter()
        .map(|challenge| challenge.eval_at_pows::<F, E, D>(&alpha_pows))
        .collect::<Result<_, _>>()?;

    let z_first = lp.m_vars >= lp.r_vars;

    let claim_to_group: Vec<(usize, usize)> = claim_to_group
        .iter()
        .zip(claim_poly_indices.iter())
        .map(|(&group_idx, &poly_idx)| (group_idx, poly_idx))
        .collect();

    Ok(RingSwitchDeferredRowEval {
        c_alphas,
        eq_tau1,
        total_blocks,
        num_blocks,
        num_claims,
        depth_open,
        depth_commit,
        depth_fold,
        #[cfg(feature = "zk")]
        d_blinding_segment_len,
        #[cfg(feature = "zk")]
        b_blinding_digit_planes_per_group,
        #[cfg(feature = "zk")]
        b_blinding_segment_len,
        block_len,
        inner_width,
        log_basis,
        n_a: lp.a_key.row_len(),
        n_d,
        n_b,
        num_commitment_groups,
        rows,
        z_first,
        claim_to_group,
        #[cfg(feature = "zk")]
        group_poly_counts: group_poly_counts.to_vec(),
        num_points,
        num_public_eval_rows,
        gamma: gamma.to_vec(),
        claim_to_point: claim_to_point.to_vec(),
    })
}

impl<E: FieldCore> RingSwitchDeferredRowEval<E> {
    /// Evaluate the prepared ring-switch row table at the supplied point.
    ///
    /// # Errors
    ///
    /// Returns an error if the setup matrix cannot be viewed at `D` or an
    /// internal offset-eq evaluation receives inconsistent dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the prepared state was built for a layout inconsistent with
    /// the provided setup, opening points, or challenge vector. Callers should
    /// build values through [`prepare_ring_switch_row_eval`] or [`ring_switch_verifier`].
    #[inline]
    pub fn eval_at_point<F, const D: usize>(
        &self,
        x_challenges: &[E],
        setup: &AkitaExpandedSetup<F>,
        opening_points: &[RingOpeningPoint<F>],
        alpha: E,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F>,
    {
        super::slice_mle::eval_at_point_parts::<F, E, D>(
            self,
            x_challenges,
            setup,
            opening_points,
            alpha,
        )
        .map(|parts| parts.sum())
    }
}

#[inline]
pub(crate) fn summarize_pow2_block_carries_base<F, E>(
    eq_low: &[E],
    offset_low: usize,
    values: &[F],
) -> [E; 2]
where
    F: FieldCore,
    E: ExtField<F>,
{
    assert!(
        values.len().is_power_of_two(),
        "peeled inner block length must be a power of two"
    );
    assert_eq!(
        eq_low.len(),
        values.len(),
        "low eq table must match peeled inner block length"
    );
    assert!(
        offset_low < values.len(),
        "low offset must lie inside the peeled block"
    );

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

    out
}

#[inline]
pub(crate) fn summarize_strided_pow2_block_carries<F, E, const D: usize>(
    eq_low: &[E],
    offset_low: usize,
    row: &[CyclotomicRing<F, D>],
    alpha_pows: &[E],
    block_count: usize,
    block_stride: usize,
    lane_offset: usize,
) -> [E; 2]
where
    F: FieldCore,
    E: ExtField<F>,
{
    debug_assert!(block_count.is_power_of_two());
    debug_assert_eq!(eq_low.len(), block_count);
    debug_assert!(offset_low < block_count);

    let inner_bits = block_count.trailing_zeros() as usize;
    let inner_mask = block_count - 1;
    let mut out = [E::zero(), E::zero()];
    for block_idx in 0..block_count {
        let sum = offset_low + block_idx;
        let carry = sum >> inner_bits;
        let low_idx = sum & inner_mask;
        let col = block_idx * block_stride + lane_offset;
        let value = eval_ring_at_pows(&row[col], alpha_pows);
        out[carry] += value * eq_low[low_idx];
    }
    out
}
