//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{
    eval_offset_eq_peeled_carry_terms, eval_offset_eq_tensor, summarize_pow2_block_carries,
};
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_algebra::{CyclotomicRing, SparseChallenge};
use akita_challenges::eval_sparse_challenge_at_pows;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use akita_transcript::{sample_challenge_scalars, Transcript};
use akita_types::{
    checked_num_claims_from_group_sizes, gadget_row_scalars, r_decomp_levels,
    validate_opening_points_for_claims, AkitaExpandedSetup, FlatRingVec, LevelParams,
    RingMatrixView, RingOpeningPoint,
};

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub struct RingSwitchVerifyOutput<F: FieldCore> {
    /// Prepared data for deferred M-table MLE evaluation.
    pub prepared_m_eval: PreparedMEval<F>,
    /// Evaluation table of alpha powers over the ring-coordinate dimension.
    pub alpha_evals_y: Vec<F>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
    /// Challenge tau0 for the stage-1 sumcheck.
    pub tau0: Vec<F>,
    /// Challenge tau1 for the stage-2 M-row combination.
    pub tau1: Vec<F>,
    /// Basis size `b = 2^log_basis`.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: F,
}

/// Precomputed challenge-derived data for deferred M-table MLE evaluation.
///
/// Stores only data that cannot be derived from context at evaluation time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else is passed by reference at evaluation time to avoid
/// duplicating setup matrix views, opening points, and gadget vectors.
pub struct PreparedMEval<F: FieldCore> {
    c_alphas: Vec<F>,
    eq_tau1: Vec<F>,
    total_blocks: usize,
    num_blocks: usize,
    num_claims: usize,
    depth_open: usize,
    depth_commit: usize,
    depth_fold: usize,
    block_len: usize,
    inner_width: usize,
    log_basis: u32,
    n_a: usize,
    n_d: usize,
    n_b: usize,
    num_commitment_groups: usize,
    rows: usize,
    z_first: bool,
    claim_to_group: Vec<(usize, usize)>,
    num_points: usize,
    num_eval_rows: usize,
    gamma: Vec<F>,
    claim_to_point: Vec<usize>,
}

/// Replay the verifier half of ring switching.
///
/// This handles multiple opening points, arbitrary claim-to-point mapping, and
/// arbitrary commitment grouping. The recursive/single-point path is the
/// `opening_points = [pt]`, `claim_to_point = [0]`,
/// `claim_group_sizes = [1]`, `num_eval_rows = 1` specialization.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or M-eval
/// preparation fails.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub fn ring_switch_verifier<F, T, const D: usize>(
    opening_points: &[RingOpeningPoint<F>],
    claim_to_point: &[usize],
    challenges: &[SparseChallenge],
    w_len: usize,
    w_commitment: &FlatRingVec<F>,
    transcript: &mut T,
    lp: &LevelParams,
    claim_group_sizes: &[usize],
    gamma: &[F],
    num_eval_rows: usize,
) -> Result<RingSwitchVerifyOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_SUMCHECK_W, w_commitment);

    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    validate_opening_points_for_claims(opening_points, claim_to_point, lp, num_claims)?;
    let num_commitment_groups = claim_group_sizes.len();

    let num_ring_elems = w_len / D;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let ring_bits = D.trailing_zeros() as usize;
    let m_rows = lp.m_row_count(num_commitment_groups, num_eval_rows);
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;

    let tau0 = sample_challenge_scalars::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);
    let tau1 = sample_challenge_scalars::<F, T>(transcript, CHALLENGE_TAU1, num_i);
    let alpha_evals_y = scalar_powers(alpha, D);
    let prepared_m_eval = prepare_m_eval::<F, D>(
        challenges,
        alpha,
        lp,
        &tau1,
        claim_group_sizes,
        gamma,
        num_eval_rows,
        opening_points.len(),
        claim_to_point,
    )?;

    Ok(RingSwitchVerifyOutput {
        prepared_m_eval,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b: 1usize << lp.log_basis,
        alpha,
    })
}

/// Prepare deferred verifier M-table evaluation data.
///
/// # Errors
///
/// Returns an error if gamma/challenge lengths do not match the claim shape,
/// the expanded tau1 table is too short for the level layout, or sparse
/// challenge evaluation fails.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "prepare_m_eval")]
pub fn prepare_m_eval<F: FieldCore + CanonicalField, const D: usize>(
    challenges: &[SparseChallenge],
    alpha: F,
    lp: &LevelParams,
    tau1: &[F],
    claim_group_sizes: &[usize],
    gamma: &[F],
    num_eval_rows: usize,
    opening_points_len: usize,
    claim_to_point: &[usize],
) -> Result<PreparedMEval<F>, AkitaError> {
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    let num_commitment_groups = claim_group_sizes.len();

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
    let rows = lp.m_row_count(num_commitment_groups, num_eval_rows);

    let eq_tau1 = EqPolynomial::evals(tau1);
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let c_alphas: Vec<F> = challenges
        .iter()
        .map(|challenge| eval_sparse_challenge_at_pows::<F, D>(challenge, &alpha_pows))
        .collect::<Result<_, _>>()?;

    let z_first = lp.m_vars >= lp.r_vars;

    let claim_to_group: Vec<(usize, usize)> = claim_group_sizes
        .iter()
        .enumerate()
        .flat_map(|(group_idx, &group_size)| {
            (0..group_size).map(move |within_group| (group_idx, within_group))
        })
        .collect();

    Ok(PreparedMEval {
        c_alphas,
        eq_tau1,
        total_blocks,
        num_blocks,
        num_claims,
        depth_open,
        depth_commit,
        depth_fold,
        block_len,
        inner_width,
        log_basis,
        n_a: lp.a_key.row_len(),
        n_d: lp.d_key.row_len(),
        n_b: lp.b_key.row_len(),
        num_commitment_groups,
        rows,
        z_first,
        claim_to_group,
        num_points,
        num_eval_rows,
        gamma: gamma.to_vec(),
        claim_to_point: claim_to_point.to_vec(),
    })
}

impl<F: FieldCore + CanonicalField> PreparedMEval<F> {
    /// Evaluate the prepared verifier M-table at the supplied point.
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
    /// build values through [`prepare_m_eval`] or [`ring_switch_verifier`].
    #[inline]
    pub fn eval_at_point<const D: usize>(
        &self,
        x_challenges: &[F],
        setup: &AkitaExpandedSetup<F>,
        opening_points: &[RingOpeningPoint<F>],
        alpha: F,
    ) -> Result<F, AkitaError> {
        let alpha_pows = scalar_powers(alpha, D);
        let g1_open = gadget_row_scalars::<F>(self.depth_open, self.log_basis);
        let g1_commit = gadget_row_scalars::<F>(self.depth_commit, self.log_basis);
        let fold_gadget = gadget_row_scalars::<F>(self.depth_fold, self.log_basis);
        let levels = r_decomp_levels::<F>(self.log_basis);
        let r_gadget = gadget_row_scalars::<F>(levels, self.log_basis);

        let stride = setup.seed.max_stride;
        let d_view = setup.shared_matrix.ring_view::<D>(self.n_d, stride);
        let b_view = setup.shared_matrix.ring_view::<D>(self.n_b, stride);
        let a_view = setup.shared_matrix.ring_view::<D>(self.n_a, stride);

        let consistency_weight = self.eq_tau1[0];
        let public_weights = &self.eq_tau1[1..(1 + self.num_eval_rows)];
        let d_start = 1 + self.num_eval_rows;
        let commitment_row_count = self.n_b * self.num_commitment_groups;
        let b_start = d_start + self.n_d;
        let a_start = b_start + commitment_row_count;
        let a_weights = &self.eq_tau1[a_start..self.rows];

        let total_blocks = self.total_blocks;
        let num_blocks = self.num_blocks;
        let num_claims = self.num_claims;
        let depth_open = self.depth_open;
        let depth_commit = self.depth_commit;
        let depth_fold = self.depth_fold;
        let block_len = self.block_len;
        let inner_width = self.inner_width;
        let n_d = self.n_d;
        let n_b = self.n_b;
        let n_a = self.n_a;
        let rows = self.rows;
        let num_points = self.num_points;
        let c_alphas = &self.c_alphas;
        let eq_tau1 = &self.eq_tau1;
        let d_weights = &eq_tau1[d_start..(d_start + n_d)];
        let claim_to_group = &self.claim_to_group;
        let claim_to_point = &self.claim_to_point;
        let gamma = &self.gamma;

        let w_len = depth_open * total_blocks;
        let t_len = depth_open * n_a * total_blocks;
        let z_total_blocks = num_points * block_len;
        let z_len = depth_fold * depth_commit * z_total_blocks;
        let r_tail_len = rows * levels;

        let is_multi_point = num_points > 1;

        let offset_z = if self.z_first { 0 } else { w_len + t_len };
        let offset_w = if self.z_first { z_len } else { 0 };
        let offset_t = if self.z_first { z_len + w_len } else { w_len };
        let block_bits = num_blocks.trailing_zeros() as usize;
        let block_low_eq = EqPolynomial::evals(&x_challenges[..block_bits]);
        let block_offset_low = offset_w & (num_blocks - 1);
        debug_assert_eq!(block_offset_low, offset_t & (num_blocks - 1));

        let opening_point_block_summaries: Vec<[F; 2]> = opening_points
            .iter()
            .map(|opening_point| {
                summarize_pow2_block_carries(&block_low_eq, block_offset_low, &opening_point.b)
            })
            .collect();
        let challenge_block_summaries: Vec<[F; 2]> = (0..num_claims)
            .map(|claim_idx| {
                let start = claim_idx * num_blocks;
                summarize_pow2_block_carries(
                    &block_low_eq,
                    block_offset_low,
                    &c_alphas[start..(start + num_blocks)],
                )
            })
            .collect();

        let mut w_carry_terms = vec![[F::zero(), F::zero()]; num_claims * depth_open];
        for (dig, &g_open) in g1_open.iter().enumerate() {
            let q_base = dig * num_claims;
            for claim_idx in 0..num_claims {
                let q = q_base + claim_idx;
                let point_idx = if is_multi_point {
                    claim_to_point[claim_idx]
                } else {
                    0
                };
                let [public_low0, public_low1] = opening_point_block_summaries[point_idx];
                let public_scale = public_weights[point_idx] * gamma[claim_idx] * g_open;
                w_carry_terms[q][0] += public_scale * public_low0;
                w_carry_terms[q][1] += public_scale * public_low1;

                let [challenge_low0, challenge_low1] = challenge_block_summaries[claim_idx];
                let challenge_scale = consistency_weight * g_open;
                w_carry_terms[q][0] += challenge_scale * challenge_low0;
                w_carry_terms[q][1] += challenge_scale * challenge_low1;
            }
        }
        let w_sep = {
            let _span = tracing::info_span!("m_eval_w_sep").entered();
            eval_offset_eq_peeled_carry_terms(x_challenges, offset_w, block_bits, &w_carry_terms)
        };
        let w_d = {
            let _span = tracing::info_span!("m_eval_w_d").entered();
            eval_d_matrix_w_residual_direct(
                x_challenges,
                offset_w,
                num_blocks,
                num_claims,
                depth_open,
                d_weights,
                d_view,
                &alpha_pows,
            )
        };

        let mut t_carry_terms = vec![[F::zero(), F::zero()]; num_claims * depth_open * n_a];
        for (a_idx, &a_weight) in a_weights.iter().enumerate() {
            for (digit_idx, &g_open) in g1_open.iter().enumerate() {
                let q_base = num_claims * (digit_idx + depth_open * a_idx);
                let scale = a_weight * g_open;
                for (claim_idx, &[challenge_low0, challenge_low1]) in
                    challenge_block_summaries.iter().enumerate()
                {
                    let q = q_base + claim_idx;
                    t_carry_terms[q][0] += scale * challenge_low0;
                    t_carry_terms[q][1] += scale * challenge_low1;
                }
            }
        }
        let t_sep = {
            let _span = tracing::info_span!("m_eval_t_sep").entered();
            eval_offset_eq_peeled_carry_terms(x_challenges, offset_t, block_bits, &t_carry_terms)
        };

        let t_b = {
            let _span = tracing::info_span!("m_eval_t_b").entered();
            eval_b_matrix_t_residual_direct(
                x_challenges,
                offset_t,
                num_blocks,
                num_claims,
                depth_open,
                n_a,
                n_b,
                eq_tau1,
                b_start,
                claim_to_group,
                b_view,
                &alpha_pows,
            )
        };

        let z_base_len = num_points * inner_width;
        let z_base: Vec<F> = {
            let _span = tracing::info_span!("m_eval_z_base").entered();
            cfg_into_iter!(0..z_base_len)
                .map(|k| {
                    let point_idx = if is_multi_point { k / inner_width } else { 0 };
                    let local_k = if is_multi_point { k % inner_width } else { k };
                    let block_idx = local_k / depth_commit;
                    let digit_idx = local_k % depth_commit;
                    let opening_point = &opening_points[point_idx];
                    let mut acc =
                        consistency_weight * opening_point.a[block_idx] * g1_commit[digit_idx];
                    for (a_idx, eq_i) in a_weights.iter().enumerate() {
                        if !eq_i.is_zero() {
                            acc +=
                                *eq_i * eval_ring_at_pows(&a_view.row(a_idx)[local_k], &alpha_pows);
                        }
                    }
                    acc
                })
                .collect()
        };

        let z_dense = {
            let _span = tracing::info_span!("m_eval_z_dense").entered();
            let z_segment: Vec<F> = cfg_into_iter!(0..z_len)
                .map(|x| {
                    let compound_dig = x / z_total_blocks;
                    let global_blk = x % z_total_blocks;
                    let dc = compound_dig / depth_fold;
                    let df = compound_dig % depth_fold;
                    let point_idx = global_blk / block_len;
                    let blk = global_blk % block_len;
                    let phys_k = point_idx * inner_width + blk * depth_commit + dc;
                    -(z_base[phys_k] * fold_gadget[df])
                })
                .collect();
            eval_offset_eq_tensor(x_challenges, offset_z, F::one(), &[z_segment.as_slice()])
        };

        let alpha_pow_d = alpha_pows[D - 1] * alpha;
        let denom = alpha_pow_d + F::one();

        let r_tail_dims_pow2 = levels.is_power_of_two();
        let offset_r = w_len + t_len + z_len;

        let r_sep = if r_tail_dims_pow2 {
            eval_offset_eq_tensor(
                x_challenges,
                offset_r,
                -denom,
                &[&r_gadget, &eq_tau1[..rows]],
            )
        } else {
            F::zero()
        };
        let r_dense = if r_tail_dims_pow2 {
            F::zero()
        } else {
            let _span = tracing::info_span!("m_eval_r_dense").entered();
            let r_tail: Vec<F> = cfg_into_iter!(0..r_tail_len)
                .map(|idx| {
                    let row_idx = idx / levels;
                    let level_idx = idx % levels;
                    -(eq_tau1[row_idx] * denom * r_gadget[level_idx])
                })
                .collect();
            eval_offset_eq_tensor(x_challenges, offset_r, F::one(), &[r_tail.as_slice()])
        };

        Ok(z_dense + w_sep + w_d + t_sep + t_b + r_sep + r_dense)
    }
}

#[inline]
fn summarize_strided_pow2_block_carries<F: FieldCore, const D: usize>(
    eq_low: &[F],
    offset_low: usize,
    row: &[CyclotomicRing<F, D>],
    alpha_pows: &[F],
    block_count: usize,
    block_stride: usize,
    lane_offset: usize,
) -> [F; 2] {
    debug_assert!(block_count.is_power_of_two());
    debug_assert_eq!(eq_low.len(), block_count);
    debug_assert!(offset_low < block_count);

    let inner_bits = block_count.trailing_zeros() as usize;
    let inner_mask = block_count - 1;
    let mut out = [F::zero(), F::zero()];
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

#[allow(clippy::too_many_arguments)]
#[inline]
fn eval_d_matrix_w_residual_direct<F: FieldCore, const D: usize>(
    x_challenges: &[F],
    offset_w: usize,
    num_blocks: usize,
    num_claims: usize,
    depth_open: usize,
    d_weights: &[F],
    d_view: RingMatrixView<'_, F, D>,
    alpha_pows: &[F],
) -> F {
    debug_assert!(num_blocks.is_power_of_two());
    let block_bits = num_blocks.trailing_zeros() as usize;
    let block_low_eq = EqPolynomial::evals(&x_challenges[..block_bits]);
    let block_offset_low = offset_w & (num_blocks - 1);
    let per_claim_d_width = num_blocks * depth_open;
    let carry_terms: Vec<[F; 2]> = cfg_into_iter!(0..(num_claims * depth_open))
        .map(|q| {
            let claim_idx = q % num_claims;
            let dig = q / num_claims;
            let lane_offset = claim_idx * per_claim_d_width + dig;
            let mut out = [F::zero(), F::zero()];
            for (di, &d_weight) in d_weights.iter().enumerate() {
                if d_weight.is_zero() {
                    continue;
                }
                let row = d_view.row(di);
                let [block_low0, block_low1] = summarize_strided_pow2_block_carries(
                    &block_low_eq,
                    block_offset_low,
                    row,
                    alpha_pows,
                    num_blocks,
                    depth_open,
                    lane_offset,
                );
                out[0] += d_weight * block_low0;
                out[1] += d_weight * block_low1;
            }
            out
        })
        .collect();
    eval_offset_eq_peeled_carry_terms(x_challenges, offset_w, block_bits, &carry_terms)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn eval_b_matrix_t_residual_direct<F: FieldCore, const D: usize>(
    x_challenges: &[F],
    offset_t: usize,
    num_blocks: usize,
    num_claims: usize,
    depth_open: usize,
    n_a: usize,
    n_b: usize,
    eq_tau1: &[F],
    b_start: usize,
    claim_to_group: &[(usize, usize)],
    b_view: RingMatrixView<'_, F, D>,
    alpha_pows: &[F],
) -> F {
    debug_assert!(num_blocks.is_power_of_two());
    let block_bits = num_blocks.trailing_zeros() as usize;
    let block_low_eq = EqPolynomial::evals(&x_challenges[..block_bits]);
    let block_offset_low = offset_t & (num_blocks - 1);
    let t_compound_per_block = n_a * depth_open;
    let t_cols_per_claim = t_compound_per_block * num_blocks;
    let carry_terms: Vec<[F; 2]> = cfg_into_iter!(0..(num_claims * n_a * depth_open))
        .map(|q| {
            let claim_idx = q % num_claims;
            let compound_dig = q / num_claims;
            let a_idx = compound_dig / depth_open;
            let digit_idx = compound_dig % depth_open;
            let (group_idx, claim_idx_within_group) = claim_to_group[claim_idx];
            let commitment_weights =
                &eq_tau1[(b_start + group_idx * n_b)..(b_start + (group_idx + 1) * n_b)];
            let lane_offset =
                claim_idx_within_group * t_cols_per_claim + a_idx * depth_open + digit_idx;
            let mut out = [F::zero(), F::zero()];
            for (row_idx, &eq_i) in commitment_weights.iter().enumerate() {
                if eq_i.is_zero() {
                    continue;
                }
                let row = b_view.row(row_idx);
                let [block_low0, block_low1] = summarize_strided_pow2_block_carries(
                    &block_low_eq,
                    block_offset_low,
                    row,
                    alpha_pows,
                    num_blocks,
                    t_compound_per_block,
                    lane_offset,
                );
                out[0] += eq_i * block_low0;
                out[1] += eq_i * block_low1;
            }
            out
        })
        .collect();
    eval_offset_eq_peeled_carry_terms(x_challenges, offset_t, block_bits, &carry_terms)
}
