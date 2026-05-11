#![allow(missing_docs)]

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{
    eval_offset_eq_peeled_carry_terms, eval_offset_eq_tensor, summarize_pow2_block_carries,
};
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::protocol::ring_switch::{compute_m_evals_x, ring_switch_build_w};
use akita_prover::{AkitaPolyOps, CommitmentProver, OneHotPoly, QuadraticEquation};
use akita_sumcheck::multilinear_eval;
use akita_transcript::labels::{
    ABSORB_EVALUATION_CLAIMS, ABSORB_EVAL_OPENINGS_FIELD, CHALLENGE_EVAL_BATCH,
    CHALLENGE_RING_SWITCH, CHALLENGE_TAU1,
};
use akita_transcript::{Blake2bTranscript, Transcript};
use akita_types::{
    append_batch_shape_to_transcript, append_batched_commitments_to_transcript,
    checked_num_claims_from_group_sizes, gadget_row_scalars, r_decomp_levels,
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
    validate_opening_points_for_claims, AkitaExpandedSetup, BasisMode, BlockOrder, LevelParams,
    RingMatrixView, RingOpeningPoint,
};
use akita_verifier::prepare_ring_switch_row_eval;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::time::{Duration, Instant};

type F = fp128::Field;
type Cfg = fp128::D32OneHot;
const D: usize = Cfg::D;
const DEFAULT_NV: usize = 32;
const ONEHOT_K: usize = 256;

#[derive(Clone)]
struct PreparedPlaygroundMEval<F: FieldCore> {
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

#[derive(Clone, Copy, Debug, Default)]
struct ComponentTimings {
    prelude: Duration,
    block_summaries: Duration,
    w_sep_build: Duration,
    w_sep_eval: Duration,
    w_d: Duration,
    t_sep_build: Duration,
    t_sep_eval: Duration,
    t_b: Duration,
    z_base: Duration,
    z_dense: Duration,
    r_tail: Duration,
    total: Duration,
}

impl ComponentTimings {
    fn add_assign(&mut self, rhs: Self) {
        self.prelude += rhs.prelude;
        self.block_summaries += rhs.block_summaries;
        self.w_sep_build += rhs.w_sep_build;
        self.w_sep_eval += rhs.w_sep_eval;
        self.w_d += rhs.w_d;
        self.t_sep_build += rhs.t_sep_build;
        self.t_sep_eval += rhs.t_sep_eval;
        self.t_b += rhs.t_b;
        self.z_base += rhs.z_base;
        self.z_dense += rhs.z_dense;
        self.r_tail += rhs.r_tail;
        self.total += rhs.total;
    }

    fn div(self, n: u32) -> Self {
        Self {
            prelude: self.prelude / n,
            block_summaries: self.block_summaries / n,
            w_sep_build: self.w_sep_build / n,
            w_sep_eval: self.w_sep_eval / n,
            w_d: self.w_d / n,
            t_sep_build: self.t_sep_build / n,
            t_sep_eval: self.t_sep_eval / n,
            t_b: self.t_b / n,
            z_base: self.z_base / n,
            z_dense: self.z_dense / n,
            r_tail: self.r_tail / n,
            total: self.total / n,
        }
    }
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let out = f();
    (out, start.elapsed())
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_flag(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| value != "0")
        .unwrap_or(default)
}

fn onehot_k_for_num_vars(nv: usize) -> usize {
    let max_supported_log_k = ONEHOT_K.trailing_zeros() as usize;
    if nv >= max_supported_log_k {
        ONEHOT_K
    } else {
        1usize << nv
    }
}

fn opening_from_poly<P: AkitaPolyOps<F, D>>(
    poly: &P,
    point: &[F],
    layout: &LevelParams,
    basis: BasisMode,
) -> F {
    let alpha_bits = D.trailing_zeros() as usize;
    let target_num_vars = alpha_bits + layout.m_vars + layout.r_vars;
    assert!(
        point.len() <= target_num_vars,
        "opening point length {} exceeds target root arity {}",
        point.len(),
        target_num_vars
    );
    let mut padded_point = point.to_vec();
    padded_point.resize(target_num_vars, F::zero());

    let inner_point = &padded_point[..alpha_bits];
    let reduced_point = &padded_point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        basis,
        BlockOrder::RowMajor,
    )
    .expect("opening point shape should match layout");

    let (y_ring, _) = poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );
    let v = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis)
        .expect("inner opening point should match ring dimension");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

#[allow(clippy::too_many_arguments)]
fn prepare_playground_m_eval<F: FieldCore + CanonicalField, const D: usize>(
    challenges: &[SparseChallenge],
    alpha: F,
    lp: &LevelParams,
    tau1: &[F],
    claim_group_sizes: &[usize],
    gamma: &[F],
    num_eval_rows: usize,
    opening_points_len: usize,
    claim_to_point: &[usize],
) -> Result<PreparedPlaygroundMEval<F>, AkitaError> {
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
        .map(|challenge| challenge.eval_at_pows::<F, D>(&alpha_pows))
        .collect::<Result<_, _>>()?;

    let claim_to_group: Vec<(usize, usize)> = claim_group_sizes
        .iter()
        .enumerate()
        .flat_map(|(group_idx, &group_size)| {
            (0..group_size).map(move |within_group| (group_idx, within_group))
        })
        .collect();

    Ok(PreparedPlaygroundMEval {
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
        z_first: lp.m_vars >= lp.r_vars,
        claim_to_group,
        num_points,
        num_eval_rows,
        gamma: gamma.to_vec(),
        claim_to_point: claim_to_point.to_vec(),
    })
}

impl<F: FieldCore + CanonicalField> PreparedPlaygroundMEval<F> {
    fn total_cols(&self) -> usize {
        let levels = r_decomp_levels::<F>(self.log_basis);
        let w_len = self.depth_open * self.total_blocks;
        let t_len = self.depth_open * self.n_a * self.total_blocks;
        let z_total_blocks = self.num_points * self.block_len;
        let z_len = self.depth_fold * self.depth_commit * z_total_blocks;
        let r_tail_len = self.rows * levels;
        w_len + t_len + z_len + r_tail_len
    }

    fn timed_eval_at_point<const D: usize>(
        &self,
        x_challenges: &[F],
        setup: &AkitaExpandedSetup<F>,
        opening_points: &[RingOpeningPoint<F>],
        alpha: F,
    ) -> Result<(F, ComponentTimings), AkitaError> {
        let total_start = Instant::now();
        let (
            (alpha_pows, g1_open, g1_commit, fold_gadget, levels, r_gadget, d_view, b_view, a_view),
            prelude,
        ) = timed(|| {
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
            (
                alpha_pows,
                g1_open,
                g1_commit,
                fold_gadget,
                levels,
                r_gadget,
                d_view,
                b_view,
                a_view,
            )
        });

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

        let ((opening_point_block_summaries, challenge_block_summaries), block_summaries) =
            timed(|| {
                let block_low_eq = EqPolynomial::evals(&x_challenges[..block_bits]);
                let block_offset_low = offset_w & (num_blocks - 1);
                debug_assert_eq!(block_offset_low, offset_t & (num_blocks - 1));
                let opening_point_block_summaries: Vec<[F; 2]> = opening_points
                    .iter()
                    .map(|opening_point| {
                        summarize_pow2_block_carries(
                            &block_low_eq,
                            block_offset_low,
                            &opening_point.b,
                        )
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
                (opening_point_block_summaries, challenge_block_summaries)
            });

        let (w_carry_terms, w_sep_build) = timed(|| {
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
            w_carry_terms
        });
        let (w_sep, w_sep_eval) = timed(|| {
            eval_offset_eq_peeled_carry_terms(x_challenges, offset_w, block_bits, &w_carry_terms)
        });

        let (w_d, w_d_time) = timed(|| {
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
        });

        let (t_carry_terms, t_sep_build) = timed(|| {
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
            t_carry_terms
        });
        let (t_sep, t_sep_eval) = timed(|| {
            eval_offset_eq_peeled_carry_terms(x_challenges, offset_t, block_bits, &t_carry_terms)
        });

        let (t_b, t_b_time) = timed(|| {
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
        });

        let (z_base, z_base_time) = timed(|| {
            let z_base_len = num_points * inner_width;
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
                .collect::<Vec<F>>()
        });

        let (z_dense, z_dense_time) = timed(|| {
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
        });

        let (r_tail, r_tail_time) = timed(|| {
            let alpha_pow_d = alpha_pows[D - 1] * alpha;
            let denom = alpha_pow_d + F::one();
            let offset_r = w_len + t_len + z_len;
            if levels.is_power_of_two() {
                eval_offset_eq_tensor(
                    x_challenges,
                    offset_r,
                    -denom,
                    &[&r_gadget, &eq_tau1[..rows]],
                )
            } else {
                let r_tail: Vec<F> = cfg_into_iter!(0..r_tail_len)
                    .map(|idx| {
                        let row_idx = idx / levels;
                        let level_idx = idx % levels;
                        -(eq_tau1[row_idx] * denom * r_gadget[level_idx])
                    })
                    .collect();
                eval_offset_eq_tensor(x_challenges, offset_r, F::one(), &[r_tail.as_slice()])
            }
        });

        let value = z_dense + w_sep + w_d + t_sep + t_b + r_tail;
        let timings = ComponentTimings {
            prelude,
            block_summaries,
            w_sep_build,
            w_sep_eval,
            w_d: w_d_time,
            t_sep_build,
            t_sep_eval,
            t_b: t_b_time,
            z_base: z_base_time,
            z_dense: z_dense_time,
            r_tail: r_tail_time,
            total: total_start.elapsed(),
        };
        Ok((value, timings))
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

fn print_duration(label: &str, duration: Duration, total: Duration) {
    let total_s = total.as_secs_f64();
    let pct = if total_s > 0.0 {
        100.0 * duration.as_secs_f64() / total_s
    } else {
        0.0
    };
    println!(
        "{label:>18}: {:>10.3} ms  {:>6.2}%",
        duration.as_secs_f64() * 1e3,
        pct
    );
}

fn print_timings(t: ComponentTimings) {
    println!("\ncomponent timings (average per eval):");
    print_duration("prelude", t.prelude, t.total);
    print_duration("block_summaries", t.block_summaries, t.total);
    print_duration("w_sep_build", t.w_sep_build, t.total);
    print_duration("w_sep_eval", t.w_sep_eval, t.total);
    print_duration("w_d", t.w_d, t.total);
    print_duration("t_sep_build", t.t_sep_build, t.total);
    print_duration("t_sep_eval", t.t_sep_eval, t.total);
    print_duration("t_b", t.t_b, t.total);
    print_duration("z_base", t.z_base, t.total);
    print_duration("z_dense", t.z_dense, t.total);
    print_duration("r_tail", t.r_tail, t.total);
    print_duration("total", t.total, t.total);
}

fn main() -> Result<(), AkitaError> {
    let nv = env_usize("AKITA_PLAYGROUND_NV", DEFAULT_NV);
    let iterations = env_usize("AKITA_PLAYGROUND_ITERS", 5);
    let check_materialized = env_flag("AKITA_PLAYGROUND_CHECK_MATERIALIZED", false);

    let layout = Cfg::commitment_layout(nv).expect("onehot commitment layout");
    let required_vars = layout.m_vars + layout.r_vars + D.trailing_zeros() as usize;
    assert!(
        required_vars <= nv,
        "layout requires {required_vars} variables but nv={nv}"
    );

    let onehot_k = onehot_k_for_num_vars(nv);
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let total_field = (layout.num_blocks * layout.block_len)
        .checked_mul(D)
        .expect("total field size overflow");
    let total_chunks = total_field / onehot_k;
    assert_eq!(total_chunks * onehot_k, total_field);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
        .collect();
    let onehot_poly = OneHotPoly::<F, D, u8>::new(onehot_k, indices).unwrap();
    let point: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();

    let setup_start = Instant::now();
    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
    let setup_elapsed = setup_start.elapsed();
    let commit_start = Instant::now();
    let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        std::slice::from_ref(&onehot_poly),
        &setup,
    )
    .expect("commitment");
    let commit_elapsed = commit_start.elapsed();

    let alpha_bits = D.trailing_zeros() as usize;
    let outer_point = &point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        outer_point,
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
        BlockOrder::RowMajor,
    )?;
    let (y_ring, w_folded) = onehot_poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );
    let opening = opening_from_poly(&onehot_poly, &point, &layout, BasisMode::Lagrange);

    let mut transcript = Blake2bTranscript::<F>::new(b"eval-at-point-playground");
    append_batch_shape_to_transcript::<F, _>(&[1], &[1], &mut transcript);
    append_batched_commitments_to_transcript(std::slice::from_ref(&commitment), &mut transcript);
    for pt in &point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, &opening);
    let gamma = vec![transcript.challenge_scalar(CHALLENGE_EVAL_BATCH)];
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

    let quad_start = Instant::now();
    let mut quad_eq = QuadraticEquation::<F, D>::new_prover(
        &setup.ntt_shared,
        vec![ring_opening_point.clone()],
        vec![0usize],
        &[&onehot_poly],
        vec![w_folded],
        &[1usize],
        layout.clone(),
        vec![hint],
        &mut transcript,
        std::slice::from_ref(&commitment),
        std::slice::from_ref(&y_ring),
        gamma.clone(),
        setup.expanded.seed.max_stride,
    )
    .expect("quadratic equation");
    let quad_elapsed = quad_start.elapsed();

    let build_w_start = Instant::now();
    let w = ring_switch_build_w::<F, D>(&mut quad_eq, &setup.expanded, &setup.ntt_shared, &layout)
        .expect("ring-switch witness");
    let build_w_elapsed = build_w_start.elapsed();

    let alpha = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);
    let rows = layout.m_row_count(1, 1);
    let num_i = rows.next_power_of_two().trailing_zeros() as usize;
    let tau1: Vec<F> = (0..num_i)
        .map(|_| transcript.challenge_scalar(CHALLENGE_TAU1))
        .collect();

    validate_opening_points_for_claims(
        std::slice::from_ref(&ring_opening_point),
        &[0usize],
        &layout,
        1,
    )?;
    let prepared = prepare_playground_m_eval::<F, D>(
        &quad_eq.challenges,
        alpha,
        &layout,
        &tau1,
        &[1usize],
        &gamma,
        1,
        1,
        &[0usize],
    )?;
    let verifier_prepared = prepare_m_eval::<F, D>(
        &quad_eq.challenges,
        alpha,
        &layout,
        &tau1,
        &[1usize],
        &gamma,
        1,
        1,
        &[0usize],
    )?;

    let total_cols = prepared.total_cols();
    let witness_cols = w.len() / D;
    assert_eq!(
        total_cols, witness_cols,
        "instrumented layout should match ring_switch_build_w witness width"
    );
    let col_bits = total_cols.next_power_of_two().trailing_zeros() as usize;
    let mut x_rng = StdRng::seed_from_u64(0x1234_5678_9abc_def0);
    let x_challenges: Vec<F> = (0..col_bits)
        .map(|_| F::from_canonical_u128_reduced(x_rng.gen::<u128>()))
        .collect();

    println!("eval_at_point playground");
    println!("  nv={nv}, D={D}, onehot_k={onehot_k}, iterations={iterations}");
    println!(
        "  layout: m_vars={}, r_vars={}, num_blocks={}, block_len={}, log_basis={}, depth_open={}, depth_commit={}, depth_fold={}",
        layout.m_vars,
        layout.r_vars,
        layout.num_blocks,
        layout.block_len,
        layout.log_basis,
        layout.num_digits_open,
        layout.num_digits_commit,
        layout.num_digits_fold
    );
    println!(
        "  rows={}, tau1_vars={}, total_cols={}, col_bits={}, w_field_len={}",
        rows,
        num_i,
        total_cols,
        col_bits,
        w.len()
    );
    println!(
        "  setup={:.3} ms, commit={:.3} ms, quad_eq={:.3} ms, ring_switch_build_w={:.3} ms",
        setup_elapsed.as_secs_f64() * 1e3,
        commit_elapsed.as_secs_f64() * 1e3,
        quad_elapsed.as_secs_f64() * 1e3,
        build_w_elapsed.as_secs_f64() * 1e3
    );

    let warm_verifier_value = verifier_prepared.eval_at_point::<D>(
        &x_challenges,
        &setup.expanded,
        std::slice::from_ref(&ring_opening_point),
        alpha,
    )?;
    let (warm_instrumented_value, _) = prepared.timed_eval_at_point::<D>(
        &x_challenges,
        &setup.expanded,
        std::slice::from_ref(&ring_opening_point),
        alpha,
    )?;
    assert_eq!(warm_instrumented_value, warm_verifier_value);

    let mut verifier_value = F::zero();
    let verifier_eval_start = Instant::now();
    for _ in 0..iterations {
        verifier_value = verifier_prepared.eval_at_point::<D>(
            &x_challenges,
            &setup.expanded,
            std::slice::from_ref(&ring_opening_point),
            alpha,
        )?;
    }
    let verifier_eval_avg = verifier_eval_start.elapsed() / iterations as u32;

    let mut total_timings = ComponentTimings::default();
    let mut instrumented_value = F::zero();
    for _ in 0..iterations {
        let (value, timings) = prepared.timed_eval_at_point::<D>(
            &x_challenges,
            &setup.expanded,
            std::slice::from_ref(&ring_opening_point),
            alpha,
        )?;
        instrumented_value = value;
        total_timings.add_assign(timings);
    }
    assert_eq!(
        instrumented_value, verifier_value,
        "instrumented evaluator must match PreparedMEval::eval_at_point"
    );

    let avg = total_timings.div(iterations as u32);
    println!(
        "\nreal PreparedMEval::eval_at_point avg: {:.3} ms",
        verifier_eval_avg.as_secs_f64() * 1e3
    );
    print_timings(avg);

    if check_materialized {
        let materialized_start = Instant::now();
        let alpha_evals_y = scalar_powers(alpha, D);
        let m_evals_x = compute_m_evals_x::<F, D>(
            &setup.expanded,
            std::slice::from_ref(&ring_opening_point),
            &[0usize],
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &layout,
            &tau1,
            &[1usize],
            &gamma,
            1,
        )?;
        let expected = multilinear_eval(&m_evals_x, &x_challenges)?;
        let materialized_elapsed = materialized_start.elapsed();
        assert_eq!(expected, verifier_value);
        println!(
            "\nmaterialized check: OK, table_len={}, elapsed={:.3} ms",
            m_evals_x.len(),
            materialized_elapsed.as_secs_f64() * 1e3
        );
    } else {
        println!(
            "\nmaterialized check skipped; set AKITA_PLAYGROUND_CHECK_MATERIALIZED=1 to enable it"
        );
    }

    Ok(())
}
