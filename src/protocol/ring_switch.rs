//! Ring switching logic for the Hachi PCS (Section 4.3).
//!
//! Handles the transition from the ring-based quadratic equation to field-based
//! sumcheck instances by expanding the ring elements into their coefficient
//! vectors and setting up the evaluation tables.

use crate::algebra::CyclotomicRing;
use crate::cfg_into_iter;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::linear::{
    decompose_rows_i8, flatten_i8_blocks, mat_vec_mul_ntt_i8, mat_vec_mul_ntt_single_i8,
};
use crate::protocol::commitment::utils::norm::detect_field_modulus;
use crate::protocol::commitment::{
    CommitmentConfig, DecompositionParams, HachiCommitmentLayout, HachiExpandedSetup,
    RingCommitment,
};
use crate::protocol::proof::HachiCommitmentHint;
use crate::protocol::quadratic_equation::{
    compute_m_a_streaming, compute_r_split_eq, QuadraticEquation,
};
use crate::protocol::sumcheck::eq_poly::EqPolynomial;
use crate::protocol::transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling};
#[cfg(test)]
use std::array::from_fn;
use std::marker::PhantomData;
use std::time::Instant;

/// Output of the ring switch protocol, containing everything needed for sumchecks.
pub struct RingSwitchOutput<F: FieldCore, const D: usize> {
    /// The witness vector w (concatenation of z and r coefficients).
    pub w: Vec<F>,
    /// Commitment to w.
    pub w_commitment: RingCommitment<F, D>,
    /// Prover hint for the w-commitment (t_hat blocks), needed for recursive opening.
    pub w_hint: HachiCommitmentHint<F, D>,
    /// Compact evaluation table of w (all entries in [-b/2, b/2), reordered for sumcheck).
    /// Populated by the prover; empty on the verifier side.
    pub w_evals: Vec<i8>,
    /// Field-element evaluation table of w (same reordering as `w_evals`).
    /// Produced alongside `w_evals` in a single pass to avoid a duplicate scan.
    pub w_evals_field: Vec<F>,
    /// Evaluation table of M_alpha(x) (tau1-weighted).
    pub m_evals_x: Vec<F>,
    /// Evaluation table of alpha powers (y dimension).
    pub alpha_evals_y: Vec<F>,
    /// Number of upper variable bits.
    pub num_u: usize,
    /// Number of lower variable bits.
    pub num_l: usize,
    /// Challenge tau0 for F_0 sumcheck.
    pub tau0: Vec<F>,
    /// Challenge tau1 for F_alpha sumcheck.
    pub tau1: Vec<F>,
    /// Basis size b = 2^LOG_BASIS.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: F,
}

/// Execute the prover side of the ring switching protocol (Section 4.3).
///
/// # Errors
///
/// Returns an error if z_pre/w_hat is missing, commitment fails, or matrix expansion fails.
#[tracing::instrument(skip_all, name = "ring_switch_prover")]
#[allow(clippy::too_many_arguments)]
pub fn ring_switch_prover<F, T, const D: usize, Cfg>(
    quad_eq: &mut QuadraticEquation<F, D, Cfg>,
    setup: &HachiExpandedSetup<F, D>,
    transcript: &mut T,
    ntt_a: &NttSlotCache<D>,
    ntt_b: &NttSlotCache<D>,
    ntt_d: &NttSlotCache<D>,
    layout: HachiCommitmentLayout,
) -> Result<RingSwitchOutput<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let w_hat = quad_eq
        .w_hat()
        .ok_or_else(|| HachiError::InvalidInput("missing w_hat in prover".to_string()))?;
    let w_hat_flat = quad_eq
        .w_hat_flat()
        .ok_or_else(|| HachiError::InvalidInput("missing w_hat_flat in prover".to_string()))?;
    let z_pre = quad_eq
        .z_pre()
        .ok_or_else(|| HachiError::InvalidInput("missing z_pre in prover".to_string()))?;
    let hint = quad_eq
        .hint()
        .ok_or_else(|| HachiError::InvalidInput("missing hint in prover".to_string()))?;
    let t_hat = &hint.t_hat;
    let w_folded = quad_eq
        .w_folded()
        .ok_or_else(|| HachiError::InvalidInput("missing w_folded in prover".to_string()))?;

    let t_rs = Instant::now();
    let r = compute_r_split_eq::<F, D, Cfg>(
        setup,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        w_hat_flat,
        t_hat,
        w_folded,
        z_pre,
        quad_eq.y(),
        ntt_a,
        ntt_b,
        ntt_d,
        layout,
    )?;
    eprintln!(
        "    [ring_switch] compute_r_split_eq: {:.2}s",
        t_rs.elapsed().as_secs_f64()
    );
    let t_wc = Instant::now();
    let w = {
        let _span = tracing::info_span!("build_w_coeffs").entered();
        build_w_coeffs::<F, D>(w_hat, t_hat, z_pre, &r, layout)
    };
    eprintln!(
        "    [ring_switch] build_w_coeffs: {:.2}s",
        t_wc.elapsed().as_secs_f64()
    );

    let t_cw = Instant::now();
    let (w_commitment, w_hint) = commit_w::<F, D, Cfg>(&w, ntt_a, ntt_b)?;
    eprintln!(
        "    [ring_switch] commit_w: {:.2}s (w_len={})",
        t_cw.elapsed().as_secs_f64(),
        w.len()
    );
    transcript.append_serde(ABSORB_SUMCHECK_W, &w_commitment);

    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

    let num_l = D.trailing_zeros() as usize;
    let num_ring_elems = w.len() / D;
    let num_u = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let m_rows = m_row_count::<Cfg>();
    let num_sc_vars = num_u + num_l;
    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;

    let tau0 = sample_tau::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);
    let tau1 = sample_tau::<F, T>(transcript, CHALLENGE_TAU1, num_i);
    let alpha_evals_y = build_alpha_evals_y(alpha, D);

    let t_par = Instant::now();
    let opening_point = quad_eq.opening_point();
    let challenges = &quad_eq.challenges;

    #[cfg(feature = "parallel")]
    let (m_evals_x_result, w_result) = rayon::join(
        || {
            let m_a = compute_m_a_streaming::<F, D, Cfg>(
                setup,
                opening_point,
                challenges,
                &alpha,
                layout,
            )?;
            build_m_evals_x_fused::<F, D>(&m_a, alpha, layout.log_basis, &tau1)
        },
        || build_w_evals_dual::<F>(&w, D),
    );
    #[cfg(not(feature = "parallel"))]
    let (m_evals_x_result, w_result) = {
        let m_a =
            compute_m_a_streaming::<F, D, Cfg>(setup, opening_point, challenges, &alpha, layout)?;
        let m_evals_x = build_m_evals_x_fused::<F, D>(&m_a, alpha, layout.log_basis, &tau1)?;
        let w_dual = build_w_evals_dual::<F>(&w, D);
        (Ok(m_evals_x), w_dual)
    };

    let m_evals_x = m_evals_x_result?;
    let (w_evals, w_evals_field, _, _) = w_result?;
    eprintln!(
        "    [ring_switch] m_a+w_evals parallel: {:.2}s",
        t_par.elapsed().as_secs_f64()
    );

    Ok(RingSwitchOutput {
        w,
        w_commitment,
        w_hint,
        w_evals,
        w_evals_field,
        m_evals_x,
        alpha_evals_y,
        num_u,
        num_l,
        tau0,
        tau1,
        b: 1usize << layout.log_basis,
        alpha,
    })
}

/// Replay the verifier side of ring switching to reconstruct evaluation tables.
///
/// # Errors
///
/// Returns an error if matrix expansion fails.
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
pub fn ring_switch_verifier<F, T, const D: usize, Cfg>(
    quad_eq: &QuadraticEquation<F, D, Cfg>,
    setup: &HachiExpandedSetup<F, D>,
    w_len: usize,
    w_commitment: &RingCommitment<F, D>,
    transcript: &mut T,
    layout: HachiCommitmentLayout,
) -> Result<RingSwitchOutput<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    transcript.append_serde(ABSORB_SUMCHECK_W, w_commitment);

    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

    let num_ring_elems = w_len / D;
    let num_u = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let num_l = D.trailing_zeros() as usize;
    let m_rows = m_row_count::<Cfg>();
    let num_sc_vars = num_u + num_l;
    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;

    let tau0 = sample_tau::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);
    let tau1 = sample_tau::<F, T>(transcript, CHALLENGE_TAU1, num_i);
    let alpha_evals_y = build_alpha_evals_y(alpha, D);

    let m_a = compute_m_a_streaming::<F, D, Cfg>(
        setup,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        &alpha,
        layout,
    )?;
    let m_evals_x = build_m_evals_x_fused::<F, D>(&m_a, alpha, layout.log_basis, &tau1)?;

    Ok(RingSwitchOutput {
        w: Vec::new(),
        w_commitment: w_commitment.clone(),
        w_hint: HachiCommitmentHint::new(Vec::new()),
        w_evals: Vec::new(),
        w_evals_field: Vec::new(),
        m_evals_x,
        alpha_evals_y,
        num_u,
        num_l,
        tau0,
        tau1,
        b: 1usize << layout.log_basis,
        alpha,
    })
}

#[cfg(test)]
pub(crate) fn compute_r_via_poly_division<F: FieldCore + CanonicalField, const D: usize>(
    m: &[Vec<CyclotomicRing<F, D>>],
    z: &[CyclotomicRing<F, D>],
    y: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    let poly_len = 2 * D - 1;
    let out = m
        .iter()
        .zip(y.iter())
        .map(|(row, y_i)| {
            let column_contribution =
                |m_ij: &CyclotomicRing<F, D>, z_j: &CyclotomicRing<F, D>| -> Vec<F> {
                    let mut local = vec![F::zero(); poly_len];
                    if m_ij.is_zero() {
                        return local;
                    }
                    let a = m_ij.coefficients();
                    let b = z_j.coefficients();
                    let is_scalar = a[1..].iter().all(|c| c.is_zero());
                    if is_scalar {
                        let scalar = a[0];
                        for s in 0..D {
                            local[s] = scalar * b[s];
                        }
                    } else {
                        for t in 0..D {
                            for s in 0..D {
                                local[t + s] = local[t + s] + a[t] * b[s];
                            }
                        }
                    }
                    local
                };

            let pointwise_add = |mut a: Vec<F>, b: Vec<F>| -> Vec<F> {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai = *ai + *bi;
                }
                a
            };

            #[cfg(feature = "parallel")]
            let mut poly = row
                .par_iter()
                .zip(z.par_iter())
                .fold(
                    || vec![F::zero(); poly_len],
                    |acc, (m_ij, z_j)| pointwise_add(acc, column_contribution(m_ij, z_j)),
                )
                .reduce(|| vec![F::zero(); poly_len], pointwise_add);

            #[cfg(not(feature = "parallel"))]
            let mut poly = row
                .iter()
                .zip(z.iter())
                .fold(vec![F::zero(); poly_len], |acc, (m_ij, z_j)| {
                    pointwise_add(acc, column_contribution(m_ij, z_j))
                });
            let y_coeffs = y_i.coefficients();
            for k in 0..D {
                poly[k] = poly[k] - y_coeffs[k];
            }
            let mut quotient = vec![F::zero(); D];
            for k in (D..poly_len).rev() {
                let q = poly[k];
                quotient[k - D] = q;
                poly[k - D] = poly[k - D] - q;
            }
            let coeffs: [F; D] = from_fn(|k| quotient[k]);
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct WCommitmentConfig<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

impl<const D: usize, Cfg: CommitmentConfig> CommitmentConfig for WCommitmentConfig<D, Cfg> {
    const D: usize = D;
    const N_A: usize = Cfg::N_A;
    const N_B: usize = Cfg::N_B;
    const N_D: usize = Cfg::N_D;
    const CHALLENGE_WEIGHT: usize = Cfg::CHALLENGE_WEIGHT;

    fn decomposition() -> DecompositionParams {
        let parent = Cfg::decomposition();
        let parent_open = parent.log_open_bound.unwrap_or(parent.log_commit_bound);
        DecompositionParams {
            log_basis: parent.log_basis,
            // w's entries are balanced digits in [-b/2, b/2), so commitment
            // decomposition needs only one level.
            log_commit_bound: parent.log_basis,
            // Opening folds w with arbitrary field-element weights, producing
            // full-field-size coefficients that need the same decomposition
            // depth as the parent's opening bound.
            log_open_bound: Some(parent_open),
        }
    }

    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        let alpha = D.trailing_zeros() as usize;
        let reduced_vars = max_num_vars.checked_sub(alpha).ok_or_else(|| {
            HachiError::InvalidSetup("max_num_vars is smaller than alpha".to_string())
        })?;
        if reduced_vars == 0 {
            return Err(HachiError::InvalidSetup(
                "max_num_vars must leave at least one outer variable".to_string(),
            ));
        }
        let r_vars = reduced_vars / 2;
        let m_vars = reduced_vars - r_vars;
        HachiCommitmentLayout::new::<Self>(m_vars, r_vars, &Self::decomposition())
    }
}

/// Total ring elements in the w polynomial, computed from the main layout.
///
/// Components: w_hat + t_hat + decomposed z_pre + decomposed r.
pub(crate) fn w_ring_element_count<F: CanonicalField, Cfg: CommitmentConfig>(
    layout: HachiCommitmentLayout,
) -> usize {
    let w_hat_count = layout.num_blocks * layout.num_digits_open;
    let t_hat_count = layout.num_blocks * Cfg::N_A * layout.num_digits_commit;
    let z_pre_count = layout.inner_width * layout.num_digits_fold;
    let r_count = (Cfg::N_D + Cfg::N_B + 2) * r_decomp_levels::<F>(layout.log_basis);
    w_hat_count + t_hat_count + z_pre_count + r_count
}

/// Compute the w-commitment layout from the main layout.
pub(crate) fn w_commitment_layout<F: CanonicalField, const D: usize, Cfg: CommitmentConfig>(
    main_layout: HachiCommitmentLayout,
) -> Result<HachiCommitmentLayout, HachiError> {
    let total = w_ring_element_count::<F, Cfg>(main_layout)
        .next_power_of_two()
        .max(1);
    let alpha = D.trailing_zeros() as usize;
    let m_vars = total.trailing_zeros() as usize;
    let max_num_vars = m_vars + alpha;
    WCommitmentConfig::<D, Cfg>::commitment_layout(max_num_vars)
}

#[tracing::instrument(skip_all, name = "commit_w")]
fn commit_w<F, const D: usize, Cfg>(
    w: &[F],
    ntt_a: &NttSlotCache<D>,
    ntt_b: &NttSlotCache<D>,
) -> Result<(RingCommitment<F, D>, HachiCommitmentHint<F, D>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    let ring_elems: Vec<CyclotomicRing<F, D>> = w
        .chunks(D)
        .map(|chunk| CyclotomicRing::from_slice(chunk))
        .collect();

    let total = ring_elems.len().next_power_of_two().max(1);
    let alpha = D.trailing_zeros() as usize;
    let m_vars_total = total.trailing_zeros() as usize;
    let max_num_vars = m_vars_total + alpha;
    let w_layout = WCommitmentConfig::<D, Cfg>::commitment_layout(max_num_vars)?;

    let num_blocks = w_layout.num_blocks;
    let block_len = w_layout.block_len;
    let depth = w_layout.num_digits_commit;
    let log_basis = w_layout.log_basis;
    let coeff_len = ring_elems.len();

    let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_blocks)
        .map(|i| {
            let start = i * block_len;
            if start >= coeff_len {
                &[] as &[CyclotomicRing<F, D>]
            } else {
                &ring_elems[start..(start + block_len).min(coeff_len)]
            }
        })
        .collect();

    let t_all = mat_vec_mul_ntt_i8(ntt_a, &block_slices, depth, log_basis);
    let t_hat_per_block: Vec<Vec<[i8; D]>> = cfg_into_iter!(t_all)
        .map(|t_i| decompose_rows_i8(&t_i, depth, log_basis))
        .collect();

    let t_hat_flat = flatten_i8_blocks(&t_hat_per_block);
    let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(ntt_b, &t_hat_flat);
    let hint = HachiCommitmentHint::new(t_hat_per_block);
    Ok((RingCommitment { u }, hint))
}

pub(crate) fn eval_ring_at<F: FieldCore, const D: usize>(r: &CyclotomicRing<F, D>, alpha: &F) -> F {
    let mut acc = F::zero();
    let mut power = F::one();
    for coeff in r.coefficients() {
        acc += *coeff * power;
        power = power * *alpha;
    }
    acc
}

pub(crate) fn r_decomp_levels<F: CanonicalField>(log_basis: u32) -> usize {
    let modulus = detect_field_modulus::<F>();
    let bits = 128 - (modulus.saturating_sub(1)).leading_zeros() as usize;
    let lb = log_basis as usize;
    let mut levels = (bits + lb.saturating_sub(1)) / lb.max(1);
    if levels == 0 {
        levels = 1;
    }

    let total_bits = levels * lb;
    if total_bits <= bits {
        let b = 1u128 << log_basis;
        let half_q = modulus / 2;
        let half_b_minus_1 = b / 2 - 1;
        let b_minus_1 = b - 1;
        let mut b_pow = 1u128;
        for _ in 0..levels {
            b_pow = b_pow.saturating_mul(b);
        }
        let max_positive = half_b_minus_1.saturating_mul((b_pow - 1) / b_minus_1);
        if max_positive < half_q {
            levels += 1;
        }
    }

    levels
}

#[cfg(test)]
pub(crate) fn expand_m_a<F: CanonicalField, const D: usize>(
    m_a: &[Vec<F>],
    alpha: F,
    log_basis: u32,
) -> Result<Vec<F>, HachiError> {
    if m_a.is_empty() {
        return Ok(Vec::new());
    }
    let rows = m_a.len();
    let cols = m_a[0].len();
    if cols == 0 {
        return Ok(vec![F::zero(); rows]);
    }
    for row in m_a.iter() {
        if row.len() != cols {
            return Err(HachiError::InvalidSize {
                expected: cols,
                actual: row.len(),
            });
        }
    }

    let levels = r_decomp_levels::<F>(log_basis);
    let total_cols = cols
        .checked_add(
            rows.checked_mul(levels)
                .ok_or_else(|| HachiError::InvalidSetup("expanded M width overflow".to_string()))?,
        )
        .ok_or_else(|| HachiError::InvalidSetup("expanded M width overflow".to_string()))?;

    let base = F::from_canonical_u128_reduced(1u128 << log_basis);
    let mut gadget_row = Vec::with_capacity(levels);
    let mut power = F::one();
    for _ in 0..levels {
        gadget_row.push(power);
        power = power * base;
    }

    let mut alpha_pow = F::one();
    for _ in 0..D {
        alpha_pow = alpha_pow * alpha;
    }
    let denom = alpha_pow + F::one();

    let mut out = vec![F::zero(); rows * total_cols];
    for (i, m_a_row) in m_a.iter().enumerate() {
        let row_start = i * total_cols;
        out[row_start..row_start + cols].copy_from_slice(m_a_row);
        let r_start = row_start + cols + i * levels;
        for (j, g) in gadget_row.iter().enumerate() {
            out[r_start + j] = -denom * *g;
        }
    }
    Ok(out)
}

/// # Errors
///
/// Returns an error if `w.len()` is not a multiple of `d`.
pub(crate) fn build_w_evals<F: FieldCore>(
    w: &[F],
    d: usize,
) -> Result<(Vec<F>, usize, usize), HachiError> {
    if d == 0 || w.len() % d != 0 {
        return Err(HachiError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let num_l = d.trailing_zeros() as usize;
    let num_ring_elems = w.len() / d;
    let num_u = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let x_len = 1usize << num_u;
    let n = x_len << num_l;

    let evals: Vec<F> = cfg_into_iter!(0..n)
        .map(|dst| {
            let x = dst & (x_len - 1);
            let y = dst >> num_u;
            let src = y + (x << num_l);
            if src < w.len() {
                w[src]
            } else {
                F::zero()
            }
        })
        .collect();
    Ok((evals, num_u, num_l))
}

/// Produce both compact `Vec<i8>` and field `Vec<F>` eval tables in one pass
/// over `w`, sharing the index computation.
pub(crate) fn build_w_evals_dual<F: FieldCore + CanonicalField>(
    w: &[F],
    d: usize,
) -> Result<(Vec<i8>, Vec<F>, usize, usize), HachiError> {
    if d == 0 || w.len() % d != 0 {
        return Err(HachiError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let num_l = d.trailing_zeros() as usize;
    let num_ring_elems = w.len() / d;
    let num_u = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let x_len = 1usize << num_u;
    let n = x_len << num_l;

    let q = (-F::one()).to_canonical_u128() + 1;
    let half_q = q / 2;

    let (compact, field): (Vec<i8>, Vec<F>) = cfg_into_iter!(0..n)
        .map(|dst| {
            let x = dst & (x_len - 1);
            let y = dst >> num_u;
            let src = y + (x << num_l);
            if src < w.len() {
                let val = w[src];
                let canonical = val.to_canonical_u128();
                let c = if canonical <= half_q {
                    canonical as i8
                } else {
                    (canonical as i128 - q as i128) as i8
                };
                (c, val)
            } else {
                (0i8, F::zero())
            }
        })
        .unzip();
    Ok((compact, field, num_u, num_l))
}

pub(crate) fn m_row_count<Cfg: CommitmentConfig>() -> usize {
    Cfg::N_D + Cfg::N_B + 1 + 1 + Cfg::N_A
}

pub(crate) fn build_m_evals_x_fused<F: FieldCore + CanonicalField, const D: usize>(
    m_a: &[Vec<F>],
    alpha: F,
    log_basis: u32,
    tau1: &[F],
) -> Result<Vec<F>, HachiError> {
    if m_a.is_empty() {
        return Ok(Vec::new());
    }
    let rows = m_a.len();
    let orig_cols = m_a[0].len();
    for row in m_a.iter() {
        if row.len() != orig_cols {
            return Err(HachiError::InvalidSize {
                expected: orig_cols,
                actual: row.len(),
            });
        }
    }

    let levels = r_decomp_levels::<F>(log_basis);
    let total_cols = orig_cols
        .checked_add(
            rows.checked_mul(levels)
                .ok_or_else(|| HachiError::InvalidSetup("expanded M width overflow".to_string()))?,
        )
        .ok_or_else(|| HachiError::InvalidSetup("expanded M width overflow".to_string()))?;

    let base = F::from_canonical_u128_reduced(1u128 << log_basis);
    let mut gadget_row = Vec::with_capacity(levels);
    let mut power = F::one();
    for _ in 0..levels {
        gadget_row.push(power);
        power = power * base;
    }

    let mut alpha_pow = F::one();
    for _ in 0..D {
        alpha_pow = alpha_pow * alpha;
    }
    let denom = alpha_pow + F::one();

    let eq_tau1 = EqPolynomial::evals(tau1);
    let x_len = total_cols.next_power_of_two();

    let out = cfg_into_iter!(0..x_len)
        .map(|x| {
            let mut acc = F::zero();
            if x < orig_cols {
                for (i, eq_i) in eq_tau1.iter().enumerate().take(rows) {
                    acc += *eq_i * m_a[i][x];
                }
            } else if x < total_cols {
                let offset = x - orig_cols;
                let i = offset / levels;
                let j = offset % levels;
                if i < rows && i < eq_tau1.len() {
                    acc = eq_tau1[i] * (-denom * gadget_row[j]);
                }
            }
            acc
        })
        .collect();
    Ok(out)
}

pub(crate) fn build_alpha_evals_y<F: FieldCore>(alpha: F, d: usize) -> Vec<F> {
    let mut out = vec![F::zero(); d];
    let mut power = F::one();
    for val in out.iter_mut() {
        *val = power;
        power = power * alpha;
    }
    out
}

pub(crate) fn sample_tau<F: FieldCore + CanonicalField, T: Transcript<F>>(
    transcript: &mut T,
    label: &[u8],
    n: usize,
) -> Vec<F> {
    (0..n).map(|_| transcript.challenge_scalar(label)).collect()
}

pub(crate) fn build_w_coeffs<F: CanonicalField, const D: usize>(
    w_hat: &[Vec<[i8; D]>],
    t_hat: &[Vec<[i8; D]>],
    z_pre: &[CyclotomicRing<F, D>],
    r: &[CyclotomicRing<F, D>],
    layout: HachiCommitmentLayout,
) -> Vec<F> {
    let log_basis = layout.log_basis;
    let num_digits_fold = layout.num_digits_fold;
    let levels = r_decomp_levels::<F>(log_basis);
    let r_hat: Vec<CyclotomicRing<F, D>> = r
        .iter()
        .flat_map(|ri| ri.balanced_decompose_pow2(levels, log_basis))
        .collect();

    let t_hat_flat = t_hat.iter().flat_map(|v| v.iter());

    let z_count = w_hat.iter().map(|v| v.len()).sum::<usize>()
        + t_hat.iter().map(|v| v.len()).sum::<usize>()
        + z_pre.len() * num_digits_fold;
    let mut out = Vec::with_capacity((z_count + r_hat.len()) * D);
    for block in w_hat {
        for digits in block {
            for &d in digits.iter() {
                out.push(F::from_i64(d as i64));
            }
        }
    }
    for digits in t_hat_flat {
        for &d in digits.iter() {
            out.push(F::from_i64(d as i64));
        }
    }
    for z_j in z_pre {
        for elem in z_j.balanced_decompose_pow2(num_digits_fold, log_basis) {
            out.extend_from_slice(elem.coefficients());
        }
    }
    for elem in &r_hat {
        out.extend_from_slice(elem.coefficients());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::compute_r_via_poly_division;
    use crate::algebra::{CyclotomicRing, Fp64};
    use std::array::from_fn;

    use crate::{FieldCore, FromSmallInt};

    fn compute_r_schoolbook<F: FieldCore, const D: usize>(
        m: &[Vec<CyclotomicRing<F, D>>],
        z: &[CyclotomicRing<F, D>],
        y: &[CyclotomicRing<F, D>],
    ) -> Vec<CyclotomicRing<F, D>> {
        let poly_len = 2 * D - 1;
        m.iter()
            .zip(y.iter())
            .map(|(row, y_i)| {
                let mut poly = vec![F::zero(); poly_len];
                for (m_ij, z_j) in row.iter().zip(z.iter()) {
                    if m_ij.is_zero() {
                        continue;
                    }
                    let a = m_ij.coefficients();
                    let b = z_j.coefficients();
                    let is_scalar = a[1..].iter().all(|c| c.is_zero());
                    if is_scalar {
                        let scalar = a[0];
                        for s in 0..D {
                            poly[s] = poly[s] + scalar * b[s];
                        }
                    } else {
                        for t in 0..D {
                            for s in 0..D {
                                poly[t + s] = poly[t + s] + a[t] * b[s];
                            }
                        }
                    }
                }
                let y_coeffs = y_i.coefficients();
                for k in 0..D {
                    poly[k] = poly[k] - y_coeffs[k];
                }
                let mut quotient = vec![F::zero(); D];
                for k in (D..poly_len).rev() {
                    let q = poly[k];
                    quotient[k - D] = q;
                    poly[k - D] = poly[k - D] - q;
                }
                let coeffs: [F; D] = from_fn(|k| quotient[k]);
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect()
    }

    #[test]
    fn compute_r_matches_schoolbook_reference() {
        type F = Fp64<4294967197>;
        const D: usize = 64;

        let m: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        if (i + j) % 3 == 0 {
                            let mut coeffs = [F::zero(); D];
                            coeffs[0] = F::from_u64((i * 5 + j + 1) as u64);
                            CyclotomicRing::from_coefficients(coeffs)
                        } else {
                            let coeffs = from_fn(|k| {
                                F::from_u64((i as u64 * 1000 + j as u64 * 100 + k as u64 + 1) % 97)
                            });
                            CyclotomicRing::from_coefficients(coeffs)
                        }
                    })
                    .collect()
            })
            .collect();
        let z: Vec<CyclotomicRing<F, D>> = (0..4)
            .map(|j| {
                let coeffs = from_fn(|k| F::from_u64((j as u64 * 37 + k as u64 + 5) % 89));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();
        let y: Vec<CyclotomicRing<F, D>> = (0..3)
            .map(|i| {
                let coeffs = from_fn(|k| F::from_u64((i as u64 * 29 + k as u64 + 7) % 83));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();

        let expected = compute_r_schoolbook(&m, &z, &y);
        let got = compute_r_via_poly_division::<F, D>(&m, &z, &y)
            .expect("ring-switch CRT+NTT path should dispatch for D=64");
        assert_eq!(got, expected);
    }
}
