//! Ring switching logic for the Hachi PCS (Section 4.3).
//!
//! Handles the transition from the ring-based quadratic equation to field-based
//! sumcheck instances by expanding the ring elements into their coefficient
//! vectors and setting up the evaluation tables.

use crate::algebra::{CyclotomicRing, SparseChallenge};
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::linear::{
    decompose_rows_i8, flatten_i8_blocks, mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_i8,
    mat_vec_mul_ntt_single_i8,
};
use crate::protocol::commitment::utils::norm::detect_field_modulus;
use crate::protocol::commitment::{
    optimal_m_r_split, CommitmentConfig, DecompositionParams, HachiCommitmentLayout,
    HachiExpandedSetup, RingCommitment,
};
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::proof::{DigitLut, FlatCommitmentHint, FlatRingVec, HachiCommitmentHint};
use crate::protocol::quadratic_equation::{compute_r_split_eq, QuadraticEquation};
use crate::protocol::sumcheck::eq_poly::EqPolynomial;
use crate::protocol::transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use crate::protocol::transcript::Transcript;
use crate::{cfg_into_iter, cfg_iter};
use crate::{CanonicalField, FieldCore, FieldSampling};
#[cfg(test)]
use std::array::from_fn;
use std::marker::PhantomData;
use std::time::Instant;

/// D-agnostic output of the ring switch protocol, containing everything
/// needed for sumchecks and level chaining.
pub struct RingSwitchOutput<F: FieldCore> {
    /// The witness vector w as balanced digits in `[-b/2, b/2)`.
    pub w: Vec<i8>,
    /// D-erased commitment to w.
    pub w_commitment: FlatRingVec<F>,
    /// D-erased prover hint for the w-commitment.
    pub w_hint: FlatCommitmentHint,
    /// Compact evaluation table of w (all entries in [-b/2, b/2), reordered for sumcheck).
    /// Populated by the prover; empty on the verifier side.
    pub w_evals: Vec<i8>,
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

/// Build the witness vector `w` from the quadratic equation state.
///
/// This is the first half of the ring switch: it computes `r` and assembles
/// `w` as a flat `Vec<i8>`. The resulting `w` is D-agnostic and can be
/// committed at any ring dimension via [`commit_w`].
///
/// # Errors
///
/// Returns an error if the quadratic equation is missing prover-side data.
#[tracing::instrument(skip_all, name = "ring_switch_build_w")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_build_w<F, const D: usize, Cfg>(
    quad_eq: &mut QuadraticEquation<F, D, Cfg>,
    setup: &HachiExpandedSetup<F>,
    ntt_a: &NttSlotCache<D>,
    ntt_b: &NttSlotCache<D>,
    ntt_d: &NttSlotCache<D>,
    layout: HachiCommitmentLayout,
) -> Result<Vec<i8>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    {
        let x: u8 = 0;
        eprintln!(
            "  [ring_switch_build_w] stack ~= {:#x}",
            &x as *const u8 as usize
        );
    }
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
    let t = hint.t().ok_or_else(|| {
        HachiError::InvalidInput("missing recomposed t in prover hint".to_string())
    })?;
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
        t,
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
    Ok(w)
}

/// Complete the ring switch after `w` has been committed.
///
/// Takes the already-committed `w` (with its D-erased commitment and hint)
/// and finishes the protocol: absorbs the commitment into the transcript,
/// samples challenges, and builds the evaluation tables for the fused sumcheck.
///
/// Only the current level's `D` is needed (for M_alpha expansion and
/// alpha_evals_y). The commitment's ring dimension is encoded in the
/// `FlatRingVec` and does not require a separate const generic.
///
/// # Errors
///
/// Returns an error if matrix expansion or evaluation-table construction fails.
#[tracing::instrument(skip_all, name = "ring_switch_finalize")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_finalize<F, T, const D: usize, Cfg>(
    quad_eq: &QuadraticEquation<F, D, Cfg>,
    setup: &HachiExpandedSetup<F>,
    transcript: &mut T,
    w: Vec<i8>,
    w_commitment: FlatRingVec<F>,
    w_hint: FlatCommitmentHint,
    layout: HachiCommitmentLayout,
) -> Result<RingSwitchOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
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
            compute_m_evals_x::<F, D, Cfg>(
                setup,
                opening_point,
                challenges,
                alpha,
                &alpha_evals_y,
                layout,
                &tau1,
            )
        },
        || build_w_evals_compact(&w, D),
    );
    #[cfg(not(feature = "parallel"))]
    let (m_evals_x_result, w_result) = {
        let m_evals_x = compute_m_evals_x::<F, D, Cfg>(
            setup,
            opening_point,
            challenges,
            alpha,
            &alpha_evals_y,
            layout,
            &tau1,
        )?;
        let w_compact = build_w_evals_compact(&w, D);
        (Ok(m_evals_x), w_compact)
    };

    let m_evals_x = m_evals_x_result?;
    let (w_evals, _, _) = w_result?;
    eprintln!(
        "    [ring_switch] m_evals_x+w_evals parallel: {:.2}s",
        t_par.elapsed().as_secs_f64()
    );

    Ok(RingSwitchOutput {
        w,
        w_commitment,
        w_hint,
        w_evals,
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

/// Execute the prover side of the ring switching protocol (Section 4.3).
///
/// Convenience wrapper that calls [`ring_switch_build_w`], [`commit_w`], and
/// [`ring_switch_finalize`] in sequence, all at the same ring dimension `D`.
///
/// # Errors
///
/// Returns an error if z_pre/w_hat is missing, commitment fails, or matrix expansion fails.
#[tracing::instrument(skip_all, name = "ring_switch_prover")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_prover<F, T, const D: usize, Cfg>(
    quad_eq: &mut QuadraticEquation<F, D, Cfg>,
    setup: &HachiExpandedSetup<F>,
    transcript: &mut T,
    ntt_a: &NttSlotCache<D>,
    ntt_b: &NttSlotCache<D>,
    ntt_d: &NttSlotCache<D>,
    layout: HachiCommitmentLayout,
) -> Result<RingSwitchOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let w = ring_switch_build_w::<F, D, Cfg>(quad_eq, setup, ntt_a, ntt_b, ntt_d, layout)?;

    let t_cw = Instant::now();
    let (w_commitment, w_hint) = commit_w::<F, D, Cfg>(&w, ntt_a, ntt_b)?;
    eprintln!(
        "    [ring_switch] commit_w: {:.2}s (w_len={})",
        t_cw.elapsed().as_secs_f64(),
        w.len()
    );

    let w_commitment_flat = FlatRingVec::from_commitment(&w_commitment);
    let w_hint_flat = FlatCommitmentHint::from_typed(w_hint);

    ring_switch_finalize::<F, T, D, Cfg>(
        quad_eq,
        setup,
        transcript,
        w,
        w_commitment_flat,
        w_hint_flat,
        layout,
    )
}

/// Replay the verifier side of ring switching to reconstruct evaluation tables.
///
/// Takes the w-commitment as a [`FlatRingVec`] so the verifier does not need
/// to know D_COMMIT (the commitment's ring dimension).
///
/// # Errors
///
/// Returns an error if matrix expansion fails.
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub fn ring_switch_verifier<F, T, const D: usize, Cfg>(
    quad_eq: &QuadraticEquation<F, D, Cfg>,
    setup: &HachiExpandedSetup<F>,
    w_len: usize,
    w_commitment: &FlatRingVec<F>,
    transcript: &mut T,
    layout: HachiCommitmentLayout,
) -> Result<RingSwitchOutput<F>, HachiError>
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

    let m_evals_x = compute_m_evals_x::<F, D, Cfg>(
        setup,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        alpha,
        &alpha_evals_y,
        layout,
        &tau1,
    )?;

    Ok(RingSwitchOutput {
        w: Vec::new(),
        w_commitment: w_commitment.clone(),
        w_hint: FlatCommitmentHint::empty(),
        w_evals: Vec::new(),
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
                                local[t + s] += a[t] * b[s];
                            }
                        }
                    }
                    local
                };

            let pointwise_add = |mut a: Vec<F>, b: Vec<F>| -> Vec<F> {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai += *bi;
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
                poly[k] -= y_coeffs[k];
            }
            let mut quotient = vec![F::zero(); D];
            for k in (D..poly_len).rev() {
                let q = poly[k];
                quotient[k - D] = q;
                poly[k - D] -= q;
            }
            let coeffs: [F; D] = from_fn(|k| quotient[k]);
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();
    Ok(out)
}

/// Derived commitment config for recursive w-openings.
///
/// Sets `log_commit_bound = log_basis` (w's entries are balanced digits) and
/// `log_open_bound = parent's open bound` (opening folds produce full-field
/// coefficients).
///
/// For `D=512, Cfg=Fp128FullCommitmentConfig`, this is equivalent to
/// [`Fp128LogBasisCommitmentConfig`](super::commitment::Fp128LogBasisCommitmentConfig).
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

    fn challenge_weight_for_ring_dim(d: usize) -> usize {
        Cfg::challenge_weight_for_ring_dim(d)
    }

    fn w_log_basis() -> u32 {
        Cfg::w_log_basis()
    }

    fn decomposition() -> DecompositionParams {
        let parent = Cfg::decomposition();
        let w_basis = Cfg::w_log_basis();
        let parent_open = parent.log_open_bound.unwrap_or(parent.log_commit_bound);
        DecompositionParams {
            log_basis: w_basis,
            // w entries come from a balanced decomposition; use w_basis for
            // the commit bound since that's the widest digit range at any
            // recursive level (level-0 entries fit in parent.log_basis <= w_basis).
            log_commit_bound: w_basis,
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
        let (m_vars, r_vars) = optimal_m_r_split::<Self>(reduced_vars);
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
    let t_hat_count = layout.num_blocks * Cfg::N_A * layout.num_digits_open;
    let z_pre_count = layout.inner_width * layout.num_digits_fold;
    let r_count = m_row_count::<Cfg>() * r_decomp_levels::<F>(layout.log_basis);
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

/// Commit the witness vector `w` (D-agnostic `Vec<i8>`) into `D`-sized ring
/// elements and compute the ring commitment.
///
/// This is the **D-boundary** in the protocol: the ring switch at level k
/// produces `w` using D_k operations, but `commit_w` re-chunks `w` into
/// D_{k+1}-sized ring elements and commits using D_{k+1} NTT caches.
///
/// For constant-D configs, D_k = D_{k+1} = D and the distinction is moot.
///
/// # Errors
///
/// Returns an error if the commitment layout derivation or NTT mat-vec fails.
#[tracing::instrument(skip_all, name = "commit_w")]
#[inline(never)]
pub fn commit_w<F, const D: usize, Cfg>(
    w: &[i8],
    ntt_a: &NttSlotCache<D>,
    ntt_b: &NttSlotCache<D>,
) -> Result<(RingCommitment<F, D>, HachiCommitmentHint<F, D>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    let (w_digits, remainder) = w.as_chunks::<D>();
    if !remainder.is_empty() {
        return Err(HachiError::InvalidSize {
            expected: D,
            actual: w.len(),
        });
    }

    let total = w_digits.len().next_power_of_two().max(1);
    let alpha = D.trailing_zeros() as usize;
    let m_vars_total = total.trailing_zeros() as usize;
    let max_num_vars = m_vars_total + alpha;
    let w_layout = WCommitmentConfig::<D, Cfg>::commitment_layout(max_num_vars)?;

    let num_blocks = w_layout.num_blocks;
    let block_len = w_layout.block_len;
    let depth_commit = w_layout.num_digits_commit;
    let depth_open = w_layout.num_digits_open;
    let log_basis = w_layout.log_basis;
    let coeff_len = w_digits.len();

    let t_all = if depth_commit == 1 {
        // `build_w_coeffs` already emits balanced base-`2^log_basis` digits, so
        // the recursive w-commitment can skip the field conversion and feed those
        // planes directly into the tiled NTT mat-vec.
        let block_slices: Vec<&[[i8; D]]> = (0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                if start >= coeff_len {
                    &[] as &[[i8; D]]
                } else {
                    &w_digits[start..(start + block_len).min(coeff_len)]
                }
            })
            .collect();
        mat_vec_mul_ntt_digits_i8(ntt_a, &block_slices)
    } else {
        let lut = DigitLut::<F>::new(log_basis);
        let ring_elems: Vec<CyclotomicRing<F, D>> = w_digits
            .iter()
            .map(|digit| {
                let coeffs = std::array::from_fn(|k| lut.get(digit[k]));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();
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
        mat_vec_mul_ntt_i8(ntt_a, &block_slices, depth_commit, log_basis)
    };
    let t_hat_per_block: Vec<Vec<[i8; D]>> = cfg_iter!(t_all)
        .map(|t_i| decompose_rows_i8(t_i, depth_open, log_basis))
        .collect();

    let t_hat_flat = flatten_i8_blocks(&t_hat_per_block);
    let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(ntt_b, &t_hat_flat);
    let hint = HachiCommitmentHint::with_t(t_hat_per_block, t_all);
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

#[inline]
fn eval_ring_at_pows<F: FieldCore, const D: usize>(
    r: &CyclotomicRing<F, D>,
    alpha_pows: &[F],
) -> F {
    debug_assert_eq!(alpha_pows.len(), D);
    r.coefficients()
        .iter()
        .zip(alpha_pows.iter())
        .fold(F::zero(), |acc, (coeff, alpha_pow)| {
            acc + *coeff * *alpha_pow
        })
}

#[inline]
fn eval_sparse_challenge_at_pows<F: FieldCore + CanonicalField, const D: usize>(
    challenge: &SparseChallenge,
    alpha_pows: &[F],
) -> Result<F, HachiError> {
    if alpha_pows.len() != D {
        return Err(HachiError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }

    debug_assert_eq!(challenge.positions.len(), challenge.coeffs.len());

    let mut acc = F::zero();
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let idx = pos as usize;
        debug_assert!(idx < D);
        debug_assert_ne!(coeff, 0);
        acc += F::from_i64(coeff as i64) * alpha_pows[idx];
    }
    Ok(acc)
}

#[inline]
fn gadget_row_scalars<F: FieldCore + CanonicalField>(levels: usize, log_basis: u32) -> Vec<F> {
    let base = F::from_canonical_u128_reduced(1u128 << log_basis);
    let mut out = Vec::with_capacity(levels);
    let mut power = F::one();
    for _ in 0..levels {
        out.push(power);
        power = power * base;
    }
    out
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

/// Produce the compact `Vec<i8>` eval table of `w` for the fused prover.
pub(crate) fn build_w_evals_compact(
    w: &[i8],
    d: usize,
) -> Result<(Vec<i8>, usize, usize), HachiError> {
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

    let compact: Vec<i8> = cfg_into_iter!(0..n)
        .map(|dst| {
            let x = dst & (x_len - 1);
            let y = dst >> num_u;
            let src = y + (x << num_l);
            if src < w.len() {
                w[src]
            } else {
                0i8
            }
        })
        .collect();
    Ok((compact, num_u, num_l))
}

pub(crate) fn m_row_count<Cfg: CommitmentConfig>() -> usize {
    Cfg::N_D + Cfg::N_B + 1 + 1 + Cfg::N_A
}

pub(crate) fn compute_m_evals_x<F: FieldCore + CanonicalField, const D: usize, Cfg>(
    setup: &HachiExpandedSetup<F>,
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    alpha: F,
    alpha_pows: &[F],
    layout: HachiCommitmentLayout,
    tau1: &[F],
) -> Result<Vec<F>, HachiError>
where
    Cfg: CommitmentConfig,
{
    if alpha_pows.len() != D {
        return Err(HachiError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }

    let depth_commit = layout.num_digits_commit;
    let depth_open = layout.num_digits_open;
    let depth_fold = layout.num_digits_fold;
    let log_basis = layout.log_basis;
    let num_blocks = opening_point.b.len();
    let block_len = layout.block_len;
    let w_len = depth_open * num_blocks;
    let t_len = depth_open * Cfg::N_A * num_blocks;
    let inner_width = block_len * depth_commit;
    let z_len = depth_fold * inner_width;
    let rows = m_row_count::<Cfg>();
    let levels = r_decomp_levels::<F>(log_basis);
    let total_cols = w_len
        .checked_add(t_len)
        .and_then(|cols| cols.checked_add(z_len))
        .and_then(|cols| cols.checked_add(rows.checked_mul(levels)?))
        .ok_or_else(|| HachiError::InvalidSetup("expanded M width overflow".to_string()))?;

    let eq_tau1 = EqPolynomial::evals(tau1);
    if eq_tau1.len() < rows {
        return Err(HachiError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let g1_open = gadget_row_scalars::<F>(depth_open, log_basis);
    let g1_commit = gadget_row_scalars::<F>(depth_commit, log_basis);
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let r_gadget = gadget_row_scalars::<F>(levels, log_basis);
    let x_len = total_cols.next_power_of_two();
    let mut out = Vec::with_capacity(x_len);

    let c_alphas: Vec<F> = challenges
        .iter()
        .map(|challenge| eval_sparse_challenge_at_pows::<F, D>(challenge, alpha_pows))
        .collect::<Result<_, _>>()?;

    let d_view = setup.D_mat.view::<D>();
    let b_view = setup.B.view::<D>();
    let a_view = setup.A.view::<D>();

    let row3_weight = eq_tau1[Cfg::N_D + Cfg::N_B];
    let row4_weight = eq_tau1[Cfg::N_D + Cfg::N_B + 1];
    let a_weights = &eq_tau1[(Cfg::N_D + Cfg::N_B + 2)..rows];

    let w_segment: Vec<F> = cfg_into_iter!(0..w_len)
        .map(|x| {
            let block_idx = x / depth_open;
            let digit_idx = x % depth_open;
            let mut acc = (row3_weight * opening_point.b[block_idx]
                + row4_weight * c_alphas[block_idx])
                * g1_open[digit_idx];
            for (row_idx, eq_i) in eq_tau1.iter().enumerate().take(Cfg::N_D) {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&d_view.row(row_idx)[x], alpha_pows);
                }
            }
            acc
        })
        .collect();
    out.extend(w_segment);

    let t_segment: Vec<F> = cfg_into_iter!(0..t_len)
        .map(|x| {
            let block_idx = x / (Cfg::N_A * depth_open);
            let rem = x % (Cfg::N_A * depth_open);
            let a_idx = rem / depth_open;
            let digit_idx = rem % depth_open;
            let mut acc = a_weights[a_idx] * c_alphas[block_idx] * g1_open[digit_idx];
            for (row_idx, eq_i) in eq_tau1[Cfg::N_D..(Cfg::N_D + Cfg::N_B)].iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&b_view.row(row_idx)[x], alpha_pows);
                }
            }
            acc
        })
        .collect();
    out.extend(t_segment);

    let z_base: Vec<F> = cfg_into_iter!(0..inner_width)
        .map(|k| {
            let block_idx = k / depth_commit;
            let digit_idx = k % depth_commit;
            let mut acc = row4_weight * opening_point.a[block_idx] * g1_commit[digit_idx];
            for (a_idx, eq_i) in a_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&a_view.row(a_idx)[k], alpha_pows);
                }
            }
            acc
        })
        .collect();

    let z_segment: Vec<F> = cfg_into_iter!(0..z_len)
        .map(|idx| {
            let k = idx / depth_fold;
            let fold_idx = idx % depth_fold;
            -(z_base[k] * fold_gadget[fold_idx])
        })
        .collect();
    out.extend(z_segment);

    let alpha_pow_d = alpha_pows[D - 1] * alpha;
    let denom = alpha_pow_d + F::one();
    let r_tail_len = rows * levels;
    let r_tail: Vec<F> = cfg_into_iter!(0..r_tail_len)
        .map(|idx| {
            let row_idx = idx / levels;
            let level_idx = idx % levels;
            -(eq_tau1[row_idx] * denom * r_gadget[level_idx])
        })
        .collect();
    out.extend(r_tail);
    out.resize(x_len, F::zero());
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
) -> Vec<i8> {
    let log_basis = layout.log_basis;
    let num_digits_fold = layout.num_digits_fold;
    let levels = r_decomp_levels::<F>(log_basis);

    let t_hat_flat = t_hat.iter().flat_map(|v| v.iter());

    let w_hat_planes: usize = w_hat.iter().map(|v| v.len()).sum();
    let t_hat_planes: usize = t_hat.iter().map(|v| v.len()).sum();
    let z_count = w_hat_planes + t_hat_planes + z_pre.len() * num_digits_fold;
    let r_hat_count = r.len() * levels;
    eprintln!(
        "    [build_w_coeffs] w_hat_planes={w_hat_planes}, t_hat_planes={t_hat_planes}, z_pre_elems={}, z_pre_planes={}, r_elems={}, r_planes={r_hat_count}, total_ring={}, total_field={}",
        z_pre.len(), z_pre.len() * num_digits_fold, r.len(), z_count + r_hat_count, (z_count + r_hat_count) * D,
    );
    let mut out = Vec::with_capacity((z_count + r_hat_count) * D);
    let mut digit_scratch = vec![[0i8; D]; num_digits_fold.max(levels)];
    for block in w_hat {
        for digits in block {
            out.extend_from_slice(digits);
        }
    }
    for digits in t_hat_flat {
        out.extend_from_slice(digits);
    }
    for z_j in z_pre {
        let z_planes = &mut digit_scratch[..num_digits_fold];
        z_j.balanced_decompose_pow2_i8_into(z_planes, log_basis);
        for plane in z_planes.iter() {
            out.extend_from_slice(plane);
        }
    }
    for ri in r {
        let r_planes = &mut digit_scratch[..levels];
        ri.balanced_decompose_pow2_i8_into(r_planes, log_basis);
        for plane in r_planes.iter() {
            out.extend_from_slice(plane);
        }
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
                            poly[s] += scalar * b[s];
                        }
                    } else {
                        for t in 0..D {
                            for s in 0..D {
                                poly[t + s] += a[t] * b[s];
                            }
                        }
                    }
                }
                let y_coeffs = y_i.coefficients();
                for k in 0..D {
                    poly[k] -= y_coeffs[k];
                }
                let mut quotient = vec![F::zero(); D];
                for k in (D..poly_len).rev() {
                    let q = poly[k];
                    quotient[k - D] = q;
                    poly[k - D] -= q;
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
