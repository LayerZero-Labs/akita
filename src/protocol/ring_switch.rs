//! Ring switching logic for the Hachi PCS (Section 4.3).
//!
//! Handles the transition from the ring-based quadratic equation to field-based
//! sumcheck instances by expanding the ring elements into their coefficient
//! vectors and setting up the evaluation tables.

use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::flat_matrix::RingMatrixView;
use crate::protocol::commitment::utils::linear::mat_vec_mul_ntt_single_i8;
use crate::protocol::commitment::utils::norm::detect_field_modulus;
use crate::protocol::commitment::HachiRootBatchSummary;
use crate::protocol::commitment::{
    hachi_recursive_level_layout_from_params, recursive_level_decomposition_from_root,
    HachiScheduleInputs,
};
use crate::protocol::config::{CommitmentConfig, CommitmentEnvelope, DecompositionParams};
use crate::protocol::hachi_poly_ops::RecursiveWitnessFlat;
use crate::protocol::quadratic_equation::{compute_r_split_eq, QuadraticEquation};
use crate::protocol::recursive_runtime::RecursiveCommitmentHintCache;
use crate::protocol::setup::HachiExpandedSetup;
use crate::{CanonicalField, FieldCore, FieldSampling};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{
    eval_offset_eq_peeled_carry_terms, eval_offset_eq_tensor, summarize_pow2_block_carries,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::{CyclotomicRing, SparseChallenge};
use akita_field::parallel::*;
use akita_field::HachiError;
use akita_transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use akita_transcript::Transcript;
use akita_types::LevelParams;
use akita_types::RingOpeningPoint;
use akita_types::{FlatDigitBlocks, FlatRingVec, HachiCommitmentHint, RingCommitment};
#[cfg(test)]
use std::array::from_fn;
use std::marker::PhantomData;

/// D-agnostic output of the ring switch protocol, containing everything
/// needed for sumchecks and level chaining.
pub(crate) struct RingSwitchOutput<F: FieldCore> {
    /// The witness vector w as balanced digits in `[-b/2, b/2)`.
    pub w: RecursiveWitnessFlat,
    /// Runtime commitment to w (prover only).
    pub w_commitment: Option<FlatRingVec<F>>,
    /// Runtime-only prover hint cache for the w-commitment.
    pub w_hint: Option<RecursiveCommitmentHintCache<F>>,
    /// Compact evaluation table of w, stored as x-outer/y-inner slices.
    /// Populated by the prover; empty on the verifier side.
    pub w_evals_compact: Vec<i8>,
    /// Physical x width before zero-extension to the next power of two.
    pub live_x_cols: usize,
    /// Evaluation table of M_alpha(x) (tau1-weighted).
    pub m_evals_x: Vec<F>,
    /// Evaluation table of alpha powers (y dimension).
    pub alpha_evals_y: Vec<F>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
    /// Challenge tau0 for F_0 sumcheck.
    pub tau0: Vec<F>,
    /// Challenge tau1 for F_alpha sumcheck.
    pub tau1: Vec<F>,
    /// Basis size b = 2^LOG_BASIS.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: F,
}

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub(crate) struct RingSwitchVerifyOutput<F: FieldCore> {
    /// Prepared data for deferred M-table MLE evaluation.
    pub prepared_m_eval: PreparedMEval<F>,
    /// Evaluation table of alpha powers (y dimension).
    pub alpha_evals_y: Vec<F>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
    /// Challenge tau0 for F_0 sumcheck.
    pub tau0: Vec<F>,
    /// Challenge tau1 for F_alpha sumcheck.
    pub tau1: Vec<F>,
    /// Basis size b = 2^LOG_BASIS.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: F,
}

/// Pre-computed challenge-derived data for deferred M-table MLE evaluation.
///
/// Stores only data that cannot be derived from context at eval time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else (setup matrix views, opening point, gadget vectors) is
/// passed by reference at eval time to avoid duplication.
pub(crate) struct PreparedMEval<F: FieldCore> {
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

/// Build the witness vector `w` from the quadratic equation state.
///
/// This is the first half of the ring switch: it computes `r` and assembles
/// `w` as a flat recursive witness. The resulting `w` is D-agnostic and can be
/// committed at any ring dimension via [`commit_w`].
///
/// # Errors
///
/// Returns an error if the quadratic equation is missing prover-side data.
#[tracing::instrument(skip_all, name = "ring_switch_build_w")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn ring_switch_build_w<F, const D: usize, Cfg>(
    quad_eq: &mut QuadraticEquation<F, D, Cfg>,
    setup: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    lp: &LevelParams,
) -> Result<RecursiveWitnessFlat, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + crate::FromSmallInt,
    Cfg: CommitmentConfig<Field = F>,
{
    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            "ring_switch_build_w"
        );
    }
    let w_hat = quad_eq
        .take_w_hat()
        .ok_or_else(|| HachiError::InvalidInput("missing w_hat in prover".to_string()))?;
    let z_pre = quad_eq
        .take_z_pre()
        .ok_or_else(|| HachiError::InvalidInput("missing centered z_pre in prover".to_string()))?;
    let mut hint = quad_eq
        .take_hint()
        .ok_or_else(|| HachiError::InvalidInput("missing hint in prover".to_string()))?;
    hint.ensure_t_recomposed(lp.num_digits_open, lp.log_basis)?;
    let (inner_opening_digits, t) = hint.into_flat_parts();
    let t = t.ok_or_else(|| {
        HachiError::InvalidInput("missing recomposed t in prover hint".to_string())
    })?;
    let w_folded = quad_eq
        .take_w_folded()
        .ok_or_else(|| HachiError::InvalidInput("missing w_folded in prover".to_string()))?;

    let r = compute_r_split_eq::<F, D>(
        lp,
        setup,
        &quad_eq.challenges,
        w_hat.flat_digits(),
        &inner_opening_digits,
        &t,
        &w_folded,
        &z_pre.centered_coeffs,
        z_pre.centered_inf_norm,
        quad_eq.y(),
        quad_eq.claim_group_sizes(),
        quad_eq.num_eval_rows(),
        lp.num_blocks,
        lp.inner_width(),
        setup.seed.max_stride,
        ntt_shared,
    )?;
    let w = {
        let _span = tracing::info_span!("build_w_coeffs").entered();
        build_w_coeffs::<F, D>(
            &w_hat,
            &inner_opening_digits,
            &z_pre.centered_coeffs,
            &r,
            lp,
        )
    };
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
pub(crate) fn ring_switch_finalize<F, T, const D: usize, Cfg>(
    quad_eq: &QuadraticEquation<F, D, Cfg>,
    setup: &HachiExpandedSetup<F>,
    transcript: &mut T,
    w: RecursiveWitnessFlat,
    w_commitment: FlatRingVec<F>,
    w_commitment_proof: &FlatRingVec<F>,
    w_hint: RecursiveCommitmentHintCache<F>,
    lp: &LevelParams,
) -> Result<RingSwitchOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    ring_switch_finalize_with_claim_groups::<F, T, D, Cfg>(
        quad_eq,
        setup,
        transcript,
        w,
        w_commitment,
        w_commitment_proof,
        w_hint,
        lp,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn ring_switch_finalize_with_claim_groups<F, T, const D: usize, Cfg>(
    quad_eq: &QuadraticEquation<F, D, Cfg>,
    setup: &HachiExpandedSetup<F>,
    transcript: &mut T,
    w: RecursiveWitnessFlat,
    w_commitment: FlatRingVec<F>,
    w_commitment_proof: &FlatRingVec<F>,
    w_hint: RecursiveCommitmentHintCache<F>,
    lp: &LevelParams,
) -> Result<RingSwitchOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    transcript.append_serde(ABSORB_SUMCHECK_W, w_commitment_proof);

    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

    let claim_group_sizes = quad_eq.claim_group_sizes();
    let _num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    let num_commitment_groups = claim_group_sizes.len();
    let num_eval_rows = quad_eq.num_eval_rows();

    let ring_bits = D.trailing_zeros() as usize;
    let num_ring_elems = w.len() / D;
    let live_x_cols = num_ring_elems;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let m_rows = lp.m_row_count(num_commitment_groups, num_eval_rows);
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;

    let tau0 = sample_tau::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);
    let tau1 = sample_tau::<F, T>(transcript, CHALLENGE_TAU1, num_i);
    let alpha_evals_y = build_alpha_evals_y(alpha, D);

    let opening_points = quad_eq.opening_points();
    let claim_to_point = quad_eq.claim_to_point();
    let challenges = &quad_eq.challenges;

    let gamma = quad_eq.gamma();

    #[cfg(feature = "parallel")]
    let (m_evals_x_result, w_result) = rayon::join(
        || {
            compute_m_evals_x::<F, D>(
                setup,
                opening_points,
                claim_to_point,
                challenges,
                alpha,
                &alpha_evals_y,
                lp,
                &tau1,
                claim_group_sizes,
                gamma,
                num_eval_rows,
            )
        },
        || build_w_evals_compact(w.as_i8_digits(), D),
    );
    #[cfg(not(feature = "parallel"))]
    let (m_evals_x_result, w_result) = {
        let m_evals_x = compute_m_evals_x::<F, D>(
            setup,
            opening_points,
            claim_to_point,
            challenges,
            alpha,
            &alpha_evals_y,
            lp,
            &tau1,
            claim_group_sizes,
            gamma,
            num_eval_rows,
        )?;
        let w_compact = build_w_evals_compact(w.as_i8_digits(), D);
        (Ok(m_evals_x), w_compact)
    };

    let m_evals_x = m_evals_x_result?;
    let (w_evals_compact, _, _) = w_result?;

    Ok(RingSwitchOutput {
        w,
        w_commitment: Some(w_commitment),
        w_hint: Some(w_hint),
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b: 1usize << lp.log_basis,
        alpha,
    })
}

/// Unified verifier entry point for ring switching: multiple opening
/// points, arbitrary claim-to-point mapping, and arbitrary commitment
/// grouping.  The recursive/single-point path is the trivial
/// `opening_points = [pt]`, `claim_to_point = [0]`, `claim_group_sizes = [1]`,
/// `num_eval_rows = 1` specialisation.
///
/// # Errors
///
/// Returns an error if matrix expansion fails.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub(crate) fn ring_switch_verifier<F, T, const D: usize>(
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
) -> Result<RingSwitchVerifyOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
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

    let tau0 = sample_tau::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);
    let tau1 = sample_tau::<F, T>(transcript, CHALLENGE_TAU1, num_i);
    let alpha_evals_y = build_alpha_evals_y(alpha, D);
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
/// For the default fp128 presets, this uses the same decomposition parameters
/// as the corresponding log-bounded preset, but at the caller-selected ring
/// dimension `D`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WCommitmentConfig<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

impl<const D: usize, Cfg: CommitmentConfig> CommitmentConfig for WCommitmentConfig<D, Cfg> {
    type Field = Cfg::Field;
    const D: usize = D;

    fn decomposition() -> DecompositionParams {
        recursive_level_decomposition_from_root(
            Cfg::decomposition(),
            Cfg::decomposition().log_basis,
        )
    }

    fn stage1_challenge_config(d: usize) -> akita_algebra::SparseChallengeConfig {
        Cfg::stage1_challenge_config(d)
    }

    #[allow(private_interfaces)]
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        Cfg::schedule_table()
    }

    fn audited_root_rank(role: crate::protocol::config::AjtaiRole, max_num_vars: usize) -> usize {
        Cfg::audited_root_rank(role, max_num_vars)
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        Cfg::envelope(max_num_vars)
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), HachiError> {
        Cfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)
    }

    fn level_params_with_log_basis(inputs: HachiScheduleInputs, log_basis: u32) -> LevelParams {
        let params = Cfg::level_params_with_log_basis(inputs, log_basis);
        debug_assert_eq!(params.ring_dimension, D);
        params
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, HachiError> {
        Cfg::root_level_params_for_layout_with_log_basis(inputs, lp)
    }

    fn root_level_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, HachiError> {
        Cfg::root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        Cfg::log_basis_at_level(inputs)
    }

    fn log_basis_search_range(inputs: HachiScheduleInputs) -> (u32, u32) {
        Cfg::log_basis_search_range(inputs)
    }

    fn schedule_key(key: crate::protocol::commitment::HachiScheduleLookupKey) -> String {
        Cfg::schedule_key(key)
    }

    fn schedule_plan(
        key: crate::protocol::commitment::HachiScheduleLookupKey,
    ) -> Result<Option<crate::protocol::commitment::HachiSchedulePlan>, HachiError> {
        Cfg::schedule_plan(key)
    }

    fn commitment_layout(_max_num_vars: usize) -> Result<LevelParams, HachiError> {
        Err(HachiError::InvalidSetup(
            "recursive w layout requires active level params".to_string(),
        ))
    }
}

/// Total ring elements in the w polynomial, computed from the main layout.
///
/// Components: w_hat + t_hat + decomposed z_pre + decomposed r.
pub(crate) fn w_ring_element_count<F: CanonicalField>(lp: &LevelParams) -> usize {
    w_ring_element_count_with_num_claims::<F>(lp, 1)
}

fn checked_num_claims_from_group_sizes(claim_group_sizes: &[usize]) -> Result<usize, HachiError> {
    if claim_group_sizes.is_empty() {
        return Err(HachiError::InvalidSetup(
            "claim groups must be nonempty".to_string(),
        ));
    }
    claim_group_sizes
        .iter()
        .try_fold(0usize, |acc, &group_size| {
            if group_size == 0 {
                return Err(HachiError::InvalidSetup(
                    "claim groups must be nonempty".to_string(),
                ));
            }
            acc.checked_add(group_size)
                .ok_or_else(|| HachiError::InvalidSetup("claim group count overflow".to_string()))
        })
}

fn w_ring_element_count_with_counts<F: CanonicalField>(
    lp: &LevelParams,
    num_claims: usize,
    num_commitment_groups: usize,
    num_points: usize,
) -> usize {
    let w_hat_count = num_claims * lp.num_blocks * lp.num_digits_open;
    let t_hat_count = num_claims * lp.num_blocks * lp.a_key.row_len() * lp.num_digits_open;
    let z_pre_count = num_points * lp.inner_width() * lp.num_digits_fold;
    // One public y-row per distinct opening point (batched_cwss_proof.tex §6).
    let r_rows = lp.m_row_count(num_commitment_groups, num_points);
    let r_count = r_rows * r_decomp_levels::<F>(lp.log_basis);
    w_hat_count + t_hat_count + z_pre_count + r_count
}

pub(crate) fn w_ring_element_count_with_num_claims<F: CanonicalField>(
    lp: &LevelParams,
    num_claims: usize,
) -> usize {
    w_ring_element_count_with_counts::<F>(lp, num_claims, num_claims, 1)
}

pub(crate) fn w_ring_element_count_with_batch_summary<F: CanonicalField>(
    lp: &LevelParams,
    batch: HachiRootBatchSummary,
) -> usize {
    w_ring_element_count_with_counts::<F>(
        lp,
        batch.num_claims,
        batch.num_commitment_groups,
        batch.num_points,
    )
}

pub(crate) fn w_ring_element_count_with_claim_groups<F: CanonicalField>(
    lp: &LevelParams,
    claim_group_sizes: &[usize],
    num_points: usize,
) -> usize {
    let num_claims = claim_group_sizes.iter().sum();
    w_ring_element_count_with_counts::<F>(lp, num_claims, claim_group_sizes.len(), num_points)
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
pub(crate) fn commit_w<F, const D: usize, Cfg>(
    w: &RecursiveWitnessFlat,
    ntt_shared: &NttSlotCache<D>,
    lp: &LevelParams,
    stride: usize,
) -> Result<(RingCommitment<F, D>, HachiCommitmentHint<F, D>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig<Field = F>,
{
    if !w.len().is_multiple_of(D) {
        return Err(HachiError::InvalidSize {
            expected: D,
            actual: w.len(),
        });
    }

    let w_layout = hachi_recursive_level_layout_from_params::<Cfg>(lp, w.len())?;

    let num_blocks = w_layout.num_blocks;
    let block_len = w_layout.block_len;
    let depth_commit = w_layout.num_digits_commit;
    let depth_open = w_layout.num_digits_open;
    let log_basis = w_layout.log_basis;
    let num_ring_elems = w.len() / D;
    tracing::debug!(
        num_ring_elems,
        num_blocks,
        block_len,
        depth_commit,
        depth_open,
        m_vars = w_layout.m_vars,
        r_vars = w_layout.r_vars,
        inner_width = w_layout.inner_width(),
        pow2_block = 1usize << w_layout.m_vars,
        "commit_w layout"
    );

    let w_view = w.view::<F, D>()?;
    let inner = w_view.commit_inner_witness(
        ntt_shared,
        lp.a_key.row_len(),
        block_len,
        num_blocks,
        depth_commit,
        depth_open,
        log_basis,
        stride,
    )?;

    let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
        ntt_shared,
        lp.b_key.row_len(),
        stride,
        inner.t_hat.flat_digits(),
    );
    let hint = HachiCommitmentHint::singleton_with_t(inner.t_hat, inner.t);
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
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    crate::protocol::commitment::digit_math::compute_num_digits_full_field(field_bits, log_basis)
}

/// # Errors
///
/// Returns an error if `w.len()` is not a multiple of `d`.
#[cfg(test)]
pub(crate) fn build_w_evals<F: FieldCore>(
    w: &[F],
    d: usize,
) -> Result<(Vec<F>, usize, usize), HachiError> {
    if !w.len().is_multiple_of(d) {
        return Err(HachiError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let ring_bits = d.trailing_zeros() as usize;
    let num_ring_elems = w.len() / d;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let x_len = 1usize << col_bits;
    let n = x_len << ring_bits;

    let evals: Vec<F> = cfg_into_iter!(0..n)
        .map(|dst| {
            let y = dst & (d - 1);
            let x = dst >> ring_bits;
            let src = y + (x << ring_bits);
            if src < w.len() {
                w[src]
            } else {
                F::zero()
            }
        })
        .collect();
    Ok((evals, col_bits, ring_bits))
}

/// Produce the compact `Vec<i8>` eval table of `w` for the fused prover.
///
/// The compact witness stays in the raw `build_w_coeffs()` order:
/// `w[x * y_len + y]`, with x outer and y inner.
pub(crate) fn build_w_evals_compact(
    w: &[i8],
    d: usize,
) -> Result<(Vec<i8>, usize, usize), HachiError> {
    if !w.len().is_multiple_of(d) {
        return Err(HachiError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let ring_bits = d.trailing_zeros() as usize;
    let live_x_cols = w.len() / d;
    let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
    Ok((w.to_vec(), col_bits, ring_bits))
}

/// Unified M-table evaluation for the batched CWSS protocol.
/// `opening_points` holds the distinct ring-level opening points used by the batch,
/// `claim_to_point` maps each flattened claim index to its opening-point index,
/// and `gamma` provides the per-claim random linear-combination coefficients.
/// The matrix carries one public y-row per distinct opening point
/// (`num_eval_rows = opening_points.len()`).
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_m_evals_x_batched")]
pub(crate) fn compute_m_evals_x<F: FieldCore + CanonicalField, const D: usize>(
    setup: &HachiExpandedSetup<F>,
    opening_points: &[RingOpeningPoint<F>],
    claim_to_point: &[usize],
    challenges: &[SparseChallenge],
    alpha: F,
    alpha_pows: &[F],
    lp: &LevelParams,
    tau1: &[F],
    claim_group_sizes: &[usize],
    gamma: &[F],
    num_eval_rows: usize,
) -> Result<Vec<F>, HachiError> {
    if alpha_pows.len() != D {
        return Err(HachiError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }
    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    validate_opening_points_for_claims(opening_points, claim_to_point, lp, num_claims)?;
    let num_commitment_groups = claim_group_sizes.len();

    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let depth_fold = lp.num_digits_fold;
    let log_basis = lp.log_basis;
    let num_blocks = lp.num_blocks;
    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| HachiError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.len() != total_blocks {
        return Err(HachiError::InvalidSize {
            expected: total_blocks,
            actual: challenges.len(),
        });
    }
    let block_len = lp.block_len;
    let w_len = depth_open * total_blocks;
    let n_a = lp.a_key.row_len();
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    let t_len = depth_open * n_a * total_blocks;
    let inner_width = block_len * depth_commit;
    let z_base_len = opening_points
        .len()
        .checked_mul(inner_width)
        .ok_or_else(|| HachiError::InvalidSetup("batched z width overflow".to_string()))?;
    let z_len = depth_fold
        .checked_mul(z_base_len)
        .ok_or_else(|| HachiError::InvalidSetup("batched z width overflow".to_string()))?;
    let rows = lp.m_row_count(num_commitment_groups, num_eval_rows);
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

    let stride = setup.seed.max_stride;
    let d_view = setup.shared_matrix.ring_view::<D>(n_d, stride);
    let b_view = setup.shared_matrix.ring_view::<D>(n_b, stride);
    let a_view = setup.shared_matrix.ring_view::<D>(n_a, stride);

    // Row layout: consistency (1) | public (num_eval_rows) | D (n_d) |
    //             B (n_b * num_commitment_groups) | A (n_a)
    let commitment_row_count = n_b * num_commitment_groups;
    let consistency_weight = eq_tau1[0];
    let public_weights = &eq_tau1[1..(1 + num_eval_rows)];
    let d_start = 1 + num_eval_rows;
    let b_start = d_start + n_d;
    let a_start = b_start + commitment_row_count;
    let a_weights = &eq_tau1[a_start..rows];
    let claim_to_group: Vec<(usize, usize)> = claim_group_sizes
        .iter()
        .enumerate()
        .flat_map(|(group_idx, &group_size)| {
            (0..group_size).map(move |within_group| (group_idx, within_group))
        })
        .collect();

    let t_compound_per_block = n_a * depth_open;

    let w_segment: Vec<F> = cfg_into_iter!(0..w_len)
        .map(|x| {
            let dig = x / total_blocks;
            let blk = x % total_blocks;
            let claim_idx = blk / num_blocks;
            let block_idx = blk % num_blocks;
            let d_phys_col = blk * depth_open + dig;
            let point_idx = claim_to_point[claim_idx];
            let opening_point = &opening_points[point_idx];
            // The public row weight is per-point: each opening point
            // contributes its own public y-row (one row per point).
            let mut acc =
                (public_weights[point_idx] * gamma[claim_idx] * opening_point.b[block_idx]
                    + consistency_weight * c_alphas[blk])
                    * g1_open[dig];
            for (di, eq_i) in eq_tau1[d_start..(d_start + n_d)].iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&d_view.row(di)[d_phys_col], alpha_pows);
                }
            }
            acc
        })
        .collect();

    let t_cols_per_claim = t_compound_per_block * num_blocks;
    let t_segment: Vec<F> = cfg_into_iter!(0..t_len)
        .map(|x| {
            let compound_dig = x / total_blocks;
            let blk = x % total_blocks;
            let a_idx = compound_dig / depth_open;
            let digit_idx = compound_dig % depth_open;
            let claim_idx = blk / num_blocks;
            let block_idx = blk % num_blocks;
            let (group_idx, claim_idx_within_group) = claim_to_group[claim_idx];
            let phys_claim_offset =
                block_idx * t_compound_per_block + a_idx * depth_open + digit_idx;
            let local_col = claim_idx_within_group * t_cols_per_claim + phys_claim_offset;
            let commitment_weights =
                &eq_tau1[(b_start + group_idx * n_b)..(b_start + (group_idx + 1) * n_b)];
            let mut acc = a_weights[a_idx] * c_alphas[blk] * g1_open[digit_idx];
            for (row_idx, eq_i) in commitment_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&b_view.row(row_idx)[local_col], alpha_pows);
                }
            }
            acc
        })
        .collect();

    let z_base: Vec<F> = cfg_into_iter!(0..z_base_len)
        .map(|k| {
            let point_idx = k / inner_width;
            let local_k = k % inner_width;
            let block_idx = local_k / depth_commit;
            let digit_idx = local_k % depth_commit;
            let opening_point = &opening_points[point_idx];
            let mut acc = consistency_weight * opening_point.a[block_idx] * g1_commit[digit_idx];
            for (a_idx, eq_i) in a_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&a_view.row(a_idx)[local_k], alpha_pows);
                }
            }
            acc
        })
        .collect();

    let num_points = opening_points.len();
    let z_total_blocks = num_points * block_len;
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

    let z_first = lp.m_vars >= lp.r_vars;
    if z_first {
        out.extend(z_segment);
        out.extend(w_segment);
        out.extend(t_segment);
    } else {
        out.extend(w_segment);
        out.extend(t_segment);
        out.extend(z_segment);
    }
    out.extend(r_tail);
    out.resize(x_len, F::zero());
    Ok(out)
}

fn validate_opening_points_for_claims<F: FieldCore>(
    opening_points: &[RingOpeningPoint<F>],
    claim_to_point: &[usize],
    lp: &LevelParams,
    num_claims: usize,
) -> Result<(), HachiError> {
    if opening_points.is_empty() {
        return Err(HachiError::InvalidInput(
            "multipoint ring switch requires at least one opening point".to_string(),
        ));
    }
    if claim_to_point.len() != num_claims {
        return Err(HachiError::InvalidSize {
            expected: num_claims,
            actual: claim_to_point.len(),
        });
    }
    for opening_point in opening_points {
        if opening_point.a.len() < lp.block_len || opening_point.b.len() != lp.num_blocks {
            return Err(HachiError::InvalidInput(
                "multipoint ring switch m-eval opening-point layout mismatch".to_string(),
            ));
        }
    }
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= opening_points.len())
    {
        return Err(HachiError::InvalidInput(
            "multipoint ring switch claim-to-point index out of range".to_string(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "prepare_m_eval")]
pub(crate) fn prepare_m_eval<F: FieldCore + CanonicalField, const D: usize>(
    challenges: &[SparseChallenge],
    alpha: F,
    lp: &LevelParams,
    tau1: &[F],
    claim_group_sizes: &[usize],
    gamma: &[F],
    num_eval_rows: usize,
    opening_points_len: usize,
    claim_to_point: &[usize],
) -> Result<PreparedMEval<F>, HachiError> {
    let alpha_pows = build_alpha_evals_y(alpha, D);
    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    let num_commitment_groups = claim_group_sizes.len();

    if gamma.len() != num_claims {
        return Err(HachiError::InvalidSize {
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
        .ok_or_else(|| HachiError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.len() != total_blocks {
        return Err(HachiError::InvalidSize {
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
        return Err(HachiError::InvalidSize {
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
    #[inline]
    pub(crate) fn eval_at_point<const D: usize>(
        &self,
        x_challenges: &[F],
        setup: &HachiExpandedSetup<F>,
        opening_points: &[RingOpeningPoint<F>],
        alpha: F,
    ) -> Result<F, HachiError> {
        let alpha_pows = build_alpha_evals_y(alpha, D);
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

fn balanced_decompose_centered_i32_i8_into<const D: usize>(
    centered: &[i32; D],
    out: &mut [[i8; D]],
    log_basis: u32,
) {
    let levels = out.len();
    assert!(
        log_basis > 0 && log_basis <= 6,
        "log_basis must be in 1..=6 for i8 output"
    );
    assert!(
        (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
        "levels * log_basis must be <= 128 + log_basis"
    );

    let half_b = 1i128 << (log_basis - 1);
    let b = half_b << 1;
    let mask = b - 1;

    for coeff_idx in 0..D {
        let mut c = centered[coeff_idx] as i128;
        for plane in out.iter_mut() {
            let d = c & mask;
            let balanced = if d >= half_b { d - b } else { d };
            c = (c - balanced) >> log_basis;
            plane[coeff_idx] = balanced as i8;
        }
    }
}

/// Transpose block-major digit planes to digit-major order (block index
/// innermost): for each compound digit index, emit all blocks in order.
fn emit_planes_block_inner<const D: usize>(
    out: &mut Vec<i8>,
    flat: &[[i8; D]],
    total_blocks: usize,
    planes_per_block: usize,
) {
    debug_assert_eq!(
        flat.len(),
        total_blocks * planes_per_block,
        "emit_planes_block_inner: flat.len()={} != total_blocks({}) * planes_per_block({})",
        flat.len(),
        total_blocks,
        planes_per_block
    );
    for compound_dig in 0..planes_per_block {
        for blk in 0..total_blocks {
            out.extend_from_slice(&flat[blk * planes_per_block + compound_dig]);
        }
    }
}

/// Decompose z_pre elements and emit in digit-major order.
///
/// z_pre has `num_points * block_len * depth_commit` elements indexed as
/// `z[point * inner_width + blk * depth_commit + dc]`.  Each decomposes into
/// `num_digits_fold` planes.
///
/// Output order: for each `(dc, df)`, emit all `(point, blk)` pairs with
/// the global block index `point * block_len + blk` innermost.
fn emit_z_pre_block_inner<const D: usize>(
    out: &mut Vec<i8>,
    z_pre_centered: &[[i32; D]],
    block_len: usize,
    depth_commit: usize,
    num_digits_fold: usize,
    log_basis: u32,
) {
    let total_elems = z_pre_centered.len();
    let inner_width = block_len * depth_commit;
    debug_assert_eq!(
        total_elems % inner_width,
        0,
        "z_pre length {total_elems} not divisible by inner_width {inner_width}",
    );
    let num_points = total_elems / inner_width;

    let mut all_planes = vec![[0i8; D]; total_elems * num_digits_fold];
    for (k, z_j) in z_pre_centered.iter().enumerate() {
        balanced_decompose_centered_i32_i8_into(
            z_j,
            &mut all_planes[k * num_digits_fold..(k + 1) * num_digits_fold],
            log_basis,
        );
    }

    for dc in 0..depth_commit {
        for df in 0..num_digits_fold {
            for pt in 0..num_points {
                for blk in 0..block_len {
                    let k = pt * inner_width + blk * depth_commit + dc;
                    out.extend_from_slice(&all_planes[k * num_digits_fold + df]);
                }
            }
        }
    }
}

/// Build the committed witness polynomial from ring-domain digit planes.
///
/// Emits field-domain coefficients in digit-major order (block index innermost)
/// with adaptive segment ordering: the segment whose block dimension is the
/// larger power of two comes first.
///
/// Segment ordering:
/// - If `m_vars >= r_vars`: z-hat (`2^m` blocks), e-hat + t-hat (`2^r` blocks), r-hat
/// - If `m_vars < r_vars`: e-hat + t-hat (`2^r` blocks), z-hat (`2^m` blocks), r-hat
///
/// Within each segment, the power-of-2 block index is the fastest-varying
/// (innermost) dimension.
///
/// `FlatDigitBlocks` stores ring-domain data in block-major order (all digit
/// planes for one block contiguously), which is natural for ring-domain matvec
/// and recomposition. This function transposes to digit-major at the
/// ring-to-field boundary. An alternative would be propagating digit-major
/// throughout `FlatDigitBlocks`, eliminating this transposition but requiring
/// restructured producers and block-level operations.
pub(crate) fn build_w_coeffs<F: CanonicalField, const D: usize>(
    w_hat: &FlatDigitBlocks<D>,
    t_hat: &FlatDigitBlocks<D>,
    z_pre_centered: &[[i32; D]],
    r: &[CyclotomicRing<F, D>],
    lp: &LevelParams,
) -> RecursiveWitnessFlat {
    let log_basis = lp.log_basis;
    let num_digits_fold = lp.num_digits_fold;
    let depth_open = lp.num_digits_open;
    let depth_commit = lp.num_digits_commit;
    let block_len = lp.block_len;
    let levels = r_decomp_levels::<F>(log_basis);

    let w_hat_planes = w_hat.flat_digits().len();
    let t_hat_planes = t_hat.flat_digits().len();
    let z_count = w_hat_planes + t_hat_planes + z_pre_centered.len() * num_digits_fold;
    let r_hat_count = r.len() * levels;
    let z_first = lp.m_vars >= lp.r_vars;
    tracing::debug!(
        w_hat_planes,
        t_hat_planes,
        z_pre_elems = z_pre_centered.len(),
        z_pre_planes = z_pre_centered.len() * num_digits_fold,
        r_elems = r.len(),
        r_planes = r_hat_count,
        total_ring = z_count + r_hat_count,
        total_field = (z_count + r_hat_count) * D,
        z_first,
        "build_w_coeffs"
    );
    let total_planes = z_count + r_hat_count;
    let total_elems = total_planes * D;

    let mut out = Vec::with_capacity(total_elems);

    let total_blocks_et = if depth_open > 0 {
        w_hat_planes / depth_open
    } else {
        0
    };
    let t_planes_per_block = if total_blocks_et > 0 {
        t_hat_planes / total_blocks_et
    } else {
        0
    };

    if z_first {
        emit_z_pre_block_inner(
            &mut out,
            z_pre_centered,
            block_len,
            depth_commit,
            num_digits_fold,
            log_basis,
        );
        emit_planes_block_inner(&mut out, w_hat.flat_digits(), total_blocks_et, depth_open);
        emit_planes_block_inner(
            &mut out,
            t_hat.flat_digits(),
            total_blocks_et,
            t_planes_per_block,
        );
    } else {
        emit_planes_block_inner(&mut out, w_hat.flat_digits(), total_blocks_et, depth_open);
        emit_planes_block_inner(
            &mut out,
            t_hat.flat_digits(),
            total_blocks_et,
            t_planes_per_block,
        );
        emit_z_pre_block_inner(
            &mut out,
            z_pre_centered,
            block_len,
            depth_commit,
            num_digits_fold,
            log_basis,
        );
    }

    let mut r_planes = vec![[0i8; D]; levels];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(levels, log_basis, q);
    for ri in r {
        r_planes.fill([0i8; D]);
        ri.balanced_decompose_pow2_i8_into_with_params(&mut r_planes, &decompose_params);
        for plane in &r_planes {
            out.extend_from_slice(plane);
        }
    }
    RecursiveWitnessFlat::from_i8_digits(out)
}

#[cfg(test)]
mod tests {
    use super::{
        build_alpha_evals_y, build_w_evals_compact, compute_m_evals_x, compute_r_via_poly_division,
        prepare_m_eval, ring_switch_build_w,
    };
    use crate::protocol::commitment_scheme::HachiCommitmentScheme;
    use crate::protocol::config::proof_optimized::fp128;
    use crate::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps};
    use crate::protocol::quadratic_equation::QuadraticEquation;
    use crate::protocol::sumcheck::hachi_stage2::relation_claim_from_rows;
    use crate::protocol::CommitmentConfig;
    use crate::{CanonicalField, CommitmentProver, Transcript};
    use akita_algebra::CyclotomicRing;
    use akita_transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
    use akita_transcript::Blake2bTranscript;
    use akita_types::AppendToTranscript;
    use akita_types::{ring_opening_point_from_field, BasisMode, BlockOrder};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
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
        type F = fp128::Field;
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

    fn direct_relation_claim<F: FieldCore + FromSmallInt>(
        w_compact: &[i8],
        alpha_evals_y: &[F],
        m_evals_x: &[F],
        live_x_cols: usize,
    ) -> F {
        (0..live_x_cols).fold(F::zero(), |acc_x, x| {
            let column_start = x * alpha_evals_y.len();
            let y_eval = alpha_evals_y
                .iter()
                .enumerate()
                .fold(F::zero(), |acc_y, (y, &alpha)| {
                    acc_y + F::from_i64(w_compact[column_start + y] as i64) * alpha
                });
            acc_x + y_eval * m_evals_x[x]
        })
    }

    #[test]
    fn full_root_rows_match_direct_relation_claim() {
        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
        const NV: usize = 12;

        let lp = Cfg::commitment_layout(NV).expect("lp");

        let mut rng = StdRng::seed_from_u64(0x5eed_cafe);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let (commitment, batched_hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<
            F,
            D,
        >>::commit(&[poly.clone()], &setup)
        .expect("commitment");

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            lp.r_vars,
            lp.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("ring opening point");
        let (y_ring, w_folded) =
            poly.evaluate_and_fold(&ring_opening_point.b, &ring_opening_point.a, lp.block_len);

        let mut transcript = Blake2bTranscript::<F>::new(b"ring-switch-row-regression");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

        let mut quad_eq = QuadraticEquation::<F, D, Cfg>::new_prover(
            &setup.ntt_shared,
            vec![ring_opening_point],
            vec![0usize],
            &[&poly],
            vec![w_folded],
            &[1usize],
            lp.clone(),
            vec![batched_hint],
            &mut transcript,
            std::slice::from_ref(&commitment),
            std::slice::from_ref(&y_ring),
            vec![F::one()],
            setup.expanded.seed.max_stride,
        )
        .expect("quadratic equation");

        let w =
            ring_switch_build_w::<F, D, Cfg>(&mut quad_eq, &setup.expanded, &setup.ntt_shared, &lp)
                .expect("ring-switch witness");
        let (w_compact, _col_bits, ring_bits) =
            build_w_evals_compact(w.as_i8_digits(), D).expect("compact witness");
        let live_x_cols = w_compact.len() >> ring_bits;

        let alpha = F::from_u64(17);
        let alpha_evals_y = build_alpha_evals_y(alpha, D);
        let rows = lp.m_row_count(1, 1);
        let num_i = rows.next_power_of_two().trailing_zeros() as usize;

        for row in 0..rows {
            let tau1: Vec<F> = (0..num_i)
                .map(|bit| {
                    if (row >> bit) & 1 == 1 {
                        F::one()
                    } else {
                        F::zero()
                    }
                })
                .collect();
            let m_evals_x = compute_m_evals_x::<F, D>(
                &setup.expanded,
                &[quad_eq.opening_point().clone()],
                &[0usize],
                &quad_eq.challenges,
                alpha,
                &alpha_evals_y,
                &lp,
                &tau1,
                &[1usize],
                &[F::one()],
                1,
            )
            .expect("m evals");
            let got = direct_relation_claim(&w_compact, &alpha_evals_y, &m_evals_x, live_x_cols);
            let expected = relation_claim_from_rows::<F, D>(
                &tau1,
                alpha,
                &quad_eq.v,
                &commitment.u,
                std::slice::from_ref(&y_ring),
            );
            assert_eq!(got, expected, "row {row} mismatch");
        }
    }

    #[test]
    fn centered_i32_decompose_matches_ring_decompose() {
        type F = fp128::Field;
        const D: usize = 128;

        let centered = from_fn(|i| ((37 * i as i32 + 11) % 95) - 47);
        let ring =
            CyclotomicRing::<F, D>::from_coefficients(from_fn(|i| F::from_i64(centered[i] as i64)));

        for (num_digits, log_basis) in [
            (7usize, 3u32),
            (10usize, 2u32),
            (5usize, 5u32),
            (4usize, 6u32),
        ] {
            let mut got = vec![[0i8; D]; num_digits];
            super::balanced_decompose_centered_i32_i8_into(&centered, &mut got, log_basis);

            let mut expected = vec![[0i8; D]; num_digits];
            ring.balanced_decompose_pow2_i8_into(&mut expected, log_basis);
            assert_eq!(
                got, expected,
                "centered i32 decomposition mismatch for num_digits={num_digits} log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn asymmetric_centering_decompose_roundtrip() {
        use crate::protocol::commitment::digit_math::compute_num_digits_full_field;

        type F = fp128::Field;
        const D: usize = 64;

        let mut rng = rand::thread_rng();

        for log_basis in [2u32, 3, 4, 5, 6] {
            let field_bits = 128u32;
            let num_digits = compute_num_digits_full_field(field_bits, log_basis);

            let ring = CyclotomicRing::<F, D>::random(&mut rng);

            let mut digits = vec![CyclotomicRing::<F, D>::zero(); num_digits];
            ring.balanced_decompose_pow2_into(&mut digits, log_basis);
            let recomposed = CyclotomicRing::gadget_recompose_pow2(&digits, log_basis);
            assert_eq!(
                ring, recomposed,
                "field-element roundtrip failed for log_basis={log_basis}, num_digits={num_digits}"
            );

            let mut i8_digits = vec![[0i8; D]; num_digits];
            ring.balanced_decompose_pow2_i8_into(&mut i8_digits, log_basis);
            let recomposed_i8 = CyclotomicRing::gadget_recompose_pow2_i8(&i8_digits, log_basis);
            assert_eq!(
                ring, recomposed_i8,
                "i8 roundtrip failed for log_basis={log_basis}, num_digits={num_digits}"
            );
        }
    }

    #[test]
    fn prepared_m_eval_matches_materialized() {
        use akita_sumcheck::multilinear_eval;

        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
        const NV: usize = 12;

        let level_params = Cfg::commitment_layout(NV).expect("commitment layout");

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let (commitment, batched_hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentProver<
            F,
            D,
        >>::commit(&[poly.clone()], &setup)
        .expect("commitment");

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            level_params.r_vars,
            level_params.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("ring opening point");
        let (y_ring, w_folded) = poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            level_params.block_len,
        );

        let mut transcript = Blake2bTranscript::<F>::new(b"prepared-m-eval-test");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

        let mut quad_eq = QuadraticEquation::<F, D, Cfg>::new_prover(
            &setup.ntt_shared,
            vec![ring_opening_point.clone()],
            vec![0usize],
            &[&poly],
            vec![w_folded],
            &[1usize],
            level_params.clone(),
            vec![batched_hint],
            &mut transcript,
            std::slice::from_ref(&commitment),
            std::slice::from_ref(&y_ring),
            vec![F::one()],
            setup.expanded.seed.max_stride,
        )
        .expect("quadratic equation");

        ring_switch_build_w::<F, D, Cfg>(
            &mut quad_eq,
            &setup.expanded,
            &setup.ntt_shared,
            &level_params,
        )
        .expect("ring-switch witness");

        let alpha = F::from_u64(42);
        let alpha_evals_y = build_alpha_evals_y(alpha, D);
        let rows = level_params.m_row_count(1, 1);
        let num_i = rows.next_power_of_two().trailing_zeros() as usize;
        let tau1: Vec<F> = (0..num_i)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let m_evals_x = compute_m_evals_x::<F, D>(
            &setup.expanded,
            &[ring_opening_point.clone()],
            &[0usize],
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            &tau1,
            &[1usize],
            &[F::one()],
            1,
        )
        .expect("m evals (materialized)");

        let x_challenges: Vec<F> = (0..m_evals_x.len().trailing_zeros() as usize)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let expected = multilinear_eval(&m_evals_x, &x_challenges).expect("multilinear_eval");

        let prepared = prepare_m_eval::<F, D>(
            &quad_eq.challenges,
            alpha,
            &level_params,
            &tau1,
            &[1usize],
            &[F::one()],
            1,
            1,
            &[],
        )
        .expect("prepare_m_eval");

        let got = prepared
            .eval_at_point::<D>(
                &x_challenges,
                &setup.expanded,
                std::slice::from_ref(&ring_opening_point),
                alpha,
            )
            .expect("eval_at_point");

        assert_eq!(
            got, expected,
            "PreparedMEval::eval_at_point must match materialized multilinear_eval"
        );
    }
}
