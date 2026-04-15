//! Ring switching logic for the Hachi PCS (Section 4.3).
//!
//! Handles the transition from the ring-based quadratic equation to field-based
//! sumcheck instances by expanding the ring elements into their coefficient
//! vectors and setting up the evaluation tables.

use crate::algebra::eq_poly::EqPolynomial;
use crate::algebra::offset_eq::{
    eval_offset_eq_peeled_carry_terms, eval_offset_eq_tensor, summarize_pow2_block_carries,
};
use crate::algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use crate::algebra::{CyclotomicRing, SparseChallenge};
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::flat_matrix::RingMatrixView;
use crate::protocol::commitment::utils::linear::mat_vec_mul_ntt_single_i8;
use crate::protocol::commitment::utils::norm::detect_field_modulus;
use crate::protocol::commitment::HachiRootBatchSummary;
use crate::protocol::commitment::{
    hachi_recursive_level_layout_from_params, recursive_level_decomposition_from_root,
    recursive_r_decomp_levels_for_bound, CommitmentConfig, CommitmentEnvelope, DecompositionParams,
    HachiCommitmentLayout, HachiExpandedSetup, HachiLevelParams, HachiScheduleInputs,
    RingCommitment,
};
use crate::protocol::hachi_poly_ops::RecursiveWitnessFlat;
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::proof::{FlatDigitBlocks, FlatRingVec, HachiCommitmentHint};
use crate::protocol::quadratic_equation::{compute_r_split_eq, QuadraticEquation};
use crate::protocol::recursive_runtime::RecursiveCommitmentHintCache;
use crate::protocol::shared_matrix_setup::SharedMatrixTensorLayout;
use crate::protocol::transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling};
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
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
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
    let hint = quad_eq
        .take_hint()
        .ok_or_else(|| HachiError::InvalidInput("missing hint in prover".to_string()))?;
    let inner_opening_digits = &hint.inner_opening_digits;
    let t = hint.t().ok_or_else(|| {
        HachiError::InvalidInput("missing recomposed t in prover hint".to_string())
    })?;
    let w_folded = quad_eq
        .take_w_folded()
        .ok_or_else(|| HachiError::InvalidInput("missing w_folded in prover".to_string()))?;

    let r = compute_r_split_eq::<F, D>(
        level_params,
        setup,
        &quad_eq.challenges,
        w_hat.flat_digits(),
        inner_opening_digits,
        t,
        &w_folded,
        &z_pre.centered_coeffs,
        z_pre.centered_inf_norm,
        quad_eq.y(),
        quad_eq.claim_group_sizes(),
        layout.num_blocks,
        layout.inner_width,
        setup.seed.max_stride(),
        ntt_shared,
    )?;
    let w = {
        let _span = tracing::info_span!("build_w_coeffs").entered();
        build_w_coeffs::<F, D>(
            &w_hat,
            inner_opening_digits,
            &z_pre.centered_coeffs,
            &r,
            layout,
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
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
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
        level_params,
        layout,
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
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> Result<RingSwitchOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    transcript.append_serde(ABSORB_SUMCHECK_W, w_commitment_proof);

    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

    let claim_group_sizes = quad_eq.claim_group_sizes();
    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    let num_commitment_groups = claim_group_sizes.len();

    let ring_bits = D.trailing_zeros() as usize;
    let num_ring_elems = w.len() / D;
    let live_x_cols = num_ring_elems;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let m_rows = if num_claims == 1 && num_commitment_groups == 1 {
        m_row_count(level_params)
    } else {
        level_params
            .m_row_count_with_commitments_and_public_outputs(num_commitment_groups, num_claims)
    };
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;

    let tau0 = sample_tau::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);
    let tau1 = sample_tau::<F, T>(transcript, CHALLENGE_TAU1, num_i);
    let alpha_evals_y = build_alpha_evals_y(alpha, D);

    let opening_points = quad_eq.opening_points();
    let claim_to_point = quad_eq.claim_to_point();
    let challenges = &quad_eq.challenges;

    #[cfg(feature = "parallel")]
    let (m_evals_x_result, w_result) = rayon::join(
        || {
            if opening_points.len() == 1 && num_claims == 1 && num_commitment_groups == 1 {
                compute_m_evals_x::<F, D>(
                    setup,
                    &opening_points[0],
                    challenges,
                    alpha,
                    &alpha_evals_y,
                    level_params,
                    layout,
                    &tau1,
                )
            } else if opening_points.len() == 1 {
                compute_m_evals_x_with_claim_groups::<F, D>(
                    setup,
                    &opening_points[0],
                    challenges,
                    alpha,
                    &alpha_evals_y,
                    level_params,
                    layout,
                    &tau1,
                    claim_group_sizes,
                )
            } else {
                compute_m_evals_x_with_opening_points_and_claim_groups::<F, D>(
                    setup,
                    opening_points,
                    claim_to_point,
                    challenges,
                    alpha,
                    &alpha_evals_y,
                    level_params,
                    layout,
                    &tau1,
                    claim_group_sizes,
                )
            }
        },
        || build_w_evals_compact(w.as_i8_digits(), D),
    );
    #[cfg(not(feature = "parallel"))]
    let (m_evals_x_result, w_result) = {
        let m_evals_x =
            if opening_points.len() == 1 && num_claims == 1 && num_commitment_groups == 1 {
                compute_m_evals_x::<F, D>(
                    setup,
                    &opening_points[0],
                    challenges,
                    alpha,
                    &alpha_evals_y,
                    level_params,
                    layout,
                    &tau1,
                )?
            } else if opening_points.len() == 1 {
                compute_m_evals_x_with_claim_groups::<F, D>(
                    setup,
                    &opening_points[0],
                    challenges,
                    alpha,
                    &alpha_evals_y,
                    level_params,
                    layout,
                    &tau1,
                    claim_group_sizes,
                )?
            } else {
                compute_m_evals_x_with_opening_points_and_claim_groups::<F, D>(
                    setup,
                    opening_points,
                    claim_to_point,
                    challenges,
                    alpha,
                    &alpha_evals_y,
                    level_params,
                    layout,
                    &tau1,
                    claim_group_sizes,
                )?
            };
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
/// Replay the verifier side of ring switching to reconstruct evaluation tables.
///
/// # Errors
///
/// Returns an error if matrix expansion fails.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub(crate) fn ring_switch_verifier<F, T, const D: usize>(
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    w_len: usize,
    w_commitment: &FlatRingVec<F>,
    transcript: &mut T,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> Result<RingSwitchVerifyOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    ring_switch_verifier_with_claim_groups::<F, T, D>(
        opening_point,
        challenges,
        w_len,
        w_commitment,
        transcript,
        level_params,
        layout,
        &[1usize],
    )
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "ring_switch_verifier_with_claim_groups")]
#[inline(never)]
pub(crate) fn ring_switch_verifier_with_claim_groups<F, T, const D: usize>(
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    w_len: usize,
    w_commitment: &FlatRingVec<F>,
    transcript: &mut T,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    claim_group_sizes: &[usize],
) -> Result<RingSwitchVerifyOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_SUMCHECK_W, w_commitment);

    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);
    if opening_point.a.len() < layout.block_len || opening_point.b.len() != layout.num_blocks {
        return Err(HachiError::InvalidInput(
            "ring switch verifier opening-point layout mismatch".to_string(),
        ));
    }

    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    let num_commitment_groups = claim_group_sizes.len();

    let num_ring_elems = w_len / D;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let ring_bits = D.trailing_zeros() as usize;
    let m_rows = if num_claims == 1 && num_commitment_groups == 1 {
        m_row_count(level_params)
    } else {
        level_params
            .m_row_count_with_commitments_and_public_outputs(num_commitment_groups, num_claims)
    };
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;

    let tau0 = sample_tau::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);
    let tau1 = sample_tau::<F, T>(transcript, CHALLENGE_TAU1, num_i);
    let alpha_evals_y = build_alpha_evals_y(alpha, D);
    let prepared_m_eval = prepare_m_eval::<F, D>(
        challenges,
        alpha,
        level_params,
        layout,
        &tau1,
        claim_group_sizes,
        1,
        &[],
    )?;

    Ok(RingSwitchVerifyOutput {
        prepared_m_eval,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b: 1usize << layout.log_basis,
        alpha,
    })
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(
    skip_all,
    name = "ring_switch_verifier_with_opening_points_and_claim_groups"
)]
#[inline(never)]
pub(crate) fn ring_switch_verifier_with_opening_points_and_claim_groups<F, T, const D: usize>(
    opening_points: &[RingOpeningPoint<F>],
    claim_to_point: &[usize],
    challenges: &[SparseChallenge],
    w_len: usize,
    w_commitment: &FlatRingVec<F>,
    transcript: &mut T,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    claim_group_sizes: &[usize],
) -> Result<RingSwitchVerifyOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_SUMCHECK_W, w_commitment);

    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    validate_opening_points_for_claims(opening_points, claim_to_point, layout, num_claims)?;
    let num_commitment_groups = claim_group_sizes.len();

    let num_ring_elems = w_len / D;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let ring_bits = D.trailing_zeros() as usize;
    let m_rows = if num_claims == 1 && num_commitment_groups == 1 && opening_points.len() == 1 {
        m_row_count(level_params)
    } else {
        level_params
            .m_row_count_with_commitments_and_public_outputs(num_commitment_groups, num_claims)
    };
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;

    let tau0 = sample_tau::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);
    let tau1 = sample_tau::<F, T>(transcript, CHALLENGE_TAU1, num_i);
    let alpha_evals_y = build_alpha_evals_y(alpha, D);
    let prepared_m_eval = prepare_m_eval::<F, D>(
        challenges,
        alpha,
        level_params,
        layout,
        &tau1,
        claim_group_sizes,
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

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        Cfg::envelope(max_num_vars)
    }

    fn stage1_challenge_config(d: usize) -> crate::algebra::SparseChallengeConfig {
        Cfg::stage1_challenge_config(d)
    }

    fn level_params_with_log_basis(
        inputs: HachiScheduleInputs,
        log_basis: u32,
    ) -> HachiLevelParams {
        let params = Cfg::level_params_with_log_basis(inputs, log_basis);
        debug_assert_eq!(params.d, D);
        params
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

    fn decomposition() -> DecompositionParams {
        recursive_level_decomposition_from_root(
            Cfg::decomposition(),
            Cfg::decomposition().log_basis,
        )
    }

    fn commitment_layout(_max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        Err(HachiError::InvalidSetup(
            "recursive w layout requires active level params".to_string(),
        ))
    }
}

/// Total ring elements in the w polynomial, computed from the main layout.
///
/// Components: w_hat + t_hat + decomposed z_pre + decomposed r.
pub(crate) fn w_ring_element_count<F: CanonicalField>(
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> usize {
    w_ring_element_count_with_num_claims::<F>(level_params, layout, 1)
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
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    num_claims: usize,
    num_commitment_groups: usize,
    num_point_sets: usize,
) -> usize {
    let w_hat_count = num_claims * layout.num_blocks * layout.num_digits_open;
    let t_hat_count = num_claims * layout.num_blocks * level_params.n_a * layout.num_digits_open;
    let z_pre_count = num_point_sets * layout.inner_width * layout.num_digits_fold;
    let r_rows = if num_claims == 1 && num_commitment_groups == 1 {
        level_params.m_row_count()
    } else {
        level_params
            .m_row_count_with_commitments_and_public_outputs(num_commitment_groups, num_claims)
    };
    let r_count = r_rows * r_decomp_levels::<F>(layout.log_basis);
    w_hat_count + t_hat_count + z_pre_count + r_count
}

pub(crate) fn w_ring_element_count_with_num_claims<F: CanonicalField>(
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    num_claims: usize,
) -> usize {
    w_ring_element_count_with_counts::<F>(level_params, layout, num_claims, num_claims, 1)
}

#[cfg(test)]
pub(crate) fn w_ring_element_count_with_num_claims_and_points<F: CanonicalField>(
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    num_claims: usize,
    num_points: usize,
) -> usize {
    w_ring_element_count_with_counts::<F>(level_params, layout, num_claims, num_claims, num_points)
}

pub(crate) fn w_ring_element_count_with_batch_summary<F: CanonicalField>(
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    batch: HachiRootBatchSummary,
) -> usize {
    w_ring_element_count_with_counts::<F>(
        level_params,
        layout,
        batch.num_claims,
        batch.num_commitment_groups,
        batch.num_points,
    )
}

pub(crate) fn w_ring_element_count_with_claim_groups<F: CanonicalField>(
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    claim_group_sizes: &[usize],
) -> usize {
    let num_claims = claim_group_sizes.iter().sum();
    w_ring_element_count_with_counts::<F>(
        level_params,
        layout,
        num_claims,
        claim_group_sizes.len(),
        1,
    )
}

pub(crate) fn w_ring_element_count_with_point_claim_groups<F: CanonicalField>(
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    claim_group_sizes: &[usize],
    num_points: usize,
) -> usize {
    let num_claims = claim_group_sizes.iter().sum();
    w_ring_element_count_with_counts::<F>(
        level_params,
        layout,
        num_claims,
        claim_group_sizes.len(),
        num_points,
    )
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
    level_params: &HachiLevelParams,
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

    let w_layout = hachi_recursive_level_layout_from_params::<Cfg>(level_params, w.len())?;

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
        inner_width = w_layout.inner_width,
        pow2_block = 1usize << w_layout.m_vars,
        "commit_w layout"
    );

    let w_view = w.view::<F, D>()?;
    let inner = w_view.commit_inner_witness(
        ntt_shared,
        level_params.n_a,
        block_len,
        num_blocks,
        depth_commit,
        depth_open,
        log_basis,
        stride,
    )?;

    let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
        ntt_shared,
        level_params.n_b,
        stride,
        inner.t_hat.flat_digits(),
    );
    let hint = HachiCommitmentHint::with_t(inner.t_hat, inner.t);
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
pub(crate) fn gadget_row_scalars<F: FieldCore + CanonicalField>(
    levels: usize,
    log_basis: u32,
) -> Vec<F> {
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
    recursive_r_decomp_levels_for_bound(field_bits, modulus / 2, log_basis)
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

pub(crate) fn m_row_count(level_params: &HachiLevelParams) -> usize {
    level_params.m_row_count()
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_m_evals_x")]
pub(crate) fn compute_m_evals_x<F: FieldCore + CanonicalField, const D: usize>(
    setup: &HachiExpandedSetup<F>,
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    alpha: F,
    alpha_pows: &[F],
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    tau1: &[F],
) -> Result<Vec<F>, HachiError> {
    compute_m_evals_x_with_claim_groups::<F, D>(
        setup,
        opening_point,
        challenges,
        alpha,
        alpha_pows,
        level_params,
        layout,
        tau1,
        &[1usize],
    )
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_m_evals_x_with_claim_groups")]
pub(crate) fn compute_m_evals_x_with_claim_groups<F: FieldCore + CanonicalField, const D: usize>(
    setup: &HachiExpandedSetup<F>,
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    alpha: F,
    alpha_pows: &[F],
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    tau1: &[F],
    claim_group_sizes: &[usize],
) -> Result<Vec<F>, HachiError> {
    if alpha_pows.len() != D {
        return Err(HachiError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }
    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    let num_commitment_groups = claim_group_sizes.len();

    let depth_commit = layout.num_digits_commit;
    let depth_open = layout.num_digits_open;
    let depth_fold = layout.num_digits_fold;
    let log_basis = layout.log_basis;
    let num_blocks = opening_point.b.len();
    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| HachiError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.len() != total_blocks {
        return Err(HachiError::InvalidSize {
            expected: total_blocks,
            actual: challenges.len(),
        });
    }
    let block_len = layout.block_len;
    let w_len = depth_open * total_blocks;
    let t_len = depth_open * level_params.n_a * total_blocks;
    let inner_width = block_len * depth_commit;
    let z_len = depth_fold * inner_width;
    let rows = if num_claims == 1 && num_commitment_groups == 1 {
        level_params.m_row_count()
    } else {
        level_params
            .m_row_count_with_commitments_and_public_outputs(num_commitment_groups, num_claims)
    };
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

    let stride = setup.seed.max_stride();
    let d_view = setup.shared_matrix.ring_view::<D>(level_params.n_d, stride);
    let b_view = setup.shared_matrix.ring_view::<D>(level_params.n_b, stride);
    let a_view = setup.shared_matrix.ring_view::<D>(level_params.n_a, stride);

    // Row layout: consistency (1) | public (num_claims) | D (n_d) |
    //             B (n_b * num_commitment_groups) | A (n_a)
    let commitment_row_count = level_params.n_b * num_commitment_groups;
    let consistency_weight = eq_tau1[0];
    let public_weights = &eq_tau1[1..(1 + num_claims)];
    let d_start = 1 + num_claims;
    let b_start = d_start + level_params.n_d;
    let a_start = b_start + commitment_row_count;
    let a_weights = &eq_tau1[a_start..rows];
    let claim_to_group: Vec<(usize, usize)> = claim_group_sizes
        .iter()
        .enumerate()
        .flat_map(|(group_idx, &group_size)| {
            (0..group_size).map(move |within_group| (group_idx, within_group))
        })
        .collect();

    // --- Digit-major segments (block index innermost) ---
    //
    // Within each segment the power-of-2 block index is the fastest-varying
    // dimension.  Adaptive ordering places the segment whose block dimension
    // is larger first.

    let n_a = level_params.n_a;
    let t_compound_per_block = n_a * depth_open;

    let w_segment: Vec<F> = cfg_into_iter!(0..w_len)
        .map(|x| {
            let dig = x / total_blocks;
            let blk = x % total_blocks;
            let claim_idx = blk / num_blocks;
            let block_idx = blk % num_blocks;
            let d_phys_col = blk * depth_open + dig;
            let mut acc = (public_weights[claim_idx] * opening_point.b[block_idx]
                + consistency_weight * c_alphas[blk])
                * g1_open[dig];
            for (di, eq_i) in eq_tau1[d_start..(d_start + level_params.n_d)]
                .iter()
                .enumerate()
            {
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
            let commitment_weights = &eq_tau1[(b_start + group_idx * level_params.n_b)
                ..(b_start + (group_idx + 1) * level_params.n_b)];
            let mut acc = a_weights[a_idx] * c_alphas[blk] * g1_open[digit_idx];
            for (row_idx, eq_i) in commitment_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&b_view.row(row_idx)[local_col], alpha_pows);
                }
            }
            acc
        })
        .collect();

    let z_base: Vec<F> = cfg_into_iter!(0..inner_width)
        .map(|k| {
            let block_idx = k / depth_commit;
            let digit_idx = k % depth_commit;
            let mut acc = consistency_weight * opening_point.a[block_idx] * g1_commit[digit_idx];
            for (a_idx, eq_i) in a_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&a_view.row(a_idx)[k], alpha_pows);
                }
            }
            acc
        })
        .collect();

    let z_segment: Vec<F> = cfg_into_iter!(0..z_len)
        .map(|x| {
            let compound_dig = x / block_len;
            let blk = x % block_len;
            let dc = compound_dig / depth_fold;
            let df = compound_dig % depth_fold;
            let phys_k = blk * depth_commit + dc;
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

    let z_first = layout.m_vars >= layout.r_vars;
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

/// Compute only the algebraic (non-matrix-backed) part of `m_evals_x`.
///
/// The full `m_evals_x` vector decomposes additively as `alg + setup`, where
/// `alg` depends only on protocol-sampled scalars (opening point, challenges,
/// eq_tau1, gadgets) and `setup` depends on the physical setup matrix entries.
/// This function returns the `alg` part.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn compute_alg_m_evals_x_with_claim_groups<
    F: FieldCore + CanonicalField,
    const D: usize,
>(
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    alpha: F,
    alpha_pows: &[F],
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    tau1: &[F],
    claim_group_sizes: &[usize],
) -> Result<Vec<F>, HachiError> {
    if alpha_pows.len() != D {
        return Err(HachiError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }
    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    let num_commitment_groups = claim_group_sizes.len();

    let depth_commit = layout.num_digits_commit;
    let depth_open = layout.num_digits_open;
    let depth_fold = layout.num_digits_fold;
    let log_basis = layout.log_basis;
    let num_blocks = opening_point.b.len();
    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| HachiError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.len() != total_blocks {
        return Err(HachiError::InvalidSize {
            expected: total_blocks,
            actual: challenges.len(),
        });
    }
    let block_len = layout.block_len;
    let w_len = depth_open * total_blocks;
    let t_len = depth_open * level_params.n_a * total_blocks;
    let inner_width = block_len * depth_commit;
    let z_len = depth_fold * inner_width;
    let rows = if num_claims == 1 && num_commitment_groups == 1 {
        level_params.m_row_count()
    } else {
        level_params
            .m_row_count_with_commitments_and_public_outputs(num_commitment_groups, num_claims)
    };
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

    let consistency_weight = eq_tau1[0];
    let public_weights = &eq_tau1[1..(1 + num_claims)];
    let a_start = 1 + num_claims + level_params.n_d + level_params.n_b * num_commitment_groups;
    let a_weights = &eq_tau1[a_start..rows];

    let w_segment: Vec<F> = cfg_into_iter!(0..w_len)
        .map(|x| {
            let dig = x / total_blocks;
            let blk = x % total_blocks;
            let claim_idx = blk / num_blocks;
            let block_idx = blk % num_blocks;
            (public_weights[claim_idx] * opening_point.b[block_idx]
                + consistency_weight * c_alphas[blk])
                * g1_open[dig]
        })
        .collect();

    let t_segment: Vec<F> = cfg_into_iter!(0..t_len)
        .map(|x| {
            let compound_dig = x / total_blocks;
            let blk = x % total_blocks;
            let a_idx = compound_dig / depth_open;
            let digit_idx = compound_dig % depth_open;
            a_weights[a_idx] * c_alphas[blk] * g1_open[digit_idx]
        })
        .collect();

    let z_base: Vec<F> = cfg_into_iter!(0..inner_width)
        .map(|k| {
            let block_idx = k / depth_commit;
            let digit_idx = k % depth_commit;
            consistency_weight * opening_point.a[block_idx] * g1_commit[digit_idx]
        })
        .collect();

    let z_segment: Vec<F> = cfg_into_iter!(0..z_len)
        .map(|x| {
            let compound_dig = x / block_len;
            let blk = x % block_len;
            let dc = compound_dig / depth_fold;
            let df = compound_dig % depth_fold;
            let phys_k = blk * depth_commit + dc;
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

    let z_first = layout.m_vars >= layout.r_vars;
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

/// Shared indexing for the delegated matrix-weight tensor on single proofs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SingleProofMatrixWeightGeometry {
    pub d_start: usize,
    pub b_start: usize,
    pub a_start: usize,
    pub depth_open: usize,
    pub depth_commit: usize,
    pub depth_fold: usize,
    pub log_basis: u32,
    pub num_blocks: usize,
    pub block_len: usize,
    pub n_d: usize,
    pub n_b: usize,
    pub n_a: usize,
    pub d_matrix_width: usize,
    pub inner_width: usize,
    pub t_compound_per_block: usize,
    pub outer_width: usize,
    pub max_row: usize,
    pub offset_z: usize,
    pub offset_w: usize,
    pub offset_t: usize,
}

pub(crate) fn single_proof_matrix_weight_geometry(
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> SingleProofMatrixWeightGeometry {
    let depth_open = layout.num_digits_open;
    let depth_commit = layout.num_digits_commit;
    let depth_fold = layout.num_digits_fold;
    let log_basis = layout.log_basis;
    let num_blocks = layout.num_blocks;
    let block_len = layout.block_len;
    let n_d = level_params.n_d;
    let n_b = level_params.n_b;
    let n_a = level_params.n_a;
    let d_matrix_width = layout.d_matrix_width;
    let inner_width = block_len * depth_commit;
    let t_compound_per_block = n_a * depth_open;
    let outer_width = t_compound_per_block * num_blocks;
    let max_row = n_d.max(n_b).max(n_a);
    let d_start = 2;
    let b_start = d_start + n_d;
    let a_start = b_start + n_b;

    let w_len = depth_open * num_blocks;
    let t_len = depth_open * n_a * num_blocks;
    let z_len = depth_fold * inner_width;
    let z_first = layout.m_vars >= layout.r_vars;
    let (offset_z, offset_w, offset_t) = if z_first {
        (0, z_len, z_len + w_len)
    } else {
        (w_len + t_len, 0, w_len)
    };

    SingleProofMatrixWeightGeometry {
        d_start,
        b_start,
        a_start,
        depth_open,
        depth_commit,
        depth_fold,
        log_basis,
        num_blocks,
        block_len,
        n_d,
        n_b,
        n_a,
        d_matrix_width,
        inner_width,
        t_compound_per_block,
        outer_width,
        max_row,
        offset_z,
        offset_w,
        offset_t,
    }
}

pub(crate) fn single_proof_matrix_weight_entry<F: FieldCore + CanonicalField>(
    row: usize,
    col: usize,
    eq_tau1: &[F],
    eq_r_x: &[F],
    geometry: SingleProofMatrixWeightGeometry,
    fold_gadget: &[F],
) -> F {
    let mut w2 = F::zero();

    if row < geometry.n_d && col < geometry.d_matrix_width {
        let blk = col / geometry.depth_open;
        let dig = col % geometry.depth_open;
        if blk < geometry.num_blocks {
            let global_x = geometry.offset_w + dig * geometry.num_blocks + blk;
            w2 += eq_tau1[geometry.d_start + row] * eq_r_x[global_x];
        }
    }
    if row < geometry.n_b && col < geometry.outer_width {
        let blk = col / geometry.t_compound_per_block;
        let remainder = col % geometry.t_compound_per_block;
        let a_idx = remainder / geometry.depth_open;
        let digit_idx = remainder % geometry.depth_open;
        if blk < geometry.num_blocks {
            let compound_dig = a_idx * geometry.depth_open + digit_idx;
            let global_x = geometry.offset_t + compound_dig * geometry.num_blocks + blk;
            w2 += eq_tau1[geometry.b_start + row] * eq_r_x[global_x];
        }
    }
    if row < geometry.n_a && col < geometry.inner_width {
        let blk_a = col / geometry.depth_commit;
        let dc = col % geometry.depth_commit;
        if blk_a < geometry.block_len {
            for (df, gadget_val) in fold_gadget.iter().enumerate() {
                let compound_dig = dc * geometry.depth_fold + df;
                let global_x = geometry.offset_z + compound_dig * geometry.block_len + blk_a;
                w2 += eq_tau1[geometry.a_start + row] * (-*gadget_val) * eq_r_x[global_x];
            }
        }
    }

    w2
}

/// Evaluate the MLE of the matrix weight tensor at `(r_row, r_col, r_k)`.
///
/// `matrix_weight[row, col, k] = alpha^k * W2[row, col]` where W2 encodes the
/// column-side weights from the D, B, and A matrix views, weighted by the
/// row weights from `eq_tau1`.
///
/// The evaluation factors as:
///   `alpha_factor(r_k) * (D_row * D_col + B_row * B_col + A_row * A_col)`
///
/// `r_x` is the stage-1 challenge point (fixes the eq_x contribution).
/// This is the single-proof specialization, it still covers arbitrary
/// per-level `n_a/n_b/n_d` row counts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_matrix_weight_at_point<F: FieldCore + CanonicalField, const D: usize>(
    r_row: &[F],
    r_col: &[F],
    r_k: &[F],
    r_x: &[F],
    alpha_pows: &[F],
    eq_tau1: &[F],
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    tensor_layout: SharedMatrixTensorLayout,
) -> Result<F, HachiError> {
    use crate::algebra::poly::multilinear_eval;

    if r_row.len() != tensor_layout.row_vars
        || r_col.len() != tensor_layout.col_vars
        || r_k.len() != tensor_layout.ring_vars
    {
        return Err(HachiError::InvalidInput(
            "matrix weight point does not match shared matrix tensor layout".to_string(),
        ));
    }

    let geometry = single_proof_matrix_weight_geometry(level_params, layout);

    let alpha_factor = multilinear_eval(alpha_pows, r_k)?;

    let eq_r_col = EqPolynomial::evals(r_col);
    let eq_r_x = EqPolynomial::evals(r_x);
    let eq_r_row = EqPolynomial::evals(r_row);

    // D row weight: Σ_{row < n_d} eq(row, r_row) * eta_D[row]
    let d_row_eval: F = (0..geometry.n_d)
        .map(|row| eq_r_row[row] * eq_tau1[geometry.d_start + row])
        .fold(F::zero(), |a, b| a + b);

    // D column weight: Σ_{col < d_matrix_width} eq(col, r_col) * eq_r_x[global_D(col)]
    let d_col_eval: F = (0..geometry.d_matrix_width)
        .map(|col| {
            let blk = col / geometry.depth_open;
            let dig = col % geometry.depth_open;
            let global_x = geometry.offset_w + dig * geometry.num_blocks + blk;
            eq_r_col[col] * eq_r_x[global_x]
        })
        .fold(F::zero(), |a, b| a + b);

    // B row weight: Σ_{row < n_b} eq(row, r_row) * eta_B[row]
    let b_row_eval: F = (0..geometry.n_b)
        .map(|row| eq_r_row[row] * eq_tau1[geometry.b_start + row])
        .fold(F::zero(), |a, b| a + b);

    // B column weight: Σ_{col < outer_width} eq(col, r_col) * eq_r_x[global_B(col)]
    let b_col_eval: F = (0..geometry.outer_width)
        .map(|col| {
            let blk = col / geometry.t_compound_per_block;
            let remainder = col % geometry.t_compound_per_block;
            let a_idx = remainder / geometry.depth_open;
            let digit_idx = remainder % geometry.depth_open;
            let compound_dig = a_idx * geometry.depth_open + digit_idx;
            let global_x = geometry.offset_t + compound_dig * geometry.num_blocks + blk;
            eq_r_col[col] * eq_r_x[global_x]
        })
        .fold(F::zero(), |a, b| a + b);

    // A row weight: Σ_{row < n_a} eq(row, r_row) * eta_A[row]
    let a_row_eval: F = (0..geometry.n_a)
        .map(|row| eq_r_row[row] * eq_tau1[geometry.a_start + row])
        .fold(F::zero(), |a, b| a + b);

    // A column weight: Σ_{col < inner_width} eq(col, r_col) * (Σ_df (-fold_gadget[df]) * eq_r_x[global_A(col, df)])
    let fold_gadget = gadget_row_scalars::<F>(geometry.depth_fold, geometry.log_basis);
    let a_col_eval: F = (0..geometry.inner_width)
        .map(|col| {
            let blk_a = col / geometry.depth_commit;
            let dc = col % geometry.depth_commit;
            let fold_sum: F = (0..geometry.depth_fold)
                .map(|df| {
                    let compound_dig = dc * geometry.depth_fold + df;
                    let global_x = geometry.offset_z + compound_dig * geometry.block_len + blk_a;
                    -fold_gadget[df] * eq_r_x[global_x]
                })
                .fold(F::zero(), |a, b| a + b);
            eq_r_col[col] * fold_sum
        })
        .fold(F::zero(), |a, b| a + b);

    let w2_eval = d_row_eval * d_col_eval + b_row_eval * b_col_eval + a_row_eval * a_col_eval;
    Ok(alpha_factor * w2_eval)
}

fn validate_opening_points_for_claims<F: FieldCore>(
    opening_points: &[RingOpeningPoint<F>],
    claim_to_point: &[usize],
    layout: HachiCommitmentLayout,
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
        if opening_point.a.len() < layout.block_len || opening_point.b.len() != layout.num_blocks {
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

#[tracing::instrument(
    skip_all,
    name = "compute_m_evals_x_with_opening_points_and_claim_groups"
)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_m_evals_x_with_opening_points_and_claim_groups<
    F: FieldCore + CanonicalField,
    const D: usize,
>(
    setup: &HachiExpandedSetup<F>,
    opening_points: &[RingOpeningPoint<F>],
    claim_to_point: &[usize],
    challenges: &[SparseChallenge],
    alpha: F,
    alpha_pows: &[F],
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    tau1: &[F],
    claim_group_sizes: &[usize],
) -> Result<Vec<F>, HachiError> {
    if alpha_pows.len() != D {
        return Err(HachiError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }
    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    validate_opening_points_for_claims(opening_points, claim_to_point, layout, num_claims)?;
    let num_commitment_groups = claim_group_sizes.len();

    let depth_commit = layout.num_digits_commit;
    let depth_open = layout.num_digits_open;
    let depth_fold = layout.num_digits_fold;
    let log_basis = layout.log_basis;
    let num_blocks = layout.num_blocks;
    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| HachiError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.len() != total_blocks {
        return Err(HachiError::InvalidSize {
            expected: total_blocks,
            actual: challenges.len(),
        });
    }
    let block_len = layout.block_len;
    let w_len = depth_open * total_blocks;
    let t_len = depth_open * level_params.n_a * total_blocks;
    let inner_width = block_len * depth_commit;
    let z_base_len = opening_points
        .len()
        .checked_mul(inner_width)
        .ok_or_else(|| HachiError::InvalidSetup("multipoint z width overflow".to_string()))?;
    let z_len = depth_fold
        .checked_mul(z_base_len)
        .ok_or_else(|| HachiError::InvalidSetup("multipoint z width overflow".to_string()))?;
    let rows = if num_claims == 1 && num_commitment_groups == 1 && opening_points.len() == 1 {
        level_params.m_row_count()
    } else {
        level_params
            .m_row_count_with_commitments_and_public_outputs(num_commitment_groups, num_claims)
    };
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

    let stride = setup.seed.max_stride();
    let d_view = setup.shared_matrix.ring_view::<D>(level_params.n_d, stride);
    let b_view = setup.shared_matrix.ring_view::<D>(level_params.n_b, stride);
    let a_view = setup.shared_matrix.ring_view::<D>(level_params.n_a, stride);

    // Row layout: consistency (1) | public (num_claims) | D (n_d) |
    //             B (n_b * num_commitment_groups) | A (n_a)
    let commitment_row_count = level_params.n_b * num_commitment_groups;
    let consistency_weight = eq_tau1[0];
    let public_weights = &eq_tau1[1..(1 + num_claims)];
    let d_start = 1 + num_claims;
    let b_start = d_start + level_params.n_d;
    let a_start = b_start + commitment_row_count;
    let a_weights = &eq_tau1[a_start..rows];
    let claim_to_group: Vec<(usize, usize)> = claim_group_sizes
        .iter()
        .enumerate()
        .flat_map(|(group_idx, &group_size)| {
            (0..group_size).map(move |within_group| (group_idx, within_group))
        })
        .collect();

    // --- Digit-major segments (block index innermost) ---

    let n_a = level_params.n_a;
    let t_compound_per_block = n_a * depth_open;

    let w_segment: Vec<F> = cfg_into_iter!(0..w_len)
        .map(|x| {
            let dig = x / total_blocks;
            let blk = x % total_blocks;
            let claim_idx = blk / num_blocks;
            let block_idx = blk % num_blocks;
            let d_phys_col = blk * depth_open + dig;
            let opening_point = &opening_points[claim_to_point[claim_idx]];
            let mut acc = (public_weights[claim_idx] * opening_point.b[block_idx]
                + consistency_weight * c_alphas[blk])
                * g1_open[dig];
            for (di, eq_i) in eq_tau1[d_start..(d_start + level_params.n_d)]
                .iter()
                .enumerate()
            {
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
            let commitment_weights = &eq_tau1[(b_start + group_idx * level_params.n_b)
                ..(b_start + (group_idx + 1) * level_params.n_b)];
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

    let z_first = layout.m_vars >= layout.r_vars;
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

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "prepare_m_eval")]
pub(crate) fn prepare_m_eval<F: FieldCore + CanonicalField, const D: usize>(
    challenges: &[SparseChallenge],
    alpha: F,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    tau1: &[F],
    claim_group_sizes: &[usize],
    opening_points_len: usize,
    claim_to_point: &[usize],
) -> Result<PreparedMEval<F>, HachiError> {
    let alpha_pows = build_alpha_evals_y(alpha, D);
    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    let num_commitment_groups = claim_group_sizes.len();

    let depth_commit = layout.num_digits_commit;
    let depth_open = layout.num_digits_open;
    let depth_fold = layout.num_digits_fold;
    let log_basis = layout.log_basis;
    let num_blocks = layout.num_blocks;
    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| HachiError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.len() != total_blocks {
        return Err(HachiError::InvalidSize {
            expected: total_blocks,
            actual: challenges.len(),
        });
    }
    let block_len = layout.block_len;
    let inner_width = block_len * depth_commit;
    let num_points = opening_points_len.max(1);
    let rows = if num_claims == 1 && num_commitment_groups == 1 && num_points <= 1 {
        level_params.m_row_count()
    } else {
        level_params
            .m_row_count_with_commitments_and_public_outputs(num_commitment_groups, num_claims)
    };

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

    let z_first = layout.m_vars >= layout.r_vars;

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
        n_a: level_params.n_a,
        n_d: level_params.n_d,
        n_b: level_params.n_b,
        num_commitment_groups,
        rows,
        z_first,
        claim_to_group,
        num_points,
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

        let stride = setup.seed.max_stride();
        let d_view = setup.shared_matrix.ring_view::<D>(self.n_d, stride);
        let b_view = setup.shared_matrix.ring_view::<D>(self.n_b, stride);
        let a_view = setup.shared_matrix.ring_view::<D>(self.n_a, stride);

        let consistency_weight = self.eq_tau1[0];
        let public_weights = &self.eq_tau1[1..(1 + self.num_claims)];
        let d_start = 1 + self.num_claims;
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
                let public_scale = public_weights[claim_idx] * g_open;
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
    layout: HachiCommitmentLayout,
) -> RecursiveWitnessFlat {
    let log_basis = layout.log_basis;
    let num_digits_fold = layout.num_digits_fold;
    let depth_open = layout.num_digits_open;
    let depth_commit = layout.num_digits_commit;
    let block_len = layout.block_len;
    let levels = r_decomp_levels::<F>(log_basis);

    let w_hat_planes = w_hat.flat_digits().len();
    let t_hat_planes = t_hat.flat_digits().len();
    let z_count = w_hat_planes + t_hat_planes + z_pre_centered.len() * num_digits_fold;
    let r_hat_count = r.len() * levels;
    let z_first = layout.m_vars >= layout.r_vars;
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
        build_alpha_evals_y, build_w_evals_compact, commit_w,
        compute_alg_m_evals_x_with_claim_groups, compute_m_evals_x,
        compute_m_evals_x_with_claim_groups, compute_r_via_poly_division, eval_ring_at_pows,
        eval_matrix_weight_at_point, gadget_row_scalars, m_row_count, prepare_m_eval,
        ring_switch_build_w,
        WCommitmentConfig,
    };
    use crate::algebra::eq_poly::EqPolynomial;
    use crate::algebra::poly::multilinear_eval;
    use crate::algebra::CyclotomicRing;
    use crate::protocol::commitment::AppendToTranscript;
    use crate::protocol::commitment::{
        hachi_recursive_level_layout_from_params, presets::fp128, HachiCommitmentCore,
        HachiScheduleInputs, RingCommitmentScheme, SmallTestCommitmentConfig,
    };
    use crate::protocol::commitment_scheme::HachiCommitmentScheme;
    use crate::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps, RecursiveWitnessFlat};
    use crate::protocol::opening_point::{ring_opening_point_from_field, BasisMode, BlockOrder};
    use crate::protocol::quadratic_equation::QuadraticEquation;
    use crate::protocol::sumcheck::hachi_stage2::relation_claim_from_rows;
    use crate::protocol::transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::protocol::CommitmentConfig;
    use crate::test_utils::F as TestF;
    use crate::{CanonicalField, CommitmentScheme, Transcript};
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

        let layout = Cfg::commitment_layout(NV).expect("layout");
        let level_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: NV,
            level: 0,
            current_w_len: 1usize << NV,
        });

        let mut rng = StdRng::seed_from_u64(0x5eed_cafe);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(NV, 1);
        let (commitment, batched_hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<
            F,
            D,
        >>::commit(&[poly.clone()], &setup)
        .expect("commitment");
        let hint = batched_hint.into_flattened();

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            layout.r_vars,
            layout.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("ring opening point");
        let (y_ring, w_folded) = poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            layout.block_len,
        );

        let mut transcript = Blake2bTranscript::<F>::new(b"ring-switch-row-regression");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

        let mut quad_eq = QuadraticEquation::<F, D, Cfg>::new_prover(
            &setup.ntt_shared,
            ring_opening_point,
            &poly,
            w_folded,
            level_params.clone(),
            hint,
            &mut transcript,
            &commitment,
            &y_ring,
            layout,
            setup.expanded.seed.max_stride(),
        )
        .expect("quadratic equation");

        let w = ring_switch_build_w::<F, D, Cfg>(
            &mut quad_eq,
            &setup.expanded,
            &setup.ntt_shared,
            &level_params,
            layout,
        )
        .expect("ring-switch witness");
        let (w_compact, _col_bits, ring_bits) =
            build_w_evals_compact(w.as_i8_digits(), D).expect("compact witness");
        let live_x_cols = w_compact.len() >> ring_bits;

        let alpha = F::from_u64(17);
        let alpha_evals_y = build_alpha_evals_y(alpha, D);
        let rows = m_row_count(&level_params);
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
                quad_eq.opening_point(),
                &quad_eq.challenges,
                alpha,
                &alpha_evals_y,
                &level_params,
                layout,
                &tau1,
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
        use crate::protocol::commitment::compute_num_digits_full_field;

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
    fn commit_w_uses_active_level_row_count() {
        type Cfg = SmallTestCommitmentConfig;
        type WCfg = WCommitmentConfig<32, Cfg>;
        const D: usize = 32;

        let (setup, _) = <HachiCommitmentCore as RingCommitmentScheme<TestF, D, Cfg>>::setup(12, 1)
            .expect("setup");
        assert!(
            setup.ntt_shared.total_elements() > 3,
            "test needs a shared cache envelope"
        );

        let w = RecursiveWitnessFlat::from_i8_digits(
            (0..(19 * D)).map(|i| ((i % 7) as i8) - 3).collect(),
        );
        let mut level_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: 12,
            level: 1,
            current_w_len: w.len(),
        });
        level_params.n_a = 3;

        let expected_layout =
            hachi_recursive_level_layout_from_params::<WCfg>(&level_params, w.len())
                .expect("layout");
        let (_commitment, hint) = commit_w::<TestF, D, WCfg>(
            &w,
            &setup.ntt_shared,
            &level_params,
            setup.expanded.seed.max_stride(),
        )
        .expect("commit w");
        let t = hint
            .t()
            .expect("commit_w should preserve recomposed t rows");

        assert_eq!(t.len(), expected_layout.num_blocks);
        assert!(
            t.iter().all(|block| block.len() == level_params.n_a),
            "every block should use the active n_a rows"
        );
        assert!(
            hint.inner_opening_digits
                .iter()
                .all(|block| block.len() == level_params.n_a * expected_layout.num_digits_open),
            "t_hat should also use the active n_a rows"
        );
    }

    #[test]
    fn prepared_m_eval_matches_materialized() {
        use crate::protocol::sumcheck::multilinear_eval;

        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
        const NV: usize = 12;

        let layout = Cfg::commitment_layout(NV).expect("layout");
        let level_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: NV,
            level: 0,
            current_w_len: 1usize << NV,
        });

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(NV, 1);
        let (commitment, batched_hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<
            F,
            D,
        >>::commit(&[poly.clone()], &setup)
        .expect("commitment");
        let hint = batched_hint.into_flattened();

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            layout.r_vars,
            layout.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("ring opening point");
        let (y_ring, w_folded) = poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            layout.block_len,
        );

        let mut transcript = Blake2bTranscript::<F>::new(b"prepared-m-eval-test");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

        let mut quad_eq = QuadraticEquation::<F, D, Cfg>::new_prover(
            &setup.ntt_shared,
            ring_opening_point.clone(),
            &poly,
            w_folded,
            level_params.clone(),
            hint,
            &mut transcript,
            &commitment,
            &y_ring,
            layout,
            setup.expanded.seed.max_stride(),
        )
        .expect("quadratic equation");

        ring_switch_build_w::<F, D, Cfg>(
            &mut quad_eq,
            &setup.expanded,
            &setup.ntt_shared,
            &level_params,
            layout,
        )
        .expect("ring-switch witness");

        let alpha = F::from_u64(42);
        let alpha_evals_y = build_alpha_evals_y(alpha, D);
        let rows = m_row_count(&level_params);
        let num_i = rows.next_power_of_two().trailing_zeros() as usize;
        let tau1: Vec<F> = (0..num_i)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let m_evals_x = compute_m_evals_x::<F, D>(
            &setup.expanded,
            &ring_opening_point,
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            layout,
            &tau1,
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
            layout,
            &tau1,
            &[1usize],
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

    #[test]
    fn alg_plus_setup_equals_m_evals_x() {
        type F = fp128::Field;
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;
        const NV: usize = 12;

        let layout = Cfg::commitment_layout(NV).expect("layout");
        let level_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: NV,
            level: 0,
            current_w_len: 1usize << NV,
        });

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let evals: Vec<F> = (0..(1usize << NV))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point: Vec<F> = (0..NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(NV, 1);
        let (commitment, batched_hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<
            F,
            D,
        >>::commit(&[poly.clone()], &setup)
        .expect("commitment");
        let hint = batched_hint.into_flattened();

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            layout.r_vars,
            layout.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("ring opening point");

        let mut transcript = Blake2bTranscript::<F>::new(b"alg-setup-split-test");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        let (y_ring, w_folded) = poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            layout.block_len,
        );
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

        let quad_eq = QuadraticEquation::<F, D, Cfg>::new_prover(
            &setup.ntt_shared,
            ring_opening_point,
            &poly,
            w_folded,
            level_params.clone(),
            hint,
            &mut transcript,
            &commitment,
            &y_ring,
            layout,
            setup.expanded.seed.max_stride(),
        )
        .expect("quadratic equation");

        let alpha = F::from_u64(42);
        let alpha_evals_y = build_alpha_evals_y(alpha, D);
        let rows = m_row_count(&level_params);
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

            let full = compute_m_evals_x_with_claim_groups::<F, D>(
                &setup.expanded,
                quad_eq.opening_point(),
                &quad_eq.challenges,
                alpha,
                &alpha_evals_y,
                &level_params,
                layout,
                &tau1,
                &[1usize],
            )
            .expect("full m_evals_x");

            let alg = compute_alg_m_evals_x_with_claim_groups::<F, D>(
                quad_eq.opening_point(),
                &quad_eq.challenges,
                alpha,
                &alpha_evals_y,
                &level_params,
                layout,
                &tau1,
                &[1usize],
            )
            .expect("alg m_evals_x");

            assert_eq!(full.len(), alg.len(), "length mismatch at row {row}");
            for (x, (f, a)) in full.iter().zip(alg.iter()).enumerate() {
                let setup_part = *f - *a;
                assert_eq!(
                    *f,
                    *a + setup_part,
                    "alg + setup != full at row={row}, x={x}"
                );
            }
        }

        // The D-rows sit at indices d_start..d_start+n_d in eq_tau1.
        // Pick a row whose eq_tau1 has weight there to confirm setup is nonzero.
        let d_start = 1 + 1; // consistency(1) + public(num_claims=1)
        let tau1_for_d: Vec<F> = (0..num_i)
            .map(|bit| {
                if (d_start >> bit) & 1 == 1 {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect();
        let full_d = compute_m_evals_x_with_claim_groups::<F, D>(
            &setup.expanded,
            quad_eq.opening_point(),
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            layout,
            &tau1_for_d,
            &[1usize],
        )
        .expect("full at d_start");
        let alg_d = compute_alg_m_evals_x_with_claim_groups::<F, D>(
            quad_eq.opening_point(),
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            layout,
            &tau1_for_d,
            &[1usize],
        )
        .expect("alg at d_start");
        let has_nonzero_setup = full_d.iter().zip(alg_d.iter()).any(|(f, a)| *f != *a);
        assert!(
            has_nonzero_setup,
            "expected nonzero setup contribution for D-row weight"
        );
    }

    fn matrix_weight_inner_product_equals_setup_residue_for_cfg<const D: usize, Cfg>(nv: usize)
    where
        Cfg: crate::protocol::shared_matrix_setup::SharedMatrixOpeningConfig<Field = fp128::Field>,
    {
        type F = fp128::Field;

        let layout = Cfg::commitment_layout(nv).expect("layout");
        let level_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: nv,
            level: 0,
            current_w_len: 1usize << nv,
        });

        let mut rng = StdRng::seed_from_u64(0xcafe_babe);
        let evals: Vec<F> = (0..(1usize << nv))
            .map(|i| F::from_u64((i % 2) as u64))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).expect("dense poly");
        let point: Vec<F> = (0..nv)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(nv, 1);
        let (commitment, batched_hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<
            F,
            D,
        >>::commit(&[poly.clone()], &setup)
        .expect("commitment");
        let hint = batched_hint.into_flattened();

        let alpha_bits = D.trailing_zeros() as usize;
        let outer_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            outer_point,
            layout.r_vars,
            layout.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("ring opening point");

        let mut transcript = Blake2bTranscript::<F>::new(b"w-env-test");
        commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
        for pt in &point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        let (y_ring, w_folded) = poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            layout.block_len,
        );
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

        let quad_eq = QuadraticEquation::<F, D, Cfg>::new_prover(
            &setup.ntt_shared,
            ring_opening_point,
            &poly,
            w_folded,
            level_params.clone(),
            hint,
            &mut transcript,
            &commitment,
            &y_ring,
            layout,
            setup.expanded.seed.max_stride(),
        )
        .expect("quadratic equation");

        let alpha = F::from_u64(42);
        let alpha_evals_y = build_alpha_evals_y(alpha, D);

        let depth_open = layout.num_digits_open;
        let depth_commit = layout.num_digits_commit;
        let depth_fold = layout.num_digits_fold;
        let log_basis = layout.log_basis;
        let num_blocks = layout.num_blocks;
        let block_len = layout.block_len;
        let n_d = level_params.n_d;
        let n_b = level_params.n_b;
        let n_a = level_params.n_a;
        let d_matrix_width = layout.d_matrix_width;
        let inner_width = block_len * depth_commit;
        let t_compound_per_block = n_a * depth_open;
        let outer_width = t_compound_per_block * num_blocks;
        let stride = setup.expanded.seed.max_stride();
        let max_row = n_d.max(n_b).max(n_a);

        let w_len = depth_open * num_blocks;
        let t_len = depth_open * n_a * num_blocks;
        let z_len = depth_fold * inner_width;
        let rows = m_row_count(&level_params);

        let z_first = layout.m_vars >= layout.r_vars;
        let (offset_z, offset_w, offset_t) = if z_first {
            (0, z_len, z_len + w_len)
        } else {
            (w_len + t_len, 0, w_len)
        };

        let num_i = rows.next_power_of_two().trailing_zeros() as usize;
        let tau1: Vec<F> = (0..num_i)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let eq_tau1_full = EqPolynomial::evals(&tau1);

        let full = compute_m_evals_x_with_claim_groups::<F, D>(
            &setup.expanded,
            quad_eq.opening_point(),
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            layout,
            &tau1,
            &[1usize],
        )
        .expect("full m_evals_x");

        let alg = compute_alg_m_evals_x_with_claim_groups::<F, D>(
            quad_eq.opening_point(),
            &quad_eq.challenges,
            alpha,
            &alpha_evals_y,
            &level_params,
            layout,
            &tau1,
            &[1usize],
        )
        .expect("alg m_evals_x");

        let setup_vec: Vec<F> = full.iter().zip(alg.iter()).map(|(f, a)| *f - *a).collect();

        let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
        let d_start: usize = 2;
        let b_start = d_start + n_d;
        let a_start = b_start + n_b;

        let x_len = full.len();
        let r_x: Vec<F> = (0..(x_len.trailing_zeros() as usize))
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let eq_r_x = EqPolynomial::evals(&r_x);

        let mut inner_product = F::zero();
        let sm_view = setup.expanded.shared_matrix.ring_view::<D>(max_row, stride);
        for row in 0..max_row {
            for col in 0..stride {
                let sbar = eval_ring_at_pows(&sm_view.row(row)[col], &alpha_evals_y);
                let mut w2 = F::zero();

                if row < n_d && col < d_matrix_width {
                    let blk = col / depth_open;
                    let dig = col % depth_open;
                    if blk < num_blocks {
                        let global_x = offset_w + dig * num_blocks + blk;
                        w2 += eq_tau1_full[d_start + row] * eq_r_x[global_x];
                    }
                }
                if row < n_b && col < outer_width {
                    let blk = col / t_compound_per_block;
                    let remainder = col % t_compound_per_block;
                    let a_idx = remainder / depth_open;
                    let digit_idx = remainder % depth_open;
                    if blk < num_blocks {
                        let compound_dig = a_idx * depth_open + digit_idx;
                        let global_x = offset_t + compound_dig * num_blocks + blk;
                        w2 += eq_tau1_full[b_start + row] * eq_r_x[global_x];
                    }
                }
                if row < n_a && col < inner_width {
                    let blk_a = col / depth_commit;
                    let dc = col % depth_commit;
                    if blk_a < block_len {
                        for df in 0..depth_fold {
                            let compound_dig = dc * depth_fold + df;
                            let global_x = offset_z + compound_dig * block_len + blk_a;
                            w2 +=
                                eq_tau1_full[a_start + row] * (-fold_gadget[df]) * eq_r_x[global_x];
                        }
                    }
                }
                inner_product += sbar * w2;
            }
        }

        let expected = multilinear_eval(&setup_vec, &r_x).expect("multilinear_eval");
        assert_eq!(
            inner_product, expected,
            "inner product <shared_matrix, matrix_weight> must equal setup(r_x)"
        );

        let tensor_layout =
            crate::protocol::shared_matrix_setup::SharedMatrixTensorLayout::from_expanded::<
                F,
                Cfg,
                D,
            >(&setup.expanded);
        let col_vars = tensor_layout.col_vars;
        let row_vars = tensor_layout.row_vars;
        let k_vars = tensor_layout.ring_vars;

        let mut weight_table = vec![F::zero(); tensor_layout.field_len()];
        for row in 0..max_row {
            for col in 0..stride {
                let mut w2 = F::zero();
                if row < n_d && col < d_matrix_width {
                    let blk = col / depth_open;
                    let dig = col % depth_open;
                    if blk < num_blocks {
                        let global_x = offset_w + dig * num_blocks + blk;
                        w2 += eq_tau1_full[d_start + row] * eq_r_x[global_x];
                    }
                }
                if row < n_b && col < outer_width {
                    let blk = col / t_compound_per_block;
                    let remainder = col % t_compound_per_block;
                    let a_idx = remainder / depth_open;
                    let digit_idx = remainder % depth_open;
                    if blk < num_blocks {
                        let compound_dig = a_idx * depth_open + digit_idx;
                        let global_x = offset_t + compound_dig * num_blocks + blk;
                        w2 += eq_tau1_full[b_start + row] * eq_r_x[global_x];
                    }
                }
                if row < n_a && col < inner_width {
                    let blk_a = col / depth_commit;
                    let dc = col % depth_commit;
                    if blk_a < block_len {
                        for df in 0..depth_fold {
                            let compound_dig = dc * depth_fold + df;
                            let global_x = offset_z + compound_dig * block_len + blk_a;
                            w2 +=
                                eq_tau1_full[a_start + row] * (-fold_gadget[df]) * eq_r_x[global_x];
                        }
                    }
                }
                for k in 0..D {
                    let flat_idx = tensor_layout.flat_index(row, col, k);
                    weight_table[flat_idx] = alpha_evals_y[k] * w2;
                }
            }
        }

        let r_row: Vec<F> = (0..row_vars)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let r_col: Vec<F> = (0..col_vars)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let r_k_point: Vec<F> = (0..k_vars)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let mut full_point = Vec::with_capacity(k_vars + col_vars + row_vars);
        full_point.extend_from_slice(&r_k_point);
        full_point.extend_from_slice(&r_col);
        full_point.extend_from_slice(&r_row);

        let expected_eval =
            multilinear_eval(&weight_table, &full_point).expect("multilinear_eval matrix_weight");

        let got_eval = eval_matrix_weight_at_point::<F, D>(
            &r_row,
            &r_col,
            &r_k_point,
            &r_x,
            &alpha_evals_y,
            &eq_tau1_full,
            &level_params,
            layout,
            tensor_layout,
        )
        .expect("eval_matrix_weight_at_point");

        assert_eq!(
            got_eval, expected_eval,
            "eval_matrix_weight_at_point must match brute-force MLE evaluation"
        );
    }

    #[test]
    fn matrix_weight_inner_product_equals_setup_residue() {
        matrix_weight_inner_product_equals_setup_residue_for_cfg::<
            { fp128::D128Full::D },
            fp128::D128Full,
        >(12);
    }

    #[test]
    fn matrix_weight_matches_setup_residue_for_multirow_onehot_root() {
        type MultiRowCfg = fp128::D32OneHot;
        const D: usize = MultiRowCfg::D;
        const NV: usize = 12;

        let level_params = MultiRowCfg::level_params(HachiScheduleInputs {
            max_num_vars: NV,
            level: 0,
            current_w_len: 1usize << NV,
        });
        assert!(
            level_params.n_a > 1,
            "fixture must exercise the multi-row delegated path"
        );

        matrix_weight_inner_product_equals_setup_residue_for_cfg::<D, MultiRowCfg>(NV);
    }
}
