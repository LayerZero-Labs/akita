//! Ring switching logic for the Hachi PCS (Section 4.3).
//!
//! Handles the transition from the ring-based quadratic equation to field-based
//! sumcheck instances by expanding the ring elements into their coefficient
//! vectors and setting up the evaluation tables.

use crate::algebra::eq_poly::EqPolynomial;
use crate::algebra::{CyclotomicRing, SparseChallenge};
use crate::cfg_into_iter;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::linear::{flatten_i8_blocks, mat_vec_mul_ntt_single_i8};
use crate::protocol::commitment::utils::norm::detect_field_modulus;
use crate::protocol::commitment::{
    hachi_recursive_level_layout_from_params, recursive_level_decomposition_from_root,
    recursive_r_decomp_levels_for_bound, CommitmentConfig, CommitmentEnvelope, DecompositionParams,
    HachiCommitmentLayout, HachiExpandedSetup, HachiLevelParams, HachiScheduleInputs,
    RingCommitment,
};
use crate::protocol::hachi_poly_ops::RecursiveWitnessFlat;
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::proof::{FlatRingVec, HachiCommitmentHint, ProofRingVec};
use crate::protocol::quadratic_equation::{compute_r_split_eq, QuadraticEquation};
use crate::protocol::recursive_runtime::RecursiveCommitmentHintCache;
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
    /// Compact evaluation table of w, stored as y-major slices of the live x prefix.
    /// Populated by the prover; empty on the verifier side.
    pub w_evals_compact: Vec<i8>,
    /// Physical x width before zero-extension to the next power of two.
    pub live_x_cols: usize,
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

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub(crate) struct RingSwitchVerifyOutput<F: FieldCore> {
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
    Cfg: CommitmentConfig,
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
    let w_hat_flat = quad_eq
        .take_w_hat_flat()
        .ok_or_else(|| HachiError::InvalidInput("missing w_hat_flat in prover".to_string()))?;
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
        &w_hat_flat,
        inner_opening_digits,
        t,
        &w_folded,
        &z_pre.centered_coeffs,
        z_pre.centered_inf_norm,
        quad_eq.y(),
        quad_eq.claim_group_sizes(),
        layout.num_blocks,
        layout.inner_width,
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
    w_commitment_proof: &ProofRingVec<F>,
    w_hint: RecursiveCommitmentHintCache<F>,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> Result<RingSwitchOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
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
    w_commitment_proof: &ProofRingVec<F>,
    w_hint: RecursiveCommitmentHintCache<F>,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> Result<RingSwitchOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    transcript.append_serde(ABSORB_SUMCHECK_W, w_commitment_proof);

    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

    let claim_group_sizes = quad_eq.claim_group_sizes();
    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    let num_commitment_groups = claim_group_sizes.len();

    let num_l = D.trailing_zeros() as usize;
    let num_ring_elems = w.len() / D;
    let live_x_cols = num_ring_elems;
    let num_u = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let m_rows = if num_claims == 1 && num_commitment_groups == 1 {
        m_row_count(level_params)
    } else {
        level_params
            .m_row_count_with_commitments_and_public_outputs(num_commitment_groups, num_claims)
    };
    let num_sc_vars = num_u + num_l;
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
    setup: &HachiExpandedSetup<F>,
    w_len: usize,
    w_commitment: &ProofRingVec<F>,
    transcript: &mut T,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> Result<RingSwitchVerifyOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    ring_switch_verifier_with_num_claims::<F, T, D>(
        opening_point,
        challenges,
        setup,
        w_len,
        w_commitment,
        transcript,
        level_params,
        layout,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "ring_switch_verifier_with_num_claims")]
#[inline(never)]
pub(crate) fn ring_switch_verifier_with_num_claims<F, T, const D: usize>(
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    setup: &HachiExpandedSetup<F>,
    w_len: usize,
    w_commitment: &ProofRingVec<F>,
    transcript: &mut T,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    num_claims: usize,
) -> Result<RingSwitchVerifyOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    let claim_group_sizes = vec![1usize; num_claims];
    ring_switch_verifier_with_claim_groups::<F, T, D>(
        opening_point,
        challenges,
        setup,
        w_len,
        w_commitment,
        transcript,
        level_params,
        layout,
        &claim_group_sizes,
    )
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "ring_switch_verifier_with_claim_groups")]
#[inline(never)]
pub(crate) fn ring_switch_verifier_with_claim_groups<F, T, const D: usize>(
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    setup: &HachiExpandedSetup<F>,
    w_len: usize,
    w_commitment: &ProofRingVec<F>,
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
    let num_commitment_groups = claim_group_sizes.len();

    let num_ring_elems = w_len / D;
    let num_u = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let num_l = D.trailing_zeros() as usize;
    let m_rows = if num_claims == 1 && num_commitment_groups == 1 {
        m_row_count(level_params)
    } else {
        level_params
            .m_row_count_with_commitments_and_public_outputs(num_commitment_groups, num_claims)
    };
    let num_sc_vars = num_u + num_l;
    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;

    let tau0 = sample_tau::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);
    let tau1 = sample_tau::<F, T>(transcript, CHALLENGE_TAU1, num_i);
    let alpha_evals_y = build_alpha_evals_y(alpha, D);
    let m_evals_x = if num_claims == 1 && num_commitment_groups == 1 {
        compute_m_evals_x::<F, D>(
            setup,
            opening_point,
            challenges,
            alpha,
            &alpha_evals_y,
            level_params,
            layout,
            &tau1,
        )?
    } else {
        compute_m_evals_x_with_claim_groups::<F, D>(
            setup,
            opening_point,
            challenges,
            alpha,
            &alpha_evals_y,
            level_params,
            layout,
            &tau1,
            claim_group_sizes,
        )?
    };

    Ok(RingSwitchVerifyOutput {
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
    setup: &HachiExpandedSetup<F>,
    w_len: usize,
    w_commitment: &ProofRingVec<F>,
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
    let num_u = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let num_l = D.trailing_zeros() as usize;
    let m_rows = if num_claims == 1 && num_commitment_groups == 1 && opening_points.len() == 1 {
        m_row_count(level_params)
    } else {
        level_params
            .m_row_count_with_commitments_and_public_outputs(num_commitment_groups, num_claims)
    };
    let num_sc_vars = num_u + num_l;
    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;

    let tau0 = sample_tau::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);
    let tau1 = sample_tau::<F, T>(transcript, CHALLENGE_TAU1, num_i);
    let alpha_evals_y = build_alpha_evals_y(alpha, D);
    let m_evals_x = if opening_points.len() == 1 {
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

    Ok(RingSwitchVerifyOutput {
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
/// For `Cfg=Fp128FullCommitmentConfig`, this uses the same decomposition
/// parameters as
/// [`Fp128LogBasisCommitmentConfig`](super::commitment::Fp128LogBasisCommitmentConfig),
/// but at the caller-selected ring dimension `D`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WCommitmentConfig<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

impl<const D: usize, Cfg: CommitmentConfig> CommitmentConfig for WCommitmentConfig<D, Cfg> {
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

    fn schedule_key(max_num_vars: usize) -> String {
        Cfg::schedule_key(max_num_vars)
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

pub(crate) fn w_ring_element_count_with_num_claims_and_points<F: CanonicalField>(
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    num_claims: usize,
    num_points: usize,
) -> usize {
    w_ring_element_count_with_counts::<F>(level_params, layout, num_claims, num_claims, num_points)
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
) -> Result<(RingCommitment<F, D>, HachiCommitmentHint<F, D>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    if w.len() % D != 0 {
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
    )?;

    let t_hat_flat = flatten_i8_blocks(&inner.t_hat);
    let u: Vec<CyclotomicRing<F, D>> =
        mat_vec_mul_ntt_single_i8(ntt_shared, level_params.n_b, &t_hat_flat);
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

/// Produce the compact `Vec<i8>` eval table of `w` for the fused prover,
/// storing only the physical x prefix for each y slice.
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
    let live_x_cols = w.len() / d;
    let num_u = live_x_cols.next_power_of_two().trailing_zeros() as usize;

    let mut compact = vec![0i8; w.len()];

    #[cfg(feature = "parallel")]
    compact
        .par_chunks_mut(live_x_cols)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, dst) in row.iter_mut().enumerate() {
                *dst = w[y + (x << num_l)];
            }
        });

    #[cfg(not(feature = "parallel"))]
    for (y, row) in compact.chunks_mut(live_x_cols).enumerate() {
        for (x, dst) in row.iter_mut().enumerate() {
            *dst = w[y + (x << num_l)];
        }
    }
    Ok((compact, num_u, num_l))
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
    compute_m_evals_x_with_num_claims::<F, D>(
        setup,
        opening_point,
        challenges,
        alpha,
        alpha_pows,
        level_params,
        layout,
        tau1,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_m_evals_x_with_num_claims")]
pub(crate) fn compute_m_evals_x_with_num_claims<F: FieldCore + CanonicalField, const D: usize>(
    setup: &HachiExpandedSetup<F>,
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    alpha: F,
    alpha_pows: &[F],
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    tau1: &[F],
    num_claims: usize,
) -> Result<Vec<F>, HachiError> {
    let claim_group_sizes = vec![1usize; num_claims];
    compute_m_evals_x_with_claim_groups::<F, D>(
        setup,
        opening_point,
        challenges,
        alpha,
        alpha_pows,
        level_params,
        layout,
        tau1,
        &claim_group_sizes,
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

    let shared_view = setup.shared_matrix.view::<D>();

    let commitment_row_count = level_params.n_b * num_commitment_groups;
    let public_row_start = level_params.n_d + commitment_row_count;
    let row3_weights = &eq_tau1[public_row_start..(public_row_start + num_claims)];
    let row4_weight = eq_tau1[public_row_start + num_claims];
    let a_weights = &eq_tau1[(public_row_start + num_claims + 1)..rows];
    let claim_to_group: Vec<(usize, usize)> = claim_group_sizes
        .iter()
        .enumerate()
        .flat_map(|(group_idx, &group_size)| {
            (0..group_size).map(move |within_group| (group_idx, within_group))
        })
        .collect();

    let w_segment: Vec<F> = cfg_into_iter!(0..w_len)
        .map(|x| {
            let blocks_per_claim = num_blocks * depth_open;
            let claim_idx = x / blocks_per_claim;
            let claim_offset = x % blocks_per_claim;
            let block_idx = claim_offset / depth_open;
            let digit_idx = claim_offset % depth_open;
            let global_block_idx = claim_idx * num_blocks + block_idx;
            let mut acc = (row3_weights[claim_idx] * opening_point.b[block_idx]
                + row4_weight * c_alphas[global_block_idx])
                * g1_open[digit_idx];
            for (row_idx, eq_i) in eq_tau1.iter().enumerate().take(level_params.n_d) {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&shared_view.row(row_idx)[x], alpha_pows);
                }
            }
            acc
        })
        .collect();
    out.extend(w_segment);

    let t_segment: Vec<F> = cfg_into_iter!(0..t_len)
        .map(|x| {
            let t_cols_per_claim = level_params.n_a * depth_open * num_blocks;
            let claim_idx = x / t_cols_per_claim;
            let claim_offset = x % t_cols_per_claim;
            let block_idx = claim_offset / (level_params.n_a * depth_open);
            let rem = claim_offset % (level_params.n_a * depth_open);
            let a_idx = rem / depth_open;
            let digit_idx = rem % depth_open;
            let global_block_idx = claim_idx * num_blocks + block_idx;
            let (group_idx, claim_idx_within_group) = claim_to_group[claim_idx];
            let local_col = claim_idx_within_group * t_cols_per_claim + claim_offset;
            let commitment_weights = &eq_tau1[(level_params.n_d + group_idx * level_params.n_b)
                ..(level_params.n_d + (group_idx + 1) * level_params.n_b)];
            let mut acc = a_weights[a_idx] * c_alphas[global_block_idx] * g1_open[digit_idx];
            for (row_idx, eq_i) in commitment_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc +=
                        *eq_i * eval_ring_at_pows(&shared_view.row(row_idx)[local_col], alpha_pows);
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
                    acc += *eq_i * eval_ring_at_pows(&shared_view.row(a_idx)[k], alpha_pows);
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
        if opening_point.a.len() != layout.block_len || opening_point.b.len() != layout.num_blocks {
            return Err(HachiError::InvalidInput(
                "multipoint ring switch opening-point layout mismatch".to_string(),
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

    let shared_view = setup.shared_matrix.view::<D>();

    let commitment_row_count = level_params.n_b * num_commitment_groups;
    let public_row_start = level_params.n_d + commitment_row_count;
    let row3_weights = &eq_tau1[public_row_start..(public_row_start + num_claims)];
    let row4_weight = eq_tau1[public_row_start + num_claims];
    let a_weights = &eq_tau1[(public_row_start + num_claims + 1)..rows];
    let claim_to_group: Vec<(usize, usize)> = claim_group_sizes
        .iter()
        .enumerate()
        .flat_map(|(group_idx, &group_size)| {
            (0..group_size).map(move |within_group| (group_idx, within_group))
        })
        .collect();

    let w_segment: Vec<F> = cfg_into_iter!(0..w_len)
        .map(|x| {
            let blocks_per_claim = num_blocks * depth_open;
            let claim_idx = x / blocks_per_claim;
            let claim_offset = x % blocks_per_claim;
            let block_idx = claim_offset / depth_open;
            let digit_idx = claim_offset % depth_open;
            let global_block_idx = claim_idx * num_blocks + block_idx;
            let opening_point = &opening_points[claim_to_point[claim_idx]];
            let mut acc = (row3_weights[claim_idx] * opening_point.b[block_idx]
                + row4_weight * c_alphas[global_block_idx])
                * g1_open[digit_idx];
            for (row_idx, eq_i) in eq_tau1.iter().enumerate().take(level_params.n_d) {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&shared_view.row(row_idx)[x], alpha_pows);
                }
            }
            acc
        })
        .collect();
    out.extend(w_segment);

    let t_segment: Vec<F> = cfg_into_iter!(0..t_len)
        .map(|x| {
            let t_cols_per_claim = level_params.n_a * depth_open * num_blocks;
            let claim_idx = x / t_cols_per_claim;
            let claim_offset = x % t_cols_per_claim;
            let block_idx = claim_offset / (level_params.n_a * depth_open);
            let rem = claim_offset % (level_params.n_a * depth_open);
            let a_idx = rem / depth_open;
            let digit_idx = rem % depth_open;
            let global_block_idx = claim_idx * num_blocks + block_idx;
            let (group_idx, claim_idx_within_group) = claim_to_group[claim_idx];
            let local_col = claim_idx_within_group * t_cols_per_claim + claim_offset;
            let commitment_weights = &eq_tau1[(level_params.n_d + group_idx * level_params.n_b)
                ..(level_params.n_d + (group_idx + 1) * level_params.n_b)];
            let mut acc = a_weights[a_idx] * c_alphas[global_block_idx] * g1_open[digit_idx];
            for (row_idx, eq_i) in commitment_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc +=
                        *eq_i * eval_ring_at_pows(&shared_view.row(row_idx)[local_col], alpha_pows);
                }
            }
            acc
        })
        .collect();
    out.extend(t_segment);

    let z_base: Vec<F> = cfg_into_iter!(0..z_base_len)
        .map(|k| {
            let point_idx = k / inner_width;
            let local_k = k % inner_width;
            let block_idx = local_k / depth_commit;
            let digit_idx = local_k % depth_commit;
            let opening_point = &opening_points[point_idx];
            let mut acc = row4_weight * opening_point.a[block_idx] * g1_commit[digit_idx];
            for (a_idx, eq_i) in a_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&shared_view.row(a_idx)[local_k], alpha_pows);
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

pub(crate) fn build_w_coeffs<F: CanonicalField, const D: usize>(
    w_hat: &[Vec<[i8; D]>],
    t_hat: &[Vec<[i8; D]>],
    z_pre_centered: &[[i32; D]],
    r: &[CyclotomicRing<F, D>],
    layout: HachiCommitmentLayout,
) -> RecursiveWitnessFlat {
    let log_basis = layout.log_basis;
    let num_digits_fold = layout.num_digits_fold;
    let levels = r_decomp_levels::<F>(log_basis);

    let w_hat_planes: usize = w_hat.iter().map(|v| v.len()).sum();
    let t_hat_planes: usize = t_hat.iter().map(|v| v.len()).sum();
    let z_count = w_hat_planes + t_hat_planes + z_pre_centered.len() * num_digits_fold;
    let r_hat_count = r.len() * levels;
    tracing::debug!(
        w_hat_planes,
        t_hat_planes,
        z_pre_elems = z_pre_centered.len(),
        z_pre_planes = z_pre_centered.len() * num_digits_fold,
        r_elems = r.len(),
        r_planes = r_hat_count,
        total_ring = z_count + r_hat_count,
        total_field = (z_count + r_hat_count) * D,
        "build_w_coeffs"
    );
    let total_planes = z_count + r_hat_count;
    let total_elems = total_planes * D;

    let mut out = Vec::with_capacity(total_elems);
    for block in w_hat {
        for digits in block {
            out.extend_from_slice(digits);
        }
    }
    for block in t_hat {
        for digits in block {
            out.extend_from_slice(digits);
        }
    }
    let mut z_planes = vec![[0i8; D]; num_digits_fold];
    for z_j in z_pre_centered {
        z_planes.fill([0i8; D]);
        balanced_decompose_centered_i32_i8_into(z_j, &mut z_planes, log_basis);
        for plane in &z_planes {
            out.extend_from_slice(plane);
        }
    }
    let mut r_planes = vec![[0i8; D]; levels];
    for ri in r {
        r_planes.fill([0i8; D]);
        ri.balanced_decompose_pow2_i8_into(&mut r_planes, log_basis);
        for plane in &r_planes {
            out.extend_from_slice(plane);
        }
    }
    RecursiveWitnessFlat::from_i8_digits(out)
}

#[cfg(test)]
mod tests {
    use super::{
        build_alpha_evals_y, build_w_evals_compact, commit_w, compute_m_evals_x,
        compute_r_via_poly_division, m_row_count, ring_switch_build_w, WCommitmentConfig,
    };
    use crate::algebra::{CyclotomicRing, Fp128, Prime128Offset5823};
    use crate::protocol::commitment::AppendToTranscript;
    use crate::protocol::commitment::{
        hachi_recursive_level_layout_from_params, Fp128FullCommitmentConfig, HachiCommitmentCore,
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
        type F = Prime128Offset5823;
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
        alpha_evals_y
            .iter()
            .enumerate()
            .fold(F::zero(), |acc_y, (y, &alpha)| {
                let row = &w_compact[y * live_x_cols..(y + 1) * live_x_cols];
                acc_y
                    + row.iter().enumerate().fold(F::zero(), |acc_x, (x, &w)| {
                        acc_x + F::from_i64(w as i64) * alpha * m_evals_x[x]
                    })
            })
    }

    #[test]
    fn full_root_rows_match_direct_relation_claim() {
        type F = Fp128<0xffffffffffffffffffffffffffffe941>;
        type Cfg = Fp128FullCommitmentConfig;
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
        let (w_compact, _num_u, num_l) =
            build_w_evals_compact(w.as_i8_digits(), D).expect("compact witness");
        let live_x_cols = w_compact.len() >> num_l;

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
        type F = Prime128Offset5823;
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
    fn commit_w_uses_active_level_row_count() {
        type Cfg = SmallTestCommitmentConfig;
        type WCfg = WCommitmentConfig<16, Cfg>;
        const D: usize = 16;

        let (setup, _) = <HachiCommitmentCore as RingCommitmentScheme<TestF, D, Cfg>>::setup(12, 1)
            .expect("setup");
        assert!(
            setup.ntt_shared.num_rows() > 3,
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
        let (_commitment, hint) =
            commit_w::<TestF, D, WCfg>(&w, &setup.ntt_shared, &level_params).expect("commit w");
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
}
