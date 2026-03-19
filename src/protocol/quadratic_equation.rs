//! Quadratic equation builder for the Hachi PCS (§4.2).
//!
//! This module encapsulates the stage-1 prover logic and the generation of
//! the quadratic equation components M, y, z, and v.

use crate::algebra::{CyclotomicRing, SparseChallenge};
#[cfg(any(test, debug_assertions))]
use crate::cfg_into_iter;
use crate::error::HachiError;
#[cfg(all(feature = "parallel", any(test, debug_assertions)))]
use crate::parallel::*;
use crate::protocol::challenges::sparse::sample_sparse_challenges;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::linear::{
    flatten_i8_blocks, mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
    unreduced_quotient_rows_ntt_cached_centered_i32,
};
use crate::protocol::commitment::{
    CommitmentConfig, HachiCommitmentLayout, HachiExpandedSetup, HachiLevelParams, RingCommitment,
};
use crate::protocol::hachi_poly_ops::{DecomposeFoldWitness, HachiPolyOps};
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::proof::HachiCommitmentHint;
#[cfg(any(test, debug_assertions))]
use crate::protocol::ring_switch::eval_ring_at;
use crate::protocol::transcript::labels::{ABSORB_PROVER_V, CHALLENGE_STAGE1_FOLD};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};
use std::iter::repeat_n;
use std::marker::PhantomData;
use std::time::Instant;

/// **Step 4.** Compute `v = D · ŵ` (first prover message).
fn compute_v<F: FieldCore + CanonicalField, const D: usize>(
    ntt_d: &NttSlotCache<D>,
    w_hat_flat: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>> {
    mat_vec_mul_ntt_single_i8(ntt_d, w_hat_flat)
}

fn flatten_w_hat<const D: usize>(w_hat: &[Vec<[i8; D]>]) -> Vec<[i8; D]> {
    w_hat.iter().flat_map(|v| v.iter().copied()).collect()
}

fn compute_z_pre<F, const D: usize, P>(
    poly: &P,
    challenges: &[SparseChallenge],
    level_params: HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> Result<DecomposeFoldWitness<F, D>, HachiError>
where
    F: FieldCore + CanonicalField,
    P: HachiPolyOps<F, D>,
{
    let z = poly.decompose_fold(
        challenges,
        layout.block_len,
        layout.num_digits_commit,
        layout.log_basis,
    );

    let norm = u128::from(z.centered_inf_norm);
    let beta = crate::protocol::commitment::beta_linf_fold_bound(
        layout.r_vars,
        level_params.challenge_weight,
        layout.log_basis,
    )?;
    if norm > beta {
        return Err(HachiError::InvalidInput(format!(
            "prover abort: ||z||_inf = {norm} > beta = {beta}"
        )));
    }

    Ok(z)
}

/// Stage-1 quadratic equation state for the Hachi protocol.
///
/// Encapsulates the relation $M(x) \cdot z = y(x) + (X^D + 1) \cdot r(x)$
/// along with intermediate prover witness data (`w_hat`, `z_pre`, `hint`).
///
/// M and z are never materialized on the hot path — split-eq factoring computes
/// their products on-the-fly via `compute_r_split_eq`, while debug/test code
/// can reconstruct reference `M_a` rows when needed.
pub struct QuadraticEquation<F: FieldCore, const D: usize, Cfg: CommitmentConfig> {
    /// Stage-1 proof vector `v = D · ŵ`.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// Stage-1 folding challenges (sparse representation).
    pub challenges: Vec<SparseChallenge>,
    /// Vector `y`.
    y: Vec<CyclotomicRing<F, D>>,
    /// Opening point (a, b) Lagrange weights.
    opening_point: RingOpeningPoint<F>,
    /// Pre-decomposition folded witness `z_pre = Σ c_i · s_i` (prover only).
    /// Replaces both `z_hat` and `z`: `z_hat = J^{-1}(z_pre)`.
    z_pre: Option<DecomposeFoldWitness<F, D>>,
    /// Decomposed `ŵ_i = G_1^{-1}(w_i)` as i8 digit planes (prover only).
    w_hat: Option<Vec<Vec<[i8; D]>>>,
    /// Flattened `w_hat` as i8 digit planes (prover only, computed once and reused).
    w_hat_flat: Option<Vec<[i8; D]>>,
    /// Pre-decomposition folded ring elements (prover only, avoids recompose roundtrip).
    w_folded: Option<Vec<CyclotomicRing<F, D>>>,
    /// Commitment hint (prover only).
    hint: Option<HachiCommitmentHint<F, D>>,

    _marker: PhantomData<Cfg>,
}

impl<F, const D: usize, Cfg> QuadraticEquation<F, D, Cfg>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    /// Prover constructor: runs §4.2 stage 1 and builds all equation components.
    ///
    /// `poly` provides the ring-level polynomial data for fold/decompose ops.
    /// `hint` carries `t_hat` from the commitment phase.
    ///
    /// # Errors
    ///
    /// Returns an error if the norm check, challenge sampling, or matrix
    /// generation fails.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "QuadraticEquation::new_prover")]
    #[inline(never)]
    pub fn new_prover<T: Transcript<F>, P: HachiPolyOps<F, D>>(
        ntt_d: &NttSlotCache<D>,
        ring_opening_point: RingOpeningPoint<F>,
        poly: &P,
        pre_folded: Vec<CyclotomicRing<F, D>>,
        level_params: HachiLevelParams,
        mut hint: HachiCommitmentHint<F, D>,
        transcript: &mut T,
        commitment: &RingCommitment<F, D>,
        y_ring: &CyclotomicRing<F, D>,
        layout: HachiCommitmentLayout,
    ) -> Result<Self, HachiError> {
        {
            let x: u8 = 0;
            tracing::trace!(
                stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
                "QuadraticEquation::new_prover"
            );
        }
        let (w_hat, w_hat_flat) = {
            let _span = tracing::info_span!("decompose_w_hat").entered();
            let depth_open = layout.num_digits_open;
            let log_basis = layout.log_basis;
            let w_hat: Vec<Vec<[i8; D]>> = pre_folded
                .iter()
                .map(|w_i| w_i.balanced_decompose_pow2_i8(depth_open, log_basis))
                .collect();
            let w_hat_flat = flatten_w_hat(&w_hat);
            (w_hat, w_hat_flat)
        };
        hint.ensure_t_recomposed(layout.num_digits_open, layout.log_basis)?;

        let v = {
            let _span =
                tracing::info_span!("compute_v", w_hat_flat_len = w_hat_flat.len()).entered();
            let mut v = compute_v(ntt_d, &w_hat_flat);
            v.truncate(level_params.n_d);
            v
        };

        transcript.append_serde(ABSORB_PROVER_V, &v);

        let challenge_cfg = Cfg::stage1_challenge_config(level_params);
        let challenges = sample_sparse_challenges::<F, T, D>(
            transcript,
            CHALLENGE_STAGE1_FOLD,
            layout.num_blocks,
            &challenge_cfg,
        )?;

        let z_pre = {
            let _span = tracing::info_span!("compute_z_pre").entered();
            compute_z_pre::<F, D, P>(poly, &challenges, level_params, layout)?
        };

        let y = generate_y::<F, D>(
            &v,
            &commitment.u,
            y_ring,
            level_params.n_d,
            level_params.n_b,
            level_params.n_a,
        )?;

        Ok(Self {
            v,
            challenges,
            y,
            opening_point: ring_opening_point,
            z_pre: Some(z_pre),
            w_hat: Some(w_hat),
            w_hat_flat: Some(w_hat_flat),
            w_folded: Some(pre_folded),
            hint: Some(hint),
            _marker: PhantomData,
        })
    }

    /// Verifier constructor: Derives challenges and computes M and y.
    ///
    /// # Errors
    ///
    /// Returns an error if challenge derivation fails.
    #[tracing::instrument(skip_all, name = "QuadraticEquation::new_verifier")]
    #[inline(never)]
    pub fn new_verifier<T: Transcript<F>>(
        ring_opening_point: RingOpeningPoint<F>,
        v: Vec<CyclotomicRing<F, D>>,
        level_params: HachiLevelParams,
        transcript: &mut T,
        commitment: &RingCommitment<F, D>,
        y_ring: &CyclotomicRing<F, D>,
        layout: HachiCommitmentLayout,
    ) -> Result<Self, HachiError> {
        let challenges = derive_stage1_challenges::<F, T, D, Cfg>(
            transcript,
            &v,
            layout.num_blocks,
            level_params,
        )?;
        let y = generate_y::<F, D>(
            &v,
            &commitment.u,
            y_ring,
            level_params.n_d,
            level_params.n_b,
            level_params.n_a,
        )?;

        Ok(Self {
            v,
            challenges,
            y,
            opening_point: ring_opening_point,
            z_pre: None,
            w_hat: None,
            w_hat_flat: None,
            w_folded: None,
            hint: None,
            _marker: PhantomData,
        })
    }

    /// Get the vector y.
    pub fn y(&self) -> &[CyclotomicRing<F, D>] {
        &self.y
    }

    /// Get the vector v.
    pub fn v(&self) -> &[CyclotomicRing<F, D>] {
        &self.v
    }

    /// Get the opening point (a, b) Lagrange weights.
    pub fn opening_point(&self) -> &RingOpeningPoint<F> {
        &self.opening_point
    }

    /// Get the pre-decomposition folded witness `z_pre` (prover only).
    pub fn z_pre(&self) -> Option<&[CyclotomicRing<F, D>]> {
        self.z_pre.as_ref().map(|witness| witness.z_pre.as_slice())
    }

    /// Get centered coefficients for each `z_pre` row (prover only).
    pub fn z_pre_centered(&self) -> Option<&[[i32; D]]> {
        self.z_pre
            .as_ref()
            .map(|witness| witness.centered_coeffs.as_slice())
    }

    /// Get `||z_pre||_inf` from the centered witness representation.
    pub fn z_pre_centered_inf_norm(&self) -> Option<u32> {
        self.z_pre.as_ref().map(|witness| witness.centered_inf_norm)
    }

    /// Take ownership of the `z_pre` witness, leaving `None` in its place.
    pub fn take_z_pre(&mut self) -> Option<DecomposeFoldWitness<F, D>> {
        self.z_pre.take()
    }

    /// Get the decomposed witness `ŵ` as i8 digit planes (prover only).
    pub fn w_hat(&self) -> Option<&[Vec<[i8; D]>]> {
        self.w_hat.as_deref()
    }

    /// Get the pre-flattened `w_hat` as i8 digit planes (prover only).
    pub fn w_hat_flat(&self) -> Option<&[[i8; D]]> {
        self.w_hat_flat.as_deref()
    }

    /// Take ownership of `w_hat`, leaving `None` in its place.
    pub fn take_w_hat(&mut self) -> Option<Vec<Vec<[i8; D]>>> {
        self.w_hat.take()
    }

    /// Get the pre-decomposition folded ring elements (prover only).
    pub fn w_folded(&self) -> Option<&[CyclotomicRing<F, D>]> {
        self.w_folded.as_deref()
    }

    /// Get the commitment hint (prover only).
    pub fn hint(&self) -> Option<&HachiCommitmentHint<F, D>> {
        self.hint.as_ref()
    }

    /// Take ownership of the hint, leaving `None` in its place.
    pub fn take_hint(&mut self) -> Option<HachiCommitmentHint<F, D>> {
        self.hint.take()
    }
}

pub(crate) fn derive_stage1_challenges<F, T, const D: usize, Cfg: CommitmentConfig>(
    transcript: &mut T,
    v: &Vec<CyclotomicRing<F, D>>,
    num_blocks: usize,
    level_params: HachiLevelParams,
) -> Result<Vec<SparseChallenge>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let challenge_cfg = Cfg::stage1_challenge_config(level_params);
    transcript.append_serde(ABSORB_PROVER_V, v);
    sample_sparse_challenges::<F, T, D>(
        transcript,
        CHALLENGE_STAGE1_FOLD,
        num_blocks,
        &challenge_cfg,
    )
}

#[cfg(any(test, debug_assertions))]
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

/// Add only the high-half quotient contribution of `challenge * ring`.
fn add_sparse_ring_product_high_half<F: FieldCore + CanonicalField, const D: usize>(
    quotient: &mut [F],
    challenge: &SparseChallenge,
    ring: &CyclotomicRing<F, D>,
) {
    let rc = ring.coefficients();
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let c = F::from_i64(coeff as i64);
        let p = pos as usize;
        let start = D.saturating_sub(p);
        for (s, &r_s) in rc.iter().enumerate().skip(start) {
            quotient[p + s - D] += c * r_s;
        }
    }
}

fn quotient_from_cyclic_and_reduced<F: FieldCore, const D: usize>(
    cyclic: &CyclotomicRing<F, D>,
    reduced: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let cyc_c = cyclic.coefficients();
    let red_c = reduced.coefficients();
    let quotient = std::array::from_fn(|k| (cyc_c[k] - red_c[k]) * F::TWO_INV);
    CyclotomicRing::from_coefficients(quotient)
}

/// Split-eq replacement for `generate_m` + `compute_r_via_poly_division`.
///
/// Computes `r` such that `M·z = y + (X^D+1)·r` without materializing M or z.
/// Uses split-eq factoring: `kron(left, gadget) · decomposed = left · pre_decomp`.
#[allow(clippy::too_many_arguments, clippy::needless_borrow)]
#[tracing::instrument(skip_all, name = "compute_r_split_eq")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_r_split_eq<F, const D: usize>(
    level_params: HachiLevelParams,
    _setup: &HachiExpandedSetup<F>,
    challenges: &[SparseChallenge],
    w_hat_flat: &[[i8; D]],
    t_hat: &[Vec<[i8; D]>],
    t: &[Vec<CyclotomicRing<F, D>>],
    w_folded: &[CyclotomicRing<F, D>],
    z_pre_centered: &[[i32; D]],
    z_pre_centered_inf_norm: u32,
    y: &[CyclotomicRing<F, D>],
    ntt_a: &NttSlotCache<D>,
    ntt_b: &NttSlotCache<D>,
    ntt_d: &NttSlotCache<D>,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
{
    let num_rows = level_params.m_row_count();

    let t_hat_flat = flatten_i8_blocks(t_hat);

    // D/B rows already know their reduced outputs `y`, so only the cyclic side
    // must be computed here; quotient = (cyc - reduced) / 2.
    let t_d = Instant::now();
    let d_cyclic = {
        let _span = tracing::info_span!("D_rows_ntt").entered();
        mat_vec_mul_ntt_single_i8_cyclic(ntt_d, w_hat_flat)
    };
    let d_time = t_d.elapsed().as_secs_f64();

    let t_b = Instant::now();
    let b_cyclic = {
        let _span = tracing::info_span!("B_rows_ntt").entered();
        mat_vec_mul_ntt_single_i8_cyclic(ntt_b, &t_hat_flat)
    };
    let b_time = t_b.elapsed().as_secs_f64();

    let t_a = Instant::now();
    let a_quotients = {
        let _span = tracing::info_span!("A_rows_ntt").entered();
        unreduced_quotient_rows_ntt_cached_centered_i32(
            ntt_a,
            z_pre_centered,
            z_pre_centered_inf_norm,
        )
    };
    let a_time = t_a.elapsed().as_secs_f64();

    let mut result = Vec::with_capacity(num_rows);
    let mut other_time = 0.0f64;
    let mut quotient_buf = vec![F::zero(); D];

    for row_idx in 0..num_rows {
        if row_idx < level_params.n_d {
            result.push(quotient_from_cyclic_and_reduced(
                &d_cyclic[row_idx],
                &y[row_idx],
            ));
        } else if row_idx < level_params.n_d + level_params.n_b {
            result.push(quotient_from_cyclic_and_reduced(
                &b_cyclic[row_idx - level_params.n_d],
                &y[row_idx],
            ));
        } else if row_idx >= level_params.n_d + level_params.n_b + 2 {
            // A-rows: NTT-accelerated A*z_pre + sparse challenge terms
            let t_row = Instant::now();
            let _span = tracing::info_span!("A_row").entered();
            let a_idx = row_idx - (level_params.n_d + level_params.n_b + 2);

            quotient_buf.fill(F::zero());
            for (i, t_rows_i) in t.iter().enumerate() {
                if let Some(t_row_i) = t_rows_i.get(a_idx) {
                    add_sparse_ring_product_high_half(&mut quotient_buf, &challenges[i], t_row_i);
                }
            }

            let a_q = a_quotients[a_idx].coefficients();
            for k in 0..D {
                quotient_buf[k] -= a_q[k];
            }
            result.push(CyclotomicRing::from_slice(&quotient_buf));
            other_time += t_row.elapsed().as_secs_f64();
        } else {
            let t_row = Instant::now();

            if row_idx == level_params.n_d + level_params.n_b {
                let _span = tracing::info_span!("bTw_row").entered();
                // `b^T · G · ŵ - y_ring` is degree < D, so its quotient is zero.
                result.push(CyclotomicRing::<F, D>::zero());
            } else {
                let _span = tracing::info_span!("challenge_fold_row").entered();
                quotient_buf.fill(F::zero());
                for (i, w_f) in w_folded.iter().enumerate() {
                    add_sparse_ring_product_high_half(&mut quotient_buf, &challenges[i], w_f);
                }
                // `a^T · G · J · z_hat` contributes only low-degree terms, so it
                // cannot affect the high-half quotient we need here.
                result.push(CyclotomicRing::from_slice(&quotient_buf));
            }
            other_time += t_row.elapsed().as_secs_f64();
        }
    }

    tracing::debug!(
        d_ntt_s = d_time,
        b_ntt_s = b_time,
        a_ntt_s = a_time,
        other_s = other_time,
        "compute_r breakdown"
    );

    Ok(result)
}

/// Reference helper for tests/debug diagnostics: split-eq replacement for
/// `generate_m` + `eval_ring_matrix_at`.
///
/// Computes the field-element evaluations of each M entry at `alpha`,
/// organized as rows of field elements, without materializing ring-valued `M`.
#[cfg(any(test, debug_assertions))]
#[tracing::instrument(skip_all, name = "compute_m_a_reference")]
pub(crate) fn compute_m_a_reference<F, const D: usize>(
    setup: &HachiExpandedSetup<F>,
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    alpha: &F,
    level_params: HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> Result<Vec<Vec<F>>, HachiError>
where
    F: FieldCore + CanonicalField,
{
    let depth_commit = layout.num_digits_commit;
    let depth_open = layout.num_digits_open;
    let depth_fold = layout.num_digits_fold;
    let log_basis = layout.log_basis;
    let num_blocks = opening_point.b.len();
    let block_len = layout.block_len;
    let w_len = depth_open * num_blocks;
    let t_len = depth_open * level_params.n_a * num_blocks;
    let z_len = depth_fold * depth_commit * block_len;
    let total_cols = w_len + t_len + z_len;

    let g1_open = gadget_row_scalars::<F>(depth_open, log_basis);
    let g1_commit = gadget_row_scalars::<F>(depth_commit, log_basis);
    let j1 = gadget_row_scalars::<F>(depth_fold, log_basis);

    let c_alphas: Vec<F> = challenges
        .iter()
        .map(|c| eval_ring_at(&c.to_dense::<F, D>().expect("valid challenge"), alpha))
        .collect();

    let d_view = setup.D_mat.view::<D>();
    let b_view = setup.B.view::<D>();

    let d_rows: Vec<Vec<F>> = cfg_into_iter!(0..d_view.num_rows())
        .map(|i| {
            let d_row = d_view.row(i);
            let mut full = vec![F::zero(); total_cols];
            for (j, ring) in d_row.iter().take(w_len).enumerate() {
                full[j] = eval_ring_at(ring, alpha);
            }
            full
        })
        .collect();

    let b_rows: Vec<Vec<F>> = cfg_into_iter!(0..b_view.num_rows())
        .map(|i| {
            let b_row = b_view.row(i);
            let mut full = vec![F::zero(); total_cols];
            for (j, ring) in b_row.iter().take(t_len).enumerate() {
                full[w_len + j] = eval_ring_at(ring, alpha);
            }
            full
        })
        .collect();

    let mut rows = Vec::with_capacity(level_params.m_row_count());
    rows.extend(d_rows.into_iter().take(level_params.n_d));
    rows.extend(b_rows.into_iter().take(level_params.n_b));

    // Row 3: b^T · G · ŵ = y_ring (ŵ uses delta_open)
    {
        let mut full = vec![F::zero(); total_cols];
        for (i, &b_i) in opening_point.b.iter().enumerate() {
            for (d, &g) in g1_open.iter().enumerate() {
                full[i * depth_open + d] = b_i * g;
            }
        }
        rows.push(full);
    }

    // Row 4: (c^T ⊗ G) · ŵ = a^T · G · J · ẑ
    {
        let mut full = vec![F::zero(); total_cols];
        for (i, &c_alpha) in c_alphas.iter().enumerate() {
            for (d, &g) in g1_open.iter().enumerate() {
                full[i * depth_open + d] = c_alpha * g;
            }
        }
        let z_offset = w_len + t_len;
        for (i, &a_i) in opening_point.a.iter().enumerate() {
            for (d, &g) in g1_commit.iter().enumerate() {
                let ag = a_i * g;
                for (t, &j) in j1.iter().enumerate() {
                    let idx = (i * depth_commit + d) * depth_fold + t;
                    full[z_offset + idx] = -(ag * j);
                }
            }
        }
        rows.push(full);
    }

    // Row 5: (c^T ⊗ G_open) · t̂ = A · J · ẑ
    // t̂ uses delta_open (t = A*s has full-field coefficients); ẑ uses delta_commit
    for a_idx in 0..level_params.n_a {
        let mut full = vec![F::zero(); total_cols];
        for (i, &c_alpha) in c_alphas.iter().enumerate() {
            for (d, &g) in g1_open.iter().enumerate() {
                let t_idx = i * (level_params.n_a * depth_open) + a_idx * depth_open + d;
                full[w_len + t_idx] = c_alpha * g;
            }
        }
        let z_offset = w_len + t_len;
        let a_view = setup.A.view::<D>();
        let a_row = a_view.row(a_idx);
        let inner_width = block_len * depth_commit;
        for (k, ring) in a_row.iter().take(inner_width).enumerate() {
            let ring_alpha = eval_ring_at(ring, alpha);
            for (t, &j) in j1.iter().enumerate() {
                full[z_offset + k * depth_fold + t] = -(ring_alpha * j);
            }
        }
        rows.push(full);
    }

    Ok(rows)
}

pub(crate) fn generate_y<F, const D: usize>(
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
    u_eval: &CyclotomicRing<F, D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore,
{
    if v.len() != n_d {
        return Err(HachiError::InvalidSize {
            expected: n_d,
            actual: v.len(),
        });
    }
    if u.len() != n_b {
        return Err(HachiError::InvalidSize {
            expected: n_b,
            actual: u.len(),
        });
    }
    let mut out = Vec::with_capacity(n_d + n_b + 1 + 1 + n_a);
    out.extend_from_slice(v);
    out.extend_from_slice(u);
    out.push(*u_eval);
    out.push(CyclotomicRing::<F, D>::zero());
    out.extend(repeat_n(CyclotomicRing::<F, D>::zero(), n_a));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::array::from_fn;

    use crate::algebra::{CyclotomicRing, SparseChallengeConfig};
    use crate::protocol::challenges::sparse::sample_sparse_challenges;
    use crate::protocol::commitment::HachiProverSetup;
    use crate::protocol::commitment::{
        HachiCommitmentCore, HachiScheduleInputs, RingCommitmentScheme,
    };
    use crate::protocol::hachi_poly_ops::DensePoly;
    use crate::protocol::proof::HachiCommitmentHint;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::test_utils::*;
    use crate::FromSmallInt;
    use crate::Transcript;

    const TRANSCRIPT_SEED: &[u8] = b"test/prover-relation";

    fn replay_challenges(v: &Vec<CyclotomicRing<F, D>>) -> Vec<CyclotomicRing<F, D>> {
        let mut transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_SEED);
        transcript.append_serde(ABSORB_PROVER_V, v);

        let challenge_cfg = SparseChallengeConfig {
            weight: TinyConfig::CHALLENGE_WEIGHT,
            nonzero_coeffs: vec![-1, 1],
        };
        let sparse = sample_sparse_challenges::<F, Blake2bTranscript<F>, D>(
            &mut transcript,
            CHALLENGE_STAGE1_FOLD,
            NUM_BLOCKS,
            &challenge_cfg,
        )
        .unwrap();
        sparse
            .iter()
            .map(|c| c.to_dense::<F, D>().unwrap())
            .collect()
    }

    struct Fixture {
        setup: HachiProverSetup<F, D>,
        commitment_u: Vec<CyclotomicRing<F, D>>,
        point: RingOpeningPoint<F>,
        blocks: Vec<Vec<CyclotomicRing<F, D>>>,
        quad_eq: QuadraticEquation<F, D, TinyConfig>,
        /// Challenges re-derived via transcript replay (cross-check).
        challenges: Vec<CyclotomicRing<F, D>>,
    }

    fn build_fixture() -> Fixture {
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::setup(16).unwrap();

        let blocks = sample_blocks();
        let w =
            <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_ring_blocks(
                &blocks, &setup,
            )
            .unwrap();

        let point = RingOpeningPoint {
            a: sample_a(),
            b: sample_b(),
        };

        let ring_coeffs: Vec<CyclotomicRing<F, D>> =
            blocks.iter().flat_map(|b| b.iter().copied()).collect();
        let poly = DensePoly::from_ring_coeffs(ring_coeffs);
        let hint = HachiCommitmentHint::new(w.t_hat);
        let mut transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_SEED);
        let y_ring = CyclotomicRing::<F, D>::zero();
        let layout = setup.layout();
        let w_folded = poly.fold_blocks(&point.a, layout.block_len);
        let level_params = TinyConfig::level_params(HachiScheduleInputs {
            max_num_vars: setup.expanded.seed.max_num_vars,
            level: 0,
            current_w_len: layout.num_blocks * layout.block_len * D,
        });
        let quad_eq = QuadraticEquation::<F, D, TinyConfig>::new_prover(
            &setup.ntt_D,
            point.clone(),
            &poly,
            w_folded,
            level_params,
            hint,
            &mut transcript,
            &w.commitment,
            &y_ring,
            layout,
        )
        .unwrap();

        let challenges = replay_challenges(&quad_eq.v);

        Fixture {
            setup,
            commitment_u: w.commitment.u.clone(),
            point,
            blocks,
            quad_eq,
            challenges,
        }
    }

    fn i8_to_ring(digits: &[[i8; D]]) -> Vec<CyclotomicRing<F, D>> {
        digits
            .iter()
            .map(|d| {
                let coeffs: [F; D] = from_fn(|i| F::from_i64(d[i] as i64));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect()
    }

    /// Row 1: D · ŵ = v
    #[test]
    fn row1_d_times_w_hat_equals_v() {
        let f = build_fixture();

        let w_hat = f.quad_eq.w_hat().unwrap();
        let w_hat_flat: Vec<CyclotomicRing<F, D>> = i8_to_ring(
            &w_hat
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect::<Vec<_>>(),
        );
        let lhs = mat_vec_mul(&f.setup.expanded.D_mat, &w_hat_flat);

        assert_eq!(lhs, f.quad_eq.v(), "Row 1 failed: D · ŵ ≠ v");
    }

    /// Row 2: B · inner opening digits = u (commitment vector)
    #[test]
    fn row2_b_times_inner_opening_digits_equals_u_commitment() {
        let f = build_fixture();

        let hint = f.quad_eq.hint().unwrap();
        let inner_opening_digits_flat_ring: Vec<CyclotomicRing<F, D>> = hint
            .inner_opening_digits
            .iter()
            .flat_map(|v| v.iter())
            .map(|plane| {
                let coeffs: [F; D] = from_fn(|k| F::from_i64(plane[k] as i64));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();
        let lhs = mat_vec_mul(&f.setup.expanded.B, &inner_opening_digits_flat_ring);

        assert_eq!(
            lhs, f.commitment_u,
            "Row 2 failed: B · inner opening digits ≠ u"
        );
    }

    /// Row 3: b^T · G_{2^r} · ŵ = u_eval
    #[test]
    fn row3_bt_gadget_w_hat_equals_u_eval() {
        let f = build_fixture();

        let w_hat = f.quad_eq.w_hat().unwrap();
        let w_recomposed: Vec<CyclotomicRing<F, D>> = w_hat
            .iter()
            .map(|w_hat_i| CyclotomicRing::gadget_recompose_pow2_i8(w_hat_i, log_basis()))
            .collect();

        let u_eval = w_recomposed
            .iter()
            .zip(f.point.b.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (w_i, b_i)| {
                acc + w_i.scale(b_i)
            });

        let u_eval_direct = f.blocks.iter().zip(f.point.b.iter()).fold(
            CyclotomicRing::<F, D>::zero(),
            |acc, (block_i, b_i)| {
                let inner: CyclotomicRing<F, D> = block_i
                    .iter()
                    .zip(f.point.a.iter())
                    .fold(CyclotomicRing::<F, D>::zero(), |acc2, (f_ij, a_j)| {
                        acc2 + f_ij.scale(a_j)
                    });
                acc + inner.scale(b_i)
            },
        );

        assert_eq!(
            u_eval, u_eval_direct,
            "Row 3 failed: b^T G ŵ ≠ Σ b_i (a^T f_i)"
        );
    }

    /// Derive z_hat from z_pre for test assertions.
    fn derive_z_hat(z_pre: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
        z_pre
            .iter()
            .flat_map(|z_j| z_j.balanced_decompose_pow2(num_digits_fold(), log_basis()))
            .collect()
    }

    /// Row 4: (c^T ⊗ G_1) · ŵ = a^T · G_{2^m} · J · ẑ
    #[test]
    fn row4_challenge_fold_w_equals_a_gadget_j_z_hat() {
        let f = build_fixture();

        let w_hat = f.quad_eq.w_hat().unwrap();
        let w: Vec<CyclotomicRing<F, D>> = w_hat
            .iter()
            .map(|w_hat_i| CyclotomicRing::gadget_recompose_pow2_i8(w_hat_i, log_basis()))
            .collect();

        let lhs = f
            .challenges
            .iter()
            .zip(w.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (c_i, w_i)| {
                acc + (*c_i * *w_i)
            });

        let z_hat = derive_z_hat(f.quad_eq.z_pre().unwrap());
        let z_recovered = recompose_z_hat(&z_hat);
        let rhs = a_transpose_gadget_times_vec(&f.point.a, &z_recovered);

        assert_eq!(lhs, rhs, "Row 4 failed: (c^T ⊗ G_1)ŵ ≠ a^T G J ẑ");
    }

    /// Row 5: (c^T ⊗ G_{n_A}) · inner opening digits = A · J · ẑ
    #[test]
    fn row5_challenge_fold_inner_opening_digits_equals_a_j_z_hat() {
        let f = build_fixture();

        let hint = f.quad_eq.hint().unwrap();
        let mut lhs = vec![CyclotomicRing::<F, D>::zero(); N_A];
        for (c_i, inner_opening_digits_i) in
            f.challenges.iter().zip(hint.inner_opening_digits.iter())
        {
            let t_i = gadget_recompose_vec_i8(inner_opening_digits_i);
            assert_eq!(t_i.len(), N_A);
            for (lhs_j, t_ij) in lhs.iter_mut().zip(t_i.iter()) {
                *lhs_j += *c_i * *t_ij;
            }
        }

        let z_hat = derive_z_hat(f.quad_eq.z_pre().unwrap());
        let z_recovered = recompose_z_hat(&z_hat);
        let rhs = mat_vec_mul(&f.setup.expanded.A, &z_recovered);

        assert_eq!(
            lhs, rhs,
            "Row 5 failed: (c^T ⊗ G_nA)inner opening digits ≠ A · J · ẑ"
        );
    }

    #[test]
    fn prove_output_shapes_are_correct() {
        let f = build_fixture();

        assert_eq!(f.quad_eq.v().len(), TinyConfig::N_D);

        let w_hat = f.quad_eq.w_hat().unwrap();
        assert_eq!(w_hat.len(), NUM_BLOCKS);
        assert!(w_hat.iter().all(|v| v.len() == num_digits_open()));

        let hint = f.quad_eq.hint().unwrap();
        assert_eq!(hint.inner_opening_digits.len(), NUM_BLOCKS);
        assert!(hint
            .inner_opening_digits
            .iter()
            .all(|v| v.len() == N_A * num_digits_open()));

        assert_eq!(
            f.quad_eq.z_pre().unwrap().len(),
            BLOCK_LEN * num_digits_commit()
        );
    }
}
