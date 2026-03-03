//! Quadratic equation builder for the Hachi PCS (§4.2).
//!
//! This module encapsulates the stage-1 prover logic and the generation of
//! the quadratic equation components M, y, z, and v.

use crate::algebra::{CyclotomicRing, SparseChallenge, SparseChallengeConfig};
use crate::cfg_iter;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::challenges::sparse::sample_sparse_challenges;
use crate::protocol::commitment::utils::crt_ntt::NttMatrixCache;
use crate::protocol::commitment::utils::linear::{
    decompose_block, mat_vec_mul_ntt_cached, MatrixSlot,
};
use crate::protocol::commitment::utils::norm::{detect_field_modulus, vec_inf_norm};
use crate::protocol::commitment::{
    CommitmentConfig, HachiCommitmentLayout, HachiExpandedSetup, HachiProverSetup,
    HachiVerifierSetup, RingCommitment,
};
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::proof::HachiCommitmentHint;
use crate::protocol::ring_switch::eval_ring_at;
use crate::protocol::transcript::labels::{ABSORB_PROVER_V, CHALLENGE_STAGE1_FOLD};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};
use std::iter::repeat_n;
use std::marker::PhantomData;

/// **Steps 1–3.** Compute `w_i = a^T G_{2^m} s_i` and decompose: `ŵ_i = G_1^{-1}(w_i)`.
///
/// Recomputes each block's `s_i` from `ring_coeffs` on the fly to avoid
/// storing all `s_i` simultaneously (which can be tens of GB at production
/// parameters).
fn compute_w_hat<F, const D: usize>(
    opening_point: &RingOpeningPoint<F>,
    ring_coeffs: &[CyclotomicRing<F, D>],
    layout: HachiCommitmentLayout,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField,
{
    let a = &opening_point.a;
    let block_len = layout.block_len;
    let delta = layout.delta;
    let log_basis = layout.log_basis;

    debug_assert_eq!(a.len(), block_len);

    let blocks: Vec<&[CyclotomicRing<F, D>]> = (0..layout.num_blocks)
        .map(|i| {
            let start = i * block_len;
            let end = (start + block_len).min(ring_coeffs.len());
            if start < ring_coeffs.len() {
                &ring_coeffs[start..end]
            } else {
                &[] as &[CyclotomicRing<F, D>]
            }
        })
        .collect();

    // w_i = Σ_j a[j] · block[j]  (dot product in ring space).
    // Then ŵ_i = G^{-1}(w_i).
    //
    // The previous implementation materialized s_i = G^{-1}(block) (block_len *
    // delta ring elements ≈ 512 MB for production params) and then recomposed
    // each entry G(s_i[j*delta..(j+1)*delta]) = block[j].  The decompose +
    // recompose is an identity, so we skip it entirely.
    cfg_iter!(blocks)
        .map(|block| {
            let mut w_i = CyclotomicRing::<F, D>::zero();
            for (b_j, a_j) in block.iter().zip(a.iter()) {
                w_i += b_j.scale(a_j);
            }
            w_i.balanced_decompose_pow2(delta, log_basis)
        })
        .collect()
}

/// **Step 4.** Compute `v = D · ŵ` (first prover message).
fn compute_v<F: FieldCore + CanonicalField, const D: usize>(
    cache: &NttMatrixCache<D>,
    w_hat_flat: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    mat_vec_mul_ntt_cached(cache, MatrixSlot::D, w_hat_flat)
}

fn flatten_w_hat<F: FieldCore, const D: usize>(
    w_hat: &[Vec<CyclotomicRing<F, D>>],
) -> Vec<CyclotomicRing<F, D>> {
    w_hat.iter().flat_map(|v| v.iter().copied()).collect()
}

/// **Steps 7–9.** Fold `z_pre = Σ c_i · s_i` and check `‖z_pre‖_∞ ≤ β`.
///
/// Returns the pre-decomposition `z` vector (before gadget decomposition into
/// `ẑ = J^{-1}(z_pre)`). Callers that need `z_hat` can apply
/// `balanced_decompose_pow2(TAU, LOG_BASIS)` themselves.
fn compute_z_pre<F, const D: usize, Cfg>(
    ring_coeffs: &[CyclotomicRing<F, D>],
    challenges: &[SparseChallenge],
    layout: HachiCommitmentLayout,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    let block_len = layout.block_len;
    let delta = layout.delta;
    let log_basis = layout.log_basis;
    let inner_width = block_len * delta;

    debug_assert_eq!(challenges.len(), layout.num_blocks);

    let mut z = vec![CyclotomicRing::<F, D>::zero(); inner_width];

    for (i, c_i) in challenges.iter().enumerate() {
        let start = i * block_len;
        let end = (start + block_len).min(ring_coeffs.len());
        let block = if start < ring_coeffs.len() {
            &ring_coeffs[start..end]
        } else {
            &[] as &[CyclotomicRing<F, D>]
        };
        let s_i = decompose_block(block, delta, log_basis);
        for (j, z_j) in z.iter_mut().enumerate() {
            *z_j += s_i[j].mul_by_sparse(c_i);
        }
    }

    let modulus = detect_field_modulus::<F>();
    let norm = vec_inf_norm(&z, modulus);
    let beta = Cfg::beta_bound(layout)?;
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
/// M and z are never materialized — split-eq factoring computes their
/// products on-the-fly via `compute_r_split_eq` and `compute_m_a_streaming`.
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
    z_pre: Option<Vec<CyclotomicRing<F, D>>>,
    /// Decomposed `ŵ_i = G_1^{-1}(w_i)` (prover only).
    w_hat: Option<Vec<Vec<CyclotomicRing<F, D>>>>,
    /// Flattened `w_hat` (prover only, computed once and reused).
    w_hat_flat: Option<Vec<CyclotomicRing<F, D>>>,
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
    /// # Errors
    ///
    /// Returns an error if the norm check, challenge sampling, or matrix
    /// generation fails.
    pub fn new_prover<T: Transcript<F>>(
        setup: &HachiProverSetup<F, D>,
        ring_opening_point: RingOpeningPoint<F>,
        hint: HachiCommitmentHint<F, D>,
        transcript: &mut T,
        commitment: &RingCommitment<F, D>,
        y_ring: &CyclotomicRing<F, D>,
    ) -> Result<Self, HachiError> {
        let layout = setup.layout();
        let w_hat = compute_w_hat::<F, D>(&ring_opening_point, &hint.ring_coeffs, layout);
        let w_hat_flat = flatten_w_hat(&w_hat);
        let v = compute_v(setup.ntt_cache()?, &w_hat_flat)?;

        transcript.append_serde(ABSORB_PROVER_V, &v);

        let challenge_cfg = SparseChallengeConfig {
            weight: Cfg::CHALLENGE_WEIGHT,
            nonzero_coeffs: vec![-1, 1],
        };
        let challenges = sample_sparse_challenges::<F, T, D>(
            transcript,
            CHALLENGE_STAGE1_FOLD,
            layout.num_blocks,
            &challenge_cfg,
        )?;

        let z_pre = compute_z_pre::<F, D, Cfg>(&hint.ring_coeffs, &challenges, layout)?;

        let y = generate_y::<F, D>(&v, &commitment.u, y_ring, Cfg::N_D, Cfg::N_B, Cfg::N_A)?;

        Ok(Self {
            v,
            challenges,
            y,
            opening_point: ring_opening_point,
            z_pre: Some(z_pre),
            w_hat: Some(w_hat),
            w_hat_flat: Some(w_hat_flat),
            hint: Some(hint),
            _marker: PhantomData,
        })
    }

    /// Verifier constructor: Derives challenges and computes M and y.
    ///
    /// # Errors
    ///
    /// Returns an error if challenge derivation fails.
    pub fn new_verifier<T: Transcript<F>>(
        setup: &HachiVerifierSetup<F, D>,
        ring_opening_point: RingOpeningPoint<F>,
        v: Vec<CyclotomicRing<F, D>>,
        transcript: &mut T,
        commitment: &RingCommitment<F, D>,
        y_ring: &CyclotomicRing<F, D>,
    ) -> Result<Self, HachiError> {
        let layout = setup.expanded.seed.layout;
        let challenges =
            derive_stage1_challenges::<F, T, D, Cfg>(transcript, &v, layout.num_blocks)?;
        let y = generate_y::<F, D>(&v, &commitment.u, y_ring, Cfg::N_D, Cfg::N_B, Cfg::N_A)?;

        Ok(Self {
            v,
            challenges,
            y,
            opening_point: ring_opening_point,
            z_pre: None,
            w_hat: None,
            w_hat_flat: None,
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
        self.z_pre.as_deref()
    }

    /// Take ownership of `z_pre`, leaving `None` in its place.
    pub fn take_z_pre(&mut self) -> Option<Vec<CyclotomicRing<F, D>>> {
        self.z_pre.take()
    }

    /// Get the decomposed witness `ŵ` (prover only).
    pub fn w_hat(&self) -> Option<&[Vec<CyclotomicRing<F, D>>]> {
        self.w_hat.as_deref()
    }

    /// Get the pre-flattened `w_hat` (prover only).
    pub fn w_hat_flat(&self) -> Option<&[CyclotomicRing<F, D>]> {
        self.w_hat_flat.as_deref()
    }

    /// Take ownership of `w_hat`, leaving `None` in its place.
    pub fn take_w_hat(&mut self) -> Option<Vec<Vec<CyclotomicRing<F, D>>>> {
        self.w_hat.take()
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
) -> Result<Vec<SparseChallenge>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let challenge_cfg = SparseChallengeConfig {
        weight: Cfg::CHALLENGE_WEIGHT,
        nonzero_coeffs: vec![-1, 1],
    };
    transcript.append_serde(ABSORB_PROVER_V, v);
    sample_sparse_challenges::<F, T, D>(
        transcript,
        CHALLENGE_STAGE1_FOLD,
        num_blocks,
        &challenge_cfg,
    )
}

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

/// Accumulate unreduced polynomial product `a * b` into `poly` (length 2D-1).
fn add_unreduced_product<F: FieldCore, const D: usize>(
    poly: &mut [F],
    a: &CyclotomicRing<F, D>,
    b: &CyclotomicRing<F, D>,
) {
    if a.is_zero() {
        return;
    }
    let ac = a.coefficients();
    let bc = b.coefficients();
    let is_scalar = ac[1..].iter().all(|c| c.is_zero());
    if is_scalar {
        let s = ac[0];
        for k in 0..D {
            poly[k] = poly[k] + s * bc[k];
        }
    } else {
        for t in 0..D {
            for s in 0..D {
                poly[t + s] = poly[t + s] + ac[t] * bc[s];
            }
        }
    }
}

/// Accumulate negated unreduced product `-a * b` into `poly`.
fn sub_unreduced_product<F: FieldCore, const D: usize>(
    poly: &mut [F],
    a: &CyclotomicRing<F, D>,
    b: &CyclotomicRing<F, D>,
) {
    if a.is_zero() {
        return;
    }
    let ac = a.coefficients();
    let bc = b.coefficients();
    let is_scalar = ac[1..].iter().all(|c| c.is_zero());
    if is_scalar {
        let s = ac[0];
        for k in 0..D {
            poly[k] = poly[k] - s * bc[k];
        }
    } else {
        for t in 0..D {
            for s in 0..D {
                poly[t + s] = poly[t + s] - ac[t] * bc[s];
            }
        }
    }
}

/// Add scalar * ring_element into the low-D coefficients of `poly`.
/// scalar * ring produces degree D-1, so no high-half contribution.
fn add_scalar_ring_product<F: FieldCore, const D: usize>(
    poly: &mut [F],
    scalar: &F,
    ring: &CyclotomicRing<F, D>,
) {
    for (k, coeff) in ring.coefficients().iter().enumerate() {
        poly[k] = poly[k] + *scalar * *coeff;
    }
}

/// Subtract scalar * ring_element from the low-D coefficients of `poly`.
fn sub_scalar_ring_product<F: FieldCore, const D: usize>(
    poly: &mut [F],
    scalar: &F,
    ring: &CyclotomicRing<F, D>,
) {
    for (k, coeff) in ring.coefficients().iter().enumerate() {
        poly[k] = poly[k] - *scalar * *coeff;
    }
}

/// Add sparse_challenge * ring_element as unreduced product into `poly`.
fn add_sparse_ring_product<F: FieldCore + CanonicalField, const D: usize>(
    poly: &mut [F],
    challenge: &SparseChallenge,
    ring: &CyclotomicRing<F, D>,
) {
    let dense: CyclotomicRing<F, D> = challenge.to_dense().expect("valid sparse challenge");
    add_unreduced_product(poly, &dense, ring);
}

/// Split-eq replacement for `generate_m` + `compute_r_via_poly_division`.
///
/// Computes `r` such that `M·z = y + (X^D+1)·r` without materializing M or z.
/// Uses split-eq factoring: `kron(left, gadget) · decomposed = left · pre_decomp`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_r_split_eq<F, const D: usize, Cfg>(
    setup: &HachiExpandedSetup<F, D>,
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    w_hat: &[Vec<CyclotomicRing<F, D>>],
    w_hat_flat: &[CyclotomicRing<F, D>],
    t_hat: &[Vec<CyclotomicRing<F, D>>],
    z_pre: &[CyclotomicRing<F, D>],
    y: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    let layout = setup.seed.layout;
    let delta = layout.delta;
    let log_basis = layout.log_basis;
    let poly_len = 2 * D - 1;
    let num_rows = Cfg::N_D + Cfg::N_B + 1 + 1 + Cfg::N_A;

    let t_hat_flat: Vec<CyclotomicRing<F, D>> =
        t_hat.iter().flat_map(|v| v.iter().copied()).collect();

    let mut result = Vec::with_capacity(num_rows);

    for (row_idx, y_i) in y.iter().enumerate().take(num_rows) {
        let mut poly = vec![F::zero(); poly_len];

        if row_idx < Cfg::N_D {
            let d_row = &setup.D[row_idx];
            for (m_ij, z_j) in d_row.iter().zip(w_hat_flat.iter()) {
                add_unreduced_product(&mut poly, m_ij, z_j);
            }
        } else if row_idx < Cfg::N_D + Cfg::N_B {
            let b_row = &setup.B[row_idx - Cfg::N_D];
            for (m_ij, z_j) in b_row.iter().zip(t_hat_flat.iter()) {
                add_unreduced_product(&mut poly, m_ij, z_j);
            }
        } else if row_idx == Cfg::N_D + Cfg::N_B {
            for (i, w_hat_i) in w_hat.iter().enumerate() {
                let w_recomp = CyclotomicRing::gadget_recompose_pow2(w_hat_i, log_basis);
                add_scalar_ring_product(&mut poly, &opening_point.b[i], &w_recomp);
            }
        } else if row_idx == Cfg::N_D + Cfg::N_B + 1 {
            for (i, w_hat_i) in w_hat.iter().enumerate() {
                let w_recomp = CyclotomicRing::gadget_recompose_pow2(w_hat_i, log_basis);
                add_sparse_ring_product(&mut poly, &challenges[i], &w_recomp);
            }
            let block_len = opening_point.a.len();
            for i in 0..block_len {
                let start = i * delta;
                let end = start + delta;
                if end <= z_pre.len() {
                    let z_pre_recomp =
                        CyclotomicRing::gadget_recompose_pow2(&z_pre[start..end], log_basis);
                    sub_scalar_ring_product(&mut poly, &opening_point.a[i], &z_pre_recomp);
                }
            }
        } else {
            let a_idx = row_idx - (Cfg::N_D + Cfg::N_B + 2);
            for (i, t_hat_i) in t_hat.iter().enumerate() {
                let start = a_idx * delta;
                let end = start + delta;
                if end <= t_hat_i.len() {
                    let t_recomp =
                        CyclotomicRing::gadget_recompose_pow2(&t_hat_i[start..end], log_basis);
                    add_sparse_ring_product(&mut poly, &challenges[i], &t_recomp);
                }
            }
            let a_row = &setup.A[a_idx];
            for (m_ij, z_j) in a_row.iter().zip(z_pre.iter()) {
                sub_unreduced_product(&mut poly, m_ij, z_j);
            }
        }

        let y_coeffs = y_i.coefficients();
        for k in 0..D {
            poly[k] = poly[k] - y_coeffs[k];
        }

        // Divide by X^D + 1
        let mut quotient = vec![F::zero(); D];
        for k in (D..poly_len).rev() {
            let q = poly[k];
            quotient[k - D] = q;
            poly[k - D] = poly[k - D] - q;
        }
        result.push(CyclotomicRing::from_slice(&quotient));
    }

    Ok(result)
}

/// Split-eq replacement for `generate_m` + `eval_ring_matrix_at`.
///
/// Computes the field-element evaluations of each M entry at `alpha`,
/// organized as rows of field elements, without materializing M.
pub(crate) fn compute_m_a_streaming<F, const D: usize, Cfg>(
    setup: &HachiExpandedSetup<F, D>,
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    alpha: &F,
) -> Result<Vec<Vec<F>>, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    let layout = setup.seed.layout;
    let delta = layout.delta;
    let tau = layout.tau;
    let log_basis = layout.log_basis;
    let num_blocks = opening_point.b.len();
    let block_len = layout.block_len;
    let w_len = delta * num_blocks;
    let t_len = delta * Cfg::N_A * num_blocks;
    let z_len = tau * delta * block_len;
    let total_cols = w_len + t_len + z_len;

    let g1 = gadget_row_scalars::<F>(delta, log_basis);
    let j1 = gadget_row_scalars::<F>(tau, log_basis);

    // Pre-evaluate alpha powers for gadget scalars (already field elements)
    // g1 and j1 are already field scalars, so eval_ring_at(constant(g), alpha) = g.

    let mut rows = Vec::with_capacity(Cfg::N_D + Cfg::N_B + 1 + 1 + Cfg::N_A);

    for d_row in setup.D.iter() {
        let mut full = vec![F::zero(); total_cols];
        for (j, ring) in d_row.iter().take(w_len).enumerate() {
            full[j] = eval_ring_at(ring, alpha);
        }
        rows.push(full);
    }

    for b_row in setup.B.iter() {
        let mut full = vec![F::zero(); total_cols];
        for (j, ring) in b_row.iter().take(t_len).enumerate() {
            full[w_len + j] = eval_ring_at(ring, alpha);
        }
        rows.push(full);
    }

    // row3: kron(b, g1) evaluated at alpha -> b[i] * g1[d] (all scalars)
    {
        let mut full = vec![F::zero(); total_cols];
        for (i, &b_i) in opening_point.b.iter().enumerate() {
            for (d, &g) in g1.iter().enumerate() {
                full[i * delta + d] = b_i * g;
            }
        }
        rows.push(full);
    }

    {
        let mut full = vec![F::zero(); total_cols];
        for (i, c) in challenges.iter().enumerate() {
            let c_alpha = eval_ring_at(&c.to_dense::<F, D>().expect("valid challenge"), alpha);
            for (d, &g) in g1.iter().enumerate() {
                full[i * delta + d] = c_alpha * g;
            }
        }
        let z_offset = w_len + t_len;
        for (i, &a_i) in opening_point.a.iter().enumerate() {
            for (d, &g) in g1.iter().enumerate() {
                let ag = a_i * g;
                for (t, &j) in j1.iter().enumerate() {
                    let idx = (i * delta + d) * tau + t;
                    full[z_offset + idx] = -(ag * j);
                }
            }
        }
        rows.push(full);
    }

    for a_idx in 0..Cfg::N_A {
        let mut full = vec![F::zero(); total_cols];
        for (i, c) in challenges.iter().enumerate() {
            let c_alpha = eval_ring_at(&c.to_dense::<F, D>().expect("valid challenge"), alpha);
            for (d, &g) in g1.iter().enumerate() {
                let t_idx = i * (Cfg::N_A * delta) + a_idx * delta + d;
                full[w_len + t_idx] = c_alpha * g;
            }
        }
        let z_offset = w_len + t_len;
        let a_row = &setup.A[a_idx];
        for (k, ring) in a_row.iter().enumerate() {
            let ring_alpha = eval_ring_at(ring, alpha);
            for (t, &j) in j1.iter().enumerate() {
                full[z_offset + k * tau + t] = -(ring_alpha * j);
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
    use crate::algebra::{CyclotomicRing, SparseChallengeConfig};
    use crate::protocol::challenges::sparse::sample_sparse_challenges;
    use crate::protocol::commitment::{HachiCommitmentCore, RingCommitmentScheme};
    use crate::protocol::proof::HachiCommitmentHint;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::test_utils::*;
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
        let hint = HachiCommitmentHint {
            t_hat: w.t_hat,
            ring_coeffs,
        };
        let mut transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_SEED);
        let y_ring = CyclotomicRing::<F, D>::zero();
        let quad_eq = QuadraticEquation::<F, D, TinyConfig>::new_prover(
            &setup,
            point.clone(),
            hint,
            &mut transcript,
            &w.commitment,
            &y_ring,
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

    /// Row 1: D · ŵ = v
    #[test]
    fn row1_d_times_w_hat_equals_v() {
        let f = build_fixture();

        let w_hat = f.quad_eq.w_hat().unwrap();
        let w_hat_flat: Vec<CyclotomicRing<F, D>> =
            w_hat.iter().flat_map(|v| v.iter().copied()).collect();
        let lhs = mat_vec_mul(&f.setup.expanded.D, &w_hat_flat);

        assert_eq!(lhs, f.quad_eq.v(), "Row 1 failed: D · ŵ ≠ v");
    }

    /// Row 2: B · t̂ = u (commitment vector)
    #[test]
    fn row2_b_times_t_hat_equals_u_commitment() {
        let f = build_fixture();

        let hint = f.quad_eq.hint().unwrap();
        let t_hat_flat: Vec<CyclotomicRing<F, D>> =
            hint.t_hat.iter().flat_map(|v| v.iter().copied()).collect();
        let lhs = mat_vec_mul(&f.setup.expanded.B, &t_hat_flat);

        assert_eq!(lhs, f.commitment_u, "Row 2 failed: B · t̂ ≠ u");
    }

    /// Row 3: b^T · G_{2^r} · ŵ = u_eval
    #[test]
    fn row3_bt_gadget_w_hat_equals_u_eval() {
        let f = build_fixture();

        let w_hat = f.quad_eq.w_hat().unwrap();
        let w_recomposed: Vec<CyclotomicRing<F, D>> = w_hat
            .iter()
            .map(|w_hat_i| CyclotomicRing::gadget_recompose_pow2(w_hat_i, LOG_BASIS))
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
            .flat_map(|z_j| z_j.balanced_decompose_pow2(tau(), LOG_BASIS))
            .collect()
    }

    /// Row 4: (c^T ⊗ G_1) · ŵ = a^T · G_{2^m} · J · ẑ
    #[test]
    fn row4_challenge_fold_w_equals_a_gadget_j_z_hat() {
        let f = build_fixture();

        let w_hat = f.quad_eq.w_hat().unwrap();
        let w: Vec<CyclotomicRing<F, D>> = w_hat
            .iter()
            .map(|w_hat_i| CyclotomicRing::gadget_recompose_pow2(w_hat_i, LOG_BASIS))
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

    /// Row 5: (c^T ⊗ G_{n_A}) · t̂ = A · J · ẑ
    #[test]
    fn row5_challenge_fold_t_equals_a_j_z_hat() {
        let f = build_fixture();

        let hint = f.quad_eq.hint().unwrap();
        let mut lhs = vec![CyclotomicRing::<F, D>::zero(); N_A];
        for (c_i, t_hat_i) in f.challenges.iter().zip(hint.t_hat.iter()) {
            let t_i = gadget_recompose_vec(t_hat_i);
            assert_eq!(t_i.len(), N_A);
            for (lhs_j, t_ij) in lhs.iter_mut().zip(t_i.iter()) {
                *lhs_j += *c_i * *t_ij;
            }
        }

        let z_hat = derive_z_hat(f.quad_eq.z_pre().unwrap());
        let z_recovered = recompose_z_hat(&z_hat);
        let rhs = mat_vec_mul(&f.setup.expanded.A, &z_recovered);

        assert_eq!(lhs, rhs, "Row 5 failed: (c^T ⊗ G_nA)t̂ ≠ A · J · ẑ");
    }

    #[test]
    fn prove_output_shapes_are_correct() {
        let f = build_fixture();

        assert_eq!(f.quad_eq.v().len(), TinyConfig::N_D);

        let w_hat = f.quad_eq.w_hat().unwrap();
        assert_eq!(w_hat.len(), NUM_BLOCKS);
        assert!(w_hat.iter().all(|v| v.len() == delta()));

        let hint = f.quad_eq.hint().unwrap();
        assert_eq!(hint.t_hat.len(), NUM_BLOCKS);
        assert!(hint.t_hat.iter().all(|v| v.len() == N_A * delta()));

        assert_eq!(f.quad_eq.z_pre().unwrap().len(), BLOCK_LEN * delta());
    }
}
