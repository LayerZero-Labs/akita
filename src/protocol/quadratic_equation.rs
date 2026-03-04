//! Quadratic equation builder for the Hachi PCS (§4.2).
//!
//! This module encapsulates the stage-1 prover logic and the generation of
//! the quadratic equation components M, y, z, and v.

use crate::algebra::{CyclotomicRing, SparseChallenge, SparseChallengeConfig};
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::challenges::sparse::sample_sparse_challenges;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::linear::{
    flatten_i8_blocks, mat_vec_mul_ntt_single_i8, unreduced_quotient_rows_ntt_cached,
    unreduced_quotient_rows_ntt_cached_i8,
};
use crate::protocol::commitment::utils::norm::{detect_field_modulus, vec_inf_norm};
use crate::protocol::commitment::{
    CommitmentConfig, HachiCommitmentLayout, HachiExpandedSetup, HachiProverSetup, RingCommitment,
};
use crate::protocol::hachi_poly_ops::HachiPolyOps;
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::proof::HachiCommitmentHint;
use crate::protocol::ring_switch::eval_ring_at;
use crate::protocol::transcript::labels::{ABSORB_PROVER_V, CHALLENGE_STAGE1_FOLD};
use crate::protocol::transcript::Transcript;
use crate::{cfg_iter, CanonicalField, FieldCore};
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

/// **Steps 7–9.** Fold `z_pre = Σ c_i · s_i` and check `‖z_pre‖_∞ ≤ β`.
///
/// Uses `HachiPolyOps::decompose_fold` to carry out the decompose + fold
/// in whatever way the polynomial implementation prefers.
fn compute_z_pre<F, const D: usize, Cfg, P>(
    poly: &P,
    challenges: &[SparseChallenge],
    layout: HachiCommitmentLayout,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
    P: HachiPolyOps<F, D>,
{
    let z = poly.decompose_fold(
        challenges,
        layout.block_len,
        layout.num_digits_commit,
        layout.log_basis,
    );

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
    pub fn new_prover<T: Transcript<F>, P: HachiPolyOps<F, D>>(
        setup: &HachiProverSetup<F, D>,
        ring_opening_point: RingOpeningPoint<F>,
        poly: &P,
        pre_folded: Vec<CyclotomicRing<F, D>>,
        hint: HachiCommitmentHint<F, D>,
        transcript: &mut T,
        commitment: &RingCommitment<F, D>,
        y_ring: &CyclotomicRing<F, D>,
        layout: HachiCommitmentLayout,
    ) -> Result<Self, HachiError> {
        let t_wh = Instant::now();
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
        eprintln!(
            "    [quad_eq] decompose_w_hat+flatten: {:.2}s (blocks={}, depth={})",
            t_wh.elapsed().as_secs_f64(),
            w_hat.len(),
            w_hat.first().map_or(0, |v| v.len())
        );

        let t_v = Instant::now();
        let v = {
            let _span = tracing::info_span!("compute_v").entered();
            compute_v(&setup.ntt_D, &w_hat_flat)
        };
        eprintln!(
            "    [quad_eq] compute_v (D*w_hat): {:.2}s (w_hat_flat_len={})",
            t_v.elapsed().as_secs_f64(),
            w_hat_flat.len()
        );

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

        let t_zp = Instant::now();
        let z_pre = {
            let _span = tracing::info_span!("compute_z_pre").entered();
            compute_z_pre::<F, D, Cfg, P>(poly, &challenges, layout)?
        };
        eprintln!(
            "    [quad_eq] compute_z_pre: {:.2}s (z_pre_len={})",
            t_zp.elapsed().as_secs_f64(),
            z_pre.len()
        );

        let y = generate_y::<F, D>(&v, &commitment.u, y_ring, Cfg::N_D, Cfg::N_B, Cfg::N_A)?;

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
    pub fn new_verifier<T: Transcript<F>>(
        ring_opening_point: RingOpeningPoint<F>,
        v: Vec<CyclotomicRing<F, D>>,
        transcript: &mut T,
        commitment: &RingCommitment<F, D>,
        y_ring: &CyclotomicRing<F, D>,
        layout: HachiCommitmentLayout,
    ) -> Result<Self, HachiError> {
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
        self.z_pre.as_deref()
    }

    /// Take ownership of `z_pre`, leaving `None` in its place.
    pub fn take_z_pre(&mut self) -> Option<Vec<CyclotomicRing<F, D>>> {
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

/// Add scalar * ring_element into the low-D coefficients of `poly`.
/// scalar * ring produces degree D-1, so no high-half contribution.
fn add_scalar_ring_product<F: FieldCore, const D: usize>(
    poly: &mut [F],
    scalar: &F,
    ring: &CyclotomicRing<F, D>,
) {
    for (k, coeff) in ring.coefficients().iter().enumerate() {
        poly[k] += *scalar * *coeff;
    }
}

/// Subtract scalar * ring_element from the low-D coefficients of `poly`.
fn sub_scalar_ring_product<F: FieldCore, const D: usize>(
    poly: &mut [F],
    scalar: &F,
    ring: &CyclotomicRing<F, D>,
) {
    for (k, coeff) in ring.coefficients().iter().enumerate() {
        poly[k] -= *scalar * *coeff;
    }
}

/// Add sparse_challenge * ring_element as unreduced product into `poly`.
///
/// Exploits sparsity: O(weight * D) instead of O(D^2) schoolbook.
fn add_sparse_ring_product<F: FieldCore + CanonicalField, const D: usize>(
    poly: &mut [F],
    challenge: &SparseChallenge,
    ring: &CyclotomicRing<F, D>,
) {
    let rc = ring.coefficients();
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let c = F::from_i64(coeff as i64);
        let p = pos as usize;
        for (s, &r_s) in rc.iter().enumerate() {
            poly[p + s] += c * r_s;
        }
    }
}

/// Split-eq replacement for `generate_m` + `compute_r_via_poly_division`.
///
/// Computes `r` such that `M·z = y + (X^D+1)·r` without materializing M or z.
/// Uses split-eq factoring: `kron(left, gadget) · decomposed = left · pre_decomp`.
#[allow(clippy::too_many_arguments, clippy::needless_borrow)]
#[tracing::instrument(skip_all, name = "compute_r_split_eq")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_r_split_eq<F, const D: usize, Cfg>(
    _setup: &HachiExpandedSetup<F, D>,
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    w_hat_flat: &[[i8; D]],
    t_hat: &[Vec<[i8; D]>],
    w_folded: &[CyclotomicRing<F, D>],
    z_pre: &[CyclotomicRing<F, D>],
    y: &[CyclotomicRing<F, D>],
    ntt_a: &NttSlotCache<D>,
    ntt_b: &NttSlotCache<D>,
    ntt_d: &NttSlotCache<D>,
    layout: HachiCommitmentLayout,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    let decomp_commit = layout.num_digits_commit;
    let decomp_open = layout.num_digits_open;
    let log_basis = layout.log_basis;
    let poly_len = 2 * D - 1;
    let num_rows = Cfg::N_D + Cfg::N_B + 1 + 1 + Cfg::N_A;

    let t_hat_flat = flatten_i8_blocks(t_hat);

    // NTT-accelerated D, B, and A rows: compute quotient = (cyc - neg) / 2
    let t_d = Instant::now();
    let d_quotients = {
        let _span = tracing::info_span!("D_rows_ntt").entered();
        unreduced_quotient_rows_ntt_cached_i8(ntt_d, w_hat_flat)
    };
    let d_time = t_d.elapsed().as_secs_f64();

    let t_b = Instant::now();
    let b_quotients = {
        let _span = tracing::info_span!("B_rows_ntt").entered();
        unreduced_quotient_rows_ntt_cached_i8(ntt_b, &t_hat_flat)
    };
    let b_time = t_b.elapsed().as_secs_f64();

    let t_a = Instant::now();
    let a_quotients = {
        let _span = tracing::info_span!("A_rows_ntt").entered();
        unreduced_quotient_rows_ntt_cached(ntt_a, z_pre)
    };
    let a_time = t_a.elapsed().as_secs_f64();

    let mut result = Vec::with_capacity(num_rows);
    let mut other_time = 0.0f64;
    let mut poly_buf = vec![F::zero(); poly_len];
    let mut quotient_buf = vec![F::zero(); D];

    for (row_idx, _y_i) in y.iter().enumerate().take(num_rows) {
        if row_idx < Cfg::N_D {
            result.push(d_quotients[row_idx]);
        } else if row_idx < Cfg::N_D + Cfg::N_B {
            result.push(b_quotients[row_idx - Cfg::N_D]);
        } else if row_idx >= Cfg::N_D + Cfg::N_B + 2 {
            // A-rows: NTT-accelerated A*z_pre + sparse challenge terms
            let t_row = Instant::now();
            let _span = tracing::info_span!("A_row").entered();
            let a_idx = row_idx - (Cfg::N_D + Cfg::N_B + 2);

            poly_buf.fill(F::zero());
            for (i, t_hat_i) in t_hat.iter().enumerate() {
                let start = a_idx * decomp_open;
                let end = start + decomp_open;
                if end <= t_hat_i.len() {
                    let t_recomp =
                        CyclotomicRing::gadget_recompose_pow2_i8(&t_hat_i[start..end], log_basis);
                    add_sparse_ring_product(&mut poly_buf, &challenges[i], &t_recomp);
                }
            }

            let a_q = a_quotients[a_idx].coefficients();
            quotient_buf.fill(F::zero());
            quotient_buf[..(poly_len - D)].copy_from_slice(&poly_buf[D..poly_len]);
            for k in 0..D {
                quotient_buf[k] -= a_q[k];
            }
            result.push(CyclotomicRing::from_slice(&quotient_buf));
            other_time += t_row.elapsed().as_secs_f64();
        } else {
            // bTw_row and challenge_fold_row: schoolbook (cheap)
            let t_row = Instant::now();
            poly_buf.fill(F::zero());

            if row_idx == Cfg::N_D + Cfg::N_B {
                let _span = tracing::info_span!("bTw_row").entered();
                for (i, w_f) in w_folded.iter().enumerate() {
                    add_scalar_ring_product(&mut poly_buf, &opening_point.b[i], w_f);
                }
            } else {
                let _span = tracing::info_span!("challenge_fold_row").entered();
                for (i, w_f) in w_folded.iter().enumerate() {
                    add_sparse_ring_product(&mut poly_buf, &challenges[i], w_f);
                }
                let block_len = opening_point.a.len();
                for i in 0..block_len {
                    let start = i * decomp_commit;
                    let end = start + decomp_commit;
                    if end <= z_pre.len() {
                        let z_pre_recomp =
                            CyclotomicRing::gadget_recompose_pow2(&z_pre[start..end], log_basis);
                        sub_scalar_ring_product(&mut poly_buf, &opening_point.a[i], &z_pre_recomp);
                    }
                }
            }

            let y_coeffs = _y_i.coefficients();
            for k in 0..D {
                poly_buf[k] -= y_coeffs[k];
            }

            quotient_buf.fill(F::zero());
            for k in (D..poly_len).rev() {
                let q = poly_buf[k];
                quotient_buf[k - D] = q;
                poly_buf[k - D] -= q;
            }
            result.push(CyclotomicRing::from_slice(&quotient_buf));
            other_time += t_row.elapsed().as_secs_f64();
        }
    }

    eprintln!(
        "      [compute_r] D(NTT): {d_time:.2}s, B(NTT): {b_time:.2}s, A(NTT): {a_time:.2}s, other: {other_time:.2}s",
    );

    Ok(result)
}

/// Split-eq replacement for `generate_m` + `eval_ring_matrix_at`.
///
/// Computes the field-element evaluations of each M entry at `alpha`,
/// organized as rows of field elements, without materializing M.
#[tracing::instrument(skip_all, name = "compute_m_a_streaming")]
pub(crate) fn compute_m_a_streaming<F, const D: usize, Cfg>(
    setup: &HachiExpandedSetup<F, D>,
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    alpha: &F,
    layout: HachiCommitmentLayout,
) -> Result<Vec<Vec<F>>, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    let depth_commit = layout.num_digits_commit;
    let depth_open = layout.num_digits_open;
    let depth_fold = layout.num_digits_fold;
    let log_basis = layout.log_basis;
    let num_blocks = opening_point.b.len();
    let block_len = layout.block_len;
    let w_len = depth_open * num_blocks;
    let t_len = depth_open * Cfg::N_A * num_blocks;
    let z_len = depth_fold * depth_commit * block_len;
    let total_cols = w_len + t_len + z_len;

    let g1_open = gadget_row_scalars::<F>(depth_open, log_basis);
    let g1_commit = gadget_row_scalars::<F>(depth_commit, log_basis);
    let j1 = gadget_row_scalars::<F>(depth_fold, log_basis);

    let c_alphas: Vec<F> = challenges
        .iter()
        .map(|c| eval_ring_at(&c.to_dense::<F, D>().expect("valid challenge"), alpha))
        .collect();

    let d_rows: Vec<Vec<F>> = cfg_iter!(setup.D)
        .map(|d_row| {
            let mut full = vec![F::zero(); total_cols];
            for (j, ring) in d_row.iter().take(w_len).enumerate() {
                full[j] = eval_ring_at(ring, alpha);
            }
            full
        })
        .collect();

    let b_rows: Vec<Vec<F>> = cfg_iter!(setup.B)
        .map(|b_row| {
            let mut full = vec![F::zero(); total_cols];
            for (j, ring) in b_row.iter().take(t_len).enumerate() {
                full[w_len + j] = eval_ring_at(ring, alpha);
            }
            full
        })
        .collect();

    let mut rows = Vec::with_capacity(Cfg::N_D + Cfg::N_B + 1 + 1 + Cfg::N_A);
    rows.extend(d_rows);
    rows.extend(b_rows);

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
    for a_idx in 0..Cfg::N_A {
        let mut full = vec![F::zero(); total_cols];
        for (i, &c_alpha) in c_alphas.iter().enumerate() {
            for (d, &g) in g1_open.iter().enumerate() {
                let t_idx = i * (Cfg::N_A * depth_open) + a_idx * depth_open + d;
                full[w_len + t_idx] = c_alpha * g;
            }
        }
        let z_offset = w_len + t_len;
        let a_row = &setup.A[a_idx];
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
    use crate::protocol::commitment::{HachiCommitmentCore, RingCommitmentScheme};
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
        let quad_eq = QuadraticEquation::<F, D, TinyConfig>::new_prover(
            &setup,
            point.clone(),
            &poly,
            w_folded,
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
        let lhs = mat_vec_mul(&f.setup.expanded.D, &w_hat_flat);

        assert_eq!(lhs, f.quad_eq.v(), "Row 1 failed: D · ŵ ≠ v");
    }

    /// Row 2: B · t̂ = u (commitment vector)
    #[test]
    fn row2_b_times_t_hat_equals_u_commitment() {
        let f = build_fixture();

        let hint = f.quad_eq.hint().unwrap();
        let t_hat_flat_ring: Vec<CyclotomicRing<F, D>> = hint
            .t_hat
            .iter()
            .flat_map(|v| v.iter())
            .map(|plane| {
                let coeffs: [F; D] = std::array::from_fn(|k| F::from_i64(plane[k] as i64));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();
        let lhs = mat_vec_mul(&f.setup.expanded.B, &t_hat_flat_ring);

        assert_eq!(lhs, f.commitment_u, "Row 2 failed: B · t̂ ≠ u");
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

    /// Row 5: (c^T ⊗ G_{n_A}) · t̂ = A · J · ẑ
    #[test]
    fn row5_challenge_fold_t_equals_a_j_z_hat() {
        let f = build_fixture();

        let hint = f.quad_eq.hint().unwrap();
        let mut lhs = vec![CyclotomicRing::<F, D>::zero(); N_A];
        for (c_i, t_hat_i) in f.challenges.iter().zip(hint.t_hat.iter()) {
            let t_i = gadget_recompose_vec_i8(t_hat_i);
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
        assert!(w_hat.iter().all(|v| v.len() == num_digits_open()));

        let hint = f.quad_eq.hint().unwrap();
        assert_eq!(hint.t_hat.len(), NUM_BLOCKS);
        assert!(hint
            .t_hat
            .iter()
            .all(|v| v.len() == N_A * num_digits_open()));

        assert_eq!(
            f.quad_eq.z_pre().unwrap().len(),
            BLOCK_LEN * num_digits_commit()
        );
    }
}
