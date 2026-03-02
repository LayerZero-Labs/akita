//! Quadratic equation builder for the Hachi PCS (§4.2).
//!
//! This module encapsulates the stage-1 prover logic and the generation of
//! the quadratic equation components M, y, z, and v.

use crate::algebra::ring::{CyclotomicRing, SparseChallenge, SparseChallengeConfig};
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
use crate::protocol::transcript::labels::{ABSORB_PROVER_V, CHALLENGE_STAGE1_FOLD};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};

/// **Steps 1–3.** Compute `w_i = a^T G_{2^m} s_i` and decompose: `ŵ_i = G_1^{-1}(w_i)`.
///
/// Recomputes each block's `s_i` from `ring_coeffs` on the fly to avoid
/// storing all `s_i` simultaneously (which can be tens of GB at production
/// parameters).
fn compute_w_hat<F, const D: usize, Cfg>(
    opening_point: &RingOpeningPoint<F>,
    ring_coeffs: &[CyclotomicRing<F, D>],
    layout: HachiCommitmentLayout,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    let a = &opening_point.a;
    let block_len = layout.block_len;
    let delta = Cfg::DELTA;
    let log_basis = Cfg::LOG_BASIS;

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

    cfg_iter!(blocks)
        .map(|block| {
            let s_i = decompose_block(block, delta, log_basis);
            let mut w_i = CyclotomicRing::<F, D>::zero();
            for (j, a_j) in a.iter().enumerate().take(block_len) {
                let start = j * delta;
                let end = start + delta;
                let recomp_j = CyclotomicRing::gadget_recompose_pow2(&s_i[start..end], log_basis);
                w_i += recomp_j.scale(a_j);
            }
            w_i.balanced_decompose_pow2(delta, log_basis)
        })
        .collect()
}

/// **Step 4.** Compute `v = D · ŵ` (first prover message).
fn compute_v<F: FieldCore + CanonicalField, const D: usize>(
    cache: &NttMatrixCache<D>,
    w_hat: &[Vec<CyclotomicRing<F, D>>],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    let w_hat_flat: Vec<CyclotomicRing<F, D>> =
        w_hat.iter().flat_map(|v| v.iter().copied()).collect();
    mat_vec_mul_ntt_cached(cache, MatrixSlot::D, &w_hat_flat)
}

/// **Steps 7–9.** Fold `z = Σ c_i · s_i`, check `‖z‖_∞ ≤ β`, and decompose `ẑ = J^{-1}(z)`.
///
/// Recomputes each block's `s_i` from `ring_coeffs` one block at a time,
/// accumulating `z[j] += c_i * s_i[j]` across blocks. Peak memory is one
/// block's `s_i` rather than the full `s` tensor.
fn compute_z_hat<F, const D: usize, Cfg>(
    ring_coeffs: &[CyclotomicRing<F, D>],
    challenges: &[SparseChallenge],
    layout: HachiCommitmentLayout,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    let block_len = layout.block_len;
    let delta = Cfg::DELTA;
    let log_basis = Cfg::LOG_BASIS;
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

    Ok(z.iter()
        .flat_map(|z_j| z_j.balanced_decompose_pow2(Cfg::TAU, Cfg::LOG_BASIS))
        .collect())
}

/// Stage-1 quadratic equation state for the Hachi protocol.
///
/// Encapsulates the relation $M(x) \cdot z = y(x) + (X^D + 1) \cdot r(x)$
/// along with intermediate prover witness data (`w_hat`, `z_hat`, `hint`).
pub struct QuadraticEquation<F: FieldCore, const D: usize, Cfg: CommitmentConfig> {
    /// Stage-1 proof vector `v = D · ŵ`.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// Stage-1 folding challenges (sparse representation).
    pub challenges: Vec<SparseChallenge>,
    /// Matrix `M`.
    m: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Vector `y`.
    y: Vec<CyclotomicRing<F, D>>,
    /// Vector `z` (prover only).
    z: Option<Vec<CyclotomicRing<F, D>>>,
    /// Decomposed `ŵ_i = G_1^{-1}(w_i)` (prover only).
    w_hat: Option<Vec<Vec<CyclotomicRing<F, D>>>>,
    /// Decomposed `ẑ = J^{-1}(z)` (prover only).
    z_hat: Option<Vec<CyclotomicRing<F, D>>>,
    /// Commitment hint (prover only).
    hint: Option<HachiCommitmentHint<F, D>>,

    _marker: std::marker::PhantomData<Cfg>,
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
        ring_opening_point: &RingOpeningPoint<F>,
        hint: &HachiCommitmentHint<F, D>,
        transcript: &mut T,
        commitment: &RingCommitment<F, D>,
        y_ring: &CyclotomicRing<F, D>,
    ) -> Result<Self, HachiError> {
        let layout = setup.layout();
        let w_hat = compute_w_hat::<F, D, Cfg>(ring_opening_point, &hint.ring_coeffs, layout);
        let v = compute_v(setup.ntt_cache()?, &w_hat)?;

        // Step 5: append v to transcript
        transcript.append_serde(ABSORB_PROVER_V, &v);

        // Step 6: sample sparse folding challenges
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

        let z_hat = compute_z_hat::<F, D, Cfg>(&hint.ring_coeffs, &challenges, layout)?;

        let m = generate_m::<F, D, Cfg>(&setup.expanded, ring_opening_point, &challenges)?;
        let y = generate_y::<F, D, Cfg>(&v, &commitment.u, y_ring)?;
        let z = generate_z(&w_hat, &hint.t_hat, &z_hat);

        Ok(Self {
            v,
            challenges,
            m,
            y,
            z: Some(z),
            w_hat: Some(w_hat),
            z_hat: Some(z_hat),
            hint: Some(hint.clone()),
            _marker: std::marker::PhantomData,
        })
    }

    /// Verifier constructor: Derives challenges and computes M and y.
    ///
    /// # Errors
    ///
    /// Returns an error if challenge derivation fails.
    pub fn new_verifier<T: Transcript<F>>(
        setup: &HachiVerifierSetup<F, D>,
        ring_opening_point: &RingOpeningPoint<F>,
        v: &Vec<CyclotomicRing<F, D>>,
        transcript: &mut T,
        commitment: &RingCommitment<F, D>,
        y_ring: &CyclotomicRing<F, D>,
    ) -> Result<Self, HachiError> {
        let layout = setup.expanded.seed.layout;
        let challenges =
            derive_stage1_challenges::<F, T, D, Cfg>(transcript, v, layout.num_blocks)?;
        let m = generate_m::<F, D, Cfg>(&setup.expanded, ring_opening_point, &challenges)?;
        let y = generate_y::<F, D, Cfg>(v, &commitment.u, y_ring)?;

        Ok(Self {
            v: v.to_vec(),
            challenges,
            m,
            y,
            z: None,
            w_hat: None,
            z_hat: None,
            hint: None,
            _marker: std::marker::PhantomData,
        })
    }

    /// Get the matrix M.
    pub fn m(&self) -> &[Vec<CyclotomicRing<F, D>>] {
        &self.m
    }

    /// Get the vector y.
    pub fn y(&self) -> &[CyclotomicRing<F, D>] {
        &self.y
    }

    /// Get the vector z (returns None if constructed by verifier).
    pub fn z(&self) -> Option<&[CyclotomicRing<F, D>]> {
        self.z.as_deref()
    }

    /// Get the vector v.
    pub fn v(&self) -> &[CyclotomicRing<F, D>] {
        &self.v
    }

    /// Get the decomposed witness `ŵ` (prover only).
    pub fn w_hat(&self) -> Option<&[Vec<CyclotomicRing<F, D>>]> {
        self.w_hat.as_deref()
    }

    /// Get the decomposed folded witness `ẑ` (prover only).
    pub fn z_hat(&self) -> Option<&[CyclotomicRing<F, D>]> {
        self.z_hat.as_deref()
    }

    /// Get the commitment hint (prover only).
    pub fn hint(&self) -> Option<&HachiCommitmentHint<F, D>> {
        self.hint.as_ref()
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

fn constant_ring<F: FieldCore, const D: usize>(value: F) -> CyclotomicRing<F, D> {
    let mut coeffs = [F::zero(); D];
    coeffs[0] = value;
    CyclotomicRing::from_coefficients(coeffs)
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

/// Kronecker product of two scalar vectors, producing constant ring elements.
/// Cost: O(1) per pair (just a field multiplication + ring construction).
fn kron_scalars<F: FieldCore, const D: usize>(
    left: &[F],
    right: &[F],
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(left.len().saturating_mul(right.len()));
    for &l in left {
        for &r in right {
            out.push(constant_ring(l * r));
        }
    }
    out
}

/// Kronecker product where the right operand is field scalars.
/// Cost: O(D) per pair instead of O(D^2).
fn kron_row_scale<F: FieldCore, const D: usize>(
    left: &[CyclotomicRing<F, D>],
    right: &[F],
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(left.len().saturating_mul(right.len()));
    for l in left {
        for &r in right {
            out.push(l.scale(&r));
        }
    }
    out
}

/// Kronecker product of sparse challenges with scalar gadget entries.
/// Cost: O(omega + D) per pair instead of O(D^2).
fn kron_sparse_scale<F: FieldCore + CanonicalField, const D: usize>(
    left: &[SparseChallenge],
    right: &[F],
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(left.len().saturating_mul(right.len()));
    for l in left {
        let dense: CyclotomicRing<F, D> = l.to_dense().expect("valid sparse challenge");
        for &r in right {
            out.push(dense.scale(&r));
        }
    }
    out
}

fn gadget_block_diag_scalars<F: FieldCore>(blocks: usize, row: &[F]) -> Vec<Vec<F>> {
    let row_len = row.len();
    let mut rows = Vec::with_capacity(blocks);
    for i in 0..blocks {
        let mut out = vec![F::zero(); blocks * row_len];
        let start = i * row_len;
        out[start..start + row_len].copy_from_slice(row);
        rows.push(out);
    }
    rows
}

pub(crate) fn generate_m<F, const D: usize, Cfg: CommitmentConfig>(
    setup: &HachiExpandedSetup<F, D>,
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError>
where
    F: FieldCore + CanonicalField,
{
    let layout = setup.seed.layout;
    let num_blocks = layout.num_blocks;
    let block_len = layout.block_len;
    let w_len = Cfg::DELTA
        .checked_mul(num_blocks)
        .ok_or_else(|| HachiError::InvalidSetup("w length overflow".to_string()))?;
    let t_len = Cfg::DELTA
        .checked_mul(Cfg::N_A)
        .and_then(|v| v.checked_mul(num_blocks))
        .ok_or_else(|| HachiError::InvalidSetup("t length overflow".to_string()))?;
    let z_len = Cfg::TAU
        .checked_mul(Cfg::DELTA)
        .and_then(|v| v.checked_mul(block_len))
        .ok_or_else(|| HachiError::InvalidSetup("z length overflow".to_string()))?;
    let total_cols = w_len
        .checked_add(t_len)
        .and_then(|v| v.checked_add(z_len))
        .ok_or_else(|| HachiError::InvalidSetup("matrix width overflow".to_string()))?;

    if opening_point.b.len() != num_blocks {
        return Err(HachiError::InvalidPointDimension {
            expected: num_blocks,
            actual: opening_point.b.len(),
        });
    }
    if opening_point.a.len() != block_len {
        return Err(HachiError::InvalidPointDimension {
            expected: block_len,
            actual: opening_point.a.len(),
        });
    }
    if challenges.len() != num_blocks {
        return Err(HachiError::InvalidSize {
            expected: num_blocks,
            actual: challenges.len(),
        });
    }
    if setup.D.len() != Cfg::N_D {
        return Err(HachiError::InvalidSize {
            expected: Cfg::N_D,
            actual: setup.D.len(),
        });
    }
    if setup.B.len() != Cfg::N_B {
        return Err(HachiError::InvalidSize {
            expected: Cfg::N_B,
            actual: setup.B.len(),
        });
    }
    if setup.A.len() != Cfg::N_A {
        return Err(HachiError::InvalidSize {
            expected: Cfg::N_A,
            actual: setup.A.len(),
        });
    }
    if setup.A.first().map(|row| row.len()) != Some(block_len * Cfg::DELTA) {
        return Err(HachiError::InvalidSetup("A row width mismatch".to_string()));
    }

    let g1 = gadget_row_scalars::<F>(Cfg::DELTA, Cfg::LOG_BASIS);
    let j1 = gadget_row_scalars::<F>(Cfg::TAU, Cfg::LOG_BASIS);

    let row3_w = kron_scalars::<F, D>(&opening_point.b, &g1);
    let row4_w = kron_sparse_scale::<F, D>(challenges, &g1);

    let ag = kron_scalars::<F, D>(&opening_point.a, &g1);
    let ag_scalars: Vec<F> = ag.iter().map(|r| r.coefficients()[0]).collect();
    let row4_z = kron_scalars::<F, D>(&ag_scalars, &j1)
        .into_iter()
        .map(|x| -x)
        .collect::<Vec<_>>();

    let g_na = gadget_block_diag_scalars::<F>(Cfg::N_A, &g1);
    let row5_mid = g_na
        .iter()
        .map(|row| kron_sparse_scale::<F, D>(challenges, row))
        .collect::<Vec<_>>();
    let row5_right = setup
        .A
        .iter()
        .map(|row| kron_row_scale(row, &j1).into_iter().map(|x| -x).collect())
        .collect::<Vec<Vec<_>>>();

    let zero = CyclotomicRing::<F, D>::zero();
    let mut rows = Vec::with_capacity(Cfg::N_D + Cfg::N_B + 1usize + 1usize + Cfg::N_A);

    for row in setup.D.iter() {
        if row.len() != w_len {
            return Err(HachiError::InvalidSetup("D row width mismatch".to_string()));
        }
        let mut full = vec![zero; total_cols];
        full[..w_len].copy_from_slice(row);
        rows.push(full);
    }

    for row in setup.B.iter() {
        if row.len() != t_len {
            return Err(HachiError::InvalidSetup("B row width mismatch".to_string()));
        }
        let mut full = vec![zero; total_cols];
        full[w_len..w_len + t_len].copy_from_slice(row);
        rows.push(full);
    }

    let mut row3 = vec![zero; total_cols];
    row3[..w_len].copy_from_slice(&row3_w);
    rows.push(row3);

    let mut row4 = vec![zero; total_cols];
    row4[..w_len].copy_from_slice(&row4_w);
    row4[w_len + t_len..].copy_from_slice(&row4_z);
    rows.push(row4);

    for (mid, right) in row5_mid.into_iter().zip(row5_right.into_iter()) {
        let mut row = vec![zero; total_cols];
        row[w_len..w_len + t_len].copy_from_slice(&mid);
        row[w_len + t_len..].copy_from_slice(&right);
        rows.push(row);
    }

    Ok(rows)
}

pub(crate) fn generate_z<F: FieldCore, const D: usize>(
    w_hat: &[Vec<CyclotomicRing<F, D>>],
    t_hat: &[Vec<CyclotomicRing<F, D>>],
    z_hat: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(
        w_hat.len()
            + t_hat.len()
            + z_hat.len()
            + w_hat.iter().map(|v| v.len()).sum::<usize>()
            + t_hat.iter().map(|v| v.len()).sum::<usize>(),
    );
    for w in w_hat {
        out.extend(w.iter().copied());
    }
    for t in t_hat {
        out.extend(t.iter().copied());
    }
    out.extend_from_slice(z_hat);
    out
}

pub(crate) fn generate_y<F, const D: usize, Cfg: CommitmentConfig>(
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
    u_eval: &CyclotomicRing<F, D>,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore,
{
    if v.len() != Cfg::N_D {
        return Err(HachiError::InvalidSize {
            expected: Cfg::N_D,
            actual: v.len(),
        });
    }
    if u.len() != Cfg::N_B {
        return Err(HachiError::InvalidSize {
            expected: Cfg::N_B,
            actual: u.len(),
        });
    }
    let mut out = Vec::with_capacity(Cfg::N_D + Cfg::N_B + 1 + 1 + Cfg::N_A);
    out.extend_from_slice(v);
    out.extend_from_slice(u);
    out.push(*u_eval);
    out.push(CyclotomicRing::<F, D>::zero());
    out.extend(std::iter::repeat_n(
        CyclotomicRing::<F, D>::zero(),
        Cfg::N_A,
    ));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::SparseChallengeConfig;
    use crate::algebra::CyclotomicRing;
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
            &point,
            &hint,
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

        let z_recovered = recompose_z_hat(f.quad_eq.z_hat().unwrap());
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

        let z_recovered = recompose_z_hat(f.quad_eq.z_hat().unwrap());
        let rhs = mat_vec_mul(&f.setup.expanded.A, &z_recovered);

        assert_eq!(lhs, rhs, "Row 5 failed: (c^T ⊗ G_nA)t̂ ≠ A · J · ẑ");
    }

    #[test]
    fn prove_output_shapes_are_correct() {
        let f = build_fixture();

        assert_eq!(f.quad_eq.v().len(), TinyConfig::N_D);

        let w_hat = f.quad_eq.w_hat().unwrap();
        assert_eq!(w_hat.len(), NUM_BLOCKS);
        assert!(w_hat.iter().all(|v| v.len() == DELTA));

        let hint = f.quad_eq.hint().unwrap();
        assert_eq!(hint.t_hat.len(), NUM_BLOCKS);
        assert!(hint.t_hat.iter().all(|v| v.len() == N_A * DELTA));

        assert_eq!(f.quad_eq.z_hat().unwrap().len(), BLOCK_LEN * DELTA * TAU);
    }
}
