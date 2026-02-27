//! Quadratic equation builder for the Hachi PCS.
//!
//! This module encapsulates the logic for generating the components M, y, z, and v
//! as described in §4.2 of the Hachi paper.

use crate::algebra::ring::{CyclotomicRing, SparseChallengeConfig};
use crate::error::HachiError;
use crate::protocol::challenges::sparse::sample_dense_challenges;
use crate::protocol::commitment::{CommitmentConfig, RingCommitment, RingCommitmentSetup};
use crate::protocol::iteration_prover::HachiProver;
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::proof::HachiCommitmentHint;
use crate::protocol::transcript::labels::{ABSORB_PROVER_V, CHALLENGE_STAGE1_FOLD};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};

/// Represents the quadratic equation components used in the Hachi protocol.
///
/// Encapsulates the relation: $M(x) \cdot z = y(x) + (X^D + 1) \cdot r(x)$.
pub struct QuadraticEquation<F: FieldCore, const D: usize, Cfg: CommitmentConfig> {
    /// Stage-1 challenge vector `v`.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// Stage-1 folding challenges.
    pub challenges: Vec<CyclotomicRing<F, D>>,
    /// Matrix `M`.
    pub m: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Vector `y`.
    pub y: Vec<CyclotomicRing<F, D>>,
    /// Vector `z` (only for prover).
    pub z: Option<Vec<CyclotomicRing<F, D>>>,

    _marker: std::marker::PhantomData<Cfg>,
}

impl<F, const D: usize, Cfg> QuadraticEquation<F, D, Cfg>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    /// Prover constructor: Runs prove_stage1 and computes all components.
    ///
    /// # Errors
    ///
    /// Returns an error if any stage fails.
    pub fn new_prover<T: Transcript<F>>(
        setup: &RingCommitmentSetup<F, D>,
        ring_opening_point: &RingOpeningPoint<F, D>,
        hint: &HachiCommitmentHint<F, D>,
        transcript: &mut T,
        commitment: &RingCommitment<F, D>,
        y_ring: &CyclotomicRing<F, D>,
    ) -> Result<Self, HachiError> {
        let mut prover = HachiProver::<F, D>::new();
        let (v, challenges) =
            prover.prove_stage1::<T, Cfg>(setup, ring_opening_point, transcript, hint)?;

        let m = generate_m::<F, D, Cfg>(setup, ring_opening_point, &challenges)?;
        let y = generate_y::<F, D, Cfg>(&v, &commitment.u, y_ring)?;
        let z = generate_z(&prover.w_hat, &hint.t_hat, &prover.z_hat);

        Ok(Self {
            v,
            challenges,
            m,
            y,
            z: Some(z),
            _marker: std::marker::PhantomData,
        })
    }

    /// Verifier constructor: Derives challenges and computes M and y.
    ///
    /// # Errors
    ///
    /// Returns an error if challenge derivation fails.
    pub fn new_verifier<T: Transcript<F>>(
        setup: &RingCommitmentSetup<F, D>,
        ring_opening_point: &RingOpeningPoint<F, D>,
        v: &Vec<CyclotomicRing<F, D>>,
        transcript: &mut T,
        commitment: &RingCommitment<F, D>,
        y_ring: &CyclotomicRing<F, D>,
    ) -> Result<Self, HachiError> {
        let challenges = derive_stage1_challenges::<F, T, D, Cfg>(transcript, v)?;
        let m = generate_m::<F, D, Cfg>(setup, ring_opening_point, &challenges)?;
        let y = generate_y::<F, D, Cfg>(v, &commitment.u, y_ring)?;

        Ok(Self {
            v: v.clone(),
            challenges,
            m,
            y,
            z: None,
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
}

// --- Helper Functions (Moved from commitment_scheme.rs) ---

pub(crate) fn derive_stage1_challenges<F, T, const D: usize, Cfg: CommitmentConfig>(
    transcript: &mut T,
    v: &Vec<CyclotomicRing<F, D>>,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let challenge_cfg = SparseChallengeConfig {
        weight: Cfg::CHALLENGE_WEIGHT,
        nonzero_coeffs: vec![-1, 1],
    };
    let num_blocks = 1usize
        .checked_shl(Cfg::R as u32)
        .ok_or_else(|| HachiError::InvalidSetup("2^R does not fit usize".to_string()))?;
    transcript.append_serde(ABSORB_PROVER_V, v);
    sample_dense_challenges::<F, T, D>(
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

fn gadget_row<F: FieldCore + CanonicalField, const D: usize>(
    levels: usize,
    log_basis: u32,
) -> Vec<CyclotomicRing<F, D>> {
    let base = F::from_canonical_u128_reduced(1u128 << log_basis);
    let mut out = Vec::with_capacity(levels);
    let mut power = F::one();
    for _ in 0..levels {
        out.push(constant_ring::<F, D>(power));
        power = power * base;
    }
    out
}

fn kron_row<F: FieldCore, const D: usize>(
    left: &[CyclotomicRing<F, D>],
    right: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(left.len().saturating_mul(right.len()));
    for l in left {
        for r in right {
            out.push(*l * *r);
        }
    }
    out
}

fn gadget_block_diag<F: FieldCore, const D: usize>(
    blocks: usize,
    row: &[CyclotomicRing<F, D>],
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let row_len = row.len();
    let mut rows = Vec::with_capacity(blocks);
    for i in 0..blocks {
        let mut out = vec![CyclotomicRing::<F, D>::zero(); blocks * row_len];
        let start = i * row_len;
        out[start..start + row_len].copy_from_slice(row);
        rows.push(out);
    }
    rows
}

pub(crate) fn generate_m<F, const D: usize, Cfg: CommitmentConfig>(
    setup: &RingCommitmentSetup<F, D>,
    opening_point: &RingOpeningPoint<F, D>,
    challenges: &[CyclotomicRing<F, D>],
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError>
where
    F: FieldCore + CanonicalField,
{
    let num_blocks = 1usize
        .checked_shl(Cfg::R as u32)
        .ok_or_else(|| HachiError::InvalidSetup("2^R does not fit usize".to_string()))?;
    let block_len = 1usize
        .checked_shl(Cfg::M as u32)
        .ok_or_else(|| HachiError::InvalidSetup("2^M does not fit usize".to_string()))?;
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

    let g1 = gadget_row::<F, D>(Cfg::DELTA, Cfg::LOG_BASIS);
    let j1 = gadget_row::<F, D>(Cfg::TAU, Cfg::LOG_BASIS);

    let row3_w = kron_row(&opening_point.b, &g1);
    let row4_w = kron_row(challenges, &g1);
    let row4_z = kron_row(&kron_row(&opening_point.a, &g1), &j1)
        .into_iter()
        .map(|x| -x)
        .collect::<Vec<_>>();

    let g_na = gadget_block_diag::<F, D>(Cfg::N_A, &g1);
    let row5_mid = g_na
        .iter()
        .map(|row| kron_row(challenges, row))
        .collect::<Vec<_>>();
    let row5_right = setup
        .A
        .iter()
        .map(|row| kron_row(row, &j1).into_iter().map(|x| -x).collect())
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
