//! Operation-centric polynomial trait for the Hachi commitment scheme.
//!
//! [`HachiPolyOps`] exposes the four operations the Hachi commit/prove paths
//! need from a polynomial, rather than raw coefficient access.  Each
//! implementation handles every operation in its own optimal way:
//!
//! - [`DensePoly`] — standard dense algorithms (decompose + NTT matvec).
//! - [`OneHotPoly`] — sparse monomial tricks, avoids all inner ring
//!   multiplications.
//!
//! # Extensibility
//!
//! This trait is coupled to power-of-2 cyclotomic rings
//! ([`CyclotomicRing<F, D>`]).  When non-power-of-2 rings are added, the trait
//! signature will change.  Additional operation methods may be added as the
//! protocol evolves.

use crate::algebra::fields::wide::HasWide;
use crate::algebra::ring::sparse_challenge::SparseChallenge;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::onehot::{
    inner_ajtai_onehot_wide, map_onehot_to_sparse_blocks, SparseBlockEntry,
};
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::linear::{decompose_rows_i8, mat_vec_mul_ntt_tiled_i8};
use crate::{cfg_fold_reduce, cfg_into_iter, cfg_iter, CanonicalField, FieldCore};
use std::array::from_fn;
use std::marker::PhantomData;

/// Operations the Hachi commitment scheme needs from a polynomial.
///
/// The four methods correspond to the four places in commit/prove that consume
/// polynomial data.  Implementations decide *how* to carry out each operation
/// (dense decompose + NTT, sparse monomial tricks, streaming, etc.).
pub trait HachiPolyOps<F: FieldCore, const D: usize>: Clone + Send + Sync {
    /// Per-polynomial cache type for the A-matrix commit path.
    ///
    /// `DensePoly` uses `NttSlotCache<D>` (CRT+NTT of A for dense mat-vec).
    /// `OneHotPoly` uses `()` (one-hot commit bypasses NTT entirely).
    type CommitCache: Send + Sync;

    /// Total number of ring elements in the polynomial.
    fn num_ring_elems(&self) -> usize;

    /// **Op 1 — prove: ring-space evaluation.**
    ///
    /// Computes the global weighted sum `y = Σᵢ scalars[i] · self[i]`.
    ///
    /// `scalars` has length >= `num_ring_elems`; excess entries are ignored.
    fn evaluate_ring(&self, scalars: &[F]) -> CyclotomicRing<F, D>;

    /// **Op 2 — prove: per-block fold.**
    ///
    /// For each contiguous block of `block_len` ring elements, computes
    /// `Σⱼ scalars[j] · self[i·block_len + j]`.
    ///
    /// Returns one ring element per block (total `ceil(num_ring_elems / block_len)`).
    /// `scalars` has length `block_len`.
    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>>;

    /// **Op 3 — prove: decompose + challenge-fold.**
    ///
    /// For each block of `block_len` ring elements:
    /// 1. Decompose: `sᵢ = G⁻¹(blockᵢ)` via `balanced_decompose_pow2(num_digits, log_basis)`.
    /// 2. Accumulate: `z += cᵢ ⊗ sᵢ` (sparse challenge multiplication).
    ///
    /// Returns `z` of length `block_len · num_digits`.
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Vec<CyclotomicRing<F, D>>;

    /// **Op 4 — commit: per-block inner Ajtai.**
    ///
    /// For each block of `block_len` ring elements:
    /// 1. `sᵢ = G⁻¹(blockᵢ)` (balanced decomposition to i8 digits).
    /// 2. `tᵢ = A · sᵢ` (matrix-vector multiply via NTT cache or sparse path).
    /// 3. `t̂ᵢ = G⁻¹(tᵢ)` (decompose rows to i8 digits).
    ///
    /// Returns one `t̂ᵢ` vector per block as `[i8; D]` digit planes.
    ///
    /// # Errors
    ///
    /// Returns an error if the cached matrix-vector multiply fails.
    fn commit_inner(
        &self,
        a_matrix: &[Vec<CyclotomicRing<F, D>>],
        ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<[i8; D]>>, HachiError>;
}

/// Dense polynomial: all ring coefficients materialized in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePoly<F: FieldCore, const D: usize> {
    /// Ring coefficients in sequential block order.
    pub coeffs: Vec<CyclotomicRing<F, D>>,
}

impl<F: FieldCore, const D: usize> DensePoly<F, D> {
    /// Pack field-element evaluations into ring elements.
    ///
    /// The first `α = log₂(D)` variables become coefficient slots within each
    /// ring element; the remaining variables index ring elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `D` is not a power of two, `num_vars < log₂(D)`, or
    /// `evals.len() != 2^num_vars`.
    pub fn from_field_evals(num_vars: usize, evals: &[F]) -> Result<Self, HachiError> {
        if D == 0 || !D.is_power_of_two() {
            return Err(HachiError::InvalidInput(format!(
                "ring degree D={D} is not a power of two"
            )));
        }
        let alpha = D.trailing_zeros() as usize;
        if num_vars < alpha {
            return Err(HachiError::InvalidInput(format!(
                "num_vars {num_vars} is smaller than alpha {alpha}"
            )));
        }
        let expected_len = 1usize
            .checked_shl(num_vars as u32)
            .ok_or_else(|| HachiError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if evals.len() != expected_len {
            return Err(HachiError::InvalidSize {
                expected: expected_len,
                actual: evals.len(),
            });
        }

        let outer_len = expected_len / D;
        let coeffs: Vec<CyclotomicRing<F, D>> = (0..outer_len)
            .map(|i| CyclotomicRing::from_slice(&evals[i * D..(i + 1) * D]))
            .collect();
        Ok(Self { coeffs })
    }

    /// Wrap an existing vector of ring elements.
    pub fn from_ring_coeffs(coeffs: Vec<CyclotomicRing<F, D>>) -> Self {
        Self { coeffs }
    }
}

impl<F, const D: usize> HachiPolyOps<F, D> for DensePoly<F, D>
where
    F: FieldCore + CanonicalField,
{
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        self.coeffs.len()
    }

    fn evaluate_ring(&self, scalars: &[F]) -> CyclotomicRing<F, D> {
        #[cfg(feature = "parallel")]
        {
            self.coeffs
                .par_iter()
                .zip(scalars.par_iter())
                .fold(
                    || CyclotomicRing::<F, D>::zero(),
                    |acc, (f_i, w_i)| acc + f_i.scale(w_i),
                )
                .reduce(|| CyclotomicRing::<F, D>::zero(), |a, b| a + b)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.coeffs
                .iter()
                .zip(scalars.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, w_i)| {
                    acc + f_i.scale(w_i)
                })
        }
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);
        cfg_into_iter!(0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                let end = (start + block_len).min(n);
                let block = &self.coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    acc += b_j.scale(&a_j);
                }
                acc
            })
            .collect()
    }

    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Vec<CyclotomicRing<F, D>> {
        let inner_width = block_len * num_digits;
        let n = self.coeffs.len();
        let coeffs = &self.coeffs;

        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;

        let z_i32: Vec<[i32; D]> = cfg_fold_reduce!(
            0..challenges.len(),
            || vec![[0i32; D]; inner_width],
            |mut z: Vec<[i32; D]>, i| {
                let c_i = &challenges[i];
                let start = i * block_len;
                let end = (start + block_len).min(n);

                let half_b = 1i128 << (log_basis - 1);
                let b_val = half_b << 1;
                let mask = b_val - 1;

                for elem_idx in 0..(end.saturating_sub(start)) {
                    let ring = &coeffs[start + elem_idx];
                    let base_j = elem_idx * num_digits;

                    for coeff_idx in 0..D {
                        let canonical = ring.coeffs[coeff_idx].to_canonical_u128();
                        let mut c: i128 = if canonical > half_q {
                            -((q - canonical) as i128)
                        } else {
                            canonical as i128
                        };

                        for digit in 0..num_digits {
                            let d = c & mask;
                            let balanced = if d >= half_b { d - b_val } else { d };
                            c = (c - balanced) >> log_basis;

                            if balanced == 0 {
                                continue;
                            }
                            let digit_i32 = balanced as i32;

                            for (&pos, &challenge_coeff) in
                                c_i.positions.iter().zip(c_i.coeffs.iter())
                            {
                                let target = coeff_idx + pos as usize;
                                let (idx, sign) = if target < D {
                                    (target, 1i32)
                                } else {
                                    (target - D, -1i32)
                                };
                                z[base_j + digit][idx] += sign * digit_i32 * challenge_coeff as i32;
                            }
                        }
                    }
                }
                z
            },
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    for (a_coeff, b_coeff) in ai.iter_mut().zip(bi.iter()) {
                        *a_coeff += b_coeff;
                    }
                }
                a
            }
        );

        z_i32
            .into_iter()
            .map(|arr| {
                let field_coeffs: [F; D] = from_fn(|k| {
                    let v = arr[k];
                    if v >= 0 {
                        F::from_canonical_u128_reduced(v as u128)
                    } else {
                        F::from_canonical_u128_reduced(q - ((-v) as u128))
                    }
                });
                CyclotomicRing::from_coefficients(field_coeffs)
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "DensePoly::commit_inner")]
    fn commit_inner(
        &self,
        _a_matrix: &[Vec<CyclotomicRing<F, D>>],
        ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<[i8; D]>>, HachiError> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);

        let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                if start >= n {
                    &[] as &[CyclotomicRing<F, D>]
                } else {
                    &self.coeffs[start..(start + block_len).min(n)]
                }
            })
            .collect();

        let t_all = mat_vec_mul_ntt_tiled_i8(ntt_a, &block_slices, num_digits, log_basis);

        let results: Vec<Vec<[i8; D]>> = cfg_into_iter!(t_all)
            .map(|t_i| decompose_rows_i8(&t_i, num_digits, log_basis))
            .collect();

        Ok(results)
    }
}

/// One-hot polynomial: sparse witness with at most one nonzero field element
/// per chunk of size `onehot_k`.
///
/// Exploits sparsity in all four operations, avoiding inner ring
/// multiplications during commit and decomposing only nonzero monomials.
#[derive(Debug, Clone)]
pub struct OneHotPoly<F: FieldCore, const D: usize> {
    onehot_k: usize,
    indices: Vec<Option<usize>>,
    m_vars: usize,
    sparse_blocks: Vec<Vec<SparseBlockEntry>>,
    _marker: PhantomData<F>,
}

impl<F: FieldCore, const D: usize> OneHotPoly<F, D> {
    /// Build a one-hot polynomial from chunk size and hot-position indices.
    ///
    /// `indices[c]` is the hot position in chunk `c` (`None` for all-zero chunks).
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent or any index is out of range.
    pub fn new(
        onehot_k: usize,
        indices: Vec<Option<usize>>,
        r_vars: usize,
        m_vars: usize,
    ) -> Result<Self, HachiError> {
        let sparse_blocks = map_onehot_to_sparse_blocks(onehot_k, &indices, r_vars, m_vars, D)?;
        Ok(Self {
            onehot_k,
            indices,
            m_vars,
            sparse_blocks,
            _marker: PhantomData,
        })
    }

    fn total_ring_elems(&self) -> usize {
        let total_field = self.indices.len() * self.onehot_k;
        total_field / D
    }
}

impl<F, const D: usize> HachiPolyOps<F, D> for OneHotPoly<F, D>
where
    F: FieldCore + CanonicalField + HasWide,
{
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        self.total_ring_elems()
    }

    fn evaluate_ring(&self, scalars: &[F]) -> CyclotomicRing<F, D> {
        let block_len = 1usize << self.m_vars;
        let mut coeffs_acc = [F::zero(); D];

        for (block_idx, entries) in self.sparse_blocks.iter().enumerate() {
            let block_offset = block_idx * block_len;
            for entry in entries {
                let ring_idx = block_offset + entry.pos_in_block;
                if ring_idx < scalars.len() {
                    let s = scalars[ring_idx];
                    for &ci in &entry.nonzero_coeffs {
                        coeffs_acc[ci] += s;
                    }
                }
            }
        }

        CyclotomicRing::from_coefficients(coeffs_acc)
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let num_blocks = self.sparse_blocks.len();
        let mut results = vec![CyclotomicRing::<F, D>::zero(); num_blocks];

        for (block_idx, entries) in self.sparse_blocks.iter().enumerate() {
            let mut coeffs_acc = [F::zero(); D];
            for entry in entries {
                if entry.pos_in_block < scalars.len() && entry.pos_in_block < block_len {
                    let s = scalars[entry.pos_in_block];
                    for &ci in &entry.nonzero_coeffs {
                        coeffs_acc[ci] += s;
                    }
                }
            }
            results[block_idx] = CyclotomicRing::from_coefficients(coeffs_acc);
        }

        results
    }

    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> Vec<CyclotomicRing<F, D>> {
        let inner_width = block_len * num_digits;
        let mut z = vec![CyclotomicRing::<F, D>::zero(); inner_width];

        // One-hot coefficients are {0,1}: balanced_decompose_pow2 produces
        // nonzero output only in digit plane 0 (the value itself). So we skip
        // materialize_block + decompose_block entirely and accumulate only at
        // the first digit plane position for each sparse entry.
        for (i, c_i) in challenges.iter().enumerate() {
            if i >= self.sparse_blocks.len() {
                continue;
            }
            for entry in &self.sparse_blocks[i] {
                let j = entry.pos_in_block * num_digits;
                let mut one_hot_coeffs = [F::zero(); D];
                for &ci in &entry.nonzero_coeffs {
                    one_hot_coeffs[ci] = F::one();
                }
                let one_hot = CyclotomicRing::from_coefficients(one_hot_coeffs);
                z[j] += one_hot.mul_by_sparse(c_i);
            }
        }

        z
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::commit_inner")]
    fn commit_inner(
        &self,
        a_matrix: &[Vec<CyclotomicRing<F, D>>],
        _ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<[i8; D]>>, HachiError> {
        let n_a = a_matrix.len();
        let zero_block_len = n_a.checked_mul(num_digits).unwrap();

        let t_hat_all: Vec<Vec<[i8; D]>> = cfg_iter!(self.sparse_blocks)
            .map(|block_entries| {
                if block_entries.is_empty() {
                    vec![[0i8; D]; zero_block_len]
                } else {
                    let t_i =
                        inner_ajtai_onehot_wide(a_matrix, block_entries, block_len, num_digits);
                    decompose_rows_i8(&t_i, num_digits, log_basis)
                }
            })
            .collect();

        Ok(t_hat_all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::commitment::{HachiCommitmentCore, RingCommitmentScheme};
    use crate::test_utils::{TinyConfig, D as TestD, F as TestF};
    use crate::FromSmallInt;

    #[test]
    fn dense_poly_from_field_evals_roundtrip() {
        let num_vars = 10;
        let len = 1usize << num_vars;
        let evals: Vec<TestF> = (0..len).map(|i| TestF::from_u64(i as u64)).collect();
        let poly = DensePoly::<TestF, TestD>::from_field_evals(num_vars, &evals).unwrap();
        assert_eq!(poly.num_ring_elems(), len / TestD);
    }

    #[test]
    fn dense_commit_inner_matches_ring_commit() {
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TestD, TinyConfig>>::setup(16)
                .unwrap();
        let layout = setup.layout();
        let num_ring = layout.num_blocks * layout.block_len;
        let evals: Vec<TestF> = (0..num_ring * TestD)
            .map(|i| TestF::from_u64(i as u64))
            .collect();

        let alpha = TestD.trailing_zeros() as usize;
        let num_vars = alpha + layout.m_vars + layout.r_vars;
        let poly = DensePoly::<TestF, TestD>::from_field_evals(num_vars, &evals).unwrap();

        let t_hat_poly = poly
            .commit_inner(
                &setup.expanded.A,
                &setup.ntt_A,
                layout.block_len,
                layout.num_digits_commit,
                layout.log_basis,
            )
            .unwrap();

        let w =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TestD, TinyConfig>>::commit_coeffs(
                &poly.coeffs,
                &setup,
            )
            .unwrap();

        assert_eq!(t_hat_poly, w.t_hat);
    }

    #[test]
    fn onehot_commit_inner_matches_ring_commit_onehot() {
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TestD, TinyConfig>>::setup(16)
                .unwrap();
        let layout = setup.layout();
        let total_ring = layout.num_blocks * layout.block_len;
        let onehot_k = TestD;
        let num_chunks = total_ring;
        let indices: Vec<Option<usize>> = (0..num_chunks).map(|i| Some(i % onehot_k)).collect();

        let poly = OneHotPoly::<TestF, TestD>::new(
            onehot_k,
            indices.clone(),
            layout.r_vars,
            layout.m_vars,
        )
        .unwrap();

        let t_hat_poly = poly
            .commit_inner(
                &setup.expanded.A,
                &setup.ntt_A,
                layout.block_len,
                layout.num_digits_commit,
                layout.log_basis,
            )
            .unwrap();

        let w =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TestD, TinyConfig>>::commit_onehot(
                onehot_k, &indices, &setup,
            )
            .unwrap();

        assert_eq!(t_hat_poly, w.t_hat);
    }
}
