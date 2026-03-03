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

use crate::algebra::ring::sparse_challenge::SparseChallenge;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::onehot::{
    inner_ajtai_onehot_t_only, map_onehot_to_sparse_blocks, SparseBlockEntry,
};
use crate::protocol::commitment::utils::crt_ntt::NttMatrixCache;
use crate::protocol::commitment::utils::linear::{
    decompose_block, decompose_rows, mat_vec_mul_ntt_cached, MatrixSlot,
};
use crate::{cfg_into_iter, cfg_iter, CanonicalField, FieldCore};

/// Operations the Hachi commitment scheme needs from a polynomial.
///
/// The four methods correspond to the four places in commit/prove that consume
/// polynomial data.  Implementations decide *how* to carry out each operation
/// (dense decompose + NTT, sparse monomial tricks, streaming, etc.).
pub trait HachiPolyOps<F: FieldCore, const D: usize>: Clone + Send + Sync {
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
    /// 1. Decompose: `sᵢ = G⁻¹(blockᵢ)` via `balanced_decompose_pow2(delta, log_basis)`.
    /// 2. Accumulate: `z += cᵢ ⊗ sᵢ` (sparse challenge multiplication).
    ///
    /// Returns `z` of length `block_len · delta`.
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        delta: usize,
        log_basis: u32,
    ) -> Vec<CyclotomicRing<F, D>>;

    /// **Op 4 — commit: per-block inner Ajtai.**
    ///
    /// For each block of `block_len` ring elements:
    /// 1. `sᵢ = G⁻¹(blockᵢ)` (balanced decomposition).
    /// 2. `tᵢ = A · sᵢ` (matrix-vector multiply).
    /// 3. `t̂ᵢ = G⁻¹(tᵢ)` (decompose rows).
    ///
    /// Returns one `t̂ᵢ` vector per block.
    ///
    /// # Errors
    ///
    /// Returns an error if the NTT-cached matrix-vector multiply fails.
    fn commit_inner(
        &self,
        a_matrix: &[Vec<CyclotomicRing<F, D>>],
        cache: &NttMatrixCache<D>,
        block_len: usize,
        delta: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError>;
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
        delta: usize,
        log_basis: u32,
    ) -> Vec<CyclotomicRing<F, D>> {
        let inner_width = block_len * delta;
        let mut z = vec![CyclotomicRing::<F, D>::zero(); inner_width];
        let n = self.coeffs.len();

        for (i, c_i) in challenges.iter().enumerate() {
            let start = i * block_len;
            let end = (start + block_len).min(n);
            let block = if start < n {
                &self.coeffs[start..end]
            } else {
                &[] as &[CyclotomicRing<F, D>]
            };
            let s_i = decompose_block(block, delta, log_basis);
            for (j, z_j) in z.iter_mut().enumerate() {
                *z_j += s_i[j].mul_by_sparse(c_i);
            }
        }

        z
    }

    fn commit_inner(
        &self,
        _a_matrix: &[Vec<CyclotomicRing<F, D>>],
        cache: &NttMatrixCache<D>,
        block_len: usize,
        delta: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);
        let n_a = _a_matrix.len();
        let zero_t_hat = vec![CyclotomicRing::<F, D>::zero(); n_a.checked_mul(delta).unwrap()];

        let results: Vec<Result<Vec<CyclotomicRing<F, D>>, HachiError>> =
            cfg_into_iter!(0..num_blocks)
                .map(|i| {
                    let start = i * block_len;
                    if start >= n {
                        return Ok(zero_t_hat.clone());
                    }
                    let end = (start + block_len).min(n);
                    let block = &self.coeffs[start..end];
                    let s_i = decompose_block(block, delta, log_basis);
                    let t_i = mat_vec_mul_ntt_cached(cache, MatrixSlot::A, &s_i)?;
                    Ok(decompose_rows(&t_i, delta, log_basis))
                })
                .collect();

        results.into_iter().collect()
    }
}

// ---------------------------------------------------------------------------
// OneHotPoly
// ---------------------------------------------------------------------------

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
    _marker: std::marker::PhantomData<F>,
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
            _marker: std::marker::PhantomData,
        })
    }

    fn total_ring_elems(&self) -> usize {
        let total_field = self.indices.len() * self.onehot_k;
        total_field / D
    }

    /// Materialize one block of ring elements (for operations that need dense data).
    fn materialize_block(&self, block_idx: usize) -> Vec<CyclotomicRing<F, D>> {
        let block_len = 1usize << self.m_vars;
        let mut block = vec![CyclotomicRing::<F, D>::zero(); block_len];

        for entry in &self.sparse_blocks[block_idx] {
            let mut coeffs = [F::zero(); D];
            for &ci in &entry.nonzero_coeffs {
                coeffs[ci] = F::one();
            }
            block[entry.pos_in_block] = CyclotomicRing::from_coefficients(coeffs);
        }

        block
    }
}

impl<F, const D: usize> HachiPolyOps<F, D> for OneHotPoly<F, D>
where
    F: FieldCore + CanonicalField,
{
    fn num_ring_elems(&self) -> usize {
        self.total_ring_elems()
    }

    fn evaluate_ring(&self, scalars: &[F]) -> CyclotomicRing<F, D> {
        let block_len = 1usize << self.m_vars;
        let mut acc = CyclotomicRing::<F, D>::zero();

        for (block_idx, entries) in self.sparse_blocks.iter().enumerate() {
            let block_offset = block_idx * block_len;
            for entry in entries {
                let ring_idx = block_offset + entry.pos_in_block;
                if ring_idx < scalars.len() {
                    let mut ring_elem = CyclotomicRing::<F, D>::zero();
                    for &ci in &entry.nonzero_coeffs {
                        ring_elem.coeffs[ci] = F::one();
                    }
                    acc += ring_elem.scale(&scalars[ring_idx]);
                }
            }
        }

        acc
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let num_blocks = self.sparse_blocks.len();
        let mut results = vec![CyclotomicRing::<F, D>::zero(); num_blocks];

        for (block_idx, entries) in self.sparse_blocks.iter().enumerate() {
            for entry in entries {
                if entry.pos_in_block < scalars.len() && entry.pos_in_block < block_len {
                    let mut ring_elem = CyclotomicRing::<F, D>::zero();
                    for &ci in &entry.nonzero_coeffs {
                        ring_elem.coeffs[ci] = F::one();
                    }
                    results[block_idx] += ring_elem.scale(&scalars[entry.pos_in_block]);
                }
            }
        }

        results
    }

    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        delta: usize,
        log_basis: u32,
    ) -> Vec<CyclotomicRing<F, D>> {
        let inner_width = block_len * delta;
        let mut z = vec![CyclotomicRing::<F, D>::zero(); inner_width];

        for (i, c_i) in challenges.iter().enumerate() {
            if i >= self.sparse_blocks.len() {
                continue;
            }
            let block = self.materialize_block(i);
            let s_i = decompose_block(&block, delta, log_basis);
            for (j, z_j) in z.iter_mut().enumerate() {
                *z_j += s_i[j].mul_by_sparse(c_i);
            }
        }

        z
    }

    fn commit_inner(
        &self,
        a_matrix: &[Vec<CyclotomicRing<F, D>>],
        _cache: &NttMatrixCache<D>,
        block_len: usize,
        delta: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError> {
        let n_a = a_matrix.len();
        let zero_t_hat = vec![CyclotomicRing::<F, D>::zero(); n_a.checked_mul(delta).unwrap()];

        let t_hat_all: Vec<Vec<CyclotomicRing<F, D>>> = cfg_iter!(self.sparse_blocks)
            .map(|block_entries| {
                if block_entries.is_empty() {
                    zero_t_hat.clone()
                } else {
                    let t_i = inner_ajtai_onehot_t_only(a_matrix, block_entries, block_len, delta);
                    decompose_rows(&t_i, delta, log_basis)
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

        let cache = setup.ntt_cache().unwrap();
        let t_hat_poly = poly
            .commit_inner(
                &setup.expanded.A,
                cache,
                layout.block_len,
                layout.delta,
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

        let cache = setup.ntt_cache().unwrap();
        let t_hat_poly = poly
            .commit_inner(
                &setup.expanded.A,
                cache,
                layout.block_len,
                layout.delta,
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
