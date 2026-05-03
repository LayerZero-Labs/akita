//! Operation-centric polynomial trait for the Hachi commitment scheme.
//!
//! [`HachiPolyOps`] exposes the operations the Hachi commit/prove paths need
//! from a caller-provided root polynomial, rather than raw coefficient access.
//! Two concrete implementations handle those root operations in their own
//! optimal way:
//!
//! - [`DensePoly`] — standard dense algorithms (decompose + NTT matvec).
//! - [`OneHotPoly`] — sparse monomial tricks, avoids all inner ring
//!   multiplications.
//! - [`MultilinearPolynomail`] — borrowed wrapper that lets one batch mix dense
//!   and one-hot multilinear polynomials under one shared scheme config/layout.
//!
//! Recursive levels do not use [`HachiPolyOps`]. They operate on
//! `RecursiveWitnessFlat` / `RecursiveWitnessView`, which model the
//! D-agnostic `w` witness produced by ring switching.
//!
//! # Module layout
//!
//! - `dense` — [`DensePoly`] and its [`HachiPolyOps`] impl.
//! - `multilinear_polynomail` — [`MultilinearPolynomail`], the canonical
//!   representation-erasing wrapper for mixed root batches.
//! - `onehot` — [`OneHotPoly`], [`OneHotIndex`], and column-sweep Ajtai
//!   commit helpers.
//! - `recursive_witness` — recursive `w` owner/view types and digit-native
//!   operations for later folding levels.
//! - `helpers` — shared internal helpers: decomposition, sparse
//!   multiply-accumulate, position-partitioned accumulation.
//! - `decompose_fold_neon` — AArch64 NEON kernel for the sparse-mul-acc
//!   hot loop (conditionally compiled).
//!
//! # Extensibility
//!
//! This trait is coupled to power-of-2 cyclotomic rings
//! ([`CyclotomicRing<F, D>`]).  When non-power-of-2 rings are added, the trait
//! signature will change.  Additional operation methods may be added as the
//! protocol evolves.

#[cfg(target_arch = "aarch64")]
mod decompose_fold_neon;
mod dense;
mod helpers;
mod multilinear_polynomail;
mod onehot;
mod recursive_witness;

pub use dense::DensePoly;
pub use multilinear_polynomail::MultilinearPolynomail;
#[cfg(test)]
pub(crate) use onehot::OneHotBlocks;
pub use onehot::{OneHotIndex, OneHotPoly};
pub(crate) use recursive_witness::{RecursiveWitnessFlat, RecursiveWitnessView};

use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::{CanonicalField, FieldCore};
use akita_algebra::ring::sparse_challenge::SparseChallenge;
use akita_algebra::CyclotomicRing;
use akita_field::HachiError;
use akita_types::FlatMatrix;
use akita_types::{DirectWitnessProof, FlatDigitBlocks};

/// Prover-side output of the decompose + challenge-fold step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecomposeFoldWitness<F: FieldCore, const D: usize> {
    /// Folded witness rows in ring form.
    pub z_pre: Vec<CyclotomicRing<F, D>>,
    /// Centered integer coefficients for each `z_pre` row.
    pub centered_coeffs: Vec<[i32; D]>,
    /// Infinity norm of `centered_coeffs`.
    pub centered_inf_norm: u32,
}

/// Prover-side output of the inner Ajtai commit step.
pub struct CommitInnerWitness<F: FieldCore, const D: usize> {
    /// Undecomposed `t_i = A * s_i` rows, grouped by block.
    pub t: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Decomposed `t_hat_i = G^{-1}(t_i)` rows in flat column-major order plus
    /// explicit block boundaries.
    pub t_hat: FlatDigitBlocks<D>,
}

fn recompose_commit_inner_blocks<F: CanonicalField, const D: usize>(
    t_hat_blocks: &FlatDigitBlocks<D>,
    num_digits_open: usize,
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError> {
    if num_digits_open == 0 {
        return Err(HachiError::InvalidSetup(
            "num_digits_open must be nonzero when recomposing commit witness".to_string(),
        ));
    }
    t_hat_blocks
        .iter_blocks()
        .map(|block| {
            if block.len() % num_digits_open != 0 {
                return Err(HachiError::InvalidSetup(format!(
                    "t_hat block has {} planes, expected a multiple of num_digits_open={num_digits_open}",
                    block.len()
                )));
            }
            Ok(block
                .chunks(num_digits_open)
                .map(|digits| CyclotomicRing::gadget_recompose_pow2_i8(digits, log_basis))
                .collect())
        })
        .collect()
}

/// Operations the Hachi commitment scheme needs from a polynomial.
///
/// Each method corresponds to a place in commit/prove that consumes polynomial
/// data.  Implementations decide *how* to carry out each operation (dense
/// decompose + NTT, sparse monomial tricks, digit-plane bypass, etc.). Most
/// heterogeneous callers should use [`MultilinearPolynomail`] and let it
/// implement this trait on their behalf.
#[allow(clippy::too_many_arguments)]
pub trait HachiPolyOps<F: FieldCore, const D: usize>: Clone + Send + Sync {
    /// Per-polynomial cache type for the A-matrix commit path.
    ///
    /// All current implementations use `NttSlotCache<D>`.
    type CommitCache: Send + Sync;

    /// Total number of ring elements in the polynomial.
    fn num_ring_elems(&self) -> usize;

    /// Total number of variables (field-element dimension).
    ///
    /// Derived from `num_ring_elems() * D`, which equals `2^num_vars`.
    fn num_vars(&self) -> usize {
        let total = self
            .num_ring_elems()
            .checked_mul(D)
            .expect("ring elems * D overflow");
        debug_assert!(
            total.is_power_of_two(),
            "total field elements must be a power of 2"
        );
        total.trailing_zeros() as usize
    }

    /// **Op 2 — prove: per-block fold.**
    ///
    /// For each contiguous block of `block_len` ring elements, computes
    /// `Σⱼ scalars[j] · self[i·block_len + j]`.
    ///
    /// Returns one ring element per block (total `ceil(num_ring_elems / block_len)`).
    /// `scalars` has length `block_len`.
    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>>;

    /// Fused fold + evaluation in a single pass over the polynomial.
    ///
    /// `eval_outer_scalars` is the per-block weight vector `b` (size `num_blocks`).
    /// `fold_scalars` is the per-element-in-block weight vector `a` (size `block_len`).
    ///
    /// The full evaluation scalars factor as `outer_weights[i*block_len + j] = b[i] * a[j]`,
    /// so `eval = Σ_i b[i] * fold(a)[i]` — derived from the fold result without
    /// materializing the full `2^(m_vars + r_vars)` weight vector.
    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let folded = self.fold_blocks(fold_scalars, block_len);
        let eval = folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + f_i.scale(s_i)
            });
        (eval, folded)
    }

    /// **Op 3 — prove: decompose + challenge-fold.**
    ///
    /// For each block of `block_len` ring elements:
    /// 1. Decompose: `sᵢ = G⁻¹(blockᵢ)` via `balanced_decompose_pow2(num_digits, log_basis)`.
    /// 2. Accumulate: `z += cᵢ ⊗ sᵢ` (sparse challenge multiplication).
    ///
    /// Returns the folded witness `z_pre` of length `block_len · num_digits`
    /// together with centered coefficient rows that later prover steps can reuse.
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> DecomposeFoldWitness<F, D>;

    /// Optional fused batched variant of [`decompose_fold`](Self::decompose_fold).
    ///
    /// Implementations can override this when many claims at one opening point
    /// admit a faster shared accumulation path. The default falls back to
    /// per-polynomial processing in the caller.
    fn decompose_fold_batched(
        _polys: &[&Self],
        _challenges: &[SparseChallenge],
        _block_len: usize,
        _num_digits: usize,
        _log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F, D>> {
        None
    }

    /// **Op 4 — commit: per-block inner Ajtai.**
    ///
    /// For each block of `block_len` ring elements:
    /// 1. `sᵢ = G⁻¹(blockᵢ)` with `num_digits_commit` levels.
    /// 2. `tᵢ = A · sᵢ` (matrix-vector multiply via NTT cache or sparse path).
    /// 3. `t̂ᵢ = G⁻¹(tᵢ)` with `num_digits_open` levels (t has full-field
    ///    coefficients regardless of s's digit count).
    ///
    /// Returns one `t̂ᵢ` vector per block as `[i8; D]` digit planes.
    ///
    /// # Errors
    ///
    /// Returns an error if the cached matrix-vector multiply fails.
    fn commit_inner(
        &self,
        a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<FlatDigitBlocks<D>, HachiError>;

    /// Like [`commit_inner`](Self::commit_inner), but also preserves the
    /// undecomposed `t_i` rows for prover-side consumers that would otherwise
    /// need to recompose `t_hat`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`commit_inner`](Self::commit_inner) fails or if the
    /// resulting `t_hat` blocks cannot be recomposed into full `t_i` rows.
    fn commit_inner_witness(
        &self,
        a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<CommitInnerWitness<F, D>, HachiError>
    where
        F: CanonicalField,
    {
        let t_hat = self.commit_inner(
            a_matrix,
            ntt_a,
            n_a,
            block_len,
            num_digits_commit,
            num_digits_open,
            log_basis,
            matrix_stride,
        )?;
        let t = recompose_commit_inner_blocks::<F, D>(&t_hat, num_digits_open, log_basis)?;
        Ok(CommitInnerWitness { t, t_hat })
    }

    /// Materialize a direct root witness for zero-fold openings.
    ///
    /// The returned witness must evaluate to the original root-opening claim
    /// under the usual public opening point. Recursive witnesses do not use
    /// this hook; it exists only so root proofs can choose a first-class
    /// direct step instead of forcing a degenerate fold.
    ///
    /// # Errors
    ///
    /// Returns an error when this root representation cannot produce a direct
    /// witness payload.
    fn direct_root_witness(&self) -> Result<DirectWitnessProof<F>, HachiError> {
        Err(HachiError::InvalidInput(
            "root-direct witness is not supported for this polynomial type".to_string(),
        ))
    }
}

impl<F, const D: usize, P> HachiPolyOps<F, D> for &P
where
    F: FieldCore,
    P: HachiPolyOps<F, D>,
{
    type CommitCache = P::CommitCache;

    fn num_ring_elems(&self) -> usize {
        <P as HachiPolyOps<F, D>>::num_ring_elems(*self)
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        <P as HachiPolyOps<F, D>>::fold_blocks(*self, scalars, block_len)
    }

    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        <P as HachiPolyOps<F, D>>::evaluate_and_fold(
            *self,
            eval_outer_scalars,
            fold_scalars,
            block_len,
        )
    }

    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        <P as HachiPolyOps<F, D>>::decompose_fold(
            *self, challenges, block_len, num_digits, log_basis,
        )
    }

    fn decompose_fold_batched(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F, D>> {
        let inner_refs: Vec<&P> = polys.iter().map(|poly| **poly).collect();
        P::decompose_fold_batched(&inner_refs, challenges, block_len, num_digits, log_basis)
    }

    fn commit_inner(
        &self,
        a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<FlatDigitBlocks<D>, HachiError> {
        <P as HachiPolyOps<F, D>>::commit_inner(
            *self,
            a_matrix,
            ntt_a,
            n_a,
            block_len,
            num_digits_commit,
            num_digits_open,
            log_basis,
            matrix_stride,
        )
    }

    fn commit_inner_witness(
        &self,
        a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<CommitInnerWitness<F, D>, HachiError>
    where
        F: CanonicalField,
    {
        <P as HachiPolyOps<F, D>>::commit_inner_witness(
            *self,
            a_matrix,
            ntt_a,
            n_a,
            block_len,
            num_digits_commit,
            num_digits_open,
            log_basis,
            matrix_stride,
        )
    }

    fn direct_root_witness(&self) -> Result<DirectWitnessProof<F>, HachiError> {
        <P as HachiPolyOps<F, D>>::direct_root_witness(*self)
    }
}
