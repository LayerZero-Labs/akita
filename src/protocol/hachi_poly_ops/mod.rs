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

use crate::algebra::ring::sparse_challenge::SparseChallenge;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
use crate::protocol::proof::{DirectWitnessProof, FlatDigitBlocks};
use crate::{CanonicalField, FieldCore};

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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_arch = "aarch64")]
    use crate::algebra::ntt::neon;
    use crate::protocol::commitment::{
        hachi_recursive_level_layout_from_params, CommitmentConfig, HachiScheduleInputs,
    };
    use crate::protocol::ring_switch::w_ring_element_count;
    use crate::protocol::setup::HachiProverSetup;
    use crate::test_utils::{TinyConfig, D as TestD, F as TestF};
    use crate::FromSmallInt;
    use std::array::from_fn;

    #[test]
    fn dense_poly_from_field_evals_roundtrip() {
        let num_vars = 10;
        let len = 1usize << num_vars;
        let evals: Vec<TestF> = (0..len).map(|i| TestF::from_u64(i as u64)).collect();
        let poly = DensePoly::<TestF, TestD>::from_field_evals(num_vars, &evals).unwrap();
        assert_eq!(poly.num_ring_elems(), len / TestD);
    }

    /// Committing the same one-hot witness through the sparse `OneHotPoly`
    /// path and through the dense `DensePoly` path (after materializing the
    /// one-hot vector) must produce byte-identical `t_hat` output. This is
    /// the real correctness cross-check: one side takes the sparse-optimised
    /// inner Ajtai, the other side the generic dense NTT matvec.
    #[test]
    fn onehot_commit_inner_matches_dense_commit_inner() {
        let setup = HachiProverSetup::<TestF, TestD>::new::<TinyConfig>(16, 1, 1).unwrap();
        let layout = TinyConfig::commitment_layout(setup.expanded.seed.max_num_vars).unwrap();
        let total_ring = layout.num_blocks * layout.block_len;
        let onehot_k = TestD;
        let num_chunks = total_ring;
        let indices: Vec<Option<usize>> = (0..num_chunks).map(|i| Some(i % onehot_k)).collect();

        let onehot_poly = OneHotPoly::<TestF, TestD>::new(onehot_k, indices.clone())
            .expect("regular onehot poly");

        let level_params = TinyConfig::level_params(HachiScheduleInputs {
            max_num_vars: setup.expanded.seed.max_num_vars,
            level: 0,
            current_w_len: layout.num_blocks * layout.block_len * TestD,
        });
        let onehot_t_hat = onehot_poly
            .commit_inner(
                &setup.expanded.shared_matrix,
                &setup.ntt_shared,
                level_params.a_key.row_len(),
                layout.block_len,
                layout.num_digits_commit,
                layout.num_digits_open,
                layout.log_basis,
                setup.expanded.seed.max_stride,
            )
            .unwrap();

        // Reference: materialize the one-hot witness as a dense ring vector
        // and commit via `DensePoly::commit_inner`.
        let alpha = TestD.trailing_zeros() as usize;
        let num_vars = alpha + layout.m_vars + layout.r_vars;
        let mut evals = vec![TestF::zero(); 1usize << num_vars];
        for (c, opt) in indices.iter().enumerate() {
            if let Some(idx) = opt {
                evals[c * onehot_k + idx] = TestF::from_u64(1);
            }
        }
        let dense_poly = DensePoly::<TestF, TestD>::from_field_evals(num_vars, &evals).unwrap();
        let dense_t_hat = dense_poly
            .commit_inner(
                &setup.expanded.shared_matrix,
                &setup.ntt_shared,
                level_params.a_key.row_len(),
                layout.block_len,
                layout.num_digits_commit,
                layout.num_digits_open,
                layout.log_basis,
                setup.expanded.seed.max_stride,
            )
            .unwrap();

        assert_eq!(onehot_t_hat, dense_t_hat);
    }

    #[test]
    fn onehot_decompose_fold_matches_dense_single_chunk_onehot() {
        let setup = HachiProverSetup::<TestF, TestD>::new::<TinyConfig>(16, 1, 1).unwrap();
        let layout = TinyConfig::commitment_layout(setup.expanded.seed.max_num_vars).unwrap();
        let total_ring = layout.num_blocks * layout.block_len;
        let onehot_k = TestD;
        let indices: Vec<Option<usize>> = (0..total_ring)
            .map(|i| (i % 11 != 0).then_some((i * 7 + 3) % onehot_k))
            .collect();

        let poly = OneHotPoly::<TestF, TestD>::new(onehot_k, indices.clone())
            .expect("regular onehot poly");

        let mut evals = vec![TestF::zero(); total_ring * onehot_k];
        for (chunk_idx, hot_idx) in indices.into_iter().enumerate() {
            if let Some(hot_idx) = hot_idx {
                evals[chunk_idx * onehot_k + hot_idx] = TestF::from_u64(1);
            }
        }

        let alpha = TestD.trailing_zeros() as usize;
        let num_vars = alpha + layout.m_vars + layout.r_vars;
        let dense = DensePoly::<TestF, TestD>::from_field_evals(num_vars, &evals).unwrap();
        let challenges: Vec<SparseChallenge> = (0..layout.num_blocks)
            .map(|i| SparseChallenge {
                positions: vec![
                    0u32,
                    ((i * 5 + 1) % TestD) as u32,
                    ((i * 9 + 2) % TestD) as u32,
                ],
                coeffs: vec![1, -1, 1],
            })
            .collect();

        let got = poly.decompose_fold(&challenges, layout.block_len, 1, layout.log_basis);
        let expected = dense.decompose_fold(&challenges, layout.block_len, 1, layout.log_basis);
        assert_eq!(got.z_pre, expected.z_pre);
        assert_eq!(got.centered_coeffs, expected.centered_coeffs);
        assert_eq!(got.centered_inf_norm, expected.centered_inf_norm);
    }

    #[test]
    fn recursive_witness_matches_dense_recursive_w_ops() {
        let log_basis = TinyConfig::decomposition().log_basis;
        let digits: Vec<i8> = (0..(3 * TestD)).map(|i| (i % 7) as i8 - 3).collect();
        let field_evals: Vec<TestF> = digits.iter().map(|&d| TestF::from_i64(d as i64)).collect();
        let total_coeffs = digits.len().next_power_of_two().max(TestD);
        let mut padded = field_evals.clone();
        padded.resize(total_coeffs, TestF::zero());

        let dense = DensePoly::<TestF, TestD>::from_field_evals(
            total_coeffs.trailing_zeros() as usize,
            &padded,
        )
        .unwrap();
        let witness = RecursiveWitnessFlat::from_i8_digits(digits.clone());
        let digit_view = witness.view::<TestF, TestD>().unwrap();

        assert_eq!(digit_view.num_ring_elems(), dense.num_ring_elems());

        let eval_scalars: Vec<TestF> = (0..digit_view.num_ring_elems())
            .map(|i| TestF::from_u64((i + 2) as u64))
            .collect();
        assert_eq!(
            digit_view.evaluate_ring(&eval_scalars),
            super::dense::test_helpers::evaluate_ring_dense(&dense, &eval_scalars)
        );

        let num_blocks = 2;
        let block_len = 2;
        let num_ring = digits.len() / TestD;
        let mut cm_digits = vec![0i8; num_blocks * block_len * TestD];
        for block_idx in 0..num_blocks {
            for col_idx in 0..block_len {
                let seq = block_idx + col_idx * num_blocks;
                if seq >= num_ring {
                    continue;
                }
                let dst = block_idx * block_len + col_idx;
                cm_digits[dst * TestD..(dst + 1) * TestD]
                    .copy_from_slice(&digits[seq * TestD..(seq + 1) * TestD]);
            }
        }
        let cm_field_evals: Vec<TestF> = cm_digits
            .iter()
            .map(|&d| TestF::from_i64(d as i64))
            .collect();
        let dense_cm = DensePoly::<TestF, TestD>::from_field_evals(
            (num_blocks * block_len * TestD).trailing_zeros() as usize,
            &cm_field_evals,
        )
        .unwrap();

        let fold_scalars: Vec<TestF> = (0..block_len)
            .map(|i| TestF::from_u64((i + 5) as u64))
            .collect();
        assert_eq!(
            digit_view.fold_blocks(&fold_scalars, block_len, num_blocks),
            dense_cm.fold_blocks(&fold_scalars, block_len)
        );

        let challenges: Vec<SparseChallenge> = (0..num_blocks)
            .map(|i| SparseChallenge {
                positions: vec![0u32, ((i + 3) % TestD) as u32],
                coeffs: vec![1, -1],
            })
            .collect();
        let got = digit_view.decompose_fold(&challenges, block_len, num_blocks, 1, log_basis);
        let expected = dense_cm.decompose_fold(&challenges, block_len, 1, log_basis);
        assert_eq!(got.z_pre, expected.z_pre);
        assert_eq!(got.centered_coeffs, expected.centered_coeffs);
        assert_eq!(got.centered_inf_norm, expected.centered_inf_norm);

        let setup = HachiProverSetup::<TestF, TestD>::new::<TinyConfig>(16, 1, 1).unwrap();
        let test_lp = TinyConfig::commitment_layout(setup.expanded.seed.max_num_vars).unwrap();
        let w_len = w_ring_element_count::<TestF>(&test_lp) * TestD;
        let w_layout =
            hachi_recursive_level_layout_from_params::<TinyConfig>(&test_lp, w_len).unwrap();
        let max_stride = setup.expanded.seed.max_stride;
        let digit_commit = digit_view
            .commit_inner(
                &setup.ntt_shared,
                test_lp.a_key.row_len(),
                block_len,
                num_blocks,
                w_layout.num_digits_commit,
                w_layout.num_digits_open,
                w_layout.log_basis,
                max_stride,
            )
            .unwrap();
        let dense_commit = dense_cm
            .commit_inner(
                &setup.expanded.shared_matrix,
                &setup.ntt_shared,
                test_lp.a_key.row_len(),
                block_len,
                w_layout.num_digits_commit,
                w_layout.num_digits_open,
                w_layout.log_basis,
                max_stride,
            )
            .unwrap();

        assert_eq!(digit_commit, dense_commit);

        let digit_witness = digit_view
            .commit_inner_witness(
                &setup.ntt_shared,
                test_lp.a_key.row_len(),
                block_len,
                num_blocks,
                w_layout.num_digits_commit,
                w_layout.num_digits_open,
                w_layout.log_basis,
                max_stride,
            )
            .unwrap();
        let dense_witness = dense_cm
            .commit_inner_witness(
                &setup.expanded.shared_matrix,
                &setup.ntt_shared,
                test_lp.a_key.row_len(),
                block_len,
                w_layout.num_digits_commit,
                w_layout.num_digits_open,
                w_layout.log_basis,
                max_stride,
            )
            .unwrap();

        assert_eq!(digit_witness.t_hat, dense_witness.t_hat);
        assert_eq!(digit_witness.t, dense_witness.t);
    }

    #[test]
    fn recursive_witness_decompose_fold_respects_mag2_challenges() {
        let digits: Vec<i8> = (0..TestD).map(|i| (i % 5) as i8 - 2).collect();
        let witness = RecursiveWitnessFlat::from_i8_digits(digits.clone());
        let poly = witness.view::<TestF, TestD>().unwrap();
        let challenge = SparseChallenge {
            positions: vec![0, 3, 11],
            coeffs: vec![2, -1, -2],
        };

        let got = poly.decompose_fold(std::slice::from_ref(&challenge), 1, 1, 1, 3);

        let ring = CyclotomicRing::<TestF, TestD>::from_coefficients(from_fn(|idx| {
            TestF::from_i64(digits[idx] as i64)
        }));
        let expected = challenge.to_dense::<TestF, TestD>().unwrap() * ring;

        assert_eq!(got.z_pre, vec![expected]);
    }

    #[test]
    fn recursive_witness_view_rejects_non_divisible_digit_length() {
        let witness = RecursiveWitnessFlat::from_i8_digits(vec![1, -1, 2]);
        let err = witness.view::<TestF, TestD>().unwrap_err();
        match err {
            HachiError::InvalidSize { expected, actual } => {
                assert_eq!(expected, TestD);
                assert_eq!(actual, 3);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sparse_mul_acc_neon_matches_scalar_for_mag2_challenges() {
        if !neon::use_neon_ntt() {
            return;
        }

        let digit_plane: [i8; TestD] = from_fn(|idx| ((idx % 7) as i8) - 3);
        let challenge = SparseChallenge {
            positions: vec![0, 5, 17, 31],
            coeffs: vec![2, -1, -2, 1],
        };

        let mut scalar = [0i32; TestD];
        helpers::sparse_mul_acc_scalar::<TestD>(&digit_plane, &challenge, &mut scalar);

        let mut via_neon = [0i32; TestD];
        unsafe {
            decompose_fold_neon::sparse_mul_acc_neon(
                digit_plane.as_ptr(),
                via_neon.as_mut_ptr(),
                TestD,
                &challenge.positions,
                &challenge.coeffs,
            );
        }

        assert_eq!(via_neon, scalar);
    }
}
