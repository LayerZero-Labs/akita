//! Prover-facing API surface for the Akita PCS.
//!
//! This crate owns prover-side polynomial backends, setup artifacts, recursive
//! witness construction, ring-switch handoff, and Akita-specific sumcheck
//! provers. Root still owns config/schedule policy during the crate cutover.

pub mod commitment;
pub mod crt_ntt;
#[cfg(target_arch = "aarch64")]
mod decompose_fold_neon;
mod dense;
pub mod dispatch;
pub mod flow;
pub mod linear;
pub mod matrix;
mod multilinear_polynomail;
pub mod ntt_cache;
mod onehot;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod poly_helpers;
pub mod prg;
pub mod quadratic_equation;
mod recursive_hint;
mod recursive_witness;
pub mod ring_switch;
mod scheme;
pub mod setup;
pub mod sumcheck;

use akita_algebra::ring::sparse_challenge::SparseChallenge;
use akita_algebra::CyclotomicRing;
use akita_field::{CanonicalField, FieldCore, HachiError};
use akita_types::{DirectWitnessProof, FlatDigitBlocks, FlatMatrix, OpeningPoints};

pub use commitment::{
    batched_commit_with_params, commit_with_params, prepare_batched_commit_inputs,
    prepare_commit_inputs, verify_root_direct_commitments_with_params, PreparedBatchedCommitInputs,
    PreparedCommitInputs,
};
pub use dense::DensePoly;
pub use flow::{
    build_final_proof_steps, build_folded_batched_proof_with_suffix, prepare_batched_prove_inputs,
    prove_fold_level_from_quadratic, prove_recursive_fold_with_params,
    prove_recursive_suffix_with_policy, prove_root_direct_from_claims,
    prove_root_direct_from_polys, prove_root_fold_from_quadratic, prove_root_fold_with_params,
    resolve_final_log_basis, PreparedBatchedProveInputs, ProveLevelOutput, RecursiveProverState,
    RecursiveSuffixOutcome, RootLevelRawOutput,
};
pub use multilinear_polynomail::MultilinearPolynomail;
pub use ntt_cache::MultiDNttCaches;
pub use onehot::{OneHotIndex, OneHotPoly};
pub use quadratic_equation::QuadraticEquation;
pub use recursive_hint::RecursiveCommitmentHintCache;
pub use recursive_witness::{RecursiveWitnessFlat, RecursiveWitnessView};
pub use ring_switch::RingSwitchOutput;
pub use scheme::CommitmentProver;
pub use setup::HachiProverSetup;
pub use sumcheck::{HachiStage1Prover, HachiStage2Prover};

/// One committed polynomial group opened at an opening point.
///
/// The `polynomials` slice is the exact group committed together by the prover
/// commitment API; `commitment` and `hint` are the corresponding outputs for
/// that group.
#[derive(Debug, Clone)]
pub struct CommittedPolynomials<'a, P, C, H> {
    /// Polynomials that were committed together as one group.
    pub polynomials: &'a [P],
    /// Commitment for `polynomials`.
    pub commitment: &'a C,
    /// Prover-side hint for `commitment`.
    pub hint: H,
}

/// Batched prover input grouped by opening point.
pub type ProverClaims<'a, F, P, C, H> =
    Vec<(OpeningPoints<'a, F>, Vec<CommittedPolynomials<'a, P, C, H>>)>;

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

/// Operations the Akita commitment scheme needs from a root polynomial.
///
/// Each method corresponds to a place in commit/prove that consumes polynomial
/// data. Implementations decide how to carry out each operation: dense
/// decomposition, sparse one-hot tricks, digit-plane bypasses, or other
/// backend-specific strategies.
#[allow(clippy::too_many_arguments)]
pub trait HachiPolyOps<F: FieldCore, const D: usize>: Clone + Send + Sync {
    /// Per-polynomial cache type for the A-matrix commit path.
    type CommitCache: Send + Sync;

    /// Total number of ring elements in the polynomial.
    fn num_ring_elems(&self) -> usize;

    /// Total number of variables (field-element dimension).
    ///
    /// Derived from `num_ring_elems() * D`, which equals `2^num_vars`.
    ///
    /// # Panics
    ///
    /// Panics if `num_ring_elems() * D` overflows `usize`.
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

    /// Prover per-block fold.
    ///
    /// For each contiguous block of `block_len` ring elements, computes
    /// `sum_j scalars[j] * self[i * block_len + j]`.
    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>>;

    /// Fused fold + evaluation in one pass over the polynomial.
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

    /// Prover decompose + challenge-fold step.
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> DecomposeFoldWitness<F, D>;

    /// Optional fused batched variant of [`Self::decompose_fold`].
    fn decompose_fold_batched(
        _polys: &[&Self],
        _challenges: &[SparseChallenge],
        _block_len: usize,
        _num_digits: usize,
        _log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F, D>> {
        None
    }

    /// Inner Ajtai commit step.
    ///
    /// # Errors
    ///
    /// Returns an error if the cached matrix-vector multiply fails.
    fn commit_inner(
        &self,
        a_matrix: &FlatMatrix<F>,
        ntt_a: &Self::CommitCache,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<FlatDigitBlocks<D>, HachiError>;

    /// Inner Ajtai commit step that also preserves undecomposed `t_i` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::commit_inner`] fails or if the resulting
    /// `t_hat` blocks cannot be recomposed into full `t_i` rows.
    fn commit_inner_witness(
        &self,
        a_matrix: &FlatMatrix<F>,
        ntt_a: &Self::CommitCache,
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
        ntt_a: &Self::CommitCache,
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
        ntt_a: &Self::CommitCache,
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
