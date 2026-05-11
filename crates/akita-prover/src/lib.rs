//! Prover-facing API surface for the Akita PCS.
//!
//! This crate owns prover-side polynomial backends, setup artifacts, recursive
//! witness construction, ring-switch handoff, and Akita-specific sumcheck
//! provers. Config and schedule policy live in `akita-config`.

pub mod api;
pub mod backend;
pub mod kernels;
pub mod protocol;

use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::{
    CommitmentGroupOccurrence, DirectWitnessProof, FlatDigitBlocks, FlatMatrix, OpeningPoints,
};

pub use api::{
    batched_commit_with_params, batched_commit_with_policy, commit_for_multipoint_with_policy,
    commit_with_params, commit_with_policy, prepare_batched_commit_inputs, prepare_commit_inputs,
    AkitaProverSetup, CommitmentProver, PreparedBatchedCommitInputs, PreparedCommitInputs,
};
pub use backend::{
    DensePoly, MultilinearPolynomial, OneHotIndex, OneHotPoly, RecursiveCommitmentHintCache,
    RecursiveWitnessFlat, RecursiveWitnessView,
};
pub use kernels::MultiDNttCaches;
pub use protocol::sumcheck::{AkitaStage1Prover, AkitaStage2Prover};
pub use protocol::QuadraticEquation;
pub use protocol::{
    build_final_proof_steps, build_folded_batched_proof_with_suffix, commit_next_w_with_policy,
    prepare_batched_prove_inputs, prove_batched_with_policy, prove_fold_level_from_quadratic,
    prove_folded_batched_with_policy, prove_recursive_fold_with_params,
    prove_recursive_level_with_policy, prove_recursive_suffix_with_policy, prove_root_direct,
    prove_root_fold_from_quadratic, prove_root_fold_with_params, resolve_final_log_basis,
    PreparedBatchedProveInputs, ProveLevelOutput, RecursiveProverState, RecursiveSuffixOutcome,
    RingSwitchOutput, RootLevelRawOutput,
};
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

/// One prover-owned committed group in a normalized opening incidence graph.
///
/// This is the prover-side counterpart to [`CommitmentGroupOccurrence`]: it keeps the
/// group commitment visible to the verifier while retaining the polynomial
/// slice and hint needed to produce each referenced opening.
#[derive(Debug, Clone)]
pub struct ProverCommitmentGroupOccurrence<'a, P, C, H> {
    /// Polynomials addressable by claim `poly_idx` values in this group.
    pub polynomials: &'a [P],
    /// Commitment for `polynomials`.
    pub commitment: &'a C,
    /// Prover-side hint for `commitment`.
    pub hint: H,
}

impl<'a, P, C, H> ProverCommitmentGroupOccurrence<'a, P, C, H> {
    /// Number of polynomials addressable by incidence claims for this group.
    pub fn poly_count(&self) -> usize {
        self.polynomials.len()
    }

    /// Verifier-visible group metadata for shared incidence validation.
    pub fn incidence_group(&self) -> CommitmentGroupOccurrence<'a, C> {
        CommitmentGroupOccurrence {
            commitment: self.commitment,
            poly_count: self.poly_count(),
        }
    }
}

impl<'a, P, C, H> From<CommittedPolynomials<'a, P, C, H>>
    for ProverCommitmentGroupOccurrence<'a, P, C, H>
{
    fn from(group: CommittedPolynomials<'a, P, C, H>) -> Self {
        Self {
            polynomials: group.polynomials,
            commitment: group.commitment,
            hint: group.hint,
        }
    }
}

/// Batched prover input grouped by opening point.
pub type ProverClaims<'a, F, P, C, H> =
    Vec<(OpeningPoints<'a, F>, Vec<CommittedPolynomials<'a, P, C, H>>)>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prover_incidence_group_exposes_verifier_metadata() {
        let polynomials = [1u64, 2, 3];
        let commitment = "commitment";
        let group = ProverCommitmentGroupOccurrence {
            polynomials: &polynomials,
            commitment: &commitment,
            hint: "hint",
        };

        let incidence_group = group.incidence_group();

        assert_eq!(group.poly_count(), 3);
        assert_eq!(incidence_group.commitment, &commitment);
        assert_eq!(incidence_group.poly_count, 3);
    }

    #[test]
    fn committed_polynomials_convert_to_prover_incidence_group() {
        let polynomials = [5u64, 6];
        let commitment = "commitment";
        let committed = CommittedPolynomials {
            polynomials: &polynomials,
            commitment: &commitment,
            hint: "hint",
        };

        let group = ProverCommitmentGroupOccurrence::from(committed);

        assert_eq!(group.polynomials, &polynomials);
        assert_eq!(group.commitment, &commitment);
        assert_eq!(group.hint, "hint");
    }
}

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
    /// Recombined inner `A * s_i` rows, grouped by block.
    pub recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Digit decompositions of `A * s_i` in flat column-major order plus
    /// explicit block boundaries.
    pub decomposed_inner_rows: FlatDigitBlocks<D>,
}

fn recompose_commit_inner_blocks<F: CanonicalField, const D: usize>(
    t_hat_blocks: &FlatDigitBlocks<D>,
    num_digits_open: usize,
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    if num_digits_open == 0 {
        return Err(AkitaError::InvalidSetup(
            "num_digits_open must be nonzero when recomposing commit witness".to_string(),
        ));
    }
    t_hat_blocks
        .iter_blocks()
        .map(|block| {
            if block.len() % num_digits_open != 0 {
                return Err(AkitaError::InvalidSetup(format!(
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
pub trait AkitaPolyOps<F: FieldCore, const D: usize>: Clone + Send + Sync {
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
    ) -> Result<FlatDigitBlocks<D>, AkitaError>;

    /// Inner Ajtai commit step that also preserves recomposed inner rows.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::commit_inner`] fails or if the resulting
    /// decomposed blocks cannot be recomposed into full inner rows.
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
    ) -> Result<CommitInnerWitness<F, D>, AkitaError>
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
        let recomposed_inner_rows =
            recompose_commit_inner_blocks::<F, D>(&t_hat, num_digits_open, log_basis)?;
        Ok(CommitInnerWitness {
            recomposed_inner_rows,
            decomposed_inner_rows: t_hat,
        })
    }

    /// Materialize a direct root witness for zero-fold openings.
    ///
    /// # Errors
    ///
    /// Returns an error when this root representation cannot produce a direct
    /// witness payload.
    fn direct_root_witness(&self) -> Result<DirectWitnessProof<F>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "root-direct witness is not supported for this polynomial type".to_string(),
        ))
    }
}

impl<F, const D: usize, P> AkitaPolyOps<F, D> for &P
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    type CommitCache = P::CommitCache;

    fn num_ring_elems(&self) -> usize {
        <P as AkitaPolyOps<F, D>>::num_ring_elems(*self)
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        <P as AkitaPolyOps<F, D>>::fold_blocks(*self, scalars, block_len)
    }

    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        <P as AkitaPolyOps<F, D>>::evaluate_and_fold(
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
        <P as AkitaPolyOps<F, D>>::decompose_fold(
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
    ) -> Result<FlatDigitBlocks<D>, AkitaError> {
        <P as AkitaPolyOps<F, D>>::commit_inner(
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
    ) -> Result<CommitInnerWitness<F, D>, AkitaError>
    where
        F: CanonicalField,
    {
        <P as AkitaPolyOps<F, D>>::commit_inner_witness(
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

    fn direct_root_witness(&self) -> Result<DirectWitnessProof<F>, AkitaError> {
        <P as AkitaPolyOps<F, D>>::direct_root_witness(*self)
    }
}
