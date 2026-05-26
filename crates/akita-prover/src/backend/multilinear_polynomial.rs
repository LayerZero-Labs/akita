//! Canonical multilinear-polynomial wrapper for dense and one-hot representations.
//!
//! This is the intended public wrapper for heterogeneous root batches. All
//! wrapped polynomials must still share the same commitment config and root
//! layout chosen by the caller, but one batch can contain both dense and
//! one-hot roots.
//!
//! Homogeneous batches still reuse the existing backend-specific batched fast
//! paths; truly mixed batches fall back to the caller's per-polynomial
//! aggregation path.

use akita_algebra::CyclotomicRing;
use akita_challenges::Challenges;
use akita_field::fields::wide::HasWide;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::FlatDigitBlocks;
use akita_types::FlatMatrix;

use crate::kernels::crt_ntt::NttSlotCache;
use crate::{
    AkitaPolyOps, CommitInnerWitness, DecomposeFoldWitness, DensePoly, OneHotIndex, OneHotPoly,
};

/// Borrowed multilinear-polynomial wrapper for dense and one-hot batches.
///
/// This erases the root representation (`DensePoly` vs `OneHotPoly`) while
/// preserving the operation-oriented `AkitaPolyOps` interface that the
/// commitment scheme consumes.
#[derive(Debug, Clone, Copy)]
pub enum MultilinearPolynomial<'a, F: FieldCore, const D: usize, I: OneHotIndex = usize> {
    /// Dense multilinear polynomial.
    Dense(&'a DensePoly<F, D>),
    /// One-hot multilinear polynomial.
    OneHot(&'a OneHotPoly<F, D, I>),
}

impl<'a, F: FieldCore, const D: usize, I: OneHotIndex> MultilinearPolynomial<'a, F, D, I> {
    /// Wrap a dense polynomial.
    pub fn dense(poly: &'a DensePoly<F, D>) -> Self {
        Self::Dense(poly)
    }

    /// Wrap a one-hot polynomial.
    pub fn onehot(poly: &'a OneHotPoly<F, D, I>) -> Self {
        Self::OneHot(poly)
    }
}

impl<'a, F: FieldCore, const D: usize, I: OneHotIndex> From<&'a DensePoly<F, D>>
    for MultilinearPolynomial<'a, F, D, I>
{
    fn from(poly: &'a DensePoly<F, D>) -> Self {
        Self::dense(poly)
    }
}

impl<'a, F: FieldCore, const D: usize, I: OneHotIndex> From<&'a OneHotPoly<F, D, I>>
    for MultilinearPolynomial<'a, F, D, I>
{
    fn from(poly: &'a OneHotPoly<F, D, I>) -> Self {
        Self::onehot(poly)
    }
}

impl<F, const D: usize, I> AkitaPolyOps<F, D> for MultilinearPolynomial<'_, F, D, I>
where
    F: FieldCore + CanonicalField + HasWide,
    I: OneHotIndex,
{
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        match self {
            Self::Dense(poly) => poly.num_ring_elems(),
            Self::OneHot(poly) => poly.num_ring_elems(),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::Dense(poly) => poly.num_vars(),
            Self::OneHot(poly) => poly.num_vars(),
        }
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        match self {
            Self::Dense(poly) => poly.fold_blocks(scalars, block_len),
            Self::OneHot(poly) => poly.fold_blocks(scalars, block_len),
        }
    }

    fn fold_blocks_ring(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        match self {
            Self::Dense(poly) => poly.fold_blocks_ring(scalars, block_len),
            Self::OneHot(poly) => poly.fold_blocks_ring(scalars, block_len),
        }
    }

    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        match self {
            Self::Dense(poly) => {
                poly.evaluate_and_fold(eval_outer_scalars, fold_scalars, block_len)
            }
            Self::OneHot(poly) => {
                poly.evaluate_and_fold(eval_outer_scalars, fold_scalars, block_len)
            }
        }
    }

    fn evaluate_and_fold_ring(
        &self,
        eval_outer_scalars: &[CyclotomicRing<F, D>],
        fold_scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        match self {
            Self::Dense(poly) => {
                poly.evaluate_and_fold_ring(eval_outer_scalars, fold_scalars, block_len)
            }
            Self::OneHot(poly) => {
                poly.evaluate_and_fold_ring(eval_outer_scalars, fold_scalars, block_len)
            }
        }
    }

    fn decompose_fold(
        polys: &[&Self],
        challenges: &Challenges,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<DecomposeFoldWitness<F, D>, AkitaError> {
        // If all polys are the same kind, delegate to the dedicated backend's
        // batched path. Otherwise fall back to per-poly: each backend handles
        // one claim's challenges via `Challenges::select_claims`, then we
        // aggregate the per-poly witnesses.
        let all_dense = polys.iter().all(|p| matches!(p, Self::Dense(_)));
        let all_onehot = polys.iter().all(|p| matches!(p, Self::OneHot(_)));
        if all_dense {
            let mut dense_polys: Vec<&DensePoly<F, D>> = Vec::with_capacity(polys.len());
            for poly in polys {
                if let Self::Dense(inner) = **poly {
                    dense_polys.push(inner);
                }
            }
            return <DensePoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold(
                &dense_polys,
                challenges,
                block_len,
                num_digits,
                log_basis,
            );
        }
        if all_onehot {
            let mut onehot_polys: Vec<&OneHotPoly<F, D, I>> = Vec::with_capacity(polys.len());
            for poly in polys {
                if let Self::OneHot(inner) = **poly {
                    onehot_polys.push(inner);
                }
            }
            return <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::decompose_fold(
                &onehot_polys,
                challenges,
                block_len,
                num_digits,
                log_basis,
            );
        }
        // Mixed Dense/OneHot: per-poly fallback. Each backend gets a single
        // claim's challenges sliced via `Challenges::select_claims`, runs its
        // own batched-of-one path, and we aggregate the resulting witnesses.
        let mut witnesses = Vec::with_capacity(polys.len());
        for (claim_idx, poly) in polys.iter().enumerate() {
            let claim_challenges = challenges.select_claims::<D>(&[claim_idx])?;
            let witness = match **poly {
                Self::Dense(inner) => <DensePoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold(
                    std::slice::from_ref(&inner),
                    &claim_challenges,
                    block_len,
                    num_digits,
                    log_basis,
                )?,
                Self::OneHot(inner) => <OneHotPoly<F, D, I> as AkitaPolyOps<F, D>>::decompose_fold(
                    std::slice::from_ref(&inner),
                    &claim_challenges,
                    block_len,
                    num_digits,
                    log_basis,
                )?,
            };
            witnesses.push(witness);
        }
        crate::protocol::quadratic_equation::aggregate_decompose_fold_witnesses(witnesses)
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
    ) -> Result<FlatDigitBlocks<D>, AkitaError> {
        match self {
            Self::Dense(poly) => poly.commit_inner(
                a_matrix,
                ntt_a,
                n_a,
                block_len,
                num_digits_commit,
                num_digits_open,
                log_basis,
                matrix_stride,
            ),
            Self::OneHot(poly) => poly.commit_inner(
                a_matrix,
                ntt_a,
                n_a,
                block_len,
                num_digits_commit,
                num_digits_open,
                log_basis,
                matrix_stride,
            ),
        }
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
    ) -> Result<CommitInnerWitness<F, D>, AkitaError>
    where
        F: CanonicalField,
    {
        match self {
            Self::Dense(poly) => poly.commit_inner_witness(
                a_matrix,
                ntt_a,
                n_a,
                block_len,
                num_digits_commit,
                num_digits_open,
                log_basis,
                matrix_stride,
            ),
            Self::OneHot(poly) => poly.commit_inner_witness(
                a_matrix,
                ntt_a,
                n_a,
                block_len,
                num_digits_commit,
                num_digits_open,
                log_basis,
                matrix_stride,
            ),
        }
    }

    fn direct_root_witness(&self) -> Result<akita_types::DirectWitnessProof<F>, AkitaError> {
        match self {
            Self::Dense(poly) => poly.direct_root_witness(),
            Self::OneHot(poly) => poly.direct_root_witness(),
        }
    }
}
