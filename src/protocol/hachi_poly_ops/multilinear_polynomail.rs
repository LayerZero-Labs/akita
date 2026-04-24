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

use super::{
    CommitInnerWitness, DecomposeFoldWitness, DensePoly, HachiPolyOps, OneHotIndex, OneHotPoly,
};
use crate::algebra::fields::wide::HasWide;
use crate::algebra::ring::sparse_challenge::SparseChallenge;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
use crate::protocol::proof::FlatDigitBlocks;
use crate::{CanonicalField, FieldCore};

/// Borrowed multilinear-polynomial wrapper for dense and one-hot batches.
///
/// This erases the root representation (`DensePoly` vs `OneHotPoly`) while
/// preserving the operation-oriented `HachiPolyOps` interface that the
/// commitment scheme consumes.
#[derive(Debug, Clone, Copy)]
pub enum MultilinearPolynomail<'a, F: FieldCore, const D: usize, I: OneHotIndex = usize> {
    /// Dense multilinear polynomial.
    Dense(&'a DensePoly<F, D>),
    /// One-hot multilinear polynomial.
    OneHot(&'a OneHotPoly<F, D, I>),
}

impl<'a, F: FieldCore, const D: usize, I: OneHotIndex> MultilinearPolynomail<'a, F, D, I> {
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
    for MultilinearPolynomail<'a, F, D, I>
{
    fn from(poly: &'a DensePoly<F, D>) -> Self {
        Self::dense(poly)
    }
}

impl<'a, F: FieldCore, const D: usize, I: OneHotIndex> From<&'a OneHotPoly<F, D, I>>
    for MultilinearPolynomail<'a, F, D, I>
{
    fn from(poly: &'a OneHotPoly<F, D, I>) -> Self {
        Self::onehot(poly)
    }
}

impl<F, const D: usize, I> HachiPolyOps<F, D> for MultilinearPolynomail<'_, F, D, I>
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

    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        match self {
            Self::Dense(poly) => poly.decompose_fold(challenges, block_len, num_digits, log_basis),
            Self::OneHot(poly) => poly.decompose_fold(challenges, block_len, num_digits, log_basis),
        }
    }

    fn decompose_fold_batched(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F, D>> {
        match *polys.first()? {
            Self::Dense(_) => {
                let mut dense_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match **poly {
                        Self::Dense(inner) => dense_polys.push(inner),
                        Self::OneHot(_) => return None,
                    }
                }
                <DensePoly<F, D> as HachiPolyOps<F, D>>::decompose_fold_batched(
                    &dense_polys,
                    challenges,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
            Self::OneHot(_) => {
                let mut onehot_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match **poly {
                        Self::OneHot(inner) => onehot_polys.push(inner),
                        Self::Dense(_) => return None,
                    }
                }
                <OneHotPoly<F, D, I> as HachiPolyOps<F, D>>::decompose_fold_batched(
                    &onehot_polys,
                    challenges,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
        }
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
    ) -> Result<CommitInnerWitness<F, D>, HachiError>
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

    fn direct_root_witness(
        &self,
    ) -> Result<crate::protocol::proof::DirectWitnessProof<F>, HachiError> {
        match self {
            Self::Dense(poly) => poly.direct_root_witness(),
            Self::OneHot(poly) => poly.direct_root_witness(),
        }
    }
}
