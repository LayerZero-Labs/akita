//! D-independent root polynomial inputs for the public PCS API.
//!
//! The low-level Hachi kernels remain const-generic over the root ring degree
//! `D`, but callers should not have to pick `D` themselves when the schedule
//! chooses it from public inputs. These types preserve the source polynomial in
//! a ring-agnostic form and materialize the typed root polynomial only after
//! the concrete commitment/proof context has selected the active root ring.

use crate::algebra::fields::wide::HasWide;
use crate::error::HachiError;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::HachiCommitmentLayout;
use crate::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps, OneHotIndex, OneHotPoly};
use crate::protocol::proof::FlatDigitBlocks;
use crate::{CanonicalField, FieldCore};

/// Dense multilinear polynomial stored as raw field evaluations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMultilinear<F: FieldCore> {
    num_vars: usize,
    evals: Vec<F>,
}

impl<F: FieldCore> DenseMultilinear<F> {
    /// Build a dense multilinear polynomial from its field evaluations.
    ///
    /// # Errors
    ///
    /// Returns an error if `evals.len() != 2^num_vars`.
    pub fn from_field_evals(num_vars: usize, evals: &[F]) -> Result<Self, HachiError> {
        let expected_len = 1usize
            .checked_shl(num_vars as u32)
            .ok_or_else(|| HachiError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if evals.len() != expected_len {
            return Err(HachiError::InvalidSize {
                expected: expected_len,
                actual: evals.len(),
            });
        }
        Ok(Self {
            num_vars,
            evals: evals.to_vec(),
        })
    }

    /// Number of multilinear variables.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Borrow the raw field evaluations.
    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    /// Materialize the typed dense root polynomial for ring degree `D`.
    ///
    /// # Errors
    ///
    /// Returns an error if the typed ring packing is invalid for `D`.
    pub fn to_typed<const D: usize>(&self) -> Result<DensePoly<F, D>, HachiError>
    where
        F: CanonicalField,
    {
        DensePoly::from_field_evals(self.num_vars, &self.evals)
    }
}

/// One-hot multilinear polynomial stored independently of the root ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OneHotMultilinear {
    num_vars: usize,
    onehot_k: usize,
    indices: Vec<Option<usize>>,
}

impl OneHotMultilinear {
    /// Build a one-hot multilinear polynomial from chunk-local hot positions.
    ///
    /// # Errors
    ///
    /// Returns an error if `onehot_k == 0`, `onehot_k` does not divide the
    /// polynomial size, or `indices.len()` does not match the expected chunk
    /// count.
    pub fn new<I: OneHotIndex>(
        num_vars: usize,
        onehot_k: usize,
        indices: Vec<Option<I>>,
    ) -> Result<Self, HachiError> {
        if onehot_k == 0 {
            return Err(HachiError::InvalidInput(
                "onehot_k must be nonzero".to_string(),
            ));
        }
        let total_evals = 1usize
            .checked_shl(num_vars as u32)
            .ok_or_else(|| HachiError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if total_evals % onehot_k != 0 {
            return Err(HachiError::InvalidInput(format!(
                "onehot_k={onehot_k} must divide 2^{num_vars}"
            )));
        }
        let expected_chunks = total_evals / onehot_k;
        if indices.len() != expected_chunks {
            return Err(HachiError::InvalidSize {
                expected: expected_chunks,
                actual: indices.len(),
            });
        }

        let indices = indices
            .into_iter()
            .map(|idx| idx.map(OneHotIndex::as_usize))
            .collect();
        Ok(Self {
            num_vars,
            onehot_k,
            indices,
        })
    }

    /// Number of multilinear variables.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Chunk size of the one-hot encoding.
    pub fn onehot_k(&self) -> usize {
        self.onehot_k
    }

    /// Borrow the chunk-local hot positions.
    pub fn indices(&self) -> &[Option<usize>] {
        &self.indices
    }

    /// Materialize the typed one-hot root polynomial for ring degree `D`.
    ///
    /// `layout` must be the root layout selected for this polynomial.
    ///
    /// # Errors
    ///
    /// Returns an error if `layout` does not match `num_vars` or if the typed
    /// one-hot packing is invalid for `D`.
    pub fn to_typed<F: FieldCore, const D: usize>(
        &self,
        layout: HachiCommitmentLayout,
    ) -> Result<OneHotPoly<F, D, usize>, HachiError> {
        let expected_num_vars = layout.required_num_vars::<D>()?;
        if expected_num_vars != self.num_vars {
            return Err(HachiError::InvalidInput(format!(
                "layout expects {expected_num_vars} variables but onehot polynomial has {}",
                self.num_vars
            )));
        }
        OneHotPoly::<F, D, usize>::new(
            self.onehot_k,
            self.indices.clone(),
            layout.r_vars,
            layout.m_vars,
        )
    }
}

/// Public root polynomial wrapper accepted by the dynamic PCS API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultilinearPolynomial<F: FieldCore> {
    /// Dense field-evaluation polynomial.
    Dense(DenseMultilinear<F>),
    /// Sparse one-hot polynomial.
    OneHot(OneHotMultilinear),
}

impl<F: FieldCore> MultilinearPolynomial<F> {
    /// Number of multilinear variables.
    pub fn num_vars(&self) -> usize {
        match self {
            Self::Dense(poly) => poly.num_vars(),
            Self::OneHot(poly) => poly.num_vars(),
        }
    }
}

impl<F: FieldCore> From<DenseMultilinear<F>> for MultilinearPolynomial<F> {
    fn from(poly: DenseMultilinear<F>) -> Self {
        Self::Dense(poly)
    }
}

impl<F: FieldCore> From<OneHotMultilinear> for MultilinearPolynomial<F> {
    fn from(poly: OneHotMultilinear) -> Self {
        Self::OneHot(poly)
    }
}

/// Typed root polynomial materialized after the active root ring is known.
#[derive(Clone)]
pub(crate) enum TypedRootPolynomial<F: FieldCore, const D: usize> {
    Dense(DensePoly<F, D>),
    OneHot(OneHotPoly<F, D, usize>),
}

impl<F: FieldCore, const D: usize> TypedRootPolynomial<F, D> {
    pub(crate) fn from_public(
        poly: &MultilinearPolynomial<F>,
        root_layout: HachiCommitmentLayout,
    ) -> Result<Self, HachiError>
    where
        F: CanonicalField,
    {
        match poly {
            MultilinearPolynomial::Dense(poly) => Ok(Self::Dense(poly.to_typed::<D>()?)),
            MultilinearPolynomial::OneHot(poly) => {
                Ok(Self::OneHot(poly.to_typed::<F, D>(root_layout)?))
            }
        }
    }
}

impl<F, const D: usize> HachiPolyOps<F, D> for TypedRootPolynomial<F, D>
where
    F: FieldCore + CanonicalField + HasWide,
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

    fn evaluate_ring(&self, scalars: &[F]) -> crate::algebra::CyclotomicRing<F, D> {
        match self {
            Self::Dense(poly) => poly.evaluate_ring(scalars),
            Self::OneHot(poly) => poly.evaluate_ring(scalars),
        }
    }

    fn fold_blocks(
        &self,
        scalars: &[F],
        block_len: usize,
    ) -> Vec<crate::algebra::CyclotomicRing<F, D>> {
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
    ) -> (
        crate::algebra::CyclotomicRing<F, D>,
        Vec<crate::algebra::CyclotomicRing<F, D>>,
    ) {
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
        challenges: &[crate::algebra::ring::sparse_challenge::SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> crate::protocol::hachi_poly_ops::DecomposeFoldWitness<F, D> {
        match self {
            Self::Dense(poly) => poly.decompose_fold(challenges, block_len, num_digits, log_basis),
            Self::OneHot(poly) => poly.decompose_fold(challenges, block_len, num_digits, log_basis),
        }
    }

    fn commit_inner(
        &self,
        a_matrix: &crate::protocol::commitment::utils::flat_matrix::FlatMatrix<F>,
        ntt_a: &crate::protocol::commitment::utils::crt_ntt::NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<FlatDigitBlocks<D>, HachiError> {
        match self {
            Self::Dense(poly) => poly.commit_inner(
                a_matrix,
                ntt_a,
                block_len,
                num_digits_commit,
                num_digits_open,
                log_basis,
            ),
            Self::OneHot(poly) => poly.commit_inner(
                a_matrix,
                ntt_a,
                block_len,
                num_digits_commit,
                num_digits_open,
                log_basis,
            ),
        }
    }

    fn commit_inner_witness(
        &self,
        a_matrix: &crate::protocol::commitment::utils::flat_matrix::FlatMatrix<F>,
        ntt_a: &crate::protocol::commitment::utils::crt_ntt::NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<crate::protocol::hachi_poly_ops::CommitInnerWitness<F, D>, HachiError>
    where
        F: CanonicalField,
    {
        match self {
            Self::Dense(poly) => poly.commit_inner_witness(
                a_matrix,
                ntt_a,
                block_len,
                num_digits_commit,
                num_digits_open,
                log_basis,
            ),
            Self::OneHot(poly) => poly.commit_inner_witness(
                a_matrix,
                ntt_a,
                block_len,
                num_digits_commit,
                num_digits_open,
                log_basis,
            ),
        }
    }

    fn commit_inner_witness_batched(
        polys: &[&Self],
        a_matrix: &crate::protocol::commitment::utils::flat_matrix::FlatMatrix<F>,
        ntt_a: &crate::protocol::commitment::utils::crt_ntt::NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<Option<Vec<crate::protocol::hachi_poly_ops::CommitInnerWitness<F, D>>>, HachiError>
    where
        F: CanonicalField,
    {
        let Some(first) = polys.first() else {
            return Ok(Some(Vec::new()));
        };
        match **first {
            Self::Dense(_) => {
                let mut dense_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match **poly {
                        Self::Dense(ref inner) => dense_polys.push(inner),
                        Self::OneHot(_) => return Ok(None),
                    }
                }
                <DensePoly<F, D> as HachiPolyOps<F, D>>::commit_inner_witness_batched(
                    &dense_polys,
                    a_matrix,
                    ntt_a,
                    block_len,
                    num_digits_commit,
                    num_digits_open,
                    log_basis,
                )
            }
            Self::OneHot(_) => {
                let mut onehot_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match **poly {
                        Self::OneHot(ref inner) => onehot_polys.push(inner),
                        Self::Dense(_) => return Ok(None),
                    }
                }
                <OneHotPoly<F, D, usize> as HachiPolyOps<F, D>>::commit_inner_witness_batched(
                    &onehot_polys,
                    a_matrix,
                    ntt_a,
                    block_len,
                    num_digits_commit,
                    num_digits_open,
                    log_basis,
                )
            }
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
