//! Tensor extension-opening packing helpers.

use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, CanonicalField, FromPrimitiveInt, MulBaseUnreduced};
use akita_field::{AkitaError, ExtField, FieldCore};
use akita_types::{
    pack_tensor_base_lift_i8_digits, CleartextWitnessProof, FlatDigitBlocks, FpExtEncoding,
};
use std::sync::Arc;

use super::dense::{
    DenseCommitView, DenseOpeningBatchView, DenseOpeningView, DenseTensorView,
};
use super::sparse_ring::{
    SparseRingCommitView, SparseRingOpeningBatchView, SparseRingOpeningView, SparseRingTensorView,
};
use crate::compute::{
    CommitInnerPlan, CommitmentComputeBackend, CpuBackend, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, DirectRootWitnessSource, OpeningBatchKernel, OpeningFoldKernel,
    OpeningFoldOutput, OpeningFoldPlan, RootCommitKernel, RootCommitSource, RootOpeningSource,
    RootPolyShape, RootTensorSource, TensorPackedWitness, TensorProjectionBatchKernel,
    TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::{
    AkitaPolyOps, CommitInnerWitness, DecomposeFoldWitness, DensePoly, RecursiveWitnessFlat,
    SparseRingPoly,
};

/// Root polynomial obtained by tensor-projecting base-field evaluations into
/// an extension-valued table.
///
/// Dense roots use the ordinary dense backend. Sparse one-hot roots use signed
/// ring coefficients so the transformed commitment path preserves sparsity.
#[derive(Debug, Clone)]
pub enum RootTensorProjectionPoly<F: FieldCore, const D: usize> {
    /// Dense transformed root polynomial.
    Dense(DensePoly<F, D>),
    /// Sparse signed-ring transformed root polynomial.
    Sparse(Arc<SparseRingPoly<F, D>>),
}

impl<F: FieldCore, const D: usize> From<DensePoly<F, D>> for RootTensorProjectionPoly<F, D> {
    fn from(poly: DensePoly<F, D>) -> Self {
        Self::Dense(poly)
    }
}

impl<F: FieldCore, const D: usize> From<SparseRingPoly<F, D>> for RootTensorProjectionPoly<F, D> {
    fn from(poly: SparseRingPoly<F, D>) -> Self {
        Self::Sparse(Arc::new(poly))
    }
}

impl<F: FieldCore, const D: usize> From<Arc<SparseRingPoly<F, D>>>
    for RootTensorProjectionPoly<F, D>
{
    fn from(poly: Arc<SparseRingPoly<F, D>>) -> Self {
        Self::Sparse(poly)
    }
}

// ===========================================================================
// PO-CUTOVER (Phase A, additive): source-typed views + CpuBackend kernels for
// `RootTensorProjectionPoly`, dispatching to the inner dense/sparse kernels.
// ===========================================================================

/// Borrowed view over a committed tensor-projected root polynomial.
#[derive(Debug, Clone, Copy)]
pub struct RootTensorProjectionView<'a, F: FieldCore, const D: usize> {
    poly: &'a RootTensorProjectionPoly<F, D>,
}

/// Same-point batch view over several tensor-projected root polynomials.
#[derive(Debug, Clone, Copy)]
pub struct RootTensorProjectionBatchView<'a, F: FieldCore, const D: usize> {
    polys: &'a [&'a RootTensorProjectionPoly<F, D>],
}

impl<F, const D: usize> RootPolyShape<F, D> for RootTensorProjectionPoly<F, D>
where
    F: FieldCore,
{
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::Dense(poly) => RootPolyShape::num_ring_elems(poly),
            Self::Sparse(poly) => RootPolyShape::num_ring_elems(poly.as_ref()),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::Dense(poly) => RootPolyShape::num_vars(poly),
            Self::Sparse(poly) => RootPolyShape::num_vars(poly.as_ref()),
        }
    }
}

impl<F, const D: usize> RootCommitSource<F, D> for RootTensorProjectionPoly<F, D>
where
    F: FieldCore,
{
    type CommitView<'a>
        = RootTensorProjectionView<'a, F, D>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        Ok(RootTensorProjectionView { poly: self })
    }
}

impl<F, const D: usize> RootOpeningSource<F, D> for RootTensorProjectionPoly<F, D>
where
    F: FieldCore,
{
    type OpeningView<'a>
        = RootTensorProjectionView<'a, F, D>
    where
        Self: 'a;

    type OpeningBatchView<'a>
        = RootTensorProjectionBatchView<'a, F, D>
    where
        Self: 'a;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        Ok(RootTensorProjectionView { poly: self })
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        Ok(RootTensorProjectionBatchView { polys })
    }
}

impl<F, const D: usize> RootTensorSource<F, D> for RootTensorProjectionPoly<F, D>
where
    F: FieldCore,
{
    type TensorView<'a>
        = RootTensorProjectionView<'a, F, D>
    where
        Self: 'a;

    type TensorBatchView<'a>
        = RootTensorProjectionBatchView<'a, F, D>
    where
        Self: 'a;

    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError> {
        Ok(RootTensorProjectionView { poly: self })
    }

    fn tensor_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::TensorBatchView<'a>, AkitaError> {
        Ok(RootTensorProjectionBatchView { polys })
    }
}

impl<F, const D: usize> DirectRootWitnessSource<F, D> for RootTensorProjectionPoly<F, D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        match self {
            Self::Dense(poly) => DirectRootWitnessSource::direct_root_witness(poly),
            Self::Sparse(poly) => DirectRootWitnessSource::direct_root_witness(poly.as_ref()),
        }
    }
}

impl<F, const D: usize> RootCommitKernel<RootTensorProjectionView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: RootTensorProjectionView<'_, F, D>,
        plan: CommitInnerPlan,
    ) -> Result<FlatDigitBlocks<D>, AkitaError> {
        match source.poly {
            RootTensorProjectionPoly::Dense(poly) => RootCommitKernel::<
                DenseCommitView<'_, F, D>,
                F,
                D,
            >::commit_inner(
                self, prepared, poly.commit_view()?, plan
            ),
            RootTensorProjectionPoly::Sparse(poly) => {
                RootCommitKernel::<SparseRingCommitView<'_, F, D>, F, D>::commit_inner(
                    self,
                    prepared,
                    poly.as_ref().commit_view()?,
                    plan,
                )
            }
        }
    }

    fn commit_inner_witness(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: RootTensorProjectionView<'_, F, D>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError> {
        match source.poly {
            RootTensorProjectionPoly::Dense(poly) => RootCommitKernel::<
                DenseCommitView<'_, F, D>,
                F,
                D,
            >::commit_inner_witness(
                self, prepared, poly.commit_view()?, plan
            ),
            RootTensorProjectionPoly::Sparse(poly) => {
                RootCommitKernel::<SparseRingCommitView<'_, F, D>, F, D>::commit_inner_witness(
                    self,
                    prepared,
                    poly.as_ref().commit_view()?,
                    plan,
                )
            }
        }
    }
}

impl<F, const D: usize> OpeningFoldKernel<RootTensorProjectionView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn evaluate_and_fold(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F, D>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        match source.poly {
            RootTensorProjectionPoly::Dense(poly) => OpeningFoldKernel::<
                DenseOpeningView<'_, F, D>,
                F,
                D,
            >::evaluate_and_fold(
                self, prepared, poly.opening_view()?, plan
            ),
            RootTensorProjectionPoly::Sparse(poly) => {
                OpeningFoldKernel::<SparseRingOpeningView<'_, F, D>, F, D>::evaluate_and_fold(
                    self,
                    prepared,
                    poly.as_ref().opening_view()?,
                    plan,
                )
            }
        }
    }

    fn decompose_fold(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F, D>, AkitaError> {
        match source.poly {
            RootTensorProjectionPoly::Dense(poly) => OpeningFoldKernel::<
                DenseOpeningView<'_, F, D>,
                F,
                D,
            >::decompose_fold(
                self, prepared, poly.opening_view()?, plan
            ),
            RootTensorProjectionPoly::Sparse(poly) => {
                OpeningFoldKernel::<SparseRingOpeningView<'_, F, D>, F, D>::decompose_fold(
                    self,
                    prepared,
                    poly.as_ref().opening_view()?,
                    plan,
                )
            }
        }
    }
}

impl<F, const D: usize> OpeningBatchKernel<RootTensorProjectionBatchView<'_, F, D>, F, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn decompose_fold_batch(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionBatchView<'_, F, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        let Some(first) = source.polys.first() else {
            return Ok(None);
        };
        match *first {
            RootTensorProjectionPoly::Dense(_) => {
                let mut dense_polys = Vec::with_capacity(source.polys.len());
                for poly in source.polys {
                    match *poly {
                        RootTensorProjectionPoly::Dense(inner) => dense_polys.push(inner),
                        RootTensorProjectionPoly::Sparse(_) => return Ok(None),
                    }
                }
                let dense_view = DensePoly::<F, D>::opening_batch(&dense_polys)?;
                OpeningBatchKernel::<DenseOpeningBatchView<'_, F, D>, F, D>::decompose_fold_batch(
                    self, prepared, dense_view, plan,
                )
            }
            RootTensorProjectionPoly::Sparse(_) => {
                let mut sparse_polys = Vec::with_capacity(source.polys.len());
                for poly in source.polys {
                    match *poly {
                        RootTensorProjectionPoly::Sparse(inner) => {
                            sparse_polys.push(inner.as_ref())
                        }
                        RootTensorProjectionPoly::Dense(_) => return Ok(None),
                    }
                }
                let sparse_view = SparseRingPoly::<F, D>::opening_batch(&sparse_polys)?;
                OpeningBatchKernel::<SparseRingOpeningBatchView<'_, F, D>, F, D>::decompose_fold_batch(
                    self, prepared, sparse_view, plan,
                )
            }
        }
    }
}

impl<F, E, const D: usize> TensorProjectionKernel<RootTensorProjectionView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    E: ExtField<F>,
{
    fn column_partials(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        match source.poly {
            RootTensorProjectionPoly::Dense(poly) => {
                TensorProjectionKernel::<DenseTensorView<'_, F, D>, F, E, D>::column_partials(
                    self,
                    prepared,
                    poly.tensor_view()?,
                    logical_point,
                )
            }
            RootTensorProjectionPoly::Sparse(poly) => {
                TensorProjectionKernel::<SparseRingTensorView<'_, F, D>, F, E, D>::column_partials(
                    self,
                    prepared,
                    poly.as_ref().tensor_view()?,
                    logical_point,
                )
            }
        }
    }

    fn packed_witness(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionView<'_, F, D>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        match source.poly {
            RootTensorProjectionPoly::Dense(poly) => {
                TensorProjectionKernel::<DenseTensorView<'_, F, D>, F, E, D>::packed_witness(
                    self,
                    prepared,
                    poly.tensor_view()?,
                )
            }
            RootTensorProjectionPoly::Sparse(poly) => {
                TensorProjectionKernel::<SparseRingTensorView<'_, F, D>, F, E, D>::packed_witness(
                    self,
                    prepared,
                    poly.as_ref().tensor_view()?,
                )
            }
        }
    }

    fn root_projection(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionView<'_, F, D>,
    ) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        match source.poly {
            RootTensorProjectionPoly::Dense(poly) => {
                TensorProjectionKernel::<DenseTensorView<'_, F, D>, F, E, D>::root_projection(
                    self,
                    prepared,
                    poly.tensor_view()?,
                )
            }
            RootTensorProjectionPoly::Sparse(poly) => {
                TensorProjectionKernel::<SparseRingTensorView<'_, F, D>, F, E, D>::root_projection(
                    self,
                    prepared,
                    poly.as_ref().tensor_view()?,
                )
            }
        }
    }
}

impl<F, E, const D: usize>
    TensorProjectionBatchKernel<RootTensorProjectionBatchView<'_, F, D>, F, E, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    E: ExtField<F>,
{
    fn column_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionBatchView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        source
            .polys
            .iter()
            .map(|poly| {
                TensorProjectionKernel::<RootTensorProjectionView<'_, F, D>, F, E, D>::column_partials(
                    self,
                    prepared,
                    poly.tensor_view()?,
                    logical_point,
                )
            })
            .collect()
    }

    fn sparse_linear_combination(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionBatchView<'_, F, D>,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError> {
        if source.polys.len() != coeffs.len() {
            return Err(AkitaError::InvalidSize {
                expected: source.polys.len(),
                actual: coeffs.len(),
            });
        }
        let mut witnesses = Vec::with_capacity(source.polys.len());
        for poly in source.polys {
            let witness = match TensorProjectionKernel::<
                RootTensorProjectionView<'_, F, D>,
                F,
                E,
                D,
            >::packed_witness(self, prepared, poly.tensor_view()?)?
            {
                TensorPackedWitness::Sparse(witness) => witness,
                TensorPackedWitness::Dense(_) => return Ok(None),
            };
            witnesses.push(witness);
        }
        Ok(Some(SparseExtensionOpeningWitness::linear_combination(
            coeffs.iter().copied().zip(witnesses.iter()),
        )?))
    }
}

/// Fold-facing polynomial wrapper for original roots and tensor-projected roots.
///
/// Non-EOR paths borrow the caller's original polynomial. EOR paths own the
/// materialized tensor projection, preserving dense and sparse projected storage.
#[derive(Debug, Clone)]
pub enum FoldInputPoly<'a, F: FieldCore, P, const D: usize> {
    /// Original, non-projected polynomial.
    Original(&'a P),
    /// Dense tensor-projected root polynomial.
    ProjectedDense(DensePoly<F, D>),
    /// Sparse signed-ring tensor-projected root polynomial.
    ProjectedSparse(Arc<SparseRingPoly<F, D>>),
}

impl<'a, F: FieldCore, P, const D: usize> FoldInputPoly<'a, F, P, D> {
    pub fn projected_dense(poly: DensePoly<F, D>) -> Self {
        Self::ProjectedDense(poly)
    }

    pub fn projected_sparse(poly: SparseRingPoly<F, D>) -> Self {
        Self::ProjectedSparse(Arc::new(poly))
    }
}

macro_rules! dispatch_fold_input {
    ($self:expr, $poly:ident => $body:expr) => {
        match $self {
            FoldInputPoly::Original($poly) => $body,
            FoldInputPoly::ProjectedDense($poly) => $body,
            FoldInputPoly::ProjectedSparse($poly) => $body,
        }
    };
}

impl<F, P, const D: usize> AkitaPolyOps<F, D> for FoldInputPoly<'_, F, P, D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    P: AkitaPolyOps<F, D>,
{
    fn num_ring_elems(&self) -> usize {
        match self {
            FoldInputPoly::Original(poly) => AkitaPolyOps::num_ring_elems(*poly),
            FoldInputPoly::ProjectedDense(poly) => AkitaPolyOps::num_ring_elems(poly),
            FoldInputPoly::ProjectedSparse(poly) => AkitaPolyOps::num_ring_elems(poly.as_ref()),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            FoldInputPoly::Original(poly) => AkitaPolyOps::num_vars(*poly),
            FoldInputPoly::ProjectedDense(poly) => AkitaPolyOps::num_vars(poly),
            FoldInputPoly::ProjectedSparse(poly) => AkitaPolyOps::num_vars(poly.as_ref()),
        }
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        match self {
            Self::Original(poly) => poly.onehot_chunk_size(),
            Self::ProjectedDense(_) | Self::ProjectedSparse(_) => None,
        }
    }

    fn base_evals(&self) -> Result<Vec<F>, AkitaError> {
        dispatch_fold_input!(self, poly => poly.base_evals())
    }

    fn fold_blocks(
        &self,
        scalars: &[F],
        block_len: usize,
    ) -> Vec<akita_algebra::CyclotomicRing<F, D>> {
        dispatch_fold_input!(self, poly => poly.fold_blocks(scalars, block_len))
    }

    fn fold_blocks_ring(
        &self,
        scalars: &[akita_algebra::CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<akita_algebra::CyclotomicRing<F, D>> {
        dispatch_fold_input!(self, poly => poly.fold_blocks_ring(scalars, block_len))
    }

    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (
        akita_algebra::CyclotomicRing<F, D>,
        Vec<akita_algebra::CyclotomicRing<F, D>>,
    ) {
        dispatch_fold_input!(self, poly => {
            poly.evaluate_and_fold(eval_outer_scalars, fold_scalars, block_len)
        })
    }

    fn evaluate_and_fold_ring(
        &self,
        eval_outer_scalars: &[akita_algebra::CyclotomicRing<F, D>],
        fold_scalars: &[akita_algebra::CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (
        akita_algebra::CyclotomicRing<F, D>,
        Vec<akita_algebra::CyclotomicRing<F, D>>,
    ) {
        dispatch_fold_input!(self, poly => {
            poly.evaluate_and_fold_ring(eval_outer_scalars, fold_scalars, block_len)
        })
    }

    fn tensor_extension_column_partials<E>(&self, logical_point: &[E]) -> Result<Vec<E>, AkitaError>
    where
        E: akita_field::MulBaseUnreduced<F>,
    {
        dispatch_fold_input!(self, poly => poly.tensor_extension_column_partials(logical_point))
    }

    fn tensor_packed_extension_evals<E>(&self) -> Result<Vec<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        dispatch_fold_input!(self, poly => poly.tensor_packed_extension_evals::<E>())
    }

    fn tensor_packed_extension_poly<E>(&self) -> Result<DensePoly<F, D>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
        E: akita_types::FpExtEncoding<F>,
    {
        dispatch_fold_input!(self, poly => poly.tensor_packed_extension_poly::<E>())
    }

    fn decompose_fold(
        &self,
        challenges: &[akita_challenges::SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> crate::DecomposeFoldWitness<F, D> {
        dispatch_fold_input!(self, poly => {
            poly.decompose_fold(challenges, block_len, num_digits, log_basis)
        })
    }

    fn decompose_fold_batched(
        polys: &[&Self],
        challenges: &[akita_challenges::SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<crate::DecomposeFoldWitness<F, D>> {
        let first = *polys.first()?;
        match first {
            Self::Original(_) => {
                let mut originals = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        Self::Original(inner) => originals.push(*inner),
                        Self::ProjectedDense(_) | Self::ProjectedSparse(_) => return None,
                    }
                }
                P::decompose_fold_batched(&originals, challenges, block_len, num_digits, log_basis)
            }
            Self::ProjectedDense(_) => {
                let mut dense_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        Self::ProjectedDense(inner) => dense_polys.push(inner),
                        Self::Original(_) | Self::ProjectedSparse(_) => return None,
                    }
                }
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_batched(
                    &dense_polys,
                    challenges,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
            Self::ProjectedSparse(_) => {
                let mut sparse_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        Self::ProjectedSparse(inner) => sparse_polys.push(inner.as_ref()),
                        Self::Original(_) | Self::ProjectedDense(_) => return None,
                    }
                }
                <SparseRingPoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_batched(
                    &sparse_polys,
                    challenges,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
        }
    }

    fn decompose_fold_tensor_batched(
        polys: &[&Self],
        tensor: &akita_challenges::TensorChallenges,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Option<crate::DecomposeFoldWitness<F, D>>, AkitaError> {
        let Some(first) = polys.first() else {
            return Ok(None);
        };
        match *first {
            Self::Original(_) => {
                let mut originals = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        Self::Original(inner) => originals.push(*inner),
                        Self::ProjectedDense(_) | Self::ProjectedSparse(_) => return Ok(None),
                    }
                }
                P::decompose_fold_tensor_batched(
                    &originals, tensor, block_len, num_digits, log_basis,
                )
            }
            Self::ProjectedDense(_) => {
                let mut dense_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        Self::ProjectedDense(inner) => dense_polys.push(inner),
                        Self::Original(_) | Self::ProjectedSparse(_) => return Ok(None),
                    }
                }
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_tensor_batched(
                    &dense_polys,
                    tensor,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
            Self::ProjectedSparse(_) => {
                let mut sparse_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        Self::ProjectedSparse(inner) => sparse_polys.push(inner.as_ref()),
                        Self::Original(_) | Self::ProjectedDense(_) => return Ok(None),
                    }
                }
                <SparseRingPoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_tensor_batched(
                    &sparse_polys,
                    tensor,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
        }
    }

    fn commit_inner<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_a: usize,
        block_len: usize,
        num_blocks: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<crate::CommitInnerWitness<F, D>, AkitaError>
    where
        F: CanonicalField,
        B: CommitmentComputeBackend<F>,
    {
        dispatch_fold_input!(self, poly => {
            poly.commit_inner(
                backend,
                prepared,
                n_a,
                block_len,
                num_blocks,
                num_digits_commit,
                num_digits_open,
                log_basis,
            )
        })
    }

    fn direct_root_witness(&self) -> Result<akita_types::CleartextWitnessProof<F>, AkitaError> {
        match self {
            FoldInputPoly::Original(poly) => AkitaPolyOps::direct_root_witness(*poly),
            FoldInputPoly::ProjectedDense(poly) => AkitaPolyOps::direct_root_witness(poly),
            FoldInputPoly::ProjectedSparse(poly) => {
                AkitaPolyOps::direct_root_witness(poly.as_ref())
            }
        }
    }
}

fn tensor_extension_split<F, E>(context: &'static str) -> Result<(usize, usize), AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let split_bits = E::EXT_DEGREE.trailing_zeros() as usize;
    let width = 1usize
        .checked_shl(split_bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput("tensor extension width overflow".to_string()))?;
    if width != E::EXT_DEGREE || !E::EXT_DEGREE.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "tensor extension {context} requires power-of-two extension degree"
        )));
    }
    Ok((split_bits, width))
}

/// Pack a logical recursive digit witness into the canonical tensor extension
/// ring-subfield layout.
///
/// For degree-one fields this is the identity. For small fields this stores
/// the extension-valued tensor table in the same ring-subfield layout used by
/// folded extension openings.
///
/// # Errors
///
/// Returns an error if the logical witness length is not compatible with the
/// full tensor split or if ring-subfield packing fails.
pub fn tensor_pack_recursive_witness<F, E, const D: usize>(
    logical_w: &RecursiveWitnessFlat,
) -> Result<RecursiveWitnessFlat, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let (_split_bits, width) = tensor_extension_split::<F, E>("packing")?;
    let packed =
        pack_tensor_base_lift_i8_digits::<D>(logical_w.as_i8_digits(), E::EXT_DEGREE, width)?;
    Ok(RecursiveWitnessFlat::from_i8_digits(packed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{AkitaError, FpExt4, Prime32Offset99};

    #[test]
    fn recursive_tensor_pack_rejects_non_divisible_digit_count() {
        type F = Prime32Offset99;
        type E = FpExt4<F>;
        const D: usize = 32;
        let witness = RecursiveWitnessFlat::from_i8_digits(vec![1, 2, 3]);

        let err = tensor_pack_recursive_witness::<F, E, D>(&witness).unwrap_err();
        assert!(matches!(
            err,
            AkitaError::InvalidSize {
                expected: 4,
                actual: 3
            }
        ));
    }
}
