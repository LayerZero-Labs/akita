//! Tensor extension-opening packing helpers.

use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, CanonicalField, FromPrimitiveInt, MulBaseUnreduced};
use akita_field::{AkitaError, ExtField, FieldCore};
use akita_types::{
    pack_tensor_base_lift_i8_digits, CleartextWitnessProof, FlatDigitBlocks, RingSubfieldEncoding,
};
use std::sync::Arc;

use super::{
    DenseOpeningBatchView, DenseTensorBatchView, SparseRingOpeningBatchView,
    SparseRingTensorBatchView,
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
    CommitInnerWitness, DecomposeFoldWitness, DensePoly, RecursiveWitnessFlat,
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

macro_rules! dispatch_root_projection {
    ($self:expr, $poly:ident => $body:expr) => {
        match $self {
            RootTensorProjectionPoly::Dense($poly) => $body,
            RootTensorProjectionPoly::Sparse($poly) => $body,
        }
    };
}

/// Borrowed dispatch view for an Akita-owned root tensor projection polynomial.
#[derive(Debug, Clone, Copy)]
pub struct RootTensorProjectionView<'a, F: FieldCore, const D: usize> {
    poly: &'a RootTensorProjectionPoly<F, D>,
}

/// Same-point batch dispatch view over root tensor projection polynomials.
#[derive(Debug, Clone, Copy)]
pub struct RootTensorProjectionBatchView<'a, F: FieldCore, const D: usize> {
    polys: &'a [&'a RootTensorProjectionPoly<F, D>],
}

impl<F, const D: usize> RootPolyShape<F, D> for RootTensorProjectionPoly<F, D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn num_ring_elems(&self) -> usize {
        match self {
            RootTensorProjectionPoly::Dense(poly) => RootPolyShape::num_ring_elems(poly),
            RootTensorProjectionPoly::Sparse(poly) => RootPolyShape::num_ring_elems(poly.as_ref()),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            RootTensorProjectionPoly::Dense(poly) => RootPolyShape::num_vars(poly),
            RootTensorProjectionPoly::Sparse(poly) => RootPolyShape::num_vars(poly.as_ref()),
        }
    }
}

impl<F, const D: usize> RootCommitSource<F, D> for RootTensorProjectionPoly<F, D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
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
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
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
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
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
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        match self {
            RootTensorProjectionPoly::Dense(poly) => {
                DirectRootWitnessSource::direct_root_witness(poly)
            }
            RootTensorProjectionPoly::Sparse(poly) => {
                DirectRootWitnessSource::direct_root_witness(poly.as_ref())
            }
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
        source.poly.commit_inner(
            self,
            prepared,
            plan.n_a,
            plan.block_len,
            plan.num_digits_commit,
            plan.num_digits_open,
            plan.log_basis,
        )
    }

    fn commit_inner_witness(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: RootTensorProjectionView<'_, F, D>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError> {
        source.poly.commit_inner_witness(
            self,
            prepared,
            plan.n_a,
            plan.block_len,
            plan.num_digits_commit,
            plan.num_digits_open,
            plan.log_basis,
        )
    }
}

impl<F, const D: usize> OpeningFoldKernel<RootTensorProjectionView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F, D>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        let (eval, folded) = match plan {
            OpeningFoldPlan::Base {
                eval_outer_scalars,
                fold_scalars,
                block_len,
            } => source
                .poly
                .evaluate_and_fold(eval_outer_scalars, fold_scalars, block_len),
            OpeningFoldPlan::Ring {
                eval_outer_scalars,
                fold_scalars,
                block_len,
            } => source
                .poly
                .evaluate_and_fold_ring(eval_outer_scalars, fold_scalars, block_len),
        };
        Ok(OpeningFoldOutput { eval, folded })
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F, D>, AkitaError> {
        Ok(source.poly.decompose_fold(
            plan.challenges,
            plan.block_len,
            plan.num_digits,
            plan.log_basis,
        ))
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
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        source.poly.tensor_extension_column_partials(logical_point)
    }

    fn packed_witness(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionView<'_, F, D>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        Ok(match source.poly.tensor_packed_extension_sparse_evals()? {
            Some(witness) => TensorPackedWitness::Sparse(witness),
            None => TensorPackedWitness::Dense(source.poly.tensor_packed_extension_evals()?),
        })
    }

    fn root_projection(
        &self,
        _prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionView<'_, F, D>,
    ) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
    where
        E: RingSubfieldEncoding<F>,
    {
        source.poly.tensor_packed_extension_root_poly::<E>()
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
        let Some(first) = source.polys.first() else {
            return Ok(Vec::new());
        };
        match *first {
            RootTensorProjectionPoly::Dense(_) => {
                let mut dense_polys = Vec::with_capacity(source.polys.len());
                for poly in source.polys {
                    match *poly {
                        RootTensorProjectionPoly::Dense(inner) => dense_polys.push(inner),
                        RootTensorProjectionPoly::Sparse(_) => {
                            return source
                                .polys
                                .iter()
                                .map(|poly| {
                                    TensorProjectionKernel::<
                                        RootTensorProjectionView<'_, F, D>,
                                        F,
                                        E,
                                        D,
                                    >::column_partials(
                                        self,
                                        prepared,
                                        poly.tensor_view()?,
                                        logical_point,
                                    )
                                })
                                .collect();
                        }
                    }
                }
                let dense_view = DensePoly::<F, D>::tensor_batch(&dense_polys)?;
                TensorProjectionBatchKernel::<DenseTensorBatchView<'_, F, D>, F, E, D>::column_partials_batch(
                    self,
                    prepared,
                    dense_view,
                    logical_point,
                )
            }
            RootTensorProjectionPoly::Sparse(_) => {
                let mut sparse_polys = Vec::with_capacity(source.polys.len());
                for poly in source.polys {
                    match *poly {
                        RootTensorProjectionPoly::Sparse(inner) => {
                            sparse_polys.push(inner.as_ref())
                        }
                        RootTensorProjectionPoly::Dense(_) => {
                            return source
                                .polys
                                .iter()
                                .map(|poly| {
                                    TensorProjectionKernel::<
                                        RootTensorProjectionView<'_, F, D>,
                                        F,
                                        E,
                                        D,
                                    >::column_partials(
                                        self,
                                        prepared,
                                        poly.tensor_view()?,
                                        logical_point,
                                    )
                                })
                                .collect();
                        }
                    }
                }
                let sparse_view = SparseRingPoly::<F, D>::tensor_batch(&sparse_polys)?;
                TensorProjectionBatchKernel::<SparseRingTensorBatchView<'_, F, D>, F, E, D>::column_partials_batch(
                    self,
                    prepared,
                    sparse_view,
                    logical_point,
                )
            }
        }
    }

    fn sparse_linear_combination(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: RootTensorProjectionBatchView<'_, F, D>,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError> {
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
                let dense_view = DensePoly::<F, D>::tensor_batch(&dense_polys)?;
                TensorProjectionBatchKernel::<DenseTensorBatchView<'_, F, D>, F, E, D>::sparse_linear_combination(
                    self,
                    prepared,
                    dense_view,
                    coeffs,
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
                let sparse_view = SparseRingPoly::<F, D>::tensor_batch(&sparse_polys)?;
                TensorProjectionBatchKernel::<SparseRingTensorBatchView<'_, F, D>, F, E, D>::sparse_linear_combination(
                    self,
                    prepared,
                    sparse_view,
                    coeffs,
                )
            }
        }
    }
}

impl<F, const D: usize> RootTensorProjectionPoly<F, D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    pub(crate) fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (
        akita_algebra::CyclotomicRing<F, D>,
        Vec<akita_algebra::CyclotomicRing<F, D>>,
    ) {
        dispatch_root_projection!(self, poly => {
            poly.evaluate_and_fold(eval_outer_scalars, fold_scalars, block_len)
        })
    }

    pub(crate) fn evaluate_and_fold_ring(
        &self,
        eval_outer_scalars: &[akita_algebra::CyclotomicRing<F, D>],
        fold_scalars: &[akita_algebra::CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (
        akita_algebra::CyclotomicRing<F, D>,
        Vec<akita_algebra::CyclotomicRing<F, D>>,
    ) {
        dispatch_root_projection!(self, poly => {
            poly.evaluate_and_fold_ring(eval_outer_scalars, fold_scalars, block_len)
        })
    }

    pub(crate) fn decompose_fold(
        &self,
        challenges: &[akita_challenges::SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> crate::DecomposeFoldWitness<F, D> {
        dispatch_root_projection!(self, poly => {
            poly.decompose_fold(challenges, block_len, num_digits, log_basis)
        })
    }

    pub(crate) fn tensor_extension_column_partials<E>(&self, logical_point: &[E]) -> Result<Vec<E>, AkitaError>
    where
        E: akita_field::MulBaseUnreduced<F>,
    {
        dispatch_root_projection!(self, poly => poly.tensor_extension_column_partials(logical_point))
    }

    pub(crate) fn tensor_packed_extension_evals<E>(&self) -> Result<Vec<E>, AkitaError>
    where
        E: akita_field::ExtField<F>,
    {
        dispatch_root_projection!(self, poly => poly.tensor_packed_extension_evals())
    }

    pub(crate) fn tensor_packed_extension_sparse_evals<E>(
        &self,
    ) -> Result<Option<crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: akita_field::ExtField<F>,
    {
        dispatch_root_projection!(self, poly => poly.tensor_packed_extension_sparse_evals())
    }

    pub(crate) fn tensor_packed_extension_root_poly<E>(
        &self,
    ) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
    where
        F: CanonicalField + akita_field::FromPrimitiveInt,
        E: akita_types::RingSubfieldEncoding<F>,
    {
        dispatch_root_projection!(self, poly => poly.tensor_packed_extension_root_poly::<E>())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_inner<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<akita_types::FlatDigitBlocks<D>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        dispatch_root_projection!(self, poly => {
            poly.commit_inner(
                backend,
                prepared,
                n_a,
                block_len,
                num_digits_commit,
                num_digits_open,
                log_basis,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_inner_witness<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<crate::CommitInnerWitness<F, D>, AkitaError>
    where
        F: CanonicalField,
        B: CommitmentComputeBackend<F>,
    {
        dispatch_root_projection!(self, poly => {
            poly.commit_inner_witness(
                backend,
                prepared,
                n_a,
                block_len,
                num_digits_commit,
                num_digits_open,
                log_basis,
            )
        })
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
    use akita_field::{AkitaError, Prime32Offset99, RingSubfieldFpExt4};

    #[test]
    fn recursive_tensor_pack_rejects_non_divisible_digit_count() {
        type F = Prime32Offset99;
        type E = RingSubfieldFpExt4<F>;
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

    #[test]
    fn root_projection_kernel_tensor_paths_match_inner_dense() {
        type F = Prime32Offset99;
        type E = RingSubfieldFpExt4<F>;
        const D: usize = 8;

        let num_vars = 4;
        let evals = (0..(1usize << num_vars))
            .map(|idx| F::from_u64(idx as u64 + 1))
            .collect::<Vec<_>>();
        let dense = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
        let root =
            DensePoly::tensor_packed_extension_root_poly::<E>(&dense)
                .unwrap();
        let point = (0..num_vars)
            .map(|idx| {
                E::from_base_slice(&[
                    F::from_u64(idx as u64 + 2),
                    F::from_u64(3 * idx as u64 + 4),
                    F::from_u64(5 * idx as u64 + 6),
                    F::from_u64(7 * idx as u64 + 8),
                ])
            })
            .collect::<Vec<_>>();
        let backend = CpuBackend;
        let tensor_view = root.tensor_view().unwrap();

        let ops_partials = root.tensor_extension_column_partials::<E>(&point).unwrap();
        let kernel_partials =
            TensorProjectionKernel::<RootTensorProjectionView<'_, F, D>, F, E, D>::column_partials(
                &backend,
                None,
                tensor_view,
                &point,
            )
            .unwrap();
        assert_eq!(kernel_partials, ops_partials);

        let ops_packed = root.tensor_packed_extension_evals::<E>().unwrap();
        let kernel_packed = match TensorProjectionKernel::<
            RootTensorProjectionView<'_, F, D>,
            F,
            E,
            D,
        >::packed_witness(&backend, None, tensor_view)
        .unwrap()
        {
            TensorPackedWitness::Dense(v) => v,
            TensorPackedWitness::Sparse(_) => {
                panic!("dense root projection kernel must return dense packed witness")
            }
        };
        assert_eq!(kernel_packed, ops_packed);
    }

    #[test]
    fn root_projection_kernel_homogeneous_dense_batch_matches_inner() {
        type F = Prime32Offset99;
        type E = RingSubfieldFpExt4<F>;
        const D: usize = 8;

        let num_vars = 4;
        let evals = (0..(1usize << num_vars))
            .map(|idx| F::from_u64(idx as u64 + 1))
            .collect::<Vec<_>>();
        let dense0 = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
        let dense1 = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
        let roots = [
            DensePoly::tensor_packed_extension_root_poly::<E>(
                &dense0,
            )
            .unwrap(),
            DensePoly::tensor_packed_extension_root_poly::<E>(
                &dense1,
            )
            .unwrap(),
        ];
        let root_refs = [&roots[0], &roots[1]];
        let point = (0..num_vars)
            .map(|idx| {
                E::from_base_slice(&[
                    F::from_u64(idx as u64 + 2),
                    F::from_u64(3 * idx as u64 + 4),
                    F::from_u64(5 * idx as u64 + 6),
                    F::from_u64(7 * idx as u64 + 8),
                ])
            })
            .collect::<Vec<_>>();
        let backend = CpuBackend;

        let dense_roots: Vec<&DensePoly<F, D>> = roots
            .iter()
            .map(|root| match root {
                RootTensorProjectionPoly::Dense(inner) => inner,
                RootTensorProjectionPoly::Sparse(_) => {
                    panic!("test roots are dense tensor projections")
                }
            })
            .collect();
        let dense_root_refs: Vec<&DensePoly<F, D>> = dense_roots.iter().copied().collect();
        let expected = DensePoly::tensor_extension_column_partials_batch::<E>(
            &dense_root_refs,
            &point,
        )
        .unwrap();
        let batch_view = RootTensorProjectionPoly::<F, D>::tensor_batch(&root_refs).unwrap();
        let got = TensorProjectionBatchKernel::<RootTensorProjectionBatchView<'_, F, D>, F, E, D>::column_partials_batch(
            &backend,
            None,
            batch_view,
            &point,
        )
        .unwrap();
        assert_eq!(got, expected);
    }
}
