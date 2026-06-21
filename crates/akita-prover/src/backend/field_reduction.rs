//! Tensor extension-opening packing helpers.

use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, CanonicalField, FromPrimitiveInt, MulBaseUnreduced};
use akita_field::{AkitaError, ExtField, FieldCore};
use akita_types::{pack_tensor_base_lift_i8_digits, CleartextWitnessProof, FpExtEncoding};
use std::sync::Arc;

use super::dense::{DenseBatchView, DenseView};
use super::recursive_witness::{OwnedSuffixWitness, SuffixWitnessBatchView, SuffixWitnessView};
use super::sparse_ring::{SparseRingBatchView, SparseRingView};
use crate::compute::{
    CommitInnerPlan, CpuBackend, DecomposeFoldBatchPlan, DecomposeFoldPlan,
    DirectRootWitnessSource, OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput,
    OpeningFoldPlan, RootCommitKernel, RootCommitSource, RootOpeningSource, RootPolyShape,
    RootTensorSource, TensorPackedWitness, TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::{
    CommitInnerWitness, DecomposeFoldWitness, DensePoly, RecursiveWitnessFlat, SparseRingPoly,
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
    /// Recursive suffix witness folded directly over its ring blocks.
    ///
    /// Unlike root representations, recursive witnesses are not tensor-projected;
    /// the fold delegates to the suffix witness's own kernels (matching the legacy
    /// dense original-root behavior). Never committed via this enum.
    Recursive(OwnedSuffixWitness<F, D>),
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
            Self::Recursive(poly) => RootPolyShape::num_ring_elems(poly),
            Self::Sparse(poly) => RootPolyShape::num_ring_elems(poly.as_ref()),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::Dense(poly) => RootPolyShape::num_vars(poly),
            Self::Recursive(poly) => RootPolyShape::num_vars(poly),
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
            Self::Recursive(poly) => DirectRootWitnessSource::direct_root_witness(poly),
            Self::Sparse(poly) => DirectRootWitnessSource::direct_root_witness(poly.as_ref()),
        }
    }

    fn base_evals(&self) -> Result<Vec<F>, AkitaError> {
        match self {
            Self::Dense(poly) => DirectRootWitnessSource::base_evals(poly),
            Self::Recursive(poly) => DirectRootWitnessSource::base_evals(poly),
            Self::Sparse(poly) => DirectRootWitnessSource::base_evals(poly.as_ref()),
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
    ) -> Result<CommitInnerWitness<F, D>, AkitaError> {
        match source.poly {
            RootTensorProjectionPoly::Dense(poly) => {
                RootCommitKernel::<DenseView<'_, F, D>, F, D>::commit_inner(
                    self,
                    prepared,
                    poly.commit_view()?,
                    plan,
                )
            }
            RootTensorProjectionPoly::Sparse(poly) => {
                RootCommitKernel::<SparseRingView<'_, F, D>, F, D>::commit_inner(
                    self,
                    prepared,
                    poly.as_ref().commit_view()?,
                    plan,
                )
            }
            RootTensorProjectionPoly::Recursive(_) => Err(AkitaError::InvalidInput(
                "recursive tensor-projection poly is a fold input and is never committed"
                    .to_string(),
            )),
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
            RootTensorProjectionPoly::Dense(poly) => {
                OpeningFoldKernel::<DenseView<'_, F, D>, F, D>::evaluate_and_fold(
                    self,
                    prepared,
                    poly.opening_view()?,
                    plan,
                )
            }
            RootTensorProjectionPoly::Sparse(poly) => {
                OpeningFoldKernel::<SparseRingView<'_, F, D>, F, D>::evaluate_and_fold(
                    self,
                    prepared,
                    poly.as_ref().opening_view()?,
                    plan,
                )
            }
            RootTensorProjectionPoly::Recursive(poly) => {
                OpeningFoldKernel::<SuffixWitnessView<F, D>, F, D>::evaluate_and_fold(
                    self,
                    prepared,
                    poly.opening_view()?,
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
            RootTensorProjectionPoly::Dense(poly) => {
                OpeningFoldKernel::<DenseView<'_, F, D>, F, D>::decompose_fold(
                    self,
                    prepared,
                    poly.opening_view()?,
                    plan,
                )
            }
            RootTensorProjectionPoly::Sparse(poly) => {
                OpeningFoldKernel::<SparseRingView<'_, F, D>, F, D>::decompose_fold(
                    self,
                    prepared,
                    poly.as_ref().opening_view()?,
                    plan,
                )
            }
            RootTensorProjectionPoly::Recursive(poly) => {
                OpeningFoldKernel::<SuffixWitnessView<F, D>, F, D>::decompose_fold(
                    self,
                    prepared,
                    poly.opening_view()?,
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
                        _ => return Ok(None),
                    }
                }
                let dense_view = DensePoly::<F, D>::opening_batch(&dense_polys)?;
                OpeningBatchKernel::<DenseBatchView<'_, F, D>, F, D>::decompose_fold_batch(
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
                        _ => return Ok(None),
                    }
                }
                let sparse_view = SparseRingPoly::<F, D>::opening_batch(&sparse_polys)?;
                OpeningBatchKernel::<SparseRingBatchView<'_, F, D>, F, D>::decompose_fold_batch(
                    self,
                    prepared,
                    sparse_view,
                    plan,
                )
            }
            RootTensorProjectionPoly::Recursive(_) => {
                let mut recursive_polys = Vec::with_capacity(source.polys.len());
                for poly in source.polys {
                    match poly {
                        RootTensorProjectionPoly::Recursive(inner) => recursive_polys.push(inner),
                        _ => return Ok(None),
                    }
                }
                let recursive_view = OwnedSuffixWitness::<F, D>::opening_batch(&recursive_polys)?;
                OpeningBatchKernel::<SuffixWitnessBatchView<F, D>, F, D>::decompose_fold_batch(
                    self,
                    prepared,
                    recursive_view,
                    plan,
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
                TensorProjectionKernel::<DenseView<'_, F, D>, F, E, D>::column_partials(
                    self,
                    prepared,
                    poly.tensor_view()?,
                    logical_point,
                )
            }
            RootTensorProjectionPoly::Sparse(poly) => {
                TensorProjectionKernel::<SparseRingView<'_, F, D>, F, E, D>::column_partials(
                    self,
                    prepared,
                    poly.as_ref().tensor_view()?,
                    logical_point,
                )
            }
            RootTensorProjectionPoly::Recursive(poly) => {
                TensorProjectionKernel::<SuffixWitnessView<F, D>, F, E, D>::column_partials(
                    self,
                    prepared,
                    poly.tensor_view()?,
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
                TensorProjectionKernel::<DenseView<'_, F, D>, F, E, D>::packed_witness(
                    self,
                    prepared,
                    poly.tensor_view()?,
                )
            }
            RootTensorProjectionPoly::Sparse(poly) => {
                TensorProjectionKernel::<SparseRingView<'_, F, D>, F, E, D>::packed_witness(
                    self,
                    prepared,
                    poly.as_ref().tensor_view()?,
                )
            }
            RootTensorProjectionPoly::Recursive(poly) => {
                TensorProjectionKernel::<SuffixWitnessView<F, D>, F, E, D>::packed_witness(
                    self,
                    prepared,
                    poly.tensor_view()?,
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
                TensorProjectionKernel::<DenseView<'_, F, D>, F, E, D>::root_projection(
                    self,
                    prepared,
                    poly.tensor_view()?,
                )
            }
            RootTensorProjectionPoly::Sparse(poly) => {
                TensorProjectionKernel::<SparseRingView<'_, F, D>, F, E, D>::root_projection(
                    self,
                    prepared,
                    poly.as_ref().tensor_view()?,
                )
            }
            RootTensorProjectionPoly::Recursive(poly) => {
                TensorProjectionKernel::<SuffixWitnessView<F, D>, F, E, D>::root_projection(
                    self,
                    prepared,
                    poly.tensor_view()?,
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
