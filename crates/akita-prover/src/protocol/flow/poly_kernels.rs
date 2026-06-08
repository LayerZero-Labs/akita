//! Thin prove-path dispatch from root polynomial views to operation kernels.

use crate::compute::{
    tensor_root_projection as dispatch_tensor_root_projection, CommitmentComputeBackend,
    OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootOpeningSource, RootProvePoly,
    RootTensorSource, TensorPackedWitness, TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::RootTensorProjectionPoly;
use akita_algebra::CyclotomicRing;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced,
};
use akita_types::{
    tensor_packed_witness_evals, CleartextWitnessProof, RingMultiplierOpeningPoint,
    RingSubfieldEncoding,
};

pub(in crate::protocol::flow) fn opening_fold_plan_from_multiplier<'a, F, const D: usize>(
    point: &'a RingMultiplierOpeningPoint<F, D>,
    block_len: usize,
) -> Result<OpeningFoldPlan<'a, F, D>, AkitaError>
where
    F: FieldCore,
{
    if let Some(base_point) = point.as_base() {
        return Ok(OpeningFoldPlan::Base {
            eval_outer_scalars: &base_point.b,
            fold_scalars: &base_point.a,
            block_len,
        });
    }
    let b = point.b_rings().ok_or_else(|| {
        AkitaError::InvalidInput("ring multiplier must carry ring b weights".to_string())
    })?;
    let a = point.a_rings().ok_or_else(|| {
        AkitaError::InvalidInput("ring multiplier must carry ring a weights".to_string())
    })?;
    Ok(OpeningFoldPlan::Ring {
        eval_outer_scalars: b,
        fold_scalars: a,
        block_len,
    })
}

pub(in crate::protocol::flow) fn evaluate_at_multiplier_point<F, P, B, const D: usize>(
    backend: &B,
    poly: &P,
    point: &RingMultiplierOpeningPoint<F, D>,
    block_len: usize,
) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError>
where
    F: FieldCore + CanonicalField,
    P: RootOpeningSource<F, D>,
    B: for<'a> OpeningFoldKernel<P::OpeningView<'a>, F, D>,
{
    let plan = opening_fold_plan_from_multiplier(point, block_len)?;
    let OpeningFoldOutput { eval, folded } =
        OpeningFoldKernel::evaluate_and_fold(backend, None, poly.opening_view()?, plan)?;
    Ok((eval, folded))
}

pub(in crate::protocol::flow) fn tensor_root_projection<F, P, E, B, const D: usize>(
    backend: &B,
    poly: &P,
) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + RingSubfieldEncoding<F>,
    P: RootTensorSource<F, D>,
    B: CommitmentComputeBackend<F> + for<'a> TensorProjectionKernel<P::TensorView<'a>, F, E, D>,
{
    dispatch_tensor_root_projection(backend, None, poly)
}

pub(in crate::protocol::flow) fn tensor_extension_column_partials_batch<
    F,
    P,
    E,
    B,
    const D: usize,
>(
    backend: &B,
    polys: &[&P],
    logical_point: &[E],
) -> Result<Vec<Vec<E>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + MulBaseUnreduced<F>,
    P: RootTensorSource<F, D>,
    B: for<'a> TensorProjectionBatchKernel<P::TensorBatchView<'a>, F, E, D>,
{
    TensorProjectionBatchKernel::column_partials_batch(
        backend,
        None,
        P::tensor_batch(polys)?,
        logical_point,
    )
}

pub(in crate::protocol::flow) fn tensor_packed_extension_sparse_linear_combination<
    F,
    P,
    E,
    B,
    const D: usize,
>(
    backend: &B,
    polys: &[&P],
    coeffs: &[E],
) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    P: RootTensorSource<F, D>,
    B: for<'a> TensorProjectionBatchKernel<P::TensorBatchView<'a>, F, E, D>,
{
    if polys.len() != coeffs.len() {
        return Err(AkitaError::InvalidSize {
            expected: polys.len(),
            actual: coeffs.len(),
        });
    }
    TensorProjectionBatchKernel::sparse_linear_combination(
        backend,
        None,
        P::tensor_batch(polys)?,
        coeffs,
    )
}

pub(in crate::protocol::flow) fn tensor_packed_extension_evals<F, P, E, B, const D: usize>(
    backend: &B,
    poly: &P,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    P: RootProvePoly<F, D>,
    B: for<'a> TensorProjectionKernel<P::TensorView<'a>, F, E, D>,
{
    match TensorProjectionKernel::packed_witness(backend, None, poly.tensor_view()?)? {
        TensorPackedWitness::Dense(evals) => Ok(evals),
        TensorPackedWitness::Sparse(_) => dense_tensor_packed_evals_from_direct_witness(poly),
    }
}

fn dense_tensor_packed_evals_from_direct_witness<F, P, E, const D: usize>(
    poly: &P,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
    P: RootProvePoly<F, D>,
{
    let num_vars = poly.num_vars();
    let witness = poly.direct_root_witness()?;
    let CleartextWitnessProof::FieldElements(field_elems) = witness else {
        return Err(AkitaError::InvalidInput(
            "root tensor projection requires field-element root witness".to_string(),
        ));
    };
    tensor_packed_witness_evals::<F, E>(num_vars, field_elems.coeffs())
}
