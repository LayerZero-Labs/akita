//! Shared one-line dispatch from root polynomial views to operation kernels.

use super::kernels::TensorProjectionKernel;
use super::backend::CommitmentComputeBackend;
use super::poly::RootTensorSource;
use crate::RootTensorProjectionPoly;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_types::RingSubfieldEncoding;

pub(crate) fn tensor_root_projection<F, P, E, B, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup<D>>,
    poly: &P,
) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + RingSubfieldEncoding<F>,
    P: RootTensorSource<F, D>,
    B: CommitmentComputeBackend<F>
        + for<'a> TensorProjectionKernel<P::TensorView<'a>, F, E, D>,
{
    TensorProjectionKernel::root_projection(backend, prepared, poly.tensor_view()?)
}
