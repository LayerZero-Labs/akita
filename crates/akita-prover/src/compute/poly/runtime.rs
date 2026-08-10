//! Runtime-dimension capability bundles for prover polynomial operations.

use super::*;

/// Ring dimensions the recursive suffix may dispatch besides the config ring `D`.
pub const RECURSIVE_SUFFIX_RING_DIMENSIONS: &[usize] = &[32, 64, 128, 256, 512, 1024];

/// Full prove-flow capability at a single root ring dimension `RING_D`:
/// opening/tensor prove kernels plus commitment rows.
pub trait ProveFlowBackendFor<F, P, E, const RING_D: usize>:
    RootProveBackend<F, P, E, RING_D> + CommitmentComputeBackend<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootProvePoly<F, RING_D>,
{
}

impl<F, P, E, const RING_D: usize, B> ProveFlowBackendFor<F, P, E, RING_D> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootProvePoly<F, RING_D>,
    B: RootProveBackend<F, P, E, RING_D> + CommitmentComputeBackend<F>,
{
}

/// [`ProveFlowBackendFor`] for `P` at every runtime-supported ring dimension.
pub trait RootProveFlowBackend<F, P, E>:
    ProveFlowBackendFor<F, P, E, 32>
    + ProveFlowBackendFor<F, P, E, 64>
    + ProveFlowBackendFor<F, P, E, 128>
    + ProveFlowBackendFor<F, P, E, 256>
    + ProveFlowBackendFor<F, P, E, 512>
    + ProveFlowBackendFor<F, P, E, 1024>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
{
}

impl<F, P, E, B> RootProveFlowBackend<F, P, E> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
    B: ProveFlowBackendFor<F, P, E, 32>
        + ProveFlowBackendFor<F, P, E, 64>
        + ProveFlowBackendFor<F, P, E, 128>
        + ProveFlowBackendFor<F, P, E, 256>
        + ProveFlowBackendFor<F, P, E, 512>
        + ProveFlowBackendFor<F, P, E, 1024>,
{
}

/// Recursive witness prove-flow capability over every runtime-supported fold
/// ring dimension.
pub trait RuntimeRecursiveWitnessProveBackend<F, E>:
    ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 32>
    + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 64>
    + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 128>
    + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 256>
    + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 512>
    + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 1024>
    + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 32>
    + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 64>
    + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 128>
    + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 256>
    + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 512>
    + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 1024>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
{
}

impl<F, E, B> RuntimeRecursiveWitnessProveBackend<F, E> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    B: ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 32>
        + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 64>
        + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 128>
        + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 256>
        + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 512>
        + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 1024>
        + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 32>
        + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 64>
        + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 128>
        + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 256>
        + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 512>
        + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 1024>,
{
}

/// Backend bundle for a full recursive prove run.
pub trait RecursiveProveBackend<F, P, E>:
    RootProveFlowBackend<F, P, E>
    + RuntimeRecursiveWitnessProveBackend<F, E>
    + RuntimeRingSwitchProveBackend<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
{
}

impl<F, P, E, B> RecursiveProveBackend<F, P, E> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
    B: RootProveFlowBackend<F, P, E>
        + RuntimeRecursiveWitnessProveBackend<F, E>
        + RuntimeRingSwitchProveBackend<F>,
{
}

/// Cluster capability bundle for [`crate::batched_prove`] with a heterogeneous
/// [`crate::ProverComputeStack`].
pub trait ProveStackFor<F, P, E, C, O, TS, R>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    TS: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
}

impl<F, P, E, C, O, TS, R> ProveStackFor<F, P, E, C, O, TS, R> for ()
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
    C: ComputeBackendSetup<F> + CommitmentComputeBackend<F>,
    O: ComputeBackendSetup<F>
        + RuntimeOpeningProveBackendFor<F, P>
        + RuntimeOpeningProveBackendFor<F, RecursiveFoldSource<F>>
        + SuffixOpeningProveBackend<F>
        + DigitRowsComputeBackend<F>,
    TS: ComputeBackendSetup<F>
        + RuntimeTensorBackendFor<F, P, E>
        + RuntimeTensorBackendFor<F, RecursiveFoldSource<F>, E>
        + SuffixTensorProveBackend<F, E>,
    R: ComputeBackendSetup<F> + RuntimeRingSwitchProveBackend<F> + DigitRowsComputeBackend<F>,
{
}
