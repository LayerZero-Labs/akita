//! D-free prover capability bundles generated from the runtime ring-dimension ladder.

use super::backend::{ComputeBackendSetup, DigitRowsComputeBackend};
use super::kernels::{RootCommitKernel, SubringCoefficientPackingBatchKernel};
use super::poly::{
    OpeningProveBackendFor, ProveFlowBackendFor, RingSwitchProveBackend, RootCommitSource,
    RootOpeningSource, RootPolyMeta, RootTensorSource, TensorBackendFor,
};
use crate::backend::{RecursiveFoldSource, RecursiveWitnessFlat};
use crate::RootTensorProjectionPoly;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{CanonicalField, ExtField, FieldCore, FromPrimitiveInt, RandomSampling};

macro_rules! runtime_capabilities {
    (
        root_and_suffix: [$first:literal $(, $rest:literal)+ $(,)?],
        ring_switch_only: [$($ring_switch_only:literal),* $(,)?]
    ) => {
        /// Ring-switch kernels at every runtime-supported ring dimension.
        pub trait RuntimeRingSwitchProveBackend<F>:
            RingSwitchProveBackend<F, $first>
            $(+ RingSwitchProveBackend<F, $rest>)*
            $(+ RingSwitchProveBackend<F, $ring_switch_only>)*
        where
            F: FieldCore + CanonicalField,
        {
        }

        impl<F, B> RuntimeRingSwitchProveBackend<F> for B
        where
            F: FieldCore + CanonicalField,
            B: RingSwitchProveBackend<F, $first>
                $(+ RingSwitchProveBackend<F, $rest>)*
                $(+ RingSwitchProveBackend<F, $ring_switch_only>)*,
        {
        }

        /// Opening kernels for suffix witnesses and root-tensor projections.
        pub trait SuffixOpeningProveBackend<F>:
            OpeningProveBackendFor<F, RecursiveWitnessFlat, $first>
            + OpeningProveBackendFor<F, RootTensorProjectionPoly<F>, $first>
            $(+ OpeningProveBackendFor<F, RecursiveWitnessFlat, $rest>
                + OpeningProveBackendFor<F, RootTensorProjectionPoly<F>, $rest>)*
        where
            F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
            <F as HasWide>::Wide: From<F> + ReduceTo<F>,
        {
        }

        impl<F, B> SuffixOpeningProveBackend<F> for B
        where
            F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
            <F as HasWide>::Wide: From<F> + ReduceTo<F>,
            B: OpeningProveBackendFor<F, RecursiveWitnessFlat, $first>
                + OpeningProveBackendFor<F, RootTensorProjectionPoly<F>, $first>
                $(+ OpeningProveBackendFor<F, RecursiveWitnessFlat, $rest>
                    + OpeningProveBackendFor<F, RootTensorProjectionPoly<F>, $rest>)*,
        {
        }

        /// Tensor kernels for suffix witnesses and root-tensor projections.
        pub trait SuffixTensorProveBackend<F, E>:
            TensorBackendFor<F, RecursiveWitnessFlat, E, $first>
            + TensorBackendFor<F, RootTensorProjectionPoly<F>, E, $first>
            $(+ TensorBackendFor<F, RecursiveWitnessFlat, E, $rest>
                + TensorBackendFor<F, RootTensorProjectionPoly<F>, E, $rest>)*
        where
            F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
            <F as HasWide>::Wide: From<F> + ReduceTo<F>,
            E: ExtField<F>,
        {
        }

        impl<F, E, B> SuffixTensorProveBackend<F, E> for B
        where
            F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
            <F as HasWide>::Wide: From<F> + ReduceTo<F>,
            E: ExtField<F>,
            B: TensorBackendFor<F, RecursiveWitnessFlat, E, $first>
                + TensorBackendFor<F, RootTensorProjectionPoly<F>, E, $first>
                $(+ TensorBackendFor<F, RecursiveWitnessFlat, E, $rest>
                    + TensorBackendFor<F, RootTensorProjectionPoly<F>, E, $rest>)*,
        {
        }

        /// Root polynomial opening sources at every runtime dimension.
        pub trait RuntimeOpeningSource<F>:
            RootOpeningSource<F, $first> $(+ RootOpeningSource<F, $rest>)*
        where
            F: FieldCore,
        {
        }

        impl<F, P> RuntimeOpeningSource<F> for P
        where
            F: FieldCore,
            P: RootOpeningSource<F, $first> $(+ RootOpeningSource<F, $rest>)*,
        {
        }

        /// Root polynomial tensor sources at every runtime dimension.
        pub trait RuntimeTensorSource<F>:
            RootTensorSource<F, $first> $(+ RootTensorSource<F, $rest>)*
        where
            F: FieldCore,
        {
        }

        impl<F, P> RuntimeTensorSource<F> for P
        where
            F: FieldCore,
            P: RootTensorSource<F, $first> $(+ RootTensorSource<F, $rest>)*,
        {
        }

        /// Root polynomial commit sources at every runtime dimension.
        pub trait RuntimeCommitSource<F>:
            RootPolyMeta<F> + RootCommitSource<F, $first> $(+ RootCommitSource<F, $rest>)*
        where
            F: FieldCore,
        {
        }

        impl<F, P> RuntimeCommitSource<F> for P
        where
            F: FieldCore,
            P: RootPolyMeta<F> + RootCommitSource<F, $first> $(+ RootCommitSource<F, $rest>)*,
        {
        }

        /// Root polynomial usable for proving at every runtime dimension.
        pub trait RuntimeRootProvePoly<F>:
            RootPolyMeta<F> + RootOpeningSource<F, $first> $(+ RootOpeningSource<F, $rest>)*
        where
            F: FieldCore,
        {
        }

        impl<F, P> RuntimeRootProvePoly<F> for P
        where
            F: FieldCore,
            P: RootPolyMeta<F> + RootOpeningSource<F, $first> $(+ RootOpeningSource<F, $rest>)*,
        {
        }

        /// Opening backend for `P` at every runtime dimension.
        pub trait RuntimeOpeningProveBackendFor<F, P>:
            OpeningProveBackendFor<F, P, $first> $(+ OpeningProveBackendFor<F, P, $rest>)*
        where
            F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
            <F as HasWide>::Wide: From<F> + ReduceTo<F>,
            P: RuntimeOpeningSource<F>,
        {
        }

        impl<F, P, B> RuntimeOpeningProveBackendFor<F, P> for B
        where
            F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
            <F as HasWide>::Wide: From<F> + ReduceTo<F>,
            P: RuntimeOpeningSource<F>,
            B: OpeningProveBackendFor<F, P, $first> $(+ OpeningProveBackendFor<F, P, $rest>)*,
        {
        }

        /// Coefficient-packing projection backend for `P` at every runtime dimension.
        pub trait RuntimeCoefficientPackingBackendFor<F, P, E>:
            ComputeBackendSetup<F>
            + for<'a> SubringCoefficientPackingBatchKernel<
                <P as RootOpeningSource<F, $first>>::OpeningBatchView<'a>, F, E, $first
            >
            $(+ for<'a> SubringCoefficientPackingBatchKernel<
                <P as RootOpeningSource<F, $rest>>::OpeningBatchView<'a>, F, E, $rest
            >)*
        where
            F: FieldCore + CanonicalField,
            E: ExtField<F>,
            P: RuntimeOpeningSource<F>,
        {
        }

        impl<F, P, E, B> RuntimeCoefficientPackingBackendFor<F, P, E> for B
        where
            F: FieldCore + CanonicalField,
            E: ExtField<F>,
            P: RuntimeOpeningSource<F>,
            B: ComputeBackendSetup<F>
                + for<'a> SubringCoefficientPackingBatchKernel<
                    <P as RootOpeningSource<F, $first>>::OpeningBatchView<'a>, F, E, $first
                >
                $(+ for<'a> SubringCoefficientPackingBatchKernel<
                    <P as RootOpeningSource<F, $rest>>::OpeningBatchView<'a>, F, E, $rest
                >)*,
        {
        }

        /// Tensor backend for `P` at every runtime dimension.
        pub trait RuntimeTensorBackendFor<F, P, E>:
            TensorBackendFor<F, P, E, $first> $(+ TensorBackendFor<F, P, E, $rest>)*
        where
            F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
            <F as HasWide>::Wide: From<F> + ReduceTo<F>,
            E: ExtField<F>,
            P: RuntimeTensorSource<F>,
        {
        }

        impl<F, P, E, B> RuntimeTensorBackendFor<F, P, E> for B
        where
            F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
            <F as HasWide>::Wide: From<F> + ReduceTo<F>,
            E: ExtField<F>,
            P: RuntimeTensorSource<F>,
            B: TensorBackendFor<F, P, E, $first> $(+ TensorBackendFor<F, P, E, $rest>)*,
        {
        }

        /// Commit backend for `P` at every runtime dimension.
        pub trait RuntimeCommitBackendFor<F, P>:
            DigitRowsComputeBackend<F>
            + for<'a> RootCommitKernel<
                <P as RootCommitSource<F, $first>>::CommitView<'a>, F, $first
            >
            $(+ for<'a> RootCommitKernel<
                <P as RootCommitSource<F, $rest>>::CommitView<'a>, F, $rest
            >)*
        where
            F: FieldCore + CanonicalField,
            P: RuntimeCommitSource<F>,
        {
        }

        impl<F, P, B> RuntimeCommitBackendFor<F, P> for B
        where
            F: FieldCore + CanonicalField,
            P: RuntimeCommitSource<F>,
            B: DigitRowsComputeBackend<F>
                + for<'a> RootCommitKernel<
                    <P as RootCommitSource<F, $first>>::CommitView<'a>, F, $first
                >
                $(+ for<'a> RootCommitKernel<
                    <P as RootCommitSource<F, $rest>>::CommitView<'a>, F, $rest
                >)*,
        {
        }

        /// Full root prove flow at every runtime dimension.
        pub trait RootProveFlowBackend<F, P, E>:
            ProveFlowBackendFor<F, P, E, $first> $(+ ProveFlowBackendFor<F, P, E, $rest>)*
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
            B: ProveFlowBackendFor<F, P, E, $first> $(+ ProveFlowBackendFor<F, P, E, $rest>)*,
        {
        }

        /// Recursive witness prove flow at every runtime dimension.
        pub trait RuntimeRecursiveWitnessProveBackend<F, E>:
            ProveFlowBackendFor<F, RecursiveWitnessFlat, E, $first>
            + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, $first>
            $(+ ProveFlowBackendFor<F, RecursiveWitnessFlat, E, $rest>
                + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, $rest>)*
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
            B: ProveFlowBackendFor<F, RecursiveWitnessFlat, E, $first>
                + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, $first>
                $(+ ProveFlowBackendFor<F, RecursiveWitnessFlat, E, $rest>
                    + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, $rest>)*,
        {
        }
    };
}

runtime_capabilities! {
    root_and_suffix: [64, 128, 256, 512, 1024],
    ring_switch_only: [16, 32]
}
