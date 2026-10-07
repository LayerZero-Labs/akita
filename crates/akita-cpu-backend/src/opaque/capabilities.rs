#![allow(private_bounds)]

//! D-free prover capability bundles generated from the runtime ring-dimension ladder.

use crate::opaque::ComputeBackendSetup;
use crate::opaque::FoldHandleBackend;
use crate::opaque::FoldRelationKernel;
use crate::opaque::SubringCoefficientPackingBatchKernel;
use crate::opaque::{
    OpeningProveBackendFor, RingSwitchProveBackend, RootOpeningSource, RootPolyMeta,
};
use jolt_field::Unreduced;
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

macro_rules! runtime_capabilities {
    (
        root_and_suffix: [$first:literal $(, $rest:literal)+ $(,)?],
        ring_switch_only: [$($ring_switch_only:literal),* $(,)?]
    ) => {
        /// Ring-switch kernels at every runtime-supported ring dimension.
        pub(crate) trait RuntimeRingSwitchProveBackend<F>:
            RingSwitchProveBackend<F, $first>
            $(+ RingSwitchProveBackend<F, $rest>)*
            $(+ RingSwitchProveBackend<F, $ring_switch_only>)*
        where
            F: Field + CanonicalEncoding,
        {
        }

        impl<F, B> RuntimeRingSwitchProveBackend<F> for B
        where
            F: Field + CanonicalEncoding,
            B: RingSwitchProveBackend<F, $first>
                $(+ RingSwitchProveBackend<F, $rest>)*
                $(+ RingSwitchProveBackend<F, $ring_switch_only>)*,
        {
        }

        /// Accepted-fold relation kernels at every suffix dimension.
        pub(crate) trait RuntimeFoldRelationBackend<F>:
            FoldHandleBackend<F>
            + FoldRelationKernel<<Self as FoldHandleBackend<F>>::AcceptedFold, F, $first>
            $(+ FoldRelationKernel<<Self as FoldHandleBackend<F>>::AcceptedFold, F, $rest>)*
        where
            F: Field + CanonicalEncoding,
        {
        }

        impl<F, B> RuntimeFoldRelationBackend<F> for B
        where
            F: Field + CanonicalEncoding,
            B: FoldHandleBackend<F>
                + FoldRelationKernel<<B as FoldHandleBackend<F>>::AcceptedFold, F, $first>
                $(+ FoldRelationKernel<<B as FoldHandleBackend<F>>::AcceptedFold, F, $rest>)*,
        {
        }

        /// Root polynomial opening sources at every runtime dimension.
        pub(crate) trait RuntimeOpeningSource<F>:
            RootOpeningSource<F, $first> $(+ RootOpeningSource<F, $rest>)*
        where
            F: Field,
        {
        }

        impl<F, P> RuntimeOpeningSource<F> for P
        where
            F: Field,
            P: RootOpeningSource<F, $first> $(+ RootOpeningSource<F, $rest>)*,
        {
        }

        /// Root polynomial usable for proving at every runtime dimension.
        pub(crate) trait RuntimeRootProvePoly<F>:
            RootPolyMeta<F> + RootOpeningSource<F, $first> $(+ RootOpeningSource<F, $rest>)*
        where
            F: Field,
        {
        }

        impl<F, P> RuntimeRootProvePoly<F> for P
        where
            F: Field,
            P: RootPolyMeta<F> + RootOpeningSource<F, $first> $(+ RootOpeningSource<F, $rest>)*,
        {
        }

        /// Opening backend for `P` at every runtime dimension.
        pub(crate) trait RuntimeOpeningProveBackendFor<F, P>:
            OpeningProveBackendFor<F, P, $first> $(+ OpeningProveBackendFor<F, P, $rest>)*
        where
            F: Field + CanonicalEncoding + Ring + Unreduced + 'static,
            <F as Unreduced>::Wide: From<F>,
            P: RuntimeOpeningSource<F>,
        {
        }

        impl<F, P, B> RuntimeOpeningProveBackendFor<F, P> for B
        where
            F: Field + CanonicalEncoding + Ring + Unreduced + 'static,
            <F as Unreduced>::Wide: From<F>,
            P: RuntimeOpeningSource<F>,
            B: OpeningProveBackendFor<F, P, $first>
                $(+ OpeningProveBackendFor<F, P, $rest>)*,
        {
        }

        /// Coefficient-packing projection backend for `P` at every runtime dimension.
        pub(crate) trait RuntimeCoefficientPackingBackendFor<F, P, E>:
            ComputeBackendSetup<F>
            + for<'a> SubringCoefficientPackingBatchKernel<
                <P as RootOpeningSource<F, $first>>::OpeningBatchView<'a>, F, E, $first
            >
            $(+ for<'a> SubringCoefficientPackingBatchKernel<
                <P as RootOpeningSource<F, $rest>>::OpeningBatchView<'a>, F, E, $rest
            >)*
        where
            F: Field + CanonicalEncoding,
            E: ExtField<F>,
            P: RuntimeOpeningSource<F>,
        {
        }

        impl<F, P, E, B> RuntimeCoefficientPackingBackendFor<F, P, E> for B
        where
            F: Field + CanonicalEncoding,
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


    };
}

runtime_capabilities! { root_and_suffix: [64, 128, 256, 512, 1024, 2048], ring_switch_only: [16, 32] }
