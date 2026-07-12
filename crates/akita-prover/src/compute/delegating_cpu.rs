//! Marker backends that delegate every compute operation to [`CpuBackend`].
//!
//! Distinct ZST types let integration tests exercise heterogeneous
//! [`super::stack::ProverComputeStack`] wiring without standing up four separate
//! hardware backends.

use super::backend::{
    CommitmentComputeBackend, ComputeBackendSetup, CyclicRowsComputeBackend,
    DigitRowsComputeBackend, RingSwitchComputeBackend,
};
use super::cpu::CpuBackend;
use super::kernels::{
    OpeningBatchKernel, OpeningFoldKernel, RingSwitchQuotientKernel, RingSwitchRelationKernel,
    RootCommitKernel, TensorProjectionBatchKernel, TensorProjectionKernel,
};
use super::operation_plans::{
    CommitInnerPlan, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningFoldOutput, OpeningFoldPlan,
    RingSwitchQuotientPlan, RingSwitchRelationPlan,
};
use super::plans::{
    DenseCommitRowsPlan, OneHotCommitRowsPlan, RecursiveWitnessCommitRowsPlan,
    RingSwitchQuotientRowsPlan, RingSwitchRelationRows, RingSwitchRelationRowsPlan,
    SparseRingCommitRowsPlan,
};
use crate::{CommitInnerWitness, DecomposeFoldWitness};
use akita_algebra::CyclotomicRing;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
};
use akita_types::{AkitaExpandedSetup, FpExtEncoding, NttCacheKey, PreparedNttPlan};
use std::sync::Arc;

macro_rules! delegate_compute_backend_setup {
    ($ty:ty) => {
        impl<F> ComputeBackendSetup<F> for $ty
        where
            F: FieldCore + CanonicalField,
        {
            type PreparedSetup = <CpuBackend as ComputeBackendSetup<F>>::PreparedSetup;

            fn prepare_expanded<const D: usize>(
                &self,
                expanded: Arc<AkitaExpandedSetup<F>>,
                plan: PreparedNttPlan,
            ) -> Result<Self::PreparedSetup, AkitaError> {
                CpuBackend.prepare_expanded::<D>(expanded, plan)
            }

            fn ensure_ntt_slot(
                &self,
                prepared: &Self::PreparedSetup,
                key: NttCacheKey,
            ) -> Result<(), AkitaError> {
                CpuBackend.ensure_ntt_slot(prepared, key)
            }

            fn with_ntt_slot<R>(
                &self,
                prepared: &Self::PreparedSetup,
                key: NttCacheKey,
                f: impl FnOnce(&crate::kernels::crt_ntt::NttSlotCacheAny) -> Result<R, AkitaError>,
            ) -> Result<R, AkitaError> {
                CpuBackend.with_ntt_slot(prepared, key, f)
            }

            fn prepared_expanded_setup<'a>(
                &self,
                prepared: &'a Self::PreparedSetup,
            ) -> &'a AkitaExpandedSetup<F> {
                CpuBackend.prepared_expanded_setup(prepared)
            }

            fn prepared_ntt_plan<'a>(
                &self,
                prepared: &'a Self::PreparedSetup,
            ) -> &'a PreparedNttPlan {
                CpuBackend.prepared_ntt_plan(prepared)
            }
        }
    };
}

macro_rules! delegate_digit_rows {
    ($ty:ty) => {
        impl<F> DigitRowsComputeBackend<F> for $ty
        where
            F: FieldCore + CanonicalField,
        {
            fn digit_rows<const D: usize>(
                &self,
                prepared: &Self::PreparedSetup,
                row_len: usize,
                digits: &[[i8; D]],
                log_basis: u32,
            ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
                CpuBackend.digit_rows(prepared, row_len, digits, log_basis)
            }
        }
    };
}

macro_rules! delegate_cyclic_rows {
    ($ty:ty) => {
        impl<F> CyclicRowsComputeBackend<F> for $ty
        where
            F: FieldCore + CanonicalField,
        {
            fn cyclic_digit_rows<const D: usize>(
                &self,
                prepared: &Self::PreparedSetup,
                row_len: usize,
                digits: &[[i8; D]],
                log_basis: u32,
            ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
                CpuBackend.cyclic_digit_rows(prepared, row_len, digits, log_basis)
            }
        }
    };
}

macro_rules! delegate_opening_kernels {
    ($ty:ty) => {
        impl<S, F, const D: usize> OpeningFoldKernel<S, F, D> for $ty
        where
            F: FieldCore + CanonicalField,
            CpuBackend: OpeningFoldKernel<S, F, D>,
        {
            fn evaluate_and_fold(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: S,
                plan: OpeningFoldPlan<'_, F, D>,
            ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
                CpuBackend.evaluate_and_fold(prepared, source, plan)
            }

            fn decompose_fold(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: S,
                plan: DecomposeFoldPlan<'_>,
            ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
                CpuBackend.decompose_fold(prepared, source, plan)
            }
        }

        impl<S, F, const D: usize> OpeningBatchKernel<S, F, D> for $ty
        where
            F: FieldCore + CanonicalField,
            CpuBackend: OpeningBatchKernel<S, F, D>,
        {
            fn decompose_fold_batch(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: S,
                plan: DecomposeFoldBatchPlan<'_>,
            ) -> Result<super::kernels::BatchDecomposeFoldOutcome<F, D>, AkitaError> {
                CpuBackend.decompose_fold_batch(prepared, source, plan)
            }
        }
    };
}

macro_rules! delegate_tensor_kernels {
    ($ty:ty) => {
        impl<S, F, E, const D: usize> TensorProjectionKernel<S, F, E, D> for $ty
        where
            F: FieldCore + CanonicalField,
            E: ExtField<F>,
            CpuBackend: TensorProjectionKernel<S, F, E, D>,
        {
            fn column_partials(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: S,
                logical_point: &[E],
            ) -> Result<Vec<E>, AkitaError>
            where
                E: akita_field::MulBaseUnreduced<F>,
            {
                CpuBackend.column_partials(prepared, source, logical_point)
            }

            fn packed_witness(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: S,
            ) -> Result<super::kernels::TensorPackedWitness<E>, AkitaError> {
                CpuBackend.packed_witness(prepared, source)
            }

            fn root_projection(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: S,
            ) -> Result<crate::backend::RootTensorProjectionPoly<F>, AkitaError>
            where
                F: FromPrimitiveInt,
                E: FpExtEncoding<F>,
            {
                CpuBackend.root_projection(prepared, source)
            }
        }

        impl<S, F, E, const D: usize> TensorProjectionBatchKernel<S, F, E, D> for $ty
        where
            F: FieldCore + CanonicalField,
            E: ExtField<F>,
            CpuBackend: TensorProjectionBatchKernel<S, F, E, D>,
        {
            fn column_partials_batch(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: S,
                logical_point: &[E],
            ) -> Result<Vec<Vec<E>>, AkitaError>
            where
                E: akita_field::MulBaseUnreduced<F>,
            {
                CpuBackend.column_partials_batch(prepared, source, logical_point)
            }

            fn sparse_linear_combination(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: S,
                coeffs: &[E],
            ) -> Result<
                Option<
                    crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness<E>,
                >,
                AkitaError,
            > {
                CpuBackend.sparse_linear_combination(prepared, source, coeffs)
            }
        }
    };
}

macro_rules! delegate_root_commit_kernel {
    ($ty:ty) => {
        impl<S, F, const D: usize> RootCommitKernel<S, F, D> for $ty
        where
            F: FieldCore + CanonicalField,
            CpuBackend: RootCommitKernel<S, F, D>,
        {
            fn commit_inner(
                &self,
                prepared: &Self::PreparedSetup,
                source: S,
                plan: CommitInnerPlan,
            ) -> Result<CommitInnerWitness<F>, AkitaError> {
                CpuBackend.commit_inner(prepared, source, plan)
            }
        }
    };
}

macro_rules! delegate_ring_switch_kernels {
    ($ty:ty) => {
        impl<S, F, const D: usize> RingSwitchRelationKernel<S, F, D> for $ty
        where
            F: FieldCore + CanonicalField,
            CpuBackend: RingSwitchRelationKernel<S, F, D>,
        {
            fn relation_rows(
                &self,
                prepared: &Self::PreparedSetup,
                source: S,
                plan: RingSwitchRelationPlan,
            ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
            where
                F: HalvingField,
            {
                CpuBackend.relation_rows(prepared, source, plan)
            }
        }

        impl<S, F, const D: usize> RingSwitchQuotientKernel<S, F, D> for $ty
        where
            F: FieldCore + CanonicalField,
            CpuBackend: RingSwitchQuotientKernel<S, F, D>,
        {
            fn quotient_rows(
                &self,
                prepared: &Self::PreparedSetup,
                source: S,
                plan: RingSwitchQuotientPlan,
            ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
            where
                F: HalvingField,
            {
                CpuBackend.quotient_rows(prepared, source, plan)
            }
        }
    };
}

/// Delegating commit-cluster marker backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct CommitCluster;

delegate_compute_backend_setup!(CommitCluster);
delegate_digit_rows!(CommitCluster);
delegate_cyclic_rows!(CommitCluster);
delegate_root_commit_kernel!(CommitCluster);

impl<F> CommitmentComputeBackend<F> for CommitCluster
where
    F: FieldCore + CanonicalField,
{
    fn dense_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: DenseCommitRowsPlan<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        CpuBackend.dense_commit_rows(prepared, plan)
    }

    fn onehot_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: OneHotCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        CpuBackend.onehot_commit_rows(prepared, plan)
    }

    fn sparse_ring_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: SparseRingCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        CpuBackend.sparse_ring_commit_rows(prepared, plan)
    }

    fn recursive_witness_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RecursiveWitnessCommitRowsPlan<'_, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        CpuBackend.recursive_witness_commit_rows(prepared, plan)
    }
}

/// Delegating opening-cluster marker backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpeningCluster;

delegate_compute_backend_setup!(OpeningCluster);
delegate_digit_rows!(OpeningCluster);
delegate_opening_kernels!(OpeningCluster);

/// Delegating tensor-cluster marker backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct TensorCluster;

delegate_compute_backend_setup!(TensorCluster);
delegate_tensor_kernels!(TensorCluster);

/// Delegating ring-switch-cluster marker backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct RingSwitchCluster;

delegate_compute_backend_setup!(RingSwitchCluster);
delegate_digit_rows!(RingSwitchCluster);
delegate_cyclic_rows!(RingSwitchCluster);
delegate_ring_switch_kernels!(RingSwitchCluster);

impl<F> RingSwitchComputeBackend<F> for RingSwitchCluster
where
    F: FieldCore + CanonicalField,
{
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RingSwitchRelationRowsPlan<'_, D>,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField,
    {
        CpuBackend.ring_switch_relation_rows(prepared, plan)
    }

    fn ring_switch_quotient_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RingSwitchQuotientRowsPlan<'_, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: HalvingField,
    {
        CpuBackend.ring_switch_quotient_rows(prepared, plan)
    }
}
