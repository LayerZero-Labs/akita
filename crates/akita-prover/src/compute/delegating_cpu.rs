//! Marker backends that delegate every compute operation to [`CpuBackend`].
//!
//! Distinct ZST types let integration tests exercise heterogeneous
//! [`super::stack::ProverComputeStack`] wiring without standing up four separate
//! hardware backends.

use super::backend::{
    CompressionComputeBackend, CompressionRowsProducts, ComputeBackendSetup,
    CyclicRowsComputeBackend, DigitRowsComputeBackend,
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
use super::plans::RingSwitchRelationRows;
use crate::{CommitInnerWitness, DecomposeFoldWitness};
use akita_algebra::CyclotomicRing;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
};
use akita_types::{AkitaExpandedSetup, FpExtEncoding, NttCacheKey};
use std::sync::Arc;

macro_rules! delegate_compute_backend_setup {
    ($ty:ty) => {
        impl<F> ComputeBackendSetup<F> for $ty
        where
            F: FieldCore + CanonicalField,
        {
            type PreparedSetup = <CpuBackend as ComputeBackendSetup<F>>::PreparedSetup;

            fn prepare_expanded(
                &self,
                expanded: Arc<AkitaExpandedSetup<F>>,
            ) -> Result<Self::PreparedSetup, AkitaError> {
                CpuBackend::DEFAULT.prepare_expanded(expanded)
            }

            fn ensure_ntt_slot(
                &self,
                prepared: &Self::PreparedSetup,
                key: NttCacheKey,
            ) -> Result<(), AkitaError> {
                CpuBackend::DEFAULT.ensure_ntt_slot(prepared, key)
            }

            fn ntt_requirement_is_cached(
                &self,
                prepared: &Self::PreparedSetup,
                requirement: crate::compute::RoutedNttRequirement,
            ) -> Result<bool, AkitaError> {
                CpuBackend::DEFAULT.ntt_requirement_is_cached(prepared, requirement)
            }

            fn planned_ntt_cache_entry_bytes(
                &self,
                prepared: &Self::PreparedSetup,
                key: NttCacheKey,
            ) -> Result<usize, AkitaError> {
                CpuBackend::DEFAULT.planned_ntt_cache_entry_bytes(prepared, key)
            }

            fn release_built_ntt_slots(
                &self,
                prepared: &Self::PreparedSetup,
            ) -> Result<usize, AkitaError> {
                CpuBackend::DEFAULT.release_built_ntt_slots(prepared)
            }

            fn prepared_expanded_setup<'a>(
                &self,
                prepared: &'a Self::PreparedSetup,
            ) -> &'a AkitaExpandedSetup<F> {
                CpuBackend::DEFAULT.prepared_expanded_setup(prepared)
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
                CpuBackend::DEFAULT.digit_rows(prepared, row_len, digits, log_basis)
            }
        }
    };
}

macro_rules! delegate_compression {
    ($ty:ty) => {
        impl<F> CompressionComputeBackend<F> for $ty
        where
            F: FieldCore + CanonicalField,
        {
            fn compression_cache_bytes(&self, prepared: &Self::PreparedSetup) -> Option<usize> {
                CpuBackend::DEFAULT.compression_cache_bytes(prepared)
            }

            fn compression_rows_products<const D: usize>(
                &self,
                prepared: &Self::PreparedSetup,
                digit_vectors: &[&[[i8; D]]],
            ) -> Result<Vec<CompressionRowsProducts<F, D>>, AkitaError> {
                CpuBackend::DEFAULT.compression_rows_products(prepared, digit_vectors)
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
                CpuBackend::DEFAULT.cyclic_digit_rows(prepared, row_len, digits, log_basis)
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
                CpuBackend::DEFAULT.evaluate_and_fold(prepared, source, plan)
            }

            fn decompose_fold(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: S,
                plan: DecomposeFoldPlan<'_>,
            ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
                CpuBackend::DEFAULT.decompose_fold(prepared, source, plan)
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
                CpuBackend::DEFAULT.decompose_fold_batch(prepared, source, plan)
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
                CpuBackend::DEFAULT.column_partials(prepared, source, logical_point)
            }

            fn packed_witness(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: S,
            ) -> Result<super::kernels::TensorPackedWitness<E>, AkitaError> {
                CpuBackend::DEFAULT.packed_witness(prepared, source)
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
                CpuBackend::DEFAULT.root_projection(prepared, source)
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
                CpuBackend::DEFAULT.column_partials_batch(prepared, source, logical_point)
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
                CpuBackend::DEFAULT.sparse_linear_combination(prepared, source, coeffs)
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
            fn commit_inner_group(
                &self,
                prepared: &Self::PreparedSetup,
                sources: Vec<S>,
                plan: CommitInnerPlan,
            ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
                CpuBackend::DEFAULT.commit_inner_group(prepared, sources, plan)
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
                CpuBackend::DEFAULT.relation_rows(prepared, source, plan)
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
                CpuBackend::DEFAULT.quotient_rows(prepared, source, plan)
            }
        }
    };
}

/// Delegating commit-cluster marker backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct CommitCluster;

delegate_compute_backend_setup!(CommitCluster);
delegate_compression!(CommitCluster);
delegate_digit_rows!(CommitCluster);
delegate_cyclic_rows!(CommitCluster);
delegate_root_commit_kernel!(CommitCluster);

/// Delegating opening-cluster marker backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpeningCluster;

delegate_compute_backend_setup!(OpeningCluster);
delegate_compression!(OpeningCluster);
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
delegate_compression!(RingSwitchCluster);
delegate_digit_rows!(RingSwitchCluster);
delegate_ring_switch_kernels!(RingSwitchCluster);
