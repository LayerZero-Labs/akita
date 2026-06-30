use crate::backend::RootTensorProjectionPoly;
use crate::compute::backend::ComputeBackendSetup;
use crate::compute::operation_plans::{
    CommitInnerPlan, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningFoldOutput, OpeningFoldPlan,
    RingSwitchQuotientPlan, RingSwitchRelationPlan,
};
use crate::compute::plans::RingSwitchRelationRows;
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::{CommitInnerWitness, DecomposeFoldWitness};
use akita_algebra::CyclotomicRing;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
    MulBaseUnreduced,
};
use akita_types::FpExtEncoding;

/// Tensor-packed root witness alternatives produced by a tensor kernel.
///
/// This is an Akita-owned *output* sum type: the set of protocol output
/// alternatives is fixed, so an enum is the right model here. It is not a
/// closed *input-source* enum, which is the pattern the open boundary forbids.
#[derive(Debug, Clone)]
pub enum TensorPackedWitness<E: FieldCore> {
    /// Dense tensor-packed evaluations (universal fallback).
    Dense(Vec<E>),
    /// Sparse tensor-packed witness preserved when the source/backend can.
    Sparse(SparseExtensionOpeningWitness<E>),
}

/// Outcome of a batched decompose-fold kernel invocation.
#[derive(Debug)]
pub enum BatchDecomposeFoldOutcome<F: FieldCore, const D: usize> {
    /// Fused batched witness produced by the kernel.
    Fused(DecomposeFoldWitness<F>),
    /// No fused path; caller should decompose-fold each polynomial and aggregate.
    FallbackPerPoly,
    /// Batch shape or challenge plan is not supported.
    Unsupported,
}

/// Inner Ajtai commit kernel over a borrowed commit source view `S`.
///
/// `S` is the extensibility hook: a downstream crate defines its own commit
/// view and implements `RootCommitKernel<MyCommitView<'_>, F, D>` for a backend
/// (for example `CpuBackend`) without touching an Akita-owned enum. Built-in
/// Akita views reduce to the standard `*_commit_rows` helpers above.
pub trait RootCommitKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// Inner commitment that preserves the recomposed inner rows.
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup,
        source: S,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError>;
}

/// Fused ring-switch relation-rows kernel over a borrowed relation view `S`.
pub trait RingSwitchRelationKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// Fused D/B cyclic rows plus A-side quotient rows.
    fn relation_rows(
        &self,
        prepared: &Self::PreparedSetup,
        source: S,
        plan: RingSwitchRelationPlan,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField;
}

/// Additional public-row quotient kernel over a borrowed quotient view `S`.
pub trait RingSwitchQuotientKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// A-side quotient rows for one additional public-row segment.
    fn quotient_rows(
        &self,
        prepared: &Self::PreparedSetup,
        source: S,
        plan: RingSwitchQuotientPlan,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: HalvingField;
}

/// Opening fold / decompose-fold kernel over a borrowed opening view `S`.
///
/// `prepared` is optional because some opening folds do not need setup-owned
/// state; setup-dependent work stays explicitly tied to the backend context.
pub trait OpeningFoldKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// Fused fold + evaluation in one pass over the source.
    fn evaluate_and_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: OpeningFoldPlan<'_, F, D>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError>;

    /// Decompose + challenge-fold step.
    fn decompose_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError>;
}

/// Batched decompose-fold kernel over a borrowed opening-batch view `S`.
pub trait OpeningBatchKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// Fused batched decompose-fold at one opening point.
    fn decompose_fold_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError>;
}

/// Tensor projection kernel over a borrowed tensor view `S` for opening at an
/// extension-field point of type `E`.
pub trait TensorProjectionKernel<S, F, E, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    /// Tensor-column partials at one logical point.
    fn column_partials(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>;

    /// Tensor-packed root witness, dense or sparse when available.
    fn packed_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
    ) -> Result<TensorPackedWitness<E>, AkitaError>;

    /// Committed tensor-projected root polynomial.
    fn root_projection(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
    ) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
    where
        F: FromPrimitiveInt,
        E: FpExtEncoding<F>;
}

/// Batched tensor projection kernel over a borrowed tensor-batch view `S`.
pub trait TensorProjectionBatchKernel<S, F, E, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    /// Tensor-column partials for a same-point batch.
    fn column_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>;

    /// Sparse linear combination of tensor-packed root witnesses.
    ///
    /// Returns `Ok(None)` when a sparse combination is unavailable for the whole
    /// batch and the caller must fall back to dense materialization.
    fn sparse_linear_combination(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>;
}
