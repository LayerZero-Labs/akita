use crate::compute::backend::ComputeBackendSetup;
use crate::compute::operation_plans::{
    CommitInnerPlan, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningFoldOutput, OpeningFoldPlan,
    RingSwitchRelationPlan, SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};
use crate::compute::plans::RingSwitchRelationRows;
use crate::{CommitInnerWitness, DecomposeFoldWitness};
use akita_error::AkitaError;
use akita_field::{CanonicalField, ExtField, FieldCore, HalvingField, MulBaseUnreduced};

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
    /// Inner commitments for a same-shape group of sources.
    ///
    /// Every source of a committed group multiplies the same commit matrix,
    /// so kernels can stream the matrix once for the whole group. Results are
    /// returned per source in input order.
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<S>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError>;
}

/// Fused ring-switch relation-rows kernel over a borrowed relation view `S`.
pub trait RingSwitchRelationKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// Fused D rows in both domains, B cyclic rows, and A-side quotient rows.
    fn relation_rows(
        &self,
        prepared: &Self::PreparedSetup,
        source: S,
        plan: RingSwitchRelationPlan,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
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
        plan: OpeningFoldPlan<'_, F>,
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

    /// Tensor-packed recursive suffix witness.
    fn packed_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
    ) -> Result<Vec<E>, AkitaError>;
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
}

/// Coefficient-packing projection over a borrowed same-shape source batch.
pub trait SubringCoefficientPackingBatchKernel<S, F, E, const D: usize>:
    ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    /// Return one canonical base-field partial buffer per claim.
    ///
    /// Every returned buffer uses
    /// `[block][extension coordinate][subring coefficient]` order.
    fn coefficient_packing_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError>;
}
