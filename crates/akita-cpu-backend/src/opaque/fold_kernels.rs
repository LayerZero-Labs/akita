//! Backend-private fold kernels and witness-derived outputs.

use crate::opaque::{
    ComputeBackendSetup, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningFoldPlan,
    ValidatedFoldRelationPlan,
};
use crate::opaque::{CpuFoldResponses, DecomposeFoldWitness};
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::RelationRowGeometry;
use jolt_field::{CanonicalEncoding, Field};

/// Opening fold / decompose-fold kernel over a borrowed opening view `S`.
///
/// `prepared` is optional because some opening folds do not need setup-owned
/// state; setup-dependent work stays explicitly tied to the backend context.
pub(crate) trait OpeningFoldKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
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
    ) -> Result<DecomposeFoldWitness, AkitaError>;
}

/// Batched decompose-fold kernel over a borrowed opening-batch view `S`.
///
/// Implementations return the final aggregate witness. A backend without a
/// fused path is responsible for folding its source polynomials individually
/// and aggregating them before returning.
pub(crate) trait OpeningBatchKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    /// Batched decompose-fold at one opening point.
    fn decompose_fold_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<CpuFoldResponses, AkitaError>;
}

/// Private A-side relation work over a backend-owned accepted fold handle.
pub(crate) trait FoldRelationKernel<H, F, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    fn a_relation_from_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        fold: &H,
        plan: &ValidatedFoldRelationPlan<'_, F>,
    ) -> Result<FoldRelationOutput<F>, AkitaError>;
}

/// Fused evaluate-and-fold output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OpeningFoldOutput<F: Field, const D: usize> {
    /// Evaluation of the polynomial at the opening point.
    pub(crate) eval: CyclotomicRing<F, D>,
    /// Folded witness rows in ring form.
    pub(crate) folded: Vec<CyclotomicRing<F, D>>,
}

/// One private relation row derived from an opaque accepted fold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RelationQuotientRow<F: Field> {
    geometry: RelationRowGeometry,
    coefficients: Vec<F>,
}

impl<F: Field> RelationQuotientRow<F> {
    pub(crate) fn new(
        geometry: RelationRowGeometry,
        coefficients: Vec<F>,
    ) -> Result<Self, AkitaError> {
        if coefficients.len() != geometry.physical_coefficient_width() {
            return Err(AkitaError::InvalidSize {
                expected: geometry.physical_coefficient_width(),
                actual: coefficients.len(),
            });
        }
        Ok(Self {
            geometry,
            coefficients,
        })
    }

    /// Canonical public row geometry.
    pub(crate) const fn geometry(&self) -> RelationRowGeometry {
        self.geometry
    }

    /// Private quotient coefficients in physical row order.
    pub(crate) fn coefficients(&self) -> &[F] {
        &self.coefficients
    }

    pub(crate) fn coeffs(&self) -> &[F] {
        &self.coefficients
    }
}

/// Opening-family-typed private output derived from an opaque accepted fold.
pub(crate) enum FoldRelationOutput<F: Field> {
    EvaluationTrace {
        a_quotients: Vec<RelationQuotientRow<F>>,
        z_consistency_high_half: RelationQuotientRow<F>,
    },
    SubringCoefficientPacking {
        a_quotients: Vec<RelationQuotientRow<F>>,
    },
}
