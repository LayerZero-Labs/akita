use crate::arithmetic::backend::ComputeBackendSetup;
use crate::opaque::operation_plans::{
    FoldProbeOutcome, RingSwitchRelationPlan, SubringCoefficientPackingPartials,
    SubringCoefficientPackingPlan, ValidatedFoldProbePlan, ValidatedTerminalFoldProbePlan,
    ValidatedTerminalZEncodingPlan,
};
use crate::opaque::plans::RingSwitchRelationRows;

use akita_error::AkitaError;

use jolt_field::{CanonicalEncoding, ExtField, Field};

/// Fused ring-switch relation-rows kernel over a borrowed relation view `S`.
pub(crate) trait RingSwitchRelationKernel<S, F, const D: usize>:
    ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    /// Fused D rows in both domains, B cyclic rows, and A-side quotient rows.
    fn relation_rows(
        &self,
        prepared: &Self::PreparedSetup,
        source: S,
        plan: RingSwitchRelationPlan,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>;
}

/// Backend-owned fold computation and admission over one validated plan.
pub(crate) trait FoldResponseKernel<S, F, const D: usize>: FoldHandleBackend<F>
where
    F: Field + CanonicalEncoding,
{
    fn probe(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: &ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<<Self as FoldHandleBackend<F>>::AcceptedFold>, AkitaError>;
}

/// Consumer-selected fold handle family shared by every runtime dimension.
///
/// Akita transports these handles but never embeds or unwraps a concrete CPU
/// representation. Dimension-specific kernels must return this family.
pub(crate) trait FoldHandleBackend<F>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    type AcceptedFold: crate::opaque::AcceptedFoldHandle;
    type AcceptedTerminalFold: Send + Sync + 'static;
}

/// Coefficient-packing projection over a borrowed same-shape source batch.
pub trait SubringCoefficientPackingBatchKernel<S, F, E, const D: usize>:
    ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
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

/// Backend-owned terminal fold computation, admission, and canonical encoding.
pub(crate) trait TerminalFoldResponseKernel<S, F, const D: usize>:
    FoldHandleBackend<F>
where
    F: Field + CanonicalEncoding,
{
    fn probe_terminal(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: &ValidatedTerminalFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<<Self as FoldHandleBackend<F>>::AcceptedTerminalFold>, AkitaError>;

    fn encode_terminal_z(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        fold: &<Self as FoldHandleBackend<F>>::AcceptedTerminalFold,
        plan: &ValidatedTerminalZEncodingPlan,
    ) -> Result<Vec<u8>, AkitaError>;
}

pub(crate) struct PreparedRelationWitness<H> {
    handle: H,
    column_bits: usize,
    coefficient_bits: usize,
}

impl<H> PreparedRelationWitness<H> {
    pub(crate) fn new(handle: H, column_bits: usize, coefficient_bits: usize) -> Self {
        Self {
            handle,
            column_bits,
            coefficient_bits,
        }
    }

    pub(crate) fn into_parts(self) -> (H, usize, usize) {
        (self.handle, self.column_bits, self.coefficient_bits)
    }
}
