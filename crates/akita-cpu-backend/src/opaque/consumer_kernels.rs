//! Private execution contracts used after consumer binding validation.
use crate::opaque::{
    ComputeBackendSetup, FoldHandleBackend, FoldProbeOutcome, PreparedRelationWitness,
    PreparedWitnessOpening, ProverHandleFamily, ValidatedFoldProbePlan,
    ValidatedRelationWitnessPlan, ValidatedTerminalFoldProbePlan, ValidatedTerminalZEncodingPlan,
};
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};

/// Backend-associated preparation of the consumer-owned Stage 1/2 witness.
pub(crate) trait RecursiveRelationWitnessKernel<H, F>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    type RelationWitness: Send + 'static;

    fn prepare_relation_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: &H,
        plan: &ValidatedRelationWitnessPlan,
    ) -> Result<PreparedRelationWitness<Self::RelationWitness>, AkitaError>;
}

pub(crate) trait CpuWitnessOpeningKernel<F, E>:
    ProverHandleFamily<F, E> + ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    type WitnessOpeningHandle;
    type WitnessEorSessionHandle;
    fn prepare_witness_opening(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness_handle: &Self::WitnessHandle,
        plan: &crate::opaque::ValidatedWitnessOpeningPlan<'_, E>,
    ) -> Result<PreparedWitnessOpening<E, Self::WitnessOpeningHandle>, AkitaError>;

    fn begin_witness_eor(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness_handle: &Self::WitnessHandle,
        opening_handle: Self::WitnessOpeningHandle,
        plan: &crate::opaque::ValidatedWitnessEorPlan<'_, E>,
    ) -> Result<Self::WitnessEorSessionHandle, AkitaError>;
}

/// Fold probing over an opaque recursive-witness handle.
pub(crate) trait RecursiveWitnessFoldKernel<H, F, const D: usize>:
    FoldHandleBackend<F>
where
    F: Field + CanonicalEncoding,
{
    fn probe_recursive_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: &H,
        plan: &ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<<Self as FoldHandleBackend<F>>::AcceptedFold>, AkitaError>;
}

/// Terminal fold probing and canonical encoding over an opaque witness handle.
pub(crate) trait RecursiveWitnessTerminalFoldKernel<H, F, const D: usize>:
    FoldHandleBackend<F>
where
    F: Field + CanonicalEncoding,
{
    fn probe_terminal_recursive_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: &H,
        plan: &ValidatedTerminalFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<<Self as FoldHandleBackend<F>>::AcceptedTerminalFold>, AkitaError>;

    fn encode_terminal_recursive_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        fold: &<Self as FoldHandleBackend<F>>::AcceptedTerminalFold,
        plan: &ValidatedTerminalZEncodingPlan,
    ) -> Result<Vec<u8>, AkitaError>;
}
