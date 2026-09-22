//! Representation-free contract between protocol orchestration and prover backends.

mod admission;
mod context;
mod handles;
mod identity;
mod messages;
mod metadata;
mod opening;
mod plans;
mod scope;
mod stage3;
mod traits;

pub use admission::ProofAdmission;
pub use context::OperationCtx;
pub use handles::{
    AcceptedFoldHandle, CommitmentHandleMetadata, OpeningSource, RecursiveWitnessHandle,
    RecursiveWitnessManifest, RelationWitnessMetadata, SourceMetadata,
};
pub use identity::ProofContext;
pub use messages::{
    FoldProbeDiagnostics, FoldProbeOutcome, NextWitnessBindingMessage, PreparedGroupOpening,
    PreparedRelationHandle, RecursiveWitnessBuildStart, RecursiveWitnessFoldInput,
    RecursiveWitnessPublicInputs, TerminalTFieldsMessage, WitnessCommitmentOutput,
};
pub use metadata::{
    AcceptedFoldMetadata, CommitmentMaterialMetadata, CommitmentRelationMaterial,
    PreparedRelationGroupPublic, RelationWitnessFinalClaims, Stage1FinalClaims,
    Stage1PublicTransition, Stage1RoundPolynomial, Stage1Step, Stage1Transition,
    TerminalCommitmentMaterialKernel,
};
pub use opening::{EorGroupRequest, OpaqueEorKernel, OpaqueOpeningKernel, PreparedEor};
pub use plans::{
    EvaluationTraceDescription, FoldProbeGeometry, PhysicalL2WeightRequest, RelationWeightRequest,
    Stage2OpeningDescription, ValidatedFoldAcceptancePlan, ValidatedFoldProbePlan,
    ValidatedRecursiveGroupOpeningPlan, ValidatedRecursiveWitnessCommitPlan,
    ValidatedRecursiveWitnessPlan, ValidatedRelationSessionPlan, ValidatedRelationWitnessPlan,
    ValidatedStage1Plan, ValidatedTerminalFoldProbePlan, ValidatedTerminalZEncodingPlan,
    WitnessCommitmentParameters,
};
pub use scope::{ProofScope, ProofScopeConsumer, ProofScopeId};
pub use stage3::{OpaqueStage3Kernel, Stage3Request};
pub use traits::{
    OpaqueProverConsumer, OpaqueRecursiveWitnessBuildKernel, OpaqueRelationWitnessKernel,
    OpaqueResourceReleaseKernel, OpaqueStage1Kernel, OpaqueStage2Kernel, OpaqueTerminalFoldKernel,
    OpaqueWitnessCommitKernel, OpaqueWitnessOpeningKernel, ProverHandleFamily,
};

/// Complete protocol capability implemented by a concrete prover backend.
pub trait ProverBackend<F: jolt_field::Field + jolt_field::CanonicalEncoding, E: jolt_field::Field>:
    ProofAdmission<F, E>
    + OpaqueOpeningKernel<F, E>
    + OpaqueEorKernel<F, E>
    + OpaqueRecursiveWitnessBuildKernel<F, E>
    + OpaqueWitnessCommitKernel<F, E>
    + OpaqueRelationWitnessKernel<F, E>
    + OpaqueStage1Kernel<F, E>
    + OpaqueStage2Kernel<F, E>
    + OpaqueStage3Kernel<F, E>
    + OpaqueTerminalFoldKernel<F, E>
    + OpaqueWitnessOpeningKernel<F, E>
    + OpaqueResourceReleaseKernel<F, E>
    + TerminalCommitmentMaterialKernel<F, Self::CommitmentMaterialHandle>
{
}

impl<F, E, B> ProverBackend<F, E> for B
where
    F: jolt_field::Field + jolt_field::CanonicalEncoding,
    E: jolt_field::Field,
    B: ProofAdmission<F, E>
        + OpaqueOpeningKernel<F, E>
        + OpaqueEorKernel<F, E>
        + OpaqueRecursiveWitnessBuildKernel<F, E>
        + OpaqueWitnessCommitKernel<F, E>
        + OpaqueRelationWitnessKernel<F, E>
        + OpaqueStage1Kernel<F, E>
        + OpaqueStage2Kernel<F, E>
        + OpaqueStage3Kernel<F, E>
        + OpaqueTerminalFoldKernel<F, E>
        + OpaqueWitnessOpeningKernel<F, E>
        + OpaqueResourceReleaseKernel<F, E>
        + TerminalCommitmentMaterialKernel<F, B::CommitmentMaterialHandle>,
{
}
