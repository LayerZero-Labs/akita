//! Backend-neutral protocol sequencing, transcript handling, and proof assembly.
//!
//! Arithmetic implementations own their source storage and mutable proof state.
//! This crate exchanges opaque handles and scheduled public messages with them.

pub mod backend;
mod opening;
pub mod protocol;
mod setup;

pub use backend::{
    AcceptedFoldHandle, CommitmentHandleMetadata, CommitmentMaterialMetadata,
    CommitmentRelationMaterial, NextWitnessBindingMessage, OpaqueEorKernel, OpaqueOpeningKernel,
    OpaqueProverConsumer, OpaqueRecursiveWitnessBuildKernel, OpaqueRelationWitnessKernel,
    OpaqueResourceReleaseKernel, OpaqueStage1Kernel, OpaqueStage2Kernel, OpaqueStage3Kernel,
    OpaqueTerminalFoldKernel, OpaqueWitnessCommitKernel, OpaqueWitnessOpeningKernel, OpeningSource,
    OperationCtx, PreparedGroupOpening, PreparedRelationHandle, ProofAdmission, ProofContext,
    ProofScope, ProofScopeConsumer, ProofScopeId, ProverBackend, ProverHandleFamily,
    RecursiveWitnessBuildStart, RecursiveWitnessFoldInput, RecursiveWitnessHandle,
    RecursiveWitnessManifest, RecursiveWitnessPublicInputs, RelationWitnessMetadata,
    SourceMetadata, TerminalCommitmentMaterialKernel, TerminalTFieldsMessage,
    WitnessCommitmentOutput,
};
pub use opening::{ProverOpeningData, SelectedProverOpeningData};
pub use protocol::{
    batched_prove, ProveLevelOutput, RecursiveSuffixOutcome, RingRelationInstance,
    RingRelationProver, RingSwitchOutput, SuffixProverState,
};
pub use setup::{PreparedSetupPrefix, SetupPrefixKernel, SetupPrefixProverRegistry};
