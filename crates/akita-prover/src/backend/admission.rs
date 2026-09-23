//! Admission checks before private proof work or transcript mutation.
use super::{ProofContext, ProofScopeConsumer, ProverHandleFamily};
use akita_error::AkitaError;
use akita_types::{
    AkitaSetupDescriptor, Commitment, FoldSchedule, GroupCommitPhaseParams, OpeningClaimsLayout,
};
use jolt_field::{CanonicalEncoding, Field};

pub trait ProofAdmission<F: Field + CanonicalEncoding, E: Field>:
    ProverHandleFamily<F, E> + ProofScopeConsumer
{
    fn begin_proof(
        &self,
        setup: &AkitaSetupDescriptor,
        plan: &FoldSchedule,
        layout: &OpeningClaimsLayout,
    ) -> Result<Self::ProofSessionHandle, AkitaError>;
    fn proof_context(
        &self,
        session: &Self::ProofSessionHandle,
        level: u32,
    ) -> Result<ProofContext, AkitaError>;
    /// Validate retained source, exact public commitment and parameters together.
    fn validate_commitment(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        handle: &Self::CommitmentHandle,
        parameters: &GroupCommitPhaseParams,
        commitment: &Commitment<F>,
    ) -> Result<Self::CommitmentMaterialHandle, AkitaError>;
}
