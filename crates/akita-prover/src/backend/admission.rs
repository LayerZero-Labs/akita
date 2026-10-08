//! Admission checks before private proof work or transcript mutation.
use super::{ProofContext, ProofScopeConsumer, ProverHandleFamily};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::{FoldSchedule, GroupCommitPhaseParams, OpeningClaimsLayout};
use akita_types::{AkitaSetupDescriptor, Commitment};
use jolt_field::{CanonicalEncoding, Field};

pub trait ProofAdmission<F: Field + CanonicalEncoding, E: Field>:
    ProverHandleFamily<F, E> + ProofScopeConsumer
{
    /// Prepare one executor's proof session before any fold computation.
    /// Validate setup, the trusted schedule row and root layout, without
    /// requiring this executor to support every level of the schedule.
    fn prepare_executor<Cfg>(
        &self,
        setup: &AkitaSetupDescriptor,
        schedules: &TrustedScheduleCatalog<Cfg>,
        plan: &FoldSchedule,
        layout: &OpeningClaimsLayout,
    ) -> Result<Self::ProofSessionHandle, AkitaError>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>;

    /// Begin one assigned level, including its successor commitment work.
    fn begin_fold(
        &self,
        session: &Self::ProofSessionHandle,
        requirements: &super::FoldExecutionRequirements<'_>,
    ) -> Result<(), AkitaError>;

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
