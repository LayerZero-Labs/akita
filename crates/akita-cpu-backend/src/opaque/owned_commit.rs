//! Commitment execution and persistence admission at the owning backend boundary.
use super::owned::{CommitOutput, CommitmentHandle, CommittedSource, SourceHandle};
use crate::commitment::{CommitmentExecutor, GroupContext, PortableStatePolicy};
use crate::opaque::CpuBackend;
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::FpExtEncoding;
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::sync::Arc;

impl CpuBackend {
    /// Import a retained commitment from another backend after validating its
    /// source and complete commitment material against this backend's setup.
    pub fn import_commitment<F, E>(
        &self,
        foreign: &CommitmentHandle<F, E>,
    ) -> Result<CommitmentHandle<F, E>, AkitaError>
    where
        F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator + 'static,
        E: Field + 'static,
    {
        self.validate_extension::<E>()?;
        let prepared = self.prepared::<F>()?;
        let committed = &foreign.committed;
        let sources = committed.source.commitment_sources();
        let layout =
            crate::commitment::resolve_polynomial_group_layout(&sources, &prepared.expanded)?;
        if layout != committed.parameters.group {
            return Err(AkitaError::InvalidInput(
                "transferred source differs from commitment parameters".into(),
            ));
        }
        let executor = CommitmentExecutor::cpu(
            self,
            prepared,
            &prepared.expanded,
            Vec::new(),
            PortableStatePolicy,
        )?;
        let plan = crate::commitment::CommitmentExecutionPlan::for_root(&committed.parameters)?;
        let (payload, retained) = executor.execute_full(&plan, &sources)?.into_parts();
        if akita_types::Commitment::new(payload) != committed.public
            || retained != committed.retained
        {
            return Err(AkitaError::InvalidInput(
                "transferred commitment does not match backend setup".into(),
            ));
        }
        Ok(CommitmentHandle {
            owner: self.owner_id(),
            committed: Arc::new(CommittedSource {
                commitment_id: self.owner().next_operation_id()?,
                source: committed.source.clone(),
                metadata: committed.metadata,
                parameters: committed.parameters,
                public: committed.public.clone(),
                retained,
            }),
        })
    }

    /// Commit an imported immutable source and retain its exact source and parameters.
    pub fn commit<Cfg>(
        &self,
        source: &SourceHandle<Cfg::Field, Cfg::ExtField>,
        context: GroupContext<'_>,
    ) -> Result<CommitOutput<Cfg::Field, Cfg::ExtField>, AkitaError>
    where
        Cfg: CommitmentConfig + 'static,
        Cfg::Field: Field
            + CanonicalEncoding
            + AkitaSerialize
            + Unreduced
            + WithCommitAccumulator
            + 'static,
        <Cfg::Field as Unreduced>::Wide: From<Cfg::Field>,
        Cfg::ExtField: FpExtEncoding<Cfg::Field> + 'static,
    {
        self.validate_config::<Cfg>()?;
        if source.owner != self.owner_id() {
            return Err(AkitaError::InvalidInput(
                "source belongs to another backend".into(),
            ));
        }
        let prepared = self.prepared::<Cfg::Field>()?;
        let executor = CommitmentExecutor::cpu(
            self,
            prepared,
            &prepared.expanded,
            Vec::new(),
            PortableStatePolicy,
        )?;
        let sources = source.storage.commitment_sources();
        let output = crate::commitment::commit::<Cfg, _, _>(
            &sources,
            &prepared.expanded,
            self.schedules::<Cfg>()?,
            &executor,
            context,
        )?;
        let private_handle = CommitmentHandle {
            owner: source.owner,
            committed: Arc::new(CommittedSource {
                commitment_id: self.owner().next_operation_id()?,
                source: source.storage.clone(),
                metadata: source.metadata,
                parameters: *output.committed_group.profile(),
                public: output.committed_group.commitment().clone(),
                retained: output.prover_state,
            }),
        };
        Ok(CommitOutput {
            committed_group: output.committed_group,
            private_handle,
        })
    }
}
