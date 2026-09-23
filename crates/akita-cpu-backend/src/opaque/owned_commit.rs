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

impl<Cfg: CommitmentConfig> CpuBackend<Cfg> {
    /// Import a retained commitment from another backend after validating its
    /// source and complete commitment material against this backend's setup.
    pub fn import_commitment<ForeignCfg>(
        &self,
        foreign: &CommitmentHandle<Cfg::Field, Cfg::ExtField, ForeignCfg>,
    ) -> Result<CommitmentHandle<Cfg::Field, Cfg::ExtField, Cfg>, AkitaError>
    where
        ForeignCfg: CommitmentConfig<Field = Cfg::Field>,
        Cfg::Field: Field
            + CanonicalEncoding
            + AkitaSerialize
            + jolt_field::Ring
            + Unreduced
            + WithCommitAccumulator
            + 'static,
        <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + jolt_field::AdditiveGroup,
        Cfg::ExtField: jolt_field::ExtField<Cfg::Field>
            + FpExtEncoding<Cfg::Field>
            + jolt_field::MulBaseUnreduced<Cfg::Field>
            + Unreduced
            + jolt_field::Fold
            + AkitaSerialize
            + 'static,
    {
        self.validate_extension::<Cfg::ExtField>()?;
        let prepared = self.prepared()?;
        let committed = &foreign.committed;
        let source = self.import_source(committed.source.dense_polynomials()?)?;
        let sources = source.storage.commitment_sources();
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
                source: source.storage,
                metadata: committed.metadata,
                parameters: committed.parameters,
                public: committed.public.clone(),
                retained,
            }),
        })
    }

    /// Commit an imported immutable source and retain its exact source and parameters.
    pub fn commit(
        &self,
        source: &SourceHandle<Cfg::Field, Cfg::ExtField, Cfg>,
        context: GroupContext<'_>,
    ) -> Result<CommitOutput<Cfg::Field, Cfg::ExtField, Cfg>, AkitaError>
    where
        Cfg::Field: Field
            + CanonicalEncoding
            + AkitaSerialize
            + Unreduced
            + WithCommitAccumulator
            + 'static,
        <Cfg::Field as Unreduced>::Wide: From<Cfg::Field>,
        Cfg::ExtField: FpExtEncoding<Cfg::Field> + 'static,
    {
        if source.owner != self.owner_id() {
            return Err(AkitaError::InvalidInput(
                "source belongs to another backend".into(),
            ));
        }
        let prepared = self.prepared()?;
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
            self.schedules()?,
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
