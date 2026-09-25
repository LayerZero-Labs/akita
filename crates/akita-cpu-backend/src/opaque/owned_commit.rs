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
        ForeignCfg: CommitmentConfig<Field = Cfg::Field, ExtField = Cfg::ExtField>,
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
        let (recomputed, retained) =
            executor.execute_full_via_outer_image(self, committed.parameters, &sources)?;
        if recomputed.commitment != committed.public || retained != committed.retained {
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
        self.commit_in_family(self.schedules()?, source, context)
    }

    /// Commit a source on this backend under another schedule family's commit
    /// parameters and producer contract.
    ///
    /// `family` selects the commit profile (a catalog row, or the explicit
    /// profile in `context`) and supplies the source class and coefficient
    /// interval that admission enforces, exactly as `CpuBackend<FamilyCfg>`
    /// would. The commitment is computed once with this backend's prepared
    /// setup and caches. The returned handle is owned by this backend, so it
    /// can be a precommitted group of this backend's openings.
    ///
    /// The handle equals what [`Self::import_commitment`] returns for the same
    /// source committed by a `CpuBackend<FamilyCfg>`, without the second
    /// commitment or the second backend. Use `import_commitment` for handles
    /// produced elsewhere: its recomputation is what checks a foreign setup
    /// against this one.
    pub fn commit_in_family<FamilyCfg>(
        &self,
        family: &akita_config::TrustedScheduleCatalog<FamilyCfg>,
        source: &SourceHandle<Cfg::Field, Cfg::ExtField, Cfg>,
        context: GroupContext<'_>,
    ) -> Result<CommitOutput<Cfg::Field, Cfg::ExtField, Cfg>, AkitaError>
    where
        FamilyCfg: CommitmentConfig<Field = Cfg::Field, ExtField = Cfg::ExtField>,
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
        executor.validate_setup(&prepared.expanded)?;
        let profile = crate::commitment::resolve_commit_params::<FamilyCfg, _>(
            &sources,
            &prepared.expanded,
            family,
            context,
        )?;
        let (committed_group, prover_state) =
            executor.execute_full_via_outer_image(self, profile, &sources)?;
        let private_handle = CommitmentHandle {
            owner: source.owner,
            committed: Arc::new(CommittedSource {
                commitment_id: self.owner().next_operation_id()?,
                source: source.storage.clone(),
                metadata: source.metadata,
                parameters: *committed_group.profile(),
                public: committed_group.commitment().clone(),
                retained: prover_state,
            }),
        };
        Ok(CommitOutput {
            committed_group,
            private_handle,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AkitaProverSetup, DensePoly};
    use akita_config::proof_optimized::fp64;
    use akita_types::{AkitaScheduleLookupKey, OpeningClaimsLayout};
    use jolt_field::Ring;

    type Cfg = fp64::Dense;
    type F = <Cfg as CommitmentConfig>::Field;

    #[test]
    fn cpu_commit_matches_full_executor_state_after_outer_image_completion() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                const NUM_VARS: usize = 14;
                let schedules =
                    akita_config::test_support::workspace_schedule_catalog::<Cfg>().unwrap();
                let layout = OpeningClaimsLayout::new(NUM_VARS, 1).unwrap();
                let key = AkitaScheduleLookupKey::single(layout.root_final_group_layout().unwrap());
                let profile = schedules.resolve_key(&key).unwrap().profiles().final_group;
                let capacity =
                    akita_config::SetupRequirements::from_catalog::<Cfg>(&schedules, NUM_VARS, 1)
                        .unwrap()
                        .matrix_capacity;
                let setup =
                    AkitaProverSetup::<F>::generate_with_capacity(NUM_VARS, 1, capacity).unwrap();
                let backend = CpuBackend::<Cfg>::new(setup.expanded.clone(), &schedules).unwrap();
                let prepared = backend.prepared().unwrap();
                let executor = CommitmentExecutor::cpu(
                    &backend,
                    prepared,
                    setup.expanded.as_ref(),
                    Vec::new(),
                    PortableStatePolicy,
                )
                .unwrap();
                let evals = (0..1usize << NUM_VARS)
                    .map(|index| F::from_u64(index as u64 + 1))
                    .collect::<Vec<_>>();
                let poly = DensePoly::<F>::from_field_evals(NUM_VARS, &evals).unwrap();
                let expected = crate::commitment::commit::<Cfg, _, _>(
                    std::slice::from_ref(&poly),
                    setup.expanded.as_ref(),
                    &schedules,
                    &executor,
                    GroupContext::explicit(&profile),
                )
                .unwrap();
                let source = backend.import_source(vec![poly]).unwrap();
                let actual = backend
                    .commit(&source, GroupContext::explicit(&profile))
                    .unwrap();

                assert_eq!(actual.committed_group, expected.committed_group);
                assert_eq!(
                    actual.private_handle.committed.retained,
                    expected.prover_state
                );

                let receiving_backend =
                    CpuBackend::<Cfg>::new(setup.expanded.clone(), &schedules).unwrap();
                let transferred = receiving_backend
                    .import_commitment(&actual.private_handle)
                    .unwrap();
                assert_eq!(transferred.committed.retained, expected.prover_state);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
