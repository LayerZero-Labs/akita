//! Commitment execution and persistence admission at the owning backend boundary.
use super::owned::{CommitOutput, CommitmentHandle, CommittedSource, SourceHandle};
use crate::commitment::{CommitmentExecutor, GroupContext, PortableStatePolicy};
use crate::opaque::CpuBackend;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::FpExtEncoding;
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::sync::Arc;

impl<F, E> CpuBackend<F, E>
where
    F: Field
        + CanonicalEncoding
        + AkitaSerialize
        + jolt_field::Ring
        + Unreduced
        + WithCommitAccumulator
        + 'static,
    F::Wide: From<F> + jolt_field::AdditiveGroup,
    E: jolt_field::ExtField<F>
        + FpExtEncoding<F>
        + jolt_field::MulBaseUnreduced<F>
        + Unreduced
        + jolt_field::Fold
        + AkitaSerialize
        + 'static,
{
    /// Import a retained commitment from another backend after validating its
    /// source and complete commitment material against this backend's setup.
    pub fn import_commitment(
        &self,
        foreign: &CommitmentHandle<F, E>,
    ) -> Result<CommitmentHandle<F, E>, AkitaError> {
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
                // Recomputation above proves the same source and commitment,
                // so the admitting contract carries over unchanged.
                producer_contract: committed.producer_contract,
                public: committed.public.clone(),
                retained,
            }),
        })
    }
}

impl<F, E> CpuBackend<F, E>
where
    F: Field + CanonicalEncoding + AkitaSerialize + Unreduced + WithCommitAccumulator + 'static,
    F::Wide: From<F>,
    E: FpExtEncoding<F> + 'static,
{
    /// Commit an imported immutable source under the producer schedule family
    /// `family`, retaining its exact source and parameters.
    ///
    /// `family` supplies the catalog row for scheduler contexts and the
    /// declared committed-source contract that admits the source. The handle
    /// records that contract. One backend can commit groups from different
    /// families that share `F` and `E`.
    pub fn commit<Cfg>(
        &self,
        family: &TrustedScheduleCatalog<Cfg>,
        source: &SourceHandle<F, E>,
        context: GroupContext<'_>,
    ) -> Result<CommitOutput<F, E>, AkitaError>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
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
        let (profile, producer_contract) = crate::commitment::resolve_commit_params::<Cfg, _>(
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
                producer_contract,
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
    use akita_prover::CommitmentHandleMetadata;
    use akita_types::{AkitaScheduleLookupKey, OpeningClaimsLayout};
    use jolt_field::Ring;

    type Cfg = fp64::Dense;
    type F = <Cfg as CommitmentConfig>::Field;
    type E = <Cfg as CommitmentConfig>::ExtField;

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
                        .matrix_capacity();
                let setup =
                    AkitaProverSetup::<F>::generate_with_capacity(NUM_VARS, 1, capacity).unwrap();
                let backend = CpuBackend::<F, E>::new(setup.expanded.clone()).unwrap();
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
                    .commit(&schedules, &source, GroupContext::explicit(&profile))
                    .unwrap();

                assert_eq!(actual.committed_group, expected.committed_group);
                assert_eq!(
                    actual.private_handle.committed.retained,
                    expected.prover_state
                );

                let receiving_backend = CpuBackend::<F, E>::new(setup.expanded.clone()).unwrap();
                let transferred = receiving_backend
                    .import_commitment(&actual.private_handle)
                    .unwrap();
                assert_eq!(transferred.committed.retained, expected.prover_state);
                assert_eq!(
                    actual.private_handle.producer_contract(),
                    Cfg::committed_source_contract().unwrap()
                );
                assert_eq!(
                    transferred.producer_contract(),
                    actual.private_handle.producer_contract()
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
