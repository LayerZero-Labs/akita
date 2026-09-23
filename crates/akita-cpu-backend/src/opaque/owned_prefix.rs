//! Validated conversion from portable setup artifacts to current-owner handles.
use super::owned::{CommitmentHandle, CommittedSource, OwnedPolynomials};
use crate::commitment::{CommitmentExecutor, DenseType, PolynomialType, PortableStatePolicy};
use crate::{CpuBackend, DensePoly};
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_prover::{PreparedSetupPrefix, SetupPrefixProverRegistry};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::{Commitment, FpExtEncoding, SetupPrefixSlotId};
use jolt_field::{
    AdditiveGroup, CanonicalEncoding, ExtField, Field, Fold, MulBaseUnreduced, Ring, Unreduced,
    WithCommitAccumulator,
};
use std::sync::Arc;

/// Setup-prefix material memoized on the owning backend.
///
/// Both members depend only on the owned setup and the slot id. The retained
/// source is shared by `Arc`, so the prefix coefficients and their digit-plane
/// cache survive across proofs instead of being rebuilt per prove call.
pub(super) struct CachedSetupPrefix<F: Field> {
    artifact: crate::commitment::SetupPrefixSlot<F>,
    source: Arc<OwnedPolynomials<DensePoly<F>>>,
}

impl<Cfg: CommitmentConfig> CpuBackend<Cfg> {
    fn setup_prefix_source<F>(
        &self,
        id: &SetupPrefixSlotId,
    ) -> Result<Arc<OwnedPolynomials<DensePoly<F>>>, AkitaError>
    where
        Cfg: CommitmentConfig<Field = F>,
        F: Field + CanonicalEncoding + 'static,
    {
        let prepared = self.prepared()?;
        let coefficients = prepared
            .expanded
            .shared_matrix()
            .as_field_slice()
            .get(..id.n_prefix()?)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup prefix exceeds backend capacity".into())
            })?
            .to_vec();
        let polynomial =
            DensePoly::from_field_evals(id.commitment_profile.group.num_vars(), coefficients)?;
        Ok(Arc::new(OwnedPolynomials {
            polynomials: vec![polynomial],
        }))
    }

    /// Return the commitment artifact and retained source for one setup prefix.
    ///
    /// Export and import deliberately share this path: profiles commonly export
    /// an artifact during setup and import it immediately before proving, so a
    /// second derivation in `import_setup_prefixes` would put setup work on the
    /// timed prover path.
    fn setup_prefix_material<F>(
        &self,
        id: &SetupPrefixSlotId,
    ) -> Result<Arc<CachedSetupPrefix<F>>, AkitaError>
    where
        Cfg: CommitmentConfig<Field = F>,
        F: Field + CanonicalEncoding + Valid + Unreduced + WithCommitAccumulator + 'static,
    {
        let prepared = self.prepared()?;
        let validated =
            crate::setup::setup_prefix::validate_setup_prefix_commitment(&prepared.expanded, id)?;
        self.memoized_setup_prefix(id, || {
            let executor = CommitmentExecutor::cpu(
                self,
                prepared,
                &prepared.expanded,
                vec![PolynomialType::Dense(DenseType::Coefficients)],
                PortableStatePolicy,
            )?;
            let artifact = crate::setup::setup_prefix::commit_validated_setup_prefix(
                &prepared.expanded,
                &executor,
                validated,
            )?;
            Ok(CachedSetupPrefix {
                artifact,
                source: self.setup_prefix_source(id)?,
            })
        })
    }

    /// Adopt material already committed by a CPU backend in this process.
    fn cache_validated_setup_prefix<F>(
        &self,
        artifact: &crate::commitment::SetupPrefixSlot<F>,
    ) -> Result<Arc<CachedSetupPrefix<F>>, AkitaError>
    where
        Cfg: CommitmentConfig<Field = F>,
        F: Field + CanonicalEncoding + 'static,
    {
        let prepared = self.prepared()?;
        crate::setup::setup_prefix::validate_setup_prefix_commitment(
            &prepared.expanded,
            &artifact.id,
        )?;
        self.memoized_setup_prefix(&artifact.id, || {
            Ok(CachedSetupPrefix {
                artifact: artifact.clone(),
                source: self.setup_prefix_source(&artifact.id)?,
            })
        })
    }

    fn issue_setup_prefix_handle<F, E>(
        &self,
        id: &SetupPrefixSlotId,
        cached: Arc<CachedSetupPrefix<F>>,
    ) -> Result<PreparedSetupPrefix<F, CommitmentHandle<F, E, Cfg>>, AkitaError>
    where
        Cfg: CommitmentConfig<Field = F>,
        F: Field
            + CanonicalEncoding
            + AkitaSerialize
            + Valid
            + Ring
            + Unreduced
            + WithCommitAccumulator
            + 'static,
        F::Wide: From<F> + AdditiveGroup,
        E: ExtField<F>
            + FpExtEncoding<F>
            + MulBaseUnreduced<F>
            + Unreduced
            + Fold
            + AkitaSerialize
            + 'static,
    {
        let artifact = &cached.artifact;
        let public_commitment = artifact
            .commitment
            .rows
            .first()
            .ok_or(AkitaError::InvalidProof)?
            .clone();
        let committed = Arc::new(CommittedSource {
            commitment_id: self.owner().next_operation_id()?,
            source: cached.source.clone(),
            metadata: akita_prover::SourceMetadata::try_new(
                1,
                id.commitment_profile.group.num_vars(),
            )?,
            parameters: id.commitment_profile,
            public: Commitment::new(public_commitment),
            retained: artifact.hint.clone(),
        });
        Ok(PreparedSetupPrefix {
            public: artifact.verifier_slot(),
            commitment_handle: CommitmentHandle {
                owner: self.owner_id(),
                committed,
            },
        })
    }

    /// Build portable setup-prefix artifacts for application-managed persistence.
    pub fn export_setup_prefixes<F>(
        &self,
        ids: &[SetupPrefixSlotId],
    ) -> Result<crate::commitment::SetupPrefixProverRegistry<F>, AkitaError>
    where
        Cfg: CommitmentConfig<Field = F>,
        F: Field + CanonicalEncoding + Valid + Unreduced + WithCommitAccumulator + 'static,
    {
        let prepared = self.prepared()?;
        let mut artifacts = crate::commitment::SetupPrefixProverRegistry::new(
            prepared.expanded.descriptor().setup_seed.clone(),
        );
        for id in ids {
            let mut artifact = self.setup_prefix_material::<F>(id)?.artifact.clone();
            // Portable public rows take their dimension from the frozen profile,
            // matching their canonical serialized representation.
            artifact.commitment.rows = artifact
                .commitment
                .rows
                .into_iter()
                .map(akita_types::RingVec::into_compact)
                .collect();
            artifacts.insert(artifact)?;
        }
        artifacts.mark_backend_validated();
        Ok(artifacts)
    }

    /// Prepare an exact public setup prefix as a reusable commitment.
    pub fn prepare_setup_prefix<F, E>(
        &self,
        id: &SetupPrefixSlotId,
    ) -> Result<PreparedSetupPrefix<F, CommitmentHandle<F, E, Cfg>>, AkitaError>
    where
        Cfg: CommitmentConfig<Field = F>,
        F: Field
            + CanonicalEncoding
            + AkitaSerialize
            + Valid
            + Ring
            + Unreduced
            + WithCommitAccumulator
            + 'static,
        F::Wide: From<F> + AdditiveGroup,
        E: ExtField<F>
            + FpExtEncoding<F>
            + MulBaseUnreduced<F>
            + Unreduced
            + Fold
            + AkitaSerialize
            + 'static,
    {
        self.validate_extension::<E>()?;

        // The commitment and the prefix coefficients are a pure function of the
        // owned setup and the slot id, so derive them at most once per backend.
        // Only the per-proof operation identity below is minted fresh, keeping
        // handle lineage exactly as it was when every call re-derived.
        let cached = self.setup_prefix_material::<F>(id)?;

        self.issue_setup_prefix_handle(id, cached)
    }

    /// Validate portable artifacts and issue handles owned by this backend.
    ///
    /// Every retained image and public commitment is checked against recomputed
    /// prefix material. Serialized backend identifiers never grant authority.
    #[allow(clippy::type_complexity)] // Retain the concrete backend handle family in the public result.
    pub fn import_setup_prefixes(
        &self,
        artifacts: &crate::commitment::SetupPrefixProverRegistry<Cfg::Field>,
        required_ids: &[SetupPrefixSlotId],
    ) -> Result<
        SetupPrefixProverRegistry<Cfg::Field, CommitmentHandle<Cfg::Field, Cfg::ExtField, Cfg>>,
        AkitaError,
    >
    where
        Cfg::Field: CanonicalEncoding
            + AkitaSerialize
            + Valid
            + Ring
            + Unreduced
            + WithCommitAccumulator
            + 'static,
        <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
        Cfg::ExtField: FpExtEncoding<Cfg::Field>
            + MulBaseUnreduced<Cfg::Field>
            + Unreduced
            + Fold
            + AkitaSerialize
            + 'static,
    {
        let prepared = self.prepared()?;
        if artifacts.setup_seed() != &prepared.expanded.descriptor().setup_seed {
            return Err(AkitaError::InvalidSetup(
                "setup prefix artifacts belong to another setup".into(),
            ));
        }
        artifacts
            .check()
            .map_err(|_| AkitaError::InvalidSetup("invalid setup prefix artifacts".into()))?;
        let mut imported = SetupPrefixProverRegistry::default();
        for id in required_ids {
            let artifact = artifacts.get(id).ok_or_else(|| {
                AkitaError::InvalidSetup("required setup prefix artifact is missing".into())
            })?;
            let slot = if artifacts.is_backend_validated() {
                let cached = self.cache_validated_setup_prefix(artifact)?;
                self.issue_setup_prefix_handle(id, cached)?
            } else {
                self.prepare_setup_prefix::<Cfg::Field, Cfg::ExtField>(id)?
            };
            if slot.public.id != artifact.id
                || slot.public.commitment.rows.len() != artifact.commitment.rows.len()
                || slot
                    .public
                    .commitment
                    .rows
                    .iter()
                    .zip(&artifact.commitment.rows)
                    .any(|(expected, actual)| {
                        expected.coeffs() != actual.coeffs()
                            || (actual.ring_dim() != 0 && actual.ring_dim() != expected.ring_dim())
                    })
                || slot.commitment_handle.committed.retained != artifact.hint
            {
                return Err(AkitaError::InvalidSetup(
                    "setup prefix artifact differs from its committed source".into(),
                ));
            }
            imported.insert(slot)?;
        }
        Ok(imported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::proof_optimized::fp128;
    use akita_serialization::{AkitaDeserialize, AkitaSerialize};
    use jolt_field::One;

    #[test]
    fn prefix_artifact_import_validates_contents_and_issues_fresh_owner_handles() {
        // Debug ring-dispatch arithmetic uses the same stack as commitment fixtures.
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                type Cfg = fp128::Dense;
                type F = fp128::Field;
                const NV: usize = 14;
                let catalog =
                    akita_config::test_support::workspace_schedule_catalog::<Cfg>().unwrap();
                let capacity =
                    akita_config::SetupRequirements::from_catalog::<Cfg>(&catalog, NV, 1)
                        .unwrap()
                        .matrix_capacity;
                let setup =
                    crate::AkitaProverSetup::<F>::generate_with_capacity(NV, 1, capacity).unwrap();
                let row = catalog
                    .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                        akita_types::PolynomialGroupLayout::new(NV, 1),
                    ))
                    .unwrap();
                let params = &row.schedule().root.params;
                let n_prefix =
                    (params.d_a() * params.outer_slice_count().get()).next_power_of_two();
                let prefix =
                    akita_types::setup_prefix_precommitted_params(params, n_prefix).unwrap();
                let id = akita_types::scheduled_setup_prefix(n_prefix, prefix)
                    .slot_id()
                    .unwrap();
                let first = CpuBackend::<Cfg>::new(setup.expanded.clone(), &catalog).unwrap();
                for offset in 1..=32 {
                    let mut invalid = id.clone();
                    invalid.natural_len = invalid.natural_len.checked_add(offset).unwrap();
                    assert!(first
                        .export_setup_prefixes::<F>(std::slice::from_ref(&invalid))
                        .is_err());
                }
                assert_eq!(first.setup_prefix_cache_len().unwrap(), 0);
                let artifacts = first
                    .export_setup_prefixes::<F>(std::slice::from_ref(&id))
                    .unwrap();
                assert_eq!(first.setup_prefix_cache_len().unwrap(), 1);
                assert!(artifacts.is_backend_validated());
                let cached_after_export = first
                    .memoized_setup_prefix(&id, || {
                        panic!("export must seed the setup-prefix cache")
                    })
                    .unwrap();
                let exported = artifacts.get(&id).unwrap();
                assert_eq!(cached_after_export.artifact.id, exported.id);
                assert_eq!(
                    cached_after_export.artifact.commitment.rows[0].coeffs(),
                    exported.commitment.rows[0].coeffs()
                );
                let mut bytes = Vec::new();
                artifacts.serialize_compressed(&mut bytes).unwrap();
                let decoded =
                    crate::commitment::SetupPrefixProverRegistry::<F>::deserialize_compressed(
                        &mut bytes.as_slice(),
                        &(),
                    )
                    .unwrap();
                assert_eq!(decoded, artifacts);
                assert!(!decoded.is_backend_validated());
                let second = CpuBackend::<Cfg>::new(setup.expanded.clone(), &catalog).unwrap();
                let imported_first = first
                    .import_setup_prefixes(&decoded, std::slice::from_ref(&id))
                    .unwrap();
                let imported_second = second
                    .import_setup_prefixes(&decoded, std::slice::from_ref(&id))
                    .unwrap();
                let a = &imported_first.get(&id).unwrap().commitment_handle;
                let b = &imported_second.get(&id).unwrap().commitment_handle;
                assert_eq!(a.owner, first.owner_id());
                assert_eq!(b.owner, second.owner_id());
                assert_ne!(a.owner, b.owner);
                assert_eq!(a.committed.public, b.committed.public);

                let imported_again = first
                    .import_setup_prefixes(&decoded, std::slice::from_ref(&id))
                    .unwrap();
                let a_again = &imported_again.get(&id).unwrap().commitment_handle;
                assert_ne!(
                    a.committed.commitment_id, a_again.committed.commitment_id,
                    "each imported handle must retain fresh operation lineage"
                );
                assert!(
                    Arc::ptr_eq(&a.committed.source, &a_again.committed.source),
                    "repeated imports on one backend must reuse the memoized prefix source"
                );

                let mut slot = decoded.get(&id).unwrap().clone();
                let mut fields = slot.commitment.rows[0].coeffs().to_vec();
                fields[0] += F::one();
                slot.commitment.rows[0] = akita_types::RingVec::from_coeffs(fields);
                let mut changed =
                    crate::commitment::SetupPrefixProverRegistry::new(decoded.setup_seed().clone());
                changed.insert(slot).unwrap();
                assert!(second
                    .import_setup_prefixes(&changed, std::slice::from_ref(&id))
                    .is_err());
                let wrong_setup =
                    crate::commitment::SetupPrefixProverRegistry::<F>::new([99; 32].into());
                assert!(second
                    .import_setup_prefixes(&wrong_setup, std::slice::from_ref(&id))
                    .is_err());
                assert!(second
                    .import_setup_prefixes(&decoded, std::slice::from_ref(&id))
                    .is_ok());
                assert!(second
                    .import_setup_prefixes(&decoded, &[])
                    .unwrap()
                    .is_empty());
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn failed_prefix_cache_generations_are_removed_and_retriable() {
        type Cfg = fp128::Dense;
        const NV: usize = 14;
        let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>().unwrap();
        let row = catalog
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                akita_types::PolynomialGroupLayout::new(NV, 1),
            ))
            .unwrap();
        let params = &row.schedule().root.params;
        let n_prefix = (params.d_a() * params.outer_slice_count().get()).next_power_of_two();
        let prefix = akita_types::setup_prefix_precommitted_params(params, n_prefix).unwrap();
        let id = akita_types::scheduled_setup_prefix(n_prefix, prefix)
            .slot_id()
            .unwrap();

        let failed = super::super::backend::SetupPrefixCache::<u64>::default();
        for offset in 0..32 {
            let mut failed_id = id.clone();
            failed_id.natural_len = failed_id.natural_len.checked_add(offset).unwrap();
            assert!(failed
                .memoized(&failed_id, || {
                    Err(AkitaError::InvalidSetup("injected prefix failure".into()))
                })
                .is_err());
        }
        assert_eq!(failed.len().unwrap(), 0);

        assert!(failed
            .memoized(&id, || {
                Err(AkitaError::InvalidSetup("injected prefix failure".into()))
            })
            .is_err());
        assert_eq!(failed.len().unwrap(), 0);
        assert_eq!(*failed.memoized(&id, || Ok(17u64)).unwrap(), 17);
        assert_eq!(failed.len().unwrap(), 1);
    }

    #[test]
    fn concurrent_prefix_cache_failure_preserves_the_successful_retry() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Barrier, Mutex};

        type Cfg = fp128::Dense;
        const NV: usize = 14;
        const WORKERS: usize = 8;
        let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>().unwrap();
        let row = catalog
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                akita_types::PolynomialGroupLayout::new(NV, 1),
            ))
            .unwrap();
        let params = &row.schedule().root.params;
        let n_prefix = (params.d_a() * params.outer_slice_count().get()).next_power_of_two();
        let prefix = akita_types::setup_prefix_precommitted_params(params, n_prefix).unwrap();
        let id = akita_types::scheduled_setup_prefix(n_prefix, prefix)
            .slot_id()
            .unwrap();

        let backend = Arc::new(super::super::backend::SetupPrefixCache::<u64>::default());
        let start = Arc::new(Barrier::new(WORKERS));
        let attempts = Arc::new(AtomicUsize::new(0));
        let outcomes = Arc::new(Mutex::new(Vec::with_capacity(WORKERS)));
        std::thread::scope(|scope| {
            for _ in 0..WORKERS {
                let backend = Arc::clone(&backend);
                let start = Arc::clone(&start);
                let attempts = Arc::clone(&attempts);
                let outcomes = Arc::clone(&outcomes);
                let id = id.clone();
                scope.spawn(move || {
                    start.wait();
                    let result = backend.memoized(&id, || {
                        if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                            Err(AkitaError::InvalidSetup("injected prefix failure".into()))
                        } else {
                            Ok(23)
                        }
                    });
                    outcomes.lock().unwrap().push(result.map(|value| *value));
                });
            }
        });

        let outcomes = outcomes.lock().unwrap();
        assert_eq!(outcomes.iter().filter(|result| result.is_err()).count(), 1);
        assert_eq!(
            outcomes
                .iter()
                .filter(|result| matches!(result, Ok(23)))
                .count(),
            WORKERS - 1
        );
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(backend.len().unwrap(), 1);
        assert_eq!(
            *backend
                .memoized(&id, || { panic!("successful retry must remain cached") })
                .unwrap(),
            23
        );
    }
}
