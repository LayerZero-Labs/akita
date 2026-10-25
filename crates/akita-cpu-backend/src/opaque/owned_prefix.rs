//! Validated conversion from portable setup artifacts to current-owner handles.
use super::owned::{CommitmentHandle, CommittedSource, OwnedPolynomials};
use crate::{CpuBackend, DensePoly};
use akita_error::AkitaError;
use akita_params::sis::{CommittedSourceClass, CommittedSourceContract};
use akita_params::{DecompositionParams, SetupPrefixSlotId};
use akita_prover::{PreparedSetupPrefix, SetupPrefixProverRegistry};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::{Commitment, FpExtEncoding};
use jolt_field::{
    AdditiveGroup, CanonicalEncoding, ExtField, Field, Fold, MulBaseUnreduced, Ring, Unreduced,
    WithCommitAccumulator,
};
use std::sync::{Arc, OnceLock};

mod streaming;
use streaming::PrefixSource;

/// Setup-prefix material memoized on the owning backend.
///
/// Export retains the artifact only. A proof creates its shared source view on
/// first use; verifier preprocessing never retains polynomial digit caches.
pub(super) struct CachedSetupPrefix<F: Field> {
    artifact: crate::commitment::SetupPrefixSlot<F>,
    source: OnceLock<Arc<OwnedPolynomials<DensePoly<F>>>>,
}

impl<F: Field + CanonicalEncoding + 'static, E> CpuBackend<F, E> {
    fn setup_prefix_source(
        &self,
        id: &SetupPrefixSlotId,
    ) -> Result<Arc<OwnedPolynomials<DensePoly<F>>>, AkitaError> {
        let prepared = self.prepared()?;
        let polynomial = DensePoly::from_setup_prefix(
            Arc::clone(&prepared.expanded),
            id.commitment_profile.group.num_vars(),
        )?;
        Ok(Arc::new(OwnedPolynomials {
            polynomials: vec![polynomial],
        }))
    }

    /// Return the memoized commitment artifact for one setup prefix.
    ///
    /// Export and import deliberately share this path: profiles commonly export
    /// an artifact during setup and import it immediately before proving, so a
    /// second derivation in `import_setup_prefixes` would put setup work on the
    /// timed prover path.
    fn setup_prefix_material(
        &self,
        id: &SetupPrefixSlotId,
    ) -> Result<Arc<CachedSetupPrefix<F>>, AkitaError>
    where
        F: Valid + Unreduced + WithCommitAccumulator,
    {
        self.prefix_material_from(PrefixSource::Expanded(&self.prepared()?.expanded), id)
    }

    /// Adopt material already committed by a CPU backend in this process.
    fn cache_validated_setup_prefix(
        &self,
        artifact: &crate::commitment::SetupPrefixSlot<F>,
    ) -> Result<Arc<CachedSetupPrefix<F>>, AkitaError> {
        let prepared = self.prepared()?;
        crate::setup::setup_prefix::validate_setup_prefix_commitment(
            prepared.expanded.shared_matrix().num_field_elements(),
            &artifact.id,
        )?;
        self.setup_prefix_cache.get_or_build(&artifact.id, || {
            let mut artifact = artifact.clone();
            artifact
                .hint
                .share_inner_rows(&self.setup_prefix_inner_rows)?;
            Ok(CachedSetupPrefix {
                artifact,
                source: OnceLock::new(),
            })
        })
    }

    /// Build portable setup-prefix artifacts for application-managed persistence.
    pub fn export_setup_prefixes(
        &self,
        ids: &[SetupPrefixSlotId],
    ) -> Result<crate::commitment::SetupPrefixProverRegistry<F>, AkitaError>
    where
        F: Valid + Unreduced + WithCommitAccumulator,
    {
        self.export_prefixes_from(PrefixSource::Expanded(&self.prepared()?.expanded), ids)
    }
}

impl<F, E> CpuBackend<F, E>
where
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
    fn issue_setup_prefix_handle(
        &self,
        id: &SetupPrefixSlotId,
        cached: Arc<CachedSetupPrefix<F>>,
    ) -> Result<PreparedSetupPrefix<F, CommitmentHandle<F, E>>, AkitaError> {
        let source = match cached.source.get() {
            Some(source) => Arc::clone(source),
            None => {
                let source = self.setup_prefix_source(id)?;
                Arc::clone(cached.source.get_or_init(|| source))
            }
        };
        let artifact = &cached.artifact;
        let public_commitment = artifact
            .commitment
            .rows
            .first()
            .ok_or_else(|| {
                AkitaError::Internal("cached setup prefix has no public commitment row".into())
            })?
            .clone();
        // Setup prefixes are public uniform field elements, planned as
        // full-width balanced digits rather than under any family's contract.
        let profile = &id.commitment_profile;
        let producer_contract = CommittedSourceContract::try_new(
            CommittedSourceClass::BalancedSignedDigit,
            DecompositionParams {
                log_basis: profile.inner.digits.log_basis,
                log_commit_bound: profile.inner.matrix.sis_modulus_profile().field_bits(),
                log_open_bound: None,
            },
        )?;
        let committed = Arc::new(CommittedSource {
            commitment_id: self.owner().next_operation_id()?,
            source,
            metadata: akita_prover::SourceMetadata::try_new(
                1,
                id.commitment_profile.group.num_vars(),
            )?,
            parameters: id.commitment_profile,
            producer_contract,
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

    /// Prepare an exact public setup prefix as a reusable commitment.
    pub fn prepare_setup_prefix(
        &self,
        id: &SetupPrefixSlotId,
    ) -> Result<PreparedSetupPrefix<F, CommitmentHandle<F, E>>, AkitaError> {
        // Retain one value per slot, allowing duplicate cold derivations without waiting.
        // Each handle receives a fresh per-proof operation identity.
        let cached = self.setup_prefix_material(id)?;

        self.issue_setup_prefix_handle(id, cached)
    }

    /// Validate portable artifacts and issue handles owned by this backend.
    ///
    /// Every retained image and public commitment is checked against recomputed
    /// prefix material. Serialized backend identifiers never grant authority.
    #[allow(clippy::type_complexity)] // Retain the concrete backend handle family in the public result.
    pub fn import_setup_prefixes(
        &self,
        artifacts: &crate::commitment::SetupPrefixProverRegistry<F>,
        required_ids: &[SetupPrefixSlotId],
    ) -> Result<SetupPrefixProverRegistry<F, CommitmentHandle<F, E>>, AkitaError> {
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
                self.prepare_setup_prefix(id)?
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
                let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>(
                    akita_params::ChunkedWitnessCfg::default_non_chunked(),
                )
                .unwrap();
                let capacity =
                    akita_config::SetupRequirements::from_catalog::<Cfg>(&catalog, NV, 1)
                        .unwrap()
                        .matrix_capacity();
                let setup =
                    crate::AkitaProverSetup::<F>::generate_with_capacity(NV, 1, capacity).unwrap();
                let row = catalog
                    .resolve_key(&akita_params::ScheduleLookupKey::single(
                        akita_params::PolynomialGroupLayout::new(NV, 1),
                    ))
                    .unwrap();
                let params = &row.schedule().root.params;
                let n_prefix =
                    (params.d_a() * params.outer_slice_count().get()).next_power_of_two();
                let prefix =
                    akita_params::setup_prefix_precommitted_params(params, n_prefix).unwrap();
                let id = akita_params::scheduled_setup_prefix(n_prefix, prefix)
                    .slot_id()
                    .unwrap();
                let first = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
                for offset in 1..=32 {
                    let mut invalid = id.clone();
                    invalid.natural_len = invalid.natural_len.checked_add(offset).unwrap();
                    assert!(first
                        .export_setup_prefixes(std::slice::from_ref(&invalid))
                        .is_err());
                }
                assert_eq!(first.setup_prefix_cache_len().unwrap(), 0);
                let artifacts = first
                    .export_setup_prefixes(std::slice::from_ref(&id))
                    .unwrap();
                assert_eq!(first.setup_prefix_cache_len().unwrap(), 1);
                assert!(artifacts.is_backend_validated());
                let cached_after_export = first
                    .setup_prefix_cache
                    .get_or_build(&id, || panic!("export must seed the setup-prefix cache"))
                    .unwrap();
                let exported = artifacts.get(&id).unwrap();
                assert!(
                    cached_after_export.source.get().is_none(),
                    "export must not prepare or retain a polynomial source"
                );
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
                let second = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
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
}

#[cfg(test)]
mod cache_tests;
