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

impl<F, E> akita_prover::SetupPrefixKernel<F, E> for CpuBackend
where
    F: Field
        + CanonicalEncoding
        + AkitaSerialize
        + Valid
        + jolt_field::PseudoMersenne
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
    CpuBackend: akita_prover::ProverHandleFamily<F, E, CommitmentHandle = CommitmentHandle<F, E>>,
{
    fn prepare_setup_prefix(
        &self,
        setup: &akita_types::AkitaSetupDescriptor,
        prefix: &SetupPrefixSlotId,
    ) -> Result<PreparedSetupPrefix<F, Self::CommitmentHandle>, AkitaError> {
        if self.prepared::<F>()?.expanded.descriptor() != setup {
            return Err(AkitaError::InvalidSetup(
                "setup prefix request belongs to another setup".into(),
            ));
        }
        self.prepare_setup_prefix::<F, E>(prefix)
    }
}

impl CpuBackend {
    /// Build portable setup-prefix artifacts for application-managed persistence.
    pub fn export_setup_prefixes<F>(
        &self,
        ids: &[SetupPrefixSlotId],
    ) -> Result<crate::commitment::SetupPrefixProverRegistry<F>, AkitaError>
    where
        F: Field + CanonicalEncoding + Valid + Unreduced + WithCommitAccumulator + 'static,
    {
        let prepared = self.prepared::<F>()?;
        let executor = CommitmentExecutor::cpu(
            self,
            prepared,
            &prepared.expanded,
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            PortableStatePolicy,
        )?;
        let mut artifacts = crate::commitment::SetupPrefixProverRegistry::new(
            prepared.expanded.descriptor().setup_seed.clone(),
        );
        for id in ids {
            let mut artifact = crate::commit_setup_prefix(&prepared.expanded, &executor, id)?;
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
        Ok(artifacts)
    }

    /// Prepare an exact public setup prefix as a reusable commitment.
    pub fn prepare_setup_prefix<F, E>(
        &self,
        id: &SetupPrefixSlotId,
    ) -> Result<PreparedSetupPrefix<F, CommitmentHandle<F, E>>, AkitaError>
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
        self.validate_extension::<E>()?;
        let prepared = self.prepared::<F>()?;
        let executor = CommitmentExecutor::cpu(
            self,
            prepared,
            &prepared.expanded,
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            PortableStatePolicy,
        )?;
        let artifact = crate::commit_setup_prefix(&prepared.expanded, &executor, id)?;
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
        let public_commitment = artifact
            .commitment
            .rows
            .first()
            .ok_or(AkitaError::InvalidProof)?
            .clone();
        let committed = Arc::new(CommittedSource {
            commitment_id: self.owner().next_operation_id()?,
            source: Arc::new(OwnedPolynomials {
                polynomials: vec![polynomial],
            }),
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

    /// Validate portable artifacts and issue handles owned by this backend.
    ///
    /// Every retained image and public commitment is checked against recomputed
    /// prefix material. Serialized backend identifiers never grant authority.
    #[allow(clippy::type_complexity)] // Retain the concrete backend handle family in the public result.
    pub fn import_setup_prefixes<Cfg>(
        &self,
        artifacts: &crate::commitment::SetupPrefixProverRegistry<Cfg::Field>,
    ) -> Result<
        SetupPrefixProverRegistry<Cfg::Field, CommitmentHandle<Cfg::Field, Cfg::ExtField>>,
        AkitaError,
    >
    where
        Cfg: CommitmentConfig + 'static,
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
        self.validate_config::<Cfg>()?;
        let prepared = self.prepared::<Cfg::Field>()?;
        if artifacts.setup_seed() != &prepared.expanded.descriptor().setup_seed {
            return Err(AkitaError::InvalidSetup(
                "setup prefix artifacts belong to another setup".into(),
            ));
        }
        artifacts
            .check()
            .map_err(|_| AkitaError::InvalidSetup("invalid setup prefix artifacts".into()))?;
        let mut imported = SetupPrefixProverRegistry::new(artifacts.setup_seed().clone());
        for (id, artifact) in artifacts.iter() {
            let slot = self.prepare_setup_prefix::<Cfg::Field, Cfg::ExtField>(id)?;
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
                let first = CpuBackend::new::<Cfg>(setup.expanded.clone(), &catalog).unwrap();
                let artifacts = first
                    .export_setup_prefixes::<F>(std::slice::from_ref(&id))
                    .unwrap();
                let mut bytes = Vec::new();
                artifacts.serialize_compressed(&mut bytes).unwrap();
                let decoded =
                    crate::commitment::SetupPrefixProverRegistry::<F>::deserialize_compressed(
                        &mut bytes.as_slice(),
                        &(),
                    )
                    .unwrap();
                assert_eq!(decoded, artifacts);
                let second = CpuBackend::new::<Cfg>(setup.expanded.clone(), &catalog).unwrap();
                let imported_first = first.import_setup_prefixes::<Cfg>(&decoded).unwrap();
                let imported_second = second.import_setup_prefixes::<Cfg>(&decoded).unwrap();
                let a = &imported_first.get(&id).unwrap().commitment_handle;
                let b = &imported_second.get(&id).unwrap().commitment_handle;
                assert_eq!(a.owner, first.owner_id());
                assert_eq!(b.owner, second.owner_id());
                assert_ne!(a.owner, b.owner);
                assert_eq!(a.committed.public, b.committed.public);

                let mut slot = decoded.get(&id).unwrap().clone();
                let mut fields = slot.commitment.rows[0].coeffs().to_vec();
                fields[0] += F::one();
                slot.commitment.rows[0] = akita_types::RingVec::from_coeffs(fields);
                let mut changed =
                    crate::commitment::SetupPrefixProverRegistry::new(decoded.setup_seed().clone());
                changed.insert(slot).unwrap();
                assert!(second.import_setup_prefixes::<Cfg>(&changed).is_err());
                let wrong_setup =
                    crate::commitment::SetupPrefixProverRegistry::<F>::new([99; 32].into());
                assert!(second.import_setup_prefixes::<Cfg>(&wrong_setup).is_err());
                assert!(second.import_setup_prefixes::<Cfg>(&decoded).is_ok());
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
