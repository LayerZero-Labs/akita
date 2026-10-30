//! Prover setup artifact and config-free setup expansion helpers.

use akita_error::AkitaError;

use crate::commitment::SetupPrefixProverRegistry;
use akita_params::{FlatMatrix, SetupMatrixCapacity, SetupPrefixSlotId};
use akita_serialization::{AkitaSerialize, SerializationError, Valid};
use akita_types::{
    sample_akita_setup_seed, AkitaExpandedSetup, AkitaSetupDescriptor, AkitaVerifierSetup,
    SetupPrefixVerifierRegistry,
};
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::sync::Arc;

/// Prover setup artifact.
///
/// Backend-prepared compute state is intentionally not stored here. Host code
/// prepares a compute backend from the expanded setup when it wants to prove.
///
/// Fields are private: a prover setup is derived from its seed in this
/// process, with commitments exported here, or issued by the release loader
/// as a [`crate::PinnedProverSetup`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaProverSetup<F: Field> {
    /// Expanded matrix stage used by the prover.
    pub(crate) expanded: Arc<AkitaExpandedSetup<F>>,
    /// Preprocessed setup-prefix commitment slots for setup-claim offloading.
    ///
    /// D-free (S4): the registry stores flat ring-coefficient commitment rows and
    /// D-free hints; concrete-D selection happens at backend-prepare /
    /// per-operation time, not on this artifact.
    pub(crate) prefix_slots: SetupPrefixProverRegistry<F>,
}

impl<F: Field> AkitaProverSetup<F> {
    /// Expanded matrix stage used by the prover.
    pub fn expanded(&self) -> &Arc<AkitaExpandedSetup<F>> {
        &self.expanded
    }

    /// Setup-prefix commitment slots for setup-claim offloading.
    pub fn prefix_slots(&self) -> &SetupPrefixProverRegistry<F> {
        &self.prefix_slots
    }

    /// Derive the setup `descriptor` describes, with no setup-prefix slots.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor is malformed.
    #[tracing::instrument(skip_all, name = "AkitaProverSetup::derive")]
    pub fn derive(descriptor: AkitaSetupDescriptor) -> Result<Self, AkitaError>
    where
        F: CanonicalEncoding,
    {
        let setup_seed = descriptor.setup_seed.clone();
        Ok(Self {
            expanded: Arc::new(AkitaExpandedSetup::derive(descriptor)?),
            prefix_slots: SetupPrefixProverRegistry::new(setup_seed),
        })
    }

    /// Commit to the setup prefixes `ids` not stored yet, from this setup's
    /// matrix, and store them.
    ///
    /// # Errors
    ///
    /// Returns an error when a slot does not fit the matrix or export fails.
    pub fn export_setup_prefixes(&mut self, ids: &[SetupPrefixSlotId]) -> Result<(), AkitaError>
    where
        F: CanonicalEncoding + Unreduced + WithCommitAccumulator + Valid + 'static,
    {
        let missing = ids
            .iter()
            .filter(|id| self.prefix_slots.get(id).is_none())
            .cloned()
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(());
        }
        // Prefix commitments depend only on the setup and the slot ids, so the
        // base field fills the exporting backend's extension parameter.
        let exported = crate::CpuBackend::<F, F>::new(self.expanded.clone())?
            .export_setup_prefixes(&missing)?;
        let validated = self.prefix_slots.is_empty() || self.prefix_slots.is_backend_validated();
        for (_, slot) in exported.iter() {
            self.prefix_slots.insert(slot.clone())?;
        }
        if validated {
            self.prefix_slots.mark_backend_validated();
        }
        tracing::info!(
            slots = self.prefix_slots.len(),
            "materialized exact setup-prefix commitments"
        );
        Ok(())
    }

    /// Generate a prover setup from already-computed setup capacity bounds.
    ///
    /// The caller supplies config-derived provisioning bounds in base-field
    /// elements. This constructor owns only the concrete prover artifact:
    /// materialization of that prefix of the public field stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacity is invalid or the setup descriptor
    /// cannot be built.
    #[tracing::instrument(skip_all, name = "AkitaProverSetup::generate_with_capacity")]
    pub fn generate_with_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        setup_capacity: SetupMatrixCapacity,
    ) -> Result<Self, AkitaError>
    where
        F: CanonicalEncoding,
    {
        Self::derive(AkitaSetupDescriptor {
            max_num_vars,
            max_num_batched_polys,
            num_field_elements: setup_capacity.num_field_elements,
            setup_seed: sample_akita_setup_seed(),
        })
    }

    /// Derive a verifier setup with an explicit public-matrix capacity.
    ///
    /// The verifier keeps only the public stream prefix needed by its direct
    /// setup scans and terminal matrix. The setup-prefix registry remains
    /// complete because offloaded claims can refer to entries beyond that
    /// materialized matrix prefix. Verifier setup also initializes a
    /// non-serialized lazy terminal NTT-prefix cache.
    ///
    /// # Errors
    ///
    /// Returns an error if `matrix_capacity` is empty or exceeds the prover
    /// matrix, or if prover prefix-slot metadata cannot be converted into
    /// verifier-visible prefix slots.
    ///
    /// # Safety
    ///
    /// Soundness contract of [`AkitaVerifierSetup::from_parts`]: the caller
    /// accepts this prover's commitments as honest. That holds for a setup
    /// derived and exported in this process; a production verifier is
    /// instead loaded from a release.
    pub unsafe fn to_verifier_setup(
        &self,
        matrix_capacity: SetupMatrixCapacity,
    ) -> Result<AkitaVerifierSetup<F>, AkitaError>
    where
        F: Valid,
    {
        if matrix_capacity.num_field_elements == 0 {
            return Err(AkitaError::InvalidSetup(
                "verifier setup matrix capacity must be non-zero".to_string(),
            ));
        }
        let prover_matrix = self.expanded.shared_matrix().as_field_slice();
        let verifier_matrix = prover_matrix
            .get(..matrix_capacity.num_field_elements)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "verifier setup requires {} field elements but prover setup has {}",
                    matrix_capacity.num_field_elements,
                    prover_matrix.len()
                ))
            })?;
        let expanded = if verifier_matrix.len() == prover_matrix.len() {
            self.expanded.clone()
        } else {
            let mut descriptor = self.expanded.descriptor().clone();
            descriptor.num_field_elements = verifier_matrix.len();
            // SAFETY: a prefix of this setup's matrix, which was derived
            // from its seed.
            Arc::new(unsafe {
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    descriptor,
                    FlatMatrix::from_flat_data(verifier_matrix.to_vec()),
                )
            })
        };
        let mut prefix_slots =
            SetupPrefixVerifierRegistry::new(self.expanded.descriptor().setup_seed.clone());
        if prefix_slots.setup_seed() != self.prefix_slots.setup_seed() {
            return Err(AkitaError::InvalidSetup(
                "setup-prefix registries belong to different public matrices".into(),
            ));
        }
        for slot in self.prefix_slots.verifier_slots() {
            prefix_slots.insert(slot)?;
        }
        // SAFETY: forwarded to the caller.
        unsafe { AkitaVerifierSetup::from_parts(expanded, prefix_slots) }
    }
}

impl<F: Field + CanonicalEncoding + Valid + AkitaSerialize> Valid for AkitaProverSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()?;
        if self.prefix_slots.setup_seed() != &self.expanded.descriptor().setup_seed {
            return Err(SerializationError::InvalidData(
                "setup-prefix registry belongs to a different public matrix".to_string(),
            ));
        }
        self.prefix_slots.check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Prime128Offset275;

    #[test]
    fn generate_with_capacity_rejects_zero_setup_len() {
        let zero_len = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity {
                num_field_elements: 0,
            },
        )
        .expect_err("zero setup length must not produce an undecodable setup");
        assert!(zero_len.to_string().contains("num_field_elements"));
    }

    #[test]
    fn verifier_setup_conversion_enforces_the_requested_prefix() {
        let setup = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity {
                num_field_elements: 8,
            },
        )
        .expect("generate setup");

        let verifier = unsafe {
            setup.to_verifier_setup(SetupMatrixCapacity {
                num_field_elements: 3,
            })
        }
        .expect("narrow verifier setup");
        assert_eq!(verifier.expanded().descriptor().num_field_elements, 3);
        assert_eq!(verifier.expanded().shared_matrix().num_field_elements(), 3);
        assert_eq!(
            verifier.expanded().shared_matrix().as_field_slice(),
            &setup.expanded.shared_matrix().as_field_slice()[..3]
        );
        assert!(
            unsafe {
                setup.to_verifier_setup(SetupMatrixCapacity {
                    num_field_elements: 9,
                })
            }
            .is_err(),
            "conversion must not extend beyond the prover matrix"
        );
    }

    #[test]
    fn verifier_conversion_rejects_a_foreign_prefix_registry() {
        let mut setup = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity {
                num_field_elements: 8,
            },
        )
        .unwrap();
        setup.prefix_slots = SetupPrefixProverRegistry::new([7; 32].into());
        assert!(unsafe {
            setup.to_verifier_setup(SetupMatrixCapacity {
                num_field_elements: 8,
            })
        }
        .is_err());
    }

    #[test]
    fn prover_setup_check_validates_prefix_slots() {
        use crate::commitment::{PortableCommitmentHandle, SetupPrefixSlot};
        use akita_params::{
            scheduled_setup_prefix, CompressionChainPlan, CompressionChainWitness,
            GroupCommitPhaseParams, GroupOpenPhaseParams, InnerCommitMatrixParams,
            OuterCommitMatrixParams, PackedNegativeBinary, PolynomialGroupLayout, SisMatrixRole,
            SisModulusProfileId, SisTableDigest, SisTableKey, DEFAULT_SIS_SECURITY_POLICY,
        };
        use akita_types::{RingVec, SetupPrefixPublicCommitment};

        const PREFIX_D: usize = 64;

        let mut setup = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity::minimum(),
        )
        .expect("generate setup");
        let decomposed =
            RingVec::from_coeffs_with_ring_dim(Vec::new(), PREFIX_D).expect("empty A-native hint");
        let inner_bound = akita_params::sis::rounded_up_role_a_inf_norm(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            PREFIX_D,
            3,
            &akita_challenges::SparseChallengeConfig::production_for_ring_dim(PREFIX_D)
                .expect("D=64 has a production challenge configuration"),
            1,
            1,
        )
        .expect("audited prefix A bound");
        let inner_digits = akita_params::sis::compute_num_digits_field_width(
            SisModulusProfileId::Q128OffsetA7F7.field_bits(),
            1,
        );
        let inner_commit_matrix = InnerCommitMatrixParams::try_new_with_min_rank(
            SisTableKey {
                policy: DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
                role: SisMatrixRole::Inner,
                ring_dimension: u32::try_from(PREFIX_D).expect("test prefix ring dimension"),
                coeff_linf_bound: inner_bound,
            },
            inner_digits,
        )
        .expect("audited prefix A matrix");
        let outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
            SisTableKey {
                policy: DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
                role: SisMatrixRole::Outer,
                ring_dimension: u32::try_from(PREFIX_D).expect("test prefix ring dimension"),
                coeff_linf_bound: 3,
            },
            inner_commit_matrix.output_rank(),
        )
        .expect("audited prefix B matrix");
        let commitment_rows = outer_commit_matrix.output_rank();
        let compression_plan = CompressionChainPlan::for_complete_source(
            SisModulusProfileId::Q128OffsetA7F7,
            commitment_rows * PREFIX_D,
        )
        .expect("prefix compression plan");
        let compression_stages = compression_plan
            .maps()
            .iter()
            .map(|map| PackedNegativeBinary::from_bytes(*map, vec![0; map.packed_digit_bytes()]))
            .collect::<Result<Vec<_>, _>>()
            .expect("zero compression stages");
        let compression_witness =
            CompressionChainWitness::new(compression_plan.clone(), compression_stages)
                .expect("zero compression witness");
        let compression_quotients = compression_plan
            .maps()
            .iter()
            .map(|map| {
                RingVec::from_coeffs_with_ring_dim(
                    vec![Prime128Offset275::default(); map.output_coefficients()],
                    map.ring_dimension(),
                )
                .expect("zero compression quotient")
            })
            .collect::<Vec<_>>();
        let hint = PortableCommitmentHandle::singleton_with_outer_compression(
            decomposed,
            &compression_witness,
            &compression_quotients,
        )
        .expect("compression-valid A-native hint");
        let commitment_params = GroupOpenPhaseParams {
            setup_natural_len: None,
            profile: GroupCommitPhaseParams {
                version: GroupCommitPhaseParams::VERSION,
                group: PolynomialGroupLayout::singleton(PREFIX_D.trailing_zeros() as usize),
                blocks: akita_params::BlockGeometry::new(1, 1, 1),
                outer_slice_count: akita_params::CommitmentSliceCount::ONE,
                inner: akita_params::RoleParams::new(
                    akita_params::GadgetDigits::new(1, inner_digits),
                    inner_commit_matrix,
                ),
                outer: akita_params::RoleParams::new(
                    akita_params::GadgetDigits::new(1, 1),
                    outer_commit_matrix,
                ),
            },
            opening: akita_params::GroupOpeningPlan::evaluation_trace(
                akita_challenges::SparseChallengeConfig::pm1_only(0),
                1,
                1,
                1,
            ),
        };
        let err = setup
            .prefix_slots
            .insert(SetupPrefixSlot {
                id: scheduled_setup_prefix(PREFIX_D, commitment_params)
                    .slot_id()
                    .expect("setup prefix group"),
                commitment: SetupPrefixPublicCommitment {
                    rows: vec![
                        RingVec::from_coeffs(vec![Prime128Offset275::default(); PREFIX_D]);
                        commitment_rows
                    ],
                },
                hint,
            })
            .expect_err("insert must reject malformed prefix slot");
        assert!(err.to_string().contains("commitment row"));
    }
}
