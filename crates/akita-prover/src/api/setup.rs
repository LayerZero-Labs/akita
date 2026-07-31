//! Prover setup artifact and config-free setup expansion helpers.

use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_serialization::{AkitaSerialize, SerializationError, Valid};
use akita_types::{
    derive_public_matrix_prefix, sample_public_matrix_id, AkitaExpandedSetup, AkitaSetupSeed,
    AkitaVerifierSetup, SetupMatrixCapacity, SetupPrefixProverRegistry,
    SetupPrefixVerifierRegistry,
};
use std::sync::Arc;

/// Prover setup artifact.
///
/// Backend-prepared compute state is intentionally not stored here. Host code
/// prepares a compute backend from the expanded setup when it wants to prove.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaProverSetup<F: FieldCore> {
    /// Expanded matrix stage used by both prover and verifier.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
    /// Preprocessed setup-prefix commitment slots for setup-claim offloading.
    ///
    /// D-free (S4): the registry stores flat ring-coefficient commitment rows and
    /// D-free hints; concrete-D selection happens at backend-prepare /
    /// per-operation time, not on this artifact.
    pub prefix_slots: SetupPrefixProverRegistry<F>,
}

impl<F: FieldCore> AkitaProverSetup<F> {
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
        F: CanonicalField + RandomSampling + AkitaSerialize,
    {
        let public_matrix_id = sample_public_matrix_id();
        let seed = AkitaSetupSeed {
            max_num_vars,
            max_num_batched_polys,
            num_field_elements: setup_capacity.num_field_elements,
            public_matrix_id: public_matrix_id.clone(),
        };
        seed.check().map_err(|err| {
            AkitaError::InvalidSetup(format!("setup seed validation failed: {err}"))
        })?;

        let shared_flat =
            derive_public_matrix_prefix::<F>(setup_capacity.num_field_elements, &public_matrix_id);
        let expanded = Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared_flat),
        );

        Ok(Self {
            expanded,
            prefix_slots: SetupPrefixProverRegistry::new(public_matrix_id),
        })
    }

    /// Derive a verifier setup from this prover setup.
    ///
    /// This copies protocol-independent setup state. Verifier setup initializes
    /// a non-serialized lazy terminal NTT-prefix cache; direct terminal checks
    /// prepare exact or covering prefixes on demand.
    ///
    /// # Errors
    ///
    /// Returns an error if prover prefix-slot metadata cannot be converted into
    /// verifier-visible prefix slots.
    pub fn verifier_setup(&self) -> Result<AkitaVerifierSetup<F>, AkitaError> {
        let mut prefix_slots =
            SetupPrefixVerifierRegistry::new(self.expanded.seed().public_matrix_id.clone());
        prefix_slots.replace_from_prover_registry(&self.prefix_slots)?;
        AkitaVerifierSetup::from_parts(self.expanded.clone(), prefix_slots)
    }

    /// Wrap an already-validated [`AkitaExpandedSetup`] in a prover setup.
    ///
    /// Use this when the caller has already run strict setup validation, for
    /// example through checked setup deserialization. This still re-checks
    /// seed-to-matrix derivation at the trust boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if the expanded setup does not match its seed.
    pub fn from_validated_expanded(expanded: AkitaExpandedSetup<F>) -> Result<Self, AkitaError>
    where
        F: CanonicalField + RandomSampling + Valid,
    {
        expanded.check().map_err(|err| {
            AkitaError::InvalidSetup(format!("expanded setup validation failed: {err}"))
        })?;
        Self::from_seed_validated_expanded(expanded)
    }

    /// Wrap a seed-validated [`AkitaExpandedSetup`] in a prover setup.
    ///
    /// This skips seed-to-matrix rederivation. Use it only when the caller
    /// just verified the matrix with `validate_public_matrix_matches_seed` in
    /// the same trust boundary, such as the disk-cache loader in
    /// `akita-setup`.
    ///
    /// # Errors
    ///
    /// Returns an error if the seed and matrix disagree or their internal shape
    /// metadata is malformed.
    pub fn from_seed_validated_expanded(expanded: AkitaExpandedSetup<F>) -> Result<Self, AkitaError>
    where
        F: CanonicalField + Valid,
    {
        expanded.seed().check().map_err(|err| {
            AkitaError::InvalidSetup(format!("expanded setup seed validation failed: {err}"))
        })?;
        expanded.shared_matrix().check().map_err(|err| {
            AkitaError::InvalidSetup(format!("expanded setup matrix validation failed: {err}"))
        })?;
        if expanded.shared_matrix().num_field_elements() != expanded.seed().num_field_elements {
            return Err(AkitaError::InvalidSetup(
                "expanded setup matrix field count does not match setup seed".to_string(),
            ));
        }
        let public_matrix_id = expanded.seed().public_matrix_id.clone();
        let expanded = Arc::new(expanded);
        Ok(Self {
            expanded,
            prefix_slots: SetupPrefixProverRegistry::new(public_matrix_id),
        })
    }
}

impl<F: FieldCore + CanonicalField + RandomSampling + Valid + AkitaSerialize> Valid
    for AkitaProverSetup<F>
{
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()?;
        if self.prefix_slots.public_matrix_id() != &self.expanded.seed().public_matrix_id {
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
    use akita_field::Prime128Offset275;

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
    fn prover_setup_check_validates_prefix_slots() {
        use akita_types::{
            setup_prefix_slot_id, AkitaCommitmentHint, CommittedGroupProfile, DigitBlocks,
            InnerCommitMatrixParams, OuterCommitMatrixParams, PolynomialGroupLayout,
            PrecommittedLevelParams, RingVec, SetupPrefixPublicCommitment, SetupPrefixSlot,
            SisMatrixRole, SisModulusProfileId, SisTableDigest, SisTableKey,
            DEFAULT_SIS_SECURITY_POLICY,
        };

        let mut setup = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity::minimum(),
        )
        .expect("generate setup");
        let decomposed = DigitBlocks::empty(64);
        let hint = AkitaCommitmentHint::singleton(decomposed);
        let inner_commit_matrix = InnerCommitMatrixParams::try_new_with_min_rank(
            SisTableKey {
                policy: DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
                role: SisMatrixRole::Inner,
                ring_dimension: 64,
                coeff_linf_bound: 32_767,
            },
            1,
        )
        .expect("audited prefix A matrix");
        let outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
            SisTableKey {
                policy: DEFAULT_SIS_SECURITY_POLICY,
                table_digest: SisTableDigest::CURRENT,
                modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
                role: SisMatrixRole::Outer,
                ring_dimension: 64,
                coeff_linf_bound: 3,
            },
            inner_commit_matrix.output_rank(),
        )
        .expect("audited prefix B matrix");
        let commitment_rows = outer_commit_matrix.output_rank();
        let commitment_params = PrecommittedLevelParams {
            layout: CommittedGroupProfile {
                version: CommittedGroupProfile::VERSION,
                group: PolynomialGroupLayout::singleton(6),
                num_live_ring_elements_per_claim: 1,
                num_positions_per_block: 1,
                num_live_blocks: 1,
                log_basis_inner: 1,
                num_digits_inner: 1,
                inner_commit_matrix,
                log_basis_outer: 1,
                num_digits_outer: 1,
                outer_commit_matrix,
            },
            log_basis_open: 1,
            fold_challenge_config: akita_challenges::SparseChallengeConfig::pm1_only(0),
            num_digits_open: 1,
            num_digits_fold: 1,
        };
        setup
            .prefix_slots
            .insert(SetupPrefixSlot {
                id: setup_prefix_slot_id(1, commitment_params),
                natural_len: 1,
                padded_len: 3,
                commitment: SetupPrefixPublicCommitment {
                    rows: vec![
                        RingVec::from_coeffs(vec![Prime128Offset275::default(); 64]);
                        commitment_rows
                    ],
                },
                hint,
            })
            .expect("insert malformed slot");

        let err = setup
            .check()
            .expect_err("prover setup check must reject invalid prefix slots");
        assert!(err.to_string().contains("padded_len"));
    }
}
