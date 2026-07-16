//! Prover setup artifact and config-free setup expansion helpers.

use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_serialization::{AkitaSerialize, SerializationError, Valid};
use akita_types::{
    derive_public_matrix_flat, dispatch_for_field, sample_public_matrix_seed, AkitaExpandedSetup,
    AkitaSetupSeed, AkitaVerifierSetup, SetupMatrixEnvelope, SetupPrefixProverRegistry,
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
    /// Setup envelope ring degree.
    #[must_use]
    pub fn gen_ring_dim(&self) -> usize {
        self.expanded.seed().gen_ring_dim
    }

    /// Reject use of this setup when the root envelope ring degree mismatches.
    ///
    /// # Errors
    ///
    /// Returns an error when `root_d` does not match [`Self::gen_ring_dim`].
    #[inline]
    pub fn ensure_root_ring_dim(&self, root_d: usize) -> Result<(), AkitaError> {
        if self.gen_ring_dim() != root_d {
            return Err(AkitaError::InvalidInput(format!(
                "setup gen_ring_dim={} does not match scheme root ring degree {root_d}",
                self.gen_ring_dim()
            )));
        }
        Ok(())
    }

    /// Generate a prover setup from already-computed setup capacity bounds.
    ///
    /// The caller supplies config-derived capacity bounds, including the
    /// setup-time generation ring dimension `gen_ring_dim` (the max ring
    /// dimension across the config's schedule policy/catalog). This constructor
    /// owns only the concrete prover artifact: matrix expansion for the chosen
    /// capacity envelope.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacity calculation overflows, `gen_ring_dim` is
    /// unsupported, or the setup descriptor cannot be built.
    #[tracing::instrument(skip_all, name = "AkitaProverSetup::generate_with_capacity")]
    pub fn generate_with_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        gen_ring_dim: usize,
        setup_envelope: SetupMatrixEnvelope,
    ) -> Result<Self, AkitaError>
    where
        F: CanonicalField + RandomSampling + AkitaSerialize,
    {
        let public_matrix_seed = sample_public_matrix_seed();
        let seed = AkitaSetupSeed {
            max_num_vars,
            max_num_batched_polys,
            gen_ring_dim,
            max_setup_len: setup_envelope.max_setup_len,
            public_matrix_seed,
        };
        seed.check().map_err(|err| {
            AkitaError::InvalidSetup(format!("setup seed validation failed: {err}"))
        })?;

        // Matrix expansion still needs the concrete generation ring dimension;
        // dispatch on the runtime `gen_ring_dim` at this single kernel-entry
        // boundary (the canonical akita-types dispatcher; D-free `Self` result).
        let expanded = dispatch_for_field!(ProtocolDispatchSlot::Envelope, F, gen_ring_dim, |D| {
            let shared_flat = derive_public_matrix_flat::<F, D>(
                setup_envelope.max_setup_len,
                &public_matrix_seed,
            );
            Ok::<_, AkitaError>(Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    seed.clone(),
                    shared_flat,
                ),
            ))
        })?;

        Ok(Self {
            expanded,
            prefix_slots: SetupPrefixProverRegistry::new(),
        })
    }

    /// Derive a verifier setup from this prover setup.
    ///
    /// # Errors
    ///
    /// Returns an error if prover prefix-slot metadata cannot be converted into
    /// verifier-visible prefix slots.
    pub fn verifier_setup(&self) -> Result<AkitaVerifierSetup<F>, AkitaError> {
        let mut prefix_slots = SetupPrefixVerifierRegistry::new();
        prefix_slots.replace_from_prover_registry(&self.prefix_slots)?;
        Ok(AkitaVerifierSetup {
            expanded: self.expanded.clone(),
            prefix_slots,
        })
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
    /// Returns an error if the setup's generation dimension is unsupported, the
    /// seed and matrix disagree, or its internal shape metadata is malformed.
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
        if expanded.shared_matrix().gen_ring_dim() != expanded.seed().gen_ring_dim {
            return Err(AkitaError::InvalidSetup(
                "expanded setup matrix generation dimension does not match setup seed".to_string(),
            ));
        }
        if expanded.shared_matrix().total_ring_elements() != expanded.seed().max_setup_len {
            return Err(AkitaError::InvalidSetup(
                "expanded setup matrix length does not match setup seed".to_string(),
            ));
        }
        let expanded = Arc::new(expanded);
        // Re-assert that the generation ring dimension is one we can actually
        // materialize a typed matrix view at (the invariant the const generic
        // `D` used to enforce at compile time). The dispatcher rejects an
        // unsupported `gen_ring_dim`; `total_ring_elements_at::<D>` re-checks the
        // matrix is an exact multiple of the ring dimension.
        let gen_ring_dim = expanded.seed().gen_ring_dim;
        dispatch_for_field!(ProtocolDispatchSlot::Envelope, F, gen_ring_dim, |D| {
            expanded.shared_matrix().total_ring_elements_at::<D>()?;
            Ok::<_, AkitaError>(())
        })?;
        Ok(Self {
            expanded,
            prefix_slots: SetupPrefixProverRegistry::new(),
        })
    }

    /// Assert that this setup's generation ring dimension is compatible with a
    /// schedule level's ring dimension.
    ///
    /// Re-asserts the §6 invariant the const generic `D` used to enforce: the
    /// setup was generated at `gen_ring_dim` (the max ring dimension across the
    /// config's schedule policy), and every level's ring dimension must divide
    /// it so the level can view a typed slice of the shared matrix. For uniform-D
    /// presets `gen_ring_dim == level_ring_dim`; this generalizes the former
    /// `gen_ring_dim == D` equality to the divisibility relation a future mixed-D
    /// catalog needs.
    ///
    /// # Errors
    ///
    /// Returns an error if `level_ring_dim` is zero or does not divide the
    /// setup's `gen_ring_dim`.
    pub fn assert_level_ring_dim_compatible(
        &self,
        level_ring_dim: usize,
    ) -> Result<(), AkitaError> {
        let gen_ring_dim = self.expanded.seed().gen_ring_dim;
        if level_ring_dim == 0 || !gen_ring_dim.is_multiple_of(level_ring_dim) {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule level ring dimension {level_ring_dim} is not compatible with setup \
                 generation ring dimension {gen_ring_dim}"
            )));
        }
        Ok(())
    }
}

impl<F: FieldCore + RandomSampling + Valid + AkitaSerialize> Valid for AkitaProverSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()?;
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
            32,
            SetupMatrixEnvelope { max_setup_len: 0 },
        )
        .expect_err("zero setup length must not produce an undecodable setup");
        assert!(zero_len.to_string().contains("max_setup_len"));
    }

    #[test]
    fn generate_with_capacity_rejects_unsupported_gen_ring_dim() {
        let err = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
            8,
            1,
            48,
            SetupMatrixEnvelope { max_setup_len: 1 },
        )
        .expect_err("unsupported gen_ring_dim must be rejected");
        assert!(
            matches!(err, AkitaError::InvalidInput(_))
                || matches!(err, AkitaError::InvalidSetup(_))
        );
    }

    #[test]
    fn assert_level_ring_dim_compatible_enforces_divisibility() {
        let setup = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
            8,
            1,
            64,
            SetupMatrixEnvelope { max_setup_len: 1 },
        )
        .expect("generate D=64 setup");
        // Uniform-D: level dim equals gen_ring_dim.
        setup
            .assert_level_ring_dim_compatible(64)
            .expect("matching level ring dimension");
        // A divisor is compatible (future mixed-D); a non-divisor is not.
        setup
            .assert_level_ring_dim_compatible(32)
            .expect("divisor level ring dimension");
        setup
            .assert_level_ring_dim_compatible(128)
            .expect_err("non-divisor level ring dimension must be rejected");
        setup
            .assert_level_ring_dim_compatible(0)
            .expect_err("zero level ring dimension must be rejected");
    }

    #[test]
    fn prover_setup_check_validates_prefix_slots() {
        use akita_types::{
            setup_prefix_slot_id, AjtaiKeyParams, AkitaCommitmentHint, DigitBlocks,
            PolynomialGroupLayout, PrecommittedGroupParams, PrecommittedLevelParams, RingVec,
            SetupPrefixPublicCommitment, SetupPrefixSlot, SisMatrixRole, SisModulusProfileId,
            SisTableDigest, DEFAULT_SIS_SECURITY_POLICY,
        };

        let mut setup = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
            8,
            1,
            64,
            SetupMatrixEnvelope { max_setup_len: 1 },
        )
        .expect("generate setup");
        let decomposed = DigitBlocks::empty(64);
        let hint = AkitaCommitmentHint::singleton(decomposed);
        let commitment_params = PrecommittedLevelParams {
            layout: PrecommittedGroupParams {
                group: PolynomialGroupLayout::singleton(6),
                num_live_ring_elements_per_claim: 1,
                num_positions_per_block: 1,
                num_live_blocks: 1,
                fold_challenge_shape: akita_challenges::TensorChallengeShape::Flat,
                log_basis: 1,
                n_a: 1,
                conservative_n_b: 1,
            },
            a_key: AjtaiKeyParams::new_unchecked(
                DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                SisModulusProfileId::Q128OffsetA7F7,
                SisMatrixRole::A,
                1,
                1,
                1,
                64,
            ),
            b_key: AjtaiKeyParams::new_unchecked(
                DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                SisModulusProfileId::Q128OffsetA7F7,
                SisMatrixRole::B,
                1,
                1,
                1,
                64,
            ),
            num_digits_commit: 1,
            num_digits_open: 1,
            num_digits_fold_one: 1,
        };
        setup
            .prefix_slots
            .insert(SetupPrefixSlot {
                id: setup_prefix_slot_id(64, 1, commitment_params),
                natural_len: 1,
                padded_len: 3,
                commitment: SetupPrefixPublicCommitment {
                    rows: vec![RingVec::from_coeffs(vec![Prime128Offset275::default(); 64])],
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
