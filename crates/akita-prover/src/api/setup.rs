//! Prover setup artifact and config-free setup expansion helpers.

use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_serialization::{AkitaSerialize, SerializationError, Valid};
use akita_types::{
    derive_public_matrix_flat, dispatch_ring_dim_result, sample_public_matrix_seed,
    AkitaExpandedSetup, AkitaSetupSeed, AkitaVerifierSetup, SetupMatrixEnvelope,
    SetupPrefixRegistry, SetupPrefixVerifierRegistry,
};
use std::sync::Arc;

/// Prover setup artifact.
///
/// Ring degree is carried in [`AkitaExpandedSetup`] seed metadata (`gen_ring_dim`),
/// not as a type parameter. Backend-prepared compute state is intentionally not
/// stored here; host code prepares a compute backend from the expanded setup
/// when it wants to prove.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaProverSetup<F: FieldCore> {
    /// Expanded matrix stage used by both prover and verifier.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
    /// Preprocessed setup-prefix commitment slots for setup-claim offloading.
    pub prefix_slots: SetupPrefixRegistry<F>,
}

impl<F: FieldCore> AkitaProverSetup<F> {
    /// Setup envelope ring degree.
    #[must_use]
    pub fn gen_ring_dim(&self) -> usize {
        self.expanded.seed().gen_ring_dim
    }

    /// Generate a prover setup from already-computed setup capacity bounds.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacity calculation overflows or the setup
    /// descriptor cannot be built.
    #[tracing::instrument(skip_all, name = "AkitaProverSetup::generate_with_capacity")]
    pub fn generate_with_capacity(
        gen_ring_dim: usize,
        max_num_vars: usize,
        max_num_batched_polys: usize,
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

        dispatch_ring_dim_result!(gen_ring_dim, |D| {
            let shared_flat = derive_public_matrix_flat::<F, D>(
                setup_envelope.max_setup_len,
                &public_matrix_seed,
            );
            let expanded = Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared_flat),
            );
            Ok(Self {
                expanded,
                prefix_slots: SetupPrefixRegistry::new(),
            })
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
    /// # Errors
    ///
    /// Returns an error if setup seed/matrix metadata is inconsistent.
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
        let gen_ring_dim = expanded.seed().gen_ring_dim;
        expanded
            .shared_matrix()
            .total_ring_elements_at_dyn(gen_ring_dim)?;
        let expanded = Arc::new(expanded);
        Ok(Self {
            expanded,
            prefix_slots: SetupPrefixRegistry::new(),
        })
    }

    /// Wrap a pre-built [`AkitaExpandedSetup`] in a prover setup.
    ///
    /// # Errors
    ///
    /// Returns an error if the expanded setup is not valid for this field.
    pub fn from_expanded(expanded: AkitaExpandedSetup<F>) -> Result<Self, AkitaError>
    where
        F: CanonicalField + RandomSampling + Valid,
    {
        Self::from_validated_expanded(expanded)
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
            32,
            8,
            1,
            SetupMatrixEnvelope { max_setup_len: 0 },
        )
        .expect_err("zero setup length must not produce an undecodable setup");
        assert!(zero_len.to_string().contains("max_setup_len"));
    }

    #[test]
    fn prover_setup_check_validates_prefix_slots() {
        use akita_types::{
            AkitaCommitmentHint, ErasedCommitmentHint, FlatDigitBlocks, FlatRingVec,
            SetupPrefixPublicCommitment, SetupPrefixSlot, SetupPrefixSlotId,
        };

        let mut setup = AkitaProverSetup::<Prime128Offset275>::generate_with_capacity(
            32,
            8,
            1,
            SetupMatrixEnvelope { max_setup_len: 1 },
        )
        .expect("generate setup");
        let decomposed = FlatDigitBlocks::<32>::from_blocks(vec![Vec::new()]);
        let recomposed = vec![Vec::new()];
        let hint =
            AkitaCommitmentHint::singleton_with_recomposed_inner_rows(decomposed, recomposed);
        setup
            .prefix_slots
            .insert(SetupPrefixSlot {
                id: SetupPrefixSlotId {
                    setup_seed_digest: [1u8; 32],
                    d_setup: 32,
                    natural_len: 1,
                    n_prefix: 3,
                    level_params_digest: [2u8; 32],
                },
                natural_len: 1,
                padded_len: 3,
                commitment: SetupPrefixPublicCommitment {
                    rows: vec![FlatRingVec::from_coeffs(vec![Prime128Offset275::zero()])],
                },
                hint: ErasedCommitmentHint::from_typed(hint),
            })
            .expect("insert malformed slot");

        let err = setup
            .check()
            .expect_err("prover setup check must reject invalid prefix slots");
        assert!(err.to_string().contains("n_prefix"));
    }
}
