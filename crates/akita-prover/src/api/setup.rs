//! Prover setup artifact and config-free setup expansion helpers.

use crate::kernels::matrix::{derive_public_matrix_flat, sample_public_matrix_seed};
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_serialization::{AkitaSerialize, SerializationError, Valid};
use akita_types::{AkitaExpandedSetup, AkitaSetupSeed, AkitaVerifierSetup};
use std::sync::Arc;

/// Prover setup artifact.
///
/// Backend-prepared compute state is intentionally not stored here. Host code
/// prepares a compute backend from the expanded setup when it wants to prove.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaProverSetup<F: FieldCore, const D: usize> {
    /// Expanded matrix stage used by both prover and verifier.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
}

impl<F: FieldCore, const D: usize> AkitaProverSetup<F, D> {
    /// Generate a prover setup from already-computed setup capacity bounds.
    ///
    /// The caller supplies config-derived capacity bounds. This constructor
    /// owns only the concrete prover artifact: matrix expansion for the chosen
    /// capacity envelope.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacity calculation overflows or the setup
    /// descriptor cannot be built.
    #[tracing::instrument(skip_all, name = "AkitaProverSetup::generate_with_capacity")]
    pub fn generate_with_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
        max_rows: usize,
        max_stride: usize,
    ) -> Result<Self, AkitaError>
    where
        F: CanonicalField + RandomSampling + AkitaSerialize,
    {
        let max_total = max_rows
            .checked_mul(max_stride)
            .ok_or_else(|| AkitaError::InvalidSetup("conservative total overflow".to_string()))?;
        let public_matrix_seed = sample_public_matrix_seed();
        let shared_flat = derive_public_matrix_flat::<F, D>(max_total, &public_matrix_seed);

        let seed = AkitaSetupSeed {
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
            max_stride,
            public_matrix_seed,
        };
        let expanded = Arc::new(
            AkitaExpandedSetup::from_parts(seed, shared_flat).map_err(|err| {
                AkitaError::InvalidSetup(format!("setup descriptor digest: {err}"))
            })?,
        );

        Ok(Self { expanded })
    }

    /// Derive a verifier setup from this prover setup.
    #[must_use]
    pub fn verifier_setup(&self) -> AkitaVerifierSetup<F> {
        AkitaVerifierSetup {
            expanded: self.expanded.clone(),
        }
    }

    /// Wrap a pre-built [`AkitaExpandedSetup`] in a prover setup.
    ///
    /// # Errors
    ///
    /// Returns an error if the expanded setup is not valid for this field.
    pub fn from_expanded(expanded: AkitaExpandedSetup<F>) -> Result<Self, AkitaError>
    where
        F: CanonicalField + Valid,
    {
        expanded
            .check()
            .map_err(|err| AkitaError::InvalidSetup(format!("expanded setup validation: {err}")))?;
        expanded.shared_matrix.total_ring_elements_at::<D>()?;
        let expanded = Arc::new(expanded);
        Ok(Self { expanded })
    }
}

impl<F: FieldCore + Valid + AkitaSerialize, const D: usize> Valid for AkitaProverSetup<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()
    }
}
