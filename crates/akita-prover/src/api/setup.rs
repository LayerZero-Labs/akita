//! Prover setup artifact and config-free setup expansion helpers.

use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_serialization::{AkitaSerialize, SerializationError, Valid};
use akita_types::{
    derive_public_matrix_flat, sample_public_matrix_seed, validate_public_matrix_matches_seed,
    AkitaExpandedSetup, AkitaSetupSeed, AkitaVerifierSetup,
};
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
        let seed = AkitaSetupSeed {
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
            max_stride,
            gen_ring_dim: D,
            total_ring_elements: max_total,
            public_matrix_seed,
        };
        seed.check().map_err(|err| {
            AkitaError::InvalidSetup(format!("setup seed validation failed: {err}"))
        })?;

        let shared_flat = derive_public_matrix_flat::<F, D>(max_total, &public_matrix_seed);
        let expanded = Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared_flat),
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
        validate_public_matrix_matches_seed(expanded.shared_matrix(), expanded.seed()).map_err(
            |err| AkitaError::InvalidSetup(format!("expanded setup validation failed: {err}")),
        )?;
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
    /// Returns an error if the setup's generation dimension does not match
    /// `D` or its internal shape metadata is malformed.
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
        if expanded.seed().gen_ring_dim != D {
            return Err(AkitaError::InvalidSetup(format!(
                "expanded setup ring dimension {} does not match prover D={D}",
                expanded.seed().gen_ring_dim
            )));
        }
        if expanded.shared_matrix().gen_ring_dim() != expanded.seed().gen_ring_dim {
            return Err(AkitaError::InvalidSetup(
                "expanded setup matrix generation dimension does not match setup seed".to_string(),
            ));
        }
        if expanded.shared_matrix().total_ring_elements() != expanded.seed().total_ring_elements {
            return Err(AkitaError::InvalidSetup(
                "expanded setup matrix length does not match setup seed".to_string(),
            ));
        }
        let expanded = Arc::new(expanded);
        expanded.shared_matrix().total_ring_elements_at::<D>()?;
        Ok(Self { expanded })
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

impl<F: FieldCore + RandomSampling + Valid + AkitaSerialize, const D: usize> Valid
    for AkitaProverSetup<F, D>
{
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;

    #[test]
    fn validated_expanded_setup_rejects_mismatched_ring_dimension() {
        let setup =
            AkitaProverSetup::<Prime128Offset275, 64>::generate_with_capacity(8, 1, 1, 1, 1)
                .expect("generate D=64 setup");
        let expanded = (*setup.expanded).clone();

        let err = AkitaProverSetup::<Prime128Offset275, 32>::from_validated_expanded(expanded)
            .expect_err("D=64 setup must not be reinterpreted as D=32");

        assert!(err.to_string().contains("ring dimension 64"));
    }

    #[test]
    fn generate_with_capacity_rejects_zero_rows_and_stride() {
        let zero_rows =
            AkitaProverSetup::<Prime128Offset275, 32>::generate_with_capacity(8, 1, 1, 0, 1)
                .expect_err("zero rows must not produce an undecodable setup");
        assert!(zero_rows.to_string().contains("total_ring_elements"));

        let zero_stride =
            AkitaProverSetup::<Prime128Offset275, 32>::generate_with_capacity(8, 1, 1, 1, 0)
                .expect_err("zero stride must not produce an undecodable setup");
        assert!(zero_stride.to_string().contains("max_stride"));
    }
}
