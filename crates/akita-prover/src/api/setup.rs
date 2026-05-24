//! Prover setup artifact and config-free setup expansion helpers.

use crate::kernels::crt_ntt::{build_ntt_slot, NttSlotCache};
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_serialization::{AkitaSerialize, SerializationError, Valid};
use akita_types::{
    derive_public_matrix_flat, sample_public_matrix_seed, validate_public_matrix_matches_seed,
    AkitaExpandedSetup, AkitaSetupSeed, AkitaVerifierSetup,
};
use std::sync::Arc;

/// Prover setup artifact (expanded setup + single shared NTT cache).
///
/// The NTT cache is tied to a specific ring dimension D and covers the full
/// shared backing matrix. Role-specific mat-vec operations use row slicing and
/// input-vector-length column clamping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaProverSetup<F: FieldCore, const D: usize> {
    /// Expanded matrix stage used by both prover and verifier.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
    /// Shared NTT cache for the backing matrix at ring dimension D.
    pub ntt_shared: NttSlotCache<D>,
}

impl<F: FieldCore, const D: usize> AkitaProverSetup<F, D> {
    /// Generate a prover setup from already-computed setup capacity bounds.
    ///
    /// The caller supplies config-derived capacity bounds. This constructor
    /// owns only the concrete prover artifact: matrix expansion plus the shared NTT
    /// cache for the chosen ring dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacity calculation overflows or if the NTT
    /// cache cannot be built for the current field/ring-dimension pair.
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
        let ntt_shared = build_ntt_slot(shared_flat.ring_view::<D>(1, max_total)?)?;
        let expanded = Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared_flat),
        );

        Ok(Self {
            expanded,
            ntt_shared,
        })
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
    /// example through checked setup deserialization. This avoids re-deriving
    /// the seed-bound public matrix a second time while rebuilding prover NTT
    /// caches.
    ///
    /// # Errors
    ///
    /// Returns an error if the NTT cache cannot be built for the current
    /// field/ring-dimension pair.
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
    /// `D`, or if the NTT cache cannot be built.
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
        let total = expanded.shared_matrix().total_ring_elements_at::<D>()?;
        let ntt_shared = build_ntt_slot(expanded.shared_matrix().ring_view::<D>(1, total)?)?;
        Ok(Self {
            expanded,
            ntt_shared,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;
    use akita_types::MAX_SETUP_MATRIX_FIELD_ELEMENTS;

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
    fn generate_with_capacity_rejects_setup_larger_than_decode_cap() {
        const D: usize = 32;
        let oversized_rows = MAX_SETUP_MATRIX_FIELD_ELEMENTS / D + 1;

        let err = AkitaProverSetup::<Prime128Offset275, D>::generate_with_capacity(
            8,
            1,
            1,
            oversized_rows,
            1,
        )
        .expect_err("generation must not create setup bytes that decode rejects");

        assert!(err.to_string().contains("length"));
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

impl<F: FieldCore + RandomSampling + Valid + AkitaSerialize, const D: usize> Valid
    for AkitaProverSetup<F, D>
{
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()
    }
}
