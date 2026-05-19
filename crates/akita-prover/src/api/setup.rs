//! Prover setup artifact and config-free setup expansion helpers.

use crate::kernels::crt_ntt::{build_ntt_slot, NttSlotCache};
use crate::kernels::matrix::{derive_public_matrix_flat, sample_public_matrix_seed};
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_serialization::{AkitaSerialize, SerializationError, Valid};
use akita_types::{AkitaExpandedSetup, AkitaSetupSeed, AkitaVerifierSetup};
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
        let shared_flat = derive_public_matrix_flat::<F, D>(max_total, &public_matrix_seed);
        let ntt_shared = build_ntt_slot(shared_flat.ring_view::<D>(1, max_total))?;

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

    /// Wrap a pre-built [`AkitaExpandedSetup`] in a prover setup by
    /// reconstructing the shared NTT cache at ring dimension `D`.
    ///
    /// # Errors
    ///
    /// Returns an error if the NTT cache cannot be built for the current
    /// field/ring-dimension pair.
    pub fn from_expanded(expanded: AkitaExpandedSetup<F>) -> Result<Self, AkitaError>
    where
        F: CanonicalField,
    {
        let expanded = Arc::new(expanded);
        let total = expanded.shared_matrix.total_ring_elements_at::<D>();
        let ntt_shared = build_ntt_slot(expanded.shared_matrix.ring_view::<D>(1, total))?;
        Ok(Self {
            expanded,
            ntt_shared,
        })
    }
}

impl<F: FieldCore + Valid + AkitaSerialize, const D: usize> Valid for AkitaProverSetup<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()
    }
}
