//! Prover setup artifact and config-free setup expansion helpers.

use crate::crt_ntt::{build_ntt_slot, NttSlotCache};
use crate::matrix::{derive_public_matrix_flat, sample_public_matrix_seed};
use akita_field::{CanonicalField, FieldCore, FieldSampling, HachiError};
use akita_serialization::{SerializationError, Valid};
use akita_types::{HachiExpandedSetup, HachiSetupSeed, HachiVerifierSetup};
use std::sync::Arc;

/// Prover setup artifact (expanded setup + single shared NTT cache).
///
/// The NTT cache is tied to a specific ring dimension D and covers the full
/// shared backing matrix. Role-specific mat-vec operations use row slicing and
/// input-vector-length column clamping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProverSetup<F: FieldCore, const D: usize> {
    /// Expanded matrix stage used by both prover and verifier.
    pub expanded: Arc<HachiExpandedSetup<F>>,
    /// Shared NTT cache for the backing matrix at ring dimension D.
    pub ntt_shared: NttSlotCache<D>,
}

impl<F: FieldCore, const D: usize> HachiProverSetup<F, D> {
    /// Generate a prover setup from already-computed setup capacity bounds.
    ///
    /// The root crate still owns config/schedule policy. This constructor owns
    /// only the concrete prover artifact: matrix expansion plus the shared NTT
    /// cache for the chosen ring dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacity calculation overflows or if the NTT
    /// cache cannot be built for the current field/ring-dimension pair.
    #[tracing::instrument(skip_all, name = "HachiProverSetup::generate_with_capacity")]
    pub fn generate_with_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
        max_rows: usize,
        max_stride: usize,
    ) -> Result<Self, HachiError>
    where
        F: CanonicalField + FieldSampling,
    {
        let max_total = max_rows
            .checked_mul(max_stride)
            .ok_or_else(|| HachiError::InvalidSetup("conservative total overflow".to_string()))?;
        let public_matrix_seed = sample_public_matrix_seed();
        let shared_flat = derive_public_matrix_flat::<F, D>(max_total, &public_matrix_seed);
        let ntt_shared = build_ntt_slot(shared_flat.ring_view::<D>(1, max_total))?;

        let expanded = Arc::new(HachiExpandedSetup {
            seed: HachiSetupSeed {
                max_num_vars,
                max_num_batched_polys,
                max_num_points,
                max_stride,
                public_matrix_seed,
            },
            shared_matrix: shared_flat,
        });

        Ok(Self {
            expanded,
            ntt_shared,
        })
    }

    /// Derive a verifier setup from this prover setup.
    #[must_use]
    pub fn verifier_setup(&self) -> HachiVerifierSetup<F> {
        HachiVerifierSetup {
            expanded: self.expanded.clone(),
        }
    }

    /// Wrap a pre-built [`HachiExpandedSetup`] in a prover setup by
    /// reconstructing the shared NTT cache at ring dimension `D`.
    ///
    /// # Errors
    ///
    /// Returns an error if the NTT cache cannot be built for the current
    /// field/ring-dimension pair.
    pub fn from_expanded(expanded: HachiExpandedSetup<F>) -> Result<Self, HachiError>
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

impl<F: FieldCore + Valid, const D: usize> Valid for HachiProverSetup<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()
    }
}
