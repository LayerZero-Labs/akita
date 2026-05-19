//! Prover setup artifact and config-free setup expansion helpers.

use crate::kernels::crt_ntt::{build_ntt_slot, NttSlotCache};
use crate::kernels::matrix::{derive_public_matrix_flat, sample_public_matrix_seed};
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_serialization::{SerializationError, Valid};
use akita_types::{
    AkitaExpandedSetup, AkitaSetupSeed, AkitaVerifierSetup, TieredSetupCommitments,
    TieredSetupProverExtras,
};
use std::sync::{Arc, OnceLock};

/// Lazy cache of the tiered setup material for `S`.
///
/// Holds both verifier-derivable [`TieredSetupCommitments`] and the
/// prover-only [`TieredSetupProverExtras`] under one
/// [`OnceLock`]. The first proof that hits a particular tier shape
/// populates the cache via [`AkitaProverSetup::tiered_s_cache_get_or_init`];
/// subsequent proofs reuse it.
#[derive(Debug, Clone)]
pub struct TieredSetupCachedMaterial<F: FieldCore, const D: usize> {
    /// Verifier-derivable B-side commitments + meta-tier binding.
    pub commitments: TieredSetupCommitments<F, D>,
    /// Prover-only digit material for recursive opening.
    pub extras: TieredSetupProverExtras<F, D>,
}

/// Prover setup artifact (expanded setup + single shared NTT cache).
///
/// The NTT cache is tied to a specific ring dimension D and covers the full
/// shared backing matrix. Role-specific mat-vec operations use row slicing and
/// input-vector-length column clamping.
///
/// `tiered_s_cache` lazily memoizes the tiered B-side commitment to `S`
/// plus the prover-only digit material required to open `S` recursively
/// at fold levels with `use_setup_claim_reduction = true`. Cloning a
/// prover setup shares the cache via the inner `Arc`.
#[derive(Debug, Clone)]
pub struct AkitaProverSetup<F: FieldCore, const D: usize> {
    /// Expanded matrix stage used by both prover and verifier.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
    /// Shared NTT cache for the backing matrix at ring dimension D.
    pub ntt_shared: NttSlotCache<D>,
    /// Tiered `S` commitment + prover extras, lazy on first use.
    pub tiered_s_cache: Arc<OnceLock<TieredSetupCachedMaterial<F, D>>>,
}

impl<F: FieldCore, const D: usize> PartialEq for AkitaProverSetup<F, D> {
    fn eq(&self, other: &Self) -> bool {
        // The tiered cache is purely an optimization and ignored for
        // equality; two prover setups with the same expanded setup +
        // NTT cache are considered equal regardless of whether the
        // tiered material has been materialized.
        self.expanded == other.expanded && self.ntt_shared == other.ntt_shared
    }
}

impl<F: FieldCore, const D: usize> Eq for AkitaProverSetup<F, D> {}

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
        F: CanonicalField + RandomSampling,
    {
        let max_total = max_rows
            .checked_mul(max_stride)
            .ok_or_else(|| AkitaError::InvalidSetup("conservative total overflow".to_string()))?;
        let public_matrix_seed = sample_public_matrix_seed();
        let shared_flat = derive_public_matrix_flat::<F, D>(max_total, &public_matrix_seed);
        let ntt_shared = build_ntt_slot(shared_flat.ring_view::<D>(1, max_total))?;

        let expanded = Arc::new(AkitaExpandedSetup {
            seed: AkitaSetupSeed {
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
            tiered_s_cache: Arc::new(OnceLock::new()),
        })
    }

    /// Derive a verifier setup from this prover setup.
    ///
    /// Book §5 / Figure 12 line 817 names `C_S` as a preprocessed
    /// verifier input. The verifier's tiered-S derivation builds an
    /// NTT slot cache over the shared matrix; pre-populate it here at
    /// setup time so the first `batched_verify` call does not pay the
    /// ~1-2 s NTT preprocessing cost. Subsequent calls hit the cache
    /// as before. Soundness unchanged (derivation is deterministic in
    /// `self.expanded`).
    pub fn verifier_setup(&self) -> AkitaVerifierSetup<F>
    where
        F: CanonicalField + 'static,
    {
        let v = AkitaVerifierSetup::new(self.expanded.clone());
        let _ = v.ntt_shared_get_or_init::<D>();
        v
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
            tiered_s_cache: Arc::new(OnceLock::new()),
        })
    }

    /// Get or lazily initialize the tiered `S` material under the supplied
    /// derivation closure.
    ///
    /// The first call to this method materializes the tiered B-side
    /// commitments and prover extras via `derive`; subsequent calls
    /// return the cached value. Cloning the setup shares the cache, so
    /// the cost is paid at most once across all callers.
    ///
    /// # Errors
    ///
    /// Returns whatever `derive` returns on its first invocation.
    ///
    /// # Panics
    ///
    /// Panics if a concurrent caller's `set` succeeded but a subsequent
    /// `get` returns `None`. This indicates an internal `OnceLock`
    /// invariant violation and is unreachable under correct std
    /// behavior.
    pub fn tiered_s_cache_get_or_init<E>(
        &self,
        derive: impl FnOnce() -> Result<TieredSetupCachedMaterial<F, D>, E>,
    ) -> Result<&TieredSetupCachedMaterial<F, D>, E> {
        if let Some(cached) = self.tiered_s_cache.get() {
            return Ok(cached);
        }
        let materialized = derive()?;
        // Race: another caller may have populated the slot between the
        // `get` above and the `set` below. `set` returns `Err(materialized)`
        // in that case; we discard the loser and return the winner.
        let _ = self.tiered_s_cache.set(materialized);
        Ok(self
            .tiered_s_cache
            .get()
            .expect("tiered_s_cache initialized in this call or by a concurrent caller"))
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for AkitaProverSetup<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()
    }
}
