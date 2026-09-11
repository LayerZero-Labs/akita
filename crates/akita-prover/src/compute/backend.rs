use crate::compute::requirements::RoutedNttRequirement;
use crate::AkitaProverSetup;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::{AkitaExpandedSetup, NttCacheKey};
use jolt_field::{CanonicalEncoding, Field};
use std::sync::Arc;

/// Process-local identity of one physical backend cache owner.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NttCacheOwnerId(usize);

impl NttCacheOwnerId {
    fn from_prepared<T>(prepared: &T) -> Self {
        Self((prepared as *const T).cast::<()>() as usize)
    }
}

/// Shared prepared-setup contract for prover compute backends.
///
/// `PreparedSetup` is keyed by exact [`NttCacheKey`] prefixes at runtime.
/// Preparation leaves derived caches empty; matrix-consuming kernels acquire
/// only the exact transform prefixes they need.
pub trait ComputeBackendSetup<F>: Send + Sync
where
    F: Field + CanonicalEncoding,
{
    /// Backend-prepared setup (ring dimension is a runtime cache key, not a type param).
    type PreparedSetup: Send + Sync;

    /// Prepare backend state from a prover setup wrapper.
    ///
    /// Returns prepared backend state with derived caches initially empty.
    fn prepare_setup(
        &self,
        setup: &AkitaProverSetup<F>,
    ) -> Result<Self::PreparedSetup, AkitaError> {
        self.prepare_expanded(setup.expanded.clone())
    }

    /// Prepare backend state from already-expanded setup data.
    ///
    /// Returns an empty NTT cache.
    fn prepare_expanded(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup, AkitaError>;

    /// Build the cache for `key` if absent.
    fn ensure_ntt_slot(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<(), AkitaError>;

    /// Whether this routed request remains resident for its operation cluster.
    ///
    /// Prewarming and planned memory reporting both use this decision. The
    /// default retains every requirement. A backend that streams an operation
    /// must override this method with the same policy used by its runtime
    /// kernel.
    fn ntt_requirement_is_cached(
        &self,
        _prepared: &Self::PreparedSetup,
        _requirement: RoutedNttRequirement,
    ) -> Result<bool, AkitaError> {
        Ok(true)
    }

    /// Process-local identity used to deduplicate physically shared cache state.
    ///
    /// The default treats the prepared value itself as the cache owner. A
    /// backend whose distinct prepared values share interior cache storage must
    /// override this method with that storage's identity.
    fn ntt_cache_owner_id(&self, prepared: &Self::PreparedSetup) -> NttCacheOwnerId {
        NttCacheOwnerId::from_prepared(prepared)
    }

    /// Planned resident bytes for one independently stored exact cache entry.
    ///
    /// The result excludes any fixed cache-container overhead so callers may
    /// sum distinct `(D, domain)` entries after max-joining their prefixes.
    fn planned_ntt_cache_entry_bytes(
        &self,
        _prepared: &Self::PreparedSetup,
        _key: NttCacheKey,
    ) -> Result<usize, AkitaError> {
        Err(AkitaError::InvalidSetup(
            "compute backend does not expose planned NTT cache bytes".into(),
        ))
    }

    /// Expanded setup used to prepare this backend context.
    fn prepared_expanded_setup<'a>(
        &self,
        prepared: &'a Self::PreparedSetup,
    ) -> &'a AkitaExpandedSetup<F>;

    /// Drop backend-designated releasable NTT slots and return the freed bytes.
    /// Slots rebuild on next use. Backends may retain small reusable caches and
    /// backends without droppable caches return `Ok(0)`.
    ///
    /// Release must not invalidate active readers. A backend may require the
    /// caller to prevent concurrent cache construction if release must leave
    /// the cache empty.
    ///
    /// # Errors
    ///
    /// Returns an error when backend-owned cache state cannot be updated.
    fn release_built_ntt_slots(&self, prepared: &Self::PreparedSetup) -> Result<usize, AkitaError> {
        let _unused = prepared;
        Ok(0)
    }

    /// Ensure explicit setup metadata and backend-prepared state match.
    fn validate_prepared_setup(
        &self,
        prepared: &Self::PreparedSetup,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<(), AkitaError> {
        let prepared_expanded = self.prepared_expanded_setup(prepared);
        if prepared_expanded.descriptor() != expanded.descriptor() {
            return Err(AkitaError::InvalidSetup(
                "prepared compute context was built for a different setup".to_string(),
            ));
        }
        Ok(())
    }
}

/// Paired negacyclic and cyclic products for one compression input.
pub struct CompressionRowsProducts<F: Field, const D: usize> {
    /// Negacyclic image committed by this map or passed to the next map.
    pub negacyclic: Vec<CyclotomicRing<F, D>>,
    /// Cyclic product used to construct the map's quotient witness.
    pub cyclic: Vec<CyclotomicRing<F, D>>,
}

/// Exact-prefix compression matrix operations.
pub trait CompressionComputeBackend<F>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    /// Current byte footprint of backend-owned compression caches, when exposed.
    ///
    /// This is operational metadata and does not participate in protocol sizing.
    fn compression_cache_bytes(&self, _prepared: &Self::PreparedSetup) -> Option<usize> {
        None
    }

    /// Exact-shape rank-one negative-binary compression products over one matrix prefix.
    ///
    /// Compression-capable backends must implement this explicitly. There is no
    /// default coefficient-form fallback that would hide missing support.
    fn compression_rows_products<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        digit_vectors: &[&[[i8; D]]],
    ) -> Result<Vec<CompressionRowsProducts<F, D>>, AkitaError>;

    /// Exact-shape negacyclic-only compression products for reduced evaluation.
    fn compression_negacyclic_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        digit_vectors: &[&[[i8; D]]],
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>;
}

/// Negacyclic digit mat-vec operations shared by commitment and protocol code.
pub trait DigitRowsComputeBackend<F>:
    ComputeBackendSetup<F> + CompressionComputeBackend<F>
where
    F: Field + CanonicalEncoding,
{
    /// Negacyclic digit mat-vec rows for an equal-width input batch.
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digit_vectors: &[&[[i8; D]]],
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>;
}

/// Cyclic digit mat-vec operations needed by ring-switch relation code.
pub trait CyclicRowsComputeBackend<F>: DigitRowsComputeBackend<F>
where
    F: Field + CanonicalEncoding,
{
    /// Cyclic single-input digit mat-vec rows.
    fn cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;
}
