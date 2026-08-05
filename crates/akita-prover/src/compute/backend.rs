use crate::compute::plans::{
    DenseCommitRowsPlan, OneHotCommitRowsPlan, RecursiveWitnessCommitRowsPlan,
    RingSwitchQuotientRowsPlan, RingSwitchRelationRows, RingSwitchRelationRowsPlan,
    SparseRingCommitRowsPlan,
};
use crate::AkitaProverSetup;
use akita_algebra::CyclotomicRing;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_types::{AkitaExpandedSetup, NttCacheKey};
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
    F: FieldCore + CanonicalField,
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

    /// Ensure explicit setup metadata and backend-prepared state match.
    fn validate_prepared_setup(
        &self,
        prepared: &Self::PreparedSetup,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<(), AkitaError> {
        let prepared_expanded = self.prepared_expanded_setup(prepared);
        if prepared_expanded.seed() != expanded.seed() {
            return Err(AkitaError::InvalidSetup(
                "prepared compute context was built for a different setup".to_string(),
            ));
        }
        Ok(())
    }
}

/// Paired negacyclic and cyclic products for one compression input.
pub struct CompressionRowsProducts<F: FieldCore, const D: usize> {
    /// Negacyclic image committed by this map or passed to the next map.
    pub negacyclic: Vec<CyclotomicRing<F, D>>,
    /// Cyclic product used to construct the map's quotient witness.
    pub cyclic: Vec<CyclotomicRing<F, D>>,
}

/// Exact-prefix compression matrix operations.
pub trait CompressionComputeBackend<F>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
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
}

/// Negacyclic digit mat-vec operations shared by commitment and protocol code.
pub trait DigitRowsComputeBackend<F>:
    ComputeBackendSetup<F> + CompressionComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    /// Negacyclic single-input digit mat-vec rows.
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;
}

/// Cyclic digit mat-vec operations needed by ring-switch relation code.
pub trait CyclicRowsComputeBackend<F>: DigitRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
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

/// Commitment row operations for migrated root/ring commitment work.
pub trait CommitmentComputeBackend<F>: DigitRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    /// Dense A-side commit rows.
    fn dense_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: DenseCommitRowsPlan<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>;

    /// One-hot A-side commit rows.
    fn onehot_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: OneHotCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>;

    /// Sparse signed-ring A-side commit rows.
    fn sparse_ring_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: SparseRingCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>;

    /// Recursive witness A-side commit rows.
    fn recursive_witness_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RecursiveWitnessCommitRowsPlan<'_, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>;
}

/// Ring-switch relation operations for migrated proving work.
pub trait RingSwitchComputeBackend<F>: CyclicRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    /// Fused cyclic/quotient rows used by ring-switch finalization.
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RingSwitchRelationRowsPlan<'_, D>,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField;

    /// A-side quotient rows for an additional public-row segment.
    fn ring_switch_quotient_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RingSwitchQuotientRowsPlan<'_, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: HalvingField;
}

/// Full first-PR prover compute surface.
pub trait ProverComputeBackend<F>:
    CommitmentComputeBackend<F> + RingSwitchComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
}

impl<F, B> ProverComputeBackend<F> for B
where
    F: FieldCore + CanonicalField,
    B: CommitmentComputeBackend<F> + RingSwitchComputeBackend<F>,
{
}
