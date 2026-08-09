use crate::compute::backend::{
    CompressionComputeBackend, CompressionRowsProducts, ComputeBackendSetup,
    CyclicRowsComputeBackend, DigitRowsComputeBackend,
};
use crate::compute::kernels::{RingSwitchQuotientKernel, RingSwitchRelationKernel};
use crate::compute::operation_plans::{RingSwitchQuotientPlan, RingSwitchRelationPlan};
use crate::compute::plans::{DenseCommitInput, RingSwitchRelationRows};
use crate::compute::requirements::{NttOperationCluster, RoutedNttRequirement};
use crate::kernels::linear::validate_compression_batch_shape;
use crate::kernels::linear::{
    digit_blocks_are_balanced, fused_quotient_matrix_extent,
    fused_split_eq_quotients_prover_bounds, fused_split_eq_quotients_streamed_prover_bounds,
    mat_vec_mul_ntt_dense_digits_i8, mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_i8,
    mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row, mat_vec_mul_ntt_raw_digits_i8,
    mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic, selected_crt_i8_capacity_profile,
    CrtI8CapacityProfile,
};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_types::{
    dispatch_for_field, prepare_ntt_cache, AkitaExpandedSetup, NttCacheKey, NttCacheMode,
    NttTransformDomain, PreparedNttCache,
};
use std::any::Any;
use std::array::from_fn;
use std::collections::HashMap;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

mod compression_cache;
mod ring_switch;
#[cfg(test)]
mod streamed_tests;

use compression_cache::CompressionNttCache;

/// CPU backend using the existing Rust/Rayon kernels.
///
/// The backend owns deployment resource limits. These limits only choose
/// equivalent CPU execution paths and do not affect proof bytes or protocol
/// parameters.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct CpuBackend {
    max_cached_ring_switch_elements: usize,
    onehot_scratch_bytes_per_worker: usize,
}

type NttSlotCell = OnceLock<Result<Arc<ErasedCpuNttCache>, AkitaError>>;

impl CpuBackend {
    /// Default maximum cached extent for a ring-switch NTT operation.
    pub const DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS: usize = 1 << 21;

    /// Default temporary one hot commitment memory per worker.
    pub const DEFAULT_ONEHOT_SCRATCH_BYTES_PER_WORKER: usize = 8 << 20;

    /// CPU backend with the default resource limits.
    pub const DEFAULT: Self = Self {
        max_cached_ring_switch_elements: Self::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS,
        onehot_scratch_bytes_per_worker: Self::DEFAULT_ONEHOT_SCRATCH_BYTES_PER_WORKER,
    };

    /// Creates a CPU backend with explicit resource limits.
    ///
    /// A zero ring-switch limit streams every stream-capable ring-switch
    /// operation. A limit of [`usize::MAX`] retains every supported operation.
    /// The one hot scratch budget must be nonzero. A commitment whose minimum
    /// tile does not fit returns [`AkitaError::InvalidSetup`].
    pub fn with_resource_limits(
        max_cached_ring_switch_elements: usize,
        onehot_scratch_bytes_per_worker: usize,
    ) -> Result<Self, AkitaError> {
        if onehot_scratch_bytes_per_worker == 0 {
            return Err(AkitaError::InvalidSetup(
                "CPU one hot scratch bytes per worker must be nonzero".into(),
            ));
        }
        Ok(Self {
            max_cached_ring_switch_elements,
            onehot_scratch_bytes_per_worker,
        })
    }

    /// Returns the largest ring-switch operation extent retained as an NTT cache.
    pub const fn max_cached_ring_switch_elements(&self) -> usize {
        self.max_cached_ring_switch_elements
    }

    /// Returns the temporary one hot commitment memory allowed per worker.
    pub const fn onehot_scratch_bytes_per_worker(&self) -> usize {
        self.onehot_scratch_bytes_per_worker
    }

    #[inline]
    pub(crate) fn ntt_operation_uses_cache(
        &self,
        cluster: NttOperationCluster,
        num_ring_elements: usize,
    ) -> bool {
        let cached = cluster != NttOperationCluster::RingSwitch
            || num_ring_elements <= self.max_cached_ring_switch_elements;
        tracing::debug!(
            ?cluster,
            num_ring_elements,
            max_cached_ring_switch_elements = self.max_cached_ring_switch_elements,
            cached,
            "CPU NTT execution policy"
        );
        cached
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// CPU-prepared setup keyed by runtime ring dimension.
///
/// NTT caches are keyed by [`NttCacheKey`] and built lazily. Each ring
/// dimension/domain pair retains only its largest requested prefix; a covering
/// cell also serves smaller requests. Each cell makes concurrent construction
/// of that prefix single-flight. Diagnostic compression caches remain in a
/// separate namespace.
#[derive(Debug)]
pub struct CpuPreparedSetup<F: FieldCore> {
    expanded: Arc<AkitaExpandedSetup<F>>,
    shared_ntt: Mutex<HashMap<NttCacheKey, Arc<NttSlotCell>>>,
    compression_ntt: CompressionNttCache,
    ntt_i8_capacity_by_ring_d: Mutex<HashMap<usize, CrtI8CapacityProfile>>,
    #[cfg(test)]
    ntt_slot_build_count: AtomicUsize,
}

struct ErasedCpuNttCache {
    ring_d: usize,
    cache_bytes: usize,
    cache: Arc<dyn Any + Send + Sync>,
}

impl core::fmt::Debug for ErasedCpuNttCache {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ErasedCpuNttCache")
            .field("ring_d", &self.ring_d)
            .field("cache_bytes", &self.cache_bytes)
            .finish_non_exhaustive()
    }
}

/// CRT/NTT profile and universal i8 capacity metadata for a prepared setup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedCrtNttProfile {
    /// Stable profile identifier used by benchmark/report tooling.
    pub profile_id: &'static str,
    /// Number of CRT primes in the selected profile.
    pub num_primes: usize,
    /// Maximum bit length of a CRT prime modulus.
    pub prime_modulus_bits: u32,
    /// Signed storage width used by the CRT NTT representation.
    pub limb_bits: u32,
    /// Largest balanced i8 log basis accepted by prover i8 kernels.
    pub max_i8_log_basis: u32,
    /// Safe accumulation width for balanced i8 digits at `max_i8_log_basis`.
    pub balanced_digit_safe_width: usize,
    /// Safe accumulation width for raw signed i8 recursive-witness inputs.
    pub raw_i8_safe_width: usize,
}

/// One initialized exact-prefix NTT cache entry for diagnostics and profiling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedNttCacheMetric {
    /// Exact cache identity.
    pub key: NttCacheKey,
    /// Bytes used by materialized transform vectors, excluding map metadata.
    pub cache_bytes: usize,
}

impl From<CrtI8CapacityProfile> for PreparedCrtNttProfile {
    fn from(profile: CrtI8CapacityProfile) -> Self {
        Self {
            profile_id: profile.profile_id,
            num_primes: profile.num_primes,
            prime_modulus_bits: profile.prime_modulus_bits,
            limb_bits: profile.limb_bits,
            max_i8_log_basis: profile.max_i8_log_basis,
            balanced_digit_safe_width: profile.balanced_digit_safe_width,
            raw_i8_safe_width: profile.raw_i8_safe_width,
        }
    }
}

impl<F: FieldCore + CanonicalField> CpuPreparedSetup<F> {
    #[cfg(test)]
    pub(crate) fn ntt_slot_build_count(&self) -> usize {
        self.ntt_slot_build_count.load(Ordering::Relaxed)
    }

    pub(crate) fn with_shared_ntt<const D: usize, R>(
        &self,
        key: NttCacheKey,
        f: impl FnOnce(&PreparedNttCache<D>) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        if key.ring_d != D {
            return Err(AkitaError::InvalidSetup(
                "NTT prefix requirement ring dimension does not match kernel".into(),
            ));
        }
        let required_num_field_elements = key.num_field_elements()?;
        if required_num_field_elements > self.expanded.shared_matrix.num_field_elements() {
            return Err(AkitaError::InvalidSetup(format!(
                "NTT prefix requires {required_num_field_elements} field elements but setup has {}",
                self.expanded.shared_matrix.num_field_elements()
            )));
        }
        let slot = prepare_ntt_slot_on_prepared(self, key)?;
        if slot.ring_d != D {
            return Err(AkitaError::InvalidSetup(format!(
                "prepared CPU NTT ring_d mismatch: stored {}, requested {D}",
                slot.ring_d
            )));
        }
        let typed = slot
            .cache
            .downcast_ref::<PreparedNttCache<D>>()
            .ok_or_else(|| AkitaError::InvalidSetup("prepared CPU NTT type mismatch".into()))?;
        f(typed)
    }

    fn with_compression_ntt<const D: usize, R>(
        &self,
        input_width: usize,
        f: impl FnOnce(&PreparedNttCache<D>) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        self.compression_ntt
            .with_ntt(self.expanded.as_ref(), input_width, f)
    }

    /// In-memory byte footprint of all shared setup NTT caches.
    pub fn shared_ntt_cache_bytes(&self) -> usize {
        self.shared_ntt
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter_map(|entry| entry.get())
            .filter_map(|result| result.as_ref().ok())
            .map(|slot| slot.cache_bytes)
            .sum()
    }

    /// Drop every built shared matrix NTT slot and return the bytes freed.
    ///
    /// The small compression NTT cache remains resident across this lifecycle
    /// boundary. Removing each released shared key ensures that a later smaller
    /// request creates its exact extent instead of rebuilding the released
    /// larger prefix. Active readers keep the released slot alive through their
    /// `Arc`.
    /// Construction already in progress is not cancelled, so callers that need
    /// an empty cache after this call must use a quiescent lifecycle boundary.
    pub fn drop_built_ntt_slots(&self) -> Result<usize, AkitaError> {
        let mut freed = 0usize;
        let mut cache = self
            .shared_ntt
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?;
        let mut released_keys = Vec::new();
        for (key, cell) in cache.iter() {
            let built_bytes = cell
                .get()
                .and_then(|result| result.as_ref().ok())
                .map(|slot| slot.cache_bytes);
            if let Some(bytes) = built_bytes {
                freed = freed.checked_add(bytes).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "released shared matrix NTT cache bytes overflow".into(),
                    )
                })?;
                released_keys.push(*key);
            }
        }
        for key in released_keys {
            cache.remove(&key);
        }
        drop(cache);
        if freed > 0 {
            tracing::info!(freed_bytes = freed, "dropped built shared matrix NTT slots");
        }
        Ok(freed)
    }

    /// Initialized shared NTT cache entries in deterministic reporting order.
    pub fn shared_ntt_cache_metrics(&self) -> Result<Vec<PreparedNttCacheMetric>, AkitaError> {
        let cache = self
            .shared_ntt
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?;
        let mut metrics = cache
            .iter()
            .filter_map(|(key, entry)| {
                entry
                    .get()
                    .and_then(|result| result.as_ref().ok())
                    .map(|slot| PreparedNttCacheMetric {
                        key: *key,
                        cache_bytes: slot.cache_bytes,
                    })
            })
            .collect::<Vec<_>>();
        metrics.sort_by_key(|metric| {
            let domain = match metric.key.domain {
                NttTransformDomain::Negacyclic => 0,
                NttTransformDomain::Cyclic => 1,
            };
            (metric.key.ring_d, domain, metric.key.num_ring_elements)
        });
        Ok(metrics)
    }

    /// Planned resident bytes for max-joined exact base-profile cache keys.
    pub fn planned_shared_ntt_cache_bytes(
        &self,
        keys: impl IntoIterator<Item = NttCacheKey>,
    ) -> Result<usize, AkitaError> {
        let mut joined = HashMap::<(usize, NttTransformDomain), usize>::new();
        for key in keys {
            if key.num_field_elements()? > self.expanded.shared_matrix.num_field_elements() {
                return Err(AkitaError::InvalidSetup(
                    "planned NTT prefix exceeds prepared public matrix".into(),
                ));
            }
            joined
                .entry((key.ring_d, key.domain))
                .and_modify(|count| *count = (*count).max(key.num_ring_elements))
                .or_insert(key.num_ring_elements);
        }
        joined
            .into_iter()
            .try_fold(0usize, |total, ((ring_d, _domain), count)| {
                let profile =
                    dispatch_for_field!(ProtocolDispatchSlot::Ntt, F, ring_d, |RING_D| {
                        selected_crt_i8_capacity_profile::<F, RING_D>()
                    })?;
                count
                    .checked_mul(ring_d)
                    .and_then(|bytes| bytes.checked_mul(profile.num_primes))
                    .and_then(|bytes| bytes.checked_mul(core::mem::size_of::<i32>()))
                    .and_then(|bytes| total.checked_add(bytes))
                    .ok_or_else(|| AkitaError::InvalidSetup("planned NTT bytes overflow".into()))
            })
    }

    /// In-memory byte footprint of exact-prefix compression NTT caches.
    pub fn compression_ntt_cache_bytes(&self) -> usize {
        self.compression_ntt.cache_bytes()
    }

    /// Complete in-memory byte footprint of all CPU NTT caches.
    pub fn ntt_cache_bytes(&self) -> Result<usize, AkitaError> {
        self.shared_ntt_cache_bytes()
            .checked_add(self.compression_ntt_cache_bytes())
            .ok_or_else(|| AkitaError::InvalidSetup("CPU NTT cache bytes overflow".into()))
    }

    /// CRT/NTT profile and universal i8 capacity metadata for `ring_d`.
    pub fn shared_ntt_profile(&self, ring_d: usize) -> Result<PreparedCrtNttProfile, AkitaError> {
        self.ntt_i8_capacity_by_ring_d
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("NTT profile lock poisoned".into()))?
            .get(&ring_d)
            .copied()
            .map(Into::into)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "prepared setup has no CRT/i8 capacity profile for ring_d={ring_d}"
                ))
            })
    }
}

fn build_ntt_slot_for_key<F: FieldCore + CanonicalField>(
    expanded: &AkitaExpandedSetup<F>,
    key: NttCacheKey,
) -> Result<ErasedCpuNttCache, AkitaError> {
    dispatch_for_field!(ProtocolDispatchSlot::Ntt, F, key.ring_d, |RING_D| {
        let view = expanded
            .shared_matrix()
            .ring_view::<RING_D>(1, key.num_ring_elements)?;
        let mode = match key.domain {
            NttTransformDomain::Negacyclic => NttCacheMode::Negacyclic,
            NttTransformDomain::Cyclic => NttCacheMode::Cyclic,
        };
        let cache = Arc::new(prepare_ntt_cache(view, mode)?);
        Ok(ErasedCpuNttCache {
            ring_d: RING_D,
            cache_bytes: cache.cache_bytes(),
            cache,
        })
    })
}

fn record_ntt_profile_on_prepared<F: FieldCore>(
    prepared: &CpuPreparedSetup<F>,
    key: NttCacheKey,
    profile: CrtI8CapacityProfile,
) -> Result<(), AkitaError> {
    prepared
        .ntt_i8_capacity_by_ring_d
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("NTT profile lock poisoned".into()))?
        .entry(key.ring_d)
        .or_insert(profile);
    Ok(())
}

fn prepare_ntt_slot_on_prepared<F: FieldCore + CanonicalField>(
    prepared: &CpuPreparedSetup<F>,
    requested_key: NttCacheKey,
) -> Result<Arc<ErasedCpuNttCache>, AkitaError> {
    let profile = dispatch_for_field!(
        ProtocolDispatchSlot::Ntt,
        F,
        requested_key.ring_d,
        |RING_D| selected_crt_i8_capacity_profile::<F, RING_D>()
    )?;
    loop {
        let (key, entry) = {
            let mut cache = prepared
                .shared_ntt
                .lock()
                .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?;
            if let Some((key, entry)) = cache
                .iter()
                .filter(|(key, _)| {
                    key.ring_d == requested_key.ring_d
                        && key.domain == requested_key.domain
                        && key.num_ring_elements >= requested_key.num_ring_elements
                })
                .min_by_key(|(key, _)| key.num_ring_elements)
                .map(|(key, entry)| (*key, Arc::clone(entry)))
            {
                (key, entry)
            } else {
                let entry = Arc::new(OnceLock::new());
                cache.insert(requested_key, Arc::clone(&entry));
                (requested_key, entry)
            }
        };
        let build_result = entry.get_or_init(|| {
            #[cfg(test)]
            prepared
                .ntt_slot_build_count
                .fetch_add(1, Ordering::Relaxed);
            build_ntt_slot_for_key(prepared.expanded.as_ref(), key).map(Arc::new)
        });
        match build_result {
            Ok(slot) => {
                // Keep smaller prefixes available until the larger build has
                // completed successfully.  A failed growth must not evict a
                // working covering candidate; once the new slot is ready the
                // smaller entries are redundant and can be reclaimed.
                let mut cache = prepared
                    .shared_ntt
                    .lock()
                    .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?;
                cache.retain(|cached_key, _| {
                    cached_key.ring_d != key.ring_d
                        || cached_key.domain != key.domain
                        || cached_key.num_ring_elements >= key.num_ring_elements
                });
                drop(cache);
                record_ntt_profile_on_prepared(prepared, key, profile)?;
                return Ok(Arc::clone(slot));
            }
            Err(error) => {
                let mut cache = prepared
                    .shared_ntt
                    .lock()
                    .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?;
                if cache
                    .get(&key)
                    .is_some_and(|current| Arc::ptr_eq(current, &entry))
                {
                    cache.remove(&key);
                }
                drop(cache);
                if key == requested_key {
                    return Err(error.clone());
                }
            }
        }
    }
}

fn ensure_ntt_slot_on_prepared<F: FieldCore + CanonicalField>(
    prepared: &CpuPreparedSetup<F>,
    key: NttCacheKey,
) -> Result<(), AkitaError> {
    prepare_ntt_slot_on_prepared(prepared, key).map(|_| ())
}

fn validate_digit_row_request(
    row_len: usize,
    row_width: usize,
    total_ring_elements: usize,
) -> Result<(), AkitaError> {
    if row_width == 0 {
        return Err(AkitaError::InvalidSetup(
            "prepared setup row width must be nonzero".to_string(),
        ));
    }
    let required = row_len.checked_mul(row_width).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "digit row request overflows: row_len={row_len} row_width={row_width}"
        ))
    })?;
    if required > total_ring_elements {
        return Err(AkitaError::InvalidSetup(format!(
            "digit row request needs {required} setup ring elements but prepared setup has {total_ring_elements}"
        )));
    }
    Ok(())
}

impl<F> ComputeBackendSetup<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    type PreparedSetup = CpuPreparedSetup<F>;

    fn prepare_expanded(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup, AkitaError> {
        Ok(CpuPreparedSetup {
            expanded,
            shared_ntt: Mutex::new(HashMap::new()),
            compression_ntt: CompressionNttCache::default(),
            ntt_i8_capacity_by_ring_d: Mutex::new(HashMap::new()),
            #[cfg(test)]
            ntt_slot_build_count: AtomicUsize::new(0),
        })
    }

    fn ensure_ntt_slot(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<(), AkitaError> {
        ensure_ntt_slot_on_prepared(prepared, key)
    }

    fn ntt_requirement_is_cached(
        &self,
        _prepared: &Self::PreparedSetup,
        requirement: RoutedNttRequirement,
    ) -> Result<bool, AkitaError> {
        Ok(self.ntt_operation_uses_cache(requirement.cluster, requirement.routing_extent))
    }

    fn release_built_ntt_slots(&self, prepared: &Self::PreparedSetup) -> Result<usize, AkitaError> {
        prepared.drop_built_ntt_slots()
    }

    fn planned_ntt_cache_entry_bytes(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<usize, AkitaError> {
        prepared.planned_shared_ntt_cache_bytes([key])
    }

    fn prepared_expanded_setup<'a>(
        &self,
        prepared: &'a Self::PreparedSetup,
    ) -> &'a AkitaExpandedSetup<F> {
        prepared.expanded.as_ref()
    }
}

impl CpuBackend {
    pub(crate) fn dense_commit_rows<F, const D: usize>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        n_a: usize,
        input: DenseCommitInput<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: FieldCore + CanonicalField,
    {
        match input {
            DenseCommitInput::CachedDigits {
                digit_block_slices,
                log_basis_inner,
            } => {
                let row_width = digit_block_slices.first().map_or(0, |digits| digits.len());
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        n_a,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| {
                        mat_vec_mul_ntt_dense_digits_i8(
                            ntt,
                            n_a,
                            row_width,
                            &digit_block_slices,
                            log_basis_inner,
                        )
                    },
                )
            }
            DenseCommitInput::CoeffBlocks {
                block_slices,
                num_digits_inner,
                log_basis_inner,
            } => {
                let row_width = block_slices.first().map_or(Ok(0usize), |block| {
                    block.len().checked_mul(num_digits_inner).ok_or_else(|| {
                        AkitaError::InvalidSetup("dense coefficient row width overflow".to_string())
                    })
                })?;
                if n_a == 1 {
                    prepared.with_shared_ntt::<D, _>(
                        NttCacheKey::from_matrix_shape(
                            D,
                            1,
                            row_width,
                            NttTransformDomain::Negacyclic,
                        )?,
                        |ntt| {
                            Ok(mat_vec_mul_ntt_i8_dense_single_row(
                                ntt,
                                row_width,
                                &block_slices,
                                num_digits_inner,
                                log_basis_inner,
                            )?
                            .into_iter()
                            .map(|ring| vec![ring])
                            .collect())
                        },
                    )
                } else {
                    prepared.with_shared_ntt::<D, _>(
                        NttCacheKey::from_matrix_shape(
                            D,
                            n_a,
                            row_width,
                            NttTransformDomain::Negacyclic,
                        )?,
                        |ntt| {
                            mat_vec_mul_ntt_i8_dense(
                                ntt,
                                n_a,
                                row_width,
                                &block_slices,
                                num_digits_inner,
                                log_basis_inner,
                            )
                        },
                    )
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn recursive_witness_commit_rows<F, const D: usize>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        coeffs: &[[i8; D]],
        n_rows: usize,
        num_positions_per_block: usize,
        num_live_blocks: usize,
        num_digits_inner: usize,
        log_basis_inner: u32,
        known_balanced_log_basis: Option<u32>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: FieldCore + CanonicalField,
    {
        let row_width = num_positions_per_block
            .checked_mul(num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("recursive A width overflow".to_string()))?;
        let minimum_ring_elems = num_live_blocks
            .saturating_sub(1)
            .checked_mul(num_positions_per_block)
            .and_then(|prefix| prefix.checked_add(1))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive witness block extent overflow".to_string())
            })?;
        if num_live_blocks == 0 || coeffs.len() < minimum_ring_elems {
            return Err(AkitaError::InvalidSetup(
                "recursive witness does not cover its live blocks".to_string(),
            ));
        }
        if num_digits_inner == 1 {
            let blocks = coeffs
                .chunks(num_positions_per_block)
                .take(num_live_blocks)
                .collect::<Vec<_>>();
            // The `num_digits_inner == 1` recursive witness is a raw signed-i8
            // coefficient stream. Degree-one fields yield balanced gadget digits
            // (fast predecomposed-digit kernel), but extension-field tensor
            // base-lift packing sums gadget digits and can push coefficients
            // past the balanced range; those must commit through the general
            // raw ring mat-vec instead of the balanced-digit LUT kernel.
            let known_balanced = known_balanced_log_basis
                .is_some_and(|source_log_basis| log_basis_inner >= source_log_basis);
            if known_balanced || digit_blocks_are_balanced(&blocks, row_width, log_basis_inner) {
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        n_rows,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| {
                        mat_vec_mul_ntt_digits_i8(ntt, n_rows, row_width, &blocks, log_basis_inner)
                    },
                )
            } else {
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        n_rows,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| mat_vec_mul_ntt_raw_digits_i8(ntt, n_rows, row_width, &blocks),
                )
            }
        } else {
            let ring_elems: Vec<CyclotomicRing<F, D>> = coeffs
                .iter()
                .map(|digit| {
                    let coeffs = from_fn(|k| F::from_i8(digit[k]));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect();
            let blocks = ring_elems
                .chunks(num_positions_per_block)
                .take(num_live_blocks)
                .collect::<Vec<_>>();
            prepared.with_shared_ntt::<D, _>(
                NttCacheKey::from_matrix_shape(
                    D,
                    n_rows,
                    row_width,
                    NttTransformDomain::Negacyclic,
                )?,
                |ntt| {
                    mat_vec_mul_ntt_i8(
                        ntt,
                        n_rows,
                        row_width,
                        &blocks,
                        num_digits_inner,
                        log_basis_inner,
                    )
                },
            )
        }
    }
}

impl<F> DigitRowsComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        validate_digit_row_request(
            row_len,
            digits.len(),
            prepared.expanded.shared_matrix.num_field_elements() / D,
        )?;
        prepared.with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(
                D,
                row_len,
                digits.len(),
                NttTransformDomain::Negacyclic,
            )?,
            |ntt| mat_vec_mul_ntt_single_i8(ntt, row_len, digits.len(), digits, log_basis),
        )
    }
}

impl<F> CompressionComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn compression_cache_bytes(&self, prepared: &Self::PreparedSetup) -> Option<usize> {
        Some(prepared.compression_ntt_cache_bytes())
    }

    fn compression_rows_products<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        digit_vectors: &[&[[i8; D]]],
    ) -> Result<Vec<CompressionRowsProducts<F, D>>, AkitaError> {
        let input_width = validate_compression_batch_shape(digit_vectors)?;
        let total_ring_elements = prepared.expanded.shared_matrix.num_field_elements() / D;
        validate_digit_row_request(1, input_width, total_ring_elements)?;
        prepared.with_compression_ntt::<D, _>(input_width, |ntt| {
            let negacyclic = mat_vec_mul_ntt_digits_i8(ntt, 1, input_width, digit_vectors, 1)?;
            let cyclic = digit_vectors
                .iter()
                .map(|digits| mat_vec_mul_ntt_single_i8_cyclic(ntt, 1, input_width, digits, 1))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(negacyclic
                .into_iter()
                .zip(cyclic)
                .map(|(negacyclic, cyclic)| CompressionRowsProducts { negacyclic, cyclic })
                .collect())
        })
    }
}

impl<F> CyclicRowsComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        validate_digit_row_request(
            row_len,
            digits.len(),
            prepared.expanded.shared_matrix.num_field_elements() / D,
        )?;
        prepared.with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(D, row_len, digits.len(), NttTransformDomain::Cyclic)?,
            |ntt| mat_vec_mul_ntt_single_i8_cyclic(ntt, row_len, digits.len(), digits, log_basis),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::RingSwitchRelationView;
    use crate::compute::backend::{
        ComputeBackendSetup, CyclicRowsComputeBackend, DigitRowsComputeBackend,
    };
    use crate::compute::{RingSwitchRelationKernel, RingSwitchRelationPlan};
    use crate::kernels::linear::{mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic};
    use crate::validation::MAX_I8_LOG_BASIS;
    use crate::AkitaProverSetup;
    use akita_field::Prime64Offset59;
    use akita_types::SetupMatrixCapacity;
    use std::sync::Arc;

    type F = Prime64Offset59;
    const D: usize = 64;

    fn setup_capacity(num_ring_elements: usize) -> SetupMatrixCapacity {
        SetupMatrixCapacity {
            num_field_elements: num_ring_elements * D,
        }
    }

    fn prepared() -> CpuPreparedSetup<F> {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
        CpuBackend::DEFAULT.prepare_setup(&setup).unwrap()
    }

    #[test]
    fn cpu_prepared_setup_identity_rejects_mismatched_setup() {
        let setup_a =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
        let setup_b =
            AkitaProverSetup::<F>::generate_with_capacity(9, 1, setup_capacity(D)).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup_a).unwrap();

        CpuBackend::DEFAULT
            .validate_prepared_setup(&prepared, setup_a.expanded.as_ref())
            .expect("matching setup");
        assert!(
            CpuBackend::DEFAULT
                .validate_prepared_setup(&prepared, setup_b.expanded.as_ref())
                .is_err(),
            "prepared context must stay bound to the setup used to create it"
        );
    }

    #[test]
    fn cpu_prepared_setup_identity_accepts_equivalent_setup() {
        let setup_a =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
        let setup_b =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
        assert!(!Arc::ptr_eq(&setup_a.expanded, &setup_b.expanded));

        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup_a).unwrap();

        CpuBackend::DEFAULT
            .validate_prepared_setup(&prepared, setup_b.expanded.as_ref())
            .expect("equivalent deterministic setup should validate");
    }

    #[test]
    fn cpu_prepared_setup_reports_checked_crt_capacity_profile() {
        let prepared = prepared();
        CpuBackend::DEFAULT
            .digit_rows::<D>(&prepared, 1, &[[1i8; D]], 2)
            .expect("build exact NTT prefix");
        let profile = prepared.shared_ntt_profile(D).expect("profile");

        assert_eq!(profile.profile_id, "Q64/3xi32");
        assert_eq!(profile.num_primes, 3);
        assert_eq!(profile.limb_bits, 32);
        assert_eq!(profile.max_i8_log_basis, MAX_I8_LOG_BASIS);
        assert!(profile.balanced_digit_safe_width > 0);
        assert!(profile.raw_i8_safe_width > 0);
    }

    #[test]
    fn prepare_setup_starts_with_empty_ntt_cache() {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
        assert!(prepared.shared_ntt.lock().unwrap().is_empty());
    }

    #[test]
    fn cpu_prepared_setup_builds_only_requested_ntt_slots() {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let partial_key = NttCacheKey {
            ring_d: D,
            num_ring_elements: 1,
            domain: NttTransformDomain::Negacyclic,
        };
        CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, partial_key)
            .expect("warm partial slot");
        assert!(prepared.shared_ntt_cache_bytes() > 0);
        let cache = prepared.shared_ntt.lock().unwrap();
        assert!(cache.contains_key(&partial_key));
        assert_eq!(cache.len(), 1);
        drop(cache);
        let miss = NttCacheKey {
            ring_d: D,
            num_ring_elements: 99_999,
            domain: NttTransformDomain::Negacyclic,
        };
        assert!(!prepared.shared_ntt.lock().unwrap().contains_key(&miss));
    }

    #[test]
    fn concurrent_same_key_ntt_warm_builds_once() {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
        let prepared = CpuBackend::DEFAULT
            .prepare_expanded(setup.expanded.clone())
            .expect("empty prepared setup");
        let key = NttCacheKey {
            ring_d: D,
            num_ring_elements: 2,
            domain: NttTransformDomain::Negacyclic,
        };

        std::thread::scope(|scope| {
            for _ in 0..8 {
                let prepared = &prepared;
                scope.spawn(move || {
                    CpuBackend::DEFAULT
                        .ensure_ntt_slot(prepared, key)
                        .expect("warm shared NTT slot");
                });
            }
        });
        CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, key)
            .expect("repeated warm is a no-op");

        assert_eq!(prepared.ntt_slot_build_count.load(Ordering::Relaxed), 1);
        assert!(prepared.shared_ntt_cache_bytes() > 0);
    }

    #[test]
    fn larger_initialized_prefix_covers_smaller_request() {
        let prepared = prepared();
        let covering_key = NttCacheKey {
            ring_d: D,
            num_ring_elements: 8,
            domain: NttTransformDomain::Negacyclic,
        };
        CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, covering_key)
            .expect("warm covering prefix");

        prepared
            .with_shared_ntt::<D, _>(
                NttCacheKey::from_matrix_shape(D, 1, 3, NttTransformDomain::Negacyclic).unwrap(),
                |_ntt| Ok(()),
            )
            .expect("reuse covering prefix");

        assert_eq!(prepared.ntt_slot_build_count.load(Ordering::Relaxed), 1);
        let cache = prepared.shared_ntt.lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&covering_key));
    }

    #[test]
    fn larger_request_replaces_smaller_cached_prefix() {
        let prepared = prepared();
        let small = NttCacheKey {
            ring_d: D,
            num_ring_elements: 3,
            domain: NttTransformDomain::Negacyclic,
        };
        let large = NttCacheKey {
            ring_d: D,
            num_ring_elements: 8,
            domain: NttTransformDomain::Negacyclic,
        };
        CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, small)
            .expect("warm small prefix");
        CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, large)
            .expect("grow to larger prefix");

        let cache = prepared.shared_ntt.lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&large));
    }

    #[test]
    fn failed_growth_retains_smaller_cached_prefix() {
        let prepared = prepared();
        let small = NttCacheKey {
            ring_d: D,
            num_ring_elements: 3,
            domain: NttTransformDomain::Negacyclic,
        };
        let oversized = NttCacheKey {
            ring_d: D,
            num_ring_elements: D + 1,
            domain: NttTransformDomain::Negacyclic,
        };

        CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, small)
            .expect("warm small prefix");
        assert!(CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, oversized)
            .is_err());
        CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, small)
            .expect("failed growth must leave the smaller prefix usable");

        let cache = prepared.shared_ntt.lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&small));
    }

    #[test]
    fn planned_cache_bytes_match_max_joined_resident_state() {
        let prepared = prepared();
        let keys = [
            NttCacheKey {
                ring_d: D,
                num_ring_elements: 3,
                domain: NttTransformDomain::Negacyclic,
            },
            NttCacheKey {
                ring_d: D,
                num_ring_elements: 8,
                domain: NttTransformDomain::Negacyclic,
            },
            NttCacheKey {
                ring_d: D,
                num_ring_elements: 2,
                domain: NttTransformDomain::Cyclic,
            },
        ];
        let planned = prepared
            .planned_shared_ntt_cache_bytes(keys)
            .expect("planned bytes");
        for key in keys {
            CpuBackend::DEFAULT
                .ensure_ntt_slot(&prepared, key)
                .expect("prewarm exact requirement");
        }

        assert_eq!(prepared.shared_ntt_cache_bytes(), planned);
        assert_eq!(prepared.shared_ntt_cache_metrics().unwrap().len(), 2);
    }

    #[test]
    fn concurrent_prefix_growth_retains_only_the_maximum() {
        let prepared = prepared();
        std::thread::scope(|scope| {
            for num_ring_elements in [2, 5, 3, 8, 4, 7] {
                let prepared = &prepared;
                scope.spawn(move || {
                    CpuBackend::DEFAULT
                        .ensure_ntt_slot(
                            prepared,
                            NttCacheKey {
                                ring_d: D,
                                num_ring_elements,
                                domain: NttTransformDomain::Cyclic,
                            },
                        )
                        .expect("grow shared NTT prefix");
                });
            }
        });

        let cache = prepared.shared_ntt.lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&NttCacheKey {
            ring_d: D,
            num_ring_elements: 8,
            domain: NttTransformDomain::Cyclic,
        }));
    }

    #[test]
    fn failed_oversized_warm_does_not_cover_valid_request() {
        let prepared = prepared();
        let oversized = NttCacheKey {
            ring_d: D,
            num_ring_elements: D + 1,
            domain: NttTransformDomain::Negacyclic,
        };
        let valid = NttCacheKey {
            ring_d: D,
            num_ring_elements: 3,
            domain: NttTransformDomain::Negacyclic,
        };

        assert!(CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, oversized)
            .is_err());
        assert!(prepared.shared_ntt.lock().unwrap().is_empty());
        CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, valid)
            .expect("failed oversized warm must not poison a valid prefix");

        let cache = prepared.shared_ntt.lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&valid));
        assert_eq!(prepared.ntt_slot_build_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn concurrent_failed_growth_leaves_valid_prefix_recoverable() {
        let prepared = prepared();
        let oversized = NttCacheKey {
            ring_d: D,
            num_ring_elements: D + 1,
            domain: NttTransformDomain::Cyclic,
        };
        let valid = NttCacheKey {
            ring_d: D,
            num_ring_elements: 8,
            domain: NttTransformDomain::Cyclic,
        };

        std::thread::scope(|scope| {
            let failed = scope.spawn(|| CpuBackend::DEFAULT.ensure_ntt_slot(&prepared, oversized));
            let warmed = scope.spawn(|| CpuBackend::DEFAULT.ensure_ntt_slot(&prepared, valid));
            assert!(failed.join().expect("oversized warm thread").is_err());
            warmed
                .join()
                .expect("valid warm thread")
                .expect("valid warm must retry a failed covering entry");
        });
        CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, valid)
            .expect("valid prefix remains available after failed growth");

        let cache = prepared.shared_ntt.lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&valid));
    }

    #[test]
    fn ring_switch_domains_keep_independent_exact_prefix_lengths() {
        let prepared = prepared();
        let e_hat = vec![[1i8; D]; 5];
        let t_hat = vec![[1i8; D]; 3];
        let z_segment = vec![[1i32; D]; 2];

        CpuBackend::DEFAULT
            .relation_rows(
                &prepared,
                RingSwitchRelationView {
                    e_hat: &e_hat,
                    t_hat: &t_hat,
                    z_segment: &z_segment,
                    z_folded_centered_inf_norm: 1,
                },
                RingSwitchRelationPlan {
                    n_d: 2,
                    n_b: 1,
                    n_a: 1,
                    log_basis_open: 2,
                    log_basis_outer: 2,
                },
            )
            .expect("ring-switch rows");

        let cache = prepared.shared_ntt.lock().unwrap();
        assert!(cache.contains_key(&NttCacheKey {
            ring_d: D,
            num_ring_elements: 10,
            domain: NttTransformDomain::Cyclic,
        }));
        assert!(cache.contains_key(&NttCacheKey {
            ring_d: D,
            num_ring_elements: 2,
            domain: NttTransformDomain::Negacyclic,
        }));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cyclic_only_ring_switch_rows_do_not_prepare_negacyclic_state() {
        let prepared = prepared();
        let t_hat = vec![[1i8; D]; 3];

        let rows = CpuBackend::DEFAULT
            .relation_rows(
                &prepared,
                RingSwitchRelationView {
                    e_hat: &[],
                    t_hat: &t_hat,
                    z_segment: &[],
                    z_folded_centered_inf_norm: 0,
                },
                RingSwitchRelationPlan {
                    n_d: 0,
                    n_b: 2,
                    n_a: 0,
                    log_basis_open: 2,
                    log_basis_outer: 2,
                },
            )
            .expect("B-only ring-switch rows");

        assert_eq!(rows.b_cyclic.len(), 2);
        assert!(rows.d_cyclic.is_empty());
        assert!(rows.a_quotients.is_empty());
        let cache = prepared.shared_ntt.lock().unwrap();
        assert!(cache.contains_key(&NttCacheKey {
            ring_d: D,
            num_ring_elements: 6,
            domain: NttTransformDomain::Cyclic,
        }));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cpu_digit_rows_match_direct_kernel() {
        let prepared = prepared();
        let digits = vec![[1i8; D], [-1i8; D], [2i8; D]];
        let log_basis = 3;
        let via_backend = CpuBackend::DEFAULT
            .digit_rows::<D>(&prepared, 2, &digits, log_basis)
            .expect("backend digit rows");
        let direct = prepared
            .with_shared_ntt::<D, _>(
                NttCacheKey::from_matrix_shape(D, 2, digits.len(), NttTransformDomain::Negacyclic)
                    .unwrap(),
                |ntt| mat_vec_mul_ntt_single_i8(ntt, 2, digits.len(), &digits, log_basis),
            )
            .expect("direct digit rows");
        assert_eq!(via_backend, direct);
    }

    #[test]
    fn cpu_digit_rows_accept_logical_input_longer_than_stride() {
        let prepared = prepared();
        let digits = vec![[1i8; D]; 12];
        let log_basis = 3;
        let via_backend = CpuBackend::DEFAULT
            .digit_rows::<D>(&prepared, 2, &digits, log_basis)
            .expect("backend digit rows");
        let direct = prepared
            .with_shared_ntt::<D, _>(
                NttCacheKey::from_matrix_shape(D, 2, digits.len(), NttTransformDomain::Negacyclic)
                    .unwrap(),
                |ntt| mat_vec_mul_ntt_single_i8(ntt, 2, digits.len(), &digits, log_basis),
            )
            .expect("direct digit rows");
        assert_eq!(via_backend, direct);
    }

    #[test]
    fn recursive_commit_ignores_commitment_padding_blocks() {
        let prepared = prepared();
        let coeffs = vec![[1i8; D]; 6];
        let rows = CpuBackend::DEFAULT
            .recursive_witness_commit_rows(&prepared, &coeffs, 1, 2, 2, 1, 3, Some(3))
            .expect("recursive commit rows");

        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn cpu_cyclic_digit_rows_match_direct_kernel() {
        let prepared = prepared();
        let digits = vec![[1i8; D], [0i8; D], [-2i8; D], [3i8; D]];
        let log_basis = 3;
        let via_backend = CpuBackend::DEFAULT
            .cyclic_digit_rows::<D>(&prepared, 2, &digits, log_basis)
            .expect("backend cyclic digit rows");
        let direct = prepared
            .with_shared_ntt::<D, _>(
                NttCacheKey::from_matrix_shape(D, 2, digits.len(), NttTransformDomain::Cyclic)
                    .unwrap(),
                |ntt| mat_vec_mul_ntt_single_i8_cyclic(ntt, 2, digits.len(), &digits, log_basis),
            )
            .expect("direct cyclic digit rows");
        assert_eq!(via_backend, direct);
    }

    #[test]
    fn cpu_ring_switch_relation_rows_use_distinct_open_and_outer_bases() {
        let prepared = prepared();
        let e_hat = vec![[1i8; D], [-1i8; D]];
        let t_hat = vec![[-1i8; D], [3i8; D]];
        let z_segment = vec![[1i32; D], [-2i32; D], [3i32; D]];
        let via_backend = CpuBackend::DEFAULT
            .relation_rows(
                &prepared,
                RingSwitchRelationView {
                    e_hat: &e_hat,
                    t_hat: &t_hat,
                    z_segment: &z_segment,
                    z_folded_centered_inf_norm: 3,
                },
                RingSwitchRelationPlan {
                    n_d: 1,
                    n_b: 1,
                    n_a: 1,
                    log_basis_open: 2,
                    log_basis_outer: 3,
                },
            )
            .expect("backend ring-switch relation rows");
        let direct = prepared
            .with_shared_ntt::<D, _>(
                NttCacheKey::from_matrix_shape(D, 1, z_segment.len(), NttTransformDomain::Cyclic)
                    .unwrap(),
                |cyclic_ntt| {
                    prepared.with_shared_ntt::<D, _>(
                        NttCacheKey::from_matrix_shape(
                            D,
                            1,
                            z_segment.len(),
                            NttTransformDomain::Negacyclic,
                        )
                        .unwrap(),
                        |negacyclic_ntt| {
                            fused_split_eq_quotients_prover_bounds(
                                negacyclic_ntt,
                                cyclic_ntt,
                                1,
                                1,
                                1,
                                &e_hat,
                                &t_hat,
                                &z_segment,
                                3,
                                2,
                                3,
                            )
                        },
                    )
                },
            )
            .expect("direct fused split-eq rows");
        assert_eq!(
            (
                via_backend.d_cyclic,
                via_backend.b_cyclic,
                via_backend.a_quotients
            ),
            direct
        );
    }
}
