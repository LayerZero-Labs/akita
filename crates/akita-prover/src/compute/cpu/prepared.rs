use super::compression_cache::CompressionNttCache;
use super::CpuBackend;
use crate::compute::backend::ComputeBackendSetup;
use crate::compute::requirements::RoutedNttRequirement;
use crate::kernels::linear::{selected_crt_i8_capacity_profile, CrtI8CapacityProfile};
use akita_error::AkitaError;
use akita_types::{
    dispatch_for_field, planned_exact_ntt_cache_bytes, prepare_ntt_cache, AkitaExpandedSetup,
    NttCacheKey, NttCacheMode, NttTransformDomain, PreparedNttCache,
};
use jolt_field::{CanonicalEncoding, Field};
use std::any::Any;
use std::collections::HashMap;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

type NttSlotCell = OnceLock<Result<Arc<ErasedCpuNttCache>, AkitaError>>;

/// CPU-prepared setup keyed by runtime ring dimension.
///
/// NTT caches are keyed by [`NttCacheKey`] and built lazily. Each ring
/// dimension/domain pair retains only its largest requested prefix; a covering
/// cell also serves smaller requests. Each cell makes concurrent construction
/// of that prefix single-flight. Diagnostic compression caches remain in a
/// separate namespace.
#[derive(Debug)]
pub struct CpuPreparedSetup<F: Field> {
    pub(super) expanded: Arc<AkitaExpandedSetup<F>>,
    pub(super) shared_ntt: Mutex<HashMap<NttCacheKey, Arc<NttSlotCell>>>,
    pub(super) compression_ntt: CompressionNttCache,
    ntt_i8_capacity_by_ring_d: Mutex<HashMap<usize, CrtI8CapacityProfile>>,
    #[cfg(test)]
    pub(super) ntt_slot_build_count: AtomicUsize,
}

pub(super) struct ErasedCpuNttCache {
    pub(super) ring_d: usize,
    pub(super) cache_bytes: usize,
    pub(super) cache: Arc<dyn Any + Send + Sync>,
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

impl<F: Field + CanonicalEncoding> CpuPreparedSetup<F> {
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

    pub(super) fn with_compression_ntt<const D: usize, R>(
        &self,
        input_width: usize,
        domains: super::compression_cache::CompressionNttDomains,
        f: impl FnOnce(&PreparedNttCache<D>) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        self.compression_ntt
            .with_ntt(self.expanded.as_ref(), input_width, domains, f)
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

    /// Drop built shared-matrix NTT slots and return their byte footprint.
    ///
    /// Compression transforms remain resident. Active readers keep released
    /// slots alive through their `Arc`; callers that need an empty cache must
    /// invoke this at a quiescent lifecycle boundary.
    pub fn drop_built_ntt_slots(&self) -> Result<usize, AkitaError> {
        let mut freed = 0usize;
        let mut cache = self
            .shared_ntt
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?;
        let mut released_keys = Vec::new();
        for (key, cell) in cache.iter() {
            if let Some(bytes) = cell
                .get()
                .and_then(|result| result.as_ref().ok())
                .map(|slot| slot.cache_bytes)
            {
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
                NttTransformDomain::I16TailBothTransforms => 2,
                NttTransformDomain::ExactNegacyclicI16 { .. } => 3,
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
            .try_fold(0usize, |total, ((ring_d, domain), count)| {
                let entry_bytes = match domain {
                    NttTransformDomain::I16TailBothTransforms => count
                        .checked_mul(ring_d)
                        .and_then(|bytes| bytes.checked_mul(2 * core::mem::size_of::<i16>()))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("planned i16-tail bytes overflow".into())
                        })?,
                    NttTransformDomain::ExactNegacyclicI16 {
                        width,
                        rhs_abs_bound,
                    } => dispatch_for_field!(ProtocolDispatchSlot::Ntt, F, ring_d, |RING_D| {
                        planned_exact_ntt_cache_bytes::<F, RING_D>(count, width, rhs_abs_bound)
                    })?,
                    NttTransformDomain::Negacyclic | NttTransformDomain::Cyclic => {
                        let profile =
                            dispatch_for_field!(ProtocolDispatchSlot::Ntt, F, ring_d, |RING_D| {
                                selected_crt_i8_capacity_profile::<F, RING_D>()
                            })?;
                        count
                            .checked_mul(ring_d)
                            .and_then(|bytes| bytes.checked_mul(profile.num_primes))
                            .and_then(|bytes| bytes.checked_mul(core::mem::size_of::<i32>()))
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("planned NTT bytes overflow".into())
                            })?
                    }
                };
                total
                    .checked_add(entry_bytes)
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

    /// CRT/NTT profile and universal i8 capacity metadata for ring degree `D`.
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

fn build_ntt_slot_for_key<F: Field + CanonicalEncoding>(
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
            NttTransformDomain::I16TailBothTransforms => NttCacheMode::I16TailBothTransforms,
            NttTransformDomain::ExactNegacyclicI16 {
                width,
                rhs_abs_bound,
            } => NttCacheMode::ExactNegacyclic {
                width,
                rhs_abs_bound,
            },
        };
        let cache = Arc::new(prepare_ntt_cache(view, mode)?);
        Ok(ErasedCpuNttCache {
            ring_d: RING_D,
            cache_bytes: cache.cache_bytes(),
            cache,
        })
    })
}

fn record_ntt_profile_on_prepared<F: Field>(
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

fn prepare_ntt_slot_on_prepared<F: Field + CanonicalEncoding>(
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

fn ensure_ntt_slot_on_prepared<F: Field + CanonicalEncoding>(
    prepared: &CpuPreparedSetup<F>,
    key: NttCacheKey,
) -> Result<(), AkitaError> {
    prepare_ntt_slot_on_prepared(prepared, key).map(|_| ())
}

pub(super) fn validate_digit_row_request(
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
    F: Field + CanonicalEncoding,
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
