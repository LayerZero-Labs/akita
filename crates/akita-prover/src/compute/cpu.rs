use crate::backend::onehot::{
    column_sweep_ajtai_onehot, column_sweep_ajtai_onehot_multi,
    column_sweep_ajtai_onehot_multi_lazy, LazyOneHotBlocks, MultiChunkEntry, SingleChunkEntry,
};
use crate::backend::sparse_ring::column_sweep_sparse;
#[cfg(feature = "compression-diagnostics")]
use crate::compute::backend::CompressionDiagnosticBackend;
use crate::compute::backend::{
    CommitmentComputeBackend, ComputeBackendSetup, CyclicRowsComputeBackend,
    DigitRowsComputeBackend, RingSwitchComputeBackend,
};
use crate::compute::plans::{
    DenseCommitInput, DenseCommitRowsPlan, OneHotCommitBlocks, OneHotCommitRowsPlan,
    RecursiveWitnessCommitRowsPlan, RingSwitchQuotientRowsPlan, RingSwitchRelationRows,
    RingSwitchRelationRowsPlan, SparseRingCommitRowsPlan,
};
#[cfg(feature = "compression-diagnostics")]
use crate::kernels::linear::validate_compression_batch_shape;
use crate::kernels::linear::{
    digit_blocks_are_balanced, fused_split_eq_quotients_prover_bounds,
    fused_split_eq_quotients_streamed_prover_bounds, mat_vec_mul_ntt_dense_digits_i8,
    mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_i8, mat_vec_mul_ntt_i8_dense,
    mat_vec_mul_ntt_i8_dense_single_row, mat_vec_mul_ntt_raw_digits_i8, mat_vec_mul_ntt_single_i8,
    mat_vec_mul_ntt_single_i8_cyclic, selected_crt_i8_capacity_profile, CrtI8CapacityProfile,
    StreamedASource,
};
use crate::validation::MAX_I8_LOG_BASIS;
use akita_algebra::CyclotomicRing;
use akita_field::unreduced::{HasCommitAccum, HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_types::{
    dispatch_for_field, prepare_ntt_cache, AkitaExpandedSetup, NttCacheKey, NttCacheMode,
    PreparedNttCache,
};
use std::any::Any;
use std::array::from_fn;
use std::collections::{HashMap, HashSet};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "compression-diagnostics")]
mod compression_cache;

#[cfg(feature = "compression-diagnostics")]
use compression_cache::CompressionNttCache;

/// CPU backend using the existing Rust/Rayon kernels.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

type NttSlotCell = OnceLock<Result<Arc<ErasedCpuNttCache>, AkitaError>>;

/// A-extent above which one-shot consumers stream per-element transforms from
/// the field form instead of building (and keeping) a cached NTT slot. At the
/// jolt 2^26 shape the cached slot costs ~2.5 KiB per ring element, so this
/// bounds any lazily built slot to ~5 GiB.
///
/// Public because callers releasing the setup matrix must retain at least
/// this prefix: cached-path consumers below the threshold read the retained
/// store, and a smaller prefix silently degrades their slot rebuilds to
/// per-call re-derivations (see
/// [`CpuPreparedSetup::release_setup_matrix_to_streaming_prefix`]).
pub const NTT_STREAM_THRESHOLD_RING_ELEMENTS: usize = 1 << 21;

/// CPU-prepared setup keyed by runtime ring dimension.
///
/// NTT caches are keyed by [`NttCacheKey`]. [`ComputeBackendSetup::prepare_setup`]
/// reserves the envelope slot on the setup contract without building it; slots
/// build lazily at the extent consumers actually request (see
/// `CpuPreparedSetup::with_shared_ntt`), or eagerly via
/// [`ComputeBackendSetup::ensure_ntt_slot`]. A built slot serves all requests
/// within its extent at its ring dimension, and its cell makes concurrent
/// first use single-flight. Diagnostic compression caches use a separate
/// exact-shape namespace because they contain only the negacyclic transform.
#[derive(Debug)]
pub struct CpuPreparedSetup<F: FieldCore> {
    expanded: Arc<AkitaExpandedSetup<F>>,
    shared_ntt: Mutex<HashMap<NttCacheKey, Arc<NttSlotCell>>>,
    shared_neg_ntt: Mutex<HashMap<NttCacheKey, Arc<NttSlotCell>>>,
    #[cfg(feature = "compression-diagnostics")]
    compression_ntt: CompressionNttCache,
    ntt_i8_capacity_by_ring_d: Mutex<HashMap<usize, CrtI8CapacityProfile>>,
    /// Keys promised at [`ComputeBackendSetup::prepare_setup`]; lazy builds outside
    /// this set emit a diagnostic warning.
    setup_contract_ntt_keys: Mutex<HashSet<NttCacheKey>>,
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
    fn envelope_ntt_key<const D: usize>(&self) -> Result<NttCacheKey, AkitaError> {
        NttCacheKey::from_envelope(&self.expanded, D)
    }

    /// Run `f` against a transformed-matrix slot covering the first
    /// `extent_ring_elements` flat ring elements of A.
    ///
    /// Slots build lazily on first use: the transformed matrix is a second
    /// full-matrix residency (5 CRT primes x 2 transforms, ~2.5x the field
    /// form), and at the jolt 2^26 shape its consumers touch under 1/6 of
    /// the setup envelope — so the build is sized to the caller-declared
    /// extent (rounded up to a power of two, capped at the envelope) rather
    /// than the envelope itself. The smallest already-built covering slot is
    /// reused, so an explicitly warmed envelope slot (`ensure_ntt_slot`)
    /// serves every request exactly as before. `get_or_init` keeps each
    /// build single-flight; concurrent first users join it instead of
    /// reporting a false cache miss.
    pub(crate) fn with_shared_ntt<const D: usize, R>(
        &self,
        extent_ring_elements: usize,
        f: impl FnOnce(&PreparedNttCache<D>) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        let envelope = self.envelope_ntt_key::<D>()?;
        let capped = extent_ring_elements.max(1).min(envelope.num_ring_elements);
        let rounded = capped
            .checked_next_power_of_two()
            .map_or(envelope.num_ring_elements, |p| {
                p.min(envelope.num_ring_elements)
            });
        let (key, entry) = {
            let mut cache = self
                .shared_ntt
                .lock()
                .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?;
            let covering = cache
                .iter()
                .filter(|(k, cell)| {
                    k.ring_d == envelope.ring_d
                        && k.num_ring_elements >= capped
                        && cell.get().is_some_and(|result| result.is_ok())
                })
                .min_by_key(|(k, _)| k.num_ring_elements)
                .map(|(k, cell)| (*k, Arc::clone(cell)));
            match covering {
                Some(found) => found,
                None => {
                    let key = NttCacheKey {
                        ring_d: envelope.ring_d,
                        num_ring_elements: rounded,
                    };
                    let cell = Arc::clone(
                        cache
                            .entry(key)
                            .or_insert_with(|| Arc::new(OnceLock::new())),
                    );
                    (key, cell)
                }
            }
        };
        let slot = entry
            .get_or_init(|| {
                #[cfg(test)]
                self.ntt_slot_build_count.fetch_add(1, Ordering::Relaxed);
                build_ntt_slot_for_key(self.expanded.as_ref(), key).map(Arc::new)
            })
            .as_ref()
            .map_err(Clone::clone)?
            .clone();
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

    #[cfg(feature = "compression-diagnostics")]
    fn with_compression_ntt<const D: usize, R>(
        &self,
        input_width: usize,
        f: impl FnOnce(&PreparedNttCache<D>) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        self.compression_ntt
            .with_ntt(self.expanded.as_ref(), input_width, f)
    }

    /// Run `f` against a negacyclic-only transformed-matrix slot.
    ///
    /// A built both-transform slot can satisfy this request. Otherwise this
    /// uses a separate cache so cyclic consumers cannot mistake a
    /// negacyclic-only slot for their stronger contract.
    pub(crate) fn with_shared_negacyclic_ntt<const D: usize, R>(
        &self,
        extent_ring_elements: usize,
        f: impl FnOnce(&PreparedNttCache<D>) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        let envelope = self.envelope_ntt_key::<D>()?;
        let capped = extent_ring_elements.max(1).min(envelope.num_ring_elements);
        let rounded = capped
            .checked_next_power_of_two()
            .map_or(envelope.num_ring_elements, |p| {
                p.min(envelope.num_ring_elements)
            });

        let both_slot = {
            let cache = self
                .shared_ntt
                .lock()
                .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?;
            cache
                .iter()
                .filter(|(key, cell)| {
                    key.ring_d == envelope.ring_d
                        && key.num_ring_elements >= capped
                        && cell.get().is_some_and(|result| result.is_ok())
                })
                .min_by_key(|(key, _)| key.num_ring_elements)
                .and_then(|(_, cell)| cell.get())
                .and_then(|result| result.as_ref().ok())
                .cloned()
        };
        if let Some(slot) = both_slot {
            return with_typed_ntt_slot::<D, _>(&slot, f);
        }

        let key = NttCacheKey {
            ring_d: envelope.ring_d,
            num_ring_elements: rounded,
        };
        let entry = {
            let mut cache = self
                .shared_neg_ntt
                .lock()
                .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?;
            Arc::clone(
                cache
                    .entry(key)
                    .or_insert_with(|| Arc::new(OnceLock::new())),
            )
        };
        let slot = entry
            .get_or_init(|| {
                #[cfg(test)]
                self.ntt_slot_build_count.fetch_add(1, Ordering::Relaxed);
                build_negacyclic_ntt_slot_for_key(self.expanded.as_ref(), key).map(Arc::new)
            })
            .as_ref()
            .map_err(Clone::clone)?
            .clone();
        with_typed_ntt_slot::<D, _>(&slot, f)
    }

    /// Release the setup matrix's backing store down to its first
    /// `keep_ring_elements` (generation dimension), returning bytes freed.
    /// The retained prefix serves slot rebuilds and small setup reads; wider
    /// consumers stream per-element from the seed or re-derive per call.
    pub fn release_setup_matrix_to_prefix(&self, keep_ring_elements: usize) -> usize {
        self.expanded
            .shared_matrix()
            .release_to_prefix(keep_ring_elements)
    }

    /// [`Self::release_setup_matrix_to_prefix`] with the retained prefix
    /// pinned to the streaming threshold, so cached-path consumers keep
    /// hitting the store instead of re-deriving.
    pub fn release_setup_matrix_to_streaming_prefix(&self) -> usize {
        self.release_setup_matrix_to_prefix(NTT_STREAM_THRESHOLD_RING_ELEMENTS)
    }

    /// Drop every built NTT slot back to its reserved (empty) state and
    /// return the bytes freed. Keys and the setup contract are kept, so the
    /// next `with_shared_ntt` use rebuilds single-flight — callers
    /// may drop between pipeline windows that don't touch A's transformed
    /// form (e.g. after the commit's terminal product, before the fold) to
    /// keep the transform out of the intervening standing footprint. A user
    /// that raced ahead with the old cell finishes against it unaffected;
    /// the swap only redirects future lookups.
    pub fn drop_built_ntt_slots(&self) -> usize {
        let freed = drop_built_slots(&self.shared_ntt) + drop_built_slots(&self.shared_neg_ntt);
        if freed > 0 {
            tracing::info!(freed_bytes = freed, "dropped built NTT slots");
        }
        freed
    }

    /// In-memory byte footprint of all shared setup NTT caches.
    pub fn shared_ntt_cache_bytes(&self) -> usize {
        built_slot_bytes(&self.shared_ntt) + built_slot_bytes(&self.shared_neg_ntt)
    }

    /// In-memory byte footprint of exact-prefix compression NTT caches.
    #[cfg(feature = "compression-diagnostics")]
    pub fn compression_ntt_cache_bytes(&self) -> usize {
        self.compression_ntt.cache_bytes()
    }

    /// CRT/NTT profile and universal i8 capacity metadata for ring degree `D`.
    pub fn shared_ntt_profile<const D: usize>(&self) -> Result<PreparedCrtNttProfile, AkitaError> {
        self.ntt_i8_capacity_by_ring_d
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("NTT profile lock poisoned".into()))?
            .get(&D)
            .copied()
            .map(Into::into)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "prepared setup has no CRT/i8 capacity profile for ring_d={D}"
                ))
            })
    }
}

fn with_typed_ntt_slot<const D: usize, R>(
    slot: &ErasedCpuNttCache,
    f: impl FnOnce(&PreparedNttCache<D>) -> Result<R, AkitaError>,
) -> Result<R, AkitaError> {
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

fn drop_built_slots(cache: &Mutex<HashMap<NttCacheKey, Arc<NttSlotCell>>>) -> usize {
    let mut freed = 0;
    if let Ok(mut cache) = cache.lock() {
        for cell in cache.values_mut() {
            let built = cell
                .get()
                .and_then(|result| result.as_ref().ok())
                .map(|slot| slot.cache_bytes);
            if let Some(bytes) = built {
                freed += bytes;
                *cell = Arc::new(OnceLock::new());
            }
        }
    }
    freed
}

fn built_slot_bytes(cache: &Mutex<HashMap<NttCacheKey, Arc<NttSlotCell>>>) -> usize {
    cache
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .values()
        .filter_map(|entry| entry.get())
        .filter_map(|result| result.as_ref().ok())
        .map(|slot| slot.cache_bytes)
        .sum()
}

fn build_ntt_slot_for_key<F: FieldCore + CanonicalField>(
    expanded: &AkitaExpandedSetup<F>,
    key: NttCacheKey,
) -> Result<ErasedCpuNttCache, AkitaError> {
    dispatch_for_field!(ProtocolDispatchSlot::Ntt, F, key.ring_d, |RING_D| {
        let matrix = expanded
            .shared_matrix()
            .covering_at_dyn(key.num_ring_elements, RING_D)?;
        let view = matrix.ring_view::<RING_D>(1, key.num_ring_elements)?;
        let cache = Arc::new(prepare_ntt_cache(view, NttCacheMode::BothTransforms)?);
        Ok(ErasedCpuNttCache {
            ring_d: RING_D,
            cache_bytes: cache.cache_bytes(),
            cache,
        })
    })
}

fn build_negacyclic_ntt_slot_for_key<F: FieldCore + CanonicalField>(
    expanded: &AkitaExpandedSetup<F>,
    key: NttCacheKey,
) -> Result<ErasedCpuNttCache, AkitaError> {
    dispatch_for_field!(ProtocolDispatchSlot::Ntt, F, key.ring_d, |RING_D| {
        let matrix = expanded
            .shared_matrix()
            .covering_at_dyn(key.num_ring_elements, RING_D)?;
        let view = matrix.ring_view::<RING_D>(1, key.num_ring_elements)?;
        // `digit_rows` chunks its accumulation independently. Width one asks
        // the exact-mode builder for only the base negacyclic transforms,
        // without an unused full-slot i16 tail.
        let cache = Arc::new(prepare_ntt_cache(
            view,
            NttCacheMode::ExactNegacyclic {
                width: 1,
                log_basis: MAX_I8_LOG_BASIS,
            },
        )?);
        tracing::info!(
            ring_d = RING_D,
            num_ring_elements = key.num_ring_elements,
            cache_bytes = cache.cache_bytes(),
            "built shared-matrix negacyclic NTT slot"
        );
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

/// Reserve a slot cell and record its CRT profile without transforming the
/// matrix; the build happens on first [`CpuPreparedSetup::with_shared_ntt`]
/// use (or an explicit [`ensure_ntt_slot_on_prepared`] warm-up).
fn reserve_ntt_slot_on_prepared<F: FieldCore + CanonicalField>(
    prepared: &CpuPreparedSetup<F>,
    key: NttCacheKey,
) -> Result<(), AkitaError> {
    let profile = dispatch_for_field!(ProtocolDispatchSlot::Ntt, F, key.ring_d, |RING_D| {
        selected_crt_i8_capacity_profile::<F, RING_D>()
    })?;
    {
        let mut cache = prepared
            .shared_ntt
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?;
        let _reserved = cache
            .entry(key)
            .or_insert_with(|| Arc::new(OnceLock::new()));
    }
    record_ntt_profile_on_prepared(prepared, key, profile)
}

fn insert_ntt_slot_on_prepared<F: FieldCore + CanonicalField>(
    prepared: &CpuPreparedSetup<F>,
    key: NttCacheKey,
) -> Result<(), AkitaError> {
    let profile = dispatch_for_field!(ProtocolDispatchSlot::Ntt, F, key.ring_d, |RING_D| {
        selected_crt_i8_capacity_profile::<F, RING_D>()
    })?;
    let entry = {
        let mut cache = prepared
            .shared_ntt
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?;
        Arc::clone(
            cache
                .entry(key)
                .or_insert_with(|| Arc::new(OnceLock::new())),
        )
    };
    let build_result = entry.get_or_init(|| {
        #[cfg(test)]
        prepared
            .ntt_slot_build_count
            .fetch_add(1, Ordering::Relaxed);
        build_ntt_slot_for_key(prepared.expanded.as_ref(), key).map(Arc::new)
    });
    build_result.as_ref().map_err(Clone::clone)?;
    record_ntt_profile_on_prepared(prepared, key, profile)
}

fn register_setup_contract_ntt_slot_on_prepared<F: FieldCore + CanonicalField>(
    prepared: &CpuPreparedSetup<F>,
    key: NttCacheKey,
) -> Result<(), AkitaError> {
    reserve_ntt_slot_on_prepared(prepared, key)?;
    prepared
        .setup_contract_ntt_keys
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("NTT contract lock poisoned".into()))?
        .insert(key);
    Ok(())
}

fn ensure_ntt_slot_on_prepared<F: FieldCore + CanonicalField>(
    prepared: &CpuPreparedSetup<F>,
    key: NttCacheKey,
) -> Result<(), AkitaError> {
    let initialized = prepared
        .shared_ntt
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?
        .get(&key)
        .is_some_and(|entry| entry.get().is_some_and(Result::is_ok));
    if initialized {
        return Ok(());
    }
    if !prepared
        .setup_contract_ntt_keys
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("NTT contract lock poisoned".into()))?
        .contains(&key)
    {
        tracing::warn!(
            target: "akita_prover::ntt_cache",
            ring_d = key.ring_d,
            num_ring_elements = key.num_ring_elements,
            setup_contract_keys = prepared
                .setup_contract_ntt_keys
                .lock()
                .map_err(|_| AkitaError::InvalidSetup("NTT contract lock poisoned".into()))?
                .len(),
            "building NTT cache slot outside setup prepare contract; \
             setup envelope or prepare path is likely undersized for this commit/prove path"
        );
    }
    insert_ntt_slot_on_prepared(prepared, key)
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

    fn prepare_expanded<const D: usize>(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup, AkitaError> {
        Ok(CpuPreparedSetup {
            expanded,
            shared_ntt: Mutex::new(HashMap::new()),
            shared_neg_ntt: Mutex::new(HashMap::new()),
            #[cfg(feature = "compression-diagnostics")]
            compression_ntt: CompressionNttCache::default(),
            ntt_i8_capacity_by_ring_d: Mutex::new(HashMap::new()),
            setup_contract_ntt_keys: Mutex::new(HashSet::new()),
            #[cfg(test)]
            ntt_slot_build_count: AtomicUsize::new(0),
        })
    }

    fn register_setup_contract_ntt_slot(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<(), AkitaError> {
        register_setup_contract_ntt_slot_on_prepared(prepared, key)
    }

    fn ensure_ntt_slot(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<(), AkitaError> {
        ensure_ntt_slot_on_prepared(prepared, key)
    }

    fn release_built_ntt_slots(&self, prepared: &Self::PreparedSetup) -> usize {
        prepared.drop_built_ntt_slots()
    }

    fn prepared_expanded_setup<'a>(
        &self,
        prepared: &'a Self::PreparedSetup,
    ) -> &'a AkitaExpandedSetup<F> {
        prepared.expanded.as_ref()
    }
}

impl<F> CommitmentComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn dense_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: DenseCommitRowsPlan<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        match plan.input {
            DenseCommitInput::CachedDigits {
                digit_block_slices,
                log_basis_inner,
            } => {
                let row_width = digit_block_slices.first().map_or(0, |digits| digits.len());
                prepared.with_shared_ntt::<D, _>(plan.n_a.saturating_mul(row_width), |ntt| {
                    mat_vec_mul_ntt_dense_digits_i8(
                        ntt,
                        plan.n_a,
                        row_width,
                        &digit_block_slices,
                        log_basis_inner,
                    )
                })
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
                if plan.n_a == 1 {
                    prepared.with_shared_ntt::<D, _>(row_width, |ntt| {
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
                    })
                } else {
                    prepared.with_shared_ntt::<D, _>(plan.n_a.saturating_mul(row_width), |ntt| {
                        mat_vec_mul_ntt_i8_dense(
                            ntt,
                            plan.n_a,
                            row_width,
                            &block_slices,
                            num_digits_inner,
                            log_basis_inner,
                        )
                    })
                }
            }
        }
    }

    fn onehot_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: OneHotCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasCommitAccum,
        F::CommitAccum: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        let active_a_cols = plan
            .num_positions_per_block
            .checked_mul(plan.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
        let a_matrix = prepared.expanded.shared_matrix.covering_at_dyn(
            plan.n_a
                .checked_mul(active_a_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("active A extent overflow".to_string()))?,
            D,
        )?;
        let a_view = a_matrix.ring_view::<D>(plan.n_a, active_a_cols)?;
        Ok(match plan.blocks {
            OneHotCommitBlocks::SingleChunk(blocks) => {
                column_sweep_ajtai_onehot::<SingleChunkEntry, F, D>(
                    &a_view,
                    &blocks.block_slices()?,
                    plan.n_a,
                    active_a_cols,
                    plan.num_digits_inner,
                )
            }
            OneHotCommitBlocks::MultiChunk(blocks) => {
                column_sweep_ajtai_onehot::<MultiChunkEntry, F, D>(
                    &a_view,
                    &blocks.block_slices()?,
                    plan.n_a,
                    active_a_cols,
                    plan.num_digits_inner,
                )
            }
            // Single-plan lazy callers route through the fused lazy sweep so
            // the entry cache stays tile-sized here too.
            OneHotCommitBlocks::SingleChunkLazy(ref source) => {
                column_sweep_ajtai_onehot_multi_lazy::<SingleChunkEntry, F, D>(
                    &a_view,
                    &[source],
                    plan.n_a,
                    active_a_cols,
                    plan.num_digits_inner,
                )?
                .pop()
                .unwrap_or_default()
            }
            OneHotCommitBlocks::MultiChunkLazy(ref source) => {
                column_sweep_ajtai_onehot_multi_lazy::<MultiChunkEntry, F, D>(
                    &a_view,
                    &[source],
                    plan.n_a,
                    active_a_cols,
                    plan.num_digits_inner,
                )?
                .pop()
                .unwrap_or_default()
            }
        })
    }

    fn onehot_commit_rows_multi<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plans: Vec<OneHotCommitRowsPlan<'_>>,
    ) -> Result<Vec<Vec<Vec<CyclotomicRing<F, D>>>>, AkitaError>
    where
        F: HasCommitAccum,
        F::CommitAccum: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        let Some(first) = plans.first() else {
            return Ok(Vec::new());
        };
        let uniform_shape = plans.iter().all(|plan| {
            plan.n_a == first.n_a
                && plan.num_positions_per_block == first.num_positions_per_block
                && plan.num_digits_inner == first.num_digits_inner
        });
        let all_single = plans
            .iter()
            .all(|plan| matches!(plan.blocks, OneHotCommitBlocks::SingleChunk(_)));
        let all_multi = plans
            .iter()
            .all(|plan| matches!(plan.blocks, OneHotCommitBlocks::MultiChunk(_)));
        let all_single_lazy = plans
            .iter()
            .all(|plan| matches!(plan.blocks, OneHotCommitBlocks::SingleChunkLazy(_)));
        let all_multi_lazy = plans
            .iter()
            .all(|plan| matches!(plan.blocks, OneHotCommitBlocks::MultiChunkLazy(_)));
        if !uniform_shape || !(all_single || all_multi || all_single_lazy || all_multi_lazy) {
            return plans
                .into_iter()
                .map(|plan| self.onehot_commit_rows::<D>(prepared, plan))
                .collect();
        }

        let active_a_cols = first
            .num_positions_per_block
            .checked_mul(first.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
        let a_matrix = prepared.expanded.shared_matrix.covering_at_dyn(
            first
                .n_a
                .checked_mul(active_a_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("active A extent overflow".to_string()))?,
            D,
        )?;
        let a_view = a_matrix.ring_view::<D>(first.n_a, active_a_cols)?;
        let n_a = first.n_a;
        let num_digits_inner = first.num_digits_inner;

        if all_single_lazy || all_multi_lazy {
            return if all_single_lazy {
                let sources: Vec<&LazyOneHotBlocks<'_, SingleChunkEntry>> = plans
                    .iter()
                    .map(|plan| match &plan.blocks {
                        OneHotCommitBlocks::SingleChunkLazy(source) => source,
                        _ => unreachable!("checked all_single_lazy"),
                    })
                    .collect();
                column_sweep_ajtai_onehot_multi_lazy::<SingleChunkEntry, F, D>(
                    &a_view,
                    &sources,
                    n_a,
                    active_a_cols,
                    num_digits_inner,
                )
            } else {
                let sources: Vec<&LazyOneHotBlocks<'_, MultiChunkEntry>> = plans
                    .iter()
                    .map(|plan| match &plan.blocks {
                        OneHotCommitBlocks::MultiChunkLazy(source) => source,
                        _ => unreachable!("checked all_multi_lazy"),
                    })
                    .collect();
                column_sweep_ajtai_onehot_multi_lazy::<MultiChunkEntry, F, D>(
                    &a_view,
                    &sources,
                    n_a,
                    active_a_cols,
                    num_digits_inner,
                )
            };
        }

        if all_single {
            let polys_blocks: Vec<Vec<&[SingleChunkEntry]>> = plans
                .iter()
                .map(|plan| match &plan.blocks {
                    OneHotCommitBlocks::SingleChunk(blocks) => blocks.block_slices(),
                    _ => unreachable!("checked all_single"),
                })
                .collect::<Result<_, _>>()?;
            Ok(column_sweep_ajtai_onehot_multi::<SingleChunkEntry, F, D>(
                &a_view,
                &polys_blocks,
                n_a,
                active_a_cols,
                num_digits_inner,
            ))
        } else {
            let polys_blocks: Vec<Vec<&[MultiChunkEntry]>> = plans
                .iter()
                .map(|plan| match &plan.blocks {
                    OneHotCommitBlocks::MultiChunk(blocks) => blocks.block_slices(),
                    _ => unreachable!("checked all_multi"),
                })
                .collect::<Result<_, _>>()?;
            Ok(column_sweep_ajtai_onehot_multi::<MultiChunkEntry, F, D>(
                &a_view,
                &polys_blocks,
                n_a,
                active_a_cols,
                num_digits_inner,
            ))
        }
    }

    fn sparse_ring_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: SparseRingCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        let active_a_cols = plan
            .num_positions_per_block
            .checked_mul(plan.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
        let a_matrix = prepared.expanded.shared_matrix.covering_at_dyn(
            plan.n_a
                .checked_mul(active_a_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("active A extent overflow".to_string()))?,
            D,
        )?;
        let a_view = a_matrix.ring_view::<D>(plan.n_a, active_a_cols)?;
        Ok(column_sweep_sparse(
            &a_view,
            &plan.blocks.block_slices()?,
            plan.n_a,
            plan.num_positions_per_block,
            plan.num_digits_inner,
        ))
    }

    fn recursive_witness_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RecursiveWitnessCommitRowsPlan<'_, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        let row_width = plan
            .num_positions_per_block
            .checked_mul(plan.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("recursive A width overflow".to_string()))?;
        let minimum_ring_elems = plan
            .num_live_blocks
            .saturating_sub(1)
            .checked_mul(plan.num_positions_per_block)
            .and_then(|prefix| prefix.checked_add(1))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive witness block extent overflow".to_string())
            })?;
        if plan.num_live_blocks == 0 || plan.coeffs.len() < minimum_ring_elems {
            return Err(AkitaError::InvalidSetup(
                "recursive witness does not cover its live blocks".to_string(),
            ));
        }
        if plan.num_digits_inner == 1 {
            let blocks = plan
                .coeffs
                .chunks(plan.num_positions_per_block)
                .take(plan.num_live_blocks)
                .collect::<Vec<_>>();
            // The `num_digits_inner == 1` recursive witness is a raw signed-i8
            // coefficient stream. Degree-one fields yield balanced gadget digits
            // (fast predecomposed-digit kernel), but extension-field tensor
            // base-lift packing sums gadget digits and can push coefficients
            // past the balanced range; those must commit through the general
            // raw ring mat-vec instead of the balanced-digit LUT kernel.
            let known_balanced = plan
                .known_balanced_log_basis
                .is_some_and(|source_log_basis| plan.log_basis_inner >= source_log_basis);
            if known_balanced || digit_blocks_are_balanced(&blocks, row_width, plan.log_basis_inner)
            {
                prepared.with_shared_ntt::<D, _>(plan.n_rows.saturating_mul(row_width), |ntt| {
                    mat_vec_mul_ntt_digits_i8(
                        ntt,
                        plan.n_rows,
                        row_width,
                        &blocks,
                        plan.log_basis_inner,
                    )
                })
            } else {
                prepared.with_shared_ntt::<D, _>(plan.n_rows.saturating_mul(row_width), |ntt| {
                    mat_vec_mul_ntt_raw_digits_i8(ntt, plan.n_rows, row_width, &blocks)
                })
            }
        } else {
            let ring_elems: Vec<CyclotomicRing<F, D>> = plan
                .coeffs
                .iter()
                .map(|digit| {
                    let coeffs = from_fn(|k| F::from_i8(digit[k]));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect();
            let blocks = ring_elems
                .chunks(plan.num_positions_per_block)
                .take(plan.num_live_blocks)
                .collect::<Vec<_>>();
            prepared.with_shared_ntt::<D, _>(plan.n_rows.saturating_mul(row_width), |ntt| {
                mat_vec_mul_ntt_i8(
                    ntt,
                    plan.n_rows,
                    row_width,
                    &blocks,
                    plan.num_digits_inner,
                    plan.log_basis_inner,
                )
            })
        }
    }
}

/// Owned parts backing a [`StreamedASource`]: the materialized matrix when it
/// still covers the extent (pre-release), else a seed deriver (post-release).
#[allow(clippy::type_complexity)]
fn streamed_a_source_parts<F: FieldCore + CanonicalField, const D: usize>(
    prepared: &CpuPreparedSetup<F>,
    extent: usize,
) -> Result<
    (
        Option<std::sync::Arc<akita_types::FlatMatrix<F>>>,
        Option<akita_types::MatrixElementDeriver<F>>,
    ),
    AkitaError,
> {
    let shared = prepared.expanded.shared_matrix();
    if let Some(matrix) = shared.materialized_covering_at_dyn(extent, D) {
        return Ok((Some(matrix), None));
    }
    if shared.gen_ring_dim() != D {
        // Seed entries are generation-dimension rings; a mismatched view
        // cannot stream per-element — fall back to a derived prefix.
        return Ok((Some(shared.covering_at_dyn(extent, D)?), None));
    }
    let full = shared.total_ring_elements_at_dyn(D)?;
    if extent > full {
        return Err(AkitaError::InvalidSetup(format!(
            "streamed A extent {extent} exceeds the setup envelope {full}"
        )));
    }
    Ok((None, Some(shared.element_deriver())))
}

/// Borrow the parts into the kernel-facing source view.
fn streamed_a_source<'a, F: FieldCore + CanonicalField, const D: usize>(
    matrix: &'a Option<std::sync::Arc<akita_types::FlatMatrix<F>>>,
    deriver: &'a Option<akita_types::MatrixElementDeriver<F>>,
    extent: usize,
) -> Result<StreamedASource<'a, F, D>, AkitaError> {
    if let Some(matrix) = matrix {
        return Ok(StreamedASource::Flat(
            matrix.ring_view::<D>(1, extent)?.as_slice(),
        ));
    }
    let deriver = deriver
        .as_ref()
        .ok_or_else(|| AkitaError::InvalidSetup("streamed A source has no backing".into()))?;
    Ok(StreamedASource::Seed {
        deriver,
        len: extent,
    })
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
            prepared
                .expanded
                .shared_matrix
                .total_ring_elements_at::<D>()?,
        )?;
        prepared.with_shared_negacyclic_ntt::<D, _>(row_len.saturating_mul(digits.len()), |ntt| {
            mat_vec_mul_ntt_single_i8(ntt, row_len, digits.len(), digits, log_basis)
        })
    }
}

#[cfg(feature = "compression-diagnostics")]
impl<F> CompressionDiagnosticBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn compression_cache_bytes(&self, prepared: &Self::PreparedSetup) -> Option<usize> {
        Some(prepared.compression_ntt_cache_bytes())
    }

    fn compression_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        digit_vectors: &[&[[i8; D]]],
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        let input_width = validate_compression_batch_shape(digit_vectors)?;
        let total_ring_elements = prepared
            .expanded
            .shared_matrix
            .total_ring_elements_at::<D>()?;
        validate_digit_row_request(1, input_width, total_ring_elements)?;
        prepared.with_compression_ntt::<D, _>(input_width, |ntt| {
            mat_vec_mul_ntt_digits_i8(ntt, 1, input_width, digit_vectors, 1)
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
            prepared
                .expanded
                .shared_matrix
                .total_ring_elements_at::<D>()?,
        )?;
        prepared.with_shared_ntt::<D, _>(row_len.saturating_mul(digits.len()), |ntt| {
            mat_vec_mul_ntt_single_i8_cyclic(ntt, row_len, digits.len(), digits, log_basis)
        })
    }
}

impl<F> RingSwitchComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RingSwitchRelationRowsPlan<'_, D>,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField,
    {
        let extent = plan
            .n_d
            .saturating_mul(plan.e_hat.len())
            .max(plan.n_b.saturating_mul(plan.t_hat.len()))
            .max(plan.n_a.saturating_mul(plan.z_segment.len()));
        // The root-level relation spans nearly the whole matrix but reads
        // each element exactly once per prove — stream its transforms from
        // the field form instead of materializing a matrix-scale NTT cache
        // for one pass. Small (deeper-level) extents keep the cached path,
        // which is shared with the per-level digit-row products.
        if extent > NTT_STREAM_THRESHOLD_RING_ELEMENTS {
            let (matrix, deriver) = streamed_a_source_parts::<F, D>(prepared, extent)?;
            let source = streamed_a_source::<F, D>(&matrix, &deriver, extent)?;
            let streamed = prepared.with_shared_ntt::<D, _>(1, |ntt| {
                fused_split_eq_quotients_streamed_prover_bounds(
                    ntt,
                    &source,
                    plan.n_d,
                    plan.n_b,
                    plan.n_a,
                    plan.e_hat,
                    plan.t_hat,
                    plan.z_segment,
                    plan.z_folded_centered_inf_norm,
                    plan.log_basis_open,
                    plan.log_basis_outer,
                )
            })?;
            if let Some((d_cyclic, b_cyclic, a_quotients)) = streamed {
                return Ok(RingSwitchRelationRows {
                    d_cyclic,
                    b_cyclic,
                    a_quotients,
                });
            }
        }
        prepared.with_shared_ntt::<D, _>(extent, |ntt| {
            let (d_cyclic, b_cyclic, a_quotients) = fused_split_eq_quotients_prover_bounds(
                ntt,
                plan.n_d,
                plan.n_b,
                plan.n_a,
                plan.e_hat,
                plan.t_hat,
                plan.z_segment,
                plan.z_folded_centered_inf_norm,
                plan.log_basis_open,
                plan.log_basis_outer,
            )?;
            Ok(RingSwitchRelationRows {
                d_cyclic,
                b_cyclic,
                a_quotients,
            })
        })
    }

    fn ring_switch_quotient_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RingSwitchQuotientRowsPlan<'_, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: HalvingField,
    {
        let extent = plan.n_a.saturating_mul(plan.z_segment.len());
        if extent > NTT_STREAM_THRESHOLD_RING_ELEMENTS {
            let (matrix, deriver) = streamed_a_source_parts::<F, D>(prepared, extent)?;
            let source = streamed_a_source::<F, D>(&matrix, &deriver, extent)?;
            let streamed = prepared.with_shared_ntt::<D, _>(1, |ntt| {
                fused_split_eq_quotients_streamed_prover_bounds(
                    ntt,
                    &source,
                    0,
                    0,
                    plan.n_a,
                    &[][..],
                    &[][..],
                    plan.z_segment,
                    plan.z_folded_centered_inf_norm,
                    1,
                    1,
                )
            })?;
            if let Some((_d_cyclic, _b_cyclic, a_quotients)) = streamed {
                return Ok(a_quotients);
            }
        }
        prepared.with_shared_ntt::<D, _>(extent, |ntt| {
            let (_d_cyclic, _b_cyclic, a_quotients) = fused_split_eq_quotients_prover_bounds(
                ntt,
                0,
                0,
                plan.n_a,
                &[][..],
                &[][..],
                plan.z_segment,
                plan.z_folded_centered_inf_norm,
                1,
                1,
            )?;
            Ok(a_quotients)
        })
    }
}

#[cfg(test)]
mod tests;
