use crate::backend::onehot::{column_sweep_ajtai_onehot, MultiChunkEntry, SingleChunkEntry};
use crate::backend::sparse_ring::column_sweep_sparse;
use crate::compute::backend::{
    CommitmentComputeBackend, CompressionComputeBackend, CompressionRowsProducts,
    ComputeBackendSetup, CyclicRowsComputeBackend, DigitRowsComputeBackend,
    RingSwitchComputeBackend,
};
use crate::compute::plans::{
    DenseCommitInput, DenseCommitRowsPlan, OneHotCommitBlocks, OneHotCommitRowsPlan,
    RecursiveWitnessCommitRowsPlan, RingSwitchQuotientRowsPlan, RingSwitchRelationRows,
    RingSwitchRelationRowsPlan, SparseRingCommitRowsPlan,
};
use crate::kernels::linear::validate_compression_batch_shape;
use crate::kernels::linear::{
    digit_blocks_are_balanced, fused_split_eq_quotients_prover_bounds,
    mat_vec_mul_ntt_dense_digits_i8, mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_i8,
    mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row, mat_vec_mul_ntt_raw_digits_i8,
    mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic, selected_crt_i8_capacity_profile,
    CrtI8CapacityProfile,
};
use akita_algebra::CyclotomicRing;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_types::{
    dispatch_for_field, ntt_cache_requires_i16_tail, prepare_ntt_cache, AkitaExpandedSetup,
    NttCacheKey, NttCacheMode, NttTransformDomain, PreparedNttCache,
};
use std::any::Any;
use std::array::from_fn;
use std::collections::HashMap;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

mod compression_cache;

use compression_cache::CompressionNttCache;

/// CPU backend using the existing Rust/Rayon kernels.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

type NttSlotCell = OnceLock<Result<Arc<ErasedCpuNttCache>, AkitaError>>;

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
                NttTransformDomain::ExactNegacyclicI16 { .. } => 2,
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
                let profile =
                    dispatch_for_field!(ProtocolDispatchSlot::Ntt, F, ring_d, |RING_D| {
                        selected_crt_i8_capacity_profile::<F, RING_D>()
                    })?;
                let base_bytes = count
                    .checked_mul(ring_d)
                    .and_then(|bytes| bytes.checked_mul(profile.num_primes))
                    .and_then(|bytes| bytes.checked_mul(core::mem::size_of::<i32>()))
                    .ok_or_else(|| AkitaError::InvalidSetup("planned NTT bytes overflow".into()))?;
                let tail_bytes = match domain {
                    NttTransformDomain::ExactNegacyclicI16 { width, log_basis }
                        if dispatch_for_field!(
                            ProtocolDispatchSlot::Ntt,
                            F,
                            ring_d,
                            |RING_D| ntt_cache_requires_i16_tail::<F, RING_D>(width, log_basis)
                        )? =>
                    {
                        count
                            .checked_mul(ring_d)
                            .and_then(|bytes| bytes.checked_mul(core::mem::size_of::<i16>()))
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("planned i16-tail bytes overflow".into())
                            })?
                    }
                    _ => 0,
                };
                total
                    .checked_add(base_bytes)
                    .and_then(|bytes| bytes.checked_add(tail_bytes))
                    .ok_or_else(|| AkitaError::InvalidSetup("planned NTT bytes overflow".into()))
            })
    }

    /// In-memory byte footprint of exact-prefix compression NTT caches.
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
            NttTransformDomain::ExactNegacyclicI16 { width, log_basis } => {
                NttCacheMode::ExactNegacyclic { width, log_basis }
            }
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
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        plan.n_a,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| {
                        mat_vec_mul_ntt_dense_digits_i8(
                            ntt,
                            plan.n_a,
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
                if log_basis_inner > 8 {
                    prepared.with_shared_ntt::<D, _>(
                        NttCacheKey::from_matrix_shape(
                            D,
                            plan.n_a,
                            row_width,
                            NttTransformDomain::ExactNegacyclicI16 {
                                width: row_width,
                                log_basis: log_basis_inner,
                            },
                        )?,
                        |ntt| {
                            block_slices
                                .iter()
                                .map(|block| {
                                    let mut rhs = vec![[0i16; D]; row_width];
                                    for (ring_idx, ring) in block.iter().enumerate() {
                                        let start = ring_idx * num_digits_inner;
                                        ring.balanced_decompose_pow2_i16_into(
                                            &mut rhs[start..start + num_digits_inner],
                                            log_basis_inner,
                                        );
                                    }
                                    ntt.mat_vec_i16::<F>(log_basis_inner, plan.n_a, &rhs)
                                })
                                .collect()
                        },
                    )
                } else if plan.n_a == 1 {
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
                            plan.n_a,
                            row_width,
                            NttTransformDomain::Negacyclic,
                        )?,
                        |ntt| {
                            mat_vec_mul_ntt_i8_dense(
                                ntt,
                                plan.n_a,
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

    fn onehot_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: OneHotCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        let active_a_cols = plan
            .num_positions_per_block
            .checked_mul(plan.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
        let a_view = prepared
            .expanded
            .shared_matrix
            .ring_view::<D>(plan.n_a, active_a_cols)?;
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
        })
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
        let a_view = prepared
            .expanded
            .shared_matrix
            .ring_view::<D>(plan.n_a, active_a_cols)?;
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
        if plan.log_basis_inner > 8 {
            return prepared.with_shared_ntt::<D, _>(
                NttCacheKey::from_matrix_shape(
                    D,
                    plan.n_rows,
                    row_width,
                    NttTransformDomain::ExactNegacyclicI16 {
                        width: row_width,
                        log_basis: plan.log_basis_inner,
                    },
                )?,
                |ntt| {
                    plan.coeffs
                        .chunks(plan.num_positions_per_block)
                        .take(plan.num_live_blocks)
                        .map(|block| {
                            let mut rhs = vec![[0i16; D]; row_width];
                            if plan.num_digits_inner == 1 {
                                for (dst, src) in rhs.iter_mut().zip(block) {
                                    *dst = from_fn(|k| i16::from(src[k]));
                                }
                            } else {
                                for (ring_idx, digit) in block.iter().enumerate() {
                                    let ring = CyclotomicRing::from_coefficients(from_fn(|k| {
                                        F::from_i8(digit[k])
                                    }));
                                    let start = ring_idx * plan.num_digits_inner;
                                    ring.balanced_decompose_pow2_i16_into(
                                        &mut rhs[start..start + plan.num_digits_inner],
                                        plan.log_basis_inner,
                                    );
                                }
                            }
                            ntt.mat_vec_i16::<F>(plan.log_basis_inner, plan.n_rows, &rhs)
                        })
                        .collect()
                },
            );
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
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        plan.n_rows,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| {
                        mat_vec_mul_ntt_digits_i8(
                            ntt,
                            plan.n_rows,
                            row_width,
                            &blocks,
                            plan.log_basis_inner,
                        )
                    },
                )
            } else {
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        plan.n_rows,
                        row_width,
                        NttTransformDomain::Negacyclic,
                    )?,
                    |ntt| mat_vec_mul_ntt_raw_digits_i8(ntt, plan.n_rows, row_width, &blocks),
                )
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
            prepared.with_shared_ntt::<D, _>(
                NttCacheKey::from_matrix_shape(
                    D,
                    plan.n_rows,
                    row_width,
                    NttTransformDomain::Negacyclic,
                )?,
                |ntt| {
                    mat_vec_mul_ntt_i8(
                        ntt,
                        plan.n_rows,
                        row_width,
                        &blocks,
                        plan.num_digits_inner,
                        plan.log_basis_inner,
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
        let mut cyclic_requirement: Option<NttCacheKey> = None;
        for (rows, width) in [
            (plan.n_d, plan.e_hat.len()),
            (plan.n_b, plan.t_hat.len()),
            (plan.n_a, plan.z_segment.len()),
        ] {
            if rows == 0 && width == 0 {
                continue;
            }
            let role_requirement =
                NttCacheKey::from_matrix_shape(D, rows, width, NttTransformDomain::Cyclic)?;
            cyclic_requirement = Some(match cyclic_requirement {
                Some(current) => current.join(role_requirement)?,
                None => role_requirement,
            });
        }
        let cyclic_requirement = cyclic_requirement.ok_or_else(|| {
            AkitaError::InvalidSetup("ring-switch relation has no active rows".into())
        })?;
        prepared.with_shared_ntt::<D, _>(cyclic_requirement, |cyclic_ntt| {
            if plan.n_a == 0 {
                let (d_cyclic, b_cyclic, a_quotients) = fused_split_eq_quotients_prover_bounds(
                    cyclic_ntt,
                    cyclic_ntt,
                    plan.n_d,
                    plan.n_b,
                    0,
                    plan.e_hat,
                    plan.t_hat,
                    &[],
                    0,
                    plan.log_basis_open,
                    plan.log_basis_outer,
                )?;
                return Ok(RingSwitchRelationRows {
                    d_cyclic,
                    b_cyclic,
                    a_quotients,
                });
            }
            let negacyclic_requirement = NttCacheKey::from_matrix_shape(
                D,
                plan.n_a,
                plan.z_segment.len(),
                NttTransformDomain::Negacyclic,
            )?;
            prepared.with_shared_ntt::<D, _>(negacyclic_requirement, |negacyclic_ntt| {
                let (d_cyclic, b_cyclic, a_quotients) = fused_split_eq_quotients_prover_bounds(
                    negacyclic_ntt,
                    cyclic_ntt,
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
        let cyclic = NttCacheKey::from_matrix_shape(
            D,
            plan.n_a,
            plan.z_segment.len(),
            NttTransformDomain::Cyclic,
        )?;
        let negacyclic = NttCacheKey::from_matrix_shape(
            D,
            plan.n_a,
            plan.z_segment.len(),
            NttTransformDomain::Negacyclic,
        )?;
        prepared.with_shared_ntt::<D, _>(cyclic, |cyclic_ntt| {
            prepared.with_shared_ntt::<D, _>(negacyclic, |negacyclic_ntt| {
                let (_d_cyclic, _b_cyclic, a_quotients) = fused_split_eq_quotients_prover_bounds(
                    negacyclic_ntt,
                    cyclic_ntt,
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::backend::{
        ComputeBackendSetup, CyclicRowsComputeBackend, DigitRowsComputeBackend,
        RingSwitchComputeBackend,
    };
    use crate::compute::plans::RingSwitchRelationRowsPlan;
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
        CpuBackend.prepare_setup(&setup).unwrap()
    }

    #[test]
    fn cpu_prepared_setup_identity_rejects_mismatched_setup() {
        let setup_a =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
        let setup_b =
            AkitaProverSetup::<F>::generate_with_capacity(9, 1, setup_capacity(D)).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup_a).unwrap();

        CpuBackend
            .validate_prepared_setup(&prepared, setup_a.expanded.as_ref())
            .expect("matching setup");
        assert!(
            CpuBackend
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

        let prepared = CpuBackend.prepare_setup(&setup_a).unwrap();

        CpuBackend
            .validate_prepared_setup(&prepared, setup_b.expanded.as_ref())
            .expect("equivalent deterministic setup should validate");
    }

    #[test]
    fn cpu_prepared_setup_reports_checked_crt_capacity_profile() {
        let prepared = prepared();
        CpuBackend
            .digit_rows::<D>(&prepared, 1, &[[1i8; D]], 2)
            .expect("build exact NTT prefix");
        let profile = prepared.shared_ntt_profile::<D>().expect("profile");

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
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared");
        assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
        assert!(prepared.shared_ntt.lock().unwrap().is_empty());
    }

    #[test]
    fn cpu_prepared_setup_builds_only_requested_ntt_slots() {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_capacity(D)).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared");
        let partial_key = NttCacheKey {
            ring_d: D,
            num_ring_elements: 1,
            domain: NttTransformDomain::Negacyclic,
        };
        CpuBackend
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
        let prepared = CpuBackend
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
                    CpuBackend
                        .ensure_ntt_slot(prepared, key)
                        .expect("warm shared NTT slot");
                });
            }
        });
        CpuBackend
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
        CpuBackend
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
        CpuBackend
            .ensure_ntt_slot(&prepared, small)
            .expect("warm small prefix");
        CpuBackend
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

        CpuBackend
            .ensure_ntt_slot(&prepared, small)
            .expect("warm small prefix");
        assert!(CpuBackend.ensure_ntt_slot(&prepared, oversized).is_err());
        CpuBackend
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
            CpuBackend
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
                    CpuBackend
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

        assert!(CpuBackend.ensure_ntt_slot(&prepared, oversized).is_err());
        assert!(prepared.shared_ntt.lock().unwrap().is_empty());
        CpuBackend
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
            let failed = scope.spawn(|| CpuBackend.ensure_ntt_slot(&prepared, oversized));
            let warmed = scope.spawn(|| CpuBackend.ensure_ntt_slot(&prepared, valid));
            assert!(failed.join().expect("oversized warm thread").is_err());
            warmed
                .join()
                .expect("valid warm thread")
                .expect("valid warm must retry a failed covering entry");
        });
        CpuBackend
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

        CpuBackend
            .ring_switch_relation_rows::<D>(
                &prepared,
                RingSwitchRelationRowsPlan {
                    n_d: 2,
                    n_b: 1,
                    n_a: 1,
                    e_hat: &e_hat,
                    t_hat: &t_hat,
                    z_segment: &z_segment,
                    z_folded_centered_inf_norm: 1,
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

        let rows = CpuBackend
            .ring_switch_relation_rows::<D>(
                &prepared,
                RingSwitchRelationRowsPlan {
                    n_d: 0,
                    n_b: 2,
                    n_a: 0,
                    e_hat: &[],
                    t_hat: &t_hat,
                    z_segment: &[],
                    z_folded_centered_inf_norm: 0,
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
        let via_backend = CpuBackend
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
        let via_backend = CpuBackend
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
        let rows = CpuBackend
            .recursive_witness_commit_rows(
                &prepared,
                RecursiveWitnessCommitRowsPlan {
                    coeffs: &coeffs,
                    n_rows: 1,
                    num_positions_per_block: 2,
                    num_live_blocks: 2,
                    num_digits_inner: 1,
                    log_basis_inner: 3,
                    known_balanced_log_basis: Some(3),
                },
            )
            .expect("recursive commit rows");

        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn recursive_commit_selects_exact_i16_from_inner_basis() {
        let prepared = prepared();
        let coeffs = vec![[1i8; D], [-1i8; D]];
        let commit = |log_basis_inner| {
            CpuBackend
                .recursive_witness_commit_rows(
                    &prepared,
                    RecursiveWitnessCommitRowsPlan {
                        coeffs: &coeffs,
                        n_rows: 1,
                        num_positions_per_block: 2,
                        num_live_blocks: 1,
                        num_digits_inner: 1,
                        log_basis_inner,
                        known_balanced_log_basis: Some(2),
                    },
                )
                .expect("recursive commit rows")
        };

        assert_eq!(commit(3), commit(11));
        assert!(prepared.shared_ntt.lock().unwrap().contains_key(
            &NttCacheKey::from_matrix_shape(
                D,
                1,
                2,
                NttTransformDomain::ExactNegacyclicI16 {
                    width: 2,
                    log_basis: 11,
                },
            )
            .unwrap()
        ));
    }

    #[test]
    fn dense_coeff_commit_selects_exact_i16_from_inner_basis() {
        let prepared = prepared();
        let block = vec![
            CyclotomicRing::from_coefficients([F::one(); D]),
            CyclotomicRing::from_coefficients([F::from_i8(-1); D]),
        ];
        let commit = |log_basis_inner| {
            CpuBackend
                .dense_commit_rows(
                    &prepared,
                    DenseCommitRowsPlan {
                        n_a: 1,
                        input: DenseCommitInput::CoeffBlocks {
                            block_slices: vec![block.as_slice()],
                            num_digits_inner: 1,
                            log_basis_inner,
                        },
                    },
                )
                .expect("dense commit rows")
        };

        assert_eq!(commit(3), commit(11));
    }

    #[test]
    fn cpu_cyclic_digit_rows_match_direct_kernel() {
        let prepared = prepared();
        let digits = vec![[1i8; D], [0i8; D], [-2i8; D], [3i8; D]];
        let log_basis = 3;
        let via_backend = CpuBackend
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
        let via_backend = CpuBackend
            .ring_switch_relation_rows::<D>(
                &prepared,
                RingSwitchRelationRowsPlan {
                    n_d: 1,
                    n_b: 1,
                    n_a: 1,
                    e_hat: &e_hat,
                    t_hat: &t_hat,
                    z_segment: &z_segment,
                    z_folded_centered_inf_norm: 3,
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
