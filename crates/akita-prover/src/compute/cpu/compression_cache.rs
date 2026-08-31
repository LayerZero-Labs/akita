//! Exact-prefix paired NTT cache used by compressed commitments.

use super::prepared::ErasedCpuNttCache;
use akita_error::AkitaError;
use akita_types::{
    prepare_compression_ntt_cache, prepare_reduced_compression_ntt_cache, AkitaExpandedSetup,
    PreparedNttCache,
};
use jolt_field::{CanonicalEncoding, Field};
use std::any::Any;
use std::collections::HashMap;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

type CompressionSlotCell = OnceLock<Result<Arc<ErasedCpuNttCache>, AkitaError>>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CompressionNttCacheKey {
    ring_d: usize,
    input_width: usize,
    domains: CompressionNttDomains,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum CompressionNttDomains {
    Negacyclic,
    Both,
}

#[derive(Debug, Default)]
pub(super) struct CompressionNttCache {
    slots: Mutex<HashMap<CompressionNttCacheKey, Arc<CompressionSlotCell>>>,
    #[cfg(test)]
    slot_build_count: AtomicUsize,
}

impl CompressionNttCache {
    pub(super) fn with_ntt<F, const D: usize, R>(
        &self,
        expanded: &AkitaExpandedSetup<F>,
        input_width: usize,
        domains: CompressionNttDomains,
        f: impl FnOnce(&PreparedNttCache<D>) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError>
    where
        F: Field + CanonicalEncoding,
    {
        let key = CompressionNttCacheKey {
            ring_d: D,
            input_width,
            domains,
        };
        let entry = {
            let mut slots = self.slots.lock().map_err(|_| {
                AkitaError::InvalidSetup("compression NTT cache lock poisoned".into())
            })?;
            Arc::clone(
                slots
                    .entry(key)
                    .or_insert_with(|| Arc::new(OnceLock::new())),
            )
        };
        let build_result = entry.get_or_init(|| {
            #[cfg(test)]
            self.slot_build_count.fetch_add(1, Ordering::Relaxed);
            build_slot::<F, D>(expanded, input_width, domains).map(Arc::new)
        });
        let slot = build_result.as_ref().map_err(Clone::clone)?;
        if slot.ring_d != D {
            return Err(AkitaError::InvalidSetup(format!(
                "prepared compression NTT ring_d mismatch: stored {}, requested {D}",
                slot.ring_d
            )));
        }
        let typed = slot
            .cache
            .downcast_ref::<PreparedNttCache<D>>()
            .ok_or_else(|| {
                AkitaError::InvalidSetup("prepared compression NTT type mismatch".into())
            })?;
        f(typed)
    }

    pub(super) fn cache_bytes(&self) -> usize {
        self.slots
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .values()
            .filter_map(|entry| entry.get())
            .filter_map(|result| result.as_ref().ok())
            .map(|slot| slot.cache_bytes)
            .sum()
    }

    #[cfg(test)]
    fn slot_count(&self) -> usize {
        self.slots
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .len()
    }

    #[cfg(test)]
    fn slot_build_count(&self) -> usize {
        self.slot_build_count.load(Ordering::Relaxed)
    }
}

fn build_slot<F: Field + CanonicalEncoding, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    input_width: usize,
    domains: CompressionNttDomains,
) -> Result<ErasedCpuNttCache, AkitaError> {
    let view = expanded.shared_matrix().ring_view::<D>(1, input_width)?;
    let cache = Arc::new(match domains {
        CompressionNttDomains::Negacyclic => prepare_reduced_compression_ntt_cache(view)?,
        CompressionNttDomains::Both => prepare_compression_ntt_cache(view)?,
    });
    if !cache.has_negacyclic()
        || cache.has_cyclic() != matches!(domains, CompressionNttDomains::Both)
    {
        return Err(AkitaError::InvalidSetup(
            "compression NTT cache domains disagree with the requested relation mode".into(),
        ));
    }
    Ok(ErasedCpuNttCache {
        ring_d: D,
        cache_bytes: cache.cache_bytes(),
        cache: cache as Arc<dyn Any + Send + Sync>,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{CpuBackend, CpuPreparedSetup};
    use super::CompressionNttDomains;
    use crate::compute::{
        CompressionComputeBackend, ComputeBackendSetup, CyclicRowsComputeBackend,
    };
    use crate::AkitaProverSetup;
    use akita_types::{NttCacheKey, NttTransformDomain, SetupMatrixCapacity};
    use jolt_field::Prime64Offset59;

    type F = Prime64Offset59;
    const D: usize = 32;

    fn setup_envelope(num_field_elements: usize) -> SetupMatrixCapacity {
        SetupMatrixCapacity { num_field_elements }
    }

    fn empty_prepared() -> CpuPreparedSetup<F> {
        let setup =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, setup_envelope(6 * D)).unwrap();
        CpuBackend::DEFAULT
            .prepare_expanded(setup.expanded)
            .expect("empty prepared setup")
    }

    #[test]
    fn cache_is_exact_prefix_and_paired() {
        let prepared = empty_prepared();
        let vectors = [vec![[0i8; D]; 3], vec![[-1i8; D]; 3]];
        let views = vectors.iter().map(Vec::as_slice).collect::<Vec<_>>();

        CpuBackend::DEFAULT
            .compression_rows_products::<D>(&prepared, &views)
            .expect("compression rows");

        let expected_bytes = 2 * 3 * D * 3 * core::mem::size_of::<i32>();
        assert_eq!(prepared.compression_ntt_cache_bytes(), expected_bytes);
        assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
        prepared
            .with_compression_ntt::<D, _>(3, CompressionNttDomains::Both, |cache| {
                assert!(cache.has_cyclic());
                Ok(())
            })
            .expect("typed compression cache");
    }

    #[test]
    fn reduced_cache_is_exact_prefix_and_negacyclic_only() {
        let prepared = empty_prepared();
        let vectors = [vec![[0i8; D]; 3], vec![[-1i8; D]; 3]];
        let views = vectors.iter().map(Vec::as_slice).collect::<Vec<_>>();

        CpuBackend::DEFAULT
            .compression_negacyclic_rows::<D>(&prepared, &views)
            .expect("reduced compression rows");

        let expected_bytes = 3 * D * 3 * core::mem::size_of::<i32>();
        assert_eq!(prepared.compression_ntt_cache_bytes(), expected_bytes);
        prepared
            .with_compression_ntt::<D, _>(3, CompressionNttDomains::Negacyclic, |cache| {
                assert!(cache.has_negacyclic());
                assert!(!cache.has_cyclic());
                Ok(())
            })
            .expect("typed reduced compression cache");
    }

    #[test]
    fn cache_cannot_alias_full_envelope_both_transform_cache() {
        let prepared = empty_prepared();
        let envelope_width = prepared.expanded.shared_matrix().num_field_elements() / D;
        let compression_digits = vec![[0i8; D]; envelope_width];
        CpuBackend::DEFAULT
            .compression_rows_products::<D>(&prepared, &[compression_digits.as_slice()])
            .expect("compression cache at the full materialized prefix length");
        assert_eq!(prepared.shared_ntt_cache_bytes(), 0);

        let envelope_key = NttCacheKey {
            ring_d: D,
            num_ring_elements: envelope_width,
            domain: NttTransformDomain::Cyclic,
        };
        CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, envelope_key)
            .expect("independent cyclic envelope cache");
        assert!(prepared.shared_ntt_cache_bytes() > 0);
        let cyclic_digits = vec![[0i8; D]; envelope_width];
        CpuBackend::DEFAULT
            .cyclic_digit_rows::<D>(&prepared, 1, &cyclic_digits, 1)
            .expect("cyclic transform remains available");
    }

    #[test]
    fn concurrent_same_shape_warm_builds_once() {
        let prepared = empty_prepared();
        let digits = vec![[0i8; D]; 3];

        std::thread::scope(|scope| {
            for _ in 0..8 {
                let prepared = &prepared;
                let digits = &digits;
                scope.spawn(move || {
                    CpuBackend::DEFAULT
                        .compression_rows_products::<D>(prepared, &[digits.as_slice()])
                        .expect("compression rows");
                });
            }
        });

        assert_eq!(prepared.compression_ntt.slot_build_count(), 1);
    }

    #[test]
    fn cache_key_includes_input_width() {
        let prepared = empty_prepared();
        for input_width in [3, 6] {
            let digits = vec![[0i8; D]; input_width];
            CpuBackend::DEFAULT
                .compression_rows_products::<D>(&prepared, &[digits.as_slice()])
                .expect("compression rows");
        }

        assert_eq!(prepared.compression_ntt.slot_count(), 2);
        assert_eq!(prepared.compression_ntt.slot_build_count(), 2);
    }

    #[test]
    fn generic_release_retains_compression_and_rebuilds_shared_ntt() {
        let prepared = empty_prepared();
        let digits = vec![[0i8; D]; 3];
        let shared_key = NttCacheKey {
            ring_d: D,
            num_ring_elements: digits.len(),
            domain: NttTransformDomain::Cyclic,
        };
        CpuBackend::DEFAULT
            .compression_rows_products::<D>(&prepared, &[digits.as_slice()])
            .expect("warm compression NTT");
        CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, shared_key)
            .expect("warm shared NTT");

        let shared_bytes = prepared.shared_ntt_cache_bytes();
        let compression_bytes = prepared.compression_ntt_cache_bytes();
        let total_bytes = shared_bytes.checked_add(compression_bytes).unwrap();
        let shared_builds = prepared.ntt_slot_build_count();
        let compression_builds = prepared.compression_ntt.slot_build_count();
        assert!(shared_bytes > 0);
        assert!(compression_bytes > 0);
        assert_eq!(prepared.ntt_cache_bytes().unwrap(), total_bytes);

        assert_eq!(
            CpuBackend::DEFAULT
                .release_built_ntt_slots(&prepared)
                .unwrap(),
            shared_bytes
        );
        assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
        assert_eq!(prepared.compression_ntt_cache_bytes(), compression_bytes);
        assert_eq!(prepared.ntt_cache_bytes().unwrap(), compression_bytes);
        assert_eq!(prepared.compression_ntt.slot_count(), 1);
        assert_eq!(
            CpuBackend::DEFAULT
                .release_built_ntt_slots(&prepared)
                .unwrap(),
            0
        );

        CpuBackend::DEFAULT
            .compression_rows_products::<D>(&prepared, &[digits.as_slice()])
            .expect("reuse retained compression NTT");
        CpuBackend::DEFAULT
            .ensure_ntt_slot(&prepared, shared_key)
            .expect("rebuild shared NTT");
        assert_eq!(prepared.ntt_cache_bytes().unwrap(), total_bytes);
        assert_eq!(prepared.ntt_slot_build_count(), shared_builds + 1);
        assert_eq!(
            prepared.compression_ntt.slot_build_count(),
            compression_builds
        );
    }

    #[test]
    fn invalid_request_does_not_warm_a_cache_slot() {
        let prepared = empty_prepared();
        let valid = vec![[0i8; D]; 3];
        let short = vec![[0i8; D]; 2];

        assert!(CpuBackend::DEFAULT
            .compression_rows_products::<D>(&prepared, &[valid.as_slice(), short.as_slice()])
            .is_err());
        assert_eq!(prepared.compression_ntt.slot_count(), 0);
        assert_eq!(prepared.compression_ntt_cache_bytes(), 0);
    }
}
