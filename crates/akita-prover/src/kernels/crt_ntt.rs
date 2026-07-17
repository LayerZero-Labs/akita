//! Protocol-facing CRT+NTT parameter dispatch and matrix caching.

use akita_algebra::ntt::prime::PrimeWidth;
use akita_algebra::ntt::tables::{Q128_NUM_PRIMES, Q32_NUM_PRIMES, Q64_NUM_PRIMES};
use akita_algebra::ring::{CrtNttParamSet, CyclotomicCrtNtt};
#[allow(unused_imports)]
use akita_field::parallel::*;
use akita_field::{cfg_iter, AkitaError, CanonicalField, FieldCore};

pub use akita_types::{select_crt_ntt_params, ProtocolCrtNttParams};
use akita_types::{NttCacheKey, RingMatrixView};

/// Pre-converted CRT+NTT cache for a flat 1D matrix, keyed by parameter family.
///
/// Stores both negacyclic (for mat-vec) and cyclic (for quotient) representations
/// as flat contiguous vectors. Callers provide `(num_rows, num_cols)` when
/// accessing elements, treating the flat data as a row-major matrix with
/// stride = `num_cols`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs, clippy::large_enum_variant)]
pub enum NttSlotCache<const D: usize> {
    /// 32-bit CRT primes.
    Q32 {
        neg: Vec<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>,
        cyc: Vec<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>,
        params: CrtNttParamSet<i32, Q32_NUM_PRIMES, D>,
    },
    /// 64-bit CRT primes.
    Q64 {
        neg: Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>,
        cyc: Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>,
        params: CrtNttParamSet<i32, Q64_NUM_PRIMES, D>,
    },
    /// 128-bit CRT primes.
    Q128 {
        neg: Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>,
        cyc: Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>,
        params: CrtNttParamSet<i32, Q128_NUM_PRIMES, D>,
    },
}

fn convert_flat_pair<F, W, const K: usize, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
    params: &CrtNttParamSet<W, K, D>,
) -> (
    Vec<CyclotomicCrtNtt<W, K, D>>,
    Vec<CyclotomicCrtNtt<W, K, D>>,
)
where
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
{
    cfg_iter!(mat.as_slice())
        .map(|ring| CyclotomicCrtNtt::from_ring_pair_with_params(ring, params))
        .unzip()
}

/// Build an NTT slot cache for a matrix view (flat 1D storage).
///
/// # Errors
///
/// Returns an error if no CRT+NTT parameter set matches the field modulus and ring degree.
#[tracing::instrument(skip_all, name = "build_ntt_slot")]
pub fn build_ntt_slot<F: FieldCore + CanonicalField, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
) -> Result<NttSlotCache<D>, AkitaError> {
    let params = select_crt_ntt_params::<F, D>()?;
    Ok(build_ntt_slot_from_params(mat, params))
}

fn build_ntt_slot_from_params<F: FieldCore + CanonicalField, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
    params: ProtocolCrtNttParams<D>,
) -> NttSlotCache<D> {
    match params {
        ProtocolCrtNttParams::Q32(p) => {
            let (neg, cyc) = convert_flat_pair(mat, &p);
            NttSlotCache::Q32 {
                neg,
                cyc,
                params: p,
            }
        }
        ProtocolCrtNttParams::Q64(p) => {
            let (neg, cyc) = convert_flat_pair(mat, &p);
            NttSlotCache::Q64 {
                neg,
                cyc,
                params: p,
            }
        }
        ProtocolCrtNttParams::Q128(p) => {
            let (neg, cyc) = convert_flat_pair(mat, &p);
            NttSlotCache::Q128 {
                neg,
                cyc,
                params: p,
            }
        }
    }
}

macro_rules! define_ntt_slot_cache_any {
    ( $( $d:literal => $v:ident ),+ $(,)? ) => {
        /// Type-erased prepared-setup NTT cache over supported NTT ring degrees.
        #[allow(clippy::large_enum_variant)]
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum NttSlotCacheAny {
            $( #[doc = concat!("Ring degree ", stringify!($d), ".")] $v(NttSlotCache<$d>), )+
        }

        impl NttSlotCacheAny {
            #[must_use]
            pub const fn ring_d(&self) -> usize {
                match self {
                    $( Self::$v(_) => $d, )+
                }
            }

            /// In-memory byte footprint of this cache entry.
            #[must_use]
            pub fn cache_bytes(&self) -> usize {
                match self {
                    $( Self::$v(cache) => cache.cache_bytes(), )+
                }
            }

            /// Cache-hit accessor. Returns [`AkitaError::InvalidSetup`] when the stored
            /// variant does not match the requested compile-time ring degree.
            pub fn as_d<const D: usize>(&self) -> Result<&NttSlotCache<D>, AkitaError> {
                if self.ring_d() != D {
                    return Err(AkitaError::InvalidSetup(format!(
                        "NTT cache ring_d mismatch: stored {}, requested {D}",
                        self.ring_d()
                    )));
                }
                // SAFETY: `ring_d()` equals `D`, so the active enum variant is the one
                // for degree `D` and the pointer cast is layout-identical.
                Ok(unsafe { self.as_d_assuming_match::<D>() })
            }

            #[inline]
            unsafe fn as_d_assuming_match<const D: usize>(&self) -> &NttSlotCache<D> {
                match self {
                    $( Self::$v(cache) => {
                        &*(cache as *const NttSlotCache<$d> as *const NttSlotCache<D>)
                    } )+
                }
            }
        }

        $( impl From<NttSlotCache<$d>> for NttSlotCacheAny {
            fn from(cache: NttSlotCache<$d>) -> Self {
                Self::$v(cache)
            }
        } )+
    };
}

define_ntt_slot_cache_any!(
    16 => D16,
    32 => D32,
    64 => D64,
    128 => D128,
    256 => D256,
    512 => D512,
    1024 => D1024,
    2048 => D2048,
);

/// Map of warmed NTT caches keyed by [`NttCacheKey`].
pub type NttCacheMap = std::collections::HashMap<NttCacheKey, std::sync::Arc<NttSlotCacheAny>>;

impl<const D: usize> NttSlotCache<D> {
    /// Total number of NTT elements stored in this cache.
    pub fn total_elements(&self) -> usize {
        match self {
            NttSlotCache::Q32 { neg, .. } => neg.len(),
            NttSlotCache::Q64 { neg, .. } => neg.len(),
            NttSlotCache::Q128 { neg, .. } => neg.len(),
        }
    }

    /// In-memory byte footprint of the negacyclic plus cyclic NTT slot
    /// vectors. Diagnostic surface for the profiler / bench report; the cache
    /// is the dominant prepared-setup allocation and is much larger than the
    /// plain setup vector it is built from.
    pub fn cache_bytes(&self) -> usize {
        match self {
            NttSlotCache::Q32 { neg, cyc, .. } => {
                (neg.len() + cyc.len())
                    * core::mem::size_of::<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>()
            }
            NttSlotCache::Q64 { neg, cyc, .. } => {
                (neg.len() + cyc.len())
                    * core::mem::size_of::<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>()
            }
            NttSlotCache::Q128 { neg, cyc, .. } => {
                (neg.len() + cyc.len())
                    * core::mem::size_of::<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{
        Prime128Offset159, Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7,
        Prime32Offset99, Prime64Offset59,
    };

    fn assert_selects_q32_params<F: CanonicalField, const D: usize>() {
        assert!(matches!(
            select_crt_ntt_params::<F, D>(),
            Ok(ProtocolCrtNttParams::Q32(_))
        ));
    }

    fn assert_selects_q64_params<F: CanonicalField, const D: usize>() {
        assert!(matches!(
            select_crt_ntt_params::<F, D>(),
            Ok(ProtocolCrtNttParams::Q64(_))
        ));
    }

    fn assert_selects_q128_params<F: CanonicalField, const D: usize>() {
        assert!(matches!(
            select_crt_ntt_params::<F, D>(),
            Ok(ProtocolCrtNttParams::Q128(_))
        ));
    }

    #[test]
    fn selects_q32_params_across_tier_ntt_band() {
        assert!(matches!(
            select_crt_ntt_params::<Prime32Offset99, 32>(),
            Err(AkitaError::InvalidSetup(_))
        ));
        assert_selects_q32_params::<Prime32Offset99, 64>();
        assert_selects_q32_params::<Prime32Offset99, 128>();
        assert_selects_q32_params::<Prime32Offset99, 256>();
    }

    #[test]
    fn selects_q64_params_across_tier_ntt_band() {
        assert_selects_q64_params::<Prime64Offset59, 32>();
        assert_selects_q64_params::<Prime64Offset59, 64>();
        assert_selects_q64_params::<Prime64Offset59, 128>();
        assert_selects_q64_params::<Prime64Offset59, 256>();
    }

    #[test]
    fn fp128_tier_ntt_accepts_d16() {
        assert_selects_q128_params::<Prime128OffsetA7F7, 16>();
    }

    #[test]
    fn profile_caps_limit_crt_ring_degree_by_modulus() {
        assert!(select_crt_ntt_params::<Prime32Offset99, 512>().is_ok());
        assert!(select_crt_ntt_params::<Prime32Offset99, 2048>().is_ok());
        assert!(select_crt_ntt_params::<Prime64Offset59, 1024>().is_ok());
        assert!(matches!(
            select_crt_ntt_params::<Prime64Offset59, 2048>(),
            Err(AkitaError::InvalidSetup(_))
        ));
        assert!(select_crt_ntt_params::<Prime128Offset275, 512>().is_ok());
        assert!(matches!(
            select_crt_ntt_params::<Prime128Offset275, 1024>(),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn selects_q128_params_for_prime275_across_challenge_ring_dims() {
        assert_selects_q128_params::<Prime128Offset275, 32>();
        assert_selects_q128_params::<Prime128Offset275, 64>();
        assert_selects_q128_params::<Prime128Offset275, 128>();
        assert_selects_q128_params::<Prime128Offset275, 256>();
    }

    #[test]
    fn selects_q128_params_for_split_only_prime159() {
        assert_selects_q128_params::<Prime128Offset159, 32>();
    }

    #[test]
    fn selects_q128_params_for_prime_a7f7_across_challenge_ring_dims() {
        assert_selects_q128_params::<Prime128OffsetA7F7, 32>();
        assert_selects_q128_params::<Prime128OffsetA7F7, 64>();
        assert_selects_q128_params::<Prime128OffsetA7F7, 128>();
    }

    #[test]
    fn selects_q128_params_for_prime2355_across_challenge_ring_dims() {
        assert_selects_q128_params::<Prime128Offset2355, 32>();
        assert_selects_q128_params::<Prime128Offset2355, 64>();
        assert_selects_q128_params::<Prime128Offset2355, 128>();
    }
}

#[cfg(test)]
mod ntt_slot_cache_any {
    use super::*;
    use akita_field::{AkitaError, Prime128OffsetA7F7, Prime32Offset99};
    use akita_types::FlatMatrix;

    fn sample_cache<F: FieldCore + CanonicalField, const D: usize>() -> NttSlotCache<D> {
        let ring = akita_algebra::CyclotomicRing::<F, D>::zero();
        let flat = FlatMatrix::from_ring_slice(&[ring]);
        build_ntt_slot(flat.ring_view::<D>(1, 1).expect("view")).expect("ntt slot")
    }

    #[test]
    fn as_d_returns_matching_variant() {
        let any: NttSlotCacheAny = sample_cache::<Prime32Offset99, 64>().into();
        assert!(any.as_d::<64>().is_ok());
    }

    #[test]
    fn as_d_rejects_ring_d_mismatch_without_panic() {
        let any: NttSlotCacheAny = sample_cache::<Prime32Offset99, 64>().into();
        let err = any.as_d::<32>().expect_err("mismatched ring_d");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn from_maps_each_supported_ring_degree() {
        let d16: NttSlotCacheAny = sample_cache::<Prime128OffsetA7F7, 16>().into();
        let d32: NttSlotCacheAny = sample_cache::<akita_field::Prime64Offset59, 32>().into();
        let d64: NttSlotCacheAny = sample_cache::<Prime32Offset99, 64>().into();
        let d128: NttSlotCacheAny = sample_cache::<Prime32Offset99, 128>().into();
        let d256: NttSlotCacheAny = sample_cache::<Prime32Offset99, 256>().into();
        assert_eq!(d16.ring_d(), 16);
        assert_eq!(d32.ring_d(), 32);
        assert_eq!(d64.ring_d(), 64);
        assert_eq!(d128.ring_d(), 128);
        assert_eq!(d256.ring_d(), 256);
    }

    #[test]
    fn ntt_cache_map_keys_by_ring_d_and_length() {
        let mut map = NttCacheMap::new();
        let key = NttCacheKey {
            ring_d: 64,
            num_ring_elements: 1,
        };
        map.insert(
            key,
            std::sync::Arc::new(sample_cache::<Prime32Offset99, 64>().into()),
        );
        assert!(map.contains_key(&key));
        assert_eq!(map.get(&key).expect("slot").ring_d(), 64);
    }
}
