//! Runtime ring-dimension NTT prepared-setup caches.

use akita_algebra::ntt::prime::PrimeWidth;
use akita_algebra::ntt::tables::{
    q128_primes, validate_profile_crt_ring_degree, Q128_MAX_RING_D, Q128_MODULUS, Q128_NUM_PRIMES,
    Q32_MAX_RING_D, Q32_MODULUS, Q32_NUM_PRIMES, Q32_PRIMES, Q64_MAX_RING_D, Q64_MODULUS,
    Q64_NUM_PRIMES, Q64_PRIMES,
};
use akita_algebra::{CrtNttParamSet, CyclotomicCrtNtt};
#[allow(unused_imports)]
use akita_field::parallel::*;
use akita_field::{
    cfg_iter, AkitaError, CanonicalField, FieldCore, Prime128Offset159, Prime128Offset2355,
    Prime128OffsetA7F7, PseudoMersenneField,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::{
    field_modulus, ntt_max_ring_d, ntt_min_ring_d, ntt_ring_degree_supported_for_field,
    proof::AkitaExpandedSetup, protocol_dispatch_tier, RingMatrixView,
};

/// Identifies one full-envelope NTT cache entry at a concrete ring degree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NttCacheKey {
    /// Ring dimension `D` for the cached transform family.
    pub ring_d: usize,
    /// Number of ring elements in the cached matrix view at `ring_d`.
    pub num_ring_elements: usize,
}

impl NttCacheKey {
    /// Build the full-envelope cache key for `ring_d` on `expanded`.
    ///
    /// # Errors
    ///
    /// Returns an error when `ring_d` does not divide the setup envelope or the
    /// matrix view length cannot be computed.
    pub fn from_envelope<F: FieldCore>(
        expanded: &AkitaExpandedSetup<F>,
        ring_d: usize,
    ) -> Result<Self, AkitaError> {
        let num_ring_elements = expanded
            .shared_matrix()
            .total_ring_elements_at_dyn(ring_d)?;
        Ok(Self {
            ring_d,
            num_ring_elements,
        })
    }
}

/// Supported protocol CRT+NTT parameter families.
#[derive(Clone)]
#[allow(missing_docs, clippy::large_enum_variant)]
pub enum ProtocolCrtNttParams<const D: usize> {
    Q32(CrtNttParamSet<i32, Q32_NUM_PRIMES, D>),
    Q64(CrtNttParamSet<i32, Q64_NUM_PRIMES, D>),
    Q128(CrtNttParamSet<i32, Q128_NUM_PRIMES, D>),
}

/// Select the canonical CRT+NTT parameter set for protocol field `F` and degree `D`.
pub fn select_crt_ntt_params<F: CanonicalField, const D: usize>(
) -> Result<ProtocolCrtNttParams<D>, AkitaError> {
    if !ntt_ring_degree_supported_for_field::<F>(D) {
        let tier = protocol_dispatch_tier::<F>();
        return Err(AkitaError::InvalidSetup(format!(
            "CRT+NTT ring degree {D} outside tier band [{}, {}] for this field",
            ntt_min_ring_d(tier),
            ntt_max_ring_d(tier),
        )));
    }

    let modulus = field_modulus::<F>();
    let split_only_q128_modulus =
        u128::MAX - (<Prime128Offset159 as PseudoMersenneField>::MODULUS_OFFSET - 1);
    let ntt_q128_modulus =
        u128::MAX - (<Prime128Offset2355 as PseudoMersenneField>::MODULUS_OFFSET - 1);
    let a7f7_q128_modulus =
        u128::MAX - (<Prime128OffsetA7F7 as PseudoMersenneField>::MODULUS_OFFSET - 1);

    if modulus <= Q32_MODULUS as u128 {
        if D >= 64 {
            validate_profile_crt_ring_degree(D, Q32_MAX_RING_D)?;
        }
        return Ok(ProtocolCrtNttParams::Q32(CrtNttParamSet::new(Q32_PRIMES)));
    }
    if modulus <= Q64_MODULUS as u128 {
        if D >= 64 {
            validate_profile_crt_ring_degree(D, Q64_MAX_RING_D)?;
        }
        return Ok(ProtocolCrtNttParams::Q64(CrtNttParamSet::new(Q64_PRIMES)));
    }
    if modulus == Q128_MODULUS
        || modulus == split_only_q128_modulus
        || modulus == ntt_q128_modulus
        || modulus == a7f7_q128_modulus
    {
        if D >= 64 {
            validate_profile_crt_ring_degree(D, Q128_MAX_RING_D)?;
        }
        return Ok(ProtocolCrtNttParams::Q128(CrtNttParamSet::new(
            q128_primes(),
        )));
    }
    Err(AkitaError::InvalidSetup(format!(
        "no CRT+NTT parameter set for modulus {modulus} and D={D}"
    )))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SmallNat {
    limbs: Vec<u32>,
}

impl SmallNat {
    fn one() -> Self {
        Self { limbs: vec![1] }
    }

    fn mul_u128(&mut self, rhs: u128) {
        if rhs == 0 {
            self.limbs = vec![0];
            return;
        }
        let mut rhs_limbs = Vec::new();
        let mut value = rhs;
        while value != 0 {
            rhs_limbs.push(value as u32);
            value >>= 32;
        }
        let mut out = vec![0u32; self.limbs.len() + rhs_limbs.len()];
        for (i, &lhs) in self.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &rhs) in rhs_limbs.iter().enumerate() {
                let index = i + j;
                let accum = u128::from(out[index]) + u128::from(lhs) * u128::from(rhs) + carry;
                out[index] = accum as u32;
                carry = accum >> 32;
            }
            let mut index = i + rhs_limbs.len();
            while carry != 0 {
                if index == out.len() {
                    out.push(0);
                }
                let accum = u128::from(out[index]) + carry;
                out[index] = accum as u32;
                carry = accum >> 32;
                index += 1;
            }
        }
        while out.len() > 1 && out.last() == Some(&0) {
            out.pop();
        }
        self.limbs = out;
    }
}

impl Ord for SmallNat {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        match self.limbs.len().cmp(&other.limbs.len()) {
            core::cmp::Ordering::Equal => self.limbs.iter().rev().cmp(other.limbs.iter().rev()),
            ordering => ordering,
        }
    }
}

impl PartialOrd for SmallNat {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn crt_width_is_safe<F: CanonicalField, const D: usize>(
    crt_product: &SmallNat,
    width: usize,
    rhs_abs_bound: u64,
) -> bool {
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let mut required = SmallNat::one();
    required.mul_u128(2);
    required.mul_u128(width as u128);
    required.mul_u128(D as u128);
    required.mul_u128(modulus / 2);
    required.mul_u128(u128::from(rhs_abs_bound));
    required < *crt_product
}

/// Conservative maximum matrix width that one signed CRT accumulator can hold.
///
/// The bound covers all `D` convolution coefficients and centered setup entries:
/// `2 * width * D * floor(q/2) * rhs_abs_bound < product(CRT primes)`.
pub fn max_safe_crt_accumulation_width<
    F: CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    params: &CrtNttParamSet<W, K, D>,
    rhs_abs_bound: u64,
) -> Option<usize> {
    if rhs_abs_bound == 0 {
        return Some(usize::MAX);
    }
    let modulus = (-F::one()).to_canonical_u128() + 1;
    if modulus <= 1 || D == 0 {
        return None;
    }
    let mut crt_product = SmallNat::one();
    for prime in &params.primes {
        crt_product.mul_u128(prime.p.to_i64() as u128);
    }
    if !crt_width_is_safe::<F, D>(&crt_product, 1, rhs_abs_bound) {
        return None;
    }
    let mut low = 1usize;
    let mut high = 2usize;
    while crt_width_is_safe::<F, D>(&crt_product, high, rhs_abs_bound) {
        low = high;
        let Some(next) = high.checked_mul(2) else {
            if crt_width_is_safe::<F, D>(&crt_product, usize::MAX, rhs_abs_bound) {
                return Some(usize::MAX);
            }
            high = usize::MAX;
            break;
        };
        high = next;
    }
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if crt_width_is_safe::<F, D>(&crt_product, mid, rhs_abs_bound) {
            low = mid;
        } else {
            high = mid;
        }
    }
    Some(low)
}

/// Negacyclic-only prepared matrix prefix used by verifier commitment checks.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs, clippy::large_enum_variant)]
pub enum VerifierNttSlot<const D: usize> {
    Q32 {
        neg: Vec<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>,
        params: CrtNttParamSet<i32, Q32_NUM_PRIMES, D>,
    },
    Q64 {
        neg: Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>,
        params: CrtNttParamSet<i32, Q64_NUM_PRIMES, D>,
    },
    Q128 {
        neg: Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>,
        params: CrtNttParamSet<i32, Q128_NUM_PRIMES, D>,
    },
}

impl<const D: usize> VerifierNttSlot<D> {
    /// Number of prepared matrix rings.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Q32 { neg, .. } => neg.len(),
            Self::Q64 { neg, .. } => neg.len(),
            Self::Q128 { neg, .. } => neg.len(),
        }
    }

    /// Whether the prepared prefix is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// In-memory byte footprint of the negacyclic entries.
    #[must_use]
    pub fn cache_bytes(&self) -> usize {
        match self {
            Self::Q32 { neg, .. } => {
                neg.len() * core::mem::size_of::<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>()
            }
            Self::Q64 { neg, .. } => {
                neg.len() * core::mem::size_of::<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>()
            }
            Self::Q128 { neg, .. } => {
                neg.len() * core::mem::size_of::<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>()
            }
        }
    }
}

/// Prepare exactly the supplied coefficient-matrix prefix in negacyclic NTT form.
#[tracing::instrument(skip_all, name = "build_verifier_negacyclic_ntt_prefix", fields(ring_d = D, rings = mat.as_slice().len()))]
pub(crate) fn build_verifier_ntt_slot<F: FieldCore + CanonicalField, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
) -> Result<VerifierNttSlot<D>, AkitaError> {
    macro_rules! convert {
        ($params:expr, $variant:ident) => {{
            let params = $params;
            let neg = cfg_iter!(mat.as_slice())
                .map(|ring| CyclotomicCrtNtt::from_ring_with_params(ring, &params))
                .collect();
            VerifierNttSlot::$variant { neg, params }
        }};
    }
    Ok(match select_crt_ntt_params::<F, D>()? {
        ProtocolCrtNttParams::Q32(params) => convert!(params, Q32),
        ProtocolCrtNttParams::Q64(params) => convert!(params, Q64),
        ProtocolCrtNttParams::Q128(params) => convert!(params, Q128),
    })
}

/// Build a type-erased exact verifier prefix for a runtime NTT cache key.
pub(crate) fn build_verifier_ntt_slot_for_key<F: FieldCore + CanonicalField>(
    expanded: &AkitaExpandedSetup<F>,
    key: NttCacheKey,
) -> Result<VerifierNttSlotAny, AkitaError> {
    // Terminal A/B roles use the union represented by the outer-role table.
    // The broader NTT table also contains setup-envelope-only dimensions and
    // made every verifier instantiation compile kernels it can never call.
    crate::dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Outer),
        F,
        key.ring_d,
        |D| {
            let matrix = expanded
                .shared_matrix()
                .ring_view::<D>(1, key.num_ring_elements)?;
            Ok(build_verifier_ntt_slot(matrix)?.into())
        }
    )
}

macro_rules! define_verifier_ntt_slot_any {
    ($( $d:literal => $variant:ident ),+ $(,)?) => {
        /// Type-erased verifier NTT prefix over supported ring degrees.
        #[derive(Debug, Clone, PartialEq, Eq)]
        #[allow(clippy::large_enum_variant)]
        pub enum VerifierNttSlotAny {
            $( $variant(VerifierNttSlot<$d>), )+
        }

        impl VerifierNttSlotAny {
            /// Runtime ring degree.
            #[must_use]
            pub const fn ring_d(&self) -> usize {
                match self { $( Self::$variant(_) => $d, )+ }
            }

            /// In-memory byte footprint.
            #[must_use]
            pub fn cache_bytes(&self) -> usize {
                match self { $( Self::$variant(slot) => slot.cache_bytes(), )+ }
            }

            /// Checked typed access.
            pub fn as_d<const D: usize>(&self) -> Result<&VerifierNttSlot<D>, AkitaError> {
                if self.ring_d() != D {
                    return Err(AkitaError::InvalidSetup(format!(
                        "verifier NTT cache ring_d mismatch: stored {}, requested {D}",
                        self.ring_d()
                    )));
                }
                // SAFETY: the runtime degree uniquely selects the identical const-generic variant.
                Ok(unsafe { self.as_d_assuming_match::<D>() })
            }

            unsafe fn as_d_assuming_match<const D: usize>(&self) -> &VerifierNttSlot<D> {
                match self {
                    $( Self::$variant(slot) => &*(slot as *const VerifierNttSlot<$d> as *const VerifierNttSlot<D>), )+
                }
            }
        }

        $( impl From<VerifierNttSlot<$d>> for VerifierNttSlotAny {
            fn from(slot: VerifierNttSlot<$d>) -> Self { Self::$variant(slot) }
        } )+
    };
}

define_verifier_ntt_slot_any!(
    16 => D16,
    32 => D32,
    64 => D64,
    128 => D128,
    256 => D256,
    512 => D512,
    1024 => D1024,
    2048 => D2048,
);

/// Derived verifier cache. It is deliberately excluded from setup serialization and equality.
#[derive(Default)]
pub(crate) struct VerifierNttCache {
    slots: Mutex<HashMap<NttCacheKey, Arc<VerifierNttSlotAny>>>,
}

impl core::fmt::Debug for VerifierNttCache {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.slots.lock() {
            Ok(slots) => formatter
                .debug_struct("VerifierNttCache")
                .field("keys", &slots.keys().collect::<Vec<_>>())
                .field(
                    "cache_bytes",
                    &slots.values().map(|slot| slot.cache_bytes()).sum::<usize>(),
                )
                .finish(),
            Err(_) => formatter
                .debug_struct("VerifierNttCache")
                .field("state", &"poisoned")
                .finish(),
        }
    }
}

impl VerifierNttCache {
    /// Build and atomically install an entry when needed.
    pub(crate) fn prepare(
        &self,
        key: NttCacheKey,
        build: impl FnOnce() -> Result<VerifierNttSlotAny, AkitaError>,
    ) -> Result<Arc<VerifierNttSlotAny>, AkitaError> {
        let covering = |slots: &HashMap<NttCacheKey, Arc<VerifierNttSlotAny>>| {
            slots
                .iter()
                .filter(|(candidate, _)| {
                    candidate.ring_d == key.ring_d
                        && candidate.num_ring_elements >= key.num_ring_elements
                })
                .min_by_key(|(candidate, _)| candidate.num_ring_elements)
                .map(|(_, slot)| Arc::clone(slot))
        };
        let slots = self
            .slots
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("verifier NTT cache lock poisoned".into()))?;
        if let Some(slot) = covering(&slots) {
            return Ok(slot);
        }
        drop(slots);
        let built = Arc::new(build()?);
        let mut slots = self
            .slots
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("verifier NTT cache lock poisoned".into()))?;
        if let Some(slot) = covering(&slots) {
            return Ok(slot);
        }
        slots.retain(|candidate, _| {
            candidate.ring_d != key.ring_d || candidate.num_ring_elements > key.num_ring_elements
        });
        slots.insert(key, Arc::clone(&built));
        Ok(built)
    }

    /// Total prepared verifier cache bytes.
    pub(crate) fn cache_bytes(&self) -> Result<usize, AkitaError> {
        Ok(self
            .slots
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("verifier NTT cache lock poisoned".into()))?
            .values()
            .map(|slot| slot.cache_bytes())
            .sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;

    #[test]
    fn q128_d64_centered_z_capacity_matches_profile_tail() {
        const D: usize = 64;
        let ProtocolCrtNttParams::Q128(params) =
            select_crt_ntt_params::<Prime128Offset275, D>().expect("Q128 parameters")
        else {
            panic!("fp128 must select Q128 parameters");
        };
        assert_eq!(
            max_safe_crt_accumulation_width::<Prime128Offset275, _, Q128_NUM_PRIMES, D>(
                &params, 2015,
            ),
            Some(32)
        );
        assert_eq!(
            max_safe_crt_accumulation_width::<Prime128Offset275, _, Q128_NUM_PRIMES, D>(
                &params, 1510,
            ),
            Some(43)
        );
    }
}
