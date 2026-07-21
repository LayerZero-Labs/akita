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

/// Prepared matrix transforms shared by prover and verifier caches.
///
/// Verifier slots leave `cyc` absent; prover slots materialize it for quotient
/// kernels. The CRT family remains statically typed in each variant.
#[derive(Debug)]
#[allow(missing_docs, clippy::large_enum_variant)]
pub enum PreparedNttSlot<const D: usize> {
    Q32 {
        neg: Vec<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>,
        cyc: Option<Vec<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i32, Q32_NUM_PRIMES, D>,
    },
    Q64 {
        neg: Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>,
        cyc: Option<Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i32, Q64_NUM_PRIMES, D>,
    },
    Q128 {
        neg: Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>,
        cyc: Option<Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i32, Q128_NUM_PRIMES, D>,
    },
}

impl<const D: usize> PreparedNttSlot<D> {
    /// In-memory byte footprint of all materialized transform entries.
    #[must_use]
    pub fn cache_bytes(&self) -> usize {
        match self {
            Self::Q32 { neg, cyc, .. } => {
                (neg.len() + cyc.as_ref().map_or(0, Vec::len))
                    * core::mem::size_of::<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>()
            }
            Self::Q64 { neg, cyc, .. } => {
                (neg.len() + cyc.as_ref().map_or(0, Vec::len))
                    * core::mem::size_of::<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>()
            }
            Self::Q128 { neg, cyc, .. } => {
                (neg.len() + cyc.as_ref().map_or(0, Vec::len))
                    * core::mem::size_of::<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>()
            }
        }
    }
}

/// Prepare exactly the supplied coefficient-matrix view in negacyclic NTT form.
#[tracing::instrument(skip_all, name = "build_negacyclic_ntt_slot", fields(ring_d = D, rings = mat.as_slice().len()))]
pub fn build_negacyclic_ntt_slot<F: FieldCore + CanonicalField, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
) -> Result<PreparedNttSlot<D>, AkitaError> {
    macro_rules! convert {
        ($params:expr, $variant:ident) => {{
            let params = $params;
            let neg = cfg_iter!(mat.as_slice())
                .map(|ring| CyclotomicCrtNtt::from_ring_with_params(ring, &params))
                .collect();
            PreparedNttSlot::$variant {
                neg,
                cyc: None,
                params,
            }
        }};
    }
    Ok(match select_crt_ntt_params::<F, D>()? {
        ProtocolCrtNttParams::Q32(params) => convert!(params, Q32),
        ProtocolCrtNttParams::Q64(params) => convert!(params, Q64),
        ProtocolCrtNttParams::Q128(params) => convert!(params, Q128),
    })
}

/// Prepare exactly the supplied coefficient-matrix view in both NTT domains.
#[tracing::instrument(skip_all, name = "build_negacyclic_and_cyclic_ntt_slot", fields(ring_d = D, rings = mat.as_slice().len()))]
pub fn build_negacyclic_and_cyclic_ntt_slot<F: FieldCore + CanonicalField, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
) -> Result<PreparedNttSlot<D>, AkitaError> {
    let params = select_crt_ntt_params::<F, D>()?;
    Ok(build_negacyclic_and_cyclic_ntt_slot_from_params(
        mat, params,
    ))
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
    W: akita_algebra::PrimeWidth,
{
    cfg_iter!(mat.as_slice())
        .map(|ring| CyclotomicCrtNtt::from_ring_pair_with_params(ring, params))
        .unzip()
}

fn build_negacyclic_and_cyclic_ntt_slot_from_params<
    F: FieldCore + CanonicalField,
    const D: usize,
>(
    mat: RingMatrixView<'_, F, D>,
    params: ProtocolCrtNttParams<D>,
) -> PreparedNttSlot<D> {
    match params {
        ProtocolCrtNttParams::Q32(params) => {
            let (neg, cyc) = convert_flat_pair(mat, &params);
            PreparedNttSlot::Q32 {
                neg,
                cyc: Some(cyc),
                params,
            }
        }
        ProtocolCrtNttParams::Q64(params) => {
            let (neg, cyc) = convert_flat_pair(mat, &params);
            PreparedNttSlot::Q64 {
                neg,
                cyc: Some(cyc),
                params,
            }
        }
        ProtocolCrtNttParams::Q128(params) => {
            let (neg, cyc) = convert_flat_pair(mat, &params);
            PreparedNttSlot::Q128 {
                neg,
                cyc: Some(cyc),
                params,
            }
        }
    }
}

/// Build a type-erased exact verifier prefix for a runtime NTT cache key.
pub(crate) fn build_verifier_ntt_slot_for_key<F: FieldCore + CanonicalField>(
    expanded: &AkitaExpandedSetup<F>,
    key: NttCacheKey,
) -> Result<PreparedNttSlotAny, AkitaError> {
    // The verifier cache key already selects one active ring dimension. Use
    // the outer-role dispatch table as the type-erased dimension registry;
    // this does not imply a terminal B role (terminals use A only).
    crate::dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Outer),
        F,
        key.ring_d,
        |D| {
            let matrix = expanded
                .shared_matrix()
                .ring_view::<D>(1, key.num_ring_elements)?;
            let slot = build_negacyclic_ntt_slot(matrix)?;
            let any: PreparedNttSlotAny = slot.into();
            Ok(any)
        }
    )
}

macro_rules! define_prepared_ntt_slot_any {
    ($( $d:literal => $variant:ident ),+ $(,)?) => {
        /// Type-erased prepared NTT slot over supported ring degrees.
        #[derive(Debug)]
        #[allow(clippy::large_enum_variant)]
        pub enum PreparedNttSlotAny {
            $( $variant(PreparedNttSlot<$d>), )+
        }

        impl PreparedNttSlotAny {
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
            pub fn as_d<const D: usize>(&self) -> Result<&PreparedNttSlot<D>, AkitaError> {
                if self.ring_d() != D {
                    return Err(AkitaError::InvalidSetup(format!(
                        "prepared NTT slot ring_d mismatch: stored {}, requested {D}",
                        self.ring_d()
                    )));
                }
                // SAFETY: the runtime degree uniquely selects the identical const-generic variant.
                Ok(unsafe { self.as_d_assuming_match::<D>() })
            }

            unsafe fn as_d_assuming_match<const D: usize>(&self) -> &PreparedNttSlot<D> {
                match self {
                    $( Self::$variant(slot) => &*(slot as *const PreparedNttSlot<$d> as *const PreparedNttSlot<D>), )+
                }
            }
        }

        $( impl From<PreparedNttSlot<$d>> for PreparedNttSlotAny {
            fn from(slot: PreparedNttSlot<$d>) -> Self { Self::$variant(slot) }
        } )+
    };
}

define_prepared_ntt_slot_any!(
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
    slots: Mutex<HashMap<NttCacheKey, Arc<PreparedNttSlotAny>>>,
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
    pub(crate) fn cache_bytes(&self) -> Result<usize, AkitaError> {
        let slots = self
            .slots
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("verifier NTT cache lock poisoned".into()))?;
        Ok(slots.values().map(|slot| slot.cache_bytes()).sum())
    }

    /// Build and atomically install an entry when needed.
    pub(crate) fn prepare(
        &self,
        key: NttCacheKey,
        build: impl FnOnce() -> Result<PreparedNttSlotAny, AkitaError>,
    ) -> Result<Arc<PreparedNttSlotAny>, AkitaError> {
        let covering = |slots: &HashMap<NttCacheKey, Arc<PreparedNttSlotAny>>| {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{
        Prime128Offset159, Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7,
        Prime32Offset99, Prime64Offset59,
    };

    fn sample_negacyclic_slot<F: FieldCore + CanonicalField, const D: usize>() -> PreparedNttSlot<D>
    {
        let ring = akita_algebra::CyclotomicRing::<F, D>::zero();
        let flat = crate::FlatMatrix::from_ring_slice(&[ring]);
        build_negacyclic_ntt_slot(flat.ring_view::<D>(1, 1).expect("view"))
            .expect("prepared NTT slot")
    }

    #[test]
    fn prepared_slot_materializes_only_requested_domains() {
        let neg_only = sample_negacyclic_slot::<Prime32Offset99, 64>();
        let ring = akita_algebra::CyclotomicRing::<Prime32Offset99, 64>::zero();
        let flat = crate::FlatMatrix::from_ring_slice(&[ring]);
        let both = build_negacyclic_and_cyclic_ntt_slot(flat.ring_view::<64>(1, 1).expect("view"))
            .expect("prepared NTT slot");
        let PreparedNttSlot::Q32 {
            cyc: neg_only_cyc, ..
        } = neg_only
        else {
            panic!("Q32 field must select Q32 transforms");
        };
        let PreparedNttSlot::Q32 { cyc: both_cyc, .. } = both else {
            panic!("Q32 field must select Q32 transforms");
        };
        assert!(neg_only_cyc.is_none());
        assert_eq!(both_cyc.as_ref().map(Vec::len), Some(1));
    }

    #[test]
    fn prepared_slot_any_rejects_ring_degree_mismatch() {
        let any: PreparedNttSlotAny = sample_negacyclic_slot::<Prime32Offset99, 64>().into();
        assert_eq!(any.ring_d(), 64);
        assert!(matches!(any.as_d::<32>(), Err(AkitaError::InvalidSetup(_))));
    }

    #[test]
    fn prepared_slot_any_maps_every_supported_ring_degree() {
        let slots: [PreparedNttSlotAny; 8] = [
            sample_negacyclic_slot::<Prime128OffsetA7F7, 16>().into(),
            sample_negacyclic_slot::<Prime64Offset59, 32>().into(),
            sample_negacyclic_slot::<Prime32Offset99, 64>().into(),
            sample_negacyclic_slot::<Prime32Offset99, 128>().into(),
            sample_negacyclic_slot::<Prime32Offset99, 256>().into(),
            sample_negacyclic_slot::<Prime32Offset99, 512>().into(),
            sample_negacyclic_slot::<Prime64Offset59, 1024>().into(),
            sample_negacyclic_slot::<Prime32Offset99, 2048>().into(),
        ];
        for (slot, expected) in slots.iter().zip([16, 32, 64, 128, 256, 512, 1024, 2048]) {
            assert_eq!(slot.ring_d(), expected);
        }
    }

    fn assert_selects_q32<F: CanonicalField, const D: usize>() {
        assert!(matches!(
            select_crt_ntt_params::<F, D>(),
            Ok(ProtocolCrtNttParams::Q32(_))
        ));
    }

    fn assert_selects_q64<F: CanonicalField, const D: usize>() {
        assert!(matches!(
            select_crt_ntt_params::<F, D>(),
            Ok(ProtocolCrtNttParams::Q64(_))
        ));
    }

    fn assert_selects_q128<F: CanonicalField, const D: usize>() {
        assert!(matches!(
            select_crt_ntt_params::<F, D>(),
            Ok(ProtocolCrtNttParams::Q128(_))
        ));
    }

    #[test]
    fn selects_supported_protocol_tier_bands() {
        assert!(matches!(
            select_crt_ntt_params::<Prime32Offset99, 32>(),
            Err(AkitaError::InvalidSetup(_))
        ));
        assert_selects_q32::<Prime32Offset99, 64>();
        assert_selects_q32::<Prime32Offset99, 128>();
        assert_selects_q32::<Prime32Offset99, 256>();

        assert_selects_q64::<Prime64Offset59, 32>();
        assert_selects_q64::<Prime64Offset59, 64>();
        assert_selects_q64::<Prime64Offset59, 128>();
        assert_selects_q64::<Prime64Offset59, 256>();

        assert_selects_q128::<Prime128OffsetA7F7, 16>();
        assert_selects_q128::<Prime128OffsetA7F7, 32>();
        assert_selects_q128::<Prime128OffsetA7F7, 64>();
        assert_selects_q128::<Prime128OffsetA7F7, 128>();
        assert_selects_q128::<Prime128Offset159, 32>();
        assert_selects_q128::<Prime128Offset2355, 32>();
        assert_selects_q128::<Prime128Offset275, 256>();
    }

    #[test]
    fn profile_caps_limit_crt_ring_degree_by_modulus() {
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
    fn selects_each_protocol_crt_family() {
        assert!(matches!(
            select_crt_ntt_params::<Prime32Offset99, 64>(),
            Ok(ProtocolCrtNttParams::Q32(_))
        ));
        assert!(matches!(
            select_crt_ntt_params::<Prime64Offset59, 64>(),
            Ok(ProtocolCrtNttParams::Q64(_))
        ));
        assert!(matches!(
            select_crt_ntt_params::<Prime128OffsetA7F7, 64>(),
            Ok(ProtocolCrtNttParams::Q128(_))
        ));
    }

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
