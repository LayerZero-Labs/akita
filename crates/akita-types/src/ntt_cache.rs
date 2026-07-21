//! Runtime ring-dimension NTT prepared-setup caches.

use akita_algebra::ntt::prime::PrimeWidth;
use akita_algebra::ntt::tables::{
    q128_primes, validate_profile_crt_ring_degree, I16_TAIL_PRIME, Q128_MAX_RING_D, Q128_MODULUS,
    Q128_NUM_PRIMES, Q32_MAX_RING_D, Q32_MODULUS, Q32_NUM_PRIMES, Q32_PRIMES, Q64_MAX_RING_D,
    Q64_MODULUS, Q64_NUM_PRIMES, Q64_PRIMES,
};
use akita_algebra::{CrtNttParamSet, CyclotomicCrtNtt, I16TailParams};
#[allow(unused_imports)]
use akita_field::parallel::*;
use akita_field::{
    cfg_iter, AkitaError, CanonicalField, FieldCore, Prime128Offset159, Prime128Offset2355,
    Prime128OffsetA7F7, PseudoMersenneField,
};
use std::any::Any;
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

fn crt_product<W: PrimeWidth, const K: usize, const D: usize>(
    params: &CrtNttParamSet<W, K, D>,
) -> SmallNat {
    let mut product = SmallNat::one();
    for prime in &params.primes {
        product.mul_u128(prime.p.to_i64() as u128);
    }
    product
}

fn required_profile_for_params<F, W, const K: usize, const D: usize>(
    params: &CrtNttParamSet<W, K, D>,
    width: usize,
    rhs_abs_bound: u64,
) -> Result<bool, AkitaError>
where
    F: CanonicalField,
    W: PrimeWidth,
{
    let mut product = crt_product(params);
    if crt_width_is_safe::<F, D>(&product, width, rhs_abs_bound) {
        return Ok(false);
    }
    product.mul_u128(I16_TAIL_PRIME.p as u128);
    if crt_width_is_safe::<F, D>(&product, width, rhs_abs_bound) {
        return Ok(true);
    }
    Err(AkitaError::InvalidSetup(format!(
        "CRT accumulation exceeds base plus i16-tail capacity for D={D}, width={width}, rhs_abs_bound={rhs_abs_bound}"
    )))
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
    let crt_product = crt_product(params);
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

/// NTT representations requested by protocol and backend consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NttCacheMode {
    /// Materialize the base-profile negacyclic and cyclic transforms.
    BothTransforms,
    /// Materialize the minimum exact negacyclic representation for balanced
    /// base-`2^log_basis` signed coefficients.
    ExactNegacyclic { width: usize, log_basis: u32 },
}

/// Optional homogeneous i16 tail attached to one prepared base profile.
///
/// This type is public only because [`PreparedNttCache`] crosses crate
/// boundaries. Its fields and construction remain private to the cache
/// implementation.
#[doc(hidden)]
#[derive(Debug)]
pub struct PreparedI16Tail<const K: usize, const D: usize> {
    negacyclic: Vec<CyclotomicCrtNtt<i16, 1, D>>,
    params: I16TailParams<K, D>,
}

/// One prepared NTT cache over the field-selected CRT profile.
///
/// Every variant always contains the base negacyclic representation. `cyc` is
/// present only for [`NttCacheMode::BothTransforms`]; `tail` is present only
/// when an exact negacyclic request exceeds the base CRT product.
#[derive(Debug)]
#[allow(missing_docs, clippy::large_enum_variant)]
pub enum PreparedNttCache<const D: usize> {
    #[non_exhaustive]
    Q32 {
        neg: Vec<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>,
        cyc: Option<Vec<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i32, Q32_NUM_PRIMES, D>,
        tail: Option<PreparedI16Tail<Q32_NUM_PRIMES, D>>,
        exact: bool,
    },
    #[non_exhaustive]
    Q64 {
        neg: Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>,
        cyc: Option<Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i32, Q64_NUM_PRIMES, D>,
        tail: Option<PreparedI16Tail<Q64_NUM_PRIMES, D>>,
        exact: bool,
    },
    #[non_exhaustive]
    Q128 {
        neg: Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>,
        cyc: Option<Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i32, Q128_NUM_PRIMES, D>,
        tail: Option<PreparedI16Tail<Q128_NUM_PRIMES, D>>,
        exact: bool,
    },
}

impl<const D: usize> PreparedNttCache<D> {
    fn validate(&self) -> Result<(), AkitaError> {
        macro_rules! validate {
            ($neg:expr, $cyc:expr, $params:expr, $tail:expr, $exact:expr) => {{
                if $exact == $cyc.is_some() {
                    return Err(AkitaError::InvalidSetup(
                        "prepared NTT cache has an unsupported domain combination".into(),
                    ));
                }
                if $cyc.as_ref().is_some_and(|cyc| cyc.len() != $neg.len()) {
                    return Err(AkitaError::InvalidSetup(
                        "prepared cyclic and negacyclic NTT lengths differ".into(),
                    ));
                }
                if $tail.as_ref().is_some_and(|tail| {
                    tail.negacyclic.is_empty() || tail.negacyclic.len() > $neg.len()
                }) {
                    return Err(AkitaError::InvalidSetup(
                        "prepared i16-tail NTT prefix is empty or exceeds its base".into(),
                    ));
                }
                if let Some(tail) = $tail.as_ref() {
                    if !$exact
                        || tail.params.wide != *$params
                        || tail.params.tail.primes != [I16_TAIL_PRIME]
                    {
                        return Err(AkitaError::InvalidSetup(
                            "prepared i16-tail NTT parameters do not match the base".into(),
                        ));
                    }
                }
            }};
        }
        match self {
            Self::Q32 {
                neg,
                cyc,
                params,
                tail,
                exact,
            } => validate!(neg, cyc, params, tail, *exact),
            Self::Q64 {
                neg,
                cyc,
                params,
                tail,
                exact,
            } => validate!(neg, cyc, params, tail, *exact),
            Self::Q128 {
                neg,
                cyc,
                params,
                tail,
                exact,
            } => validate!(neg, cyc, params, tail, *exact),
        }
        Ok(())
    }

    /// In-memory byte footprint of all materialized matrix transforms.
    #[must_use]
    pub fn cache_bytes(&self) -> usize {
        macro_rules! bytes {
            ($neg:expr, $cyc:expr, $tail:expr, $k:expr) => {{
                let base_entries = $neg.len() + $cyc.as_ref().map_or(0, Vec::len);
                let base = base_entries * D * $k * core::mem::size_of::<i32>();
                let tail = $tail.as_ref().map_or(0, |tail| {
                    tail.negacyclic.len() * D * core::mem::size_of::<i16>()
                });
                base + tail
            }};
        }
        match self {
            Self::Q32 { neg, cyc, tail, .. } => bytes!(neg, cyc, tail, Q32_NUM_PRIMES),
            Self::Q64 { neg, cyc, tail, .. } => bytes!(neg, cyc, tail, Q64_NUM_PRIMES),
            Self::Q128 { neg, cyc, tail, .. } => bytes!(neg, cyc, tail, Q128_NUM_PRIMES),
        }
    }

    /// Whether the cyclic base representation was materialized.
    #[must_use]
    pub const fn has_cyclic(&self) -> bool {
        match self {
            Self::Q32 { cyc, .. } => cyc.is_some(),
            Self::Q64 { cyc, .. } => cyc.is_some(),
            Self::Q128 { cyc, .. } => cyc.is_some(),
        }
    }

    /// Whether the exactness tail was materialized.
    #[must_use]
    pub const fn has_i16_tail(&self) -> bool {
        match self {
            Self::Q32 { tail, .. } => tail.is_some(),
            Self::Q64 { tail, .. } => tail.is_some(),
            Self::Q128 { tail, .. } => tail.is_some(),
        }
    }

    /// Compute a shape-checked exact signed-i16 matrix product.
    pub fn mat_vec_i16<F: FieldCore + CanonicalField>(
        &self,
        log_basis: u32,
        num_rows: usize,
        rhs: &[[i16; D]],
    ) -> Result<Vec<akita_algebra::CyclotomicRing<F, D>>, AkitaError> {
        self.validate()?;
        match self {
            Self::Q32 {
                neg,
                params,
                tail,
                exact,
                ..
            } => {
                if !matches!(
                    select_crt_ntt_params::<F, D>()?,
                    ProtocolCrtNttParams::Q32(_)
                ) {
                    return Err(AkitaError::InvalidSetup(
                        "prepared NTT field profile mismatch".into(),
                    ));
                }
                mat_vec_i16_from_cache(neg, params, tail.as_ref(), *exact, log_basis, num_rows, rhs)
            }
            Self::Q64 {
                neg,
                params,
                tail,
                exact,
                ..
            } => {
                if !matches!(
                    select_crt_ntt_params::<F, D>()?,
                    ProtocolCrtNttParams::Q64(_)
                ) {
                    return Err(AkitaError::InvalidSetup(
                        "prepared NTT field profile mismatch".into(),
                    ));
                }
                mat_vec_i16_from_cache(neg, params, tail.as_ref(), *exact, log_basis, num_rows, rhs)
            }
            Self::Q128 {
                neg,
                params,
                tail,
                exact,
                ..
            } => {
                if !matches!(
                    select_crt_ntt_params::<F, D>()?,
                    ProtocolCrtNttParams::Q128(_)
                ) {
                    return Err(AkitaError::InvalidSetup(
                        "prepared NTT field profile mismatch".into(),
                    ));
                }
                mat_vec_i16_from_cache(neg, params, tail.as_ref(), *exact, log_basis, num_rows, rhs)
            }
        }
    }
}

fn mat_vec_i16_from_cache<F, const K: usize, const D: usize>(
    neg: &[CyclotomicCrtNtt<i32, K, D>],
    params: &CrtNttParamSet<i32, K, D>,
    tail: Option<&PreparedI16Tail<K, D>>,
    exact: bool,
    log_basis: u32,
    num_rows: usize,
    rhs: &[[i16; D]],
) -> Result<Vec<akita_algebra::CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if !exact {
        return Err(AkitaError::InvalidSetup(
            "signed-i16 matvec requested from a cyclic cache".into(),
        ));
    }
    if !(1..=16).contains(&log_basis) {
        return Err(AkitaError::InvalidProof);
    }
    let width = rhs.len();
    if width == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let bound = 1i32 << (log_basis - 1);
    if rhs
        .iter()
        .flatten()
        .any(|&digit| !(-bound..bound).contains(&i32::from(digit)))
    {
        return Err(AkitaError::InvalidProof);
    }
    let needs_tail =
        required_profile_for_params::<F, _, K, D>(params, width, 1u64 << (log_basis - 1))?;
    if needs_tail {
        let tail = tail.ok_or_else(|| {
            AkitaError::InvalidSetup("prepared exact NTT cache is missing its required tail".into())
        })?;
        akita_algebra::mat_vec_i16_with_tail(
            neg,
            &tail.negacyclic,
            num_rows,
            width,
            rhs,
            &tail.params,
        )
    } else {
        CyclotomicCrtNtt::mat_vec_i16(neg, num_rows, width, rhs, params)
    }
}

fn validate_cache_mode(mode: NttCacheMode) -> Result<(), AkitaError> {
    if let NttCacheMode::ExactNegacyclic { width, log_basis } = mode {
        if width == 0 {
            return Err(AkitaError::InvalidSetup(
                "exact negacyclic NTT width must be nonzero".into(),
            ));
        }
        if !(1..=16).contains(&log_basis) {
            return Err(AkitaError::InvalidSetup(
                "exact negacyclic log_basis must be in 1..=16".into(),
            ));
        }
    }
    Ok(())
}

/// Return whether an exact balanced-digit request requires the i16 tail.
pub fn ntt_cache_requires_i16_tail<F: CanonicalField, const D: usize>(
    width: usize,
    log_basis: u32,
) -> Result<bool, AkitaError> {
    let mode = NttCacheMode::ExactNegacyclic { width, log_basis };
    validate_cache_mode(mode)?;
    let rhs_abs_bound = 1u64 << (log_basis - 1);
    Ok(match select_crt_ntt_params::<F, D>()? {
        ProtocolCrtNttParams::Q32(params) => {
            required_profile_for_params::<F, _, Q32_NUM_PRIMES, D>(&params, width, rhs_abs_bound)?
        }
        ProtocolCrtNttParams::Q64(params) => {
            required_profile_for_params::<F, _, Q64_NUM_PRIMES, D>(&params, width, rhs_abs_bound)?
        }
        ProtocolCrtNttParams::Q128(params) => {
            required_profile_for_params::<F, _, Q128_NUM_PRIMES, D>(&params, width, rhs_abs_bound)?
        }
    })
}

/// Prepare exactly the NTT representations requested by `mode`.
#[tracing::instrument(skip_all, name = "prepare_ntt_cache", fields(ring_d = D, rings = matrix.as_slice().len(), ?mode))]
pub fn prepare_ntt_cache<F: FieldCore + CanonicalField, const D: usize>(
    matrix: RingMatrixView<'_, F, D>,
    mode: NttCacheMode,
) -> Result<PreparedNttCache<D>, AkitaError> {
    prepare_ntt_cache_with_tail_prefix(matrix, mode, None)
}

fn prepare_ntt_cache_with_tail_prefix<F: FieldCore + CanonicalField, const D: usize>(
    matrix: RingMatrixView<'_, F, D>,
    mode: NttCacheMode,
    tail_prefix_len: Option<usize>,
) -> Result<PreparedNttCache<D>, AkitaError> {
    validate_cache_mode(mode)?;
    if matches!(mode, NttCacheMode::ExactNegacyclic { width, .. } if width > matrix.as_slice().len())
    {
        return Err(AkitaError::InvalidSetup(
            "exact negacyclic NTT matrix is shorter than its row width".into(),
        ));
    }
    if tail_prefix_len.is_some_and(|len| len > matrix.as_slice().len()) {
        return Err(AkitaError::InvalidSetup(
            "i16-tail NTT prefix exceeds the prepared base prefix".into(),
        ));
    }
    macro_rules! prepare {
        ($params:expr, $variant:ident, $k:expr) => {{
            let params = $params;
            match mode {
                NttCacheMode::BothTransforms => {
                    let (neg, cyc) = convert_flat_pair(matrix, &params);
                    PreparedNttCache::$variant {
                        neg,
                        cyc: Some(cyc),
                        params,
                        tail: None,
                        exact: false,
                    }
                }
                NttCacheMode::ExactNegacyclic { width, log_basis } => {
                    let rhs_abs_bound = 1u64 << (log_basis - 1);
                    let needs_tail =
                        required_profile_for_params::<F, _, $k, D>(&params, width, rhs_abs_bound)?;
                    let neg = cfg_iter!(matrix.as_slice())
                        .map(|ring| CyclotomicCrtNtt::from_ring_with_params(ring, &params))
                        .collect();
                    let requested_tail_len = if needs_tail {
                        Some(tail_prefix_len.unwrap_or(matrix.as_slice().len()))
                    } else {
                        tail_prefix_len.filter(|&len| len > 0)
                    };
                    let tail = if let Some(tail_len) = requested_tail_len {
                        if tail_len == 0 {
                            return Err(AkitaError::InvalidSetup(
                                "required i16-tail NTT prefix is empty".into(),
                            ));
                        }
                        let tail_params = CrtNttParamSet::<i16, 1, D>::new([I16_TAIL_PRIME]);
                        let tail_rings = matrix.as_slice().get(..tail_len).ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "i16-tail NTT prefix exceeds the base matrix".into(),
                            )
                        })?;
                        let negacyclic = cfg_iter!(tail_rings)
                            .map(|ring| CyclotomicCrtNtt::from_ring_with_params(ring, &tail_params))
                            .collect();
                        Some(PreparedI16Tail {
                            negacyclic,
                            params: I16TailParams::new(params.clone(), tail_params),
                        })
                    } else {
                        None
                    };
                    PreparedNttCache::$variant {
                        neg,
                        cyc: None,
                        params,
                        tail,
                        exact: true,
                    }
                }
            }
        }};
    }
    let prepared = match select_crt_ntt_params::<F, D>()? {
        ProtocolCrtNttParams::Q32(params) => prepare!(params, Q32, Q32_NUM_PRIMES),
        ProtocolCrtNttParams::Q64(params) => prepare!(params, Q64, Q64_NUM_PRIMES),
        ProtocolCrtNttParams::Q128(params) => prepare!(params, Q128, Q128_NUM_PRIMES),
    };
    prepared.validate()?;
    Ok(prepared)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct VerifierNttCacheKey {
    ring_d: usize,
}

struct ErasedVerifierNttCache {
    ring_d: usize,
    base_prefix_len: usize,
    tail_prefix_len: usize,
    cache_bytes: usize,
    cache: Arc<dyn Any + Send + Sync>,
}

impl core::fmt::Debug for ErasedVerifierNttCache {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ErasedVerifierNttCache")
            .field("ring_d", &self.ring_d)
            .field("base_prefix_len", &self.base_prefix_len)
            .field("tail_prefix_len", &self.tail_prefix_len)
            .field("cache_bytes", &self.cache_bytes)
            .finish_non_exhaustive()
    }
}

/// Derived verifier cache. It is deliberately excluded from setup serialization and equality.
#[derive(Default)]
pub(crate) struct VerifierNttCache {
    slots: Mutex<HashMap<VerifierNttCacheKey, Arc<ErasedVerifierNttCache>>>,
}

impl core::fmt::Debug for VerifierNttCache {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.slots.lock() {
            Ok(slots) => formatter
                .debug_struct("VerifierNttCache")
                .field("keys", &slots.keys().collect::<Vec<_>>())
                .field(
                    "cache_bytes",
                    &slots.values().map(|slot| slot.cache_bytes).sum::<usize>(),
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
        Ok(slots.values().map(|slot| slot.cache_bytes).sum())
    }

    /// Build, erase, and atomically install an entry when needed.
    pub(crate) fn prepare<F: FieldCore + CanonicalField, const D: usize>(
        &self,
        expanded: &AkitaExpandedSetup<F>,
        matrix: NttCacheKey,
        tail_prefix_len: usize,
        mode: NttCacheMode,
    ) -> Result<Arc<PreparedNttCache<D>>, AkitaError> {
        let NttCacheMode::ExactNegacyclic { width, log_basis } = mode else {
            return Err(AkitaError::InvalidSetup(
                "verifier NTT cache requires exact negacyclic mode".into(),
            ));
        };
        if matrix.ring_d != D {
            return Err(AkitaError::InvalidSetup(format!(
                "verifier NTT cache ring_d mismatch: key {}, requested {D}",
                matrix.ring_d
            )));
        }
        let with_i16_tail = ntt_cache_requires_i16_tail::<F, D>(width, log_basis)?;
        if with_i16_tail != (tail_prefix_len > 0) {
            return Err(AkitaError::InvalidSetup(
                "verifier tail prefix disagrees with exactness requirement".into(),
            ));
        }
        if tail_prefix_len > matrix.num_ring_elements {
            return Err(AkitaError::InvalidSetup(
                "verifier tail prefix exceeds its base prefix".into(),
            ));
        }
        if width > matrix.num_ring_elements {
            return Err(AkitaError::InvalidSetup(
                "verifier NTT matrix prefix is shorter than its row width".into(),
            ));
        }
        let key = VerifierNttCacheKey { ring_d: D };
        let mut slots = self
            .slots
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("verifier NTT cache lock poisoned".into()))?;
        if let Some(slot) = slots.get(&key) {
            if slot.base_prefix_len >= matrix.num_ring_elements
                && slot.tail_prefix_len >= tail_prefix_len
            {
                return downcast_verifier_cache::<D>(Arc::clone(slot));
            }
        }
        let base_prefix_len = slots.get(&key).map_or(matrix.num_ring_elements, |slot| {
            slot.base_prefix_len.max(matrix.num_ring_elements)
        });
        let tail_prefix_len = slots.get(&key).map_or(tail_prefix_len, |slot| {
            slot.tail_prefix_len.max(tail_prefix_len)
        });
        let view = expanded
            .shared_matrix()
            .ring_view::<D>(1, base_prefix_len)?;
        let prepared = Arc::new(prepare_ntt_cache_with_tail_prefix(
            view,
            mode,
            Some(tail_prefix_len),
        )?);
        if prepared.has_i16_tail() != (tail_prefix_len > 0) {
            return Err(AkitaError::InvalidSetup(
                "prepared verifier NTT layout disagrees with exactness selection".into(),
            ));
        }
        let built = Arc::new(ErasedVerifierNttCache {
            ring_d: D,
            base_prefix_len,
            tail_prefix_len,
            cache_bytes: prepared.cache_bytes(),
            cache: prepared,
        });
        slots.insert(key, Arc::clone(&built));
        downcast_verifier_cache::<D>(built)
    }
}

fn downcast_verifier_cache<const D: usize>(
    erased: Arc<ErasedVerifierNttCache>,
) -> Result<Arc<PreparedNttCache<D>>, AkitaError> {
    if erased.ring_d != D {
        return Err(AkitaError::InvalidSetup(format!(
            "prepared verifier NTT ring_d mismatch: stored {}, requested {D}",
            erased.ring_d
        )));
    }
    Arc::clone(&erased.cache)
        .downcast::<PreparedNttCache<D>>()
        .map_err(|_| AkitaError::InvalidSetup("prepared verifier NTT type mismatch".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::CyclotomicRing;
    use akita_field::{Prime128Offset275, Prime32Offset99, Prime64Offset59};
    use core::mem::size_of;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn flat_zeros<F: FieldCore, const D: usize>(len: usize) -> crate::FlatMatrix<F> {
        crate::FlatMatrix::from_ring_slice(&vec![CyclotomicRing::<F, D>::zero(); len])
    }

    #[test]
    fn prepare_materializes_exactly_the_requested_layout() {
        const D: usize = 64;
        let flat = flat_zeros::<Prime32Offset99, D>(10);
        let view = flat.ring_view::<D>(1, 10).expect("matrix view");
        let both = prepare_ntt_cache(view, NttCacheMode::BothTransforms).expect("both transforms");
        assert!(both.has_cyclic());
        assert!(!both.has_i16_tail());

        let view = flat.ring_view::<D>(1, 10).expect("matrix view");
        let base = prepare_ntt_cache(
            view,
            NttCacheMode::ExactNegacyclic {
                width: 5,
                log_basis: 16,
            },
        )
        .expect("base negacyclic");
        assert!(!base.has_cyclic());
        assert!(!base.has_i16_tail());

        let flat = flat_zeros::<Prime128Offset275, D>(10);
        let view = flat.ring_view::<D>(1, 10).expect("matrix view");
        let tail = prepare_ntt_cache(
            view,
            NttCacheMode::ExactNegacyclic {
                width: 5,
                log_basis: 16,
            },
        )
        .expect("tail negacyclic");
        assert!(!tail.has_cyclic());
        assert!(tail.has_i16_tail());
        assert_eq!(
            tail.cache_bytes(),
            10 * D * (Q128_NUM_PRIMES * size_of::<i32>() + size_of::<i16>())
        );
    }

    #[test]
    fn exact_mode_rejects_invalid_balanced_bounds() {
        const D: usize = 64;
        let flat = flat_zeros::<Prime64Offset59, D>(1);
        for mode in [
            NttCacheMode::ExactNegacyclic {
                width: 0,
                log_basis: 10,
            },
            NttCacheMode::ExactNegacyclic {
                width: 1,
                log_basis: 0,
            },
            NttCacheMode::ExactNegacyclic {
                width: 1,
                log_basis: 17,
            },
        ] {
            let view = flat.ring_view::<D>(1, 1).expect("matrix view");
            assert!(matches!(
                prepare_ntt_cache(view, mode),
                Err(AkitaError::InvalidSetup(_))
            ));
        }
    }

    #[test]
    fn exact_selector_changes_layout_at_the_strict_capacity_boundary() {
        const D: usize = 64;
        let ProtocolCrtNttParams::Q128(params) =
            select_crt_ntt_params::<Prime128Offset275, D>().expect("Q128 params")
        else {
            panic!("Q128 field must select Q128 params");
        };
        let safe = max_safe_crt_accumulation_width::<Prime128Offset275, i32, Q128_NUM_PRIMES, D>(
            &params,
            1 << 15,
        )
        .expect("one term fits");
        assert!(!ntt_cache_requires_i16_tail::<Prime128Offset275, D>(safe, 16).unwrap());
        assert!(ntt_cache_requires_i16_tail::<Prime128Offset275, D>(safe + 1, 16).unwrap());
    }

    #[test]
    fn signed_i16_cache_checks_shape_and_digit_class() {
        const D: usize = 64;
        let flat = flat_zeros::<Prime32Offset99, D>(2);
        let cache = prepare_ntt_cache(
            flat.ring_view::<D>(1, 2).expect("matrix view"),
            NttCacheMode::ExactNegacyclic {
                width: 2,
                log_basis: 10,
            },
        )
        .expect("cache");
        assert!(cache
            .mat_vec_i16::<Prime32Offset99>(10, 1, &[[511; D], [-512; D]])
            .is_ok());
        assert!(matches!(
            cache.mat_vec_i16::<Prime32Offset99>(10, 1, &[[512; D], [0; D]]),
            Err(AkitaError::InvalidProof)
        ));
        assert!(cache
            .mat_vec_i16::<Prime32Offset99>(10, 1, &[[0; D]])
            .is_ok());

        let short = prepare_ntt_cache(
            flat.ring_view::<D>(1, 1).expect("matrix view"),
            NttCacheMode::ExactNegacyclic {
                width: 1,
                log_basis: 10,
            },
        )
        .expect("short cache");
        assert!(matches!(
            short.mat_vec_i16::<Prime32Offset99>(10, 1, &[[0; D], [0; D]]),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn erased_cache_mismatches_return_errors_without_panicking() {
        const D: usize = 64;
        let flat = flat_zeros::<Prime32Offset99, D>(1);
        let cache = Arc::new(
            prepare_ntt_cache(
                flat.ring_view::<D>(1, 1).expect("matrix view"),
                NttCacheMode::ExactNegacyclic {
                    width: 1,
                    log_basis: 8,
                },
            )
            .expect("cache"),
        );
        let bytes = cache.cache_bytes();
        let wrong_degree = Arc::new(ErasedVerifierNttCache {
            ring_d: D,
            base_prefix_len: 1,
            tail_prefix_len: 0,
            cache_bytes: bytes,
            cache: Arc::clone(&cache) as Arc<dyn Any + Send + Sync>,
        });
        let result = catch_unwind(AssertUnwindSafe(|| {
            downcast_verifier_cache::<32>(wrong_degree)
        }));
        assert!(matches!(result, Ok(Err(AkitaError::InvalidSetup(_)))));

        let wrong_type = Arc::new(ErasedVerifierNttCache {
            ring_d: D,
            base_prefix_len: 1,
            tail_prefix_len: 0,
            cache_bytes: 0,
            cache: Arc::new(17usize),
        });
        let result = catch_unwind(AssertUnwindSafe(|| {
            downcast_verifier_cache::<D>(wrong_type)
        }));
        assert!(matches!(result, Ok(Err(AkitaError::InvalidSetup(_)))));
    }
}
