//! Runtime ring-dimension NTT prepared-setup caches.

use akita_algebra::ntt::ifma52::{ifma52_enabled, IFMA52_PRIMES};
use akita_algebra::ntt::prime::PrimeWidth;
use akita_algebra::ntt::tables::{
    q128_primes, validate_profile_crt_ring_degree, I16_TAIL_PRIME, Q128_MAX_RING_D, Q128_MODULUS,
    Q128_NUM_PRIMES, Q32_MAX_RING_D, Q32_MODULUS, Q32_NUM_PRIMES, Q32_PRIMES, Q64_MAX_RING_D,
    Q64_MODULUS, Q64_NUM_PRIMES, Q64_PRIMES,
};
use akita_algebra::{
    CrtCapacity, CrtNttParamSet, CyclotomicCrtNtt, I16TailParams, Ifma52NttMatrix, Ifma52Params,
    MontCoeff,
};
use akita_error::AkitaError;
#[allow(unused_imports)]
use jolt_field::solinas::parallel::*;
use jolt_field::{cfg_iter, CanonicalEncoding, Field, Prime128OffsetA7F7, PseudoMersenne};
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::dispatch::compression_ring_dim_supported_for_tier;
use crate::{
    balanced_signed_digit_abs_bound, field_modulus, ntt_max_ring_d, ntt_min_ring_d,
    ntt_ring_degree_supported_for_field, protocol_dispatch_tier, AkitaExpandedSetup,
    ProtocolRingDispatchTierId, RingMatrixView, SisModulusProfileId,
};

mod exact;
mod prepared_artifact;

#[cfg(test)]
use exact::ifma52_cache_enabled;
use exact::{exact_cache_plan, ifma52_cache_enabled_for_ring_dimension, prepare_exact_ntt_cache};
pub use exact::{ntt_cache_requires_exactness_tail, planned_exact_ntt_cache_bytes};
pub(crate) use prepared_artifact::decode_riscv64_scalar_q128_cache;
pub use prepared_artifact::{
    build_riscv64_scalar_q128_cache_artifact, prepared_verifier_ntt_cache_metadata,
    PreparedVerifierNttCacheBinding, PreparedVerifierNttCacheMetadata,
    PREPARED_VERIFIER_NTT_CACHE_MAX_BYTES,
};

/// Transform representation stored by one exact-prefix NTT cache entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NttTransformDomain {
    /// Base-profile negacyclic transforms only.
    Negacyclic,
    /// Base-profile cyclic transforms only.
    Cyclic,
    /// Exactness-only 14-bit tail in both negacyclic and cyclic domains.
    ///
    /// Ring-switch quotient kernels combine this cache with the ordinary CRT
    /// prefix when the field-selected representation cannot fit one centered
    /// product term.
    I16TailBothTransforms,
    /// Exact negacyclic transforms for a signed-i16 matrix product.
    ///
    /// Both values participate in the cache identity because exact CRT sizing
    /// depends on the active matrix row width and coefficient bound.
    ExactNegacyclicI16 { width: usize, rhs_abs_bound: u64 },
}

/// Exact public-matrix prefix required at one ring dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NttPrefixRequirement {
    /// Ring dimension used to interpret the flat public field stream.
    pub ring_dimension: usize,
    /// Number of ring elements in the required prefix.
    pub num_ring_elements: usize,
}

impl NttPrefixRequirement {
    /// Derive an exact prefix from the matrix rows and active row width passed
    /// to a consuming kernel.
    pub fn from_matrix_shape(
        ring_dimension: usize,
        num_rows: usize,
        active_width: usize,
    ) -> Result<Self, AkitaError> {
        if ring_dimension == 0 {
            return Err(AkitaError::InvalidSetup(
                "NTT prefix ring dimension must be nonzero".into(),
            ));
        }
        let num_ring_elements = num_rows
            .checked_mul(active_width)
            .filter(|count| *count > 0)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("NTT prefix matrix shape overflows or is empty".into())
            })?;
        Ok(Self {
            ring_dimension,
            num_ring_elements,
        })
    }

    /// Join overlapping public-matrix prefixes by maximum length.
    pub fn join(self, other: Self) -> Result<Self, AkitaError> {
        if self.ring_dimension != other.ring_dimension {
            return Err(AkitaError::InvalidSetup(
                "cannot join NTT prefixes at different ring dimensions".into(),
            ));
        }
        Ok(Self {
            ring_dimension: self.ring_dimension,
            num_ring_elements: self.num_ring_elements.max(other.num_ring_elements),
        })
    }

    /// Exact number of public field elements covered by this prefix.
    pub fn num_field_elements(self) -> Result<usize, AkitaError> {
        self.num_ring_elements
            .checked_mul(self.ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("NTT prefix field count overflow".into()))
    }
}

/// Identifies one prepared NTT prefix at a concrete ring degree and domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NttCacheKey {
    /// Ring dimension `D` for the cached transform family.
    pub ring_d: usize,
    /// Number of ring elements in the cached matrix view at `ring_d`.
    pub num_ring_elements: usize,
    /// Transform domain materialized for this prefix.
    pub domain: NttTransformDomain,
}

impl NttCacheKey {
    /// Attach a transform domain to an exact prefix requirement.
    #[must_use]
    pub const fn new(requirement: NttPrefixRequirement, domain: NttTransformDomain) -> Self {
        Self {
            ring_d: requirement.ring_dimension,
            num_ring_elements: requirement.num_ring_elements,
            domain,
        }
    }

    /// Derive a cache key from the exact matrix shape passed to a kernel.
    pub fn from_matrix_shape(
        ring_dimension: usize,
        num_rows: usize,
        active_width: usize,
        domain: NttTransformDomain,
    ) -> Result<Self, AkitaError> {
        Ok(Self::new(
            NttPrefixRequirement::from_matrix_shape(ring_dimension, num_rows, active_width)?,
            domain,
        ))
    }

    /// Join covering prefixes that share a ring dimension and transform domain.
    pub fn join(self, other: Self) -> Result<Self, AkitaError> {
        if self.ring_d != other.ring_d || self.domain != other.domain {
            return Err(AkitaError::InvalidSetup(
                "cannot join NTT cache keys with different dimensions or domains".into(),
            ));
        }
        Ok(Self {
            ring_d: self.ring_d,
            num_ring_elements: self.num_ring_elements.max(other.num_ring_elements),
            domain: self.domain,
        })
    }

    /// Exact number of public field elements covered by this cache key.
    pub fn num_field_elements(self) -> Result<usize, AkitaError> {
        self.num_ring_elements
            .checked_mul(self.ring_d)
            .ok_or_else(|| AkitaError::InvalidSetup("NTT cache field count overflow".into()))
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
///
/// This is the ordinary protocol selector. Compression-only ring degrees must use
/// [`select_compression_crt_ntt_params`] instead of widening this gate.
pub fn select_crt_ntt_params<F: Field + CanonicalEncoding, const D: usize>(
) -> Result<ProtocolCrtNttParams<D>, AkitaError> {
    let tier = protocol_dispatch_tier::<F>();
    if !ntt_ring_degree_supported_for_field::<F>(D) {
        return Err(AkitaError::InvalidSetup(format!(
            "CRT+NTT ring degree {D} outside tier band [{}, {}] for this field",
            ntt_min_ring_d(tier),
            ntt_max_ring_d(tier),
        )));
    }
    select_crt_ntt_params_for_modulus::<F, D>()
}

/// Select CRT+NTT params for the compressed-commitment diagnostic ladder.
///
/// Ordinary protocol callers must keep using [`select_crt_ntt_params`].
pub fn select_compression_crt_ntt_params<F: Field + CanonicalEncoding, const D: usize>(
) -> Result<ProtocolCrtNttParams<D>, AkitaError> {
    let tier = protocol_dispatch_tier::<F>();
    if !compression_ring_dim_supported_for_tier(tier, D) {
        return Err(AkitaError::InvalidSetup(format!(
            "compression CRT+NTT ring degree {D} is outside the compression surface for this field"
        )));
    }
    select_crt_ntt_params_for_modulus::<F, D>()
}

fn select_crt_ntt_params_for_modulus<F: Field + CanonicalEncoding, const D: usize>(
) -> Result<ProtocolCrtNttParams<D>, AkitaError> {
    let modulus = field_modulus::<F>()?;
    let a7f7_q128_modulus = u128::MAX - (<Prime128OffsetA7F7 as PseudoMersenne>::OFFSET - 1);

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
    if modulus == Q128_MODULUS || modulus == a7f7_q128_modulus {
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

fn required_profile_for_params<F, W, const K: usize, const D: usize>(
    params: &CrtNttParamSet<W, K, D>,
    width: usize,
    rhs_abs_bound: u64,
) -> Result<bool, AkitaError>
where
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
{
    let capacity = params.crt_capacity();
    if capacity.supports::<F, D>(width, rhs_abs_bound) {
        return Ok(false);
    }
    if capacity
        .with_prime_modulus(I16_TAIL_PRIME.p as u128)
        .supports::<F, D>(width, rhs_abs_bound)
    {
        return Ok(true);
    }
    Err(AkitaError::InvalidSetup(format!(
        "CRT accumulation exceeds base plus i16-tail capacity for D={D}, width={width}, rhs_abs_bound={rhs_abs_bound}"
    )))
}

/// Whether a centered ring-switch product term needs the 14-bit exactness
/// tail in addition to the protocol CRT prefix.
pub fn centered_quotient_requires_i16_tail(
    profile: SisModulusProfileId,
    ring_dimension: usize,
    rhs_abs_bound: u64,
) -> Result<bool, AkitaError> {
    let capacity = match profile {
        SisModulusProfileId::Q32Offset99 => {
            CrtCapacity::from_prime_moduli(Q32_PRIMES.map(|prime| prime.p as u128))
        }
        SisModulusProfileId::Q64Offset59 => {
            CrtCapacity::from_prime_moduli(Q64_PRIMES.map(|prime| prime.p as u128))
        }
        SisModulusProfileId::Q128OffsetA7F7 => {
            CrtCapacity::from_prime_moduli(q128_primes().map(|prime| prime.p as u128))
        }
    };
    if capacity.supports_modulus(1, ring_dimension, profile.modulus(), rhs_abs_bound) {
        return Ok(false);
    }
    if capacity
        .with_prime_modulus(I16_TAIL_PRIME.p as u128)
        .supports_modulus(1, ring_dimension, profile.modulus(), rhs_abs_bound)
    {
        return Ok(true);
    }
    Err(AkitaError::InvalidSetup(format!(
        "centered quotient term exceeds base plus i16-tail capacity for D={ring_dimension}, rhs_abs_bound={rhs_abs_bound}"
    )))
}

fn dense_i8_exact_ifma52_is_profitable(
    field_modulus: u128,
    ring_dimension: usize,
    width: usize,
    rhs_abs_bound: u64,
    ifma52_cache_enabled: bool,
) -> bool {
    if field_modulus <= Q64_MODULUS as u128 || !ifma52_cache_enabled {
        return false;
    }
    let capacity = CrtCapacity::from_prime_moduli(IFMA52_PRIMES.map(u128::from));
    !capacity.supports_modulus(width, ring_dimension, field_modulus, rhs_abs_bound)
        && capacity
            .with_prime_modulus(q128_primes()[0].p as u128)
            .supports_modulus(width, ring_dimension, field_modulus, rhs_abs_bound)
}

/// Whether a dense signed-i8 commitment should use one exact AVX-512 IFMA52
/// accumulation instead of bounded portable CRT chunks.
///
/// This selects only q128 rows that need the 30-bit tail for a complete IFMA52
/// accumulation. AVX2, NEON, scalar execution, and rows that fit the three
/// base IFMA52 limbs retain the chunked i8 kernel.
pub fn dense_i8_commit_prefers_exact_ifma52(
    field_modulus: u128,
    ring_dimension: usize,
    width: usize,
    rhs_abs_bound: u64,
) -> bool {
    dense_i8_exact_ifma52_is_profitable(
        field_modulus,
        ring_dimension,
        width,
        rhs_abs_bound,
        ifma52_cache_enabled_for_ring_dimension(ring_dimension),
    )
}

/// Field-typed form of [`centered_quotient_requires_i16_tail`] used by the
/// runtime kernel dispatch.
pub fn centered_quotient_requires_i16_tail_for_field<
    F: Field + CanonicalEncoding,
    const D: usize,
>(
    rhs_abs_bound: u64,
) -> Result<bool, AkitaError> {
    match select_crt_ntt_params::<F, D>()? {
        ProtocolCrtNttParams::Q32(params) => {
            required_profile_for_params::<F, _, Q32_NUM_PRIMES, D>(&params, 1, rhs_abs_bound)
        }
        ProtocolCrtNttParams::Q64(params) => {
            required_profile_for_params::<F, _, Q64_NUM_PRIMES, D>(&params, 1, rhs_abs_bound)
        }
        ProtocolCrtNttParams::Q128(params) => {
            required_profile_for_params::<F, _, Q128_NUM_PRIMES, D>(&params, 1, rhs_abs_bound)
        }
    }
}

/// NTT representations requested by protocol and backend consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NttCacheMode {
    /// Materialize only the base-profile negacyclic transform.
    Negacyclic,
    /// Materialize only the base-profile cyclic transform.
    Cyclic,
    /// Materialize the base-profile negacyclic and cyclic transforms.
    BothTransforms,
    /// Materialize only the exactness tail in both transform domains.
    I16TailBothTransforms,
    /// Materialize the minimum exact negacyclic representation for signed
    /// coefficients whose absolute value is at most `rhs_abs_bound`.
    ExactNegacyclic { width: usize, rhs_abs_bound: u64 },
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

/// Read-only view of an exactness-only i16 tail pair.
#[derive(Clone, Copy)]
pub struct PreparedNttTailPairView<'a, const D: usize> {
    negacyclic: &'a [CyclotomicCrtNtt<i16, 1, D>],
    cyclic: &'a [CyclotomicCrtNtt<i16, 1, D>],
    params: &'a CrtNttParamSet<i16, 1, D>,
}

impl<'a, const D: usize> PreparedNttTailPairView<'a, D> {
    /// Borrow the negacyclic tail transforms.
    #[must_use]
    pub const fn negacyclic(self) -> &'a [CyclotomicCrtNtt<i16, 1, D>] {
        self.negacyclic
    }

    /// Borrow the cyclic tail transforms.
    #[must_use]
    pub const fn cyclic(self) -> &'a [CyclotomicCrtNtt<i16, 1, D>] {
        self.cyclic
    }

    /// Borrow the 14-bit tail parameters.
    #[must_use]
    pub const fn params(self) -> &'a CrtNttParamSet<i16, 1, D> {
        self.params
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct PreparedIfma52Tail<W: PrimeWidth, const D: usize> {
    negacyclic: Vec<CyclotomicCrtNtt<W, 1, D>>,
    params: CrtNttParamSet<W, 1, D>,
}

/// One prepared NTT cache over the field-selected CRT profile.
///
/// Its representation is opaque so matrices, parameters, exactness metadata,
/// and optional tails cannot be mutated independently across crate boundaries.
#[derive(Debug)]
pub struct PreparedNttCache<const D: usize>(PreparedNttCacheRepr<D>);

/// Read-only typed view of one prepared i32 base profile.
#[derive(Clone, Copy)]
pub struct PreparedNttBaseView<'a, W: PrimeWidth, const K: usize, const D: usize> {
    negacyclic: Option<&'a [CyclotomicCrtNtt<W, K, D>]>,
    cyclic: Option<&'a [CyclotomicCrtNtt<W, K, D>]>,
    params: &'a CrtNttParamSet<W, K, D>,
}

impl<'a, W: PrimeWidth, const K: usize, const D: usize> PreparedNttBaseView<'a, W, K, D> {
    /// Borrow the negacyclic domain when it was materialized.
    #[must_use]
    pub const fn negacyclic(self) -> Option<&'a [CyclotomicCrtNtt<W, K, D>]> {
        self.negacyclic
    }

    /// Borrow the cyclic domain when it was materialized.
    #[must_use]
    pub const fn cyclic(self) -> Option<&'a [CyclotomicCrtNtt<W, K, D>]> {
        self.cyclic
    }

    /// Borrow the parameters bound to both prepared domains.
    #[must_use]
    pub const fn params(self) -> &'a CrtNttParamSet<W, K, D> {
        self.params
    }
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum PreparedNttCacheRepr<const D: usize> {
    I16TailPair {
        neg: Vec<CyclotomicCrtNtt<i16, 1, D>>,
        cyc: Vec<CyclotomicCrtNtt<i16, 1, D>>,
        params: CrtNttParamSet<i16, 1, D>,
    },
    #[non_exhaustive]
    Q32 {
        neg: Option<Vec<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>>,
        cyc: Option<Vec<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i32, Q32_NUM_PRIMES, D>,
        tail: Option<PreparedI16Tail<Q32_NUM_PRIMES, D>>,
        exact: bool,
    },
    #[non_exhaustive]
    Q32Ifma52 {
        neg: Ifma52NttMatrix<1, D>,
        tail: Option<PreparedIfma52Tail<i16, D>>,
    },
    #[non_exhaustive]
    Q64 {
        neg: Option<Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>>,
        cyc: Option<Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i32, Q64_NUM_PRIMES, D>,
        tail: Option<PreparedI16Tail<Q64_NUM_PRIMES, D>>,
        exact: bool,
    },
    #[non_exhaustive]
    Q64Ifma52 { neg: Ifma52NttMatrix<2, D> },
    #[non_exhaustive]
    Q128 {
        neg: Option<Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>>,
        cyc: Option<Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i32, Q128_NUM_PRIMES, D>,
        tail: Option<PreparedI16Tail<Q128_NUM_PRIMES, D>>,
        exact: bool,
    },
    #[non_exhaustive]
    Q128Ifma52 {
        neg: Ifma52NttMatrix<3, D>,
        tail: Option<PreparedIfma52Tail<i32, D>>,
    },
}

impl<const D: usize> PreparedNttCacheRepr<D> {
    fn validate(&self) -> Result<(), AkitaError> {
        macro_rules! validate {
            ($neg:expr, $cyc:expr, $params:expr, $tail:expr, $exact:expr) => {{
                if $neg.is_none() && $cyc.is_none() {
                    return Err(AkitaError::InvalidSetup(
                        "prepared NTT cache has no transform domain".into(),
                    ));
                }
                if $exact && ($neg.is_none() || $cyc.is_some()) {
                    return Err(AkitaError::InvalidSetup(
                        "prepared NTT cache has an unsupported domain combination".into(),
                    ));
                }
                if $tail.as_ref().is_some_and(|tail| {
                    tail.negacyclic.is_empty()
                        || $neg
                            .as_ref()
                            .is_none_or(|neg| tail.negacyclic.len() > neg.len())
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
            Self::I16TailPair { neg, cyc, params } => {
                if neg.is_empty() || neg.len() != cyc.len() || params.primes != [I16_TAIL_PRIME] {
                    return Err(AkitaError::InvalidSetup(
                        "prepared i16-tail transform pair is inconsistent".into(),
                    ));
                }
            }
            Self::Q32 {
                neg,
                cyc,
                params,
                tail,
                exact,
            } => validate!(neg, cyc, params, tail, *exact),
            Self::Q32Ifma52 { neg, tail } => {
                if neg.is_empty()
                    || neg.has_i16_tail() != tail.is_some()
                    || tail.as_ref().is_some_and(|tail| {
                        tail.negacyclic.is_empty()
                            || tail.negacyclic.len() > neg.len()
                            || tail.params.primes != [I16_TAIL_PRIME]
                    })
                {
                    return Err(AkitaError::InvalidSetup(
                        "prepared mixed IFMA52 NTT cache is inconsistent".into(),
                    ));
                }
            }
            Self::Q64 {
                neg,
                cyc,
                params,
                tail,
                exact,
            } => validate!(neg, cyc, params, tail, *exact),
            Self::Q64Ifma52 { neg, .. } => {
                if neg.is_empty() {
                    return Err(AkitaError::InvalidSetup(
                        "prepared IFMA52 NTT cache is empty".into(),
                    ));
                }
            }
            Self::Q128 {
                neg,
                cyc,
                params,
                tail,
                exact,
            } => validate!(neg, cyc, params, tail, *exact),
            Self::Q128Ifma52 { neg, tail } => {
                if neg.is_empty()
                    || neg.has_i32_tail() != tail.is_some()
                    || tail.as_ref().is_some_and(|tail| {
                        tail.negacyclic.is_empty()
                            || tail.negacyclic.len() > neg.len()
                            || tail.params.primes != [q128_primes()[0]]
                    })
                {
                    return Err(AkitaError::InvalidSetup(
                        "prepared mixed IFMA52 NTT cache is inconsistent".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// In-memory byte footprint of all materialized matrix transforms.
    #[must_use]
    fn cache_bytes(&self) -> usize {
        macro_rules! bytes {
            ($neg:expr, $cyc:expr, $tail:expr, $k:expr) => {{
                let base_entries =
                    $neg.as_ref().map_or(0, Vec::len) + $cyc.as_ref().map_or(0, Vec::len);
                let base = base_entries * D * $k * core::mem::size_of::<i32>();
                let tail = $tail.as_ref().map_or(0, |tail| {
                    tail.negacyclic.len() * D * core::mem::size_of::<i16>()
                });
                base + tail
            }};
        }
        match self {
            Self::I16TailPair { neg, cyc, .. } => {
                (neg.len() + cyc.len()) * D * core::mem::size_of::<i16>()
            }
            Self::Q32 { neg, cyc, tail, .. } => bytes!(neg, cyc, tail, Q32_NUM_PRIMES),
            Self::Q32Ifma52 { neg, tail, .. } => {
                neg.cache_bytes()
                    + tail.as_ref().map_or(0, |tail| {
                        tail.negacyclic.len() * D * core::mem::size_of::<i16>()
                    })
            }
            Self::Q64 { neg, cyc, tail, .. } => bytes!(neg, cyc, tail, Q64_NUM_PRIMES),
            Self::Q64Ifma52 { neg, .. } => neg.cache_bytes(),
            Self::Q128 { neg, cyc, tail, .. } => bytes!(neg, cyc, tail, Q128_NUM_PRIMES),
            Self::Q128Ifma52 { neg, tail, .. } => {
                neg.cache_bytes()
                    + tail.as_ref().map_or(0, |tail| {
                        tail.negacyclic.len() * D * core::mem::size_of::<i32>()
                    })
            }
        }
    }

    /// Whether the cyclic base representation was materialized.
    #[must_use]
    const fn has_cyclic(&self) -> bool {
        match self {
            Self::I16TailPair { .. } => true,
            Self::Q32 { cyc, .. } => cyc.is_some(),
            Self::Q32Ifma52 { .. } => false,
            Self::Q64 { cyc, .. } => cyc.is_some(),
            Self::Q64Ifma52 { .. } => false,
            Self::Q128 { cyc, .. } => cyc.is_some(),
            Self::Q128Ifma52 { .. } => false,
        }
    }

    /// Whether the base negacyclic representation was materialized.
    #[must_use]
    const fn has_negacyclic(&self) -> bool {
        match self {
            Self::I16TailPair { .. } => true,
            Self::Q32 { neg, .. } => neg.is_some(),
            Self::Q32Ifma52 { .. } => true,
            Self::Q64 { neg, .. } => neg.is_some(),
            Self::Q64Ifma52 { .. } => true,
            Self::Q128 { neg, .. } => neg.is_some(),
            Self::Q128Ifma52 { .. } => true,
        }
    }

    /// Whether an exactness tail was materialized.
    #[must_use]
    const fn has_exactness_tail(&self) -> bool {
        match self {
            Self::I16TailPair { .. } => true,
            Self::Q32 { tail, .. } => tail.is_some(),
            Self::Q32Ifma52 { tail, .. } => tail.is_some(),
            Self::Q64 { tail, .. } => tail.is_some(),
            Self::Q64Ifma52 { .. } => false,
            Self::Q128 { tail, .. } => tail.is_some(),
            Self::Q128Ifma52 { tail, .. } => tail.is_some(),
        }
    }

    /// Compute a shape-checked exact signed-i16 matrix product.
    #[inline]
    fn mat_vec_i16<F: Field + CanonicalEncoding>(
        &self,
        log_basis: u32,
        num_rows: usize,
        rhs: &[[i16; D]],
    ) -> Result<Vec<akita_algebra::CyclotomicRing<F, D>>, AkitaError> {
        self.validate()?;
        let prepared_tier = match self {
            Self::I16TailPair { .. } => {
                return Err(AkitaError::InvalidSetup(
                    "signed-i16 matvec requested from a tail-only cache".into(),
                ));
            }
            Self::Q32 { .. } | Self::Q32Ifma52 { .. } => ProtocolRingDispatchTierId::Fp32,
            Self::Q64 { .. } | Self::Q64Ifma52 { .. } => ProtocolRingDispatchTierId::Fp64,
            Self::Q128 { .. } | Self::Q128Ifma52 { .. } => ProtocolRingDispatchTierId::Fp128,
        };
        if protocol_dispatch_tier::<F>() != prepared_tier {
            return Err(AkitaError::InvalidSetup(
                "prepared NTT field profile mismatch".into(),
            ));
        }
        match self {
            Self::I16TailPair { .. } => Err(AkitaError::InvalidSetup(
                "signed-i16 matvec requested from a tail-only cache".into(),
            )),
            Self::Q32 {
                neg,
                params,
                tail,
                exact,
                ..
            } => {
                let neg = neg.as_deref().ok_or_else(|| {
                    AkitaError::InvalidSetup("negacyclic NTT domain not prepared".into())
                })?;
                mat_vec_i16_from_cache(neg, params, tail.as_ref(), *exact, log_basis, num_rows, rhs)
            }
            Self::Q32Ifma52 { neg, tail } => {
                let rhs_abs_bound = validate_i16_rhs(log_basis, rhs)?;
                if !neg
                    .crt_capacity()
                    .supports::<F, D>(rhs.len(), rhs_abs_bound)
                {
                    return Err(AkitaError::InvalidSetup(
                        "signed-i16 matvec exceeds prepared IFMA52 capacity".into(),
                    ));
                }
                if let Some(tail) = tail {
                    neg.mat_vec_i16_with_tail(&tail.negacyclic, num_rows, rhs, &tail.params)
                } else {
                    neg.mat_vec_i16(num_rows, rhs)
                }
            }
            Self::Q64 {
                neg,
                params,
                tail,
                exact,
                ..
            } => {
                let neg = neg.as_deref().ok_or_else(|| {
                    AkitaError::InvalidSetup("negacyclic NTT domain not prepared".into())
                })?;
                mat_vec_i16_from_cache(neg, params, tail.as_ref(), *exact, log_basis, num_rows, rhs)
            }
            Self::Q64Ifma52 { neg } => {
                let rhs_abs_bound = validate_i16_rhs(log_basis, rhs)?;
                if !neg
                    .crt_capacity()
                    .supports::<F, D>(rhs.len(), rhs_abs_bound)
                {
                    return Err(AkitaError::InvalidSetup(
                        "signed-i16 matvec exceeds prepared IFMA52 capacity".into(),
                    ));
                }
                neg.mat_vec_i16(num_rows, rhs)
            }
            Self::Q128 {
                neg,
                params,
                tail,
                exact,
                ..
            } => {
                let neg = neg.as_deref().ok_or_else(|| {
                    AkitaError::InvalidSetup("negacyclic NTT domain not prepared".into())
                })?;
                mat_vec_i16_from_cache(neg, params, tail.as_ref(), *exact, log_basis, num_rows, rhs)
            }
            Self::Q128Ifma52 { neg, tail } => {
                let rhs_abs_bound = validate_i16_rhs(log_basis, rhs)?;
                if !neg
                    .crt_capacity()
                    .supports::<F, D>(rhs.len(), rhs_abs_bound)
                {
                    return Err(AkitaError::InvalidSetup(
                        "signed-i16 matvec exceeds prepared IFMA52 capacity".into(),
                    ));
                }
                if let Some(tail) = tail {
                    neg.mat_vec_i16_with_tail(&tail.negacyclic, num_rows, rhs, &tail.params)
                } else {
                    neg.mat_vec_i16(num_rows, rhs)
                }
            }
        }
    }
}

impl<const D: usize> PreparedNttCache<D> {
    /// In-memory byte footprint of all materialized matrix transforms.
    #[must_use]
    pub fn cache_bytes(&self) -> usize {
        self.0.cache_bytes()
    }

    /// Whether the cyclic base representation was materialized.
    #[must_use]
    pub const fn has_cyclic(&self) -> bool {
        self.0.has_cyclic()
    }

    /// Whether the base negacyclic representation was materialized.
    #[must_use]
    pub const fn has_negacyclic(&self) -> bool {
        self.0.has_negacyclic()
    }

    /// Whether an exactness tail was materialized.
    #[must_use]
    pub const fn has_exactness_tail(&self) -> bool {
        self.0.has_exactness_tail()
    }

    /// Whether this exact cache uses the AVX-512IFMA residue representation.
    #[must_use]
    pub const fn uses_ifma52(&self) -> bool {
        matches!(
            self.0,
            PreparedNttCacheRepr::Q32Ifma52 { .. }
                | PreparedNttCacheRepr::Q64Ifma52 { .. }
                | PreparedNttCacheRepr::Q128Ifma52 { .. }
        )
    }

    /// Borrow the Q32 i32 base domains and their bound parameters.
    #[must_use]
    pub fn q32_base(&self) -> Option<PreparedNttBaseView<'_, i32, Q32_NUM_PRIMES, D>> {
        match &self.0 {
            PreparedNttCacheRepr::Q32 {
                neg, cyc, params, ..
            } => Some(PreparedNttBaseView {
                negacyclic: neg.as_deref(),
                cyclic: cyc.as_deref(),
                params,
            }),
            _ => None,
        }
    }

    /// Borrow the Q64 i32 base domains and their bound parameters.
    #[must_use]
    pub fn q64_base(&self) -> Option<PreparedNttBaseView<'_, i32, Q64_NUM_PRIMES, D>> {
        match &self.0 {
            PreparedNttCacheRepr::Q64 {
                neg, cyc, params, ..
            } => Some(PreparedNttBaseView {
                negacyclic: neg.as_deref(),
                cyclic: cyc.as_deref(),
                params,
            }),
            _ => None,
        }
    }

    /// Borrow the Q128 i32 base domains and their bound parameters.
    #[must_use]
    pub fn q128_base(&self) -> Option<PreparedNttBaseView<'_, i32, Q128_NUM_PRIMES, D>> {
        match &self.0 {
            PreparedNttCacheRepr::Q128 {
                neg, cyc, params, ..
            } => Some(PreparedNttBaseView {
                negacyclic: neg.as_deref(),
                cyclic: cyc.as_deref(),
                params,
            }),
            _ => None,
        }
    }

    /// Borrow an exactness-only paired tail cache.
    #[must_use]
    pub fn i16_tail_pair(&self) -> Option<PreparedNttTailPairView<'_, D>> {
        match &self.0 {
            PreparedNttCacheRepr::I16TailPair { neg, cyc, params } => {
                Some(PreparedNttTailPairView {
                    negacyclic: neg,
                    cyclic: cyc,
                    params,
                })
            }
            _ => None,
        }
    }

    /// Compute a shape-checked exact signed-i16 matrix product.
    #[inline]
    pub fn mat_vec_i16<F: Field + CanonicalEncoding>(
        &self,
        log_basis: u32,
        num_rows: usize,
        rhs: &[[i16; D]],
    ) -> Result<Vec<akita_algebra::CyclotomicRing<F, D>>, AkitaError> {
        self.0.mat_vec_i16(log_basis, num_rows, rhs)
    }
}

#[inline]
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
    F: Field + CanonicalEncoding,
{
    if !exact {
        return Err(AkitaError::InvalidSetup(
            "signed-i16 matvec requested from a cyclic cache".into(),
        ));
    }
    let width = rhs.len();
    let rhs_abs_bound = validate_i16_rhs(log_basis, rhs)?;
    let needs_tail = required_profile_for_params::<F, _, K, D>(params, width, rhs_abs_bound)?;
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

fn validate_i16_rhs<const D: usize>(log_basis: u32, rhs: &[[i16; D]]) -> Result<u64, AkitaError> {
    let Some(bound) = balanced_signed_digit_abs_bound(log_basis) else {
        return Err(AkitaError::InvalidProof);
    };
    if rhs.is_empty() {
        return Err(AkitaError::InvalidProof);
    }
    // At basis 16 every i16 value is already inside the balanced digit range.
    if bound == 1u64 << (i16::BITS - 1) {
        return Ok(bound);
    }
    let digits_valid =
        akita_algebra::ntt::i16_values_in_balanced_range(rhs.as_flattened(), bound as i16);
    if !digits_valid {
        return Err(AkitaError::InvalidProof);
    }
    Ok(bound)
}

fn validate_cache_mode(mode: NttCacheMode) -> Result<(), AkitaError> {
    if let NttCacheMode::ExactNegacyclic {
        width,
        rhs_abs_bound,
    } = mode
    {
        if width == 0 {
            return Err(AkitaError::InvalidSetup(
                "exact negacyclic NTT width must be nonzero".into(),
            ));
        }
        if rhs_abs_bound == 0 {
            return Err(AkitaError::InvalidSetup(
                "exact negacyclic RHS absolute bound must be nonzero".into(),
            ));
        }
    }
    Ok(())
}

/// Prepare exactly the NTT representations requested by `mode`.
#[tracing::instrument(skip_all, name = "prepare_ntt_cache", fields(ring_d = D, rings = matrix.as_slice().len(), ?mode))]
pub fn prepare_ntt_cache<F: Field + CanonicalEncoding, const D: usize>(
    matrix: RingMatrixView<'_, F, D>,
    mode: NttCacheMode,
) -> Result<PreparedNttCache<D>, AkitaError> {
    prepare_ntt_cache_with_tail_prefix(matrix, mode, None, select_crt_ntt_params::<F, D>()?)
}

/// Prepare the exact-prefix paired-transform cache used by compressed commitments.
///
/// Uses [`select_compression_crt_ntt_params`] so compression-only ring degrees do
/// not widen the ordinary protocol NTT selector.
#[tracing::instrument(
    skip_all,
    name = "prepare_compression_ntt_cache",
    fields(ring_d = D, rings = matrix.as_slice().len())
)]
pub fn prepare_compression_ntt_cache<F: Field + CanonicalEncoding, const D: usize>(
    matrix: RingMatrixView<'_, F, D>,
) -> Result<PreparedNttCache<D>, AkitaError> {
    prepare_ntt_cache_with_tail_prefix(
        matrix,
        NttCacheMode::BothTransforms,
        None,
        select_compression_crt_ntt_params::<F, D>()?,
    )
}

/// Prepare the exact-prefix negacyclic-only cache used by reduced-evaluation
/// compressed commitments.
///
/// This uses the compression CRT selector because compression-only ring
/// degrees need not belong to the ordinary protocol NTT ladder.
#[tracing::instrument(
    skip_all,
    name = "prepare_reduced_compression_ntt_cache",
    fields(ring_d = D, rings = matrix.as_slice().len())
)]
pub fn prepare_reduced_compression_ntt_cache<F: Field + CanonicalEncoding, const D: usize>(
    matrix: RingMatrixView<'_, F, D>,
) -> Result<PreparedNttCache<D>, AkitaError> {
    prepare_ntt_cache_with_tail_prefix(
        matrix,
        NttCacheMode::Negacyclic,
        None,
        select_compression_crt_ntt_params::<F, D>()?,
    )
}

fn prepare_ntt_cache_with_tail_prefix<F: Field + CanonicalEncoding, const D: usize>(
    matrix: RingMatrixView<'_, F, D>,
    mode: NttCacheMode,
    tail_prefix_len: Option<usize>,
    selected: ProtocolCrtNttParams<D>,
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
    if let NttCacheMode::ExactNegacyclic {
        width,
        rhs_abs_bound,
    } = mode
    {
        let plan = exact_cache_plan::<F, D>(selected, width, rhs_abs_bound)?;
        return prepare_exact_ntt_cache(matrix, tail_prefix_len, plan);
    }
    macro_rules! prepare {
        ($params:expr, $variant:ident) => {{
            let params = $params;
            match mode {
                NttCacheMode::Negacyclic => PreparedNttCacheRepr::$variant {
                    neg: Some(
                        cfg_iter!(matrix.as_slice())
                            .map(|ring| CyclotomicCrtNtt::from_ring(ring, &params))
                            .collect(),
                    ),
                    cyc: None,
                    params,
                    tail: None,
                    exact: false,
                },
                NttCacheMode::Cyclic => PreparedNttCacheRepr::$variant {
                    neg: None,
                    cyc: Some(
                        cfg_iter!(matrix.as_slice())
                            .map(|ring| CyclotomicCrtNtt::from_ring_cyclic(ring, &params))
                            .collect(),
                    ),
                    params,
                    tail: None,
                    exact: false,
                },
                NttCacheMode::BothTransforms => {
                    let (neg, cyc) = convert_flat_pair(matrix, &params);
                    PreparedNttCacheRepr::$variant {
                        neg: Some(neg),
                        cyc: Some(cyc),
                        params,
                        tail: None,
                        exact: false,
                    }
                }
                NttCacheMode::I16TailBothTransforms => {
                    return Err(AkitaError::InvalidSetup(
                        "i16-tail cache bypassed its dedicated representation".into(),
                    ));
                }
                NttCacheMode::ExactNegacyclic { .. } => {
                    return Err(AkitaError::InvalidSetup(
                        "exact NTT cache bypassed its representation plan".into(),
                    ));
                }
            }
        }};
    }
    if mode == NttCacheMode::I16TailBothTransforms {
        let params = CrtNttParamSet::<i16, 1, D>::new([I16_TAIL_PRIME]);
        let (neg, cyc) = convert_flat_pair(matrix, &params);
        let prepared = PreparedNttCacheRepr::I16TailPair { neg, cyc, params };
        prepared.validate()?;
        return Ok(PreparedNttCache(prepared));
    }
    let prepared = match selected {
        ProtocolCrtNttParams::Q32(params) => prepare!(params, Q32),
        ProtocolCrtNttParams::Q64(params) => prepare!(params, Q64),
        ProtocolCrtNttParams::Q128(params) => prepare!(params, Q128),
    };
    prepared.validate()?;
    Ok(PreparedNttCache(prepared))
}

fn convert_flat_pair<F, W, const K: usize, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
    params: &CrtNttParamSet<W, K, D>,
) -> (
    Vec<CyclotomicCrtNtt<W, K, D>>,
    Vec<CyclotomicCrtNtt<W, K, D>>,
)
where
    F: Field + CanonicalEncoding,
    W: akita_algebra::PrimeWidth,
{
    cfg_iter!(mat.as_slice())
        .map(|ring| CyclotomicCrtNtt::from_ring_pair_with_params(ring, params))
        .unzip()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct VerifierNttCacheKey {
    ring_d: usize,
    width: usize,
    rhs_abs_bound: u64,
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

    pub(crate) fn install_trusted<const D: usize>(
        &self,
        metadata: PreparedVerifierNttCacheMetadata,
        prepared: PreparedNttCache<D>,
    ) -> Result<(), AkitaError> {
        if metadata.ring_dimension != D
            || !prepared.has_negacyclic()
            || prepared.has_cyclic()
            || prepared.has_exactness_tail() != (metadata.tail_prefix_len > 0)
        {
            return Err(AkitaError::InvalidSetup(
                "trusted prepared verifier cache has inconsistent geometry".into(),
            ));
        }
        let key = VerifierNttCacheKey {
            ring_d: D,
            width: metadata.width,
            rhs_abs_bound: metadata.rhs_abs_bound,
        };
        let prepared = Arc::new(prepared);
        let built = Arc::new(ErasedVerifierNttCache {
            ring_d: D,
            base_prefix_len: metadata.base_prefix_len,
            tail_prefix_len: metadata.tail_prefix_len,
            cache_bytes: prepared.cache_bytes(),
            cache: prepared,
        });
        self.slots
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("verifier NTT cache lock poisoned".into()))?
            .insert(key, built);
        Ok(())
    }

    /// Build, erase, and atomically install an entry when needed.
    pub(crate) fn prepare<F: Field + CanonicalEncoding, const D: usize>(
        &self,
        expanded: &AkitaExpandedSetup<F>,
        matrix: NttCacheKey,
        tail_prefix_len: usize,
        mode: NttCacheMode,
    ) -> Result<Arc<PreparedNttCache<D>>, AkitaError> {
        let NttCacheMode::ExactNegacyclic {
            width,
            rhs_abs_bound,
        } = mode
        else {
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
        let with_exactness_tail = ntt_cache_requires_exactness_tail::<F, D>(width, rhs_abs_bound)?;
        if with_exactness_tail != (tail_prefix_len > 0) {
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
        let key = VerifierNttCacheKey {
            ring_d: D,
            width,
            rhs_abs_bound,
        };
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
            select_crt_ntt_params::<F, D>()?,
        )?);
        if prepared.has_exactness_tail() != (tail_prefix_len > 0) {
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
mod tests;
