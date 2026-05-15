//! Protocol-facing CRT+NTT parameter dispatch and matrix caching.

use akita_algebra::ntt::prime::PrimeWidth;
use akita_algebra::ntt::tables::{
    q128_primes, q64_primes, MAX_CRT_RING_DEGREE, Q128_MODULUS, Q128_NUM_PRIMES, Q32_MODULUS,
    Q32_NUM_PRIMES, Q32_PRIMES, Q64_MODULUS, Q64_NUM_PRIMES, RING_DEGREE,
};
use akita_algebra::ring::{CrtNttParamSet, CyclotomicCrtNtt};
#[allow(unused_imports)]
use akita_field::parallel::*;
use akita_field::{
    cfg_into_iter, cfg_join, AkitaError, CanonicalField, FieldCore, PseudoMersenneField,
};
use akita_field::{Prime128Offset159, Prime128Offset2355, Prime128OffsetA7F7};

use akita_types::RingMatrixView;

/// Supported protocol CRT+NTT parameter families.
#[derive(Clone)]
#[allow(missing_docs, clippy::large_enum_variant)]
pub enum ProtocolCrtNttParams<const D: usize> {
    Q32(CrtNttParamSet<i16, Q32_NUM_PRIMES, D>),
    Q64(CrtNttParamSet<i32, Q64_NUM_PRIMES, D>),
    Q128(CrtNttParamSet<i32, Q128_NUM_PRIMES, D>),
}

/// Select a CRT+NTT parameter set from field modulus and ring degree.
///
/// Dispatch policy:
/// - `q <= 2^32-99` and `D <= 64`: Q32 (`i16`, K=4)
/// - `q <= 2^64-59` and `D <= 1024`: Q64 (`i32`, K=3)
/// - `q ∈ { 2^128-275, 2^128-159, 2^128-2355, 2^128-2^32+22537 }` and
///   `D <= 1024`: Q128 (`i32`, K=5)
/// - otherwise: explicit setup error
///
/// # Errors
///
/// Returns an error if `D` is unsupported or no CRT/NTT parameter family
/// matches the field modulus.
pub fn select_crt_ntt_params<F: CanonicalField, const D: usize>(
) -> Result<ProtocolCrtNttParams<D>, AkitaError> {
    if !D.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(format!(
            "CRT+NTT requires power-of-two ring degree, got D={D}"
        )));
    }
    if D > MAX_CRT_RING_DEGREE {
        return Err(AkitaError::InvalidSetup(format!(
            "CRT+NTT supports D <= {MAX_CRT_RING_DEGREE}, got D={D}"
        )));
    }

    let modulus = detect_field_modulus::<F>();
    let split_only_q128_modulus =
        u128::MAX - (<Prime128Offset159 as PseudoMersenneField>::MODULUS_OFFSET - 1);
    let ntt_q128_modulus =
        u128::MAX - (<Prime128Offset2355 as PseudoMersenneField>::MODULUS_OFFSET - 1);
    let a7f7_q128_modulus =
        u128::MAX - (<Prime128OffsetA7F7 as PseudoMersenneField>::MODULUS_OFFSET - 1);

    if modulus == Q128_MODULUS
        || modulus == split_only_q128_modulus
        || modulus == ntt_q128_modulus
        || modulus == a7f7_q128_modulus
    {
        return Ok(ProtocolCrtNttParams::Q128(CrtNttParamSet::new(
            q128_primes(),
        )));
    }

    if modulus <= Q32_MODULUS as u128 {
        if D <= RING_DEGREE {
            return Ok(ProtocolCrtNttParams::Q32(CrtNttParamSet::new(Q32_PRIMES)));
        }
        return Ok(ProtocolCrtNttParams::Q64(CrtNttParamSet::new(q64_primes())));
    }

    if modulus <= Q64_MODULUS as u128 {
        return Ok(ProtocolCrtNttParams::Q64(CrtNttParamSet::new(q64_primes())));
    }

    Err(AkitaError::InvalidSetup(format!(
        "no CRT+NTT parameter set for modulus {modulus} and D={D}; supported ranges: <= {Q64_MODULUS} (with Q32/Q64 dispatch) or q in {{{Q128_MODULUS}, {split_only_q128_modulus}, {ntt_q128_modulus}, {a7f7_q128_modulus}}}"
    )))
}

fn detect_field_modulus<F: CanonicalField>() -> u128 {
    (-F::one()).to_canonical_u128() + 1
}

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
        neg: Vec<CyclotomicCrtNtt<i16, Q32_NUM_PRIMES, D>>,
        cyc: Vec<CyclotomicCrtNtt<i16, Q32_NUM_PRIMES, D>>,
        params: CrtNttParamSet<i16, Q32_NUM_PRIMES, D>,
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

fn convert_flat<F, W, const K: usize, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicCrtNtt<W, K, D>>
where
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
{
    let total = mat.num_rows() * mat.num_cols();
    cfg_into_iter!(0..total)
        .map(|idx| {
            let r = idx / mat.num_cols();
            let c = idx % mat.num_cols();
            CyclotomicCrtNtt::from_ring_with_params(&mat.row(r)[c], params)
        })
        .collect()
}

fn convert_flat_cyclic<F, W, const K: usize, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicCrtNtt<W, K, D>>
where
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
{
    let total = mat.num_rows() * mat.num_cols();
    cfg_into_iter!(0..total)
        .map(|idx| {
            let r = idx / mat.num_cols();
            let c = idx % mat.num_cols();
            CyclotomicCrtNtt::from_ring_cyclic(&mat.row(r)[c], params)
        })
        .collect()
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
            let (neg, cyc) = cfg_join!(|| convert_flat(mat, &p), || convert_flat_cyclic(mat, &p));
            NttSlotCache::Q32 {
                neg,
                cyc,
                params: p,
            }
        }
        ProtocolCrtNttParams::Q64(p) => {
            let (neg, cyc) = cfg_join!(|| convert_flat(mat, &p), || convert_flat_cyclic(mat, &p));
            NttSlotCache::Q64 {
                neg,
                cyc,
                params: p,
            }
        }
        ProtocolCrtNttParams::Q128(p) => {
            let (neg, cyc) = cfg_join!(|| convert_flat(mat, &p), || convert_flat_cyclic(mat, &p));
            NttSlotCache::Q128 {
                neg,
                cyc,
                params: p,
            }
        }
    }
}

impl<const D: usize> NttSlotCache<D> {
    /// Total number of NTT elements stored in this cache.
    pub fn total_elements(&self) -> usize {
        match self {
            NttSlotCache::Q32 { neg, .. } => neg.len(),
            NttSlotCache::Q64 { neg, .. } => neg.len(),
            NttSlotCache::Q128 { neg, .. } => neg.len(),
        }
    }
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use akita_algebra::CyclotomicRing;
    use akita_field::{
        Prime128Offset159, Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7,
        Prime32Offset99, Prime64Offset59,
    };

    fn assert_selects_q128_params<F: CanonicalField, const D: usize>() {
        assert!(matches!(
            select_crt_ntt_params::<F, D>(),
            Ok(ProtocolCrtNttParams::Q128(_))
        ));
    }

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

    fn deterministic_ring<F: FieldCore + CanonicalField, const D: usize>() -> CyclotomicRing<F, D> {
        let q = (-F::one()).to_canonical_u128() + 1;
        let coeffs = std::array::from_fn(|i| {
            let value = match i % 8 {
                0 => 0,
                1 => 1,
                2 => q - 1,
                3 => q / 2,
                4 => q / 2 + 1,
                5 => 17,
                6 => q - 17,
                _ => {
                    (i as u128)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .wrapping_add(0xd1b5_4a32_d192_ed03)
                        % q
                }
            };
            F::from_canonical_u128_reduced(value)
        });
        CyclotomicRing::from_coefficients(coeffs)
    }

    fn assert_crt_ntt_roundtrip<F: FieldCore + CanonicalField, const D: usize>() {
        let ring = deterministic_ring::<F, D>();
        match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
            ProtocolCrtNttParams::Q32(params) => {
                let neg = CyclotomicCrtNtt::from_ring_with_params(&ring, &params);
                assert_eq!(neg.to_ring_with_params::<F>(&params), ring);
                let cyc = CyclotomicCrtNtt::from_ring_cyclic(&ring, &params);
                assert_eq!(cyc.to_ring_cyclic::<F>(&params), ring);
            }
            ProtocolCrtNttParams::Q64(params) => {
                let neg = CyclotomicCrtNtt::from_ring_with_params(&ring, &params);
                assert_eq!(neg.to_ring_with_params::<F>(&params), ring);
                let cyc = CyclotomicCrtNtt::from_ring_cyclic(&ring, &params);
                assert_eq!(cyc.to_ring_cyclic::<F>(&params), ring);
            }
            ProtocolCrtNttParams::Q128(params) => {
                let neg = CyclotomicCrtNtt::from_ring_with_params(&ring, &params);
                assert_eq!(neg.to_ring_with_params::<F>(&params), ring);
                let cyc = CyclotomicCrtNtt::from_ring_cyclic(&ring, &params);
                assert_eq!(cyc.to_ring_cyclic::<F>(&params), ring);
            }
        }
    }

    #[test]
    fn selects_q32_params_for_prime32_offset99_d64() {
        assert_selects_q32_params::<Prime32Offset99, 64>();
    }

    #[test]
    fn selects_q64_params_for_prime64_offset59_across_small_protocol_ring_dims() {
        assert_selects_q64_params::<Prime64Offset59, 32>();
        assert_selects_q64_params::<Prime64Offset59, 64>();
    }

    #[test]
    fn roundtrips_prime32_offset99_with_q32_params() {
        assert_crt_ntt_roundtrip::<Prime32Offset99, 64>();
    }

    #[test]
    fn roundtrips_prime64_offset59_with_q64_params() {
        assert_crt_ntt_roundtrip::<Prime64Offset59, 32>();
        assert_crt_ntt_roundtrip::<Prime64Offset59, 64>();
    }

    #[test]
    fn roundtrips_prime128_offset275_with_q128_params() {
        assert_crt_ntt_roundtrip::<Prime128Offset275, 64>();
    }

    #[test]
    fn selects_q128_params_for_prime275_across_small_protocol_ring_dims() {
        assert_selects_q128_params::<Prime128Offset275, 32>();
        assert_selects_q128_params::<Prime128Offset275, 64>();
        assert_selects_q128_params::<Prime128Offset275, 128>();
    }

    #[test]
    fn selects_q128_params_for_split_only_prime159() {
        assert_selects_q128_params::<Prime128Offset159, 32>();
    }

    #[test]
    fn selects_q128_params_for_prime_a7f7_across_small_protocol_ring_dims() {
        assert_selects_q128_params::<Prime128OffsetA7F7, 32>();
        assert_selects_q128_params::<Prime128OffsetA7F7, 64>();
        assert_selects_q128_params::<Prime128OffsetA7F7, 128>();
    }

    #[test]
    fn selects_q128_params_for_prime2355_across_small_protocol_ring_dims() {
        assert_selects_q128_params::<Prime128Offset2355, 32>();
        assert_selects_q128_params::<Prime128Offset2355, 64>();
        assert_selects_q128_params::<Prime128Offset2355, 128>();
    }
}
