//! Protocol-facing CRT+NTT parameter dispatch and matrix caching.

use crate::algebra::ntt::prime::PrimeWidth;
use crate::algebra::ntt::tables::{
    q128_primes, q64_primes, MAX_CRT_RING_DEGREE, Q128_MODULUS, Q128_NUM_PRIMES, Q32_MODULUS,
    Q32_NUM_PRIMES, Q32_PRIMES, Q64_MODULUS, Q64_NUM_PRIMES, RING_DEGREE,
};
use crate::algebra::ring::{CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing};
use crate::cfg_iter;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::{CanonicalField, FieldCore};

use super::norm::detect_field_modulus;

/// Supported protocol CRT+NTT parameter families.
pub(crate) enum ProtocolCrtNttParams<const D: usize> {
    Q32(CrtNttParamSet<i16, Q32_NUM_PRIMES, D>),
    Q64(CrtNttParamSet<i32, Q64_NUM_PRIMES, D>),
    Q128(CrtNttParamSet<i32, Q128_NUM_PRIMES, D>),
}

/// Select a CRT+NTT parameter set from field modulus and ring degree.
///
/// Dispatch policy:
/// - `q <= 2^32-99` and `D <= 64`: Q32 (`i16`)
/// - `q <= 2^64-59` and `D <= 1024`: Q64 (`i32`, conservative K=5)
/// - `q == 2^128-275` and `D <= 1024`: Q128 (`i32`, K=5)
/// - otherwise: explicit setup error
pub(crate) fn select_crt_ntt_params<F: CanonicalField, const D: usize>(
) -> Result<ProtocolCrtNttParams<D>, HachiError> {
    if !D.is_power_of_two() {
        return Err(HachiError::InvalidSetup(format!(
            "CRT+NTT requires power-of-two ring degree, got D={D}"
        )));
    }
    if D > MAX_CRT_RING_DEGREE {
        return Err(HachiError::InvalidSetup(format!(
            "CRT+NTT supports D <= {MAX_CRT_RING_DEGREE}, got D={D}"
        )));
    }

    let modulus = detect_field_modulus::<F>();

    if modulus == Q128_MODULUS {
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

    Err(HachiError::InvalidSetup(format!(
        "no CRT+NTT parameter set for modulus {modulus} and D={D}; supported ranges: <= {Q64_MODULUS} (with Q32/Q64 dispatch) or exactly {Q128_MODULUS}"
    )))
}

/// Pre-converted CRT+NTT matrices, keyed by parameter family.
///
/// Avoids repeated coefficient-to-NTT conversion on every dense mat-vec.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(non_snake_case, missing_docs)]
pub enum NttMatrixCache<const D: usize> {
    /// 32-bit CRT primes.
    Q32 {
        A: Vec<Vec<CyclotomicCrtNtt<i16, Q32_NUM_PRIMES, D>>>,
        B: Vec<Vec<CyclotomicCrtNtt<i16, Q32_NUM_PRIMES, D>>>,
        D: Vec<Vec<CyclotomicCrtNtt<i16, Q32_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i16, Q32_NUM_PRIMES, D>,
    },
    /// 64-bit CRT primes.
    Q64 {
        A: Vec<Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>>,
        B: Vec<Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>>,
        D: Vec<Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i32, Q64_NUM_PRIMES, D>,
    },
    /// 128-bit CRT primes.
    Q128 {
        A: Vec<Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>>,
        B: Vec<Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>>,
        D: Vec<Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i32, Q128_NUM_PRIMES, D>,
    },
}

fn convert_mat<F, W, const K: usize, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicCrtNtt<W, K, D>>>
where
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
{
    cfg_iter!(mat)
        .map(|row| {
            row.iter()
                .map(|a| CyclotomicCrtNtt::from_ring_with_params(a, params))
                .collect()
        })
        .collect()
}

#[allow(non_snake_case)]
pub(crate) fn build_ntt_cache<F: FieldCore + CanonicalField, const D: usize>(
    a: &[Vec<CyclotomicRing<F, D>>],
    b: &[Vec<CyclotomicRing<F, D>>],
    d: &[Vec<CyclotomicRing<F, D>>],
) -> Result<NttMatrixCache<D>, HachiError> {
    let params = select_crt_ntt_params::<F, D>()?;
    let cache = match params {
        ProtocolCrtNttParams::Q32(p) => NttMatrixCache::Q32 {
            A: convert_mat(a, &p),
            B: convert_mat(b, &p),
            D: convert_mat(d, &p),
            params: p,
        },
        ProtocolCrtNttParams::Q64(p) => NttMatrixCache::Q64 {
            A: convert_mat(a, &p),
            B: convert_mat(b, &p),
            D: convert_mat(d, &p),
            params: p,
        },
        ProtocolCrtNttParams::Q128(p) => NttMatrixCache::Q128 {
            A: convert_mat(a, &p),
            B: convert_mat(b, &p),
            D: convert_mat(d, &p),
            params: p,
        },
    };
    Ok(cache)
}
