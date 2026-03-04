//! Protocol-facing CRT+NTT parameter dispatch and matrix caching.

use crate::algebra::ntt::prime::PrimeWidth;
use crate::algebra::ntt::tables::{
    q128_primes, q64_primes, MAX_CRT_RING_DEGREE, Q128_MODULUS, Q128_NUM_PRIMES, Q32_MODULUS,
    Q32_NUM_PRIMES, Q32_PRIMES, Q64_MODULUS, Q64_NUM_PRIMES, RING_DEGREE,
};
use crate::algebra::ring::{CrtNttParamSet, CyclotomicCrtNtt};
use crate::cfg_into_iter;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::{CanonicalField, FieldCore};

use super::flat_matrix::RingMatrixView;
use super::norm::detect_field_modulus;

/// Supported protocol CRT+NTT parameter families.
#[derive(Clone)]
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

/// Pre-converted CRT+NTT cache for a single matrix, keyed by parameter family.
///
/// Stores both negacyclic (for mat-vec) and cyclic (for quotient) representations
/// to avoid repeated coefficient-to-NTT conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum NttSlotCache<const D: usize> {
    /// 32-bit CRT primes.
    Q32 {
        neg: Vec<Vec<CyclotomicCrtNtt<i16, Q32_NUM_PRIMES, D>>>,
        cyc: Vec<Vec<CyclotomicCrtNtt<i16, Q32_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i16, Q32_NUM_PRIMES, D>,
    },
    /// 64-bit CRT primes.
    Q64 {
        neg: Vec<Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>>,
        cyc: Vec<Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i32, Q64_NUM_PRIMES, D>,
    },
    /// 128-bit CRT primes.
    Q128 {
        neg: Vec<Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>>,
        cyc: Vec<Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>>,
        params: CrtNttParamSet<i32, Q128_NUM_PRIMES, D>,
    },
}

fn convert_mat<F, W, const K: usize, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicCrtNtt<W, K, D>>>
where
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
{
    cfg_into_iter!(0..mat.num_rows())
        .map(|i| {
            mat.row(i)
                .iter()
                .map(|a| CyclotomicCrtNtt::from_ring_with_params(a, params))
                .collect()
        })
        .collect()
}

fn convert_mat_cyclic<F, W, const K: usize, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicCrtNtt<W, K, D>>>
where
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
{
    cfg_into_iter!(0..mat.num_rows())
        .map(|i| {
            mat.row(i)
                .iter()
                .map(|a| CyclotomicCrtNtt::from_ring_cyclic(a, params))
                .collect()
        })
        .collect()
}

/// Build an NTT slot cache for a single matrix.
///
/// # Errors
///
/// Returns an error if no CRT+NTT parameter set matches the field modulus and ring degree.
#[tracing::instrument(skip_all, name = "build_ntt_slot")]
pub fn build_ntt_slot<F: FieldCore + CanonicalField, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
) -> Result<NttSlotCache<D>, HachiError> {
    let params = select_crt_ntt_params::<F, D>()?;
    Ok(build_ntt_slot_from_params(mat, params))
}

fn build_ntt_slot_from_params<F: FieldCore + CanonicalField, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
    params: ProtocolCrtNttParams<D>,
) -> NttSlotCache<D> {
    match params {
        ProtocolCrtNttParams::Q32(p) => NttSlotCache::Q32 {
            neg: convert_mat(mat, &p),
            cyc: convert_mat_cyclic(mat, &p),
            params: p,
        },
        ProtocolCrtNttParams::Q64(p) => NttSlotCache::Q64 {
            neg: convert_mat(mat, &p),
            cyc: convert_mat_cyclic(mat, &p),
            params: p,
        },
        ProtocolCrtNttParams::Q128(p) => NttSlotCache::Q128 {
            neg: convert_mat(mat, &p),
            cyc: convert_mat_cyclic(mat, &p),
            params: p,
        },
    }
}

/// Build NTT slot caches for three matrices, computing CRT+NTT parameters once.
///
/// # Errors
///
/// Returns an error if no CRT+NTT parameter set matches the field modulus and ring degree.
#[tracing::instrument(skip_all, name = "build_ntt_slots")]
#[allow(non_snake_case)]
pub fn build_ntt_slots<F: FieldCore + CanonicalField, const D: usize>(
    A: RingMatrixView<'_, F, D>,
    B: RingMatrixView<'_, F, D>,
    D_mat: RingMatrixView<'_, F, D>,
) -> Result<(NttSlotCache<D>, NttSlotCache<D>, NttSlotCache<D>), HachiError> {
    let params = select_crt_ntt_params::<F, D>()?;
    let slot_a = build_ntt_slot_from_params(A, params.clone());
    let slot_b = build_ntt_slot_from_params(B, params.clone());
    let slot_d = build_ntt_slot_from_params(D_mat, params);
    Ok((slot_a, slot_b, slot_d))
}
