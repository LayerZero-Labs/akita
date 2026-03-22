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
#[allow(clippy::large_enum_variant)]
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
        "no CRT+NTT parameter set for modulus {modulus} and D={D}; supported ranges: <= {Q64_MODULUS} (with Q32/Q64 dispatch) or {Q128_MODULUS}"
    )))
}

/// Pre-converted CRT+NTT cache for a single matrix, keyed by parameter family.
///
/// Stores both negacyclic (for mat-vec) and cyclic (for quotient) representations
/// to avoid repeated coefficient-to-NTT conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs, clippy::large_enum_variant)]
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

/// Row-bounded access macro for `NttSlotCache` variants.
macro_rules! impl_row_views {
    () => {
        /// Return the first `n` rows of the negacyclic NTT cache.
        pub fn neg_rows(&self, n: usize) -> NttRowView<'_, D> {
            match self {
                NttSlotCache::Q32 { neg, params, .. } => NttRowView::Q32(&neg[..n], params),
                NttSlotCache::Q64 { neg, params, .. } => NttRowView::Q64(&neg[..n], params),
                NttSlotCache::Q128 { neg, params, .. } => NttRowView::Q128(&neg[..n], params),
            }
        }

        /// Return the first `n` rows of the cyclic NTT cache.
        pub fn cyc_rows(&self, n: usize) -> NttRowView<'_, D> {
            match self {
                NttSlotCache::Q32 { cyc, params, .. } => NttRowView::Q32(&cyc[..n], params),
                NttSlotCache::Q64 { cyc, params, .. } => NttRowView::Q64(&cyc[..n], params),
                NttSlotCache::Q128 { cyc, params, .. } => NttRowView::Q128(&cyc[..n], params),
            }
        }
    };
}

impl<const D: usize> NttSlotCache<D> {
    impl_row_views!();
}

/// A row-bounded view into an NTT cache, carrying the params reference.
///
/// Consumers can use this to pass row-sliced NTT data to mat-vec functions
/// without copying. The view borrows from the parent `NttSlotCache`.
#[derive(Debug)]
#[allow(missing_docs, clippy::large_enum_variant)]
pub enum NttRowView<'a, const D: usize> {
    Q32(
        &'a [Vec<CyclotomicCrtNtt<i16, Q32_NUM_PRIMES, D>>],
        &'a CrtNttParamSet<i16, Q32_NUM_PRIMES, D>,
    ),
    Q64(
        &'a [Vec<CyclotomicCrtNtt<i32, Q64_NUM_PRIMES, D>>],
        &'a CrtNttParamSet<i32, Q64_NUM_PRIMES, D>,
    ),
    Q128(
        &'a [Vec<CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D>>],
        &'a CrtNttParamSet<i32, Q128_NUM_PRIMES, D>,
    ),
}
