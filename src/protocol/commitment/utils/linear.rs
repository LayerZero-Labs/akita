//! Linear algebra helpers for ring commitment.

#[cfg(target_arch = "aarch64")]
use crate::algebra::ntt::neon;
use crate::algebra::ntt::{MontCoeff, PrimeWidth};
use crate::algebra::{CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, DigitMontLut};
#[cfg(test)]
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::{cfg_fold_reduce, cfg_into_iter, cfg_iter};
use crate::{CanonicalField, FieldCore};
use std::array::from_fn;
use std::mem::size_of;

use super::crt_ntt::NttSlotCache;
#[cfg(test)]
use super::crt_ntt::{select_crt_ntt_params, ProtocolCrtNttParams};

#[cfg(test)]
pub(crate) fn mat_vec_mul_unchecked<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(mat.len());
    for row in mat {
        debug_assert_eq!(row.len(), vec.len());
        let mut acc = CyclotomicRing::<F, D>::zero();
        for (a, x) in row.iter().zip(vec.iter()) {
            acc += *a * *x;
        }
        out.push(acc);
    }
    out
}

#[inline]
fn accumulate_pointwise_product_into<W: PrimeWidth, const K: usize, const D: usize>(
    acc: &mut CyclotomicCrtNtt<W, K, D>,
    lhs: &CyclotomicCrtNtt<W, K, D>,
    rhs: &CyclotomicCrtNtt<W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    #[cfg(target_arch = "aarch64")]
    if neon::use_neon_ntt() {
        for k in 0..K {
            let prime = params.primes[k];
            unsafe {
                if size_of::<W>() == size_of::<i32>() {
                    neon::pointwise_mul_acc_i32(
                        acc.limbs[k].as_mut_ptr() as *mut i32,
                        lhs.limbs[k].as_ptr() as *const i32,
                        rhs.limbs[k].as_ptr() as *const i32,
                        D,
                        prime.p.to_i64() as i32,
                        prime.pinv.to_i64() as i32,
                    );
                } else {
                    neon::pointwise_mul_acc_i16(
                        acc.limbs[k].as_mut_ptr() as *mut i16,
                        lhs.limbs[k].as_ptr() as *const i16,
                        rhs.limbs[k].as_ptr() as *const i16,
                        D,
                        prime.p.to_i64() as i16,
                        prime.pinv.to_i64() as i16,
                    );
                }
            }
        }
        return;
    }

    for k in 0..K {
        let prime = params.primes[k];
        let acc_limb = &mut acc.limbs[k];
        let lhs_limb = &lhs.limbs[k];
        let rhs_limb = &rhs.limbs[k];
        for ((acc_coeff, lhs_coeff), rhs_coeff) in acc_limb
            .iter_mut()
            .zip(lhs_limb.iter())
            .zip(rhs_limb.iter())
        {
            let prod = prime.mul(*lhs_coeff, *rhs_coeff);
            let sum = MontCoeff::from_raw(acc_coeff.raw().wrapping_add(prod.raw()));
            *acc_coeff = prime.reduce_range(sum);
        }
    }
}

#[cfg(test)]
fn precompute_dense_mat_ntt_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicCrtNtt<W, K, D>>> {
    mat.iter()
        .map(|row| {
            row.iter()
                .map(|a| CyclotomicCrtNtt::from_ring_with_params(a, params))
                .collect()
        })
        .collect()
}

#[cfg(test)]
fn mat_vec_mul_dense_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let ntt_vec: Vec<CyclotomicCrtNtt<W, K, D>> = vec
        .iter()
        .map(|v| CyclotomicCrtNtt::from_ring_with_params(v, params))
        .collect();

    mat.iter()
        .map(|row| {
            debug_assert_eq!(row.len(), ntt_vec.len());
            let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
            for (a, x_ntt) in row.iter().zip(ntt_vec.iter()) {
                let a_ntt = CyclotomicCrtNtt::from_ring_with_params(a, params);
                accumulate_pointwise_product_into(&mut acc, &a_ntt, x_ntt, params);
            }
            acc.to_ring_with_params(params)
        })
        .collect()
}

#[cfg(test)]
fn mat_vec_mul_dense_many_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vecs: &[Vec<CyclotomicRing<F, D>>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let ntt_mat = precompute_dense_mat_ntt_with_params(mat, params);
    vecs.iter()
        .map(|vec| {
            let ntt_vec: Vec<CyclotomicCrtNtt<W, K, D>> = vec
                .iter()
                .map(|v| CyclotomicCrtNtt::from_ring_with_params(v, params))
                .collect();

            ntt_mat
                .iter()
                .map(|row_ntt| {
                    debug_assert_eq!(row_ntt.len(), ntt_vec.len());
                    let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
                    for (a_ntt, x_ntt) in row_ntt.iter().zip(ntt_vec.iter()) {
                        accumulate_pointwise_product_into(&mut acc, a_ntt, x_ntt, params);
                    }
                    acc.to_ring_with_params(params)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn mat_vec_mul_crt_ntt<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    let params = select_crt_ntt_params::<F, D>()?;
    let out = match &params {
        ProtocolCrtNttParams::Q32(p) => mat_vec_mul_dense_with_params(mat, vec, p),
        ProtocolCrtNttParams::Q64(p) => mat_vec_mul_dense_with_params(mat, vec, p),
        ProtocolCrtNttParams::Q128(p) => mat_vec_mul_dense_with_params(mat, vec, p),
    };
    Ok(out)
}

#[cfg(test)]
pub(crate) fn mat_vec_mul_crt_ntt_many<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vecs: &[Vec<CyclotomicRing<F, D>>],
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError> {
    let params = select_crt_ntt_params::<F, D>()?;
    let out = match &params {
        ProtocolCrtNttParams::Q32(p) => mat_vec_mul_dense_many_with_params(mat, vecs, p),
        ProtocolCrtNttParams::Q64(p) => mat_vec_mul_dense_many_with_params(mat, vecs, p),
        ProtocolCrtNttParams::Q128(p) => mat_vec_mul_dense_many_with_params(mat, vecs, p),
    };
    Ok(out)
}

fn unreduced_quotient_ntt<F, W, const K: usize, const D: usize>(
    ntt_row: &[CyclotomicCrtNtt<W, K, D>],
    cyc_row: &[CyclotomicCrtNtt<W, K, D>],
    vec_neg: &[CyclotomicCrtNtt<W, K, D>],
    vec_cyc: &[CyclotomicCrtNtt<W, K, D>],
    params: &CrtNttParamSet<W, K, D>,
) -> CyclotomicRing<F, D>
where
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
{
    let n = ntt_row.len().min(vec_neg.len());

    let mut acc_neg = CyclotomicCrtNtt::<W, K, D>::zero();
    let mut acc_cyc = CyclotomicCrtNtt::<W, K, D>::zero();

    for j in 0..n {
        accumulate_pointwise_product_into(&mut acc_neg, &ntt_row[j], &vec_neg[j], params);
        accumulate_pointwise_product_into(&mut acc_cyc, &cyc_row[j], &vec_cyc[j], params);
    }

    let neg_ring: CyclotomicRing<F, D> = acc_neg.to_ring_with_params(params);
    let cyc_ring: CyclotomicRing<F, D> = acc_cyc.to_ring_cyclic(params);

    let neg_coeffs = neg_ring.coefficients();
    let cyc_coeffs = cyc_ring.coefficients();
    let quotient: [F; D] = from_fn(|k| (cyc_coeffs[k] - neg_coeffs[k]) * F::TWO_INV);
    CyclotomicRing::from_coefficients(quotient)
}

macro_rules! dispatch_slot_quotient {
    ($slot:expr, $vec:expr, $convert_neg:ident, $convert_cyc:ident, $quotient_fn:ident) => {{
        match $slot {
            NttSlotCache::Q32 {
                neg,
                cyc,
                params: p,
            } => {
                let v = $vec;
                let n = neg.first().map_or(0, |r| r.len().min(v.len()));
                let v_neg: Vec<_> = cfg_iter!(v[..n])
                    .map(|x| CyclotomicCrtNtt::$convert_neg(x, p))
                    .collect();
                let v_cyc: Vec<_> = cfg_iter!(v[..n])
                    .map(|x| CyclotomicCrtNtt::$convert_cyc(x, p))
                    .collect();
                cfg_into_iter!(0..neg.len())
                    .map(|i| $quotient_fn(&neg[i], &cyc[i], &v_neg, &v_cyc, p))
                    .collect()
            }
            NttSlotCache::Q64 {
                neg,
                cyc,
                params: p,
            } => {
                let v = $vec;
                let n = neg.first().map_or(0, |r| r.len().min(v.len()));
                let v_neg: Vec<_> = cfg_iter!(v[..n])
                    .map(|x| CyclotomicCrtNtt::$convert_neg(x, p))
                    .collect();
                let v_cyc: Vec<_> = cfg_iter!(v[..n])
                    .map(|x| CyclotomicCrtNtt::$convert_cyc(x, p))
                    .collect();
                cfg_into_iter!(0..neg.len())
                    .map(|i| $quotient_fn(&neg[i], &cyc[i], &v_neg, &v_cyc, p))
                    .collect()
            }
            NttSlotCache::Q128 {
                neg,
                cyc,
                params: p,
            } => {
                let v = $vec;
                let n = neg.first().map_or(0, |r| r.len().min(v.len()));
                let v_neg: Vec<_> = cfg_iter!(v[..n])
                    .map(|x| CyclotomicCrtNtt::$convert_neg(x, p))
                    .collect();
                let v_cyc: Vec<_> = cfg_iter!(v[..n])
                    .map(|x| CyclotomicCrtNtt::$convert_cyc(x, p))
                    .collect();
                cfg_into_iter!(0..neg.len())
                    .map(|i| $quotient_fn(&neg[i], &cyc[i], &v_neg, &v_cyc, p))
                    .collect()
            }
        }
    }};
}

/// Compute unreduced quotients for matrix rows against a witness vector.
///
/// For each row: `r_i = high_part(sum_j row_ij * vec_j) = (cyc - neg) / 2`.
/// Vec NTT conversions and matrix cyclic NTT are precomputed once (not per-row).
pub fn unreduced_quotient_rows_ntt_cached<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    dispatch_slot_quotient!(
        slot,
        vec,
        from_ring_with_params,
        from_ring_cyclic,
        unreduced_quotient_ntt
    )
}

macro_rules! dispatch_slot {
    ($slot:expr, $func:ident $(, $arg:expr)*) => {{
        match $slot {
            NttSlotCache::Q32 { neg, params: p, .. } => $func(neg, $($arg,)* p),
            NttSlotCache::Q64 { neg, params: p, .. } => $func(neg, $($arg,)* p),
            NttSlotCache::Q128 { neg, params: p, .. } => $func(neg, $($arg,)* p),
        }
    }};
}

/// Flatten a nested `Vec<Vec<[i8; D]>>` into a contiguous `Vec<[i8; D]>` using
/// bulk memcpy per block, avoiding element-by-element iteration.
pub fn flatten_i8_blocks<const D: usize>(blocks: &[Vec<[i8; D]>]) -> Vec<[i8; D]> {
    let total: usize = blocks.iter().map(|b| b.len()).sum();
    let mut flat = Vec::with_capacity(total);
    for block in blocks {
        flat.extend_from_slice(block);
    }
    flat
}

/// Basis-decompose a block of ring elements into `block.len() * num_digits` gadget components.
pub fn decompose_block<F: FieldCore + CanonicalField, const D: usize>(
    block: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = vec![CyclotomicRing::<F, D>::zero(); block.len() * num_digits];
    for (i, coeff_vec) in block.iter().enumerate() {
        coeff_vec.balanced_decompose_pow2_into(
            &mut out[i * num_digits..(i + 1) * num_digits],
            log_basis,
        );
    }
    out
}

/// Decompose each ring element in `rows` into `num_digits` gadget components.
pub fn decompose_rows<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = vec![CyclotomicRing::<F, D>::zero(); rows.len() * num_digits];
    for (i, row) in rows.iter().enumerate() {
        row.balanced_decompose_pow2_into(&mut out[i * num_digits..(i + 1) * num_digits], log_basis);
    }
    out
}

/// Decompose each ring element where the last digit carries the remainder.
///
/// # Panics
///
/// Panics if `delta == 0`.
pub fn decompose_rows_with_carry<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    delta: usize,
    log_basis: u32,
) -> Vec<CyclotomicRing<F, D>> {
    if rows.is_empty() {
        return Vec::new();
    }
    assert!(delta > 0, "levels must be positive");

    let mut out = vec![CyclotomicRing::<F, D>::zero(); rows.len() * delta];

    #[cfg(feature = "parallel")]
    out.par_chunks_mut(delta)
        .zip(rows.par_iter())
        .for_each(|(dst_chunk, row)| {
            row.balanced_decompose_pow2_with_carry_into(dst_chunk, log_basis)
        });

    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(delta)
        .zip(rows.iter())
        .for_each(|(dst_chunk, row)| {
            row.balanced_decompose_pow2_with_carry_into(dst_chunk, log_basis)
        });

    out
}

/// Like [`decompose_block`] but outputs `[i8; D]` digit planes instead of ring elements.
pub fn decompose_block_i8<F: FieldCore + CanonicalField, const D: usize>(
    block: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<[i8; D]> {
    let mut out = Vec::with_capacity(block.len() * num_digits);
    for coeff_vec in block {
        out.extend(coeff_vec.balanced_decompose_pow2_i8(num_digits, log_basis));
    }
    out
}

/// Like [`decompose_rows`] but outputs `[i8; D]` digit planes instead of ring elements.
pub fn decompose_rows_i8<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<[i8; D]> {
    let mut out = Vec::with_capacity(rows.len() * num_digits);
    for row in rows {
        out.extend(row.balanced_decompose_pow2_i8(num_digits, log_basis));
    }
    out
}

#[inline]
fn is_zero_plane<const D: usize>(plane: &[i8; D]) -> bool {
    plane.iter().all(|&d| d == 0)
}

#[cfg(target_arch = "aarch64")]
const TARGET_L2_CACHE_BYTES: usize = 4 * 1024 * 1024;
#[cfg(target_arch = "x86_64")]
const TARGET_L2_CACHE_BYTES: usize = 1024 * 1024;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const TARGET_L2_CACHE_BYTES: usize = 1024 * 1024;

#[inline]
#[allow(dead_code)]
fn add_ntt_into<W: PrimeWidth, const K: usize, const D: usize>(
    acc: &mut CyclotomicCrtNtt<W, K, D>,
    other: &CyclotomicCrtNtt<W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    #[cfg(target_arch = "aarch64")]
    if neon::use_neon_ntt() {
        for k in 0..K {
            let prime = params.primes[k];
            unsafe {
                if size_of::<W>() == size_of::<i32>() {
                    neon::add_reduce_i32(
                        acc.limbs[k].as_mut_ptr() as *mut i32,
                        other.limbs[k].as_ptr() as *const i32,
                        D,
                        prime.p.to_i64() as i32,
                    );
                } else {
                    neon::add_reduce_i16(
                        acc.limbs[k].as_mut_ptr() as *mut i16,
                        other.limbs[k].as_ptr() as *const i16,
                        D,
                        prime.p.to_i64() as i16,
                    );
                }
            }
        }
        return;
    }

    for k in 0..K {
        let prime = params.primes[k];
        for d in 0..D {
            let sum =
                MontCoeff::from_raw(acc.limbs[k][d].raw().wrapping_add(other.limbs[k][d].raw()));
            acc.limbs[k][d] = prime.reduce_range(sum);
        }
    }
}

/// Column-tiled A*x across multiple blocks simultaneously.
///
/// Each rayon thread owns one column tile of `ntt_mat` (sized to fit in L2
/// cache) and iterates over all blocks, accumulating partial NTT results.
/// The matrix is loaded from DRAM exactly once. A final reduction sums
/// partial accumulators across tiles for each block.
///
/// Accepts raw ring-coefficient slices per block. Decomposes to i8 digits
/// on-the-fly per tile to avoid materializing all digits at once.
/// Tile width is auto-computed from ring parameters and target L2 cache size.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_i8")]
pub fn mat_vec_mul_ntt_i8<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    dispatch_slot!(
        slot,
        mat_vec_mul_i8_with_params,
        blocks,
        num_digits,
        log_basis
    )
}

/// Column-tiled A*x across multiple blocks of pre-decomposed i8 digit planes.
///
/// This is the `num_digits_commit = 1` specialization of
/// [`mat_vec_mul_ntt_i8`]. It skips the `CyclotomicRing -> i8 digit plane`
/// decomposition entirely because the caller already holds each coefficient as a
/// balanced digit plane.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_digits_i8")]
pub fn mat_vec_mul_ntt_digits_i8<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    blocks: &[&[[i8; D]]],
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    dispatch_slot!(slot, mat_vec_mul_digits_i8_with_params, blocks)
}

fn mat_vec_mul_digits_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    blocks: &[&[[i8; D]]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let num_blocks = blocks.len();
    if num_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let inner_width = ntt_mat.first().map_or(0, |row| row.len());
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks];
    }

    let lut = DigitMontLut::new(params);
    let tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let num_tiles = inner_width.div_ceil(tw);

    let final_accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>> = cfg_fold_reduce!(
        0..num_tiles,
        || vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks],
        |mut accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>, tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(inner_width);

            for block_idx in 0..num_blocks {
                let block = blocks[block_idx];
                if tile_start >= block.len() {
                    continue;
                }
                let block_tile_end = tile_end.min(block.len());
                for (j, digit) in block[tile_start..block_tile_end].iter().enumerate() {
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                    for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(
                            acc,
                            &mat_row[tile_start + j],
                            &ntt_d,
                            params,
                        );
                    }
                }
            }
            accs
        },
        |mut a: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>, b| {
            for block_idx in 0..num_blocks {
                for row in 0..n_a {
                    add_ntt_into(&mut a[block_idx][row], &b[block_idx][row], params);
                }
            }
            a
        }
    );

    cfg_into_iter!(final_accs)
        .map(|row_accs| {
            row_accs
                .into_iter()
                .map(|acc| acc.to_ring_with_params(params))
                .collect()
        })
        .collect()
}

fn mat_vec_mul_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let num_blocks = blocks.len();
    if num_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let inner_width = ntt_mat.first().map_or(0, |row| row.len());
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks];
    }

    let lut = DigitMontLut::new(params);
    let tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let num_tiles = inner_width.div_ceil(tw);

    let final_accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>> = cfg_fold_reduce!(
        0..num_tiles,
        || vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks],
        |mut accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>, tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(inner_width);
            let ring_start = tile_start / num_digits;
            let ring_end = ((tile_end - 1) / num_digits) + 1;
            let digit_offset = tile_start - ring_start * num_digits;
            let tile_len = tile_end - tile_start;

            for block_idx in 0..num_blocks {
                let block = blocks[block_idx];
                if ring_start >= block.len() {
                    continue;
                }
                let block_ring_end = ring_end.min(block.len());
                let partial_coeffs = &block[ring_start..block_ring_end];
                let all_digits = decompose_block_i8(partial_coeffs, num_digits, log_basis);
                let available = all_digits.len().saturating_sub(digit_offset);
                let n = tile_len.min(available);

                for (j, digit) in all_digits[digit_offset..digit_offset + n]
                    .iter()
                    .enumerate()
                {
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                    for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(
                            acc,
                            &mat_row[tile_start + j],
                            &ntt_d,
                            params,
                        );
                    }
                }
            }
            accs
        },
        |mut a: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>, b| {
            for block_idx in 0..num_blocks {
                for row in 0..n_a {
                    add_ntt_into(&mut a[block_idx][row], &b[block_idx][row], params);
                }
            }
            a
        }
    );

    cfg_into_iter!(final_accs)
        .map(|row_accs| {
            row_accs
                .into_iter()
                .map(|acc| acc.to_ring_with_params(params))
                .collect()
        })
        .collect()
}

/// Column-tiled mat-vec for a single pre-decomposed i8 digit vector.
///
/// Same tiling strategy as [`mat_vec_mul_ntt_i8`] but for a single
/// input vector of i8 digit planes (already decomposed). Tiles the matrix
/// columns to keep each tile in L2, eliminating the full `ntt_vec`
/// materialization of the non-tiled path.
/// Tile width is auto-computed from ring parameters and target L2 cache size.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_single_i8")]
pub fn mat_vec_mul_ntt_single_i8<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    vec: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>> {
    dispatch_slot!(slot, mat_vec_mul_single_i8_with_params, vec)
}

fn mat_vec_mul_single_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    vec: &[[i8; D]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let n_a = ntt_mat.len();
    let inner_width = ntt_mat.first().map_or(0, |row| row.len());
    if inner_width == 0 || n_a == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); n_a];
    }

    let lut = DigitMontLut::new(params);
    let vec_len = vec.len().min(inner_width);
    let tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let num_tiles = vec_len.div_ceil(tw);

    let final_accs: Vec<CyclotomicCrtNtt<W, K, D>> = cfg_fold_reduce!(
        0..num_tiles,
        || vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a],
        |mut accs: Vec<CyclotomicCrtNtt<W, K, D>>, tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(vec_len);
            for (j, digit) in vec[tile_start..tile_end].iter().enumerate() {
                if is_zero_plane(digit) {
                    continue;
                }
                let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                    accumulate_pointwise_product_into(
                        acc,
                        &mat_row[tile_start + j],
                        &ntt_d,
                        params,
                    );
                }
            }
            accs
        },
        |mut a: Vec<CyclotomicCrtNtt<W, K, D>>, b| {
            for row in 0..n_a {
                add_ntt_into(&mut a[row], &b[row], params);
            }
            a
        }
    );

    final_accs
        .into_iter()
        .map(|acc| acc.to_ring_with_params(params))
        .collect()
}

/// Like [`unreduced_quotient_rows_ntt_cached`] but accepts i8 digit planes
/// instead of ring elements, using direct i8 -> CRT+NTT conversion.
/// Column-tiled with zero-skip for all-zero digit planes.
#[tracing::instrument(skip_all, name = "unreduced_quotient_rows_ntt_cached_i8")]
pub fn unreduced_quotient_rows_ntt_cached_i8<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    vec: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>> {
    match slot {
        NttSlotCache::Q32 {
            neg,
            cyc,
            params: p,
        } => quotient_single_i8_with_params(neg, cyc, vec, p),
        NttSlotCache::Q64 {
            neg,
            cyc,
            params: p,
        } => quotient_single_i8_with_params(neg, cyc, vec, p),
        NttSlotCache::Q128 {
            neg,
            cyc,
            params: p,
        } => quotient_single_i8_with_params(neg, cyc, vec, p),
    }
}

fn quotient_single_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_neg: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    ntt_cyc: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    vec: &[[i8; D]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let n_a = ntt_neg.len();
    let inner_width = ntt_neg.first().map_or(0, |row| row.len());
    if inner_width == 0 || n_a == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); n_a];
    }

    let lut = DigitMontLut::new(params);
    let vec_len = vec.len().min(inner_width);
    let tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let num_tiles = vec_len.div_ceil(tw);

    let zero = CyclotomicCrtNtt::<W, K, D>::zero();

    let (final_neg, final_cyc): (
        Vec<CyclotomicCrtNtt<W, K, D>>,
        Vec<CyclotomicCrtNtt<W, K, D>>,
    ) = cfg_fold_reduce!(
        0..num_tiles,
        || (vec![zero.clone(); n_a], vec![zero.clone(); n_a]),
        |mut accs: (
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>
        ),
         tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(vec_len);
            for (j, digit) in vec[tile_start..tile_end].iter().enumerate() {
                if is_zero_plane(digit) {
                    continue;
                }
                let ntt_d_neg = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                let ntt_d_cyc = CyclotomicCrtNtt::from_i8_cyclic_with_lut(digit, params, &lut);
                let col = tile_start + j;
                for (row, (acc_neg, acc_cyc)) in
                    accs.0.iter_mut().zip(accs.1.iter_mut()).enumerate()
                {
                    accumulate_pointwise_product_into(
                        acc_neg,
                        &ntt_neg[row][col],
                        &ntt_d_neg,
                        params,
                    );
                    accumulate_pointwise_product_into(
                        acc_cyc,
                        &ntt_cyc[row][col],
                        &ntt_d_cyc,
                        params,
                    );
                }
            }
            accs
        },
        |mut a: (
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>
        ),
         b| {
            for row in 0..n_a {
                add_ntt_into(&mut a.0[row], &b.0[row], params);
                add_ntt_into(&mut a.1[row], &b.1[row], params);
            }
            a
        }
    );

    final_neg
        .into_iter()
        .zip(final_cyc)
        .map(|(neg_acc, cyc_acc)| {
            let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring_with_params(params);
            let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
            let neg_c = neg_ring.coefficients();
            let cyc_c = cyc_ring.coefficients();
            let q: [F; D] = from_fn(|k| (cyc_c[k] - neg_c[k]) * F::TWO_INV);
            CyclotomicRing::from_coefficients(q)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        mat_vec_mul_crt_ntt, mat_vec_mul_crt_ntt_many, mat_vec_mul_digits_i8_with_params,
        mat_vec_mul_i8_with_params, mat_vec_mul_unchecked, precompute_dense_mat_ntt_with_params,
    };
    use crate::algebra::{CyclotomicRing, Fp64};
    use crate::protocol::commitment::utils::crt_ntt::{
        select_crt_ntt_params, ProtocolCrtNttParams,
    };
    use crate::FromSmallInt;

    #[test]
    fn dense_mat_vec_matches_schoolbook_q32_d64() {
        type F = Fp64<4294967197>;
        const D: usize = 64;
        let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        let coeffs = std::array::from_fn(|k| {
                            F::from_u64((i as u64 * 10_000 + j as u64 * 100 + k as u64 + 1) % 97)
                        });
                        CyclotomicRing::from_coefficients(coeffs)
                    })
                    .collect()
            })
            .collect();
        let vec: Vec<CyclotomicRing<F, D>> = (0..4)
            .map(|j| {
                let coeffs =
                    std::array::from_fn(|k| F::from_u64((j as u64 * 50 + k as u64 + 3) % 89));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();

        let schoolbook = mat_vec_mul_unchecked(&mat, &vec);
        let crt_ntt = mat_vec_mul_crt_ntt(&mat, &vec).expect("Q32 dispatch should succeed");
        assert_eq!(schoolbook, crt_ntt);
    }

    #[test]
    fn dense_mat_vec_matches_schoolbook_q64_dispatch_for_large_d() {
        type F = Fp64<4294967197>;
        const D: usize = 128;
        let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..2)
            .map(|i| {
                (0..2)
                    .map(|j| {
                        let coeffs = std::array::from_fn(|k| {
                            F::from_u64((i as u64 * 20_000 + j as u64 * 300 + k as u64 + 7) % 113)
                        });
                        CyclotomicRing::from_coefficients(coeffs)
                    })
                    .collect()
            })
            .collect();
        let vec: Vec<CyclotomicRing<F, D>> = (0..2)
            .map(|j| {
                let coeffs =
                    std::array::from_fn(|k| F::from_u64((j as u64 * 70 + k as u64 + 11) % 101));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();

        let schoolbook = mat_vec_mul_unchecked(&mat, &vec);
        let crt_ntt = mat_vec_mul_crt_ntt(&mat, &vec).expect("Q64 dispatch should succeed");
        assert_eq!(schoolbook, crt_ntt);
    }

    #[test]
    fn dense_mat_vec_many_matches_individual_crt_ntt_q32_d64() {
        type F = Fp64<4294967197>;
        const D: usize = 64;
        let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        let coeffs = std::array::from_fn(|k| {
                            F::from_u64((i as u64 * 10_000 + j as u64 * 100 + k as u64 + 1) % 97)
                        });
                        CyclotomicRing::from_coefficients(coeffs)
                    })
                    .collect()
            })
            .collect();

        let vecs: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
            .map(|seed| {
                (0..4)
                    .map(|j| {
                        let coeffs = std::array::from_fn(|k| {
                            F::from_u64((seed as u64 * 700 + j as u64 * 50 + k as u64 + 3) % 89)
                        });
                        CyclotomicRing::from_coefficients(coeffs)
                    })
                    .collect()
            })
            .collect();

        let expected: Vec<Vec<CyclotomicRing<F, D>>> = vecs
            .iter()
            .map(|v| mat_vec_mul_crt_ntt(&mat, v).expect("single CRT+NTT mat-vec should succeed"))
            .collect();

        let got =
            mat_vec_mul_crt_ntt_many(&mat, &vecs).expect("batched CRT+NTT mat-vec should succeed");
        assert_eq!(expected, got);
    }

    #[test]
    fn mat_vec_mul_digits_i8_matches_num_digits_one_roundtrip() {
        type F = Fp64<4294967197>;
        const D: usize = 64;
        let log_basis = 3;

        let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
            .map(|i| {
                (0..6)
                    .map(|j| {
                        let coeffs = std::array::from_fn(|k| {
                            let raw = (i as i64 * 19 + j as i64 * 7 + k as i64) % 7;
                            F::from_i64(raw - 3)
                        });
                        CyclotomicRing::from_coefficients(coeffs)
                    })
                    .collect()
            })
            .collect();

        let digit_blocks: Vec<Vec<[i8; D]>> = vec![
            (0..6)
                .map(|j| std::array::from_fn(|k| ((j + 2 * k) % 7) as i8 - 3))
                .collect(),
            (0..4)
                .map(|j| std::array::from_fn(|k| ((2 * j + k) % 7) as i8 - 3))
                .collect(),
            vec![],
        ];

        let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = digit_blocks
            .iter()
            .map(|block| {
                block
                    .iter()
                    .map(|digit| {
                        let coeffs = std::array::from_fn(|k| F::from_i64(digit[k] as i64));
                        CyclotomicRing::from_coefficients(coeffs)
                    })
                    .collect()
            })
            .collect();

        let ring_block_slices: Vec<&[CyclotomicRing<F, D>]> =
            ring_blocks.iter().map(Vec::as_slice).collect();
        let digit_block_slices: Vec<&[[i8; D]]> = digit_blocks.iter().map(Vec::as_slice).collect();

        match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
            ProtocolCrtNttParams::Q32(params) => {
                let ntt_mat = precompute_dense_mat_ntt_with_params(&mat, &params);
                let via_roundtrip =
                    mat_vec_mul_i8_with_params(&ntt_mat, &ring_block_slices, 1, log_basis, &params);
                let direct =
                    mat_vec_mul_digits_i8_with_params(&ntt_mat, &digit_block_slices, &params);
                assert_eq!(via_roundtrip, direct);
            }
            _ => panic!("unexpected parameter family"),
        }
    }
}
