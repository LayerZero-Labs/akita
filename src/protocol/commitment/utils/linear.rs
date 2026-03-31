//! Linear algebra helpers for ring commitment.

#[cfg(all(target_arch = "aarch64", feature = "parallel"))]
use crate::algebra::ntt::neon;
#[cfg(feature = "parallel")]
use crate::algebra::ntt::MontCoeff;
use crate::algebra::ntt::PrimeWidth;
use crate::algebra::{
    CenteredMontLut, CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, DigitMontLut,
};
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
use crate::error::HachiError;

#[inline(always)]
pub(crate) fn try_centered_i8<F: CanonicalField>(coeff: F, q: u128, half_q: u128) -> Option<i8> {
    let canonical = coeff.to_canonical_u128();
    let centered = if canonical > half_q {
        -((q - canonical) as i128)
    } else {
        canonical as i128
    };
    if (i8::MIN as i128..=i8::MAX as i128).contains(&centered) {
        Some(centered as i8)
    } else {
        None
    }
}

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
    acc.add_assign_pointwise_mul_with_params(lhs, rhs, params);
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
    cfg_iter!(mat)
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

/// Like [`unreduced_quotient_rows_ntt_cached`] but accepts centered i32
/// coefficient rows instead of field-backed ring elements.
#[tracing::instrument(skip_all, name = "unreduced_quotient_rows_ntt_cached_centered_i32")]
pub fn unreduced_quotient_rows_ntt_cached_centered_i32<
    F: FieldCore + CanonicalField,
    const D: usize,
>(
    slot: &NttSlotCache<D>,
    vec: &[[i32; D]],
    max_abs: u32,
) -> Vec<CyclotomicRing<F, D>> {
    match slot {
        NttSlotCache::Q32 {
            neg,
            cyc,
            params: p,
        } => quotient_single_centered_i32_with_params(neg, cyc, vec, max_abs, p),
        NttSlotCache::Q64 {
            neg,
            cyc,
            params: p,
        } => quotient_single_centered_i32_with_params(neg, cyc, vec, max_abs, p),
        NttSlotCache::Q128 {
            neg,
            cyc,
            params: p,
        } => quotient_single_centered_i32_with_params(neg, cyc, vec, max_abs, p),
    }
}

macro_rules! dispatch_slot {
    ($slot:expr, $num_rows:expr, $func:ident $(, $arg:expr)*) => {{
        let n = $num_rows;
        match $slot {
            NttSlotCache::Q32 { neg, params: p, .. } => $func(&neg[..n], $($arg,)* p),
            NttSlotCache::Q64 { neg, params: p, .. } => $func(&neg[..n], $($arg,)* p),
            NttSlotCache::Q128 { neg, params: p, .. } => $func(&neg[..n], $($arg,)* p),
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

/// Decompose each ring element in `rows` into `[i8; D]` digit planes.
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

#[inline]
fn is_zero_centered_row<const D: usize>(row: &[i32; D]) -> bool {
    row.iter().all(|&d| d == 0)
}

#[cfg(target_arch = "aarch64")]
const TARGET_L2_CACHE_BYTES: usize = 4 * 1024 * 1024;
#[cfg(target_arch = "x86_64")]
const TARGET_L2_CACHE_BYTES: usize = 1024 * 1024;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const TARGET_L2_CACHE_BYTES: usize = 1024 * 1024;
const CENTERED_LUT_MAX_ABS: u32 = (1 << 16) - 1;

#[cfg(feature = "parallel")]
#[inline]
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
    num_rows: usize,
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    dispatch_slot!(
        slot,
        num_rows,
        mat_vec_mul_i8_with_params,
        blocks,
        num_digits,
        log_basis
    )
}

/// Strided variant of [`mat_vec_mul_ntt_i8`] for recursive witnesses.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_i8_strided")]
pub fn mat_vec_mul_ntt_i8_strided<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    coeffs: &[CyclotomicRing<F, D>],
    num_blocks: usize,
    block_len: usize,
    num_digits: usize,
    log_basis: u32,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    dispatch_slot!(
        slot,
        num_rows,
        mat_vec_mul_i8_strided_with_params,
        coeffs,
        num_blocks,
        block_len,
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
    num_rows: usize,
    blocks: &[&[[i8; D]]],
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    dispatch_slot!(slot, num_rows, mat_vec_mul_digits_i8_with_params, blocks)
}

/// Strided variant of [`mat_vec_mul_ntt_digits_i8`] for recursive witnesses.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_digits_i8_strided")]
pub fn mat_vec_mul_ntt_digits_i8_strided<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    dispatch_slot!(
        slot,
        num_rows,
        mat_vec_mul_digits_i8_strided_with_params,
        coeffs,
        num_blocks,
        block_len
    )
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
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let max_data_width = blocks.iter().map(|b| b.len()).max().unwrap_or(0);
    let inner_width = mat_width.min(max_data_width);
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks];
    }

    if n_a <= 2 && num_blocks >= 16 {
        return mat_vec_mul_digits_i8_block_parallel(ntt_mat, blocks, params);
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

fn mat_vec_mul_digits_i8_strided_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    if num_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let inner_width = mat_width.min(block_len);
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks];
    }

    if n_a <= 2 && num_blocks >= 16 {
        return mat_vec_mul_digits_i8_strided_block_parallel(
            ntt_mat,
            coeffs,
            num_blocks,
            inner_width,
            params,
        );
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

            for col in tile_start..tile_end {
                let seq_start = col * num_blocks;
                if seq_start >= coeffs.len() {
                    break;
                }
                let live_blocks = num_blocks.min(coeffs.len() - seq_start);
                let coeffs_for_col = &coeffs[seq_start..seq_start + live_blocks];
                for (block_idx, digit) in coeffs_for_col.iter().enumerate() {
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                    for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
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

/// Block-parallel fast path for small `n_a` and many blocks.
///
/// Parallelizes over blocks (high fanout) instead of column tiles (low fanout).
/// With many blocks but few matrix rows, the old tile-based approach had limited
/// parallelism (few tiles) while this path gives num_blocks-way parallelism.
fn mat_vec_mul_digits_i8_block_parallel<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    blocks: &[&[[i8; D]]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let n_a = ntt_mat.len();
    let lut = DigitMontLut::new(params);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut accs: Vec<CyclotomicCrtNtt<W, K, D>> =
                vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];

            for (j, digit) in block.iter().enumerate() {
                if is_zero_plane(digit) {
                    continue;
                }
                let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                    accumulate_pointwise_product_into(acc, &mat_row[j], &ntt_d, params);
                }
            }

            accs.into_iter()
                .map(|acc| acc.to_ring_with_params(params))
                .collect()
        })
        .collect()
}

#[allow(dead_code)]
fn mat_vec_mul_digits_i8_strided_block_parallel<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let n_a = ntt_mat.len();
    let lut = DigitMontLut::new(params);

    cfg_into_iter!(0..num_blocks)
        .map(|block_idx| {
            let mut accs: Vec<CyclotomicCrtNtt<W, K, D>> =
                vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];

            for col in 0..block_len {
                let seq = block_idx + col * num_blocks;
                let Some(digit) = coeffs.get(seq) else {
                    break;
                };
                if is_zero_plane(digit) {
                    continue;
                }
                let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                    accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                }
            }

            accs.into_iter()
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
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let max_data_width = blocks
        .iter()
        .map(|b| b.len() * num_digits)
        .max()
        .unwrap_or(0);
    let inner_width = mat_width.min(max_data_width);
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

fn mat_vec_mul_i8_strided_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    coeffs: &[CyclotomicRing<F, D>],
    num_blocks: usize,
    block_len: usize,
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    if num_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let inner_width = mat_width.min(block_len.saturating_mul(num_digits));
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

            for (block_idx, block_accs) in accs.iter_mut().enumerate() {
                let mut partial_coeffs = Vec::with_capacity(ring_end.saturating_sub(ring_start));
                for col in ring_start..ring_end {
                    let seq = block_idx + col * num_blocks;
                    let Some(coeff) = coeffs.get(seq) else {
                        break;
                    };
                    partial_coeffs.push(*coeff);
                }
                if partial_coeffs.is_empty() {
                    continue;
                }

                let all_digits = decompose_block_i8(&partial_coeffs, num_digits, log_basis);
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
                    for (acc, mat_row) in block_accs.iter_mut().zip(ntt_mat.iter()) {
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
    num_rows: usize,
    vec: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>> {
    match slot {
        NttSlotCache::Q32 { neg, params: p, .. } => {
            mat_vec_mul_single_i8_with_params(&neg[..num_rows], vec, p)
        }
        NttSlotCache::Q64 { neg, params: p, .. } => {
            mat_vec_mul_single_i8_with_params(&neg[..num_rows], vec, p)
        }
        NttSlotCache::Q128 { neg, params: p, .. } => {
            mat_vec_mul_single_i8_with_params(&neg[..num_rows], vec, p)
        }
    }
}

/// Cyclic-domain variant of [`mat_vec_mul_ntt_single_i8`].
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_single_i8_cyclic")]
pub fn mat_vec_mul_ntt_single_i8_cyclic<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    vec: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>> {
    match slot {
        NttSlotCache::Q32 { cyc, params: p, .. } => {
            mat_vec_mul_single_i8_cyclic_with_params(&cyc[..num_rows], vec, p)
        }
        NttSlotCache::Q64 { cyc, params: p, .. } => {
            mat_vec_mul_single_i8_cyclic_with_params(&cyc[..num_rows], vec, p)
        }
        NttSlotCache::Q128 { cyc, params: p, .. } => {
            mat_vec_mul_single_i8_cyclic_with_params(&cyc[..num_rows], vec, p)
        }
    }
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

#[cfg(test)]
fn block_plane_len<const D: usize>(blocks: &[Vec<[i8; D]>], inner_width: usize) -> usize {
    blocks
        .iter()
        .fold(0usize, |acc, block| acc.saturating_add(block.len()))
        .min(inner_width)
}

#[cfg(test)]
fn mat_vec_mul_single_i8_blocks_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    blocks: &[Vec<[i8; D]>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let n_a = ntt_mat.len();
    let inner_width = ntt_mat.first().map_or(0, |row| row.len());
    if inner_width == 0 || n_a == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); n_a];
    }

    let lut = DigitMontLut::new(params);
    let vec_len = block_plane_len(blocks, inner_width);
    let tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let num_tiles = vec_len.div_ceil(tw);

    let final_accs: Vec<CyclotomicCrtNtt<W, K, D>> = cfg_fold_reduce!(
        0..num_tiles,
        || vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a],
        |mut accs: Vec<CyclotomicCrtNtt<W, K, D>>, tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(vec_len);
            let mut global_col = 0usize;
            for block in blocks {
                if global_col >= tile_end {
                    break;
                }
                let block_end = global_col.saturating_add(block.len()).min(vec_len);
                if block_end <= tile_start {
                    global_col = global_col.saturating_add(block.len());
                    continue;
                }
                let local_start = tile_start.saturating_sub(global_col);
                let local_end = block_end - global_col;
                for (local_idx, digit) in block[local_start..local_end].iter().enumerate() {
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let col = global_col + local_start + local_idx;
                    let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                    for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                    }
                }
                global_col = global_col.saturating_add(block.len());
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

fn mat_vec_mul_single_i8_cyclic_with_params<
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
                let ntt_d = CyclotomicCrtNtt::from_i8_cyclic_with_lut(digit, params, &lut);
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
        .map(|acc| acc.to_ring_cyclic(params))
        .collect()
}

#[cfg(test)]
fn mat_vec_mul_single_i8_cyclic_blocks_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    blocks: &[Vec<[i8; D]>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let n_a = ntt_mat.len();
    let inner_width = ntt_mat.first().map_or(0, |row| row.len());
    if inner_width == 0 || n_a == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); n_a];
    }

    let lut = DigitMontLut::new(params);
    let vec_len = block_plane_len(blocks, inner_width);
    let tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let num_tiles = vec_len.div_ceil(tw);

    let final_accs: Vec<CyclotomicCrtNtt<W, K, D>> = cfg_fold_reduce!(
        0..num_tiles,
        || vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a],
        |mut accs: Vec<CyclotomicCrtNtt<W, K, D>>, tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(vec_len);
            let mut global_col = 0usize;
            for block in blocks {
                if global_col >= tile_end {
                    break;
                }
                let block_end = global_col.saturating_add(block.len()).min(vec_len);
                if block_end <= tile_start {
                    global_col = global_col.saturating_add(block.len());
                    continue;
                }
                let local_start = tile_start.saturating_sub(global_col);
                let local_end = block_end - global_col;
                for (local_idx, digit) in block[local_start..local_end].iter().enumerate() {
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let col = global_col + local_start + local_idx;
                    let ntt_d = CyclotomicCrtNtt::from_i8_cyclic_with_lut(digit, params, &lut);
                    for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                    }
                }
                global_col = global_col.saturating_add(block.len());
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
        .map(|acc| acc.to_ring_cyclic(params))
        .collect()
}

fn quotient_single_centered_i32_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_neg: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    ntt_cyc: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    vec: &[[i32; D]],
    max_abs: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let n_a = ntt_neg.len();
    let inner_width = ntt_neg.first().map_or(0, |row| row.len());
    if inner_width == 0 || n_a == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); n_a];
    }

    let vec_len = vec.len().min(inner_width);
    let tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let num_tiles = vec_len.div_ceil(tw);
    let zero = CyclotomicCrtNtt::<W, K, D>::zero();
    let centered_lut = (max_abs <= CENTERED_LUT_MAX_ABS)
        .then(|| CenteredMontLut::<W, K>::new(params, max_abs as i32));

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
            for (j, coeffs) in vec[tile_start..tile_end].iter().enumerate() {
                if is_zero_centered_row(coeffs) {
                    continue;
                }
                let (ntt_d_neg, ntt_d_cyc) = if let Some(lut) = centered_lut.as_ref() {
                    CyclotomicCrtNtt::from_centered_i32_pair_with_lut(coeffs, params, lut)
                } else {
                    CyclotomicCrtNtt::from_centered_i32_pair_with_params(coeffs, params)
                };
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

/// Minimum number of Rayon work-units for the fused kernel.
///
/// The fused kernel replaces three separate `cfg_fold_reduce` calls
/// (each creating ~N tiles) with a single call. To preserve the ~3N total
/// work-units that rayon::join previously provided, we enforce at least
/// this many tiles so Rayon's work-stealing keeps all cores busy.
const MIN_FUSED_TILES: usize = 30;

/// Fused column-tiled kernel for the three split-eq mat-vec products.
///
/// Replaces three separate NTT-cached mat-vec calls (D-cyclic, B-cyclic,
/// A-quotient) with a single pass over the shared NTT cache. Within each
/// column tile, cache entries are loaded once and reused across all three
/// products with their exact row bounds, eliminating redundant DRAM reads.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fused_split_eq_quotients_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_neg: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    ntt_cyc: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    n_d: usize,
    n_b: usize,
    n_a: usize,
    w_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_pre: &[[i32; D]],
    z_pre_max_abs: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> (
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
) {
    let mat_width = ntt_cyc.first().map_or(0, |row| row.len());
    let w_len = w_hat.len().min(mat_width);
    let t_len = t_hat.len().min(mat_width);
    let z_len = z_pre.len().min(mat_width);
    let max_col = w_len.max(t_len).max(z_len);

    if max_col == 0 {
        return (
            vec![CyclotomicRing::<F, D>::zero(); n_d],
            vec![CyclotomicRing::<F, D>::zero(); n_b],
            vec![CyclotomicRing::<F, D>::zero(); n_a],
        );
    }

    let lut = DigitMontLut::new(params);
    let centered_lut = (z_pre_max_abs <= CENTERED_LUT_MAX_ABS)
        .then(|| CenteredMontLut::<W, K>::new(params, z_pre_max_abs as i32));

    let base_tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let tw = base_tw.min(max_col.div_ceil(MIN_FUSED_TILES).max(1));
    let num_tiles = max_col.div_ceil(tw);
    let zero = CyclotomicCrtNtt::<W, K, D>::zero();

    let (d_accs, b_accs, a_neg_accs, a_cyc_accs) = cfg_fold_reduce!(
        0..num_tiles,
        || (
            vec![zero.clone(); n_d],
            vec![zero.clone(); n_b],
            vec![zero.clone(); n_a],
            vec![zero.clone(); n_a],
        ),
        |mut accs: (
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
        ),
         tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(max_col);

            for j in tile_start..tile_end {
                if j < w_len && !is_zero_plane(&w_hat[j]) {
                    let ntt_w = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&w_hat[j], params, &lut);
                    for (acc_d, cyc_row) in accs.0.iter_mut().zip(ntt_cyc.iter()) {
                        accumulate_pointwise_product_into(acc_d, &cyc_row[j], &ntt_w, params);
                    }
                }

                if j < t_len && !is_zero_plane(&t_hat[j]) {
                    let ntt_t = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&t_hat[j], params, &lut);
                    for (acc_b, cyc_row) in accs.1.iter_mut().zip(ntt_cyc.iter()) {
                        accumulate_pointwise_product_into(acc_b, &cyc_row[j], &ntt_t, params);
                    }
                }

                if j < z_len && !is_zero_centered_row(&z_pre[j]) {
                    let (ntt_z_neg, ntt_z_cyc) = if let Some(ref clut) = centered_lut {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_lut(&z_pre[j], params, clut)
                    } else {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_params(&z_pre[j], params)
                    };
                    for ((acc_neg, acc_cyc), (neg_row, cyc_row)) in accs
                        .2
                        .iter_mut()
                        .zip(accs.3.iter_mut())
                        .zip(ntt_neg.iter().zip(ntt_cyc.iter()))
                    {
                        accumulate_pointwise_product_into(acc_neg, &neg_row[j], &ntt_z_neg, params);
                        accumulate_pointwise_product_into(acc_cyc, &cyc_row[j], &ntt_z_cyc, params);
                    }
                }
            }
            accs
        },
        |mut a: (
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
        ),
         b| {
            for r in 0..n_d {
                add_ntt_into(&mut a.0[r], &b.0[r], params);
            }
            for r in 0..n_b {
                add_ntt_into(&mut a.1[r], &b.1[r], params);
            }
            for r in 0..n_a {
                add_ntt_into(&mut a.2[r], &b.2[r], params);
                add_ntt_into(&mut a.3[r], &b.3[r], params);
            }
            a
        }
    );

    let d_result = d_accs
        .into_iter()
        .map(|acc| acc.to_ring_cyclic(params))
        .collect();

    let b_result = b_accs
        .into_iter()
        .map(|acc| acc.to_ring_cyclic(params))
        .collect();

    let a_result = a_neg_accs
        .into_iter()
        .zip(a_cyc_accs)
        .map(|(neg_acc, cyc_acc)| {
            let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring_with_params(params);
            let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
            let neg_c = neg_ring.coefficients();
            let cyc_c = cyc_ring.coefficients();
            let q: [F; D] = from_fn(|k| (cyc_c[k] - neg_c[k]) * F::TWO_INV);
            CyclotomicRing::from_coefficients(q)
        })
        .collect();

    (d_result, b_result, a_result)
}

/// Fused split-eq quotient kernel dispatching over [`NttSlotCache`] variants.
///
/// Computes three NTT-cached mat-vec products in a single tiled pass:
/// - D-cyclic: `cyc[0..n_d] · w_hat` (cyclic domain)
/// - B-cyclic: `cyc[0..n_b] · t_hat` (cyclic domain)
/// - A-quotient: `(cyc[0..n_a]·z_cyc − neg[0..n_a]·z_neg) / 2`
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "fused_split_eq_quotients")]
pub fn fused_split_eq_quotients<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    w_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_pre: &[[i32; D]],
    z_pre_max_abs: u32,
) -> (
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
) {
    match slot {
        NttSlotCache::Q32 {
            neg,
            cyc,
            params: p,
        } => fused_split_eq_quotients_with_params(
            neg,
            cyc,
            n_d,
            n_b,
            n_a,
            w_hat,
            t_hat,
            z_pre,
            z_pre_max_abs,
            p,
        ),
        NttSlotCache::Q64 {
            neg,
            cyc,
            params: p,
        } => fused_split_eq_quotients_with_params(
            neg,
            cyc,
            n_d,
            n_b,
            n_a,
            w_hat,
            t_hat,
            z_pre,
            z_pre_max_abs,
            p,
        ),
        NttSlotCache::Q128 {
            neg,
            cyc,
            params: p,
        } => fused_split_eq_quotients_with_params(
            neg,
            cyc,
            n_d,
            n_b,
            n_a,
            w_hat,
            t_hat,
            z_pre,
            z_pre_max_abs,
            p,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        flatten_i8_blocks, mat_vec_mul_crt_ntt, mat_vec_mul_crt_ntt_many,
        mat_vec_mul_digits_i8_with_params, mat_vec_mul_i8_with_params,
        mat_vec_mul_single_i8_blocks_with_params, mat_vec_mul_single_i8_cyclic_blocks_with_params,
        mat_vec_mul_single_i8_cyclic_with_params, mat_vec_mul_single_i8_with_params,
        mat_vec_mul_unchecked, precompute_dense_mat_ntt_with_params,
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

    #[test]
    fn single_i8_block_matvec_matches_flat_q32_d64() {
        type F = Fp64<4294967197>;
        const D: usize = 64;
        let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..2)
            .map(|i| {
                (0..7)
                    .map(|j| {
                        CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                            let raw = ((13 * i as i64 + 7 * j as i64 + k as i64) % 11) - 5;
                            F::from_i64(raw)
                        }))
                    })
                    .collect()
            })
            .collect();
        let blocks: Vec<Vec<[i8; D]>> = vec![
            (0..3)
                .map(|j| std::array::from_fn(|k| ((2 * j + k) % 7) as i8 - 3))
                .collect(),
            (0..2)
                .map(|j| std::array::from_fn(|k| ((3 * j + 2 * k) % 9) as i8 - 4))
                .collect(),
            (0..2)
                .map(|j| std::array::from_fn(|k| ((j + 4 * k) % 5) as i8 - 2))
                .collect(),
        ];
        let flat = flatten_i8_blocks(&blocks);

        match select_crt_ntt_params::<F, D>().expect("CRT+NTT params should exist") {
            ProtocolCrtNttParams::Q32(params) => {
                let ntt_mat = precompute_dense_mat_ntt_with_params(&mat, &params);
                let expected: Vec<CyclotomicRing<F, D>> =
                    mat_vec_mul_single_i8_with_params(&ntt_mat, &flat, &params);
                let blockwise: Vec<CyclotomicRing<F, D>> =
                    mat_vec_mul_single_i8_blocks_with_params(&ntt_mat, &blocks, &params);
                let expected_cyc: Vec<CyclotomicRing<F, D>> =
                    mat_vec_mul_single_i8_cyclic_with_params(&ntt_mat, &flat, &params);
                let blockwise_cyc: Vec<CyclotomicRing<F, D>> =
                    mat_vec_mul_single_i8_cyclic_blocks_with_params(&ntt_mat, &blocks, &params);
                assert_eq!(blockwise, expected);
                assert_eq!(blockwise_cyc, expected_cyc);
            }
            _ => panic!("unexpected parameter family"),
        }
    }
}
