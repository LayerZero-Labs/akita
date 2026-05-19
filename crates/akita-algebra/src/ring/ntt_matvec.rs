//! NTT-based Ajtai matrix-vector kernels shared by prover and verifier.

#[cfg(all(target_arch = "aarch64", feature = "parallel"))]
use crate::ntt::neon;
use crate::ntt::MontCoeff;
use crate::ntt::PrimeWidth;
use crate::ring::crt_ntt_cache::NttSlotCache;
use crate::ring::cyclotomic::BalancedDecomposePow2I8Params;
use crate::ring::{CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, DigitMontLut};
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore};
use std::mem::size_of;

#[cfg(test)]
use crate::ring::crt_ntt_cache::{select_crt_ntt_params, ProtocolCrtNttParams};
#[cfg(test)]
use akita_field::AkitaError;

#[inline]
fn accumulate_pointwise_product_into<W: PrimeWidth, const K: usize, const D: usize>(
    acc: &mut CyclotomicCrtNtt<W, K, D>,
    lhs: &CyclotomicCrtNtt<W, K, D>,
    rhs: &CyclotomicCrtNtt<W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    acc.add_assign_pointwise_mul_with_params(lhs, rhs, params);
}

macro_rules! dispatch_slot {
    ($slot:expr, $num_rows:expr, $num_cols:expr, $func:ident $(, $arg:expr)*) => {{
        let nr: usize = $num_rows;
        let nc: usize = $num_cols;
        match $slot {
            NttSlotCache::Q32 { neg, params: p, .. } => {
                let rows: Vec<&[_]> = (0..nr).map(|i| &neg[i * nc..(i + 1) * nc]).collect();
                $func(&rows, $($arg,)* p)
            }
            NttSlotCache::Q64 { neg, params: p, .. } => {
                let rows: Vec<&[_]> = (0..nr).map(|i| &neg[i * nc..(i + 1) * nc]).collect();
                $func(&rows, $($arg,)* p)
            }
            NttSlotCache::Q128 { neg, params: p, .. } => {
                let rows: Vec<&[_]> = (0..nr).map(|i| &neg[i * nc..(i + 1) * nc]).collect();
                $func(&rows, $($arg,)* p)
            }
        }
    }};
}

/// Flatten nested digit blocks into one contiguous vector.
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
    let mut out = vec![[0i8; D]; block.len() * num_digits];
    decompose_rows_i8_into(block, &mut out, num_digits, log_basis);
    out
}

/// Decompose each ring element in `rows` into `[i8; D]` digit planes.
pub fn decompose_rows_i8<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<[i8; D]> {
    let mut out = vec![[0i8; D]; rows.len() * num_digits];
    decompose_rows_i8_into(rows, &mut out, num_digits, log_basis);
    out
}

/// Decompose each ring element in `rows` into a preallocated flat digit buffer.
///
/// # Panics
///
/// Panics if `out.len() != rows.len() * num_digits`.
pub fn decompose_rows_i8_into<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    out: &mut [[i8; D]],
    num_digits: usize,
    log_basis: u32,
) {
    assert_eq!(
        out.len(),
        rows.len() * num_digits,
        "flat digit output length must match rows * num_digits",
    );
    if num_digits == 0 {
        return;
    }
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

    #[cfg(feature = "parallel")]
    out.par_chunks_mut(num_digits)
        .zip(rows.par_iter())
        .for_each(|(dst_chunk, row)| {
            row.balanced_decompose_pow2_i8_into_with_params(dst_chunk, &decompose_params)
        });

    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(num_digits)
        .zip(rows.iter())
        .for_each(|(dst_chunk, row)| {
            row.balanced_decompose_pow2_i8_into_with_params(dst_chunk, &decompose_params)
        });
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
const SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS: usize = 4;
const SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS: usize = 16;

#[inline]
fn aligned_i8_tile_width(raw_width: usize, inner_width: usize, num_digits: usize) -> usize {
    debug_assert!(inner_width > 0);
    debug_assert!(num_digits > 0);

    if inner_width <= num_digits {
        return inner_width;
    }

    let clamped = raw_width.min(inner_width).max(num_digits);
    ((clamped / num_digits).max(1)) * num_digits
}

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
    num_cols: usize,
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_i8_with_params,
        blocks,
        num_digits,
        log_basis
    )
}

/// Dense-optimized variant of [`mat_vec_mul_ntt_i8`].
///
/// Skips the full-plane zero scans that are useful for sparse inputs but are
/// almost always wasted work on dense witnesses.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_i8_dense")]
pub fn mat_vec_mul_ntt_i8_dense<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_i8_dense_with_params,
        blocks,
        num_digits,
        log_basis
    )
}

/// Single-row dense variant of [`mat_vec_mul_ntt_i8_dense`].
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_i8_dense_single_row")]
pub fn mat_vec_mul_ntt_i8_dense_single_row<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_cols: usize,
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
) -> Vec<CyclotomicRing<F, D>> {
    dispatch_slot!(
        slot,
        1usize,
        num_cols,
        mat_vec_mul_i8_dense_single_row_with_params,
        blocks,
        num_digits,
        log_basis
    )
}

/// Strided variant of [`mat_vec_mul_ntt_i8`] for recursive witnesses.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_i8_strided")]
pub fn mat_vec_mul_ntt_i8_strided<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    coeffs: &[CyclotomicRing<F, D>],
    num_blocks: usize,
    block_len: usize,
    num_digits: usize,
    log_basis: u32,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    dispatch_slot!(
        slot,
        num_rows,
        num_cols,
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
    num_cols: usize,
    blocks: &[&[[i8; D]]],
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_digits_i8_with_params,
        blocks
    )
}

/// Strided variant of [`mat_vec_mul_ntt_digits_i8`] for recursive witnesses.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_digits_i8_strided")]
pub fn mat_vec_mul_ntt_digits_i8_strided<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    dispatch_slot!(
        slot,
        num_rows,
        num_cols,
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
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
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

    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
    {
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
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
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

    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
    {
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
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    if ntt_mat.len() == 1 {
        return mat_vec_mul_digits_i8_single_row_block_parallel(ntt_mat, blocks, params)
            .into_iter()
            .map(|ring| vec![ring])
            .collect();
    }
    if ntt_mat.len() == 2 {
        return mat_vec_mul_digits_i8_two_row_block_parallel(ntt_mat, blocks, params);
    }
    if ntt_mat.len() == 3 {
        return mat_vec_mul_digits_i8_three_row_block_parallel(ntt_mat, blocks, params);
    }

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

fn mat_vec_mul_digits_i8_single_row_block_parallel<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    debug_assert_eq!(ntt_mat.len(), 1);
    let lut = DigitMontLut::new(params);
    let mat_row = &ntt_mat[0];

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];

            for (j, digit) in block.iter().enumerate() {
                if is_zero_plane(digit) {
                    continue;
                }
                acc.add_assign_pointwise_mul_i8_with_lut_scratch(
                    &mat_row[j],
                    digit,
                    params,
                    &lut,
                    &mut rhs_scratch,
                );
            }

            acc.to_ring_with_params(params)
        })
        .collect()
}

fn mat_vec_mul_digits_i8_two_row_block_parallel<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    debug_assert_eq!(ntt_mat.len(), 2);
    let lut = DigitMontLut::new(params);
    let mat_row0 = &ntt_mat[0];
    let mat_row1 = &ntt_mat[1];

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut acc0 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc1 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];

            for (j, digit) in block.iter().enumerate() {
                if is_zero_plane(digit) {
                    continue;
                }
                CyclotomicCrtNtt::add_assign_pointwise_mul_i8_pair_with_lut_scratch(
                    [&mut acc0, &mut acc1],
                    [&mat_row0[j], &mat_row1[j]],
                    digit,
                    params,
                    &lut,
                    &mut rhs_scratch,
                );
            }

            vec![
                acc0.to_ring_with_params(params),
                acc1.to_ring_with_params(params),
            ]
        })
        .collect()
}

fn mat_vec_mul_digits_i8_three_row_block_parallel<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    debug_assert_eq!(ntt_mat.len(), 3);
    let lut = DigitMontLut::new(params);
    let mat_row0 = &ntt_mat[0];
    let mat_row1 = &ntt_mat[1];
    let mat_row2 = &ntt_mat[2];

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut acc0 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc1 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc2 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];

            for (j, digit) in block.iter().enumerate() {
                if is_zero_plane(digit) {
                    continue;
                }
                CyclotomicCrtNtt::add_assign_pointwise_mul_i8_triple_with_lut_scratch(
                    [&mut acc0, &mut acc1, &mut acc2],
                    [&mat_row0[j], &mat_row1[j], &mat_row2[j]],
                    digit,
                    params,
                    &lut,
                    &mut rhs_scratch,
                );
            }

            vec![
                acc0.to_ring_with_params(params),
                acc1.to_ring_with_params(params),
                acc2.to_ring_with_params(params),
            ]
        })
        .collect()
}

fn mat_vec_mul_digits_i8_strided_block_parallel<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    if ntt_mat.len() == 1 {
        return mat_vec_mul_digits_i8_single_row_strided_block_parallel(
            ntt_mat, coeffs, num_blocks, block_len, params,
        )
        .into_iter()
        .map(|ring| vec![ring])
        .collect();
    }
    if ntt_mat.len() == 2 {
        return mat_vec_mul_digits_i8_two_row_strided_block_parallel(
            ntt_mat, coeffs, num_blocks, block_len, params,
        );
    }
    if ntt_mat.len() == 3 {
        return mat_vec_mul_digits_i8_three_row_strided_block_parallel(
            ntt_mat, coeffs, num_blocks, block_len, params,
        );
    }

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

fn mat_vec_mul_digits_i8_single_row_strided_block_parallel<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    debug_assert_eq!(ntt_mat.len(), 1);
    let lut = DigitMontLut::new(params);
    let mat_row = &ntt_mat[0];

    cfg_into_iter!(0..num_blocks)
        .map(|block_idx| {
            let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];

            for (col, mat_coeff) in mat_row.iter().take(block_len).enumerate() {
                let seq = block_idx + col * num_blocks;
                let Some(digit) = coeffs.get(seq) else {
                    break;
                };
                if is_zero_plane(digit) {
                    continue;
                }
                acc.add_assign_pointwise_mul_i8_with_lut_scratch(
                    mat_coeff,
                    digit,
                    params,
                    &lut,
                    &mut rhs_scratch,
                );
            }

            acc.to_ring_with_params(params)
        })
        .collect()
}

fn mat_vec_mul_digits_i8_two_row_strided_block_parallel<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    debug_assert_eq!(ntt_mat.len(), 2);
    let lut = DigitMontLut::new(params);
    let mat_row0 = &ntt_mat[0];
    let mat_row1 = &ntt_mat[1];

    cfg_into_iter!(0..num_blocks)
        .map(|block_idx| {
            let mut acc0 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc1 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];

            for (col, (mat_coeff0, mat_coeff1)) in mat_row0
                .iter()
                .zip(mat_row1.iter())
                .take(block_len)
                .enumerate()
            {
                let seq = block_idx + col * num_blocks;
                let Some(digit) = coeffs.get(seq) else {
                    break;
                };
                if is_zero_plane(digit) {
                    continue;
                }
                CyclotomicCrtNtt::add_assign_pointwise_mul_i8_pair_with_lut_scratch(
                    [&mut acc0, &mut acc1],
                    [mat_coeff0, mat_coeff1],
                    digit,
                    params,
                    &lut,
                    &mut rhs_scratch,
                );
            }

            vec![
                acc0.to_ring_with_params(params),
                acc1.to_ring_with_params(params),
            ]
        })
        .collect()
}

fn mat_vec_mul_digits_i8_three_row_strided_block_parallel<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    debug_assert_eq!(ntt_mat.len(), 3);
    let lut = DigitMontLut::new(params);
    let mat_row0 = &ntt_mat[0];
    let mat_row1 = &ntt_mat[1];
    let mat_row2 = &ntt_mat[2];

    cfg_into_iter!(0..num_blocks)
        .map(|block_idx| {
            let mut acc0 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc1 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc2 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];

            for (col, ((mat_coeff0, mat_coeff1), mat_coeff2)) in mat_row0
                .iter()
                .zip(mat_row1.iter())
                .zip(mat_row2.iter())
                .take(block_len)
                .enumerate()
            {
                let seq = block_idx + col * num_blocks;
                let Some(digit) = coeffs.get(seq) else {
                    break;
                };
                if is_zero_plane(digit) {
                    continue;
                }
                CyclotomicCrtNtt::add_assign_pointwise_mul_i8_triple_with_lut_scratch(
                    [&mut acc0, &mut acc1, &mut acc2],
                    [mat_coeff0, mat_coeff1, mat_coeff2],
                    digit,
                    params,
                    &lut,
                    &mut rhs_scratch,
                );
            }

            vec![
                acc0.to_ring_with_params(params),
                acc1.to_ring_with_params(params),
                acc2.to_ring_with_params(params),
            ]
        })
        .collect()
}

fn mat_vec_mul_i8_block_parallel_with_params_impl<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let n_a = ntt_mat.len();
    let lut = DigitMontLut::new(params);
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut accs: Vec<CyclotomicCrtNtt<W, K, D>> =
                vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];
            let mut digit_buf = vec![[0i8; D]; num_digits];
            let mut col = 0usize;

            for coeff_vec in block.iter() {
                coeff_vec
                    .balanced_decompose_pow2_i8_into_with_params(&mut digit_buf, &decompose_params);
                for digit in &digit_buf {
                    if CHECK_ZERO && is_zero_plane(digit) {
                        col += 1;
                        continue;
                    }
                    let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                    for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                    }
                    col += 1;
                }
            }

            accs.into_iter()
                .map(|acc| acc.to_ring_with_params(params))
                .collect()
        })
        .collect()
}

fn mat_vec_mul_i8_block_parallel_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_i8_block_parallel_with_params_impl::<F, W, K, D, true>(
        ntt_mat, blocks, num_digits, log_basis, params,
    )
}

fn mat_vec_mul_i8_dense_block_parallel_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    if ntt_mat.len() == 1 {
        return mat_vec_mul_i8_dense_single_row_with_params(
            ntt_mat, blocks, num_digits, log_basis, params,
        )
        .into_iter()
        .map(|ring| vec![ring])
        .collect();
    }
    if ntt_mat.len() == 2 {
        return mat_vec_mul_i8_dense_two_row_fused_with_params(
            ntt_mat, blocks, num_digits, log_basis, params,
        );
    }
    if ntt_mat.len() == 3 {
        return mat_vec_mul_i8_dense_three_row_fused_with_params(
            ntt_mat, blocks, num_digits, log_basis, params,
        );
    }

    mat_vec_mul_i8_block_parallel_with_params_impl::<F, W, K, D, false>(
        ntt_mat, blocks, num_digits, log_basis, params,
    )
}

fn mat_vec_mul_i8_dense_single_row_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    debug_assert_eq!(ntt_mat.len(), 1);
    let lut = DigitMontLut::new(params);
    let mat_row = &ntt_mat[0];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut digit_buf = vec![[0i8; D]; num_digits];
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];
            let mut col = 0usize;

            for coeff_vec in block.iter() {
                coeff_vec
                    .balanced_decompose_pow2_i8_into_with_params(&mut digit_buf, &decompose_params);
                for digit in &digit_buf {
                    acc.add_assign_pointwise_mul_i8_with_lut_scratch(
                        &mat_row[col],
                        digit,
                        params,
                        &lut,
                        &mut rhs_scratch,
                    );
                    col += 1;
                }
            }

            acc.to_ring_with_params(params)
        })
        .collect()
}

fn mat_vec_mul_i8_dense_two_row_fused_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    debug_assert_eq!(ntt_mat.len(), 2);
    let lut = DigitMontLut::new(params);
    let mat_row0 = &ntt_mat[0];
    let mat_row1 = &ntt_mat[1];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut acc0 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc1 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut digit_buf = vec![[0i8; D]; num_digits];
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];
            let mut col = 0usize;

            for coeff_vec in block.iter() {
                coeff_vec
                    .balanced_decompose_pow2_i8_into_with_params(&mut digit_buf, &decompose_params);
                for digit in &digit_buf {
                    CyclotomicCrtNtt::add_assign_pointwise_mul_i8_pair_with_lut_scratch(
                        [&mut acc0, &mut acc1],
                        [&mat_row0[col], &mat_row1[col]],
                        digit,
                        params,
                        &lut,
                        &mut rhs_scratch,
                    );
                    col += 1;
                }
            }

            vec![
                acc0.to_ring_with_params(params),
                acc1.to_ring_with_params(params),
            ]
        })
        .collect()
}

fn mat_vec_mul_i8_dense_three_row_fused_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    debug_assert_eq!(ntt_mat.len(), 3);
    let lut = DigitMontLut::new(params);
    let mat_row0 = &ntt_mat[0];
    let mat_row1 = &ntt_mat[1];
    let mat_row2 = &ntt_mat[2];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut acc0 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc1 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc2 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut digit_buf = vec![[0i8; D]; num_digits];
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];
            let mut col = 0usize;

            for coeff_vec in block.iter() {
                coeff_vec
                    .balanced_decompose_pow2_i8_into_with_params(&mut digit_buf, &decompose_params);
                for digit in &digit_buf {
                    CyclotomicCrtNtt::add_assign_pointwise_mul_i8_triple_with_lut_scratch(
                        [&mut acc0, &mut acc1, &mut acc2],
                        [&mat_row0[col], &mat_row1[col], &mat_row2[col]],
                        digit,
                        params,
                        &lut,
                        &mut rhs_scratch,
                    );
                    col += 1;
                }
            }

            vec![
                acc0.to_ring_with_params(params),
                acc1.to_ring_with_params(params),
                acc2.to_ring_with_params(params),
            ]
        })
        .collect()
}

fn mat_vec_mul_i8_strided_block_parallel_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[CyclotomicRing<F, D>],
    num_blocks: usize,
    block_len: usize,
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let n_a = ntt_mat.len();
    let lut = DigitMontLut::new(params);
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

    cfg_into_iter!(0..num_blocks)
        .map(|block_idx| {
            let mut accs: Vec<CyclotomicCrtNtt<W, K, D>> =
                vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];
            let mut digit_buf = vec![[0i8; D]; num_digits];
            let mut mat_col = 0usize;

            for col in 0..block_len {
                let seq = block_idx + col * num_blocks;
                let Some(coeff) = coeffs.get(seq) else {
                    break;
                };
                coeff
                    .balanced_decompose_pow2_i8_into_with_params(&mut digit_buf, &decompose_params);
                for digit in &digit_buf {
                    if !is_zero_plane(digit) {
                        let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                        for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                            accumulate_pointwise_product_into(
                                acc,
                                &mat_row[mat_col],
                                &ntt_d,
                                params,
                            );
                        }
                    }
                    mat_col += 1;
                }
            }

            accs.into_iter()
                .map(|acc| acc.to_ring_with_params(params))
                .collect()
        })
        .collect()
}

fn mat_vec_mul_i8_with_params_impl<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
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

    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
    {
        return if CHECK_ZERO {
            mat_vec_mul_i8_block_parallel_with_params(
                ntt_mat, blocks, num_digits, log_basis, params,
            )
        } else {
            mat_vec_mul_i8_dense_block_parallel_with_params(
                ntt_mat, blocks, num_digits, log_basis, params,
            )
        };
    }

    let lut = DigitMontLut::new(params);
    let raw_tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    // Keep full tiles on ring boundaries so on-the-fly decomposition does not
    // re-expand the same ring when adjacent tiles meet mid digit-pack.
    let tw = aligned_i8_tile_width(raw_tw, inner_width, num_digits);
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
                    if CHECK_ZERO && is_zero_plane(digit) {
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
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_i8_with_params_impl::<F, W, K, D, true>(
        ntt_mat, blocks, num_digits, log_basis, params,
    )
}

fn mat_vec_mul_i8_dense_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_i8_with_params_impl::<F, W, K, D, false>(
        ntt_mat, blocks, num_digits, log_basis, params,
    )
}

fn mat_vec_mul_i8_strided_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
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

    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
    {
        return mat_vec_mul_i8_strided_block_parallel_with_params(
            ntt_mat, coeffs, num_blocks, block_len, num_digits, log_basis, params,
        );
    }

    let lut = DigitMontLut::new(params);
    let raw_tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    // Keep full tiles on ring boundaries so on-the-fly decomposition does not
    // re-expand the same ring when adjacent tiles meet mid digit-pack.
    let tw = aligned_i8_tile_width(raw_tw, inner_width, num_digits);
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
    num_cols: usize,
    vec: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>> {
    match slot {
        NttSlotCache::Q32 { neg, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &neg[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_with_params(&rows, vec, p)
        }
        NttSlotCache::Q64 { neg, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &neg[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_with_params(&rows, vec, p)
        }
        NttSlotCache::Q128 { neg, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &neg[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_with_params(&rows, vec, p)
        }
    }
}

/// Cyclic-domain variant of [`mat_vec_mul_ntt_single_i8`].
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_single_i8_cyclic")]
pub fn mat_vec_mul_ntt_single_i8_cyclic<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    vec: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>> {
    match slot {
        NttSlotCache::Q32 { cyc, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &cyc[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_cyclic_with_params(&rows, vec, p)
        }
        NttSlotCache::Q64 { cyc, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &cyc[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_cyclic_with_params(&rows, vec, p)
        }
        NttSlotCache::Q128 { cyc, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &cyc[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_cyclic_with_params(&rows, vec, p)
        }
    }
}

fn mat_vec_mul_single_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    vec: &[[i8; D]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    mat_vec_mul_single_i8_with_params_tile_reduce(ntt_mat, vec, params, true)
}

fn mat_vec_mul_single_i8_cyclic_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    vec: &[[i8; D]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    mat_vec_mul_single_i8_with_params_tile_reduce(ntt_mat, vec, params, false)
}

/// Tiled mat-vec where each tile is reconstructed to `F` and tiles are
/// summed in `F`.
///
/// The L2-tile width `tw` is chosen so a single tile's per-coefficient
/// integer bound `tw · D · |F_max| · |i8_max|` stays well below the
/// CRT product `P` of `params` (`K` ~30-bit primes). Per-tile CRT
/// reconstruction therefore returns the correct mod-`p_F` value, and
/// summing tiles in `F` keeps the running total within `F`'s native
/// modular arithmetic — bypassing the wraparound that an across-tile
/// NTT-domain accumulation would suffer once the full sum exceeds `P`.
/// This is the "option b" fix for CRT overflow in wide mat-vec products
/// (book §5.4 tiered chunks at production parameters).
fn mat_vec_mul_single_i8_with_params_tile_reduce<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    vec: &[[i8; D]],
    params: &CrtNttParamSet<W, K, D>,
    negacyclic: bool,
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

    cfg_fold_reduce!(
        0..num_tiles,
        || vec![CyclotomicRing::<F, D>::zero(); n_a],
        |mut accs: Vec<CyclotomicRing<F, D>>, tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(vec_len);
            let mut tile_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];
            for (j, digit) in vec[tile_start..tile_end].iter().enumerate() {
                if is_zero_plane(digit) {
                    continue;
                }
                let ntt_d = if negacyclic {
                    CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut)
                } else {
                    CyclotomicCrtNtt::from_i8_cyclic_with_lut(digit, params, &lut)
                };
                for (acc, mat_row) in tile_accs.iter_mut().zip(ntt_mat.iter()) {
                    accumulate_pointwise_product_into(
                        acc,
                        &mat_row[tile_start + j],
                        &ntt_d,
                        params,
                    );
                }
            }
            for (acc_field, tile_ntt) in accs.iter_mut().zip(tile_accs.into_iter()) {
                let tile_ring: CyclotomicRing<F, D> = if negacyclic {
                    tile_ntt.to_ring_with_params(params)
                } else {
                    tile_ntt.to_ring_cyclic(params)
                };
                *acc_field += tile_ring;
            }
            accs
        },
        |mut a: Vec<CyclotomicRing<F, D>>, b: Vec<CyclotomicRing<F, D>>| {
            for (a_i, b_i) in a.iter_mut().zip(b.into_iter()) {
                *a_i += b_i;
            }
            a
        }
    )
}
