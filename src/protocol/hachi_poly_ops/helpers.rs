//! Shared internal helpers for the decompose-fold and commit-inner pipelines.
//!
//! Contains balanced-digit decomposition, sparse multiply-accumulate kernels,
//! position-partitioned accumulation strategies, and the final witness
//! construction used by all three [`super::HachiPolyOps`] implementations.

use crate::algebra::ring::sparse_challenge::SparseChallenge;
use crate::algebra::CyclotomicRing;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::onehot::{RegularOneHotEntry, SparseBlockEntry};
use crate::protocol::hachi_poly_ops::DecomposeFoldWitness;
use crate::{cfg_into_iter, cfg_iter, CanonicalField};
use std::array::from_fn;

#[cfg(target_arch = "aarch64")]
use crate::algebra::ntt::neon;

#[cfg(target_arch = "aarch64")]
use super::decompose_fold_neon;

pub(super) struct DecomposeParams {
    pub half_q: u128,
    pub q: u128,
    pub mask: i128,
    pub half_b: i128,
    pub b_val: i128,
    pub log_basis: u32,
}

/// Decompose all D coefficients of a ring element into balanced base-b digits,
/// storing results in digit-major order for subsequent SIMD scatter.
///
/// Uses K=3 interleaved carry chains to saturate ALU throughput (3x ILP gain
/// over processing one coefficient at a time on out-of-order cores).
///
/// `digit_buf` is `[num_digits][D]` in i8, OVERWRITTEN (not accumulated).
#[inline(never)]
pub(super) fn decompose_ring_interleaved<F: CanonicalField, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    digit_buf: &mut [Vec<i8>],
    num_digits: usize,
    p: &DecomposeParams,
) {
    let bulk_end = D - (D % 3);

    for base in (0..bulk_end).step_by(3) {
        let mut c0 = to_signed(ring.coeffs[base].to_canonical_u128(), p);
        let mut c1 = to_signed(ring.coeffs[base + 1].to_canonical_u128(), p);
        let mut c2 = to_signed(ring.coeffs[base + 2].to_canonical_u128(), p);

        for plane in digit_buf.iter_mut().take(num_digits) {
            let d0 = extract_balanced_digit(&mut c0, p);
            let d1 = extract_balanced_digit(&mut c1, p);
            let d2 = extract_balanced_digit(&mut c2, p);
            plane[base] = d0 as i8;
            plane[base + 1] = d1 as i8;
            plane[base + 2] = d2 as i8;
        }
    }

    for idx in bulk_end..D {
        let mut c = to_signed(ring.coeffs[idx].to_canonical_u128(), p);
        for plane in digit_buf.iter_mut().take(num_digits) {
            plane[idx] = extract_balanced_digit(&mut c, p) as i8;
        }
    }
}

#[inline(never)]
pub(super) fn decompose_ring_single_digit<F: CanonicalField, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    digit_plane: &mut [i8; D],
    p: &DecomposeParams,
) {
    for (dst, coeff) in digit_plane.iter_mut().zip(ring.coeffs.iter()) {
        let centered = to_signed(coeff.to_canonical_u128(), p);
        debug_assert!(
            centered >= -(1i128 << (p.log_basis - 1)) && centered < (1i128 << (p.log_basis - 1))
        );
        *dst = centered as i8;
    }
}

#[inline(always)]
fn to_signed(canonical: u128, p: &DecomposeParams) -> i128 {
    if canonical > p.half_q {
        -((p.q - canonical) as i128)
    } else {
        canonical as i128
    }
}

#[inline(always)]
pub(super) fn try_centered_i8<F: CanonicalField>(coeff: F, q: u128, half_q: u128) -> Option<i8> {
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

pub(super) fn try_small_i8_cache_from_ring_coeffs<F: CanonicalField, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
) -> Option<Vec<[i8; D]>> {
    let q = (-F::one()).to_canonical_u128() + 1;
    let half_q = q / 2;
    let mut out = Vec::with_capacity(coeffs.len());

    for ring in coeffs {
        let mut digits = [0i8; D];
        for (dst, coeff) in digits.iter_mut().zip(ring.coeffs.iter()) {
            *dst = try_centered_i8(*coeff, q, half_q)?;
        }
        out.push(digits);
    }

    Some(out)
}

#[inline(always)]
fn extract_balanced_digit(c: &mut i128, p: &DecomposeParams) -> i32 {
    let d = *c & p.mask;
    let balanced = if d >= p.half_b { d - p.b_val } else { d };
    *c = (*c - balanced) >> p.log_basis;
    balanced as i32
}

/// Scalar sparse-multiply-accumulate: accumulate `challenge * digit_plane`
/// into `acc` using the rotate-and-add formulation.
///
/// `digit_plane` is `[i8; D]`, `acc` is `[i32; D]`.
/// Each challenge term rotates the digit plane and adds/subtracts contiguously.
#[inline(always)]
fn sparse_mul_acc_add_scalar<const D: usize>(digit_plane: &[i8], acc: &mut [i32; D], p: usize) {
    let split = D - p;
    for i in 0..split {
        acc[i + p] += digit_plane[i] as i32;
    }
    for i in split..D {
        acc[i - split] -= digit_plane[i] as i32;
    }
}

#[inline(always)]
fn sparse_mul_acc_sub_scalar<const D: usize>(digit_plane: &[i8], acc: &mut [i32; D], p: usize) {
    let split = D - p;
    for i in 0..split {
        acc[i + p] -= digit_plane[i] as i32;
    }
    for i in split..D {
        acc[i - split] += digit_plane[i] as i32;
    }
}

pub(super) fn sparse_mul_acc_scalar<const D: usize>(
    digit_plane: &[i8],
    challenge: &SparseChallenge,
    acc: &mut [i32; D],
) {
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let p = pos as usize;
        match coeff {
            1 => sparse_mul_acc_add_scalar::<D>(digit_plane, acc, p),
            -1 => sparse_mul_acc_sub_scalar::<D>(digit_plane, acc, p),
            2 => {
                sparse_mul_acc_add_scalar::<D>(digit_plane, acc, p);
                sparse_mul_acc_add_scalar::<D>(digit_plane, acc, p);
            }
            -2 => {
                sparse_mul_acc_sub_scalar::<D>(digit_plane, acc, p);
                sparse_mul_acc_sub_scalar::<D>(digit_plane, acc, p);
            }
            _ => {
                let split = D - p;
                let c = coeff as i32;
                for i in 0..split {
                    acc[i + p] += c * digit_plane[i] as i32;
                }
                for i in split..D {
                    acc[i - split] -= c * digit_plane[i] as i32;
                }
            }
        }
    }
}

/// Dispatch to NEON or scalar sparse-multiply-accumulate.
#[inline(always)]
pub(super) fn sparse_mul_acc<const D: usize>(
    digit_plane: &[i8],
    challenge: &SparseChallenge,
    acc: &mut [i32; D],
) {
    #[cfg(target_arch = "aarch64")]
    {
        if neon::use_neon_ntt()
            && challenge
                .coeffs
                .iter()
                .all(|&coeff| coeff.unsigned_abs() <= 2)
        {
            unsafe {
                decompose_fold_neon::sparse_mul_acc_neon(
                    digit_plane.as_ptr(),
                    acc.as_mut_ptr(),
                    D,
                    &challenge.positions,
                    &challenge.coeffs,
                );
            }
            return;
        }
    }
    sparse_mul_acc_scalar::<D>(digit_plane, challenge, acc);
}

/// Precompute dense rotation table for a sparse challenge.
///
/// `table[c]` holds the i32 coefficients of `challenge * X^c` in the ring
/// `Z[X]/(X^D + 1)`.  Because D is a power of two, `X^D = -1`, so
/// positions that wrap past D get negated.
///
/// The table is 16 KB for D=64, fitting entirely in L1 cache.
#[inline(always)]
fn fill_rotated_challenge<const D: usize>(table: &mut [[i32; D]], challenge: &SparseChallenge) {
    debug_assert!(D.is_power_of_two());
    debug_assert!(table.len() >= D);

    let mut dense = [0i32; D];
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        dense[pos as usize] = coeff as i32;
    }

    for (ci, row) in table.iter_mut().enumerate().take(D) {
        let split = D - ci;
        row[ci..D].copy_from_slice(&dense[..split]);
        for (dst, src) in row[..ci].iter_mut().zip(dense[split..].iter()) {
            *dst = -*src;
        }
    }
}

/// Position-parallel accumulation for sparse one-hot witnesses.
///
/// Used by [`super::onehot::OneHotPoly::decompose_fold_sparse_onehot`].
pub(super) fn sparse_onehot_accumulate<const D: usize>(
    sparse_blocks: &[Vec<SparseBlockEntry>],
    challenges: &[SparseChallenge],
    num_blocks: usize,
    inner_width: usize,
    num_digits: usize,
) -> Vec<[i32; D]> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let pos_chunk = inner_width.div_ceil(num_threads);

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..num_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            let pos_end = (pos_start + pos_chunk).min(inner_width);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i32; D]; len];
            let mut rotated = vec![[0i32; D]; D];

            for block_idx in 0..num_blocks {
                let entries = &sparse_blocks[block_idx];
                let lo = entries.partition_point(|e| e.pos_in_block * num_digits < pos_start);
                let hi = entries.partition_point(|e| e.pos_in_block * num_digits < pos_end);
                if lo >= hi {
                    continue;
                }

                fill_rotated_challenge::<D>(&mut rotated, &challenges[block_idx]);

                for entry in &entries[lo..hi] {
                    let local_pos = entry.pos_in_block * num_digits - pos_start;
                    for &ci in &entry.nonzero_coeffs {
                        let rot = &rotated[ci];
                        let dst = &mut acc[local_pos];
                        for k in 0..D {
                            dst[k] += rot[k];
                        }
                    }
                }
            }

            acc
        })
        .collect();

    chunks.into_iter().flatten().collect()
}

/// Position-partitioned accumulation for regular one-hot witnesses where each
/// nonzero ring element has exactly one hot coefficient.
pub(super) fn regular_onehot_accumulate<const D: usize>(
    regular_blocks: &[Vec<RegularOneHotEntry>],
    challenges: &[SparseChallenge],
    num_blocks: usize,
    block_len: usize,
) -> Vec<[i32; D]> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len).max(1);
    let pos_chunk = block_len.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            let pos_end = (pos_start + pos_chunk).min(block_len);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i32; D]; len];
            let mut rotated = vec![[0i32; D]; D];

            for block_idx in 0..num_blocks {
                let entries = &regular_blocks[block_idx];
                let lo = entries.partition_point(|entry| entry.pos_in_block() < pos_start);
                let hi = entries.partition_point(|entry| entry.pos_in_block() < pos_end);
                if lo >= hi {
                    continue;
                }

                fill_rotated_challenge::<D>(&mut rotated, &challenges[block_idx]);
                for entry in &entries[lo..hi] {
                    let dst = &mut acc[entry.pos_in_block() - pos_start];
                    let rot = &rotated[entry.coeff_idx()];
                    for k in 0..D {
                        dst[k] += rot[k];
                    }
                }
            }

            acc
        })
        .collect();

    chunks.into_iter().flatten().collect()
}

pub(super) fn signed_accum_to_ring<F: CanonicalField, const D: usize>(
    coeff_accum: [i32; D],
    modulus: u128,
) -> CyclotomicRing<F, D> {
    let coeffs = from_fn(|k| {
        let v = coeff_accum[k];
        if v >= 0 {
            F::from_canonical_u128_reduced(v as u128)
        } else {
            F::from_canonical_u128_reduced(modulus - ((-v) as u128))
        }
    });
    CyclotomicRing::from_coefficients(coeffs)
}

/// Position-partitioned accumulation for
/// [`RecursiveWitnessView::decompose_fold`](super::RecursiveWitnessView).
pub(super) fn balanced_digit_decompose_fold_partitioned<const D: usize>(
    coeffs: &[[i8; D]],
    challenges: &[SparseChallenge],
    active_blocks: usize,
    block_len: usize,
    num_digits: usize,
    inner_width: usize,
) -> Vec<[i32; D]> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(inner_width).max(1);
    let pos_chunk = inner_width.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            let pos_end = (pos_start + pos_chunk).min(inner_width);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i32; D]; len];

            let elem_start = pos_start / num_digits;
            let elem_end = pos_end.div_ceil(num_digits);

            for (challenge, block_idx) in challenges[..active_blocks].iter().zip(0..) {
                let block_start: usize = block_idx * block_len;
                let block_end = (block_start + block_len).min(coeffs.len());

                let lo = elem_start.min(block_end.saturating_sub(block_start));
                let hi = elem_end.min(block_end.saturating_sub(block_start));

                for elem_idx in lo..hi {
                    let out_pos = elem_idx * num_digits;
                    if out_pos >= pos_start && out_pos < pos_end {
                        sparse_mul_acc::<D>(
                            &coeffs[block_start + elem_idx],
                            challenge,
                            &mut acc[out_pos - pos_start],
                        );
                    }
                }
            }
            acc
        })
        .collect();

    chunks.into_iter().flatten().collect()
}

pub(super) fn build_decompose_fold_witness<F: CanonicalField, const D: usize>(
    centered_coeffs: Vec<[i32; D]>,
    modulus: u128,
) -> DecomposeFoldWitness<F, D> {
    let centered_inf_norm = centered_coeffs
        .iter()
        .flat_map(|row| row.iter())
        .map(|coeff| coeff.unsigned_abs())
        .max()
        .unwrap_or(0);
    let z_pre = cfg_iter!(centered_coeffs)
        .map(|coeff_accum| signed_accum_to_ring::<F, D>(*coeff_accum, modulus))
        .collect();
    DecomposeFoldWitness {
        z_pre,
        centered_coeffs,
        centered_inf_norm,
    }
}
