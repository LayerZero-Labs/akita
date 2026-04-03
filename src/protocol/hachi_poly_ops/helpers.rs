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
use crate::protocol::commitment::utils::linear::try_centered_i8;
use crate::protocol::hachi_poly_ops::DecomposeFoldWitness;
use crate::CanonicalField;
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
    digit_buf: &mut [[i8; D]],
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

#[inline(always)]
fn add_scaled_rotated_row<const D: usize>(acc: &mut [i32; D], row: &[i32; D], scale: i32) {
    match scale {
        1 => {
            for k in 0..D {
                acc[k] += row[k];
            }
        }
        -1 => {
            for k in 0..D {
                acc[k] -= row[k];
            }
        }
        2 => {
            for k in 0..D {
                acc[k] += row[k] << 1;
            }
        }
        -2 => {
            for k in 0..D {
                acc[k] -= row[k] << 1;
            }
        }
        _ => {
            for k in 0..D {
                acc[k] += scale * row[k];
            }
        }
    }
}

fn decompose_ring_full_challenge_accumulate<F: CanonicalField, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    rotated: &[[i32; D]],
    acc: &mut [[i32; D]],
    p: &DecomposeParams,
) {
    let bulk_end = D - (D % 3);

    for base in (0..bulk_end).step_by(3) {
        let mut c0 = to_signed(ring.coeffs[base].to_canonical_u128(), p);
        let mut c1 = to_signed(ring.coeffs[base + 1].to_canonical_u128(), p);
        let mut c2 = to_signed(ring.coeffs[base + 2].to_canonical_u128(), p);
        let rot0 = &rotated[base];
        let rot1 = &rotated[base + 1];
        let rot2 = &rotated[base + 2];

        for plane in acc.iter_mut() {
            let d0 = extract_balanced_digit(&mut c0, p);
            if d0 != 0 {
                add_scaled_rotated_row(plane, rot0, d0);
            }

            let d1 = extract_balanced_digit(&mut c1, p);
            if d1 != 0 {
                add_scaled_rotated_row(plane, rot1, d1);
            }

            let d2 = extract_balanced_digit(&mut c2, p);
            if d2 != 0 {
                add_scaled_rotated_row(plane, rot2, d2);
            }
        }
    }

    for (idx, rot) in rotated.iter().enumerate().take(D).skip(bulk_end) {
        let mut c = to_signed(ring.coeffs[idx].to_canonical_u128(), p);
        for plane in acc.iter_mut() {
            let digit = extract_balanced_digit(&mut c, p);
            if digit != 0 {
                add_scaled_rotated_row(plane, rot, digit);
            }
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

    let actual_threads = num_threads.min(inner_width.max(1));
    let pos_chunk = inner_width.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= inner_width {
                return Vec::new();
            }
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

/// Like [`sparse_onehot_accumulate`], but takes borrowed block slices so callers
/// can batch across many polynomials without cloning sparse block storage.
pub(super) fn sparse_onehot_accumulate_slices<const D: usize>(
    sparse_blocks: &[&[SparseBlockEntry]],
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
                let entries = sparse_blocks[block_idx];
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

/// Like [`regular_onehot_accumulate`], but takes borrowed block slices so callers
/// can batch across many polynomials without cloning regular block storage.
pub(super) fn regular_onehot_accumulate_slices<const D: usize>(
    regular_blocks: &[&[RegularOneHotEntry]],
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
                let entries = regular_blocks[block_idx];
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
    num_blocks: usize,
    num_digits: usize,
    inner_width: usize,
) -> Vec<[i32; D]> {
    debug_assert_eq!(
        num_digits, 1,
        "multi-digit decomposition is not implemented for partitioned accumulation"
    );
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(inner_width).max(1);
    let pos_chunk = inner_width.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= inner_width {
                return Vec::new();
            }
            let pos_end = (pos_start + pos_chunk).min(inner_width);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i32; D]; len];

            let elem_start = pos_start / num_digits;
            let elem_end = pos_end.div_ceil(num_digits);

            let lo = elem_start.min(block_len);
            let hi = elem_end.min(block_len);
            for col in lo..hi {
                let out_pos = col * num_digits;
                if out_pos < pos_start || out_pos >= pos_end {
                    continue;
                }

                let seq_start = col * num_blocks;
                if seq_start >= coeffs.len() {
                    break;
                }
                let available_blocks = active_blocks.min(coeffs.len() - seq_start);
                for (challenge, coeff) in challenges[..available_blocks]
                    .iter()
                    .zip(coeffs[seq_start..seq_start + available_blocks].iter())
                {
                    sparse_mul_acc::<D>(coeff, challenge, &mut acc[out_pos - pos_start]);
                }
            }
            acc
        })
        .collect();

    chunks.into_iter().flatten().collect()
}

/// Element-partitioned accumulation for multi-digit dense witnesses.
///
/// Each worker owns a disjoint element range within the block and accumulates
/// all digit planes for that range across every active challenge block. This
/// avoids the large whole-output reductions in the older block-partitioned
/// path while still decomposing each owned ring element only once per block.
pub(super) fn balanced_ring_decompose_fold_partitioned<F: CanonicalField, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
    challenges: &[SparseChallenge],
    block_len: usize,
    num_digits: usize,
    p: &DecomposeParams,
) -> Vec<[i32; D]> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len.max(1)).max(1);
    let elem_chunk = block_len.div_ceil(actual_threads);
    let use_fused_full_challenge = D == 32
        && !challenges.is_empty()
        && challenges
            .iter()
            .all(|challenge| challenge.positions.len() == D && challenge.coeffs.len() == D);
    let mut out = vec![[0i32; D]; block_len * num_digits];

    #[cfg(feature = "parallel")]
    out.par_chunks_mut(elem_chunk * num_digits)
        .enumerate()
        .for_each(|(tid, acc)| {
            let elem_start = tid * elem_chunk;
            if elem_start >= block_len {
                return;
            }
            let elems_in_chunk = acc.len() / num_digits;
            let elem_end = elem_start + elems_in_chunk;
            let mut digit_buf = (!use_fused_full_challenge).then(|| vec![[0i8; D]; num_digits]);
            let mut rotated = use_fused_full_challenge.then(|| vec![[0i32; D]; D]);

            for (block_idx, challenge) in challenges.iter().enumerate() {
                let block_start = block_idx * block_len;
                if block_start >= coeffs.len() {
                    break;
                }
                let coeff_start = block_start + elem_start;
                if coeff_start >= coeffs.len() {
                    continue;
                }
                let coeff_end = (block_start + elem_end).min(coeffs.len());
                if let Some(rotated) = rotated.as_mut() {
                    fill_rotated_challenge::<D>(rotated, challenge);
                    for (local_elem_idx, ring) in coeffs[coeff_start..coeff_end].iter().enumerate()
                    {
                        let base = local_elem_idx * num_digits;
                        decompose_ring_full_challenge_accumulate::<F, D>(
                            ring,
                            rotated,
                            &mut acc[base..base + num_digits],
                            p,
                        );
                    }
                } else if let Some(digit_buf) = digit_buf.as_mut() {
                    for (local_elem_idx, ring) in coeffs[coeff_start..coeff_end].iter().enumerate()
                    {
                        decompose_ring_interleaved::<F, D>(ring, digit_buf, num_digits, p);
                        let base = local_elem_idx * num_digits;
                        for digit in 0..num_digits {
                            sparse_mul_acc::<D>(
                                &digit_buf[digit],
                                challenge,
                                &mut acc[base + digit],
                            );
                        }
                    }
                }
            }
        });

    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(elem_chunk * num_digits)
        .enumerate()
        .for_each(|(tid, acc)| {
            let elem_start = tid * elem_chunk;
            if elem_start >= block_len {
                return;
            }
            let elems_in_chunk = acc.len() / num_digits;
            let elem_end = elem_start + elems_in_chunk;
            let mut digit_buf = (!use_fused_full_challenge).then(|| vec![[0i8; D]; num_digits]);
            let mut rotated = use_fused_full_challenge.then(|| vec![[0i32; D]; D]);

            for (block_idx, challenge) in challenges.iter().enumerate() {
                let block_start = block_idx * block_len;
                if block_start >= coeffs.len() {
                    break;
                }
                let coeff_start = block_start + elem_start;
                if coeff_start >= coeffs.len() {
                    continue;
                }
                let coeff_end = (block_start + elem_end).min(coeffs.len());
                if let Some(rotated) = rotated.as_mut() {
                    fill_rotated_challenge::<D>(rotated, challenge);
                    for (local_elem_idx, ring) in coeffs[coeff_start..coeff_end].iter().enumerate()
                    {
                        let base = local_elem_idx * num_digits;
                        decompose_ring_full_challenge_accumulate::<F, D>(
                            ring,
                            rotated,
                            &mut acc[base..base + num_digits],
                            p,
                        );
                    }
                } else if let Some(digit_buf) = digit_buf.as_mut() {
                    for (local_elem_idx, ring) in coeffs[coeff_start..coeff_end].iter().enumerate()
                    {
                        decompose_ring_interleaved::<F, D>(ring, digit_buf, num_digits, p);
                        let base = local_elem_idx * num_digits;
                        for digit in 0..num_digits {
                            sparse_mul_acc::<D>(
                                &digit_buf[digit],
                                challenge,
                                &mut acc[base + digit],
                            );
                        }
                    }
                }
            }
        });

    out
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

#[cfg(test)]
mod tests {
    use super::{
        decompose_ring_full_challenge_accumulate, decompose_ring_interleaved,
        fill_rotated_challenge, sparse_mul_acc, DecomposeParams,
    };
    use crate::algebra::ring::sparse_challenge::SparseChallenge;
    use crate::algebra::{CyclotomicRing, Fp64};
    use crate::{CanonicalField, FieldCore, FromSmallInt};

    #[test]
    fn fused_full_challenge_accumulate_matches_generic_sparse_path() {
        type F = Fp64<4294967197>;
        const D: usize = 32;
        let num_digits = 4;
        let ring = CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
            let v = ((7 * k as i64) % 17) - 8;
            F::from_i64(v)
        }));
        let challenge = SparseChallenge {
            positions: (0..D as u32).collect(),
            coeffs: (0..D)
                .map(|k| match k % 5 {
                    0 => -3,
                    1 => -1,
                    2 => 1,
                    3 => 2,
                    _ => 4,
                })
                .collect(),
        };
        let q = (-F::one()).to_canonical_u128() + 1;
        let params = DecomposeParams {
            half_q: q / 2,
            q,
            mask: (1i128 << 3) - 1,
            half_b: 1i128 << 2,
            b_val: 1i128 << 3,
            log_basis: 3,
        };

        let mut generic_digits = vec![[0i8; D]; num_digits];
        decompose_ring_interleaved::<F, D>(&ring, &mut generic_digits, num_digits, &params);
        let mut generic_acc = vec![[0i32; D]; num_digits];
        for digit in 0..num_digits {
            sparse_mul_acc::<D>(&generic_digits[digit], &challenge, &mut generic_acc[digit]);
        }

        let mut rotated = vec![[0i32; D]; D];
        fill_rotated_challenge::<D>(&mut rotated, &challenge);
        let mut fused_acc = vec![[0i32; D]; num_digits];
        decompose_ring_full_challenge_accumulate::<F, D>(&ring, &rotated, &mut fused_acc, &params);

        assert_eq!(fused_acc, generic_acc);
    }
}
