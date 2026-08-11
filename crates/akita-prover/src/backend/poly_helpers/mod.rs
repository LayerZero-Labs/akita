//! Shared internal helpers for the decompose-fold and commit-inner pipelines.
//!
//! Contains balanced-digit decomposition, sparse multiply-accumulate kernels,
//! position-partitioned accumulation strategies, and the final witness
//! construction used by dense, one-hot, and sparse-ring backends.
//! implementations.

mod decompose_fold_partitioned;
mod rotated_accum;

pub use decompose_fold_partitioned::{
    balanced_ring_decompose_fold_partitioned, balanced_tight_digit_fold_partitioned,
    cached_digit_decompose_fold_partitioned,
};

use crate::kernels::linear::try_centered_i8;
use crate::DecomposeFoldWitness;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore};
use std::array::from_fn;

#[cfg(target_arch = "aarch64")]
use crate::kernels::neon_decompose_fold as decompose_fold_neon;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::kernels::avx_decompose_fold as decompose_fold_avx;

/// Whether the SIMD `decompose-fold` dispatch is enabled.
///
/// On aarch64 this delegates to [`akita_algebra::ntt::neon::use_neon_ntt`]
/// so a single `AKITA_SCALAR_NTT=1` env var disables both the NEON NTT and
/// the NEON decompose-fold for A/B benchmarks. On x86 we read the same env
/// var locally (the NEON module isn't compiled, so we can't share the
/// helper across crates without re-introducing a hoist into `akita-algebra`).
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx2")
))]
fn use_simd_decompose_fold() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        akita_algebra::ntt::neon::use_neon_ntt()
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("AKITA_SCALAR_NTT").map_or(true, |v| v != "1"))
    }
}

pub struct DecomposeParams {
    pub threshold: u128,
    pub q: u128,
    pub mask: i128,
    pub half_b: i128,
    pub b_val: i128,
    pub log_basis: u32,
    pub overflow_possible: bool,
}

/// Decompose all D coefficients of a ring element into balanced base-b digits,
/// storing results in digit-major order for subsequent SIMD scatter.
///
/// Uses K=3 interleaved carry chains to saturate ALU throughput (3x ILP gain
/// over processing one coefficient at a time on out-of-order cores).
///
/// `digit_buf` is `[num_digits][D]` in i8, OVERWRITTEN (not accumulated).
#[inline(never)]
pub fn decompose_ring_interleaved<F: CanonicalField, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    digit_buf: &mut [[i8; D]],
    num_digits: usize,
    p: &DecomposeParams,
) {
    if p.overflow_possible {
        decompose_ring_interleaved_overflow(ring, digit_buf, num_digits, p);
    } else {
        decompose_ring_interleaved_fast(ring, digit_buf, num_digits, p);
    }
}

/// Signed-i16 counterpart of [`decompose_ring_interleaved`] for bases above 8.
#[inline(never)]
pub fn decompose_ring_interleaved_i16<F: CanonicalField, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    digit_buf: &mut [[i16; D]],
    num_digits: usize,
    p: &DecomposeParams,
) {
    let bulk_end = D - (D % 3);
    for base in (0..bulk_end).step_by(3) {
        let canonical = [
            ring.coeffs[base].to_canonical_u128(),
            ring.coeffs[base + 1].to_canonical_u128(),
            ring.coeffs[base + 2].to_canonical_u128(),
        ];
        let (mut carries, first_digits) = if p.overflow_possible {
            let (c0, d0) = peel_first_balanced_digit_i32(canonical[0], p);
            let (c1, d1) = peel_first_balanced_digit_i32(canonical[1], p);
            let (c2, d2) = peel_first_balanced_digit_i32(canonical[2], p);
            ([c0, c1, c2], Some([d0, d1, d2]))
        } else {
            (canonical.map(|coefficient| to_signed(coefficient, p)), None)
        };
        for (digit_index, plane) in digit_buf.iter_mut().take(num_digits).enumerate() {
            let digits = if digit_index == 0 {
                first_digits.unwrap_or_else(|| {
                    carries
                        .each_mut()
                        .map(|carry| extract_balanced_digit(carry, p))
                })
            } else {
                carries
                    .each_mut()
                    .map(|carry| extract_balanced_digit(carry, p))
            };
            plane[base] = digits[0] as i16;
            plane[base + 1] = digits[1] as i16;
            plane[base + 2] = digits[2] as i16;
        }
    }
    for idx in bulk_end..D {
        let canonical = ring.coeffs[idx].to_canonical_u128();
        let (mut carry, first_digit) = if p.overflow_possible {
            let (carry, digit) = peel_first_balanced_digit_i32(canonical, p);
            (carry, Some(digit))
        } else {
            (to_signed(canonical, p), None)
        };
        for (digit_index, plane) in digit_buf.iter_mut().take(num_digits).enumerate() {
            plane[idx] = if digit_index == 0 {
                first_digit.unwrap_or_else(|| extract_balanced_digit(&mut carry, p))
            } else {
                extract_balanced_digit(&mut carry, p)
            } as i16;
        }
    }
}

fn decompose_ring_interleaved_fast<F: CanonicalField, const D: usize>(
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

fn decompose_ring_interleaved_overflow<F: CanonicalField, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    digit_buf: &mut [[i8; D]],
    num_digits: usize,
    p: &DecomposeParams,
) {
    let (first_plane, remaining) = digit_buf
        .split_first_mut()
        .expect("decompose_ring_interleaved_overflow requires at least one plane");
    let bulk_end = D - (D % 3);

    for base in (0..bulk_end).step_by(3) {
        let canonical0 = ring.coeffs[base].to_canonical_u128();
        let canonical1 = ring.coeffs[base + 1].to_canonical_u128();
        let canonical2 = ring.coeffs[base + 2].to_canonical_u128();

        let (mut c0, d0) = peel_first_balanced_digit_i32(canonical0, p);
        let (mut c1, d1) = peel_first_balanced_digit_i32(canonical1, p);
        let (mut c2, d2) = peel_first_balanced_digit_i32(canonical2, p);

        first_plane[base] = d0 as i8;
        first_plane[base + 1] = d1 as i8;
        first_plane[base + 2] = d2 as i8;

        for plane in remaining.iter_mut().take(num_digits - 1) {
            let d0 = extract_balanced_digit(&mut c0, p);
            let d1 = extract_balanced_digit(&mut c1, p);
            let d2 = extract_balanced_digit(&mut c2, p);
            plane[base] = d0 as i8;
            plane[base + 1] = d1 as i8;
            plane[base + 2] = d2 as i8;
        }
    }

    for idx in bulk_end..D {
        let canonical = ring.coeffs[idx].to_canonical_u128();
        let (mut c, d0) = peel_first_balanced_digit_i32(canonical, p);
        first_plane[idx] = d0 as i8;
        for plane in remaining.iter_mut().take(num_digits - 1) {
            plane[idx] = extract_balanced_digit(&mut c, p) as i8;
        }
    }
}

#[inline(never)]
pub fn decompose_ring_single_digit<F: CanonicalField, const D: usize>(
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
pub(crate) fn to_signed(canonical: u128, p: &DecomposeParams) -> i128 {
    if canonical > p.threshold {
        -((p.q - canonical) as i128)
    } else {
        canonical as i128
    }
}

pub fn try_small_i8_cache_from_ring_coeffs<F: CanonicalField, const D: usize>(
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
pub(crate) fn extract_balanced_digit(c: &mut i128, p: &DecomposeParams) -> i32 {
    debug_assert!(p.log_basis < 31);
    if p.log_basis == 2 {
        let d = (*c as i32) & 3;
        let balanced = if d >= 2 { d - 4 } else { d };
        *c = (*c - i128::from(balanced)) >> 2;
        return balanced;
    }

    let d = (*c as i32) & (p.mask as i32);
    let balanced = if d >= p.half_b as i32 {
        d - p.b_val as i32
    } else {
        d
    };
    *c = (*c - i128::from(balanced)) >> p.log_basis;
    balanced
}

#[inline(always)]
pub(crate) fn peel_first_balanced_digit_i32(canonical: u128, p: &DecomposeParams) -> (i128, i32) {
    if canonical <= p.threshold {
        let mut c = canonical as i128;
        let d = extract_balanced_digit(&mut c, p);
        return (c, d);
    }

    let diff = p.q - canonical;
    if diff <= i128::MAX as u128 {
        let mut c = -(diff as i128);
        let d = extract_balanced_digit(&mut c, p);
        return (c, d);
    }

    let mask = p.mask as u128;
    let half_b = p.half_b as u128;
    let b_val = p.b_val as u128;
    let r = canonical.wrapping_sub(p.q) & mask;
    let balanced = if r >= half_b {
        r as i32 - b_val as i32
    } else {
        r as i32
    };
    let diff_adj = if balanced >= 0 {
        diff + balanced as u128
    } else {
        diff - ((-balanced) as u128)
    };
    debug_assert!(diff_adj & mask == 0);
    (-((diff_adj >> p.log_basis) as i128), balanced)
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

pub(crate) fn sparse_mul_acc_scalar<const D: usize>(
    digit_plane: &[i8; D],
    challenge: &SparseChallenge,
    acc: &mut [i32; D],
) {
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let p = pos as usize;
        match coeff {
            1 => sparse_mul_acc_add_scalar::<D>(digit_plane, acc, p),
            -1 => sparse_mul_acc_sub_scalar::<D>(digit_plane, acc, p),
            2 => {
                let split = D - p;
                for i in 0..split {
                    acc[i + p] += 2 * i32::from(digit_plane[i]);
                }
                for i in split..D {
                    acc[i - split] -= 2 * i32::from(digit_plane[i]);
                }
            }
            -2 => {
                let split = D - p;
                for i in 0..split {
                    acc[i + p] -= 2 * i32::from(digit_plane[i]);
                }
                for i in split..D {
                    acc[i - split] += 2 * i32::from(digit_plane[i]);
                }
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

pub(crate) fn sparse_mul_acc_i16_scalar<const D: usize>(
    digit_plane: &[i16; D],
    challenge: &SparseChallenge,
    acc: &mut [i32; D],
) {
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let p = pos as usize;
        let split = D - p;
        let scale = i32::from(coeff);
        for i in 0..split {
            acc[i + p] += scale * i32::from(digit_plane[i]);
        }
        for i in split..D {
            acc[i - split] -= scale * i32::from(digit_plane[i]);
        }
    }
}

/// Dispatch to NEON / AVX2 / scalar sparse-multiply-accumulate.
#[inline(always)]
pub(crate) fn sparse_mul_acc<const D: usize>(
    digit_plane: &[i8; D],
    challenge: &SparseChallenge,
    acc: &mut [i32; D],
) {
    assert_eq!(challenge.positions.len(), challenge.coeffs.len());
    assert!(challenge
        .positions
        .iter()
        .all(|&position| position < D as u32));
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    ))]
    {
        if use_simd_decompose_fold()
            && challenge
                .coeffs
                .iter()
                .all(|&coeff| coeff.unsigned_abs() <= 2)
        {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                decompose_fold_neon::sparse_mul_acc_neon(
                    digit_plane.as_ptr(),
                    acc.as_mut_ptr(),
                    D,
                    &challenge.positions,
                    &challenge.coeffs,
                );
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            unsafe {
                decompose_fold_avx::sparse_mul_acc_avx(
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

pub(crate) fn sparse_mul_acc_pm1<const D: usize>(
    digit_plane: &[i8; D],
    positive: &[u32],
    negative: &[u32],
    acc: &mut [i32; D],
) {
    debug_assert!(positive
        .iter()
        .chain(negative)
        .all(|&position| position < D as u32));
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    ))]
    if use_simd_decompose_fold() {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            decompose_fold_neon::sparse_mul_acc_pm1_neon(
                digit_plane.as_ptr(),
                acc.as_mut_ptr(),
                D,
                positive,
                negative,
            );
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            decompose_fold_avx::sparse_mul_acc_pm1_avx(
                digit_plane.as_ptr(),
                acc.as_mut_ptr(),
                D,
                positive,
                negative,
            );
        }
        return;
    }
    for &position in positive {
        sparse_mul_acc_add_scalar(digit_plane, acc, position as usize);
    }
    for &position in negative {
        sparse_mul_acc_sub_scalar(digit_plane, acc, position as usize);
    }
}

/// Signed-i16 sparse multiply-accumulate for large inner bases.
#[inline(always)]
pub(crate) fn sparse_mul_acc_i16<const D: usize>(
    digit_plane: &[i16; D],
    challenge: &SparseChallenge,
    acc: &mut [i32; D],
) {
    assert_eq!(challenge.positions.len(), challenge.coeffs.len());
    assert!(challenge
        .positions
        .iter()
        .all(|&position| position < D as u32));
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    ))]
    {
        if use_simd_decompose_fold()
            && challenge
                .coeffs
                .iter()
                .all(|&coeff| coeff.unsigned_abs() <= 2)
        {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                decompose_fold_neon::sparse_mul_acc_i16_neon(
                    digit_plane.as_ptr(),
                    acc.as_mut_ptr(),
                    D,
                    &challenge.positions,
                    &challenge.coeffs,
                );
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            unsafe {
                decompose_fold_avx::sparse_mul_acc_i16_avx(
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
    sparse_mul_acc_i16_scalar::<D>(digit_plane, challenge, acc);
}

pub(crate) fn sparse_mul_acc_i16_pm1<const D: usize>(
    digit_plane: &[i16; D],
    positive: &[u32],
    negative: &[u32],
    acc: &mut [i32; D],
) {
    debug_assert!(positive
        .iter()
        .chain(negative)
        .all(|&position| position < D as u32));
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    ))]
    if use_simd_decompose_fold() {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            decompose_fold_neon::sparse_mul_acc_i16_pm1_neon(
                digit_plane.as_ptr(),
                acc.as_mut_ptr(),
                D,
                positive,
                negative,
            );
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            decompose_fold_avx::sparse_mul_acc_i16_pm1_avx(
                digit_plane.as_ptr(),
                acc.as_mut_ptr(),
                D,
                positive,
                negative,
            );
        }
        return;
    }
    for (&position, scale) in positive
        .iter()
        .map(|position| (position, 1))
        .chain(negative.iter().map(|position| (position, -1)))
    {
        let position = position as usize;
        let split = D - position;
        for i in 0..split {
            acc[i + position] += scale * i32::from(digit_plane[i]);
        }
        for i in split..D {
            acc[i - split] -= scale * i32::from(digit_plane[i]);
        }
    }
}

#[inline(always)]
fn narrow_segment_acc_i8(src: &[i8], dst: &mut [i16], scale: i16) {
    debug_assert_eq!(src.len(), dst.len());
    match scale {
        1 => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc += i16::from(value)),
        -1 => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc -= i16::from(value)),
        2 => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc += i16::from(value) << 1),
        -2 => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc -= i16::from(value) << 1),
        _ => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc += i16::from(value) * scale),
    }
}

#[inline(always)]
fn narrow_segment_acc_i16(src: &[i16], dst: &mut [i16], scale: i16) {
    debug_assert_eq!(src.len(), dst.len());
    match scale {
        1 => src.iter().zip(dst).for_each(|(&value, acc)| *acc += value),
        -1 => src.iter().zip(dst).for_each(|(&value, acc)| *acc -= value),
        2 => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc += value << 1),
        -2 => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc -= value << 1),
        _ => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc += value * scale),
    }
}

/// Accumulate one sparse negacyclic product into a proven-safe i16 partial sum.
///
/// The caller must prove that the complete partial sum stays in the i16 range.
#[inline(always)]
pub(crate) fn sparse_mul_acc_narrow<const D: usize>(
    digit_plane: &[i8; D],
    challenge: &SparseChallenge,
    acc: &mut [i16; D],
) {
    debug_assert_eq!(challenge.positions.len(), challenge.coeffs.len());
    sparse_mul_acc_narrow_terms(digit_plane, &challenge.positions, &challenge.coeffs, acc);
}

#[inline(always)]
fn sparse_mul_acc_narrow_terms<const D: usize>(
    digit_plane: &[i8; D],
    positions: &[u32],
    coefficients: &[i8],
    acc: &mut [i16; D],
) {
    debug_assert_eq!(positions.len(), coefficients.len());
    for (&position, &coefficient) in positions.iter().zip(coefficients) {
        debug_assert!(position < D as u32);
        let position = position as usize;
        let split = D - position;
        let scale = i16::from(coefficient);
        narrow_segment_acc_i8(&digit_plane[..split], &mut acc[position..], scale);
        if position > 0 {
            narrow_segment_acc_i8(&digit_plane[split..], &mut acc[..position], -scale);
        }
    }
}

/// Signed-i16 source variant of [`sparse_mul_acc_narrow`].
#[inline(always)]
pub(crate) fn sparse_mul_acc_i16_narrow<const D: usize>(
    digit_plane: &[i16; D],
    challenge: &SparseChallenge,
    acc: &mut [i16; D],
) {
    debug_assert_eq!(challenge.positions.len(), challenge.coeffs.len());
    sparse_mul_acc_i16_narrow_terms(digit_plane, &challenge.positions, &challenge.coeffs, acc);
}

#[inline(always)]
fn sparse_mul_acc_i16_narrow_terms<const D: usize>(
    digit_plane: &[i16; D],
    positions: &[u32],
    coefficients: &[i8],
    acc: &mut [i16; D],
) {
    debug_assert_eq!(positions.len(), coefficients.len());
    for (&position, &coefficient) in positions.iter().zip(coefficients) {
        debug_assert!(position < D as u32);
        let position = position as usize;
        let split = D - position;
        let scale = i16::from(coefficient);
        narrow_segment_acc_i16(&digit_plane[..split], &mut acc[position..], scale);
        if position > 0 {
            narrow_segment_acc_i16(&digit_plane[split..], &mut acc[..position], -scale);
        }
    }
}

/// Precompute dense rotation table for a sparse challenge.
///
/// `table[c]` holds the small signed coefficients of `challenge * X^c` in the ring
/// `Z[X]/(X^D + 1)`.  Because D is a power of two, `X^D = -1`, so
/// positions that wrap past D get negated.
///
/// The table is 8 KB for D=64, fitting comfortably in L1 cache.
#[inline(always)]
pub fn fill_rotated_challenge<const D: usize>(table: &mut [[i16; D]], challenge: &SparseChallenge) {
    debug_assert!(D.is_power_of_two());
    debug_assert!(table.len() >= D);

    let mut dense = [0i16; D];
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        dense[pos as usize] = i16::from(coeff);
    }

    for (ci, row) in table.iter_mut().enumerate().take(D) {
        let split = D - ci;
        row[ci..D].copy_from_slice(&dense[..split]);
        for (dst, src) in row[..ci].iter_mut().zip(dense[split..].iter()) {
            *dst = -*src;
        }
    }
}

pub fn signed_accum_to_ring<F: CanonicalField, const D: usize>(
    coeff_accum: [i32; D],
    modulus: u128,
) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(signed_accum_to_coefficients(coeff_accum, modulus))
}

fn signed_accum_to_coefficients<F: CanonicalField, const D: usize>(
    coeff_accum: [i32; D],
    modulus: u128,
) -> [F; D] {
    from_fn(|k| {
        let v = coeff_accum[k];
        if v >= 0 {
            F::from_canonical_u128_reduced(v as u128)
        } else {
            F::from_canonical_u128_reduced(modulus - ((-v) as u128))
        }
    })
}

pub fn build_decompose_fold_witness<F: CanonicalField, const D: usize>(
    centered_coeffs: Vec<[i32; D]>,
    modulus: u128,
) -> DecomposeFoldWitness<F> {
    let centered_inf_norm = centered_coeffs
        .iter()
        .flat_map(|row| row.iter())
        .map(|coeff| coeff.unsigned_abs())
        .max()
        .unwrap_or(0);
    let z_folded_coeffs = cfg_iter!(centered_coeffs)
        .map(|coeff_accum| signed_accum_to_coefficients::<F, D>(*coeff_accum, modulus))
        .collect();
    DecomposeFoldWitness::from_coefficient_parts(
        z_folded_coeffs,
        centered_coeffs,
        centered_inf_norm,
    )
}

/// Fused base-field fold + evaluation shared by backends that do not specialize it.
pub(crate) fn fused_evaluate_and_fold_base<F, const D: usize>(
    folded: Vec<CyclotomicRing<F, D>>,
    live_block_weights: &[F],
) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>)
where
    F: CanonicalField,
{
    let eval = folded
        .iter()
        .zip(live_block_weights.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
            acc + f_i.scale(s_i)
        });
    (eval, folded)
}

/// Fused ring-multiplier fold + evaluation shared by backends that do not specialize it.
pub(crate) fn fused_evaluate_and_fold_ring<F, const D: usize>(
    folded: Vec<CyclotomicRing<F, D>>,
    live_block_weights: &[CyclotomicRing<F, D>],
) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>)
where
    F: FieldCore,
{
    let eval = folded
        .iter()
        .zip(live_block_weights.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
            acc + (*f_i * *s_i)
        });
    (eval, folded)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
