//! Shared internal helpers for the decompose-fold and commit-inner pipelines.
//!
//! Contains balanced-digit decomposition, sparse multiply-accumulate kernels,
//! position-partitioned accumulation strategies, and the final witness
//! construction used by dense, one-hot, and sparse-ring backends.

mod decompose_fold_partitioned;
mod narrow_accum;
mod rotated_accum;

pub use decompose_fold_partitioned::balanced_ring_decompose_fold_partitioned;
pub(crate) use decompose_fold_partitioned::cached_digit_decompose_fold_partitioned;
pub(crate) use decompose_fold_partitioned::packed_tight_digit_fold_partitioned;

use crate::kernels::linear::try_centered_i8;
#[cfg(target_arch = "aarch64")]
use crate::kernels::neon_decompose_fold as decompose_fold_neon;
use crate::opaque::DecomposeFoldWitness;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_error::AkitaError;
use akita_types::SubfieldMultiplierOpeningPoint;
use jolt_field::{CanonicalEncoding, Field};

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

pub(crate) fn balanced_ring_decompose_fold_chunked<F, const D: usize>(
    rings: &[CyclotomicRing<F, D>],
    challenges: &[SparseChallenge],
    chunk_ranges: &[std::ops::Range<usize>],
    num_positions_per_block: usize,
    params: &BalancedDecomposePow2Params<F>,
) -> Vec<DecomposeFoldWitness>
where
    F: Field + CanonicalEncoding,
{
    chunk_ranges
        .iter()
        .map(|range| {
            let ring_start = range.start * num_positions_per_block;
            let ring_end = (range.end * num_positions_per_block).min(rings.len());
            let coefficients = balanced_ring_decompose_fold_partitioned(
                &rings[ring_start.min(rings.len())..ring_end],
                &challenges[range.clone()],
                num_positions_per_block,
                params,
            );
            DecomposeFoldWitness::from_centered_rows(coefficients)
        })
        .collect()
}

pub(crate) fn try_small_i8_cache_from_ring_coeffs<F: Field + CanonicalEncoding, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
) -> Option<Vec<[i8; D]>> {
    let q = (-F::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
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

/// Fused base-field fold + evaluation shared by backends that do not specialize it.
pub(crate) fn fused_evaluate_and_fold_base<F, const D: usize>(
    folded: Vec<CyclotomicRing<F, D>>,
    live_block_weights: &[F],
) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>)
where
    F: Field + CanonicalEncoding,
{
    let mut eval = CyclotomicRing::<F, D>::zero();
    for (folded_block, &live_block_weight) in folded.iter().zip(live_block_weights) {
        folded_block.scale_accumulate_into(&mut eval, live_block_weight);
    }
    (eval, folded)
}

/// Contract folded arbitrary-ring rows with materialized sparse ring multipliers.
pub(crate) fn fused_evaluate_and_fold_materialized<F, const D: usize>(
    folded: Vec<CyclotomicRing<F, D>>,
    live_block_weights: &[CyclotomicRing<F, D>],
) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>)
where
    F: Field + CanonicalEncoding,
{
    let mut eval = CyclotomicRing::<F, D>::zero();
    for (folded_block, live_block_weight) in folded.iter().zip(live_block_weights) {
        folded_block.mul_accumulate_sparse_rhs_into(live_block_weight, &mut eval);
    }
    (eval, folded)
}

/// Fused outer evaluation over compact proper-extension multipliers.
pub(crate) fn fused_evaluate_and_fold_subfield<F, const D: usize>(
    folded: Vec<CyclotomicRing<F, D>>,
    multipliers: &SubfieldMultiplierOpeningPoint<F>,
) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError>
where
    F: Field + CanonicalEncoding,
{
    let mut eval = CyclotomicRing::<F, D>::zero();
    for (block_idx, folded_block) in folded.iter().enumerate() {
        multipliers.accumulate_fold_product(block_idx, folded_block, &mut eval)?;
    }
    Ok((eval, folded))
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
