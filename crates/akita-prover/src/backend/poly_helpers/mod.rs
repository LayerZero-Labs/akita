//! Shared internal helpers for the decompose-fold and commit-inner pipelines.
//!
//! Contains balanced-digit decomposition, sparse multiply-accumulate kernels,
//! position-partitioned accumulation strategies, and the final witness
//! construction used by dense, one-hot, and sparse-ring backends.
//! implementations.

mod decompose_fold_partitioned;
mod rotated_accum;

pub use decompose_fold_partitioned::{
    balanced_digit_decompose_fold_partitioned, balanced_ring_decompose_fold_partitioned,
    cached_digit_decompose_fold_partitioned,
};

use crate::kernels::linear::try_centered_i8;
use crate::DecomposeFoldWitness;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::sis::fold_response_shift;
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

pub fn sparse_mul_acc_scalar<const D: usize>(
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

/// Dispatch to NEON / AVX2 / scalar sparse-multiply-accumulate.
#[inline(always)]
pub fn sparse_mul_acc<const D: usize>(
    digit_plane: &[i8],
    challenge: &SparseChallenge,
    acc: &mut [i32; D],
) {
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

fn balanced_decompose_i128_i8<const D: usize>(
    centered: &[i128; D],
    out: &mut [[i8; D]],
    log_basis: u32,
) {
    let half_b = 1i128 << (log_basis - 1);
    let b = half_b << 1;
    let mask = b - 1;

    for coeff_idx in 0..D {
        let mut c = centered[coeff_idx];
        for plane in out.iter_mut() {
            let d = c & mask;
            let balanced = if d >= half_b { d - b } else { d };
            c = (c - balanced) >> log_basis;
            plane[coeff_idx] = balanced as i8;
        }
    }
}

pub(crate) fn committed_fold_digits_from_centered<const D: usize>(
    centered_coeffs: &[[i32; D]],
    log_basis: u32,
    num_digits_fold: usize,
) -> Result<(u128, Vec<[i8; D]>), AkitaError> {
    let committed_shift = fold_response_shift(log_basis, num_digits_fold);
    let shift_i128 = i128::try_from(committed_shift).map_err(|_| {
        AkitaError::InvalidSetup("fold response shift exceeds prover i128 range".to_string())
    })?;
    let mut committed_digits = vec![[0i8; D]; centered_coeffs.len() * num_digits_fold];
    for (row_idx, row) in centered_coeffs.iter().enumerate() {
        let shifted: [i128; D] = std::array::from_fn(|idx| i128::from(row[idx]) - shift_i128);
        balanced_decompose_i128_i8(
            &shifted,
            &mut committed_digits[row_idx * num_digits_fold..(row_idx + 1) * num_digits_fold],
            log_basis,
        );
    }
    Ok((committed_shift, committed_digits))
}

/// Balanced digit planes for each chunked window's semantic fold `z_i - eta`,
/// concatenated in chunk order for witness emission and quotient kernels.
pub(crate) fn committed_fold_digits_concat_per_chunk<const D: usize>(
    z_folded_centered_per_chunk: &[Vec<[i32; D]>],
    log_basis: u32,
    num_digits_fold: usize,
    expected_shift: u128,
) -> Result<Vec<[i8; D]>, AkitaError> {
    let mut out = Vec::new();
    for chunk in z_folded_centered_per_chunk {
        let (shift, digits) =
            committed_fold_digits_from_centered(chunk, log_basis, num_digits_fold)?;
        if shift != expected_shift {
            return Err(AkitaError::InvalidInput(
                "chunked fold witness shift mismatch".to_string(),
            ));
        }
        out.extend_from_slice(&digits);
    }
    Ok(out)
}

pub fn build_decompose_fold_witness<F: CanonicalField, const D: usize>(
    centered_coeffs: Vec<[i32; D]>,
    modulus: u128,
    log_basis: u32,
    num_digits_fold: usize,
) -> DecomposeFoldWitness<F, D> {
    let centered_inf_norm = centered_coeffs
        .iter()
        .flat_map(|row| row.iter())
        .map(|coeff| coeff.unsigned_abs())
        .max()
        .unwrap_or(0);
    let (committed_shift, committed_digits) =
        committed_fold_digits_from_centered(&centered_coeffs, log_basis, num_digits_fold)
            .expect("fold response shift must fit i128 for prover fold digits");
    let z_folded_rings = cfg_iter!(centered_coeffs)
        .map(|coeff_accum| signed_accum_to_ring::<F, D>(*coeff_accum, modulus))
        .collect();
    DecomposeFoldWitness {
        z_folded_rings,
        centered_coeffs,
        centered_inf_norm,
        log_basis,
        num_digits_fold,
        committed_shift,
        committed_digits,
    }
}

/// Fused base-field fold + evaluation shared by backends that do not specialize it.
pub(crate) fn fused_evaluate_and_fold_base<F, const D: usize>(
    folded: Vec<CyclotomicRing<F, D>>,
    eval_outer_scalars: &[F],
) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>)
where
    F: CanonicalField,
{
    let eval = folded
        .iter()
        .zip(eval_outer_scalars.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
            acc + f_i.scale(s_i)
        });
    (eval, folded)
}

/// Fused ring-multiplier fold + evaluation shared by backends that do not specialize it.
pub(crate) fn fused_evaluate_and_fold_ring<F, const D: usize>(
    folded: Vec<CyclotomicRing<F, D>>,
    eval_outer_scalars: &[CyclotomicRing<F, D>],
) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>)
where
    F: FieldCore,
{
    let eval = folded
        .iter()
        .zip(eval_outer_scalars.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
            acc + (*f_i * *s_i)
        });
    (eval, folded)
}

#[cfg(test)]
mod tests {
    use super::rotated_accum::{
        decompose_ring_full_challenge_accumulate, should_use_rotated_challenge,
    };
    use super::{
        balanced_ring_decompose_fold_partitioned, decompose_ring_interleaved,
        fill_rotated_challenge, sparse_mul_acc, sparse_mul_acc_scalar, DecomposeParams,
    };
    use akita_algebra::CyclotomicRing;
    use akita_challenges::SparseChallenge;
    use akita_field::CanonicalField;
    use akita_field::{Fp64, Prime128Offset275};
    use akita_types::sis::compute_num_digits_full_field;

    /// SIMD-vs-scalar parity for the sparse-multiply-accumulate decompose-fold
    /// kernel, exercising whichever SIMD backend is active (NEON / AVX2 /
    /// AVX-512). Restricted to `|coeff| <= 2` so the SIMD fast path fires.
    /// `D = 128` matches typical small-field schedules and gives both kernels
    /// multiple full-width iterations to chew through.
    #[test]
    fn sparse_mul_acc_simd_matches_scalar_small_coeffs() {
        const D: usize = 128;

        // Construct a small-coefficient challenge that hits both positive and
        // negative paths for both magnitudes 1 and 2. Positions cover both the
        // pure-prefix (split == D, no wrap) and the wrap-around case.
        let positions: Vec<u32> = (0..32u32).map(|k| k * 4).collect();
        let coeffs: Vec<i8> = (0..32)
            .map(|k| match k % 4 {
                0 => 1,
                1 => -1,
                2 => 2,
                _ => -2,
            })
            .collect();
        let challenge = SparseChallenge { positions, coeffs };

        let digit_plane: [i8; D] = std::array::from_fn(|k| (((7 * k as i64) % 13) - 6) as i8);

        let mut simd_acc = [0i32; D];
        let mut scalar_acc = [0i32; D];

        sparse_mul_acc::<D>(&digit_plane, &challenge, &mut simd_acc);
        sparse_mul_acc_scalar::<D>(&digit_plane, &challenge, &mut scalar_acc);

        assert_eq!(
            simd_acc, scalar_acc,
            "SIMD sparse_mul_acc disagreed with scalar reference"
        );
    }

    /// Edge case: challenge with `pos == 0` so `split == D` and the second
    /// (wrap) segment is empty.
    #[test]
    fn sparse_mul_acc_simd_zero_position() {
        const D: usize = 64;
        let challenge = SparseChallenge {
            positions: vec![0],
            coeffs: vec![1],
        };
        let digit_plane: [i8; D] = std::array::from_fn(|k| (k as i8) - 32);

        let mut simd_acc = [0i32; D];
        let mut scalar_acc = [0i32; D];
        sparse_mul_acc::<D>(&digit_plane, &challenge, &mut simd_acc);
        sparse_mul_acc_scalar::<D>(&digit_plane, &challenge, &mut scalar_acc);

        assert_eq!(simd_acc, scalar_acc);
    }

    /// Edge case: challenge with `pos == D - 1` so `split == 1` and the
    /// post-split (wrap) segment is the bulk of the work.
    #[test]
    fn sparse_mul_acc_simd_max_position() {
        const D: usize = 64;
        let challenge = SparseChallenge {
            positions: vec![(D - 1) as u32],
            coeffs: vec![-2],
        };
        let digit_plane: [i8; D] = std::array::from_fn(|k| ((k as i8) - 32).wrapping_mul(3));

        let mut simd_acc = [0i32; D];
        let mut scalar_acc = [0i32; D];
        sparse_mul_acc::<D>(&digit_plane, &challenge, &mut simd_acc);
        sparse_mul_acc_scalar::<D>(&digit_plane, &challenge, &mut scalar_acc);

        assert_eq!(simd_acc, scalar_acc);
    }

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
        let log_basis = 3u32;
        let threshold = akita_algebra::ring::cyclotomic::decompose_centering_threshold(
            num_digits, log_basis, q,
        );
        let params = DecomposeParams {
            threshold,
            q,
            mask: (1i128 << log_basis) - 1,
            half_b: 1i128 << (log_basis - 1),
            b_val: 1i128 << log_basis,
            log_basis,
            overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
        };

        let mut generic_digits = vec![[0i8; D]; num_digits];
        decompose_ring_interleaved::<F, D>(&ring, &mut generic_digits, num_digits, &params);
        let mut generic_acc = vec![[0i32; D]; num_digits];
        for digit in 0..num_digits {
            sparse_mul_acc::<D>(&generic_digits[digit], &challenge, &mut generic_acc[digit]);
        }

        let mut rotated = vec![[0i16; D]; D];
        fill_rotated_challenge::<D>(&mut rotated, &challenge);
        let mut fused_acc = vec![[0i32; D]; num_digits];
        decompose_ring_full_challenge_accumulate::<F, D>(&ring, &rotated, &mut fused_acc, &params);

        assert_eq!(fused_acc, generic_acc);
    }

    #[test]
    fn partitioned_full_challenge_accumulate_matches_generic_sparse_path() {
        type F = Fp64<4294967197>;
        const D: usize = 32;
        let block_len = 3;
        let num_digits = 4;
        let coeffs: Vec<_> = (0..6)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    let v = (((idx * 11 + k * 7) as i64) % 19) - 9;
                    F::from_i64(v)
                }))
            })
            .collect();
        let challenges = vec![
            SparseChallenge {
                positions: (0..D as u32).collect(),
                coeffs: (0..D)
                    .map(|k| match k % 4 {
                        0 => -2,
                        1 => -1,
                        2 => 1,
                        _ => 3,
                    })
                    .collect(),
            },
            SparseChallenge {
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
            },
        ];
        let q = (-F::one()).to_canonical_u128() + 1;
        let log_basis = 3u32;
        let threshold = akita_algebra::ring::cyclotomic::decompose_centering_threshold(
            num_digits, log_basis, q,
        );
        let params = DecomposeParams {
            threshold,
            q,
            mask: (1i128 << log_basis) - 1,
            half_b: 1i128 << (log_basis - 1),
            b_val: 1i128 << log_basis,
            log_basis,
            overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
        };

        let fused = balanced_ring_decompose_fold_partitioned::<F, D>(
            &coeffs,
            &challenges,
            block_len,
            num_digits,
            &params,
        );

        let mut generic = vec![[0i32; D]; block_len * num_digits];
        let mut digit_buf = vec![[0i8; D]; num_digits];
        for (block_idx, challenge) in challenges.iter().enumerate() {
            let block_start = block_idx * block_len;
            for local_idx in 0..block_len {
                let ring = &coeffs[block_start + local_idx];
                decompose_ring_interleaved::<F, D>(ring, &mut digit_buf, num_digits, &params);
                let base = local_idx * num_digits;
                for digit in 0..num_digits {
                    sparse_mul_acc::<D>(&digit_buf[digit], challenge, &mut generic[base + digit]);
                }
            }
        }

        assert_eq!(fused, generic);
    }

    #[test]
    fn partitioned_high_density_d32_challenge_uses_rotated_path() {
        type F = Fp64<4294967197>;
        const D: usize = 32;
        let block_len = 3;
        let num_digits = 4;
        let coeffs: Vec<_> = (0..6)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    let v = (((idx * 13 + k * 5) as i64) % 23) - 11;
                    F::from_i64(v)
                }))
            })
            .collect();
        let high_density = SparseChallenge {
            positions: vec![
                0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 28, 29, 30, 31,
            ],
            coeffs: vec![
                2, 2, -1, 4, 1, -1, 5, 4, -3, -4, -3, -6, 2, -8, -4, -3, -7, -3, 4, -1, 4, -4, 5,
                -2, -4, 6, 6, -3, 4, 4,
            ],
        };
        let sparse = SparseChallenge {
            positions: vec![1, 7, 19],
            coeffs: vec![2, -1, 3],
        };
        assert!(should_use_rotated_challenge::<D>(&high_density));
        assert!(!should_use_rotated_challenge::<D>(&sparse));
        let challenges = vec![high_density, sparse];
        let q = (-F::one()).to_canonical_u128() + 1;
        let log_basis = 3u32;
        let threshold = akita_algebra::ring::cyclotomic::decompose_centering_threshold(
            num_digits, log_basis, q,
        );
        let params = DecomposeParams {
            threshold,
            q,
            mask: (1i128 << log_basis) - 1,
            half_b: 1i128 << (log_basis - 1),
            b_val: 1i128 << log_basis,
            log_basis,
            overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
        };

        let mixed = balanced_ring_decompose_fold_partitioned::<F, D>(
            &coeffs,
            &challenges,
            block_len,
            num_digits,
            &params,
        );

        let mut generic = vec![[0i32; D]; block_len * num_digits];
        let mut digit_buf = vec![[0i8; D]; num_digits];
        for (block_idx, challenge) in challenges.iter().enumerate() {
            let block_start = block_idx * block_len;
            for local_idx in 0..block_len {
                let ring = &coeffs[block_start + local_idx];
                decompose_ring_interleaved::<F, D>(ring, &mut digit_buf, num_digits, &params);
                let base = local_idx * num_digits;
                for digit in 0..num_digits {
                    sparse_mul_acc::<D>(&digit_buf[digit], challenge, &mut generic[base + digit]);
                }
            }
        }

        assert_eq!(mixed, generic);
    }

    #[test]
    fn partitioned_high_density_d64_challenge_uses_rotated_path() {
        type F = Fp64<4294967197>;
        const D: usize = 64;
        let block_len = 2;
        let num_digits = 3;
        let coeffs: Vec<_> = (0..4)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    let v = (((idx * 17 + k * 7) as i64) % 31) - 15;
                    F::from_i64(v)
                }))
            })
            .collect();
        let high_density = SparseChallenge {
            positions: (0..42u32).collect(),
            coeffs: (0..42)
                .map(|k| match k % 4 {
                    0 => -2,
                    1 => -1,
                    2 => 1,
                    _ => 2,
                })
                .collect(),
        };
        let sparse = SparseChallenge {
            positions: vec![1, 17, 33, 49],
            coeffs: vec![2, -1, 1, -2],
        };
        assert!(should_use_rotated_challenge::<D>(&high_density));
        assert!(!should_use_rotated_challenge::<D>(&sparse));
        let challenges = vec![high_density, sparse];
        let q = (-F::one()).to_canonical_u128() + 1;
        let log_basis = 4u32;
        let threshold = akita_algebra::ring::cyclotomic::decompose_centering_threshold(
            num_digits, log_basis, q,
        );
        let params = DecomposeParams {
            threshold,
            q,
            mask: (1i128 << log_basis) - 1,
            half_b: 1i128 << (log_basis - 1),
            b_val: 1i128 << log_basis,
            log_basis,
            overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
        };

        let mixed = balanced_ring_decompose_fold_partitioned::<F, D>(
            &coeffs,
            &challenges,
            block_len,
            num_digits,
            &params,
        );

        let mut generic = vec![[0i32; D]; block_len * num_digits];
        let mut digit_buf = vec![[0i8; D]; num_digits];
        for (block_idx, challenge) in challenges.iter().enumerate() {
            let block_start = block_idx * block_len;
            for local_idx in 0..block_len {
                let ring = &coeffs[block_start + local_idx];
                decompose_ring_interleaved::<F, D>(ring, &mut digit_buf, num_digits, &params);
                let base = local_idx * num_digits;
                for digit in 0..num_digits {
                    sparse_mul_acc::<D>(&digit_buf[digit], challenge, &mut generic[base + digit]);
                }
            }
        }

        assert_eq!(mixed, generic);
    }

    #[test]
    fn fp128_overflow_paths_match_direct_and_fused_sparse_path() {
        type F = Prime128Offset275;
        const D: usize = 32;

        let log_basis = 4u32;
        let num_digits = compute_num_digits_full_field(128, log_basis);
        let q = (-F::one()).to_canonical_u128() + 1;
        let threshold = akita_algebra::ring::cyclotomic::decompose_centering_threshold(
            num_digits, log_basis, q,
        );
        let i128_max = i128::MAX as u128;
        let boundary_values = [
            0,
            threshold,
            threshold + 1,
            q - i128_max - 1,
            q - i128_max,
            q - 1,
        ];
        let ring = CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
            F::from_canonical_u128_reduced(boundary_values[k % boundary_values.len()])
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
        let params = DecomposeParams {
            threshold,
            q,
            mask: (1i128 << log_basis) - 1,
            half_b: 1i128 << (log_basis - 1),
            b_val: 1i128 << log_basis,
            log_basis,
            overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
        };

        assert!(
            params.overflow_possible,
            "test must exercise the overflow path"
        );

        let mut actual_digits = vec![[0i8; D]; num_digits];
        decompose_ring_interleaved::<F, D>(&ring, &mut actual_digits, num_digits, &params);
        let mut expected_digits = vec![[0i8; D]; num_digits];
        ring.balanced_decompose_pow2_i8_into(&mut expected_digits, log_basis);
        assert_eq!(actual_digits, expected_digits);

        let mut generic_acc = vec![[0i32; D]; num_digits];
        for digit in 0..num_digits {
            sparse_mul_acc::<D>(&actual_digits[digit], &challenge, &mut generic_acc[digit]);
        }

        let mut rotated = vec![[0i16; D]; D];
        fill_rotated_challenge::<D>(&mut rotated, &challenge);
        let mut fused_acc = vec![[0i32; D]; num_digits];
        decompose_ring_full_challenge_accumulate::<F, D>(&ring, &rotated, &mut fused_acc, &params);
        assert_eq!(fused_acc, generic_acc);
    }
}
