//! Width-aware AVX-512 CRT NTT transforms for `i32` limbs.
//!
//! The generic AVX2 transform in the parent module runs every butterfly stage
//! at 128 bits (four `i32` lanes), which leaves most lanes idle on the wide
//! stages where the butterfly run length is 8, 16, 32, ... For an AVX-512 host
//! each stage instead uses the widest vector that divides its run length:
//! 512-bit (16 lanes) when the half-length `len >= 16`, 256-bit (8 lanes) at
//! `len == 8`, 128-bit (4 lanes) at `len == 4`, and a scalar tail for the
//! `len <= 2` stages that no SIMD width can cover without intra-register
//! shuffles. Because `len` is always a power of two, each stage hits exactly
//! one width with no remainder.
//!
//! These kernels are selected only when `avx_ntt_mode()` reports
//! `AvxNttMode::Avx512`; `AKITA_AVX512_NTT=0` keeps the 128-bit AVX2 path for
//! A/B comparison.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::montgomery::forward_dif_tail_i32_avx2;
use super::{
    mont_mul_16x_i32_avx512, mont_mul_4x_i32_avx2, mont_mul_8x_i32_avx2,
    reduce_range_16x_i32_avx512, reduce_range_4x_i32_avx2, reduce_range_8x_i32_avx2,
};
use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::prime::{MontCoeff, NttPrime};

/// Forward DIF butterfly stages over the cyclic NTT, width-aware.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW (and AVX2) are available. `D` is a
/// power of two.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx2")]
unsafe fn forward_dif_stages<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p512 = _mm512_set1_epi32(prime.p);
    let pinv512 = _mm512_set1_epi32(prime.pinv);
    let p256 = _mm256_set1_epi32(prime.p);
    let pinv256 = _mm256_set1_epi32(prime.pinv);
    let p128 = _mm_set1_epi32(prime.p);
    let pinv128 = _mm_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;
    let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;

    let mut len = D / 2;
    while len >= 4 {
        let base = len - 1;
        let mut start = 0usize;
        while start < D {
            if len >= 16 {
                let mut j = 0usize;
                while j < len {
                    // SAFETY: `start + j + len < D` by the stage bounds; AVX-512
                    // proven by this function's safety contract.
                    unsafe {
                        let u = _mm512_loadu_si512(a_ptr.add(start + j) as *const __m512i);
                        let v = _mm512_loadu_si512(a_ptr.add(start + j + len) as *const __m512i);
                        let w = _mm512_loadu_si512(tw_ptr.add(base + j) as *const __m512i);
                        _mm512_storeu_si512(
                            a_ptr.add(start + j) as *mut __m512i,
                            reduce_range_16x_i32_avx512(_mm512_add_epi32(u, v), p512),
                        );
                        _mm512_storeu_si512(
                            a_ptr.add(start + j + len) as *mut __m512i,
                            mont_mul_16x_i32_avx512(_mm512_sub_epi32(u, v), w, p512, pinv512),
                        );
                    }
                    j += 16;
                }
            } else if len == 8 {
                // SAFETY: one 8-lane butterfly covers the whole run; AVX2 proven.
                unsafe {
                    let u = _mm256_loadu_si256(a_ptr.add(start) as *const __m256i);
                    let v = _mm256_loadu_si256(a_ptr.add(start + len) as *const __m256i);
                    let w = _mm256_loadu_si256(tw_ptr.add(base) as *const __m256i);
                    _mm256_storeu_si256(
                        a_ptr.add(start) as *mut __m256i,
                        reduce_range_8x_i32_avx2(_mm256_add_epi32(u, v), p256),
                    );
                    _mm256_storeu_si256(
                        a_ptr.add(start + len) as *mut __m256i,
                        mont_mul_8x_i32_avx2(_mm256_sub_epi32(u, v), w, p256, pinv256),
                    );
                }
            } else {
                // SAFETY: one 4-lane butterfly covers the whole run; SSE proven.
                unsafe {
                    let u = _mm_loadu_si128(a_ptr.add(start) as *const __m128i);
                    let v = _mm_loadu_si128(a_ptr.add(start + len) as *const __m128i);
                    let w = _mm_loadu_si128(tw_ptr.add(base) as *const __m128i);
                    _mm_storeu_si128(
                        a_ptr.add(start) as *mut __m128i,
                        reduce_range_4x_i32_avx2(_mm_add_epi32(u, v), p128),
                    );
                    _mm_storeu_si128(
                        a_ptr.add(start + len) as *mut __m128i,
                        mont_mul_4x_i32_avx2(_mm_sub_epi32(u, v), w, p128, pinv128),
                    );
                }
            }
            start += 2 * len;
        }
        len /= 2;
    }

    if D.is_multiple_of(16) {
        // SAFETY: AVX2 proven by this function's safety contract.
        unsafe { forward_dif_tail_i32_avx2::<D>(a_ptr, tw_ptr, p128, pinv128) };
    } else {
        while len > 0 {
            let base = len - 1;
            let mut start = 0usize;
            while start < D {
                for j in 0..len {
                    let w = tw.fwd_twiddles[base + j];
                    let u = a[start + j];
                    let v = a[start + j + len];
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                }
                start += 2 * len;
            }
            len /= 2;
        }
    }
}

/// Inverse DIT butterfly stages over the cyclic NTT, width-aware.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW (and AVX2) are available. `D` is a
/// power of two.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx2")]
unsafe fn inverse_dit_stages<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p512 = _mm512_set1_epi32(prime.p);
    let pinv512 = _mm512_set1_epi32(prime.pinv);
    let p256 = _mm256_set1_epi32(prime.p);
    let pinv256 = _mm256_set1_epi32(prime.pinv);
    let p128 = _mm_set1_epi32(prime.p);
    let pinv128 = _mm_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;
    let tw_ptr = tw.inv_twiddles.as_ptr() as *const i32;

    let mut len = 1usize;
    while len < D {
        let base = len - 1;
        let mut start = 0usize;
        while start < D {
            if len >= 16 {
                let mut j = 0usize;
                while j < len {
                    // SAFETY: `start + j + len < D` by the stage bounds; AVX-512
                    // proven by this function's safety contract.
                    unsafe {
                        let w = _mm512_loadu_si512(tw_ptr.add(base + j) as *const __m512i);
                        let u = _mm512_loadu_si512(a_ptr.add(start + j) as *const __m512i);
                        let v_raw =
                            _mm512_loadu_si512(a_ptr.add(start + j + len) as *const __m512i);
                        let v = mont_mul_16x_i32_avx512(v_raw, w, p512, pinv512);
                        _mm512_storeu_si512(
                            a_ptr.add(start + j) as *mut __m512i,
                            reduce_range_16x_i32_avx512(_mm512_add_epi32(u, v), p512),
                        );
                        _mm512_storeu_si512(
                            a_ptr.add(start + j + len) as *mut __m512i,
                            reduce_range_16x_i32_avx512(_mm512_sub_epi32(u, v), p512),
                        );
                    }
                    j += 16;
                }
            } else if len == 8 {
                // SAFETY: one 8-lane butterfly covers the whole run; AVX2 proven.
                unsafe {
                    let w = _mm256_loadu_si256(tw_ptr.add(base) as *const __m256i);
                    let u = _mm256_loadu_si256(a_ptr.add(start) as *const __m256i);
                    let v_raw = _mm256_loadu_si256(a_ptr.add(start + len) as *const __m256i);
                    let v = mont_mul_8x_i32_avx2(v_raw, w, p256, pinv256);
                    _mm256_storeu_si256(
                        a_ptr.add(start) as *mut __m256i,
                        reduce_range_8x_i32_avx2(_mm256_add_epi32(u, v), p256),
                    );
                    _mm256_storeu_si256(
                        a_ptr.add(start + len) as *mut __m256i,
                        reduce_range_8x_i32_avx2(_mm256_sub_epi32(u, v), p256),
                    );
                }
            } else if len == 4 {
                // SAFETY: one 4-lane butterfly covers the whole run; SSE proven.
                unsafe {
                    let w = _mm_loadu_si128(tw_ptr.add(base) as *const __m128i);
                    let u = _mm_loadu_si128(a_ptr.add(start) as *const __m128i);
                    let v_raw = _mm_loadu_si128(a_ptr.add(start + len) as *const __m128i);
                    let v = mont_mul_4x_i32_avx2(v_raw, w, p128, pinv128);
                    _mm_storeu_si128(
                        a_ptr.add(start) as *mut __m128i,
                        reduce_range_4x_i32_avx2(_mm_add_epi32(u, v), p128),
                    );
                    _mm_storeu_si128(
                        a_ptr.add(start + len) as *mut __m128i,
                        reduce_range_4x_i32_avx2(_mm_sub_epi32(u, v), p128),
                    );
                }
            } else {
                for j in 0..len {
                    let w = tw.inv_twiddles[base + j];
                    let u = a[start + j];
                    let v = prime.mul(a[start + j + len], w);
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
                }
            }
            start += 2 * len;
        }
        len *= 2;
    }
}

/// Elementwise `a[i] = mont_mul(a[i], table[i])`, 16 lanes at a time.
///
/// # Safety
///
/// AVX-512F/DQ/BW must be available.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx2")]
unsafe fn mont_mul_table<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    table: &[MontCoeff<i32>; D],
) {
    let p512 = _mm512_set1_epi32(prime.p);
    let pinv512 = _mm512_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;
    let t_ptr = table.as_ptr() as *const i32;
    let mut i = 0;
    while i + 16 <= D {
        // SAFETY: loop bound and this function's safety contract.
        unsafe {
            let ai = _mm512_loadu_si512(a_ptr.add(i) as *const __m512i);
            let ti = _mm512_loadu_si512(t_ptr.add(i) as *const __m512i);
            _mm512_storeu_si512(
                a_ptr.add(i) as *mut __m512i,
                mont_mul_16x_i32_avx512(ai, ti, p512, pinv512),
            );
        }
        i += 16;
    }
    while i < D {
        a[i] = prime.mul(a[i], table[i]);
        i += 1;
    }
}

/// Elementwise `a[i] = mont_mul(a[i], scalar)`, 16 lanes at a time.
///
/// # Safety
///
/// AVX-512F/DQ/BW must be available.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx2")]
unsafe fn mont_mul_broadcast<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    scalar: MontCoeff<i32>,
) {
    let p512 = _mm512_set1_epi32(prime.p);
    let pinv512 = _mm512_set1_epi32(prime.pinv);
    let sv = _mm512_set1_epi32(scalar.raw());
    let a_ptr = a.as_mut_ptr() as *mut i32;
    let mut i = 0;
    while i + 16 <= D {
        // SAFETY: loop bound and this function's safety contract.
        unsafe {
            let ai = _mm512_loadu_si512(a_ptr.add(i) as *const __m512i);
            _mm512_storeu_si512(
                a_ptr.add(i) as *mut __m512i,
                mont_mul_16x_i32_avx512(ai, sv, p512, pinv512),
            );
        }
        i += 16;
    }
    while i < D {
        a[i] = prime.mul(a[i], scalar);
        i += 1;
    }
}

/// Elementwise `a[i] = reduce_range(a[i])`, 16 lanes at a time.
///
/// # Safety
///
/// AVX-512F/DQ/BW must be available.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx2")]
unsafe fn reduce_range_all<const D: usize>(a: &mut [MontCoeff<i32>; D], prime: NttPrime<i32>) {
    let p512 = _mm512_set1_epi32(prime.p);
    let a_ptr = a.as_mut_ptr() as *mut i32;
    let mut i = 0;
    while i + 16 <= D {
        // SAFETY: loop bound and this function's safety contract.
        unsafe {
            let ai = _mm512_loadu_si512(a_ptr.add(i) as *const __m512i);
            _mm512_storeu_si512(
                a_ptr.add(i) as *mut __m512i,
                reduce_range_16x_i32_avx512(ai, p512),
            );
        }
        i += 16;
    }
    while i < D {
        a[i] = prime.reduce_range(a[i]);
        i += 1;
    }
}

/// Width-aware AVX-512 forward negacyclic NTT for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx2")]
pub(super) unsafe fn forward_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    // SAFETY: feature contract forwarded to each helper.
    unsafe {
        mont_mul_table(a, prime, &tw.psi_pows);
        forward_dif_stages(a, prime, tw);
        if !D.is_multiple_of(16) {
            reduce_range_all(a, prime);
        }
    }
}

/// Width-aware AVX-512 inverse negacyclic NTT for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx2")]
pub(super) unsafe fn inverse_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    // SAFETY: feature contract forwarded to each helper.
    unsafe {
        inverse_dit_stages(a, prime, tw);
        mont_mul_table(a, prime, &tw.d_inv_psi_inv);
    }
}

/// Width-aware AVX-512 forward cyclic NTT for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx2")]
pub(super) unsafe fn forward_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    // SAFETY: feature contract forwarded to each helper.
    unsafe {
        forward_dif_stages(a, prime, tw);
        if !D.is_multiple_of(16) {
            reduce_range_all(a, prime);
        }
    }
}

/// Width-aware AVX-512 inverse cyclic NTT for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx2")]
pub(super) unsafe fn inverse_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    // SAFETY: feature contract forwarded to each helper.
    unsafe {
        inverse_dit_stages(a, prime, tw);
        mont_mul_broadcast(a, prime, tw.d_inv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_avx512_ntt() -> bool {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512bw")
    }

    fn random_mont_array<const D: usize>(prime: NttPrime<i32>, seed: u64) -> [MontCoeff<i32>; D] {
        let mut state = seed;
        std::array::from_fn(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = ((state >> 33) as i64 % i64::from(prime.p)) as i32;
            prime.from_canonical(val)
        })
    }

    fn scalar_forward<const D: usize>(
        a: &mut [MontCoeff<i32>; D],
        prime: NttPrime<i32>,
        tw: &NttTwiddles<i32, D>,
    ) {
        for (ai, psi) in a.iter_mut().zip(tw.psi_pows.iter()) {
            *ai = prime.mul(*ai, *psi);
        }
        scalar_forward_cyclic(a, prime, tw);
    }

    fn scalar_inverse<const D: usize>(
        a: &mut [MontCoeff<i32>; D],
        prime: NttPrime<i32>,
        tw: &NttTwiddles<i32, D>,
    ) {
        scalar_inverse_cyclic_stages(a, prime, tw);
        for (ai, fused) in a.iter_mut().zip(tw.d_inv_psi_inv.iter()) {
            *ai = prime.mul(*ai, *fused);
        }
    }

    fn scalar_forward_cyclic<const D: usize>(
        a: &mut [MontCoeff<i32>; D],
        prime: NttPrime<i32>,
        tw: &NttTwiddles<i32, D>,
    ) {
        let mut len = D / 2;
        while len > 0 {
            let base = len - 1;
            let mut start = 0usize;
            while start < D {
                for j in 0..len {
                    let w = tw.fwd_twiddles[base + j];
                    let u = a[start + j];
                    let v = a[start + j + len];
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                }
                start += 2 * len;
            }
            len /= 2;
        }
        prime.reduce_range_in_place(a);
    }

    fn scalar_inverse_cyclic<const D: usize>(
        a: &mut [MontCoeff<i32>; D],
        prime: NttPrime<i32>,
        tw: &NttTwiddles<i32, D>,
    ) {
        scalar_inverse_cyclic_stages(a, prime, tw);
        for ai in a.iter_mut() {
            *ai = prime.mul(*ai, tw.d_inv);
        }
    }

    fn scalar_inverse_cyclic_stages<const D: usize>(
        a: &mut [MontCoeff<i32>; D],
        prime: NttPrime<i32>,
        tw: &NttTwiddles<i32, D>,
    ) {
        let mut len = 1usize;
        while len < D {
            let base = len - 1;
            let mut start = 0usize;
            while start < D {
                for j in 0..len {
                    let w = tw.inv_twiddles[base + j];
                    let u = a[start + j];
                    let v = prime.mul(a[start + j + len], w);
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
                }
                start += 2 * len;
            }
            len *= 2;
        }
    }

    fn check_degree<const D: usize>() {
        if !has_avx512_ntt() {
            return;
        }
        for (idx, raw_prime) in [1073707009_i32, 1073692673, 1073668097]
            .into_iter()
            .enumerate()
        {
            let prime = NttPrime::compute(raw_prime);
            let tw = NttTwiddles::<i32, D>::compute(prime);
            let input = random_mont_array::<D>(prime, 0x5150_9e37 + D as u64 + idx as u64);

            let mut wide = input;
            let mut scalar = input;
            unsafe { forward_ntt_i32(&mut wide, prime, &tw) };
            scalar_forward(&mut scalar, prime, &tw);
            assert_eq!(wide, scalar, "forward mismatch D={D} prime={raw_prime}");

            unsafe { inverse_ntt_i32(&mut wide, prime, &tw) };
            scalar_inverse(&mut scalar, prime, &tw);
            assert_eq!(wide, scalar, "inverse mismatch D={D} prime={raw_prime}");

            let mut wide_cyclic = input;
            let mut scalar_cyclic = input;
            unsafe { forward_ntt_cyclic_i32(&mut wide_cyclic, prime, &tw) };
            scalar_forward_cyclic(&mut scalar_cyclic, prime, &tw);
            assert_eq!(
                wide_cyclic, scalar_cyclic,
                "cyclic forward mismatch D={D} prime={raw_prime}"
            );

            unsafe { inverse_ntt_cyclic_i32(&mut wide_cyclic, prime, &tw) };
            scalar_inverse_cyclic(&mut scalar_cyclic, prime, &tw);
            assert_eq!(
                wide_cyclic, scalar_cyclic,
                "cyclic inverse mismatch D={D} prime={raw_prime}"
            );
        }
    }

    #[test]
    fn wide512_i32_transforms_match_scalar_supported_degrees() {
        check_degree::<32>();
        check_degree::<64>();
        check_degree::<128>();
        check_degree::<256>();
    }
}
