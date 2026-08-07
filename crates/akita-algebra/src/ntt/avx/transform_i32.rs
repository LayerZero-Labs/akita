#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::montgomery::{
    forward_dif_tail_i32_avx2, inverse_dit_head_i32_avx2, mont_mul_4x_i32_avx2,
    mont_mul_8x_i32_avx2, reduce_range_4x_i32_avx2, reduce_range_8x_i32_avx2,
};
use super::{d32, wide512};
use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::forward_dif_tail_eligible;
use crate::ntt::prime::{MontCoeff, NttPrime};

/// Fuse two adjacent forward DIF stages while their four data quarters are in
/// registers. The arithmetic and reduction points are identical to executing
/// stages `len` and `len / 2` separately; only the intermediate array pass is
/// removed.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn forward_dif_radix4_stage_i32_avx2(
    a_ptr: *mut i32,
    d: usize,
    len: usize,
    tw_ptr: *const i32,
    p128: __m128i,
    pinv128: __m128i,
    p256: __m256i,
    pinv256: __m256i,
) {
    let half = len / 2;
    let outer_twiddle_base = len - 1;
    let inner_twiddle_base = half - 1;
    let mut start = 0usize;
    while start < d {
        let mut j = 0usize;
        if half >= 8 {
            while j < half {
                // SAFETY: the caller supplies a valid power-of-two DIF stage;
                // all four quarters and twiddle vectors are in bounds.
                unsafe {
                    let q0 = _mm256_loadu_si256(a_ptr.add(start + j) as *const __m256i);
                    let q1 = _mm256_loadu_si256(a_ptr.add(start + j + half) as *const __m256i);
                    let q2 = _mm256_loadu_si256(a_ptr.add(start + j + len) as *const __m256i);
                    let q3 =
                        _mm256_loadu_si256(a_ptr.add(start + j + len + half) as *const __m256i);
                    let outer0 =
                        _mm256_loadu_si256(tw_ptr.add(outer_twiddle_base + j) as *const __m256i);
                    let outer1 = _mm256_loadu_si256(
                        tw_ptr.add(outer_twiddle_base + half + j) as *const __m256i
                    );
                    let inner =
                        _mm256_loadu_si256(tw_ptr.add(inner_twiddle_base + j) as *const __m256i);

                    let sum0 = reduce_range_8x_i32_avx2(_mm256_add_epi32(q0, q2), p256);
                    let diff0 =
                        mont_mul_8x_i32_avx2(_mm256_sub_epi32(q0, q2), outer0, p256, pinv256);
                    let sum1 = reduce_range_8x_i32_avx2(_mm256_add_epi32(q1, q3), p256);
                    let diff1 =
                        mont_mul_8x_i32_avx2(_mm256_sub_epi32(q1, q3), outer1, p256, pinv256);

                    _mm256_storeu_si256(
                        a_ptr.add(start + j) as *mut __m256i,
                        reduce_range_8x_i32_avx2(_mm256_add_epi32(sum0, sum1), p256),
                    );
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + half) as *mut __m256i,
                        mont_mul_8x_i32_avx2(_mm256_sub_epi32(sum0, sum1), inner, p256, pinv256),
                    );
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + len) as *mut __m256i,
                        reduce_range_8x_i32_avx2(_mm256_add_epi32(diff0, diff1), p256),
                    );
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + len + half) as *mut __m256i,
                        mont_mul_8x_i32_avx2(_mm256_sub_epi32(diff0, diff1), inner, p256, pinv256),
                    );
                }
                j += 8;
            }
        } else {
            debug_assert_eq!(half, 4);
            // SAFETY: the caller supplies a valid power-of-two DIF stage; all
            // four quarters and twiddle vectors are in bounds.
            unsafe {
                let q0 = _mm_loadu_si128(a_ptr.add(start) as *const __m128i);
                let q1 = _mm_loadu_si128(a_ptr.add(start + half) as *const __m128i);
                let q2 = _mm_loadu_si128(a_ptr.add(start + len) as *const __m128i);
                let q3 = _mm_loadu_si128(a_ptr.add(start + len + half) as *const __m128i);
                let outer0 = _mm_loadu_si128(tw_ptr.add(outer_twiddle_base) as *const __m128i);
                let outer1 =
                    _mm_loadu_si128(tw_ptr.add(outer_twiddle_base + half) as *const __m128i);
                let inner = _mm_loadu_si128(tw_ptr.add(inner_twiddle_base) as *const __m128i);

                let sum0 = reduce_range_4x_i32_avx2(_mm_add_epi32(q0, q2), p128);
                let diff0 = mont_mul_4x_i32_avx2(_mm_sub_epi32(q0, q2), outer0, p128, pinv128);
                let sum1 = reduce_range_4x_i32_avx2(_mm_add_epi32(q1, q3), p128);
                let diff1 = mont_mul_4x_i32_avx2(_mm_sub_epi32(q1, q3), outer1, p128, pinv128);

                _mm_storeu_si128(
                    a_ptr.add(start) as *mut __m128i,
                    reduce_range_4x_i32_avx2(_mm_add_epi32(sum0, sum1), p128),
                );
                _mm_storeu_si128(
                    a_ptr.add(start + half) as *mut __m128i,
                    mont_mul_4x_i32_avx2(_mm_sub_epi32(sum0, sum1), inner, p128, pinv128),
                );
                _mm_storeu_si128(
                    a_ptr.add(start + len) as *mut __m128i,
                    reduce_range_4x_i32_avx2(_mm_add_epi32(diff0, diff1), p128),
                );
                _mm_storeu_si128(
                    a_ptr.add(start + len + half) as *mut __m128i,
                    mont_mul_4x_i32_avx2(_mm_sub_epi32(diff0, diff1), inner, p128, pinv128),
                );
            }
        }
        start += 2 * len;
    }
}

/// Fuse two adjacent inverse DIT stages while their four data quarters are in
/// registers. This preserves the radix-2 arithmetic and reduction order while
/// removing the intermediate array pass.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn inverse_dit_radix4_stage_i32_avx2(
    a_ptr: *mut i32,
    d: usize,
    len: usize,
    tw_ptr: *const i32,
    p128: __m128i,
    pinv128: __m128i,
    p256: __m256i,
    pinv256: __m256i,
) {
    let outer_len = 2 * len;
    let inner_twiddle_base = len - 1;
    let outer_twiddle_base = outer_len - 1;
    let mut start = 0usize;
    while start < d {
        let mut j = 0usize;
        if len >= 8 {
            while j < len {
                // SAFETY: the caller supplies two valid adjacent DIT stages;
                // all four quarters and twiddle vectors are in bounds.
                unsafe {
                    let q0 = _mm256_loadu_si256(a_ptr.add(start + j) as *const __m256i);
                    let q1 = _mm256_loadu_si256(a_ptr.add(start + j + len) as *const __m256i);
                    let q2 = _mm256_loadu_si256(a_ptr.add(start + j + outer_len) as *const __m256i);
                    let q3 = _mm256_loadu_si256(
                        a_ptr.add(start + j + outer_len + len) as *const __m256i
                    );
                    let inner =
                        _mm256_loadu_si256(tw_ptr.add(inner_twiddle_base + j) as *const __m256i);
                    let outer0 =
                        _mm256_loadu_si256(tw_ptr.add(outer_twiddle_base + j) as *const __m256i);
                    let outer1 = _mm256_loadu_si256(
                        tw_ptr.add(outer_twiddle_base + len + j) as *const __m256i
                    );

                    let inner1 = mont_mul_8x_i32_avx2(q1, inner, p256, pinv256);
                    let sum0 = reduce_range_8x_i32_avx2(_mm256_add_epi32(q0, inner1), p256);
                    let diff0 = reduce_range_8x_i32_avx2(_mm256_sub_epi32(q0, inner1), p256);
                    let inner3 = mont_mul_8x_i32_avx2(q3, inner, p256, pinv256);
                    let sum1 = reduce_range_8x_i32_avx2(_mm256_add_epi32(q2, inner3), p256);
                    let diff1 = reduce_range_8x_i32_avx2(_mm256_sub_epi32(q2, inner3), p256);
                    let outer_sum = mont_mul_8x_i32_avx2(sum1, outer0, p256, pinv256);
                    let outer_diff = mont_mul_8x_i32_avx2(diff1, outer1, p256, pinv256);

                    _mm256_storeu_si256(
                        a_ptr.add(start + j) as *mut __m256i,
                        reduce_range_8x_i32_avx2(_mm256_add_epi32(sum0, outer_sum), p256),
                    );
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + len) as *mut __m256i,
                        reduce_range_8x_i32_avx2(_mm256_add_epi32(diff0, outer_diff), p256),
                    );
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + outer_len) as *mut __m256i,
                        reduce_range_8x_i32_avx2(_mm256_sub_epi32(sum0, outer_sum), p256),
                    );
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + outer_len + len) as *mut __m256i,
                        reduce_range_8x_i32_avx2(_mm256_sub_epi32(diff0, outer_diff), p256),
                    );
                }
                j += 8;
            }
        } else {
            debug_assert_eq!(len, 4);
            // SAFETY: the caller supplies two valid adjacent DIT stages; all
            // four quarters and twiddle vectors are in bounds.
            unsafe {
                let q0 = _mm_loadu_si128(a_ptr.add(start) as *const __m128i);
                let q1 = _mm_loadu_si128(a_ptr.add(start + len) as *const __m128i);
                let q2 = _mm_loadu_si128(a_ptr.add(start + outer_len) as *const __m128i);
                let q3 = _mm_loadu_si128(a_ptr.add(start + outer_len + len) as *const __m128i);
                let inner = _mm_loadu_si128(tw_ptr.add(inner_twiddle_base) as *const __m128i);
                let outer0 = _mm_loadu_si128(tw_ptr.add(outer_twiddle_base) as *const __m128i);
                let outer1 =
                    _mm_loadu_si128(tw_ptr.add(outer_twiddle_base + len) as *const __m128i);

                let inner1 = mont_mul_4x_i32_avx2(q1, inner, p128, pinv128);
                let sum0 = reduce_range_4x_i32_avx2(_mm_add_epi32(q0, inner1), p128);
                let diff0 = reduce_range_4x_i32_avx2(_mm_sub_epi32(q0, inner1), p128);
                let inner3 = mont_mul_4x_i32_avx2(q3, inner, p128, pinv128);
                let sum1 = reduce_range_4x_i32_avx2(_mm_add_epi32(q2, inner3), p128);
                let diff1 = reduce_range_4x_i32_avx2(_mm_sub_epi32(q2, inner3), p128);
                let outer_sum = mont_mul_4x_i32_avx2(sum1, outer0, p128, pinv128);
                let outer_diff = mont_mul_4x_i32_avx2(diff1, outer1, p128, pinv128);

                _mm_storeu_si128(
                    a_ptr.add(start) as *mut __m128i,
                    reduce_range_4x_i32_avx2(_mm_add_epi32(sum0, outer_sum), p128),
                );
                _mm_storeu_si128(
                    a_ptr.add(start + len) as *mut __m128i,
                    reduce_range_4x_i32_avx2(_mm_add_epi32(diff0, outer_diff), p128),
                );
                _mm_storeu_si128(
                    a_ptr.add(start + outer_len) as *mut __m128i,
                    reduce_range_4x_i32_avx2(_mm_sub_epi32(sum0, outer_sum), p128),
                );
                _mm_storeu_si128(
                    a_ptr.add(start + outer_len + len) as *mut __m128i,
                    reduce_range_4x_i32_avx2(_mm_sub_epi32(diff0, outer_diff), p128),
                );
            }
        }
        start += 2 * outer_len;
    }
}

/// AVX2 forward negacyclic NTT for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn forward_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
    use_avx512: bool,
) {
    if D == 32 {
        // SAFETY: the branch proves the concrete array and twiddle degree.
        unsafe {
            return d32::forward_ntt_i32(
                &mut *(a as *mut _ as *mut [MontCoeff<i32>; 32]),
                prime,
                &*(tw as *const _ as *const NttTwiddles<i32, 32>),
            );
        }
    }
    if use_avx512 {
        // SAFETY: Avx512 mode is selected only when AVX-512F/DQ/BW were detected.
        unsafe {
            return wide512::forward_ntt_i32(a, prime, tw);
        }
    }

    let p_d = _mm_set1_epi32(prime.p);
    let pinv_d = _mm_set1_epi32(prime.pinv);
    let p256 = _mm256_set1_epi32(prime.p);
    let pinv256 = _mm256_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let psi_ptr = tw.psi_pows.as_ptr() as *const i32;
    let mut i = 0;
    while i + 8 <= D {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let ai = _mm256_loadu_si256(a_ptr.add(i) as *const __m256i);
            let psi = _mm256_loadu_si256(psi_ptr.add(i) as *const __m256i);
            _mm256_storeu_si256(
                a_ptr.add(i) as *mut __m256i,
                mont_mul_8x_i32_avx2(ai, psi, p256, pinv256),
            );
        }
        i += 8;
    }
    while i < D {
        a[i] = prime.mul(a[i], tw.psi_pows[i]);
        i += 1;
    }

    let mut len = D / 2;
    while len >= 8 {
        // SAFETY: `len` and `len / 2` are valid adjacent DIF stages.
        unsafe {
            forward_dif_radix4_stage_i32_avx2(
                a_ptr,
                D,
                len,
                tw.fwd_twiddles.as_ptr() as *const i32,
                p_d,
                pinv_d,
                p256,
                pinv256,
            );
        }
        len /= 4;
    }
    while len >= 4 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            let mut j = 0usize;
            if len >= 8 {
                while j < len {
                    // SAFETY: guaranteed by stage bounds and this function's safety contract.
                    unsafe {
                        let u = _mm256_loadu_si256(a_ptr.add(start + j) as *const __m256i);
                        let v = _mm256_loadu_si256(a_ptr.add(start + j + len) as *const __m256i);
                        let w = _mm256_loadu_si256(tw_ptr.add(twiddle_base + j) as *const __m256i);
                        _mm256_storeu_si256(
                            a_ptr.add(start + j) as *mut __m256i,
                            reduce_range_8x_i32_avx2(_mm256_add_epi32(u, v), p256),
                        );
                        _mm256_storeu_si256(
                            a_ptr.add(start + j + len) as *mut __m256i,
                            mont_mul_8x_i32_avx2(_mm256_sub_epi32(u, v), w, p256, pinv256),
                        );
                    }
                    j += 8;
                }
            } else {
                while j < len {
                    // SAFETY: guaranteed by stage bounds and this function's safety contract.
                    unsafe {
                        let u = _mm_loadu_si128(a_ptr.add(start + j) as *const __m128i);
                        let v = _mm_loadu_si128(a_ptr.add(start + j + len) as *const __m128i);
                        let w = _mm_loadu_si128(tw_ptr.add(twiddle_base + j) as *const __m128i);
                        _mm_storeu_si128(
                            a_ptr.add(start + j) as *mut __m128i,
                            reduce_range_4x_i32_avx2(_mm_add_epi32(u, v), p_d),
                        );
                        _mm_storeu_si128(
                            a_ptr.add(start + j + len) as *mut __m128i,
                            mont_mul_4x_i32_avx2(_mm_sub_epi32(u, v), w, p_d, pinv_d),
                        );
                    }
                    j += 4;
                }
            }
            start += 2 * len;
        }
        len /= 2;
    }

    if forward_dif_tail_eligible::<D>() {
        // SAFETY: guaranteed by this function's safety contract.
        unsafe {
            forward_dif_tail_i32_avx2::<D>(
                a_ptr,
                tw.fwd_twiddles.as_ptr() as *const i32,
                p_d,
                pinv_d,
            );
        }
    } else {
        while len > 0 {
            let twiddle_base = len - 1;
            let mut start = 0usize;
            while start < D {
                for j in 0..len {
                    let w = tw.fwd_twiddles[twiddle_base + j];
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
        // SAFETY: guaranteed by this function's safety contract.
        unsafe { reduce_range_in_place_i32(a, prime, p_d) };
    }
}

/// AVX2 inverse negacyclic NTT for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn inverse_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
    use_avx512: bool,
) {
    if D == 32 {
        // SAFETY: the branch proves the concrete array and twiddle degree.
        unsafe {
            return d32::inverse_ntt_i32(
                &mut *(a as *mut _ as *mut [MontCoeff<i32>; 32]),
                prime,
                &*(tw as *const _ as *const NttTwiddles<i32, 32>),
            );
        }
    }
    if use_avx512 {
        // SAFETY: Avx512 mode is selected only when AVX-512F/DQ/BW were detected.
        unsafe {
            return wide512::inverse_ntt_i32(a, prime, tw);
        }
    }

    let p_d = _mm_set1_epi32(prime.p);
    let pinv_d = _mm_set1_epi32(prime.pinv);
    let p256 = _mm256_set1_epi32(prime.p);
    let pinv256 = _mm256_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = 1usize;
    if D.is_multiple_of(16) {
        // SAFETY: the divisibility check covers every 16-element block.
        unsafe {
            inverse_dit_head_i32_avx2::<D>(
                a_ptr,
                tw.inv_twiddles.as_ptr() as *const i32,
                p_d,
                pinv_d,
            );
        }
        len = 4;
    }
    while len < D / 2 {
        // SAFETY: `len` and `2 * len` are valid adjacent DIT stages.
        unsafe {
            inverse_dit_radix4_stage_i32_avx2(
                a_ptr,
                D,
                len,
                tw.inv_twiddles.as_ptr() as *const i32,
                p_d,
                pinv_d,
                p256,
                pinv256,
            );
        }
        len *= 4;
    }
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 8 {
                let mut j = 0usize;
                while j < len {
                    // SAFETY: guaranteed by stage bounds and this function's safety contract.
                    unsafe {
                        let w = _mm256_loadu_si256(tw_ptr.add(twiddle_base + j) as *const __m256i);
                        let u = _mm256_loadu_si256(a_ptr.add(start + j) as *const __m256i);
                        let v_raw =
                            _mm256_loadu_si256(a_ptr.add(start + j + len) as *const __m256i);
                        let v = mont_mul_8x_i32_avx2(v_raw, w, p256, pinv256);
                        _mm256_storeu_si256(
                            a_ptr.add(start + j) as *mut __m256i,
                            reduce_range_8x_i32_avx2(_mm256_add_epi32(u, v), p256),
                        );
                        _mm256_storeu_si256(
                            a_ptr.add(start + j + len) as *mut __m256i,
                            reduce_range_8x_i32_avx2(_mm256_sub_epi32(u, v), p256),
                        );
                    }
                    j += 8;
                }
            } else if len >= 4 {
                let mut j = 0usize;
                while j < len {
                    // SAFETY: guaranteed by stage bounds and this function's safety contract.
                    unsafe {
                        let w = _mm_loadu_si128(tw_ptr.add(twiddle_base + j) as *const __m128i);
                        let u = _mm_loadu_si128(a_ptr.add(start + j) as *const __m128i);
                        let v_raw = _mm_loadu_si128(a_ptr.add(start + j + len) as *const __m128i);
                        let v = mont_mul_4x_i32_avx2(v_raw, w, p_d, pinv_d);
                        _mm_storeu_si128(
                            a_ptr.add(start + j) as *mut __m128i,
                            reduce_range_4x_i32_avx2(_mm_add_epi32(u, v), p_d),
                        );
                        _mm_storeu_si128(
                            a_ptr.add(start + j + len) as *mut __m128i,
                            reduce_range_4x_i32_avx2(_mm_sub_epi32(u, v), p_d),
                        );
                    }
                    j += 4;
                }
            } else {
                for j in 0..len {
                    let w = tw.inv_twiddles[twiddle_base + j];
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

    let fused_ptr = tw.d_inv_psi_inv.as_ptr() as *const i32;
    let mut i = 0;
    while i + 8 <= D {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let ai = _mm256_loadu_si256(a_ptr.add(i) as *const __m256i);
            let fused = _mm256_loadu_si256(fused_ptr.add(i) as *const __m256i);
            _mm256_storeu_si256(
                a_ptr.add(i) as *mut __m256i,
                mont_mul_8x_i32_avx2(ai, fused, p256, pinv256),
            );
        }
        i += 8;
    }
    while i < D {
        a[i] = prime.mul(a[i], tw.d_inv_psi_inv[i]);
        i += 1;
    }
}

/// AVX2 forward cyclic NTT for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn forward_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
    use_avx512: bool,
) {
    if D == 32 {
        // SAFETY: the branch proves the concrete array and twiddle degree.
        unsafe {
            return d32::forward_ntt_cyclic_i32(
                &mut *(a as *mut _ as *mut [MontCoeff<i32>; 32]),
                prime,
                &*(tw as *const _ as *const NttTwiddles<i32, 32>),
            );
        }
    }
    if use_avx512 {
        // SAFETY: Avx512 mode is selected only when AVX-512F/DQ/BW were detected.
        unsafe {
            return wide512::forward_ntt_cyclic_i32(a, prime, tw);
        }
    }

    let p_d = _mm_set1_epi32(prime.p);
    let pinv_d = _mm_set1_epi32(prime.pinv);
    let p256 = _mm256_set1_epi32(prime.p);
    let pinv256 = _mm256_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = D / 2;
    while len >= 8 {
        // SAFETY: `len` and `len / 2` are valid adjacent DIF stages.
        unsafe {
            forward_dif_radix4_stage_i32_avx2(
                a_ptr,
                D,
                len,
                tw.fwd_twiddles.as_ptr() as *const i32,
                p_d,
                pinv_d,
                p256,
                pinv256,
            );
        }
        len /= 4;
    }
    while len >= 4 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            let mut j = 0usize;
            if len >= 8 {
                while j < len {
                    // SAFETY: guaranteed by stage bounds and this function's safety contract.
                    unsafe {
                        let u = _mm256_loadu_si256(a_ptr.add(start + j) as *const __m256i);
                        let v = _mm256_loadu_si256(a_ptr.add(start + j + len) as *const __m256i);
                        let w = _mm256_loadu_si256(tw_ptr.add(twiddle_base + j) as *const __m256i);
                        _mm256_storeu_si256(
                            a_ptr.add(start + j) as *mut __m256i,
                            reduce_range_8x_i32_avx2(_mm256_add_epi32(u, v), p256),
                        );
                        _mm256_storeu_si256(
                            a_ptr.add(start + j + len) as *mut __m256i,
                            mont_mul_8x_i32_avx2(_mm256_sub_epi32(u, v), w, p256, pinv256),
                        );
                    }
                    j += 8;
                }
            } else {
                while j < len {
                    // SAFETY: guaranteed by stage bounds and this function's safety contract.
                    unsafe {
                        let u = _mm_loadu_si128(a_ptr.add(start + j) as *const __m128i);
                        let v = _mm_loadu_si128(a_ptr.add(start + j + len) as *const __m128i);
                        let w = _mm_loadu_si128(tw_ptr.add(twiddle_base + j) as *const __m128i);
                        _mm_storeu_si128(
                            a_ptr.add(start + j) as *mut __m128i,
                            reduce_range_4x_i32_avx2(_mm_add_epi32(u, v), p_d),
                        );
                        _mm_storeu_si128(
                            a_ptr.add(start + j + len) as *mut __m128i,
                            mont_mul_4x_i32_avx2(_mm_sub_epi32(u, v), w, p_d, pinv_d),
                        );
                    }
                    j += 4;
                }
            }
            start += 2 * len;
        }
        len /= 2;
    }

    if forward_dif_tail_eligible::<D>() {
        // SAFETY: guaranteed by this function's safety contract.
        unsafe {
            forward_dif_tail_i32_avx2::<D>(
                a_ptr,
                tw.fwd_twiddles.as_ptr() as *const i32,
                p_d,
                pinv_d,
            );
        }
    } else {
        while len > 0 {
            let twiddle_base = len - 1;
            let mut start = 0usize;
            while start < D {
                for j in 0..len {
                    let w = tw.fwd_twiddles[twiddle_base + j];
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
        // SAFETY: guaranteed by this function's safety contract.
        unsafe { reduce_range_in_place_i32(a, prime, p_d) };
    }
}

/// AVX2 inverse cyclic NTT for one `i32` CRT limb.
///
/// # Safety
///
/// The caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn inverse_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
    use_avx512: bool,
) {
    if D == 32 {
        // SAFETY: the branch proves the concrete array and twiddle degree.
        unsafe {
            return d32::inverse_ntt_cyclic_i32(
                &mut *(a as *mut _ as *mut [MontCoeff<i32>; 32]),
                prime,
                &*(tw as *const _ as *const NttTwiddles<i32, 32>),
            );
        }
    }
    if use_avx512 {
        // SAFETY: Avx512 mode is selected only when AVX-512F/DQ/BW were detected.
        unsafe {
            return wide512::inverse_ntt_cyclic_i32(a, prime, tw);
        }
    }

    let p_d = _mm_set1_epi32(prime.p);
    let pinv_d = _mm_set1_epi32(prime.pinv);
    let p256 = _mm256_set1_epi32(prime.p);
    let pinv256 = _mm256_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = 1usize;
    if D.is_multiple_of(16) {
        // SAFETY: the divisibility check covers every 16-element block.
        unsafe {
            inverse_dit_head_i32_avx2::<D>(
                a_ptr,
                tw.inv_twiddles.as_ptr() as *const i32,
                p_d,
                pinv_d,
            );
        }
        len = 4;
    }
    while len < D / 2 {
        // SAFETY: `len` and `2 * len` are valid adjacent DIT stages.
        unsafe {
            inverse_dit_radix4_stage_i32_avx2(
                a_ptr,
                D,
                len,
                tw.inv_twiddles.as_ptr() as *const i32,
                p_d,
                pinv_d,
                p256,
                pinv256,
            );
        }
        len *= 4;
    }
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 8 {
                let mut j = 0usize;
                while j < len {
                    // SAFETY: guaranteed by stage bounds and this function's safety contract.
                    unsafe {
                        let w = _mm256_loadu_si256(tw_ptr.add(twiddle_base + j) as *const __m256i);
                        let u = _mm256_loadu_si256(a_ptr.add(start + j) as *const __m256i);
                        let v_raw =
                            _mm256_loadu_si256(a_ptr.add(start + j + len) as *const __m256i);
                        let v = mont_mul_8x_i32_avx2(v_raw, w, p256, pinv256);
                        _mm256_storeu_si256(
                            a_ptr.add(start + j) as *mut __m256i,
                            reduce_range_8x_i32_avx2(_mm256_add_epi32(u, v), p256),
                        );
                        _mm256_storeu_si256(
                            a_ptr.add(start + j + len) as *mut __m256i,
                            reduce_range_8x_i32_avx2(_mm256_sub_epi32(u, v), p256),
                        );
                    }
                    j += 8;
                }
            } else if len >= 4 {
                let mut j = 0usize;
                while j < len {
                    // SAFETY: guaranteed by stage bounds and this function's safety contract.
                    unsafe {
                        let w = _mm_loadu_si128(tw_ptr.add(twiddle_base + j) as *const __m128i);
                        let u = _mm_loadu_si128(a_ptr.add(start + j) as *const __m128i);
                        let v_raw = _mm_loadu_si128(a_ptr.add(start + j + len) as *const __m128i);
                        let v = mont_mul_4x_i32_avx2(v_raw, w, p_d, pinv_d);
                        _mm_storeu_si128(
                            a_ptr.add(start + j) as *mut __m128i,
                            reduce_range_4x_i32_avx2(_mm_add_epi32(u, v), p_d),
                        );
                        _mm_storeu_si128(
                            a_ptr.add(start + j + len) as *mut __m128i,
                            reduce_range_4x_i32_avx2(_mm_sub_epi32(u, v), p_d),
                        );
                    }
                    j += 4;
                }
            } else {
                for j in 0..len {
                    let w = tw.inv_twiddles[twiddle_base + j];
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

    let d_inv = _mm256_set1_epi32(tw.d_inv.raw());
    let mut i = 0;
    while i + 8 <= D {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let ai = _mm256_loadu_si256(a_ptr.add(i) as *const __m256i);
            _mm256_storeu_si256(
                a_ptr.add(i) as *mut __m256i,
                mont_mul_8x_i32_avx2(ai, d_inv, p256, pinv256),
            );
        }
        i += 8;
    }
    while i < D {
        a[i] = prime.mul(a[i], tw.d_inv);
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn reduce_range_in_place_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    p: __m128i,
) {
    let a_ptr = a.as_mut_ptr() as *mut i32;
    let mut i = 0;
    while i + 4 <= D {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let ai = _mm_loadu_si128(a_ptr.add(i) as *const __m128i);
            _mm_storeu_si128(
                a_ptr.add(i) as *mut __m128i,
                reduce_range_4x_i32_avx2(ai, p),
            );
        }
        i += 4;
    }
    while i < D {
        a[i] = prime.reduce_range(a[i]);
        i += 1;
    }
}
