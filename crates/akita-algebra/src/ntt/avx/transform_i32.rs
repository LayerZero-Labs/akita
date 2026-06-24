#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::montgomery::{
    forward_dif_tail_i32_avx2, mont_mul_4x_i32_avx2, reduce_range_4x_i32_avx2,
};
use super::{avx_ntt_mode, d32, wide512, AvxNttMode};
use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::forward_dif_tail_eligible;
use crate::ntt::prime::{MontCoeff, NttPrime};

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
    if matches!(avx_ntt_mode(), Some(AvxNttMode::Avx512)) {
        // SAFETY: Avx512 mode is selected only when AVX-512F/DQ/BW were detected.
        unsafe {
            return wide512::forward_ntt_i32(a, prime, tw);
        }
    }

    let p_d = _mm_set1_epi32(prime.p);
    let pinv_d = _mm_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let psi_ptr = tw.psi_pows.as_ptr() as *const i32;
    let mut i = 0;
    while i + 4 <= D {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let ai = _mm_loadu_si128(a_ptr.add(i) as *const __m128i);
            let psi = _mm_loadu_si128(psi_ptr.add(i) as *const __m128i);
            _mm_storeu_si128(
                a_ptr.add(i) as *mut __m128i,
                mont_mul_4x_i32_avx2(ai, psi, p_d, pinv_d),
            );
        }
        i += 4;
    }
    while i < D {
        a[i] = prime.mul(a[i], tw.psi_pows[i]);
        i += 1;
    }

    let mut len = D / 2;
    while len >= 4 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            let mut j = 0usize;
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
    if matches!(avx_ntt_mode(), Some(AvxNttMode::Avx512)) {
        // SAFETY: Avx512 mode is selected only when AVX-512F/DQ/BW were detected.
        unsafe {
            return wide512::inverse_ntt_i32(a, prime, tw);
        }
    }

    let p_d = _mm_set1_epi32(prime.p);
    let pinv_d = _mm_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
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
    while i + 4 <= D {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let ai = _mm_loadu_si128(a_ptr.add(i) as *const __m128i);
            let fused = _mm_loadu_si128(fused_ptr.add(i) as *const __m128i);
            _mm_storeu_si128(
                a_ptr.add(i) as *mut __m128i,
                mont_mul_4x_i32_avx2(ai, fused, p_d, pinv_d),
            );
        }
        i += 4;
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
    if matches!(avx_ntt_mode(), Some(AvxNttMode::Avx512)) {
        // SAFETY: Avx512 mode is selected only when AVX-512F/DQ/BW were detected.
        unsafe {
            return wide512::forward_ntt_cyclic_i32(a, prime, tw);
        }
    }

    let p_d = _mm_set1_epi32(prime.p);
    let pinv_d = _mm_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = D / 2;
    while len >= 4 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            let mut j = 0usize;
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
    if matches!(avx_ntt_mode(), Some(AvxNttMode::Avx512)) {
        // SAFETY: Avx512 mode is selected only when AVX-512F/DQ/BW were detected.
        unsafe {
            return wide512::inverse_ntt_cyclic_i32(a, prime, tw);
        }
    }

    let p_d = _mm_set1_epi32(prime.p);
    let pinv_d = _mm_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
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

    let d_inv = _mm_set1_epi32(tw.d_inv.raw());
    let mut i = 0;
    while i + 4 <= D {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let ai = _mm_loadu_si128(a_ptr.add(i) as *const __m128i);
            _mm_storeu_si128(
                a_ptr.add(i) as *mut __m128i,
                mont_mul_4x_i32_avx2(ai, d_inv, p_d, pinv_d),
            );
        }
        i += 4;
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
