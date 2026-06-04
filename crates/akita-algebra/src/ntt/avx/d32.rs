#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::prime::{MontCoeff, NttPrime};

#[target_feature(enable = "avx2")]
pub(super) unsafe fn forward_ntt_i32(
    a: &mut [MontCoeff<i32>; 32],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, 32>,
) {
    let p_d = _mm_set1_epi32(prime.p);
    let pinv_d = _mm_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;
    let psi_ptr = tw.psi_pows.as_ptr() as *const i32;

    for i in (0..32).step_by(4) {
        unsafe {
            let ai = _mm_loadu_si128(a_ptr.add(i) as *const __m128i);
            let psi = _mm_loadu_si128(psi_ptr.add(i) as *const __m128i);
            _mm_storeu_si128(
                a_ptr.add(i) as *mut __m128i,
                super::mont_mul_4x_i32_avx2(ai, psi, p_d, pinv_d),
            );
        }
    }

    unsafe { forward_ntt_cyclic_i32(a, prime, tw) };
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_ntt_i32(
    a: &mut [MontCoeff<i32>; 32],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, 32>,
) {
    let p_d = _mm_set1_epi32(prime.p);
    let pinv_d = _mm_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    unsafe { inverse_ntt_core_i32(a, prime, tw, p_d, pinv_d) };

    let fused_ptr = tw.d_inv_psi_inv.as_ptr() as *const i32;
    for i in (0..32).step_by(4) {
        unsafe {
            let ai = _mm_loadu_si128(a_ptr.add(i) as *const __m128i);
            let fused = _mm_loadu_si128(fused_ptr.add(i) as *const __m128i);
            _mm_storeu_si128(
                a_ptr.add(i) as *mut __m128i,
                super::mont_mul_4x_i32_avx2(ai, fused, p_d, pinv_d),
            );
        }
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn forward_ntt_cyclic_i32(
    a: &mut [MontCoeff<i32>; 32],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, 32>,
) {
    let p_d = _mm_set1_epi32(prime.p);
    let pinv_d = _mm_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;
    let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;

    macro_rules! fwd4 {
        ($lo:expr, $hi:expr, $tw:expr) => {{
            let u = _mm_loadu_si128(a_ptr.add($lo) as *const __m128i);
            let v = _mm_loadu_si128(a_ptr.add($hi) as *const __m128i);
            let w = _mm_loadu_si128(tw_ptr.add($tw) as *const __m128i);
            _mm_storeu_si128(
                a_ptr.add($lo) as *mut __m128i,
                super::reduce_range_4x_i32_avx2(_mm_add_epi32(u, v), p_d),
            );
            _mm_storeu_si128(
                a_ptr.add($hi) as *mut __m128i,
                super::mont_mul_4x_i32_avx2(_mm_sub_epi32(u, v), w, p_d, pinv_d),
            );
        }};
    }

    unsafe {
        fwd4!(0, 16, 15);
        fwd4!(4, 20, 19);
        fwd4!(8, 24, 23);
        fwd4!(12, 28, 27);
        fwd4!(0, 8, 7);
        fwd4!(4, 12, 11);
        fwd4!(16, 24, 7);
        fwd4!(20, 28, 11);
        fwd4!(0, 4, 3);
        fwd4!(8, 12, 3);
        fwd4!(16, 20, 3);
        fwd4!(24, 28, 3);
    }

    forward_scalar_stage_i32(a, prime, tw, 2);
    forward_scalar_stage_i32(a, prime, tw, 1);
    unsafe { reduce_range_in_place_i32(a, p_d) };
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_ntt_cyclic_i32(
    a: &mut [MontCoeff<i32>; 32],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, 32>,
) {
    let p_d = _mm_set1_epi32(prime.p);
    let pinv_d = _mm_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    unsafe { inverse_ntt_core_i32(a, prime, tw, p_d, pinv_d) };

    let d_inv = _mm_set1_epi32(tw.d_inv.raw());
    for i in (0..32).step_by(4) {
        unsafe {
            let ai = _mm_loadu_si128(a_ptr.add(i) as *const __m128i);
            _mm_storeu_si128(
                a_ptr.add(i) as *mut __m128i,
                super::mont_mul_4x_i32_avx2(ai, d_inv, p_d, pinv_d),
            );
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn inverse_ntt_core_i32(
    a: &mut [MontCoeff<i32>; 32],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, 32>,
    p_d: __m128i,
    pinv_d: __m128i,
) {
    let a_ptr = a.as_mut_ptr() as *mut i32;
    let tw_ptr = tw.inv_twiddles.as_ptr() as *const i32;

    inverse_scalar_stage_i32(a, prime, tw, 1);
    inverse_scalar_stage_i32(a, prime, tw, 2);

    macro_rules! inv4 {
        ($lo:expr, $hi:expr, $tw:expr) => {{
            let w = _mm_loadu_si128(tw_ptr.add($tw) as *const __m128i);
            let u = _mm_loadu_si128(a_ptr.add($lo) as *const __m128i);
            let v_raw = _mm_loadu_si128(a_ptr.add($hi) as *const __m128i);
            let v = super::mont_mul_4x_i32_avx2(v_raw, w, p_d, pinv_d);
            _mm_storeu_si128(
                a_ptr.add($lo) as *mut __m128i,
                super::reduce_range_4x_i32_avx2(_mm_add_epi32(u, v), p_d),
            );
            _mm_storeu_si128(
                a_ptr.add($hi) as *mut __m128i,
                super::reduce_range_4x_i32_avx2(_mm_sub_epi32(u, v), p_d),
            );
        }};
    }

    unsafe {
        inv4!(0, 4, 3);
        inv4!(8, 12, 3);
        inv4!(16, 20, 3);
        inv4!(24, 28, 3);
        inv4!(0, 8, 7);
        inv4!(4, 12, 11);
        inv4!(16, 24, 7);
        inv4!(20, 28, 11);
        inv4!(0, 16, 15);
        inv4!(4, 20, 19);
        inv4!(8, 24, 23);
        inv4!(12, 28, 27);
    }
}

fn forward_scalar_stage_i32(
    a: &mut [MontCoeff<i32>; 32],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, 32>,
    len: usize,
) {
    let twiddle_base = len - 1;
    let mut start = 0usize;
    while start < 32 {
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
}

fn inverse_scalar_stage_i32(
    a: &mut [MontCoeff<i32>; 32],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, 32>,
    len: usize,
) {
    let twiddle_base = len - 1;
    let mut start = 0usize;
    while start < 32 {
        for j in 0..len {
            let w = tw.inv_twiddles[twiddle_base + j];
            let u = a[start + j];
            let v = prime.mul(a[start + j + len], w);
            let sum = u.raw().wrapping_add(v.raw());
            let diff = u.raw().wrapping_sub(v.raw());
            a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
            a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
        }
        start += 2 * len;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn reduce_range_in_place_i32(a: &mut [MontCoeff<i32>; 32], p: __m128i) {
    let a_ptr = a.as_mut_ptr() as *mut i32;
    for i in (0..32).step_by(4) {
        unsafe {
            let ai = _mm_loadu_si128(a_ptr.add(i) as *const __m128i);
            _mm_storeu_si128(
                a_ptr.add(i) as *mut __m128i,
                super::reduce_range_4x_i32_avx2(ai, p),
            );
        }
    }
}
