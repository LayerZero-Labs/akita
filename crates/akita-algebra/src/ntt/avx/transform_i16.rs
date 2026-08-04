#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::montgomery::{mont_mul_16x_i16_avx2, reduce_range_16x_i16_avx2};
use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::prime::{MontCoeff, NttPrime};

/// AVX2 forward negacyclic NTT for one i16 CRT limb.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn forward_ntt_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p = _mm256_set1_epi16(prime.p);
    let pinv = _mm256_set1_epi16(prime.pinv);
    let ptr = a.as_mut_ptr() as *mut i16;

    let mut i = 0usize;
    while i + 16 <= D {
        unsafe {
            let values = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            let psi = _mm256_loadu_si256(tw.psi_pows.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(
                ptr.add(i) as *mut __m256i,
                mont_mul_16x_i16_avx2(values, psi, p, pinv),
            );
        }
        i += 16;
    }
    while i < D {
        a[i] = prime.mul(a[i], tw.psi_pows[i]);
        i += 1;
    }

    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let mut start = 0usize;
        while start < D {
            let mut j = 0usize;
            while j + 16 <= len {
                unsafe {
                    let u = _mm256_loadu_si256(ptr.add(start + j) as *const __m256i);
                    let v = _mm256_loadu_si256(ptr.add(start + j + len) as *const __m256i);
                    let w = _mm256_loadu_si256(
                        tw.fwd_twiddles.as_ptr().add(twiddle_base + j) as *const __m256i
                    );
                    _mm256_storeu_si256(
                        ptr.add(start + j) as *mut __m256i,
                        reduce_range_16x_i16_avx2(_mm256_add_epi16(u, v), p),
                    );
                    _mm256_storeu_si256(
                        ptr.add(start + j + len) as *mut __m256i,
                        mont_mul_16x_i16_avx2(_mm256_sub_epi16(u, v), w, p, pinv),
                    );
                }
                j += 16;
            }
            while j < len {
                let w = tw.fwd_twiddles[twiddle_base + j];
                let u = a[start + j];
                let v = a[start + j + len];
                a[start + j] =
                    prime.reduce_range(MontCoeff::from_raw(u.raw().wrapping_add(v.raw())));
                a[start + j + len] =
                    prime.mul(MontCoeff::from_raw(u.raw().wrapping_sub(v.raw())), w);
                j += 1;
            }
            start += 2 * len;
        }
        len /= 2;
    }

    let mut i = 0usize;
    while i + 16 <= D {
        unsafe {
            let values = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            _mm256_storeu_si256(
                ptr.add(i) as *mut __m256i,
                reduce_range_16x_i16_avx2(values, p),
            );
        }
        i += 16;
    }
    while i < D {
        a[i] = prime.reduce_range(a[i]);
        i += 1;
    }
}

/// AVX2 inverse negacyclic NTT for one i16 CRT limb.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn inverse_ntt_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p = _mm256_set1_epi16(prime.p);
    let pinv = _mm256_set1_epi16(prime.pinv);
    let ptr = a.as_mut_ptr() as *mut i16;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let mut start = 0usize;
        while start < D {
            let mut j = 0usize;
            while j + 16 <= len {
                unsafe {
                    let w = _mm256_loadu_si256(
                        tw.inv_twiddles.as_ptr().add(twiddle_base + j) as *const __m256i
                    );
                    let u = _mm256_loadu_si256(ptr.add(start + j) as *const __m256i);
                    let v_raw = _mm256_loadu_si256(ptr.add(start + j + len) as *const __m256i);
                    let v = mont_mul_16x_i16_avx2(v_raw, w, p, pinv);
                    _mm256_storeu_si256(
                        ptr.add(start + j) as *mut __m256i,
                        reduce_range_16x_i16_avx2(_mm256_add_epi16(u, v), p),
                    );
                    _mm256_storeu_si256(
                        ptr.add(start + j + len) as *mut __m256i,
                        reduce_range_16x_i16_avx2(_mm256_sub_epi16(u, v), p),
                    );
                }
                j += 16;
            }
            while j < len {
                let w = tw.inv_twiddles[twiddle_base + j];
                let u = a[start + j];
                let v = prime.mul(a[start + j + len], w);
                a[start + j] =
                    prime.reduce_range(MontCoeff::from_raw(u.raw().wrapping_add(v.raw())));
                a[start + j + len] =
                    prime.reduce_range(MontCoeff::from_raw(u.raw().wrapping_sub(v.raw())));
                j += 1;
            }
            start += 2 * len;
        }
        len *= 2;
    }

    let mut i = 0usize;
    while i + 16 <= D {
        unsafe {
            let values = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            let scale = _mm256_loadu_si256(tw.d_inv_psi_inv.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(
                ptr.add(i) as *mut __m256i,
                mont_mul_16x_i16_avx2(values, scale, p, pinv),
            );
        }
        i += 16;
    }
    while i < D {
        a[i] = prime.mul(a[i], tw.d_inv_psi_inv[i]);
        i += 1;
    }
}
