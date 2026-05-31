//! x86 runtime dispatch helpers for CRT NTT SIMD kernels.
//!
//! `AKITA_SCALAR_NTT=1` forces the scalar fallback for all CRT NTT SIMD.
//! `AKITA_AVX_NTT=0` disables only x86 CRT NTT SIMD. AVX-512 kernels are the
//! default when the host advertises the required features; `AKITA_AVX512_NTT=0`
//! forces the AVX2 path for A/B testing or hosts with AVX-512 frequency
//! penalties.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::sync::OnceLock;

pub mod batch;
mod d32;

use super::butterfly::NttTwiddles;
use super::prime::{MontCoeff, NttPrime};

/// Runtime-selected x86 CRT NTT SIMD mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvxNttMode {
    /// AVX2 kernels using 256-bit integer vectors.
    Avx2,
    /// AVX-512 kernels using 512-bit integer vectors.
    Avx512,
}

#[derive(Debug, Clone, Copy)]
struct AvxCpuFeatures {
    avx2: bool,
    avx512f: bool,
    avx512dq: bool,
    avx512bw: bool,
}

impl AvxCpuFeatures {
    #[inline]
    const fn has_avx512_ntt(self) -> bool {
        self.avx512f && self.avx512dq && self.avx512bw
    }
}

/// Return the enabled x86 CRT NTT SIMD mode, if any.
///
/// The result is cached because this function sits on hot dispatch boundaries.
pub fn avx_ntt_mode() -> Option<AvxNttMode> {
    static MODE: OnceLock<Option<AvxNttMode>> = OnceLock::new();
    *MODE.get_or_init(|| {
        select_avx_ntt_mode(
            std::env::var("AKITA_SCALAR_NTT").ok().as_deref(),
            std::env::var("AKITA_AVX_NTT").ok().as_deref(),
            std::env::var("AKITA_AVX512_NTT").ok().as_deref(),
            detect_cpu_features(),
        )
    })
}

/// Whether the host may use AVX2 transform kernels.
///
/// AVX-512 mode currently selects AVX-512 pointwise kernels, but the transform
/// kernels are AVX2-shaped because the supported CRT degrees spend most stages
/// at four or fewer useful `i32` lanes.
pub fn use_avx2_transform_ntt() -> bool {
    avx_ntt_mode().is_some() && std::is_x86_feature_detected!("avx2")
}

#[inline]
fn select_avx_ntt_mode(
    scalar_ntt: Option<&str>,
    avx_ntt: Option<&str>,
    avx512_ntt: Option<&str>,
    cpu: AvxCpuFeatures,
) -> Option<AvxNttMode> {
    if scalar_ntt == Some("1") || avx_ntt == Some("0") {
        return None;
    }
    // AVX-512 is the default when available. `AKITA_AVX512_NTT=0` opts back out
    // to AVX2 for A/B comparison or hosts that downclock under AVX-512.
    if avx512_ntt != Some("0") && cpu.has_avx512_ntt() {
        return Some(AvxNttMode::Avx512);
    }
    if cpu.avx2 {
        return Some(AvxNttMode::Avx2);
    }
    None
}

#[inline]
fn detect_cpu_features() -> AvxCpuFeatures {
    AvxCpuFeatures {
        avx2: std::is_x86_feature_detected!("avx2"),
        avx512f: std::is_x86_feature_detected!("avx512f"),
        avx512dq: std::is_x86_feature_detected!("avx512dq"),
        avx512bw: std::is_x86_feature_detected!("avx512bw"),
    }
}

#[target_feature(enable = "avx2")]
unsafe fn mont_mul_8x_i32_avx2(a: __m256i, b: __m256i, p: __m256i, pinv: __m256i) -> __m256i {
    let even_products = _mm256_mul_epi32(a, b);
    let a_odd = _mm256_srli_epi64::<32>(a);
    let b_odd = _mm256_srli_epi64::<32>(b);
    let odd_products = _mm256_mul_epi32(a_odd, b_odd);

    let even = mont_reduce_i32_products_avx2(even_products, p, pinv);
    let odd = mont_reduce_i32_products_avx2(odd_products, p, pinv);
    _mm256_or_si256(even, _mm256_slli_epi64::<32>(odd))
}

#[target_feature(enable = "avx2")]
unsafe fn mont_reduce_i32_products_avx2(c: __m256i, p: __m256i, pinv: __m256i) -> __m256i {
    let t = _mm256_mullo_epi32(c, pinv);
    let tp = _mm256_mul_epi32(t, p);
    let diff = _mm256_sub_epi64(c, tp);
    // Keep the high 32-bit two's-complement pattern from each 64-bit lane.
    // AVX2 has no arithmetic i64 shift, but the low half after this logical
    // shift is exactly the scalar `(diff >> 32) as i32` bit pattern.
    _mm256_srli_epi64::<32>(diff)
}

#[target_feature(enable = "avx2")]
unsafe fn reduce_range_8x_i32_avx2(a: __m256i, p: __m256i) -> __m256i {
    let one = _mm256_set1_epi32(1);
    let p_minus_one = _mm256_sub_epi32(p, one);
    let ge_mask = _mm256_cmpgt_epi32(a, p_minus_one);
    let after_sub = _mm256_sub_epi32(a, _mm256_and_si256(p, ge_mask));

    let zero = _mm256_setzero_si256();
    let lt_mask = _mm256_cmpgt_epi32(zero, after_sub);
    _mm256_add_epi32(after_sub, _mm256_and_si256(p, lt_mask))
}

#[target_feature(enable = "avx2")]
unsafe fn mont_mul_4x_i32_avx2(a: __m128i, b: __m128i, p: __m128i, pinv: __m128i) -> __m128i {
    let even_products = _mm_mul_epi32(a, b);
    let a_odd = _mm_srli_epi64::<32>(a);
    let b_odd = _mm_srli_epi64::<32>(b);
    let odd_products = _mm_mul_epi32(a_odd, b_odd);

    let even = mont_reduce_i32_products_128_avx2(even_products, p, pinv);
    let odd = mont_reduce_i32_products_128_avx2(odd_products, p, pinv);
    _mm_or_si128(even, _mm_slli_epi64::<32>(odd))
}

#[target_feature(enable = "avx2")]
unsafe fn mont_reduce_i32_products_128_avx2(c: __m128i, p: __m128i, pinv: __m128i) -> __m128i {
    let t = _mm_mullo_epi32(c, pinv);
    let tp = _mm_mul_epi32(t, p);
    let diff = _mm_sub_epi64(c, tp);
    _mm_srli_epi64::<32>(diff)
}

#[target_feature(enable = "avx2")]
unsafe fn reduce_range_4x_i32_avx2(a: __m128i, p: __m128i) -> __m128i {
    let one = _mm_set1_epi32(1);
    let p_minus_one = _mm_sub_epi32(p, one);
    let ge_mask = _mm_cmpgt_epi32(a, p_minus_one);
    let after_sub = _mm_sub_epi32(a, _mm_and_si128(p, ge_mask));

    let zero = _mm_setzero_si128();
    let lt_mask = _mm_cmpgt_epi32(zero, after_sub);
    _mm_add_epi32(after_sub, _mm_and_si128(p, lt_mask))
}

#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
unsafe fn mont_mul_16x_i32_avx512(a: __m512i, b: __m512i, p: __m512i, pinv: __m512i) -> __m512i {
    let even_products = _mm512_mul_epi32(a, b);
    let a_odd = _mm512_srli_epi64::<32>(a);
    let b_odd = _mm512_srli_epi64::<32>(b);
    let odd_products = _mm512_mul_epi32(a_odd, b_odd);

    let even = mont_reduce_i32_products_avx512(even_products, p, pinv);
    let odd = mont_reduce_i32_products_avx512(odd_products, p, pinv);
    _mm512_or_si512(even, _mm512_slli_epi64::<32>(odd))
}

#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
unsafe fn mont_reduce_i32_products_avx512(c: __m512i, p: __m512i, pinv: __m512i) -> __m512i {
    let t = _mm512_mullo_epi32(c, pinv);
    let tp = _mm512_mul_epi32(t, p);
    let diff = _mm512_sub_epi64(c, tp);
    _mm512_srli_epi64::<32>(diff)
}

#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
unsafe fn reduce_range_16x_i32_avx512(a: __m512i, p: __m512i) -> __m512i {
    let one = _mm512_set1_epi32(1);
    let p_minus_one = _mm512_sub_epi32(p, one);
    let ge_mask = _mm512_cmpgt_epi32_mask(a, p_minus_one);
    let after_sub = _mm512_mask_sub_epi32(a, ge_mask, a, p);

    let zero = _mm512_setzero_si512();
    let lt_mask = _mm512_cmplt_epi32_mask(after_sub, zero);
    _mm512_mask_add_epi32(after_sub, lt_mask, after_sub, p)
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
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
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
            } else {
                for j in 0..len {
                    let w = tw.fwd_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = a[start + j + len];
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                }
            }
            start += 2 * len;
        }
        len /= 2;
    }

    // SAFETY: guaranteed by this function's safety contract.
    unsafe { reduce_range_in_place_i32(a, prime, p_d) };
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

    let p_d = _mm_set1_epi32(prime.p);
    let pinv_d = _mm_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
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
            } else {
                for j in 0..len {
                    let w = tw.fwd_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = a[start + j + len];
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                }
            }
            start += 2 * len;
        }
        len /= 2;
    }

    // SAFETY: guaranteed by this function's safety contract.
    unsafe { reduce_range_in_place_i32(a, prime, p_d) };
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

#[target_feature(enable = "avx2")]
unsafe fn mont_mul_16x_i16_avx2(a: __m256i, b: __m256i, p: __m256i, pinv: __m256i) -> __m256i {
    let a_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a));
    let b_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b));
    let a_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(a));
    let b_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(b));

    let prod_lo = mont_mul_8x_i16_as_i32_avx2(a_lo, b_lo, p, pinv);
    let prod_hi = mont_mul_8x_i16_as_i32_avx2(a_hi, b_hi, p, pinv);
    let packed = _mm256_packs_epi32(prod_lo, prod_hi);
    _mm256_permute4x64_epi64::<0xd8>(packed)
}

#[target_feature(enable = "avx2")]
unsafe fn mont_mul_8x_i16_as_i32_avx2(
    a: __m256i,
    b: __m256i,
    p: __m256i,
    pinv: __m256i,
) -> __m256i {
    let c = _mm256_mullo_epi32(a, b);
    let t_wrapped = _mm256_mullo_epi32(c, pinv);
    let t = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(t_wrapped));
    let tp = _mm256_mullo_epi32(t, p);
    _mm256_srai_epi32::<16>(_mm256_sub_epi32(c, tp))
}

#[target_feature(enable = "avx2")]
unsafe fn reduce_range_16x_i16_avx2(a: __m256i, p: __m256i) -> __m256i {
    let one = _mm256_set1_epi16(1);
    let p_minus_one = _mm256_sub_epi16(p, one);
    let ge_mask = _mm256_cmpgt_epi16(a, p_minus_one);
    let after_sub = _mm256_sub_epi16(a, _mm256_and_si256(p, ge_mask));

    let zero = _mm256_setzero_si256();
    let lt_mask = _mm256_cmpgt_epi16(zero, after_sub);
    _mm256_add_epi16(after_sub, _mm256_and_si256(p, lt_mask))
}

/// AVX2 pointwise multiply-accumulate for one `i32` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))`.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `acc`, `lhs`, and `rhs` must be
/// valid for `d` `i32` elements. `acc` must be writable and must not alias in
/// a way that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pointwise_mul_acc_i32(
    acc: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    let p_v = _mm256_set1_epi32(p);
    let pinv_v = _mm256_set1_epi32(pinv);
    let mut i = 0;
    while i + 8 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let l = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let r = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            let prod = mont_mul_8x_i32_avx2(l, r, p_v, pinv_v);
            let sum = _mm256_add_epi32(a, prod);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_8x_i32_avx2(sum, p_v),
            );
        }
        i += 8;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let prod = prime.mul(
                    MontCoeff::from_raw(*lhs.add(i)),
                    MontCoeff::from_raw(*rhs.add(i)),
                );
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(prod.raw()));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX-512 pointwise multiply-accumulate for one `i32` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))`.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available. `acc`, `lhs`, and
/// `rhs` must be valid for `d` `i32` elements. `acc` must be writable and must
/// not alias in a way that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub(crate) unsafe fn pointwise_mul_acc_i32_avx512(
    acc: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    let p_v = _mm512_set1_epi32(p);
    let pinv_v = _mm512_set1_epi32(pinv);
    let mut i = 0;
    while i + 16 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm512_loadu_si512(acc.add(i) as *const __m512i);
            let l = _mm512_loadu_si512(lhs.add(i) as *const __m512i);
            let r = _mm512_loadu_si512(rhs.add(i) as *const __m512i);
            let prod = mont_mul_16x_i32_avx512(l, r, p_v, pinv_v);
            let sum = _mm512_add_epi32(a, prod);
            _mm512_storeu_si512(
                acc.add(i) as *mut __m512i,
                reduce_range_16x_i32_avx512(sum, p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let prod = prime.mul(
                    MontCoeff::from_raw(*lhs.add(i)),
                    MontCoeff::from_raw(*rhs.add(i)),
                );
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(prod.raw()));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX2 add-and-reduce for one `i32` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + other[i])`.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `acc` and `other` must be valid
/// for `d` `i32` elements. `acc` must be writable and must not alias in a way
/// that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx2")]
pub unsafe fn add_reduce_i32(acc: *mut i32, other: *const i32, d: usize, p: i32) {
    let p_v = _mm256_set1_epi32(p);
    let mut i = 0;
    while i + 8 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let b = _mm256_loadu_si256(other.add(i) as *const __m256i);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_8x_i32_avx2(_mm256_add_epi32(a, b), p_v),
            );
        }
        i += 8;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(*other.add(i)));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX-512 add-and-reduce for one `i32` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + other[i])`.
///
/// # Safety
///
/// The caller must ensure AVX-512F/DQ/BW are available. `acc` and `other` must
/// be valid for `d` `i32` elements. `acc` must be writable and must not alias in
/// a way that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx512f,avx512dq,avx512bw")]
pub unsafe fn add_reduce_i32_avx512(acc: *mut i32, other: *const i32, d: usize, p: i32) {
    let p_v = _mm512_set1_epi32(p);
    let mut i = 0;
    while i + 16 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm512_loadu_si512(acc.add(i) as *const __m512i);
            let b = _mm512_loadu_si512(other.add(i) as *const __m512i);
            _mm512_storeu_si512(
                acc.add(i) as *mut __m512i,
                reduce_range_16x_i32_avx512(_mm512_add_epi32(a, b), p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(*other.add(i)));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX2 pointwise multiply-accumulate for one `i16` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))`.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `acc`, `lhs`, and `rhs` must be
/// valid for `d` `i16` elements. `acc` must be writable and must not alias in
/// a way that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pointwise_mul_acc_i16(
    acc: *mut i16,
    lhs: *const i16,
    rhs: *const i16,
    d: usize,
    p: i16,
    pinv: i16,
) {
    let p_v = _mm256_set1_epi32(p as i32);
    let pinv_v = _mm256_set1_epi32(pinv as i32);
    let p_i16 = _mm256_set1_epi16(p);
    let mut i = 0;
    while i + 16 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let l = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let r = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            let prod = mont_mul_16x_i16_avx2(l, r, p_v, pinv_v);
            let sum = _mm256_add_epi16(a, prod);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_16x_i16_avx2(sum, p_i16),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let prod = prime.mul(
                    MontCoeff::from_raw(*lhs.add(i)),
                    MontCoeff::from_raw(*rhs.add(i)),
                );
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(prod.raw()));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

/// AVX2 add-and-reduce for one `i16` CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + other[i])`.
///
/// # Safety
///
/// The caller must ensure AVX2 is available. `acc` and `other` must be valid
/// for `d` `i16` elements. `acc` must be writable and must not alias in a way
/// that violates Rust's mutable-reference rules.
#[target_feature(enable = "avx2")]
pub unsafe fn add_reduce_i16(acc: *mut i16, other: *const i16, d: usize, p: i16) {
    let p_v = _mm256_set1_epi16(p);
    let mut i = 0;
    while i + 16 <= d {
        // SAFETY: guaranteed by this function's safety contract and loop bound.
        unsafe {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let b = _mm256_loadu_si256(other.add(i) as *const __m256i);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_16x_i16_avx2(_mm256_add_epi16(a, b), p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let prime = NttPrime::compute(p);
        while i < d {
            // SAFETY: guaranteed by this function's safety contract and loop bound.
            unsafe {
                let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(*other.add(i)));
                *acc.add(i) = prime.reduce_range(sum).raw();
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AVX2_ONLY: AvxCpuFeatures = AvxCpuFeatures {
        avx2: true,
        avx512f: false,
        avx512dq: false,
        avx512bw: false,
    };

    const AVX512_CAPABLE: AvxCpuFeatures = AvxCpuFeatures {
        avx2: true,
        avx512f: true,
        avx512dq: true,
        avx512bw: true,
    };

    #[test]
    fn avx_mode_defaults_to_avx2_when_supported() {
        assert_eq!(
            select_avx_ntt_mode(None, None, None, AVX2_ONLY),
            Some(AvxNttMode::Avx2)
        );
    }

    #[test]
    fn avx512_is_default_when_available() {
        assert_eq!(
            select_avx_ntt_mode(None, None, None, AVX512_CAPABLE),
            Some(AvxNttMode::Avx512)
        );
        assert_eq!(
            select_avx_ntt_mode(None, None, Some("1"), AVX512_CAPABLE),
            Some(AvxNttMode::Avx512)
        );
    }

    #[test]
    fn avx512_can_be_opted_out_to_avx2() {
        assert_eq!(
            select_avx_ntt_mode(None, None, Some("0"), AVX512_CAPABLE),
            Some(AvxNttMode::Avx2)
        );
    }

    #[test]
    fn scalar_kill_switch_precedes_avx_flags() {
        assert_eq!(
            select_avx_ntt_mode(Some("1"), None, Some("1"), AVX512_CAPABLE),
            None
        );
    }

    #[test]
    fn avx_kill_switch_disables_x86_ntt_simd() {
        assert_eq!(
            select_avx_ntt_mode(None, Some("0"), Some("1"), AVX512_CAPABLE),
            None
        );
    }

    #[test]
    fn avx512_opt_in_falls_back_to_avx2_without_full_features() {
        let missing_bw = AvxCpuFeatures {
            avx512bw: false,
            ..AVX512_CAPABLE
        };
        assert_eq!(
            select_avx_ntt_mode(None, None, Some("1"), missing_bw),
            Some(AvxNttMode::Avx2)
        );
    }

    fn random_mont_array_i32<const D: usize>(
        prime: NttPrime<i32>,
        seed: u64,
    ) -> [MontCoeff<i32>; D] {
        let mut state = seed;
        std::array::from_fn(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = ((state >> 33) as i64 % prime.p as i64) as i32;
            prime.from_canonical(val)
        })
    }

    fn random_mont_array_i16<const D: usize>(
        prime: NttPrime<i16>,
        seed: u64,
    ) -> [MontCoeff<i16>; D] {
        let mut state = seed;
        std::array::from_fn(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = ((state >> 33) as i64 % prime.p as i64) as i16;
            prime.from_canonical(val)
        })
    }

    fn edge_mont_array_i32<const D: usize>(prime: NttPrime<i32>) -> [MontCoeff<i32>; D] {
        let values = [
            0,
            1,
            -1,
            prime.p - 1,
            1 - prime.p,
            prime.p / 2,
            -(prime.p / 2),
            0x4000_1234_i32,
            -0x3fff_4321_i32,
        ];
        std::array::from_fn(|i| MontCoeff::from_raw(values[i % values.len()]))
    }

    fn edge_mont_array_i16<const D: usize>(prime: NttPrime<i16>) -> [MontCoeff<i16>; D] {
        let values = [
            0,
            1,
            -1,
            prime.p - 1,
            1 - prime.p,
            prime.p / 2,
            -(prime.p / 2),
            0x3a5a_i16,
            -0x3211_i16,
        ];
        std::array::from_fn(|i| MontCoeff::from_raw(values[i % values.len()]))
    }

    fn scalar_pointwise_i32<const D: usize>(
        acc: &mut [MontCoeff<i32>; D],
        lhs: &[MontCoeff<i32>; D],
        rhs: &[MontCoeff<i32>; D],
        prime: NttPrime<i32>,
    ) {
        for i in 0..D {
            let prod = prime.mul(lhs[i], rhs[i]);
            let sum = MontCoeff::from_raw(acc[i].raw().wrapping_add(prod.raw()));
            acc[i] = prime.reduce_range(sum);
        }
    }

    fn scalar_pointwise_i16<const D: usize>(
        acc: &mut [MontCoeff<i16>; D],
        lhs: &[MontCoeff<i16>; D],
        rhs: &[MontCoeff<i16>; D],
        prime: NttPrime<i16>,
    ) {
        for i in 0..D {
            let prod = prime.mul(lhs[i], rhs[i]);
            let sum = MontCoeff::from_raw(acc[i].raw().wrapping_add(prod.raw()));
            acc[i] = prime.reduce_range(sum);
        }
    }

    fn scalar_add_reduce_i32<const D: usize>(
        acc: &mut [MontCoeff<i32>; D],
        other: &[MontCoeff<i32>; D],
        prime: NttPrime<i32>,
    ) {
        for i in 0..D {
            let sum = MontCoeff::from_raw(acc[i].raw().wrapping_add(other[i].raw()));
            acc[i] = prime.reduce_range(sum);
        }
    }

    fn scalar_add_reduce_i16<const D: usize>(
        acc: &mut [MontCoeff<i16>; D],
        other: &[MontCoeff<i16>; D],
        prime: NttPrime<i16>,
    ) {
        for i in 0..D {
            let sum = MontCoeff::from_raw(acc[i].raw().wrapping_add(other[i].raw()));
            acc[i] = prime.reduce_range(sum);
        }
    }

    fn scalar_forward_ntt_i32<const D: usize>(
        a: &mut [MontCoeff<i32>; D],
        prime: NttPrime<i32>,
        tw: &NttTwiddles<i32, D>,
    ) {
        for (ai, psi) in a.iter_mut().zip(tw.psi_pows.iter()) {
            *ai = prime.mul(*ai, *psi);
        }
        scalar_forward_ntt_cyclic_i32(a, prime, tw);
    }

    fn scalar_inverse_ntt_i32<const D: usize>(
        a: &mut [MontCoeff<i32>; D],
        prime: NttPrime<i32>,
        tw: &NttTwiddles<i32, D>,
    ) {
        let mut len = 1usize;
        while len < D {
            let twiddle_base = len - 1;
            let mut start = 0usize;
            while start < D {
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
            len *= 2;
        }
        for (ai, fused) in a.iter_mut().zip(tw.d_inv_psi_inv.iter()) {
            *ai = prime.mul(*ai, *fused);
        }
    }

    fn scalar_forward_ntt_cyclic_i32<const D: usize>(
        a: &mut [MontCoeff<i32>; D],
        prime: NttPrime<i32>,
        tw: &NttTwiddles<i32, D>,
    ) {
        let mut len = D / 2;
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
        prime.reduce_range_in_place(a);
    }

    fn scalar_inverse_ntt_cyclic_i32<const D: usize>(
        a: &mut [MontCoeff<i32>; D],
        prime: NttPrime<i32>,
        tw: &NttTwiddles<i32, D>,
    ) {
        let mut len = 1usize;
        while len < D {
            let twiddle_base = len - 1;
            let mut start = 0usize;
            while start < D {
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
            len *= 2;
        }
        for c in a.iter_mut() {
            *c = prime.mul(*c, tw.d_inv);
        }
    }

    #[test]
    fn avx2_ntt_i32_transforms_match_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let prime = NttPrime::compute(1073707009_i32);
        let tw = NttTwiddles::<i32, 64>::compute(prime);
        let input = random_mont_array_i32::<64>(prime, 0x5150);

        let mut avx_fwd = input;
        let mut scalar_fwd = input;
        // SAFETY: guarded by runtime AVX2 detection above.
        unsafe { forward_ntt_i32(&mut avx_fwd, prime, &tw) };
        scalar_forward_ntt_i32(&mut scalar_fwd, prime, &tw);
        assert_eq!(avx_fwd, scalar_fwd);

        let mut avx_inv = avx_fwd;
        let mut scalar_inv = scalar_fwd;
        // SAFETY: guarded by runtime AVX2 detection above.
        unsafe { inverse_ntt_i32(&mut avx_inv, prime, &tw) };
        scalar_inverse_ntt_i32(&mut scalar_inv, prime, &tw);
        assert_eq!(avx_inv, scalar_inv);

        let mut avx_cyclic = input;
        let mut scalar_cyclic = input;
        // SAFETY: guarded by runtime AVX2 detection above.
        unsafe { forward_ntt_cyclic_i32(&mut avx_cyclic, prime, &tw) };
        scalar_forward_ntt_cyclic_i32(&mut scalar_cyclic, prime, &tw);
        assert_eq!(avx_cyclic, scalar_cyclic);

        // SAFETY: guarded by runtime AVX2 detection above.
        unsafe { inverse_ntt_cyclic_i32(&mut avx_cyclic, prime, &tw) };
        scalar_inverse_ntt_cyclic_i32(&mut scalar_cyclic, prime, &tw);
        assert_eq!(avx_cyclic, scalar_cyclic);
    }

    #[test]
    fn avx2_pointwise_mul_acc_i32_matches_scalar_with_tail() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let prime = NttPrime::compute(1073707009_i32);
        const D: usize = 19;
        let acc_init = random_mont_array_i32::<D>(prime, 0x1111);
        let lhs = edge_mont_array_i32::<D>(prime);
        let rhs = random_mont_array_i32::<D>(prime, 0x3333);

        let mut avx_acc = acc_init;
        // SAFETY: guarded by the runtime AVX2 detection above.
        unsafe {
            pointwise_mul_acc_i32(
                avx_acc.as_mut_ptr() as *mut i32,
                lhs.as_ptr() as *const i32,
                rhs.as_ptr() as *const i32,
                D,
                prime.p,
                prime.pinv,
            );
        }

        let mut scalar_acc = acc_init;
        scalar_pointwise_i32(&mut scalar_acc, &lhs, &rhs, prime);
        assert_eq!(avx_acc, scalar_acc);
    }

    #[test]
    fn avx2_add_reduce_i32_matches_scalar_with_tail() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let prime = NttPrime::compute(1073707009_i32);
        const D: usize = 19;
        let acc_init = random_mont_array_i32::<D>(prime, 0x4444);
        let other = edge_mont_array_i32::<D>(prime);

        let mut avx_acc = acc_init;
        // SAFETY: guarded by the runtime AVX2 detection above.
        unsafe {
            add_reduce_i32(
                avx_acc.as_mut_ptr() as *mut i32,
                other.as_ptr() as *const i32,
                D,
                prime.p,
            );
        }

        let mut scalar_acc = acc_init;
        scalar_add_reduce_i32(&mut scalar_acc, &other, prime);
        assert_eq!(avx_acc, scalar_acc);
    }

    #[test]
    fn avx512_pointwise_mul_acc_i32_matches_scalar_with_tail() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512bw"))
        {
            return;
        }
        let prime = NttPrime::compute(1073707009_i32);
        const D: usize = 29;
        let acc_init = random_mont_array_i32::<D>(prime, 0x5151);
        let lhs = edge_mont_array_i32::<D>(prime);
        let rhs = random_mont_array_i32::<D>(prime, 0x7171);

        let mut avx_acc = acc_init;
        // SAFETY: guarded by runtime AVX-512 feature detection above.
        unsafe {
            pointwise_mul_acc_i32_avx512(
                avx_acc.as_mut_ptr() as *mut i32,
                lhs.as_ptr() as *const i32,
                rhs.as_ptr() as *const i32,
                D,
                prime.p,
                prime.pinv,
            );
        }

        let mut scalar_acc = acc_init;
        scalar_pointwise_i32(&mut scalar_acc, &lhs, &rhs, prime);
        assert_eq!(avx_acc, scalar_acc);
    }

    #[test]
    fn avx512_add_reduce_i32_matches_scalar_with_tail() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512bw"))
        {
            return;
        }
        let prime = NttPrime::compute(1073707009_i32);
        const D: usize = 29;
        let acc_init = random_mont_array_i32::<D>(prime, 0x8181);
        let other = edge_mont_array_i32::<D>(prime);

        let mut avx_acc = acc_init;
        // SAFETY: guarded by runtime AVX-512 feature detection above.
        unsafe {
            add_reduce_i32_avx512(
                avx_acc.as_mut_ptr() as *mut i32,
                other.as_ptr() as *const i32,
                D,
                prime.p,
            );
        }

        let mut scalar_acc = acc_init;
        scalar_add_reduce_i32(&mut scalar_acc, &other, prime);
        assert_eq!(avx_acc, scalar_acc);
    }

    #[test]
    fn avx2_pointwise_mul_acc_i16_matches_scalar_with_tail() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let prime = NttPrime::compute(15361_i16);
        const D: usize = 23;
        let acc_init = random_mont_array_i16::<D>(prime, 0xaaaa);
        let lhs = edge_mont_array_i16::<D>(prime);
        let rhs = random_mont_array_i16::<D>(prime, 0xcccc);

        let mut avx_acc = acc_init;
        // SAFETY: guarded by the runtime AVX2 detection above.
        unsafe {
            pointwise_mul_acc_i16(
                avx_acc.as_mut_ptr() as *mut i16,
                lhs.as_ptr() as *const i16,
                rhs.as_ptr() as *const i16,
                D,
                prime.p,
                prime.pinv,
            );
        }

        let mut scalar_acc = acc_init;
        scalar_pointwise_i16(&mut scalar_acc, &lhs, &rhs, prime);
        assert_eq!(avx_acc, scalar_acc);
    }

    #[test]
    fn avx2_add_reduce_i16_matches_scalar_with_tail() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let prime = NttPrime::compute(15361_i16);
        const D: usize = 23;
        let acc_init = random_mont_array_i16::<D>(prime, 0xdddd);
        let other = edge_mont_array_i16::<D>(prime);

        let mut avx_acc = acc_init;
        // SAFETY: guarded by the runtime AVX2 detection above.
        unsafe {
            add_reduce_i16(
                avx_acc.as_mut_ptr() as *mut i16,
                other.as_ptr() as *const i16,
                D,
                prime.p,
            );
        }

        let mut scalar_acc = acc_init;
        scalar_add_reduce_i16(&mut scalar_acc, &other, prime);
        assert_eq!(avx_acc, scalar_acc);
    }
}
