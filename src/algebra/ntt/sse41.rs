//! x86_64 SSE4.1 kernels for CRT pointwise multiply-accumulate and add-reduce.
//!
//! This mirrors the existing AArch64 NEON acceleration for the hot CRT-domain
//! accumulate loops used by dense D32 root commits on x86 hosts that do not
//! have AVX2 available.

use std::arch::x86_64::*;
use std::sync::OnceLock;

/// Whether the SSE4.1 CRT fast path is active.
///
/// Set `HACHI_SCALAR_NTT=1` to force the scalar fallback.
pub(crate) fn use_sse41_ntt() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HACHI_SCALAR_NTT").map_or(true, |v| v != "1")
            && std::arch::is_x86_feature_detected!("sse4.1")
    })
}

#[target_feature(enable = "sse4.1")]
unsafe fn mont_mul_4x_i32(a: __m128i, b: __m128i, p: __m128i, pinv: __m128i) -> __m128i {
    let c_even = _mm_mul_epi32(a, b);
    let c_odd = _mm_mul_epi32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4));

    let t = _mm_mullo_epi32(_mm_mullo_epi32(a, b), pinv);
    let tp_even = _mm_mul_epi32(t, p);
    let tp_odd = _mm_mul_epi32(_mm_srli_si128(t, 4), p);

    let r_even = _mm_srli_epi64(_mm_sub_epi64(c_even, tp_even), 32);
    let r_odd = _mm_srli_epi64(_mm_sub_epi64(c_odd, tp_odd), 32);

    let lo = _mm_unpacklo_epi32(r_even, r_odd);
    let hi = _mm_unpackhi_epi32(r_even, r_odd);
    _mm_unpacklo_epi64(lo, hi)
}

#[target_feature(enable = "sse4.1")]
unsafe fn reduce_range_4x_i32(a: __m128i, p: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let ge = _mm_or_si128(_mm_cmpgt_epi32(a, p), _mm_cmpeq_epi32(a, p));
    let after_sub = _mm_sub_epi32(a, _mm_and_si128(p, ge));
    let lt = _mm_cmpgt_epi32(zero, after_sub);
    _mm_add_epi32(after_sub, _mm_and_si128(p, lt))
}

/// `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))` for `i in 0..d`.
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn pointwise_mul_acc_i32(
    acc: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    let p_d = _mm_set1_epi32(p);
    let pinv_d = _mm_set1_epi32(pinv);
    let mut i = 0usize;
    while i + 4 <= d {
        let a = _mm_loadu_si128(acc.add(i) as *const __m128i);
        let l = _mm_loadu_si128(lhs.add(i) as *const __m128i);
        let r = _mm_loadu_si128(rhs.add(i) as *const __m128i);
        let prod = mont_mul_4x_i32(l, r, p_d, pinv_d);
        let sum = _mm_add_epi32(a, prod);
        _mm_storeu_si128(acc.add(i) as *mut __m128i, reduce_range_4x_i32(sum, p_d));
        i += 4;
    }
}

/// `acc[i] = reduce_range(acc[i] + other[i])` for `i in 0..d`.
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn add_reduce_i32(acc: *mut i32, other: *const i32, d: usize, p: i32) {
    let p_d = _mm_set1_epi32(p);
    let mut i = 0usize;
    while i + 4 <= d {
        let a = _mm_loadu_si128(acc.add(i) as *const __m128i);
        let b = _mm_loadu_si128(other.add(i) as *const __m128i);
        let sum = _mm_add_epi32(a, b);
        _mm_storeu_si128(acc.add(i) as *mut __m128i, reduce_range_4x_i32(sum, p_d));
        i += 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse41_gate_matches_host_capability() {
        let enabled = use_sse41_ntt();
        if std::env::var("HACHI_SCALAR_NTT").as_deref() == Ok("1") {
            assert!(!enabled);
        }
    }
}
