//! x86_64 SIMD kernels for CRT+NTT pointwise operations.
//!
//! Provides runtime-selected AVX2 kernels and, when the crate is compiled with
//! the matching target features, explicit AVX-512 kernels. Set
//! `AKITA_SCALAR_NTT=1` to force the scalar fallback, or
//! `AKITA_X86_NTT=avx2|avx512|auto` to choose a backend for benchmarking.

use std::arch::x86_64::*;
use std::sync::OnceLock;

use super::prime::{MontCoeff, NttPrime};

#[derive(Clone, Copy)]
enum X86Mode {
    #[cfg(all(
        target_feature = "avx2",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512bw",
        target_feature = "avx512vl"
    ))]
    Avx512,
    Avx2,
}

/// Returns whether an x86 SIMD NTT kernel is enabled at runtime.
///
/// `AKITA_SCALAR_NTT=1` disables this path. `AKITA_X86_NTT` may be set to
/// `auto`, `avx2`, `avx512`, or a scalar/off value.
pub fn use_x86_ntt() -> bool {
    selected_mode().is_some()
}

fn selected_mode() -> Option<X86Mode> {
    static MODE: OnceLock<Option<X86Mode>> = OnceLock::new();
    *MODE.get_or_init(detect_mode)
}

fn detect_mode() -> Option<X86Mode> {
    if std::env::var("AKITA_SCALAR_NTT").map_or(false, |v| v == "1") {
        return None;
    }

    match std::env::var("AKITA_X86_NTT")
        .unwrap_or_else(|_| "auto".to_owned())
        .to_ascii_lowercase()
        .as_str()
    {
        "0" | "off" | "false" | "scalar" => None,
        "avx2" => avx2_available().then_some(X86Mode::Avx2),
        "avx512" | "avx512f" => avx512_available().then_some(avx512_mode()?),
        _ => {
            if avx2_available() {
                Some(X86Mode::Avx2)
            } else if avx512_available() {
                avx512_mode()
            } else {
                None
            }
        }
    }
}

#[inline]
fn avx2_available() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(all(
    target_feature = "avx2",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
))]
#[inline]
fn avx512_available() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
}

#[cfg(not(all(
    target_feature = "avx2",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
)))]
#[inline]
fn avx512_available() -> bool {
    false
}

#[cfg(all(
    target_feature = "avx2",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
))]
#[inline]
fn avx512_mode() -> Option<X86Mode> {
    Some(X86Mode::Avx512)
}

#[cfg(not(all(
    target_feature = "avx2",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
)))]
#[inline]
fn avx512_mode() -> Option<X86Mode> {
    None
}

/// SIMD pointwise multiply-accumulate for one i32 CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))`.
///
/// # Safety
///
/// `acc`, `lhs`, and `rhs` must be valid for `d` elements and must not alias in
/// a way that violates Rust's mutable-reference rules.
pub unsafe fn pointwise_mul_acc_i32(
    acc: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    debug_assert!((p as i64) < (1i64 << 30));
    match selected_mode() {
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512bw",
            target_feature = "avx512vl"
        ))]
        Some(X86Mode::Avx512) => avx512::pointwise_mul_acc_i32(acc, lhs, rhs, d, p, pinv),
        Some(X86Mode::Avx2) => avx2::pointwise_mul_acc_i32(acc, lhs, rhs, d, p, pinv),
        None => scalar_pointwise_mul_acc_i32(acc, lhs, rhs, d, p),
    }
}

/// SIMD pointwise multiply-accumulate for one i16 CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))`.
///
/// # Safety
///
/// `acc`, `lhs`, and `rhs` must be valid for `d` elements and must not alias in
/// a way that violates Rust's mutable-reference rules.
pub unsafe fn pointwise_mul_acc_i16(
    acc: *mut i16,
    lhs: *const i16,
    rhs: *const i16,
    d: usize,
    p: i16,
    pinv: i16,
) {
    debug_assert!((p as i64) < (1i64 << 14));
    match selected_mode() {
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512bw",
            target_feature = "avx512vl"
        ))]
        Some(X86Mode::Avx512) => avx512::pointwise_mul_acc_i16(acc, lhs, rhs, d, p, pinv),
        Some(X86Mode::Avx2) => avx2::pointwise_mul_acc_i16(acc, lhs, rhs, d, p, pinv),
        None => scalar_pointwise_mul_acc_i16(acc, lhs, rhs, d, p),
    }
}

/// SIMD add-and-reduce for one i32 CRT limb.
///
/// Computes `acc[i] = reduce_range(acc[i] + other[i])`.
///
/// # Safety
///
/// `acc` and `other` must be valid for `d` elements and must not alias in a way
/// that violates Rust's mutable-reference rules.
#[cfg(feature = "parallel")]
pub unsafe fn add_reduce_i32(acc: *mut i32, other: *const i32, d: usize, p: i32) {
    debug_assert!((p as i64) < (1i64 << 30));
    match selected_mode() {
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512bw",
            target_feature = "avx512vl"
        ))]
        Some(X86Mode::Avx512) => avx512::add_reduce_i32(acc, other, d, p),
        Some(X86Mode::Avx2) => avx2::add_reduce_i32(acc, other, d, p),
        None => scalar_add_reduce_i32(acc, other, d, p),
    }
}

/// SIMD add-and-reduce for one i16 CRT limb.
///
/// # Safety
///
/// `acc` and `other` must be valid for `d` elements and must not alias in a way
/// that violates Rust's mutable-reference rules.
#[cfg(feature = "parallel")]
pub unsafe fn add_reduce_i16(acc: *mut i16, other: *const i16, d: usize, p: i16) {
    debug_assert!((p as i64) < (1i64 << 14));
    match selected_mode() {
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512bw",
            target_feature = "avx512vl"
        ))]
        Some(X86Mode::Avx512) => avx512::add_reduce_i16(acc, other, d, p),
        Some(X86Mode::Avx2) => avx2::add_reduce_i16(acc, other, d, p),
        None => scalar_add_reduce_i16(acc, other, d, p),
    }
}

unsafe fn scalar_pointwise_mul_acc_i32(
    acc: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
) {
    let prime = NttPrime::compute(p);
    for i in 0..d {
        let prod = prime.mul(
            MontCoeff::from_raw(*lhs.add(i)),
            MontCoeff::from_raw(*rhs.add(i)),
        );
        *acc.add(i) = prime
            .add_reduce(MontCoeff::from_raw(*acc.add(i)), prod)
            .raw();
    }
}

unsafe fn scalar_pointwise_mul_acc_i16(
    acc: *mut i16,
    lhs: *const i16,
    rhs: *const i16,
    d: usize,
    p: i16,
) {
    let prime = NttPrime::compute(p);
    for i in 0..d {
        let prod = prime.mul(
            MontCoeff::from_raw(*lhs.add(i)),
            MontCoeff::from_raw(*rhs.add(i)),
        );
        *acc.add(i) = prime
            .add_reduce(MontCoeff::from_raw(*acc.add(i)), prod)
            .raw();
    }
}

#[cfg(feature = "parallel")]
unsafe fn scalar_add_reduce_i32(acc: *mut i32, other: *const i32, d: usize, p: i32) {
    let prime = NttPrime::compute(p);
    for i in 0..d {
        *acc.add(i) = prime
            .add_reduce(
                MontCoeff::from_raw(*acc.add(i)),
                MontCoeff::from_raw(*other.add(i)),
            )
            .raw();
    }
}

#[cfg(feature = "parallel")]
unsafe fn scalar_add_reduce_i16(acc: *mut i16, other: *const i16, d: usize, p: i16) {
    let prime = NttPrime::compute(p);
    for i in 0..d {
        *acc.add(i) = prime
            .add_reduce(
                MontCoeff::from_raw(*acc.add(i)),
                MontCoeff::from_raw(*other.add(i)),
            )
            .raw();
    }
}

mod avx2 {
    use super::*;

    #[target_feature(enable = "avx2,sse4.1")]
    pub(super) unsafe fn pointwise_mul_acc_i32(
        acc: *mut i32,
        lhs: *const i32,
        rhs: *const i32,
        d: usize,
        p: i32,
        pinv: i32,
    ) {
        let p4 = _mm_set1_epi32(p);
        let pinv4 = _mm_set1_epi32(pinv);
        let p8 = _mm256_set1_epi32(p);
        let mut i = 0;
        while i + 8 <= d {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let l = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let r = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            let prod = mont_mul_8x_i32(l, r, p4, pinv4);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_8x_i32(_mm256_add_epi32(a, prod), p8),
            );
            i += 8;
        }
        while i + 4 <= d {
            let a = _mm_loadu_si128(acc.add(i) as *const __m128i);
            let l = _mm_loadu_si128(lhs.add(i) as *const __m128i);
            let r = _mm_loadu_si128(rhs.add(i) as *const __m128i);
            let prod = mont_mul_4x_i32(l, r, p4, pinv4);
            _mm_storeu_si128(
                acc.add(i) as *mut __m128i,
                reduce_range_4x_i32(_mm_add_epi32(a, prod), _mm_set1_epi32(p)),
            );
            i += 4;
        }
        scalar_pointwise_mul_acc_i32(acc.add(i), lhs.add(i), rhs.add(i), d - i, p);
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn pointwise_mul_acc_i16(
        acc: *mut i16,
        lhs: *const i16,
        rhs: *const i16,
        d: usize,
        p: i16,
        pinv: i16,
    ) {
        let p32 = _mm256_set1_epi32(p as i32);
        let pinv32 = _mm256_set1_epi32(pinv as i32);
        let p16 = _mm256_set1_epi16(p);
        let mut i = 0;
        while i + 16 <= d {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let l = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let r = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            let prod = mont_mul_16x_i16(l, r, p32, pinv32);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_16x_i16(_mm256_add_epi16(a, prod), p16),
            );
            i += 16;
        }
        while i + 8 <= d {
            let a = _mm_loadu_si128(acc.add(i) as *const __m128i);
            let l = _mm_loadu_si128(lhs.add(i) as *const __m128i);
            let r = _mm_loadu_si128(rhs.add(i) as *const __m128i);
            let prod = mont_mul_8x_i16(l, r, p32, pinv32);
            _mm_storeu_si128(
                acc.add(i) as *mut __m128i,
                reduce_range_8x_i16(_mm_add_epi16(a, prod), _mm_set1_epi16(p)),
            );
            i += 8;
        }
        scalar_pointwise_mul_acc_i16(acc.add(i), lhs.add(i), rhs.add(i), d - i, p);
    }

    #[cfg(feature = "parallel")]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn add_reduce_i32(acc: *mut i32, other: *const i32, d: usize, p: i32) {
        let p8 = _mm256_set1_epi32(p);
        let mut i = 0;
        while i + 8 <= d {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let b = _mm256_loadu_si256(other.add(i) as *const __m256i);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_8x_i32(_mm256_add_epi32(a, b), p8),
            );
            i += 8;
        }
        scalar_add_reduce_i32(acc.add(i), other.add(i), d - i, p);
    }

    #[cfg(feature = "parallel")]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn add_reduce_i16(acc: *mut i16, other: *const i16, d: usize, p: i16) {
        let p16 = _mm256_set1_epi16(p);
        let mut i = 0;
        while i + 16 <= d {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let b = _mm256_loadu_si256(other.add(i) as *const __m256i);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_16x_i16(_mm256_add_epi16(a, b), p16),
            );
            i += 16;
        }
        scalar_add_reduce_i16(acc.add(i), other.add(i), d - i, p);
    }

    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn mont_mul_8x_i32(a: __m256i, b: __m256i, p: __m128i, pinv: __m128i) -> __m256i {
        let lo = mont_mul_4x_i32(
            _mm256_castsi256_si128(a),
            _mm256_castsi256_si128(b),
            p,
            pinv,
        );
        let hi = mont_mul_4x_i32(
            _mm256_extracti128_si256::<1>(a),
            _mm256_extracti128_si256::<1>(b),
            p,
            pinv,
        );
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
    }

    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn mont_mul_4x_i32(a: __m128i, b: __m128i, p: __m128i, pinv: __m128i) -> __m128i {
        let c02 = _mm_mul_epi32(a, b);
        let c13 = _mm_mul_epi32(_mm_srli_si128::<4>(a), _mm_srli_si128::<4>(b));
        let c_lo = pack_i64_low32(c02, c13);
        let t = _mm_mullo_epi32(c_lo, pinv);
        let tp02 = _mm_mul_epi32(t, p);
        let tp13 = _mm_mul_epi32(_mm_srli_si128::<4>(t), p);
        pack_i64_high32(_mm_sub_epi64(c02, tp02), _mm_sub_epi64(c13, tp13))
    }

    #[inline(always)]
    unsafe fn pack_i64_low32(even: __m128i, odd: __m128i) -> __m128i {
        let even_low = _mm_shuffle_epi32::<0x88>(even);
        let odd_low = _mm_shuffle_epi32::<0x88>(odd);
        _mm_unpacklo_epi32(even_low, odd_low)
    }

    #[inline(always)]
    unsafe fn pack_i64_high32(even: __m128i, odd: __m128i) -> __m128i {
        let even_high = _mm_shuffle_epi32::<0xdd>(even);
        let odd_high = _mm_shuffle_epi32::<0xdd>(odd);
        _mm_unpacklo_epi32(even_high, odd_high)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mont_mul_16x_i16(a: __m256i, b: __m256i, p: __m256i, pinv: __m256i) -> __m256i {
        let lo = mont_mul_8x_i16(
            _mm256_castsi256_si128(a),
            _mm256_castsi256_si128(b),
            p,
            pinv,
        );
        let hi = mont_mul_8x_i16(
            _mm256_extracti128_si256::<1>(a),
            _mm256_extracti128_si256::<1>(b),
            p,
            pinv,
        );
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mont_mul_8x_i16(a: __m128i, b: __m128i, p: __m256i, pinv: __m256i) -> __m128i {
        let a32 = _mm256_cvtepi16_epi32(a);
        let b32 = _mm256_cvtepi16_epi32(b);
        let c = _mm256_mullo_epi32(a32, b32);
        let t = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(_mm256_mullo_epi32(c, pinv)));
        let r = _mm256_srai_epi32::<16>(_mm256_sub_epi32(c, _mm256_mullo_epi32(t, p)));
        pack_i32_to_i16(r)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn pack_i32_to_i16(a: __m256i) -> __m128i {
        _mm_packs_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn reduce_range_8x_i32(a: __m256i, p: __m256i) -> __m256i {
        let ge = _mm256_cmpgt_epi32(a, _mm256_sub_epi32(p, _mm256_set1_epi32(1)));
        let after_sub = _mm256_sub_epi32(a, _mm256_and_si256(ge, p));
        let lt = _mm256_cmpgt_epi32(_mm256_setzero_si256(), after_sub);
        _mm256_add_epi32(after_sub, _mm256_and_si256(lt, p))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn reduce_range_4x_i32(a: __m128i, p: __m128i) -> __m128i {
        let ge = _mm_cmpgt_epi32(a, _mm_sub_epi32(p, _mm_set1_epi32(1)));
        let after_sub = _mm_sub_epi32(a, _mm_and_si128(ge, p));
        let lt = _mm_cmpgt_epi32(_mm_setzero_si128(), after_sub);
        _mm_add_epi32(after_sub, _mm_and_si128(lt, p))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn reduce_range_16x_i16(a: __m256i, p: __m256i) -> __m256i {
        let ge = _mm256_cmpgt_epi16(a, _mm256_sub_epi16(p, _mm256_set1_epi16(1)));
        let after_sub = _mm256_sub_epi16(a, _mm256_and_si256(ge, p));
        let lt = _mm256_cmpgt_epi16(_mm256_setzero_si256(), after_sub);
        _mm256_add_epi16(after_sub, _mm256_and_si256(lt, p))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn reduce_range_8x_i16(a: __m128i, p: __m128i) -> __m128i {
        let ge = _mm_cmpgt_epi16(a, _mm_sub_epi16(p, _mm_set1_epi16(1)));
        let after_sub = _mm_sub_epi16(a, _mm_and_si128(ge, p));
        let lt = _mm_cmpgt_epi16(_mm_setzero_si128(), after_sub);
        _mm_add_epi16(after_sub, _mm_and_si128(lt, p))
    }
}

#[cfg(all(
    target_feature = "avx2",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
))]
mod avx512 {
    use super::*;

    pub(super) unsafe fn pointwise_mul_acc_i32(
        acc: *mut i32,
        lhs: *const i32,
        rhs: *const i32,
        d: usize,
        p: i32,
        pinv: i32,
    ) {
        let p8 = _mm256_set1_epi32(p);
        let pinv8 = _mm256_set1_epi32(pinv);
        let mut i = 0;
        while i + 8 <= d {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let l = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let r = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            let prod = mont_mul_8x_i32(l, r, p8, pinv8);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_8x_i32(_mm256_add_epi32(a, prod), p8),
            );
            i += 8;
        }
        avx2::pointwise_mul_acc_i32(acc.add(i), lhs.add(i), rhs.add(i), d - i, p, pinv);
    }

    pub(super) unsafe fn pointwise_mul_acc_i16(
        acc: *mut i16,
        lhs: *const i16,
        rhs: *const i16,
        d: usize,
        p: i16,
        pinv: i16,
    ) {
        let p32 = _mm512_set1_epi32(p as i32);
        let pinv32 = _mm512_set1_epi32(pinv as i32);
        let p16 = _mm256_set1_epi16(p);
        let mut i = 0;
        while i + 16 <= d {
            let a = _mm256_loadu_si256(acc.add(i) as *const __m256i);
            let l = _mm256_loadu_si256(lhs.add(i) as *const __m256i);
            let r = _mm256_loadu_si256(rhs.add(i) as *const __m256i);
            let prod = mont_mul_16x_i16(l, r, p32, pinv32);
            _mm256_storeu_si256(
                acc.add(i) as *mut __m256i,
                reduce_range_16x_i16(_mm256_add_epi16(a, prod), p16),
            );
            i += 16;
        }
        avx2::pointwise_mul_acc_i16(acc.add(i), lhs.add(i), rhs.add(i), d - i, p, pinv);
    }

    #[cfg(feature = "parallel")]
    pub(super) unsafe fn add_reduce_i32(acc: *mut i32, other: *const i32, d: usize, p: i32) {
        let p16 = _mm512_set1_epi32(p);
        let mut i = 0;
        while i + 16 <= d {
            let a = _mm512_loadu_si512(acc.add(i) as *const __m512i);
            let b = _mm512_loadu_si512(other.add(i) as *const __m512i);
            _mm512_storeu_si512(
                acc.add(i) as *mut __m512i,
                reduce_range_16x_i32(_mm512_add_epi32(a, b), p16),
            );
            i += 16;
        }
        avx2::add_reduce_i32(acc.add(i), other.add(i), d - i, p);
    }

    #[cfg(feature = "parallel")]
    pub(super) unsafe fn add_reduce_i16(acc: *mut i16, other: *const i16, d: usize, p: i16) {
        let p32 = _mm512_set1_epi16(p);
        let mut i = 0;
        while i + 32 <= d {
            let a = _mm512_loadu_si512(acc.add(i) as *const __m512i);
            let b = _mm512_loadu_si512(other.add(i) as *const __m512i);
            _mm512_storeu_si512(
                acc.add(i) as *mut __m512i,
                reduce_range_32x_i16(_mm512_add_epi16(a, b), p32),
            );
            i += 32;
        }
        avx2::add_reduce_i16(acc.add(i), other.add(i), d - i, p);
    }

    unsafe fn mont_mul_8x_i32(a: __m256i, b: __m256i, p: __m256i, pinv: __m256i) -> __m256i {
        let a64 = _mm512_cvtepi32_epi64(a);
        let b64 = _mm512_cvtepi32_epi64(b);
        let c = _mm512_mullo_epi64(a64, b64);
        let c_lo = _mm512_cvtepi64_epi32(c);
        let t = _mm256_mullo_epi32(c_lo, pinv);
        let t64 = _mm512_cvtepi32_epi64(t);
        let p64 = _mm512_cvtepi32_epi64(p);
        let diff = _mm512_sub_epi64(c, _mm512_mullo_epi64(t64, p64));
        _mm512_cvtepi64_epi32(_mm512_srai_epi64::<32>(diff))
    }

    unsafe fn mont_mul_16x_i16(a: __m256i, b: __m256i, p: __m512i, pinv: __m512i) -> __m256i {
        let a32 = _mm512_cvtepi16_epi32(a);
        let b32 = _mm512_cvtepi16_epi32(b);
        let c = _mm512_mullo_epi32(a32, b32);
        let t = _mm512_srai_epi32::<16>(_mm512_slli_epi32::<16>(_mm512_mullo_epi32(c, pinv)));
        let r = _mm512_srai_epi32::<16>(_mm512_sub_epi32(c, _mm512_mullo_epi32(t, p)));
        _mm512_cvtepi32_epi16(r)
    }

    unsafe fn reduce_range_16x_i32(a: __m512i, p: __m512i) -> __m512i {
        let ge = _mm512_cmpgt_epi32_mask(a, _mm512_sub_epi32(p, _mm512_set1_epi32(1)));
        let after_sub = _mm512_mask_sub_epi32(a, ge, a, p);
        let lt = _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), after_sub);
        _mm512_mask_add_epi32(after_sub, lt, after_sub, p)
    }

    unsafe fn reduce_range_8x_i32(a: __m256i, p: __m256i) -> __m256i {
        let ge = _mm256_cmpgt_epi32_mask(a, _mm256_sub_epi32(p, _mm256_set1_epi32(1)));
        let after_sub = _mm256_mask_sub_epi32(a, ge, a, p);
        let lt = _mm256_cmpgt_epi32_mask(_mm256_setzero_si256(), after_sub);
        _mm256_mask_add_epi32(after_sub, lt, after_sub, p)
    }

    unsafe fn reduce_range_32x_i16(a: __m512i, p: __m512i) -> __m512i {
        let ge = _mm512_cmpgt_epi16_mask(a, _mm512_sub_epi16(p, _mm512_set1_epi16(1)));
        let after_sub = _mm512_mask_sub_epi16(a, ge, a, p);
        let lt = _mm512_cmpgt_epi16_mask(_mm512_setzero_si512(), after_sub);
        _mm512_mask_add_epi16(after_sub, lt, after_sub, p)
    }

    unsafe fn reduce_range_16x_i16(a: __m256i, p: __m256i) -> __m256i {
        let ge = _mm256_cmpgt_epi16_mask(a, _mm256_sub_epi16(p, _mm256_set1_epi16(1)));
        let after_sub = _mm256_mask_sub_epi16(a, ge, a, p);
        let lt = _mm256_cmpgt_epi16_mask(_mm256_setzero_si256(), after_sub);
        _mm256_mask_add_epi16(after_sub, lt, after_sub, p)
    }
}
