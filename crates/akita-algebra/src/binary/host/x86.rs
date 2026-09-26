use std::arch::x86_64::{
    __m128i, __m512i, _mm512_clmulepi64_epi128, _mm512_loadu_si512,
    _mm512_mask_compressstoreu_epi64, _mm512_maskz_mov_epi64, _mm512_set1_epi64, _mm512_set_epi64,
    _mm512_shuffle_epi32, _mm512_slli_epi64, _mm512_srli_epi64, _mm512_storeu_si512,
    _mm512_xor_si512, _mm_clmulepi64_si128, _mm_shuffle_epi32, _mm_xor_si128,
};

use super::{reduce128, reduce64, BinaryField128, BinaryField192};

#[inline]
#[target_feature(enable = "pclmulqdq")]
fn clmul<const SELECT: i32>(a: __m128i, b: __m128i) -> u128 {
    // SAFETY: every bit pattern is valid in both representations.
    unsafe { std::mem::transmute::<__m128i, u128>(_mm_clmulepi64_si128::<SELECT>(a, b)) }
}

#[inline]
#[target_feature(enable = "sse2")]
fn swap64_128(value: __m128i) -> __m128i {
    _mm_shuffle_epi32::<0x4e>(value)
}

#[target_feature(enable = "pclmulqdq")]
pub(super) unsafe fn multiply128(a: BinaryField128, b: BinaryField128) -> BinaryField128 {
    // SAFETY: the transparent array representation has no invalid bit patterns.
    let (a, b) = unsafe {
        (
            std::mem::transmute::<BinaryField128, __m128i>(a),
            std::mem::transmute::<BinaryField128, __m128i>(b),
        )
    };
    let d0 = clmul::<0>(a, b);
    let d1 = clmul::<0x11>(a, b);
    let cross = clmul::<0>(
        _mm_xor_si128(a, swap64_128(a)),
        _mm_xor_si128(b, swap64_128(b)),
    ) ^ d0
        ^ d1;
    reduce128([
        d0 as u64,
        (d0 >> 64) as u64 ^ cross as u64,
        d1 as u64 ^ (cross >> 64) as u64,
        (d1 >> 64) as u64,
    ])
}

#[target_feature(enable = "pclmulqdq")]
pub(super) unsafe fn multiply192(a: BinaryField192, b: BinaryField192) -> BinaryField192 {
    let [a0, a1, a2] = a.0;
    let [b0, b1, b2] = b.0;
    // SAFETY: every array bit pattern is valid in an integer vector.
    let (a01, a02, a12, b01, b02, b12) = unsafe {
        (
            std::mem::transmute::<[u64; 2], __m128i>([a0, a1]),
            std::mem::transmute::<[u64; 2], __m128i>([a0, a2]),
            std::mem::transmute::<[u64; 2], __m128i>([a1, a2]),
            std::mem::transmute::<[u64; 2], __m128i>([b0, b1]),
            std::mem::transmute::<[u64; 2], __m128i>([b0, b2]),
            std::mem::transmute::<[u64; 2], __m128i>([b1, b2]),
        )
    };
    let d0 = clmul::<0>(a01, b01);
    let d1 = clmul::<0x11>(a01, b01);
    let d2 = clmul::<0x11>(a02, b02);
    let mixed = |a, b| {
        clmul::<0>(
            _mm_xor_si128(a, swap64_128(a)),
            _mm_xor_si128(b, swap64_128(b)),
        )
    };
    let c01 = mixed(a01, b01) ^ d0 ^ d1;
    let c02 = mixed(a02, b02) ^ d0 ^ d2;
    let c12 = mixed(a12, b12) ^ d1 ^ d2;
    BinaryField192([
        reduce64(d0 ^ c12),
        reduce64(c01 ^ c12 ^ d2),
        reduce64(d1 ^ c02 ^ d2),
    ])
}

#[target_feature(enable = "pclmulqdq")]
pub(super) unsafe fn equality128(point: &[BinaryField128], output: &mut [BinaryField128]) {
    if output.is_empty() {
        return;
    }
    output[0] = BinaryField128::ONE;
    for (axis, &r) in point.iter().enumerate() {
        let width = 1 << axis;
        let (low, high) = output.split_at_mut(width);
        for j in 0..width {
            // SAFETY: the enclosing function carries the same target feature.
            let hi = unsafe { multiply128(low[j], r) };
            high[j] = hi;
            low[j] += hi;
        }
    }
}

#[target_feature(enable = "pclmulqdq")]
pub(super) unsafe fn equality192(point: &[BinaryField192], output: &mut [BinaryField192]) {
    if output.is_empty() {
        return;
    }
    output[0] = BinaryField192::ONE;
    for (axis, &r) in point.iter().enumerate() {
        let width = 1 << axis;
        let (low, high) = output.split_at_mut(width);
        for j in 0..width {
            // SAFETY: the enclosing function carries the same target feature.
            let hi = unsafe { multiply192(low[j], r) };
            high[j] = hi;
            low[j] += hi;
        }
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
fn swap64(value: __m512i) -> __m512i {
    _mm512_shuffle_epi32::<0x4e>(value)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn xor3(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_xor_si512(_mm512_xor_si512(a, b), c)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn reduce128_vec(low: __m512i, high: __m512i) -> __m512i {
    let mut first = high;
    first = _mm512_xor_si512(first, _mm512_slli_epi64::<1>(high));
    first = _mm512_xor_si512(first, _mm512_slli_epi64::<2>(high));
    first = _mm512_xor_si512(first, _mm512_slli_epi64::<7>(high));

    let overflow = xor3(
        _mm512_srli_epi64::<63>(high),
        _mm512_srli_epi64::<62>(high),
        _mm512_srli_epi64::<57>(high),
    );
    let carry = _mm512_maskz_mov_epi64(0xaa, swap64(overflow));
    first = _mm512_xor_si512(first, carry);

    let overflow = _mm512_maskz_mov_epi64(0x55, swap64(overflow));
    let mut second = overflow;
    second = _mm512_xor_si512(second, _mm512_slli_epi64::<1>(overflow));
    second = _mm512_xor_si512(second, _mm512_slli_epi64::<2>(overflow));
    second = _mm512_xor_si512(second, _mm512_slli_epi64::<7>(overflow));
    xor3(low, first, second)
}

#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
fn multiply128_vec(a: __m512i, b: __m512i) -> __m512i {
    let d0 = _mm512_clmulepi64_epi128::<0>(a, b);
    let d1 = _mm512_clmulepi64_epi128::<0x11>(a, b);
    let cross = xor3(
        _mm512_clmulepi64_epi128::<0>(
            _mm512_xor_si512(a, swap64(a)),
            _mm512_xor_si512(b, swap64(b)),
        ),
        d0,
        d1,
    );
    let cross = swap64(cross);
    let low = _mm512_xor_si512(d0, _mm512_maskz_mov_epi64(0xaa, cross));
    let high = _mm512_xor_si512(d1, _mm512_maskz_mov_epi64(0x55, cross));
    reduce128_vec(low, high)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn pack192(values: *const BinaryField192, coefficient: usize) -> __m512i {
    // SAFETY: callers provide four initialized, consecutive field elements.
    unsafe {
        _mm512_set_epi64(
            0,
            (*values.add(3)).0[coefficient] as i64,
            0,
            (*values.add(2)).0[coefficient] as i64,
            0,
            (*values.add(1)).0[coefficient] as i64,
            0,
            (*values).0[coefficient] as i64,
        )
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
fn reduce64_vec(product: __m512i) -> __m512i {
    let low = _mm512_maskz_mov_epi64(0x55, product);
    let high = _mm512_maskz_mov_epi64(0x55, swap64(product));
    let mut first = high;
    first = _mm512_xor_si512(first, _mm512_slli_epi64::<1>(high));
    first = _mm512_xor_si512(first, _mm512_slli_epi64::<3>(high));
    first = _mm512_xor_si512(first, _mm512_slli_epi64::<4>(high));
    let overflow = xor3(
        _mm512_srli_epi64::<63>(high),
        _mm512_srli_epi64::<61>(high),
        _mm512_srli_epi64::<60>(high),
    );
    let mut second = overflow;
    second = _mm512_xor_si512(second, _mm512_slli_epi64::<1>(overflow));
    second = _mm512_xor_si512(second, _mm512_slli_epi64::<3>(overflow));
    second = _mm512_xor_si512(second, _mm512_slli_epi64::<4>(overflow));
    xor3(low, first, second)
}

#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn multiply192_vec(values: *const BinaryField192, b: BinaryField192) -> [BinaryField192; 4] {
    let a0 = pack192(values, 0);
    let a1 = pack192(values, 1);
    let a2 = pack192(values, 2);
    let b0 = _mm512_set1_epi64(b.0[0] as i64);
    let b1 = _mm512_set1_epi64(b.0[1] as i64);
    let b2 = _mm512_set1_epi64(b.0[2] as i64);
    let d0 = _mm512_clmulepi64_epi128::<0>(a0, b0);
    let d1 = _mm512_clmulepi64_epi128::<0>(a1, b1);
    let d2 = _mm512_clmulepi64_epi128::<0>(a2, b2);
    let c01 = xor3(
        _mm512_clmulepi64_epi128::<0>(_mm512_xor_si512(a0, a1), _mm512_xor_si512(b0, b1)),
        d0,
        d1,
    );
    let c02 = xor3(
        _mm512_clmulepi64_epi128::<0>(_mm512_xor_si512(a0, a2), _mm512_xor_si512(b0, b2)),
        d0,
        d2,
    );
    let c12 = xor3(
        _mm512_clmulepi64_epi128::<0>(_mm512_xor_si512(a1, a2), _mm512_xor_si512(b1, b2)),
        d1,
        d2,
    );
    let c0 = reduce64_vec(_mm512_xor_si512(d0, c12));
    let c1 = reduce64_vec(xor3(c01, c12, d2));
    let c2 = reduce64_vec(xor3(d1, c02, d2));
    let (mut w0, mut w1, mut w2) = ([0u64; 4], [0u64; 4], [0u64; 4]);
    // SAFETY: all arrays hold four writable u64 lanes selected from the vectors.
    unsafe {
        _mm512_mask_compressstoreu_epi64(w0.as_mut_ptr().cast(), 0x55, c0);
        _mm512_mask_compressstoreu_epi64(w1.as_mut_ptr().cast(), 0x55, c1);
        _mm512_mask_compressstoreu_epi64(w2.as_mut_ptr().cast(), 0x55, c2);
    }
    std::array::from_fn(|i| BinaryField192([w0[i], w1[i], w2[i]]))
}

#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn equality128_vec4(point: &[BinaryField128], output: &mut [BinaryField128]) {
    if output.is_empty() {
        return;
    }
    output[0] = BinaryField128::ONE;
    for (axis, &r) in point.iter().enumerate() {
        let width = 1 << axis;
        let (low, high) = output.split_at_mut(width);
        let mut j = 0;
        if width < 4 {
            while j < width {
                // SAFETY: AVX-512 VPCLMUL implies scalar PCLMUL support.
                let hi = unsafe { multiply128(low[j], r) };
                high[j] = hi;
                low[j] += hi;
                j += 1;
            }
            continue;
        }
        let repeated = _mm512_set_epi64(
            r.0[1] as i64,
            r.0[0] as i64,
            r.0[1] as i64,
            r.0[0] as i64,
            r.0[1] as i64,
            r.0[0] as i64,
            r.0[1] as i64,
            r.0[0] as i64,
        );
        while j < width {
            // SAFETY: each loop iteration accesses four initialized fields in
            // disjoint, in-bounds halves of the pre-sized table.
            unsafe {
                let old = _mm512_loadu_si512(low.as_ptr().add(j).cast());
                let hi = multiply128_vec(old, repeated);
                _mm512_storeu_si512(high.as_mut_ptr().add(j).cast(), hi);
                _mm512_storeu_si512(low.as_mut_ptr().add(j).cast(), _mm512_xor_si512(old, hi));
            }
            j += 4;
        }
    }
}

#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn equality192_vec4(point: &[BinaryField192], output: &mut [BinaryField192]) {
    if output.is_empty() {
        return;
    }
    output[0] = BinaryField192::ONE;
    for (axis, &r) in point.iter().enumerate() {
        let width = 1 << axis;
        let (low, high) = output.split_at_mut(width);
        let mut j = 0;
        if width < 4 {
            while j < width {
                // SAFETY: AVX-512 VPCLMUL implies scalar PCLMUL support.
                let hi = unsafe { multiply192(low[j], r) };
                high[j] = hi;
                low[j] += hi;
                j += 1;
            }
            continue;
        }
        while j < width {
            // SAFETY: each loop iteration accesses four initialized fields in
            // disjoint, in-bounds halves of the pre-sized table.
            let products = unsafe { multiply192_vec(low.as_ptr().add(j), r) };
            high[j..j + 4].copy_from_slice(&products);
            for (value, product) in low[j..j + 4].iter_mut().zip(products) {
                *value += product;
            }
            j += 4;
        }
    }
}
