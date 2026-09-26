//! Four-lane AVX-512/VPCLMUL kernels over the same adjacent-pair packed layout.

use super::{x86::product_words, PackedBinary162};
use crate::binary::{product, BinaryField162 as F};
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx512f,avx512bw,pclmulqdq,vpclmulqdq")]
unsafe fn product_vectors(a: [__m512i; 3], b: [__m512i; 3]) -> [__m512i; 6] {
    let clmul = |x, y| _mm512_clmulepi64_epi128::<0>(x, y);
    [
        clmul(a[0], b[0]),
        clmul(a[1], b[1]),
        clmul(a[2], b[2]),
        clmul(_mm512_xor_si512(a[0], a[1]), _mm512_xor_si512(b[0], b[1])),
        clmul(_mm512_xor_si512(a[0], a[2]), _mm512_xor_si512(b[0], b[2])),
        clmul(_mm512_xor_si512(a[1], a[2]), _mm512_xor_si512(b[1], b[2])),
    ]
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn load_eight(values: &PackedBinary162, index: usize) -> [__m512i; 3] {
    std::array::from_fn(|word| {
        // SAFETY: callers establish that eight elements beginning at `index`
        // exist, and unaligned loads accept the Vec allocation's alignment.
        unsafe { _mm512_loadu_si512(values.words[word].as_ptr().add(index).cast()) }
    })
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn low_words(value: __m512i) -> __m512i {
    _mm512_and_si512(value, _mm512_set_epi64(0, -1, 0, -1, 0, -1, 0, -1))
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn pair_deltas(value: __m512i) -> __m512i {
    _mm512_xor_si512(value, _mm512_bsrli_epi128::<8>(value))
}

/// Reduce four independent polynomial products in the low word of each lane.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn reduce_vectors(terms: [__m512i; 6]) -> [__m512i; 3] {
    let high = |value| _mm512_bsrli_epi128::<8>(value);
    let d0 = terms[0];
    let d1 = terms[1];
    let d2 = terms[2];
    let c01 = _mm512_xor_si512(_mm512_xor_si512(terms[3], d0), d1);
    let c02 = _mm512_xor_si512(_mm512_xor_si512(terms[4], d0), d2);
    let c12 = _mm512_xor_si512(_mm512_xor_si512(terms[5], d1), d2);
    let p = [
        low_words(d0),
        _mm512_xor_si512(high(d0), low_words(c01)),
        _mm512_xor_si512(_mm512_xor_si512(low_words(d1), high(c01)), low_words(c02)),
        _mm512_xor_si512(_mm512_xor_si512(high(d1), high(c02)), low_words(c12)),
        _mm512_xor_si512(low_words(d2), high(c12)),
        high(d2),
    ];

    let h0 = _mm512_xor_si512(_mm512_srli_epi64::<34>(p[2]), _mm512_slli_epi64::<30>(p[3]));
    let h1 = _mm512_xor_si512(_mm512_srli_epi64::<34>(p[3]), _mm512_slli_epi64::<30>(p[4]));
    let h2 = _mm512_xor_si512(_mm512_srli_epi64::<34>(p[4]), _mm512_slli_epi64::<30>(p[5]));
    let j0 = _mm512_xor_si512(
        _mm512_xor_si512(h0, _mm512_srli_epi64::<17>(h1)),
        _mm512_slli_epi64::<47>(h2),
    );
    let j1 = _mm512_xor_si512(h1, _mm512_srli_epi64::<17>(h2));
    [
        _mm512_xor_si512(p[0], j0),
        _mm512_xor_si512(_mm512_xor_si512(p[1], j1), _mm512_slli_epi64::<17>(j0)),
        _mm512_and_si512(
            _mm512_xor_si512(
                _mm512_xor_si512(p[2], h2),
                _mm512_xor_si512(_mm512_srli_epi64::<47>(j0), _mm512_slli_epi64::<17>(j1)),
            ),
            _mm512_set_epi64(
                0,
                (1_i64 << 34) - 1,
                0,
                (1_i64 << 34) - 1,
                0,
                (1_i64 << 34) - 1,
                0,
                (1_i64 << 34) - 1,
            ),
        ),
    ]
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn store_four(value: __m512i, destination: *mut u64) {
    let contiguous = _mm512_permutexvar_epi64(_mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0), value);
    // SAFETY: callers provide four writable words; unaligned stores are valid.
    unsafe { _mm256_storeu_si256(destination.cast(), _mm512_castsi512_si256(contiguous)) };
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn folded_product_words(terms: &[__m512i; 6]) -> [u64; 6] {
    // SAFETY: each vector is exactly four independent 128-bit products.
    let lanes: [[u128; 4]; 6] = unsafe { std::mem::transmute(*terms) };
    let p = lanes.map(|v| v[0] ^ v[1] ^ v[2] ^ v[3]);
    product::karatsuba_product(p[0], p[1], p[2], p[3], p[4], p[5])
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn xor_vectors(sum: &mut [__m512i; 6], terms: [__m512i; 6]) {
    for (sum, term) in sum.iter_mut().zip(terms) {
        *sum = _mm512_xor_si512(*sum, term);
    }
}

#[inline]
fn xor_words(sum: &mut [u64; 6], terms: [u64; 6]) {
    for (sum, term) in sum.iter_mut().zip(terms) {
        *sum ^= term;
    }
}

#[target_feature(enable = "avx512f,avx512bw,pclmulqdq,vpclmulqdq")]
pub(super) unsafe fn round_product_vec4(
    lhs: &PackedBinary162,
    rhs: &PackedBinary162,
    current_claim: F,
) -> [F; 3] {
    let zero = _mm512_setzero_si512();
    let mut sums = [[zero; 6]; 2];
    let mut index = 0;
    while index + 7 < lhs.len() {
        // SAFETY: the loop condition establishes eight available elements.
        unsafe {
            let a = load_eight(lhs, index);
            let b = load_eight(rhs, index);
            xor_vectors(&mut sums[0], product_vectors(a, b));
            xor_vectors(
                &mut sums[1],
                product_vectors(a.map(|v| pair_deltas(v)), b.map(|v| pair_deltas(v))),
            );
        }
        index += 8;
    }

    // SAFETY: this function carries AVX-512 for lane extraction.
    let mut words = unsafe { sums.map(|sum| folded_product_words(&sum)) };
    while index < lhs.len() {
        let a0 = lhs.element(index);
        let b0 = rhs.element(index);
        // SAFETY: this function carries PCLMUL.
        xor_words(&mut words[0], unsafe { product_words(a0, b0) });
        let (a1, b1) = if index + 1 < lhs.len() {
            (lhs.element(index + 1), rhs.element(index + 1))
        } else {
            (F::ZERO, F::ZERO)
        };
        // SAFETY: this function carries PCLMUL.
        xor_words(&mut words[1], unsafe { product_words(a0 + a1, b0 + b1) });
        index += 2;
    }
    let [constant, quadratic] = words.map(product::reduce);
    [constant, current_claim + quadratic, quadratic]
}

#[target_feature(enable = "avx512f,avx512bw,pclmulqdq,vpclmulqdq")]
pub(super) unsafe fn fold_in_place_vec4(values: &mut PackedBinary162, r: F) {
    let old_len = values.len();
    let new_len = old_len.div_ceil(2);
    let scalar_r = r;
    let r = r.to_words().map(|word| {
        _mm512_set_epi64(
            0,
            word as i64,
            0,
            word as i64,
            0,
            word as i64,
            0,
            word as i64,
        )
    });
    let mut input = 0;
    while input + 7 < old_len {
        // SAFETY: the loop condition establishes eight available elements.
        unsafe {
            let loaded = load_eight(values, input);
            let even = loaded.map(|v| low_words(v));
            let deltas = loaded.map(|v| pair_deltas(v));
            let product = reduce_vectors(product_vectors(deltas, r));
            let output = input / 2;
            for word in 0..3 {
                store_four(
                    _mm512_xor_si512(even[word], product[word]),
                    values.words[word].as_mut_ptr().add(output),
                );
            }
        }
        input += 8;
    }
    while input < old_len {
        let output = input / 2;
        let even = values.element(input);
        let odd = if input + 1 < old_len {
            values.element(input + 1)
        } else {
            F::ZERO
        };
        // SAFETY: this function carries PCLMUL.
        values.set_element(
            output,
            even + unsafe { product::x86_multiply(even + odd, scalar_r) },
        );
        input += 2;
    }
    values.truncate(new_len);
}
