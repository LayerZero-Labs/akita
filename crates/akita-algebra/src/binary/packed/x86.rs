//! x86 PCLMUL and two-lane VPCLMUL packed-buffer kernels.

use std::arch::x86_64::{
    __m128i, __m256i, _mm256_and_si256, _mm256_castsi256_si128, _mm256_clmulepi64_epi128,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_permute4x64_epi64, _mm256_set_epi64x,
    _mm256_setzero_si256, _mm256_slli_epi64, _mm256_srli_epi64, _mm256_srli_si256,
    _mm256_xor_si256, _mm_storeu_si128,
};

use super::{fold_with, round_product_with, PackedBinary162};
use crate::binary::{product, BinaryField162 as F};

#[target_feature(enable = "pclmulqdq")]
unsafe fn product_words(a: F, b: F) -> [u64; 6] {
    let [a0, a1, a2] = a.to_words();
    let [b0, b1, b2] = b.to_words();
    // SAFETY: the enclosing function carries the required target feature.
    unsafe {
        product::karatsuba_product(
            product::x86_clmul(a0, b0),
            product::x86_clmul(a1, b1),
            product::x86_clmul(a2, b2),
            product::x86_clmul(a0 ^ a1, b0 ^ b1),
            product::x86_clmul(a0 ^ a2, b0 ^ b2),
            product::x86_clmul(a1 ^ a2, b1 ^ b2),
        )
    }
}

#[target_feature(enable = "pclmulqdq")]
pub(super) unsafe fn round_product(
    lhs: &PackedBinary162,
    rhs: &PackedBinary162,
    current_claim: F,
) -> [F; 3] {
    round_product_with(lhs, rhs, current_claim, |a, b| {
        // SAFETY: the enclosing function carries the same target feature.
        unsafe { product_words(a, b) }
    })
}

#[target_feature(enable = "pclmulqdq")]
pub(super) unsafe fn fold_in_place(values: &mut PackedBinary162, r: F) {
    fold_with(values, r, |a, b| {
        // SAFETY: the enclosing function carries the same target feature.
        unsafe { product::x86_multiply(a, b) }
    });
}

#[inline]
#[target_feature(enable = "avx2,pclmulqdq,vpclmulqdq")]
unsafe fn product_vectors(a: [__m256i; 3], b: [__m256i; 3]) -> [__m256i; 6] {
    let clmul = |x, y| _mm256_clmulepi64_epi128::<0>(x, y);
    [
        clmul(a[0], b[0]),
        clmul(a[1], b[1]),
        clmul(a[2], b[2]),
        clmul(_mm256_xor_si256(a[0], a[1]), _mm256_xor_si256(b[0], b[1])),
        clmul(_mm256_xor_si256(a[0], a[2]), _mm256_xor_si256(b[0], b[2])),
        clmul(_mm256_xor_si256(a[1], a[2]), _mm256_xor_si256(b[1], b[2])),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_four(values: &PackedBinary162, index: usize) -> [__m256i; 3] {
    std::array::from_fn(|word| {
        // SAFETY: callers establish that four elements beginning at `index`
        // exist, and unaligned loads accept the Vec allocation's alignment.
        unsafe { _mm256_loadu_si256(values.words[word].as_ptr().add(index).cast()) }
    })
}

#[inline]
#[target_feature(enable = "avx2")]
fn low_words(value: __m256i) -> __m256i {
    _mm256_and_si256(value, _mm256_set_epi64x(0, -1, 0, -1))
}

#[inline]
#[target_feature(enable = "avx2")]
fn pair_deltas(value: __m256i) -> __m256i {
    _mm256_xor_si256(value, _mm256_srli_si256::<8>(value))
}

/// Reduce two independent polynomial products in the low word of each lane.
#[inline]
#[target_feature(enable = "avx2")]
fn reduce_vectors(terms: [__m256i; 6]) -> [__m256i; 3] {
    let high = |value| _mm256_srli_si256::<8>(value);
    let d0 = terms[0];
    let d1 = terms[1];
    let d2 = terms[2];
    let c01 = _mm256_xor_si256(_mm256_xor_si256(terms[3], d0), d1);
    let c02 = _mm256_xor_si256(_mm256_xor_si256(terms[4], d0), d2);
    let c12 = _mm256_xor_si256(_mm256_xor_si256(terms[5], d1), d2);
    let p = [
        low_words(d0),
        _mm256_xor_si256(high(d0), low_words(c01)),
        _mm256_xor_si256(_mm256_xor_si256(low_words(d1), high(c01)), low_words(c02)),
        _mm256_xor_si256(_mm256_xor_si256(high(d1), high(c02)), low_words(c12)),
        _mm256_xor_si256(low_words(d2), high(c12)),
        high(d2),
    ];

    let h0 = _mm256_xor_si256(_mm256_srli_epi64::<34>(p[2]), _mm256_slli_epi64::<30>(p[3]));
    let h1 = _mm256_xor_si256(_mm256_srli_epi64::<34>(p[3]), _mm256_slli_epi64::<30>(p[4]));
    let h2 = _mm256_xor_si256(_mm256_srli_epi64::<34>(p[4]), _mm256_slli_epi64::<30>(p[5]));
    let j0 = _mm256_xor_si256(
        _mm256_xor_si256(h0, _mm256_srli_epi64::<17>(h1)),
        _mm256_slli_epi64::<47>(h2),
    );
    let j1 = _mm256_xor_si256(h1, _mm256_srli_epi64::<17>(h2));
    [
        _mm256_xor_si256(p[0], j0),
        _mm256_xor_si256(_mm256_xor_si256(p[1], j1), _mm256_slli_epi64::<17>(j0)),
        _mm256_and_si256(
            _mm256_xor_si256(
                _mm256_xor_si256(p[2], h2),
                _mm256_xor_si256(_mm256_srli_epi64::<47>(j0), _mm256_slli_epi64::<17>(j1)),
            ),
            _mm256_set_epi64x(0, (1_i64 << 34) - 1, 0, (1_i64 << 34) - 1),
        ),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_two(value: __m256i, destination: *mut u64) {
    let contiguous = _mm256_permute4x64_epi64::<0xd8>(value);
    // SAFETY: callers provide room for two words; the store is unaligned.
    unsafe { _mm_storeu_si128(destination.cast(), _mm256_castsi256_si128(contiguous)) };
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lane(value: __m256i, high: bool) -> u128 {
    // SAFETY: AVX2 is established by the caller and all bit patterns are valid.
    unsafe {
        let value = if high {
            _mm256_extracti128_si256::<1>(value)
        } else {
            _mm256_castsi256_si128(value)
        };
        std::mem::transmute::<__m128i, u128>(value)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn folded_product_words(terms: &[__m256i; 6]) -> [u64; 6] {
    // SAFETY: the enclosing function carries AVX2 for lane extraction.
    let products: [u128; 6] =
        unsafe { std::array::from_fn(|term| lane(terms[term], false) ^ lane(terms[term], true)) };
    product::karatsuba_product(
        products[0],
        products[1],
        products[2],
        products[3],
        products[4],
        products[5],
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn xor_vectors(sum: &mut [__m256i; 6], terms: [__m256i; 6]) {
    for (sum, term) in sum.iter_mut().zip(terms) {
        *sum = _mm256_xor_si256(*sum, term);
    }
}

#[inline]
fn xor_words(sum: &mut [u64; 6], terms: [u64; 6]) {
    for (sum, term) in sum.iter_mut().zip(terms) {
        *sum ^= term;
    }
}

#[target_feature(enable = "avx2,pclmulqdq,vpclmulqdq")]
pub(super) unsafe fn round_product_vec2(
    lhs: &PackedBinary162,
    rhs: &PackedBinary162,
    current_claim: F,
) -> [F; 3] {
    let zero = _mm256_setzero_si256();
    let mut sums = [[zero; 6]; 2];
    let mut index = 0;
    while index + 3 < lhs.len() {
        // SAFETY: the loop condition establishes four available elements.
        unsafe {
            let a = load_four(lhs, index);
            let b = load_four(rhs, index);
            xor_vectors(&mut sums[0], product_vectors(a, b));
            xor_vectors(
                &mut sums[1],
                product_vectors(a.map(|v| pair_deltas(v)), b.map(|v| pair_deltas(v))),
            );
        }
        index += 4;
    }

    // SAFETY: this function carries AVX2 for lane extraction.
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

#[target_feature(enable = "avx2,pclmulqdq,vpclmulqdq")]
pub(super) unsafe fn fold_in_place_vec2(values: &mut PackedBinary162, r: F) {
    let old_len = values.len();
    let new_len = old_len.div_ceil(2);
    let scalar_r = r;
    let r = r
        .to_words()
        .map(|word| _mm256_set_epi64x(0, word as i64, 0, word as i64));
    let mut input = 0;
    while input + 3 < old_len {
        // SAFETY: the loop condition establishes four available elements.
        unsafe {
            let loaded = load_four(values, input);
            let even = loaded.map(|v| low_words(v));
            let deltas = loaded.map(|v| pair_deltas(v));
            let product = reduce_vectors(product_vectors(deltas, r));
            let output = input / 2;
            for word in 0..3 {
                store_two(
                    _mm256_xor_si256(even[word], product[word]),
                    values.words[word].as_mut_ptr().add(output),
                );
            }
        }
        input += 4;
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
