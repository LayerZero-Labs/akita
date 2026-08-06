//! AVX-512IFMA kernels for canonical 50-bit residues.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{Ifma52Prime, Ifma52Twiddles, MASK, RADIX};

#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn reduce_once(value: __m512i, modulus: __m512i) -> __m512i {
    let mask = _mm512_cmp_epu64_mask(value, modulus, _MM_CMPINT_NLT);
    _mm512_mask_sub_epi64(value, mask, value, modulus)
}

#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn normalize(value: __m512i, prime: __m512i, two_prime: __m512i) -> __m512i {
    // SAFETY: inherited target features.
    let value = unsafe { reduce_once(value, two_prime) };
    unsafe { reduce_once(value, prime) }
}

#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn mul_constant(
    value: __m512i,
    multiplier: __m512i,
    precondition: __m512i,
    modulus: u64,
) -> __m512i {
    let zero = _mm512_setzero_si512();
    let quotient = _mm512_madd52hi_epu64(zero, precondition, value);
    let product = _mm512_madd52lo_epu64(zero, multiplier, value);
    let result = _mm512_madd52lo_epu64(
        product,
        quotient,
        _mm512_set1_epi64((RADIX - modulus) as i64),
    );
    _mm512_and_si512(result, _mm512_set1_epi64(MASK as i64))
}

#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn mul_variable(lhs: __m512i, rhs: __m512i, prime: Ifma52Prime) -> __m512i {
    let zero = _mm512_setzero_si512();
    let product_high = _mm512_madd52hi_epu64(zero, lhs, rhs);
    let product_low = _mm512_madd52lo_epu64(zero, lhs, rhs);
    let shifted = _mm512_or_si512(
        _mm512_srli_epi64::<48>(product_low),
        _mm512_slli_epi64::<4>(product_high),
    );
    let quotient = _mm512_madd52hi_epu64(zero, shifted, _mm512_set1_epi64(prime.barrett as i64));
    let result = _mm512_madd52lo_epu64(
        product_low,
        quotient,
        _mm512_set1_epi64((RADIX - prime.modulus) as i64),
    );
    let result = _mm512_and_si512(result, _mm512_set1_epi64(MASK as i64));
    // SAFETY: inherited target features.
    unsafe { reduce_once(result, _mm512_set1_epi64(prime.modulus as i64)) }
}

#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn multiply_table<const D: usize>(
    values: &mut [u64; D],
    table: &[u64; D],
    preconditions: &[u64; D],
    prime: Ifma52Prime,
) {
    let modulus = _mm512_set1_epi64(prime.modulus as i64);
    let twice_modulus = _mm512_set1_epi64((2 * prime.modulus) as i64);
    let mut index = 0;
    while index < D {
        // SAFETY: supported degrees are multiples of eight, and each load/store
        // covers `index..index + 8` within all three arrays.
        unsafe {
            let value = _mm512_loadu_si512(values.as_ptr().add(index).cast());
            let multiplier = _mm512_loadu_si512(table.as_ptr().add(index).cast());
            let precondition = _mm512_loadu_si512(preconditions.as_ptr().add(index).cast());
            let product = mul_constant(value, multiplier, precondition, prime.modulus);
            _mm512_storeu_si512(
                values.as_mut_ptr().add(index).cast(),
                normalize(product, modulus, twice_modulus),
            );
        }
        index += 8;
    }
}

#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn paired_lanes(value: __m512i, len: usize) -> (__m512i, __m512i, __mmask8) {
    let (lower, upper, mask) = match len {
        1 => (
            _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0),
            _mm512_set_epi64(7, 7, 5, 5, 3, 3, 1, 1),
            0xaa,
        ),
        2 => (
            _mm512_set_epi64(5, 4, 5, 4, 1, 0, 1, 0),
            _mm512_set_epi64(7, 6, 7, 6, 3, 2, 3, 2),
            0xcc,
        ),
        4 => (
            _mm512_set_epi64(3, 2, 1, 0, 3, 2, 1, 0),
            _mm512_set_epi64(7, 6, 5, 4, 7, 6, 5, 4),
            0xf0,
        ),
        _ => unreachable!("small IFMA stage length"),
    };
    (
        _mm512_permutexvar_epi64(lower, value),
        _mm512_permutexvar_epi64(upper, value),
        mask,
    )
}

#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn forward_small_stage<const D: usize>(
    values: &mut [u64; D],
    len: usize,
    twiddles: &[u64; 8],
    preconditions: &[u64; 8],
    prime: Ifma52Prime,
) {
    let modulus = _mm512_set1_epi64(prime.modulus as i64);
    let twice_modulus = _mm512_set1_epi64((2 * prime.modulus) as i64);
    // SAFETY: fixed tables contain eight lanes.
    let (multipliers, preconditions) = unsafe {
        (
            _mm512_loadu_si512(twiddles.as_ptr().cast()),
            _mm512_loadu_si512(preconditions.as_ptr().cast()),
        )
    };
    for start in (0..D).step_by(8) {
        // SAFETY: supported degrees are multiples of eight.
        unsafe {
            let pointer = values.as_mut_ptr().add(start);
            let value = _mm512_loadu_si512(pointer.cast());
            let (x, y, upper_mask) = paired_lanes(value, len);
            let sum = reduce_once(_mm512_add_epi64(x, y), modulus);
            let difference =
                reduce_once(_mm512_sub_epi64(_mm512_add_epi64(x, modulus), y), modulus);
            let product = normalize(
                mul_constant(difference, multipliers, preconditions, prime.modulus),
                modulus,
                twice_modulus,
            );
            _mm512_storeu_si512(
                pointer.cast(),
                _mm512_mask_blend_epi64(upper_mask, sum, product),
            );
        }
    }
}

#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn inverse_small_stage<const D: usize>(
    values: &mut [u64; D],
    len: usize,
    twiddles: &[u64; 8],
    preconditions: &[u64; 8],
    prime: Ifma52Prime,
) {
    let modulus = _mm512_set1_epi64(prime.modulus as i64);
    let twice_modulus = _mm512_set1_epi64((2 * prime.modulus) as i64);
    // SAFETY: fixed tables contain eight lanes.
    let (multipliers, preconditions) = unsafe {
        (
            _mm512_loadu_si512(twiddles.as_ptr().cast()),
            _mm512_loadu_si512(preconditions.as_ptr().cast()),
        )
    };
    for start in (0..D).step_by(8) {
        // SAFETY: supported degrees are multiples of eight.
        unsafe {
            let pointer = values.as_mut_ptr().add(start);
            let value = _mm512_loadu_si512(pointer.cast());
            let (x, y, upper_mask) = paired_lanes(value, len);
            let product = normalize(
                mul_constant(y, multipliers, preconditions, prime.modulus),
                modulus,
                twice_modulus,
            );
            let sum = reduce_once(_mm512_add_epi64(x, product), modulus);
            let difference = reduce_once(
                _mm512_sub_epi64(_mm512_add_epi64(x, modulus), product),
                modulus,
            );
            _mm512_storeu_si512(
                pointer.cast(),
                _mm512_mask_blend_epi64(upper_mask, sum, difference),
            );
        }
    }
}

#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn forward_butterfly(
    x: __m512i,
    y: __m512i,
    multiplier: __m512i,
    precondition: __m512i,
    prime: Ifma52Prime,
    twice_modulus: __m512i,
) -> (__m512i, __m512i) {
    // SAFETY: inherited target features.
    unsafe {
        let x = reduce_once(x, twice_modulus);
        let y = reduce_once(y, twice_modulus);
        (
            _mm512_add_epi64(x, y),
            mul_constant(
                _mm512_sub_epi64(_mm512_add_epi64(x, twice_modulus), y),
                multiplier,
                precondition,
                prime.modulus,
            ),
        )
    }
}

#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn inverse_butterfly(
    x: __m512i,
    y: __m512i,
    multiplier: __m512i,
    precondition: __m512i,
    prime: Ifma52Prime,
    twice_modulus: __m512i,
) -> (__m512i, __m512i) {
    // SAFETY: inherited target features.
    unsafe {
        let x = reduce_once(x, twice_modulus);
        let product = mul_constant(y, multiplier, precondition, prime.modulus);
        (
            _mm512_add_epi64(x, product),
            _mm512_sub_epi64(_mm512_add_epi64(x, twice_modulus), product),
        )
    }
}

#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn forward_radix4_stage<const D: usize>(
    values: &mut [u64; D],
    len: usize,
    prime: Ifma52Prime,
    twiddles: &Ifma52Twiddles<D>,
) {
    let half = len / 2;
    let outer_base = len - 1;
    let inner_base = half - 1;
    let twice_modulus = _mm512_set1_epi64((2 * prime.modulus) as i64);
    for start in (0..D).step_by(2 * len) {
        let mut index = 0;
        while index < half {
            // SAFETY: radix-four stage geometry keeps every eight-lane access in bounds.
            unsafe {
                let q0_pointer = values.as_mut_ptr().add(start + index);
                let q1_pointer = values.as_mut_ptr().add(start + index + half);
                let q2_pointer = values.as_mut_ptr().add(start + index + len);
                let q3_pointer = values.as_mut_ptr().add(start + index + len + half);
                let q0 = _mm512_loadu_si512(q0_pointer.cast());
                let q1 = _mm512_loadu_si512(q1_pointer.cast());
                let q2 = _mm512_loadu_si512(q2_pointer.cast());
                let q3 = _mm512_loadu_si512(q3_pointer.cast());
                let outer0 =
                    _mm512_loadu_si512(twiddles.forward.as_ptr().add(outer_base + index).cast());
                let outer0_precon = _mm512_loadu_si512(
                    twiddles
                        .forward_precon
                        .as_ptr()
                        .add(outer_base + index)
                        .cast(),
                );
                let outer1 = _mm512_loadu_si512(
                    twiddles
                        .forward
                        .as_ptr()
                        .add(outer_base + half + index)
                        .cast(),
                );
                let outer1_precon = _mm512_loadu_si512(
                    twiddles
                        .forward_precon
                        .as_ptr()
                        .add(outer_base + half + index)
                        .cast(),
                );
                let inner =
                    _mm512_loadu_si512(twiddles.forward.as_ptr().add(inner_base + index).cast());
                let inner_precon = _mm512_loadu_si512(
                    twiddles
                        .forward_precon
                        .as_ptr()
                        .add(inner_base + index)
                        .cast(),
                );
                let (sum0, difference0) =
                    forward_butterfly(q0, q2, outer0, outer0_precon, prime, twice_modulus);
                let (sum1, difference1) =
                    forward_butterfly(q1, q3, outer1, outer1_precon, prime, twice_modulus);
                let (out0, out1) =
                    forward_butterfly(sum0, sum1, inner, inner_precon, prime, twice_modulus);
                let (out2, out3) = forward_butterfly(
                    difference0,
                    difference1,
                    inner,
                    inner_precon,
                    prime,
                    twice_modulus,
                );
                _mm512_storeu_si512(q0_pointer.cast(), out0);
                _mm512_storeu_si512(q1_pointer.cast(), out1);
                _mm512_storeu_si512(q2_pointer.cast(), out2);
                _mm512_storeu_si512(q3_pointer.cast(), out3);
            }
            index += 8;
        }
    }
}

#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn inverse_radix4_stage<const D: usize>(
    values: &mut [u64; D],
    len: usize,
    prime: Ifma52Prime,
    twiddles: &Ifma52Twiddles<D>,
) {
    let outer_len = 2 * len;
    let inner_base = len - 1;
    let outer_base = outer_len - 1;
    let twice_modulus = _mm512_set1_epi64((2 * prime.modulus) as i64);
    for start in (0..D).step_by(4 * len) {
        let mut index = 0;
        while index < len {
            // SAFETY: radix-four stage geometry keeps every eight-lane access in bounds.
            unsafe {
                let q0_pointer = values.as_mut_ptr().add(start + index);
                let q1_pointer = values.as_mut_ptr().add(start + index + len);
                let q2_pointer = values.as_mut_ptr().add(start + index + outer_len);
                let q3_pointer = values.as_mut_ptr().add(start + index + outer_len + len);
                let q0 = _mm512_loadu_si512(q0_pointer.cast());
                let q1 = _mm512_loadu_si512(q1_pointer.cast());
                let q2 = _mm512_loadu_si512(q2_pointer.cast());
                let q3 = _mm512_loadu_si512(q3_pointer.cast());
                let inner =
                    _mm512_loadu_si512(twiddles.inverse.as_ptr().add(inner_base + index).cast());
                let inner_precon = _mm512_loadu_si512(
                    twiddles
                        .inverse_precon
                        .as_ptr()
                        .add(inner_base + index)
                        .cast(),
                );
                let outer0 =
                    _mm512_loadu_si512(twiddles.inverse.as_ptr().add(outer_base + index).cast());
                let outer0_precon = _mm512_loadu_si512(
                    twiddles
                        .inverse_precon
                        .as_ptr()
                        .add(outer_base + index)
                        .cast(),
                );
                let outer1 = _mm512_loadu_si512(
                    twiddles
                        .inverse
                        .as_ptr()
                        .add(outer_base + len + index)
                        .cast(),
                );
                let outer1_precon = _mm512_loadu_si512(
                    twiddles
                        .inverse_precon
                        .as_ptr()
                        .add(outer_base + len + index)
                        .cast(),
                );
                let (sum0, difference0) =
                    inverse_butterfly(q0, q1, inner, inner_precon, prime, twice_modulus);
                let (sum1, difference1) =
                    inverse_butterfly(q2, q3, inner, inner_precon, prime, twice_modulus);
                let (out0, out2) =
                    inverse_butterfly(sum0, sum1, outer0, outer0_precon, prime, twice_modulus);
                let (out1, out3) = inverse_butterfly(
                    difference0,
                    difference1,
                    outer1,
                    outer1_precon,
                    prime,
                    twice_modulus,
                );
                _mm512_storeu_si512(q0_pointer.cast(), out0);
                _mm512_storeu_si512(q1_pointer.cast(), out1);
                _mm512_storeu_si512(q2_pointer.cast(), out2);
                _mm512_storeu_si512(q3_pointer.cast(), out3);
            }
            index += 8;
        }
    }
}

/// Forward negacyclic transform.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub(super) unsafe fn forward<const D: usize>(
    values: &mut [u64; D],
    prime: Ifma52Prime,
    twiddles: &Ifma52Twiddles<D>,
) {
    // SAFETY: inherited target features and validated degree.
    unsafe { multiply_table(values, &twiddles.psi, &twiddles.psi_precon, prime) };
    let twice_modulus = _mm512_set1_epi64((2 * prime.modulus) as i64);
    let mut len = D / 2;
    while len >= 16 {
        // SAFETY: inherited target features and stage geometry.
        unsafe { forward_radix4_stage(values, len, prime, twiddles) };
        len /= 4;
    }
    while len >= 8 {
        let base = len - 1;
        for start in (0..D).step_by(2 * len) {
            let mut index = 0;
            while index < len {
                // SAFETY: stage geometry keeps eight-lane accesses in bounds.
                unsafe {
                    let x_pointer = values.as_mut_ptr().add(start + index);
                    let y_pointer = values.as_mut_ptr().add(start + index + len);
                    let multiplier =
                        _mm512_loadu_si512(twiddles.forward.as_ptr().add(base + index).cast());
                    let precondition = _mm512_loadu_si512(
                        twiddles.forward_precon.as_ptr().add(base + index).cast(),
                    );
                    let (sum, product) = forward_butterfly(
                        _mm512_loadu_si512(x_pointer.cast()),
                        _mm512_loadu_si512(y_pointer.cast()),
                        multiplier,
                        precondition,
                        prime,
                        twice_modulus,
                    );
                    _mm512_storeu_si512(x_pointer.cast(), sum);
                    _mm512_storeu_si512(y_pointer.cast(), product);
                }
                index += 8;
            }
        }
        len /= 2;
    }
    let modulus = _mm512_set1_epi64(prime.modulus as i64);
    for start in (0..D).step_by(8) {
        // SAFETY: supported degrees are multiples of eight.
        unsafe {
            let pointer = values.as_mut_ptr().add(start);
            let value = _mm512_loadu_si512(pointer.cast());
            _mm512_storeu_si512(pointer.cast(), normalize(value, modulus, twice_modulus));
        }
    }
    for (stage, small_len) in [4, 2, 1].into_iter().enumerate() {
        // SAFETY: inherited target features and fixed small-stage geometry.
        unsafe {
            forward_small_stage(
                values,
                small_len,
                &twiddles.forward_small[stage],
                &twiddles.forward_small_precon[stage],
                prime,
            )
        };
    }
}

/// Inverse negacyclic transform.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub(super) unsafe fn inverse<const D: usize>(
    values: &mut [u64; D],
    prime: Ifma52Prime,
    twiddles: &Ifma52Twiddles<D>,
) {
    for (stage, small_len) in [1, 2, 4].into_iter().enumerate() {
        // SAFETY: inherited target features and fixed small-stage geometry.
        unsafe {
            inverse_small_stage(
                values,
                small_len,
                &twiddles.inverse_small[stage],
                &twiddles.inverse_small_precon[stage],
                prime,
            )
        };
    }
    let mut len = 8;
    while len < D / 2 {
        // SAFETY: inherited target features and stage geometry.
        unsafe { inverse_radix4_stage(values, len, prime, twiddles) };
        len *= 4;
    }
    let twice_modulus = _mm512_set1_epi64((2 * prime.modulus) as i64);
    while len < D {
        let base = len - 1;
        for start in (0..D).step_by(2 * len) {
            let mut index = 0;
            while index < len {
                // SAFETY: stage geometry keeps eight-lane accesses in bounds.
                unsafe {
                    let x_pointer = values.as_mut_ptr().add(start + index);
                    let y_pointer = values.as_mut_ptr().add(start + index + len);
                    let multiplier =
                        _mm512_loadu_si512(twiddles.inverse.as_ptr().add(base + index).cast());
                    let precondition = _mm512_loadu_si512(
                        twiddles.inverse_precon.as_ptr().add(base + index).cast(),
                    );
                    let (sum, difference) = inverse_butterfly(
                        _mm512_loadu_si512(x_pointer.cast()),
                        _mm512_loadu_si512(y_pointer.cast()),
                        multiplier,
                        precondition,
                        prime,
                        twice_modulus,
                    );
                    _mm512_storeu_si512(x_pointer.cast(), sum);
                    _mm512_storeu_si512(y_pointer.cast(), difference);
                }
                index += 8;
            }
        }
        len *= 2;
    }
    // SAFETY: inherited target features and validated degree.
    unsafe {
        multiply_table(
            values,
            &twiddles.inverse_scale,
            &twiddles.inverse_scale_precon,
            prime,
        )
    };
}

/// Accumulate one pointwise product.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub(super) unsafe fn pointwise_accumulate<const D: usize>(
    accumulator: &mut [u64; D],
    lhs: &[u64; D],
    rhs: &[u64; D],
    prime: Ifma52Prime,
) {
    let modulus = _mm512_set1_epi64(prime.modulus as i64);
    for index in (0..D).step_by(8) {
        // SAFETY: supported degrees are multiples of eight.
        unsafe {
            let accumulator_pointer = accumulator.as_mut_ptr().add(index);
            let value = _mm512_loadu_si512(accumulator_pointer.cast());
            let lhs = _mm512_loadu_si512(lhs.as_ptr().add(index).cast());
            let rhs = _mm512_loadu_si512(rhs.as_ptr().add(index).cast());
            let product = mul_variable(lhs, rhs, prime);
            _mm512_storeu_si512(
                accumulator_pointer.cast(),
                reduce_once(_mm512_add_epi64(value, product), modulus),
            );
        }
    }
}
