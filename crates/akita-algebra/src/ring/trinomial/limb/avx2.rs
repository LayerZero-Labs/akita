//! Eight signed 32-bit lanes with exact rounded-quotient multiplication.

use super::arithmetic::PackedTwiddles;
use super::{TrinomialLimbDomain, TrinomialLimbSlots, DEGREE, LANES};
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn rounded_quotient(a: __m256i, q: __m256i, odd_q: __m256i, round: __m256i) -> __m256i {
    // SAFETY: Called only within this backend's feature-enabled entries.
    unsafe {
        let even = _mm256_srli_epi64::<31>(_mm256_add_epi64(_mm256_mul_epi32(a, q), round));
        // Duplicating odd dwords leaves the unused halves unconstrained by zero
        // extension, avoiding LLVM's signed-to-unsigned multiply expansion on the
        // baseline target. The invariant odd quotient is prepared once per split.
        let odd = _mm256_slli_epi64::<1>(_mm256_add_epi64(
            _mm256_mul_epi32(_mm256_shuffle_epi32::<0xf5>(a), odd_q),
            round,
        ));
        // Logical shifts preserve the low 32 bits of each signed quotient.
        // The odd quotients occupy the high halves of their 64-bit lanes.
        _mm256_blend_epi32::<0xAA>(even, odd)
    }
}

#[inline(always)]
pub(super) unsafe fn mul(
    a: __m256i,
    w: (__m256i, __m256i, __m256i),
    p: __m256i,
    round: __m256i,
) -> __m256i {
    // SAFETY: Only register operations. Centered twiddles have |q|<=2^30,
    // excluding SQRDMULH saturation. Both signed 64-bit products plus 2^30
    // fit i64. Low products/subtraction recover the exact bounded remainder.
    unsafe {
        let t = rounded_quotient(a, w.1, w.2, round);
        _mm256_sub_epi32(_mm256_mullo_epi32(a, w.0), _mm256_mullo_epi32(t, p))
    }
}

#[inline(always)]
unsafe fn reduce_one(a: __m256i, shift: __m128i, bias: __m256i, p: __m256i) -> __m256i {
    // SAFETY: Called only within this backend's feature-enabled entries.
    unsafe {
        // Domain construction proves B4+bias < i32::MAX and q=2^(31-k).
        // Therefore floor((a*q+2^30)/2^31) = (a+2^(k-1)) >> k exactly.
        let t = _mm256_sra_epi32(_mm256_add_epi32(a, bias), shift);
        _mm256_sub_epi32(a, _mm256_mullo_epi32(t, p))
    }
}

#[inline(always)]
unsafe fn center(a: __m256i, p: __m256i, half: __m256i) -> __m256i {
    // SAFETY: Called only within this backend's feature-enabled entries.
    unsafe {
        // Input has |a| <= 3h < p+h; one correction centers it without overflow.
        let a = _mm256_sub_epi32(a, _mm256_and_si256(_mm256_cmpgt_epi32(a, half), p));
        let negative_half = _mm256_sub_epi32(_mm256_setzero_si256(), half);
        _mm256_add_epi32(a, _mm256_and_si256(_mm256_cmpgt_epi32(negative_half, a), p))
    }
}

#[inline(always)]
pub(super) unsafe fn weights(w: &PackedTwiddles) -> (__m256i, __m256i, __m256i) {
    // SAFETY: Separate contiguous arrays each contain eight initialized lanes.
    unsafe {
        let q = _mm256_loadu_si256(w.quotients.as_ptr().cast());
        (
            _mm256_loadu_si256(w.values.as_ptr().cast()),
            q,
            _mm256_shuffle_epi32::<0xf5>(q),
        )
    }
}

pub(super) struct ButterflyRegisters {
    omega: (__m256i, __m256i, __m256i),
    shift: __m128i,
    bias: __m256i,
    p: __m256i,
    half: __m256i,
    round: __m256i,
}

impl ButterflyRegisters {
    #[inline(always)]
    pub(super) unsafe fn new(domain: &TrinomialLimbDomain, inverse: bool) -> Self {
        // SAFETY: Called only within a feature-enabled x86 entry. Domain
        // admission validates the normalization shift and lazy bounds.
        unsafe {
            let w = if inverse {
                domain.inverse_omega
            } else {
                domain.omega
            };
            Self {
                omega: (
                    _mm256_set1_epi32(w.value),
                    _mm256_set1_epi32(w.quotient),
                    _mm256_set1_epi32(w.quotient),
                ),
                shift: _mm_cvtsi32_si128((31 - domain.one.quotient.trailing_zeros()) as i32),
                bias: _mm256_set1_epi32(1 << ((31 - domain.one.quotient.trailing_zeros()) - 1)),
                p: _mm256_set1_epi32(domain.prime() as i32),
                half: _mm256_set1_epi32((domain.prime() / 2) as i32),
                round: _mm256_set1_epi64x(1 << 30),
            }
        }
    }
}

#[inline(always)]
pub(super) unsafe fn triple<const INVERSE: bool, const NORMALIZE: bool>(
    a: __m256i,
    mut b: __m256i,
    mut c: __m256i,
    r: (__m256i, __m256i, __m256i),
    r2: (__m256i, __m256i, __m256i),
    registers: &ButterflyRegisters,
) -> (__m256i, __m256i, __m256i) {
    // SAFETY: The caller's feature entry establishes AVX2. Forward inputs
    // obey B0..B4; inverse inputs are centered and obey I. Every transient
    // add/subtract fits i32, and each inverse level restores centered h.
    unsafe {
        let ButterflyRegisters {
            omega,
            shift,
            bias,
            p,
            half,
            round,
        } = *registers;
        if !INVERSE {
            b = mul(b, r, p, round);
            c = mul(c, r2, p, round);
        }
        let d = mul(_mm256_sub_epi32(b, c), omega, p, round);
        let mut o0 = _mm256_add_epi32(_mm256_add_epi32(a, b), c);
        let mut o1 = _mm256_add_epi32(_mm256_sub_epi32(a, c), d);
        let mut o2 = _mm256_sub_epi32(_mm256_sub_epi32(a, b), d);
        if INVERSE {
            o0 = center(o0, p, half);
            o1 = center(mul(o1, r, p, round), p, half);
            o2 = center(mul(o2, r2, p, round), p, half);
        } else if NORMALIZE {
            o0 = reduce_one(o0, shift, bias, p);
            o1 = reduce_one(o1, shift, bias, p);
            o2 = reduce_one(o2, shift, bias, p);
        }
        (o0, o1, o2)
    }
}

#[inline(always)]
pub(super) unsafe fn row<const INVERSE: bool, const NORMALIZE: bool>(
    ptr: *mut __m256i,
    offset: usize,
    step: usize,
    r: (__m256i, __m256i, __m256i),
    r2: (__m256i, __m256i, __m256i),
    registers: &ButterflyRegisters,
) {
    // SAFETY: Split metadata supplies three disjoint complete rows within
    // the fixed array; triple enforces the shared register arithmetic.
    unsafe {
        let (a, b, c) = triple::<INVERSE, NORMALIZE>(
            _mm256_loadu_si256(ptr.add(offset)),
            _mm256_loadu_si256(ptr.add(offset + step)),
            _mm256_loadu_si256(ptr.add(offset + 2 * step)),
            r,
            r2,
            registers,
        );
        _mm256_storeu_si256(ptr.add(offset), a);
        _mm256_storeu_si256(ptr.add(offset + step), b);
        _mm256_storeu_si256(ptr.add(offset + 2 * step), c);
    }
}

#[inline(always)]
pub(super) unsafe fn transform_inner<const INVERSE: bool>(
    domain: &TrinomialLimbDomain,
    values: &mut [i32; DEGREE],
    skip_first: bool,
) {
    // SAFETY: Split metadata is constructed by the domain and partitions
    // the 81 rows into disjoint triples at strides 27,9,3,1 (reversed for
    // inverse). Skipping the first forward split retains B2-B4. Normalizing
    // stride-one outputs performs the same final multiplication by one as
    // the scalar backend, retaining identical lazy representatives.
    unsafe {
        let registers = ButterflyRegisters::new(domain, INVERSE);
        let ptr = values.as_mut_ptr().cast::<__m256i>();
        for index in usize::from(skip_first)..domain.splits.len() {
            let split = &domain.splits[if INVERSE {
                domain.splits.len() - 1 - index
            } else {
                index
            }];
            let index = if INVERSE { 2 } else { 0 };
            let r = weights(&split.packed[index]);
            let r2 = weights(&split.packed[index + 1]);
            if split.stride == 1 {
                row::<INVERSE, true>(ptr, split.start, 1, r, r2, &registers);
            } else {
                for offset in split.start..split.start + split.stride {
                    row::<INVERSE, false>(ptr, offset, split.stride, r, r2, &registers);
                }
            }
        }
    }
}

/// A complete centered forward call inside one feature boundary.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn forward_centered(
    domain: &TrinomialLimbDomain,
    input: &[i32],
    values: &mut [i32; DEGREE],
) {
    // SAFETY: Dispatch establishes the ISA; the public boundary validates input.
    unsafe {
        super::x86::basis_change::<false>(domain, input, values);
        transform_inner::<false>(domain, values, false);
    }
}

/// A complete inverse call inside one feature boundary.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_centered(
    domain: &TrinomialLimbDomain,
    input: &[i32; DEGREE],
    output: &mut [i32],
) {
    // SAFETY: Dispatch establishes the ISA; public validation fixes output size.
    unsafe {
        let mut values = input.map(|v| domain.arithmetic.center(i64::from(v)));
        transform_inner::<true>(domain, &mut values, false);
        super::x86::basis_change::<true>(domain, &values, output);
    }
}

/// Gather, signed table evaluation, and remaining levels in one feature entry.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn forward_interleaved(
    domain: &TrinomialLimbDomain,
    bits: &[u64; 11],
    tables: &[[[i32; 3 * LANES]; 256]],
    values: &mut [i32; DEGREE],
) {
    // SAFETY: Bits and output are validated; the domain owns all table rows.
    unsafe {
        let mut indices = [0; super::PIECE];
        super::gather::indices(bits, &mut indices);
        super::interleaved::table_avx2(tables, &indices, values);
        transform_inner::<false>(domain, values, true);
    }
}

/// Caller must establish AVX2 support before entering this backend.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn transform(
    domain: &TrinomialLimbDomain,
    values: &mut [i32; DEGREE],
    inverse: bool,
    skip_first: bool,
) {
    // SAFETY: AVX2 is required by this function's contract. The domain owns
    // valid split metadata; fixed-size values supply all 81 complete rows.
    unsafe {
        if inverse {
            transform_inner::<true>(domain, values, skip_first);
        } else {
            transform_inner::<false>(domain, values, skip_first);
        }
    }
}

/// Caller must establish AVX2 support and retain the 128-term checkpoint.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn accumulate(
    sums: &mut [i64; DEGREE],
    lhs: &[i32; DEGREE],
    rhs: &[i32; DEGREE],
) {
    // SAFETY: Each complete four-lane input load stays within its array;
    // widening produces four signed i64 products in the original order.
    // The matching four-lane sum load/store is in bounds. The existing
    // h+128*(p-1)^2<i64::MAX checkpoint bound prevents signed overflow.
    unsafe {
        let mut add = |offset: usize| {
            let a = _mm256_cvtepi32_epi64(_mm_loadu_si128(lhs.as_ptr().add(offset).cast()));
            let b = _mm256_cvtepi32_epi64(_mm_loadu_si128(rhs.as_ptr().add(offset).cast()));
            let ptr = sums.as_mut_ptr().add(offset).cast::<__m256i>();
            let sum = _mm256_add_epi64(_mm256_loadu_si256(ptr), _mm256_mul_epi32(a, b));
            _mm256_storeu_si256(ptr, sum);
        };
        // Four independent sum updates expose latency overlap on baseline x86.
        // The final eight coefficients are a complete, bounded tail.
        for block in (0..DEGREE - 8).step_by(16) {
            add(block);
            add(block + 4);
            add(block + 8);
            add(block + 12);
        }
        add(DEGREE - 8);
        add(DEGREE - 4);
    }
}

/// Caller must establish AVX2 support and retain the 128-term checkpoint.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn accumulate_batch(
    sums: &mut [i64; DEGREE],
    lhs: &[TrinomialLimbSlots],
    rhs: &[TrinomialLimbSlots],
) {
    // SAFETY: Complete eight-lane chunks cover DEGREE exactly. Each i32
    // load contains four lanes, and each i64 load/store contains four lanes.
    // The caller splits at checkpoints, so every intermediate sum fits i64.
    unsafe {
        for offset in (0..DEGREE).step_by(8) {
            let ptr = sums.as_mut_ptr().add(offset).cast::<__m256i>();
            let mut low = _mm256_loadu_si256(ptr);
            let mut high = _mm256_loadu_si256(ptr.add(1));
            for (a, b) in lhs.iter().zip(rhs) {
                let a_ptr = a.values.as_ptr().add(offset);
                let b_ptr = b.values.as_ptr().add(offset);
                let a_low = _mm256_cvtepi32_epi64(_mm_loadu_si128(a_ptr.cast()));
                let b_low = _mm256_cvtepi32_epi64(_mm_loadu_si128(b_ptr.cast()));
                let a_high = _mm256_cvtepi32_epi64(_mm_loadu_si128(a_ptr.add(4).cast()));
                let b_high = _mm256_cvtepi32_epi64(_mm_loadu_si128(b_ptr.add(4).cast()));
                low = _mm256_add_epi64(low, _mm256_mul_epi32(a_low, b_low));
                high = _mm256_add_epi64(high, _mm256_mul_epi32(a_high, b_high));
            }
            _mm256_storeu_si256(ptr, low);
            _mm256_storeu_si256(ptr.add(1), high);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::arithmetic::{self, Arithmetic, Twiddle};
    use super::*;

    #[target_feature(enable = "avx2")]
    unsafe fn mul_lanes(arithmetic: Arithmetic, input: [i32; LANES], w: Twiddle) -> [i32; LANES] {
        let mut output = [0; LANES];
        // SAFETY: AVX2 is required on entry. Complete loads/stores cover the
        // initialized arrays, and centered twiddles satisfy |q|<=2^30.
        unsafe {
            let result = mul(
                _mm256_loadu_si256(input.as_ptr().cast()),
                (
                    _mm256_set1_epi32(w.value),
                    _mm256_set1_epi32(w.quotient),
                    _mm256_set1_epi32(w.quotient),
                ),
                _mm256_set1_epi32(arithmetic.prime as i32),
                _mm256_set1_epi64x(1 << 30),
            );
            _mm256_storeu_si256(output.as_mut_ptr().cast(), result);
        }
        output
    }

    #[test]
    fn avx2_constant_products_match_i128_at_lane_and_lazy_extremes() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for p in TrinomialLimbDomain::ADMITTED_PRIMES {
            let arithmetic = Arithmetic { prime: p };
            let mut operands = vec![i32::MIN, i32::MAX, -1, 0, 1];
            for bound in arithmetic::forward_bounds(p) {
                operands.extend([bound as i32, -(bound as i32)]);
            }
            let half = (p / 2) as i32;
            for weight in [-half, -1, 0, 1, half] {
                let twiddle = Twiddle::new(arithmetic, weight);
                let expected_quotient = (i128::from(weight) * (1i128 << 31) + i128::from(p / 2))
                    .div_euclid(i128::from(p));
                assert_eq!(i128::from(twiddle.quotient), expected_quotient);
                for offset in 0..operands.len() {
                    let lanes =
                        std::array::from_fn(|lane| operands[(offset + lane) % operands.len()]);
                    // SAFETY: Runtime detection above establishes AVX2.
                    let result = unsafe { mul_lanes(arithmetic, lanes, twiddle) };
                    for (a, actual) in lanes.into_iter().zip(result) {
                        let quotient = (i128::from(a) * expected_quotient + (1i128 << 30)) >> 31;
                        let expected =
                            i128::from(a) * i128::from(weight) - quotient * i128::from(p);
                        assert_eq!(i128::from(actual), expected);
                        assert_eq!(actual, arithmetic.mul(a, twiddle));
                        assert!(
                            i64::from(actual).abs()
                                <= arithmetic::multiply_bound(p, i64::from(a).abs())
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "avx2/normalization.rs"]
mod normalization;
