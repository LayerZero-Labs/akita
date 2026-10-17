//! Raw two-row accumulation tiles; the caller splits at 128-term checkpoints.

use super::{TrinomialLimbSlots, DEGREE};
use std::arch::x86_64::*;

/// Accumulate two matrix rows against two source columns, in row-major order.
///
/// The caller detects AVX2, validates all slot tags and equal input lengths,
/// and bounds each intermediate sum by the existing checkpoint invariant.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn accumulate_2x2(
    sums: [&mut [i64; DEGREE]; 4],
    matrix: [&[TrinomialLimbSlots]; 2],
    sources: [&[TrinomialLimbSlots]; 2],
) {
    // SAFETY: The caller detects AVX2. Each offset selects eight complete
    // coefficients in the fixed arrays. The caller validates equal lengths
    // and splits at checkpoints, so signed products and sums fit i64.
    unsafe {
        let [s00, s01, s10, s11] = sums.map(|sum| sum.as_mut_ptr().cast::<__m256i>());
        let terms = matrix[0].len();
        let [matrix0, matrix1] = matrix.map(|row| row.as_ptr());
        let [source0, source1] = sources.map(|column| column.as_ptr());
        for offset in (0..DEGREE).step_by(8) {
            let block = offset / 4;
            let mut s00l = _mm256_loadu_si256(s00.add(block));
            let mut s00h = _mm256_loadu_si256(s00.add(block + 1));
            let mut s01l = _mm256_loadu_si256(s01.add(block));
            let mut s01h = _mm256_loadu_si256(s01.add(block + 1));
            let mut s10l = _mm256_loadu_si256(s10.add(block));
            let mut s10h = _mm256_loadu_si256(s10.add(block + 1));
            let mut s11l = _mm256_loadu_si256(s11.add(block));
            let mut s11h = _mm256_loadu_si256(s11.add(block + 1));
            for term in 0..terms {
                let a = (*matrix0.add(term)).values.as_ptr().add(offset);
                let b = (*matrix1.add(term)).values.as_ptr().add(offset);
                let al = _mm256_cvtepi32_epi64(_mm_loadu_si128(a.cast()));
                let ah = _mm256_cvtepi32_epi64(_mm_loadu_si128(a.add(4).cast()));
                let bl = _mm256_cvtepi32_epi64(_mm_loadu_si128(b.cast()));
                let bh = _mm256_cvtepi32_epi64(_mm_loadu_si128(b.add(4).cast()));
                let c = (*source0.add(term)).values.as_ptr().add(offset);
                let cl = _mm256_cvtepi32_epi64(_mm_loadu_si128(c.cast()));
                let ch = _mm256_cvtepi32_epi64(_mm_loadu_si128(c.add(4).cast()));
                s00l = _mm256_add_epi64(s00l, _mm256_mul_epi32(al, cl));
                s00h = _mm256_add_epi64(s00h, _mm256_mul_epi32(ah, ch));
                s10l = _mm256_add_epi64(s10l, _mm256_mul_epi32(bl, cl));
                s10h = _mm256_add_epi64(s10h, _mm256_mul_epi32(bh, ch));
                let d = (*source1.add(term)).values.as_ptr().add(offset);
                let dl = _mm256_cvtepi32_epi64(_mm_loadu_si128(d.cast()));
                let dh = _mm256_cvtepi32_epi64(_mm_loadu_si128(d.add(4).cast()));
                s01l = _mm256_add_epi64(s01l, _mm256_mul_epi32(al, dl));
                s01h = _mm256_add_epi64(s01h, _mm256_mul_epi32(ah, dh));
                s11l = _mm256_add_epi64(s11l, _mm256_mul_epi32(bl, dl));
                s11h = _mm256_add_epi64(s11h, _mm256_mul_epi32(bh, dh));
            }
            _mm256_storeu_si256(s00.add(block), s00l);
            _mm256_storeu_si256(s00.add(block + 1), s00h);
            _mm256_storeu_si256(s01.add(block), s01l);
            _mm256_storeu_si256(s01.add(block + 1), s01h);
            _mm256_storeu_si256(s10.add(block), s10l);
            _mm256_storeu_si256(s10.add(block + 1), s10h);
            _mm256_storeu_si256(s11.add(block), s11l);
            _mm256_storeu_si256(s11.add(block + 1), s11h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::trinomial::limb::TrinomialLimbDomain;

    fn slots(prime: u32, term: usize, lane: usize, scenario: usize) -> TrinomialLimbSlots {
        let bound = (prime - 1) as i32;
        TrinomialLimbSlots {
            prime,
            values: std::array::from_fn(|i| match scenario {
                0 => bound,
                1 => -bound,
                2 => {
                    if (term + lane + i).is_multiple_of(2) {
                        bound
                    } else {
                        -bound
                    }
                }
                _ => match (i + 5 * term + 3 * lane) % 7 {
                    0 => 0,
                    1 => 1,
                    2 => -1,
                    3 => bound,
                    4 => -bound,
                    5 => bound / 2,
                    _ => -(bound / 2),
                },
            }),
        }
    }

    #[test]
    fn limb_tiles_match_scalar_raw_checkpoint_bounded_sums() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
            for count in [1, 7, 8, 127, 128] {
                for scenario in 0..4 {
                    let matrix: [Vec<_>; 2] = std::array::from_fn(|row| {
                        (0..count)
                            .map(|term| slots(prime, term, row, scenario))
                            .collect()
                    });
                    let sources: [Vec<_>; 2] = std::array::from_fn(|column| {
                        (0..count)
                            .map(|term| slots(prime, term, column + 2, (scenario + column) % 4))
                            .collect()
                    });
                    let initial: [[i64; DEGREE]; 4] = std::array::from_fn(|cell| {
                        std::array::from_fn(|i| {
                            if (cell + i).is_multiple_of(2) {
                                i64::from(prime / 2)
                            } else {
                                -i64::from(prime / 2)
                            }
                        })
                    });
                    let mut expected = initial;
                    for (row, matrix) in matrix.iter().enumerate() {
                        for (column, source) in sources.iter().enumerate() {
                            for (a, b) in matrix.iter().zip(source) {
                                for ((sum, &a), &b) in expected[2 * row + column]
                                    .iter_mut()
                                    .zip(&a.values)
                                    .zip(&b.values)
                                {
                                    *sum += i64::from(a) * i64::from(b);
                                }
                            }
                        }
                    }
                    let mut actual = initial;
                    // SAFETY: AVX2 was detected above. Equal-length inputs
                    // contain count<=128 terms, and the carry is at most h.
                    unsafe {
                        accumulate_2x2(
                            actual.each_mut(),
                            [matrix[0].as_slice(), matrix[1].as_slice()],
                            [sources[0].as_slice(), sources[1].as_slice()],
                        )
                    };
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}
