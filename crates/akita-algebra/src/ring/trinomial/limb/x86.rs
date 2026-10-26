//! Shared eight-term evaluation/interpolation in an x86 SIMD entry.

use super::{TrinomialLimbDomain, LANES, PIECE};
use std::arch::x86_64::*;

#[inline(always)]
pub(super) unsafe fn basis_change<const INVERSE: bool>(
    domain: &TrinomialLimbDomain,
    input: &[i32],
    output: &mut [i32],
) {
    // SAFETY: Caller validates degree-648 buffers. Evaluation gathers eight
    // coefficients; interpolation reads eight consecutive slots. All operands
    // are centered, so eight products sum to at most 8h^2 < 2^57 in magnitude.
    // SIMD partial sums obey the same bound; the final reduction is exact.
    unsafe {
        let weights = if INVERSE {
            &domain.interpolation
        } else {
            &domain.evaluation
        };
        for t in 0..PIECE {
            let gathered: [i32; LANES];
            let ptr = if INVERSE {
                input.as_ptr().add(t * LANES)
            } else {
                gathered = std::array::from_fn(|j| input[t + PIECE * j]);
                gathered.as_ptr()
            };
            let a = _mm256_loadu_si256(ptr.cast());
            let odd_a = _mm256_shuffle_epi32::<0xf5>(a);
            for (lane, weights) in weights.iter().enumerate() {
                let b = _mm256_loadu_si256(weights.as_ptr().cast());
                let products = _mm256_add_epi64(
                    _mm256_mul_epi32(a, b),
                    _mm256_mul_epi32(odd_a, _mm256_shuffle_epi32::<0xf5>(b)),
                );
                let pair = _mm_add_epi64(
                    _mm256_castsi256_si128(products),
                    _mm256_extracti128_si256::<1>(products),
                );
                let sum = _mm_cvtsi128_si64(_mm_add_epi64(pair, _mm_unpackhi_epi64(pair, pair)));
                let index = if INVERSE {
                    t + PIECE * lane
                } else {
                    t * LANES + lane
                };
                output[index] = domain.arithmetic.center(sum);
            }
        }
    }
}
