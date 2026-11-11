//! Shared four-state Keccak-f1600 permutation for public setup streams.

use core::arch::x86_64::*;

/// Permute four independent Keccak-f1600 states in word-major layout.
///
/// State layout is x + 5*y; each vector contains that word from four sponges.
///
/// # Safety
/// The caller must check AVX2 support, including OS support, before calling.
#[target_feature(enable = "avx2")]
pub unsafe fn permute(state: &mut [[u64; 4]; 25]) {
    const ROTATION: [i64; 25] = [
        0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56,
        14,
    ];
    const ROUND: [u64; 24] = [
        0x0000000000000001,
        0x0000000000008082,
        0x800000000000808a,
        0x8000000080008000,
        0x000000000000808b,
        0x0000000080000001,
        0x8000000080008081,
        0x8000000000008009,
        0x000000000000008a,
        0x0000000000000088,
        0x0000000080008009,
        0x000000008000000a,
        0x000000008000808b,
        0x800000000000008b,
        0x8000000000008089,
        0x8000000000008003,
        0x8000000000008002,
        0x8000000000000080,
        0x000000000000800a,
        0x800000008000000a,
        0x8000000080008081,
        0x8000000000008080,
        0x0000000080000001,
        0x8000000080008008,
    ];
    let mut a: [__m256i; 25] =
        state.map(|w| _mm256_set_epi64x(w[3] as i64, w[2] as i64, w[1] as i64, w[0] as i64));
    for round in ROUND {
        let mut c = [_mm256_setzero_si256(); 5];
        for x in 0..5 {
            for y in 0..5 {
                c[x] = _mm256_xor_si256(c[x], a[x + 5 * y]);
            }
        }
        for x in 0..5 {
            let next = c[(x + 1) % 5];
            let rotated =
                _mm256_or_si256(_mm256_slli_epi64::<1>(next), _mm256_srli_epi64::<63>(next));
            let d = _mm256_xor_si256(c[(x + 4) % 5], rotated);
            for y in 0..5 {
                a[x + 5 * y] = _mm256_xor_si256(a[x + 5 * y], d);
            }
        }
        let mut b = [_mm256_setzero_si256(); 25];
        for x in 0..5 {
            for y in 0..5 {
                let i = x + 5 * y;
                // AVX2 variable shifts by 64 yield zero, so rotation zero works.
                b[y + 5 * ((2 * x + 3 * y) % 5)] = _mm256_or_si256(
                    _mm256_sllv_epi64(a[i], _mm256_set1_epi64x(ROTATION[i])),
                    _mm256_srlv_epi64(a[i], _mm256_set1_epi64x(64 - ROTATION[i])),
                );
            }
        }
        for x in 0..5 {
            for y in 0..5 {
                a[x + 5 * y] = _mm256_xor_si256(
                    b[x + 5 * y],
                    _mm256_andnot_si256(b[(x + 1) % 5 + 5 * y], b[(x + 2) % 5 + 5 * y]),
                );
            }
        }
        a[0] = _mm256_xor_si256(a[0], _mm256_set1_epi64x(round as i64));
    }
    for (word, vector) in state.iter_mut().zip(a) {
        // SAFETY: AVX2 is a caller precondition; word is a writable 32-byte
        // array and storeu does not require vector alignment.
        unsafe { _mm256_storeu_si256(word.as_mut_ptr().cast(), vector) };
    }
}
