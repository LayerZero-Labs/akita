//! Independent source transforms sharing unchanged YMM butterfly weights.

use super::{avx2, interleaved, vbmi, TrinomialLimbDomain, DEGREE, LANES, PIECE};
use std::arch::x86_64::*;

/// Transform two or four sources with one traversal of the remaining splits.
///
/// The caller detects AVX2 and AVX-512F/VL/BW/VBMI/BITALG and validates the
/// output prime tags. All output arrays are distinct mutable references; the
/// domain owns complete lookup tables and validated split metadata.
/// `SOURCES` is restricted to two or four at compile time.
#[target_feature(enable = "avx2,avx512f,avx512vl,avx512bw,avx512vbmi,avx512bitalg")]
pub(super) unsafe fn forward_interleaved_words<const SOURCES: usize>(
    domain: &TrinomialLimbDomain,
    words: &[[u128; 4]; SOURCES],
    tables: &[[[i32; 3 * LANES]; 256]],
    mut values: [&mut [i32; DEGREE]; SOURCES],
) {
    // This assertion is evaluated at monomorphization, never at runtime.
    const { assert!(SOURCES == 2 || SOURCES == 4) };
    // SAFETY: The caller detects AVX2 and AVX-512F/VL/BW/VBMI/BITALG.
    // The same index permutation and signed table evaluation populate each
    // source separately. The fixed arrays cover all 81 complete YMM rows;
    // the compile-time source count proves every constant pointer index below.
    unsafe {
        for (source, output) in words.iter().zip(values.iter_mut()) {
            let mut gathered = [0; PIECE];
            vbmi::indices(source, &mut gathered);
            interleaved::table_avx2(tables, &gathered, output);
        }
        let registers = avx2::ButterflyRegisters::new(domain, false);
        let pointers = values.map(|output| output.as_mut_ptr().cast::<__m256i>());
        for split in domain.splits.iter().skip(1) {
            let r = avx2::weights(&split.packed[0]);
            let r2 = avx2::weights(&split.packed[1]);
            if split.stride == 1 {
                avx2::row::<false, true>(pointers[0], split.start, 1, r, r2, &registers);
                avx2::row::<false, true>(pointers[1], split.start, 1, r, r2, &registers);
                if SOURCES == 4 {
                    avx2::row::<false, true>(pointers[2], split.start, 1, r, r2, &registers);
                    avx2::row::<false, true>(pointers[3], split.start, 1, r, r2, &registers);
                }
            } else {
                for offset in split.start..split.start + split.stride {
                    avx2::row::<false, false>(pointers[0], offset, split.stride, r, r2, &registers);
                    avx2::row::<false, false>(pointers[1], offset, split.stride, r, r2, &registers);
                    if SOURCES == 4 {
                        avx2::row::<false, false>(
                            pointers[2],
                            offset,
                            split.stride,
                            r,
                            r2,
                            &registers,
                        );
                        avx2::row::<false, false>(
                            pointers[3],
                            offset,
                            split.stride,
                            r,
                            r2,
                            &registers,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::trinomial::limb::Backend;

    fn packed(words: &[u128; 4]) -> [u64; 11] {
        let mut output = [0; 11];
        for (component, &word) in words.iter().enumerate() {
            for bit in 0..128 {
                let coefficient = 4 * bit + component;
                output[coefficient / 64] |= ((word >> bit) as u64 & 1) << (coefficient % 64);
            }
        }
        output
    }

    fn check(domain: &TrinomialLimbDomain, words: &[[u128; 4]]) {
        let mut actual = vec![domain.zero_slots(); words.len()];
        let mut expected = vec![domain.zero_slots(); words.len()];
        for (source, output) in words.iter().zip(&mut expected) {
            domain
                .forward_interleaved_bits_backend(&packed(source), output, Backend::Scalar)
                .unwrap();
        }
        assert!(domain
            .try_forward_interleaved_words_batch(words, &mut actual)
            .unwrap());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.values, expected.values);
            assert_eq!(actual.prime, expected.prime);
        }
    }

    fn random_words(seed: &mut u64) -> [u128; 4] {
        std::array::from_fn(|_| {
            let mut next = || {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                u128::from(*seed)
            };
            next() | (next() << 64)
        })
    }

    #[test]
    fn limb_source_batches_match_scalar_all_primes_basis_extrema_and_random() {
        if !super::super::backend::vbmi_available() {
            return;
        }
        let mut seed = 0x43af_90de_85b2_7189;
        for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
            let mut domain = TrinomialLimbDomain::new(prime).unwrap();
            domain.backend = Backend::Avx512VbmiBatch;
            for words in [
                [[0; 4]; 4],
                [[u128::MAX; 4]; 4],
                [[0; 4], [u128::MAX; 4], [0; 4], [u128::MAX; 4]],
                [[u128::MAX; 4], [0; 4], [u128::MAX; 4], [0; 4]],
            ] {
                check(&domain, &words);
                check(&domain, &words[..2]);
            }
            for side in 0..4 {
                for bit in 0..512 {
                    let mut words: [[u128; 4]; 4] =
                        std::array::from_fn(|_| random_words(&mut seed));
                    words[side] = [0; 4];
                    words[side][bit / 128] = 1 << (bit % 128);
                    check(&domain, &words);
                    if side < 2 {
                        check(&domain, &words[..2]);
                    }
                }
            }
            for _ in 0..128 {
                let words: [[u128; 4]; 4] = std::array::from_fn(|_| random_words(&mut seed));
                check(&domain, &words);
                check(&domain, &words[..2]);
            }
        }
    }
}
