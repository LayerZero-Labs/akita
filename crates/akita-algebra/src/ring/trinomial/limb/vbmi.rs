//! Direct four-word source indices using VBMI byte selection and BITALG bits.

use super::{TrinomialLimbDomain, DEGREE, LANES, PIECE};
use std::arch::x86_64::*;

#[repr(C, align(64))]
struct Control {
    bytes: [u8; 64],
    bits: [u8; 64],
    valid: u64,
}

const fn control(group: usize) -> Control {
    let mut result = Control {
        bytes: [0; 64],
        bits: [0; 64],
        valid: 0,
    };
    let mut position = 0;
    while position < 64 {
        let t = group * 8 + position / 8;
        let j = position % 8;
        let n = t + 81 * j;
        if t < 81 && n < 512 {
            let component = n % 4;
            let source_bit = n / 4;
            result.bytes[position] = (16 * component + source_bit / 8) as u8;
            // Each output byte has its own qword containing eight selected
            // source bytes. BITALG addresses bit s%8 of selected byte j.
            result.bits[position] = (8 * j + source_bit % 8) as u8;
            result.valid |= 1 << position;
        }
        position += 1;
    }
    result
}

const CONTROLS: [Control; 11] = [
    control(0),
    control(1),
    control(2),
    control(3),
    control(4),
    control(5),
    control(6),
    control(7),
    control(8),
    control(9),
    control(10),
];

#[inline(always)]
unsafe fn group<const GROUP: usize>(source: __m512i) -> u64 {
    // SAFETY: The caller detects AVX2, AVX-512F/VL/BW/VBMI/BITALG. Both
    // controls are complete 64-byte arrays. Every byte selector is in 0..64;
    // every bit selector is in 0..64 within its destination qword.
    unsafe {
        let selectors = &CONTROLS[GROUP];
        let selected =
            _mm512_permutexvar_epi8(_mm512_loadu_si512(selectors.bytes.as_ptr().cast()), source);
        _mm512_bitshuffle_epi64_mask(selected, _mm512_loadu_si512(selectors.bits.as_ptr().cast()))
            & selectors.valid
    }
}

/// Gather exactly the indices of four interleaved little-endian source words.
///
/// The caller must detect AVX2, AVX-512F/VL/BW/VBMI/BITALG before entering.
/// No source bit outside the 512 supplied bits contributes to any index.
#[target_feature(enable = "avx2,avx512f,avx512vl,avx512bw,avx512vbmi,avx512bitalg")]
#[inline]
pub(super) unsafe fn indices(words: &[u128; 4], out: &mut [u8; 81]) {
    // SAFETY: The caller detects AVX2, AVX-512F/VL/BW/VBMI/BITALG. Four
    // x86_64 little-endian u128 words occupy exactly this 64-byte input load.
    // Ten complete groups write bytes 0..80; the final group writes only 80.
    unsafe {
        let source = _mm512_loadu_si512(words.as_ptr().cast());
        out[0..8].copy_from_slice(&group::<0>(source).to_le_bytes());
        out[8..16].copy_from_slice(&group::<1>(source).to_le_bytes());
        out[16..24].copy_from_slice(&group::<2>(source).to_le_bytes());
        out[24..32].copy_from_slice(&group::<3>(source).to_le_bytes());
        out[32..40].copy_from_slice(&group::<4>(source).to_le_bytes());
        out[40..48].copy_from_slice(&group::<5>(source).to_le_bytes());
        out[48..56].copy_from_slice(&group::<6>(source).to_le_bytes());
        out[56..64].copy_from_slice(&group::<7>(source).to_le_bytes());
        out[64..72].copy_from_slice(&group::<8>(source).to_le_bytes());
        out[72..80].copy_from_slice(&group::<9>(source).to_le_bytes());
        out[80] = group::<10>(source) as u8;
    }
}

/// Direct source indices followed by the unchanged signed tables and butterflies.
#[target_feature(enable = "avx2,avx512f,avx512vl,avx512bw,avx512vbmi,avx512bitalg")]
pub(super) unsafe fn forward_interleaved(
    domain: &TrinomialLimbDomain,
    words: &[u128; 4],
    tables: &[[[i32; 3 * LANES]; 256]],
    values: &mut [i32; DEGREE],
) {
    // SAFETY: The caller detects AVX2 and AVX-512F/VL/BW/VBMI/BITALG.
    // Complete source/output arrays and domain-owned table rows are validated.
    unsafe {
        let mut gathered = [0; PIECE];
        indices(words, &mut gathered);
        super::interleaved::table_avx2(tables, &gathered, values);
        super::avx2::transform_inner::<false>(domain, values, true);
    }
}

#[cfg(test)]
mod tests {
    use super::super::{backend, Backend};
    use super::*;

    fn packed(words: &[u128; 4]) -> [u64; 11] {
        // Build coefficient-order bits one bit at a time, independently of
        // the SIMD byte/bit selector derivation, then use the scalar gather.
        let mut packed = [0; 11];
        for (component, word) in words.iter().copied().enumerate() {
            for bit in 0..128 {
                let position = 4 * bit + component;
                packed[position / 64] |= ((word >> bit) as u64 & 1) << (position % 64);
            }
        }
        packed
    }

    fn assert_indices(words: &[u128; 4]) {
        let packed = packed(words);
        let mut expected = [0; 81];
        super::super::gather::indices(&packed, &mut expected);
        let mut actual = [0; 81];
        // SAFETY: The runtime guard in the test detects AVX2 and
        // AVX-512F/VL/BW/VBMI/BITALG before calling this helper.
        unsafe { indices(words, &mut actual) };
        assert_eq!(actual, expected);
    }

    #[test]
    fn limb_vbmi_indices_match_every_source_basis_zero_ones_and_random() {
        if !backend::vbmi_available() {
            return;
        }
        assert_indices(&[0; 4]);
        assert_indices(&[u128::MAX; 4]);
        for position in 0..512 {
            let mut words = [0; 4];
            words[position / 128] = 1 << (position % 128);
            assert_indices(&words);
        }
        let mut seed = 0x9b7e_2c01_a458_630d_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..1_000 {
            let words = std::array::from_fn(|_| u128::from(next()) | (u128::from(next()) << 64));
            assert_indices(&words);
        }
    }

    fn assert_slots(domain: &TrinomialLimbDomain, words: &[u128; 4]) {
        let mut expected = domain.zero_slots();
        domain
            .forward_interleaved_bits_backend(&packed(words), &mut expected, Backend::Scalar)
            .unwrap();
        let mut actual = domain.zero_slots();
        assert!(domain
            .try_forward_interleaved_words_batch(
                std::slice::from_ref(words),
                std::slice::from_mut(&mut actual)
            )
            .unwrap());
        assert_eq!(actual.values, expected.values);
        assert_eq!(actual.prime, expected.prime);
    }

    #[test]
    fn limb_vbmi_source_slots_match_scalar_for_all_primes_and_basis_inputs() {
        if !backend::vbmi_available() {
            return;
        }
        let mut seed = 0x68d9_027b_4cae_1f35_u64;
        for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
            let mut domain = TrinomialLimbDomain::new(prime).unwrap();
            domain.backend = Backend::Avx512VbmiBatch;
            assert_slots(&domain, &[0; 4]);
            assert_slots(&domain, &[u128::MAX; 4]);
            for position in 0..512 {
                let mut words = [0; 4];
                words[position / 128] = 1 << (position % 128);
                assert_slots(&domain, &words);
            }
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            for _ in 0..64 {
                let words =
                    std::array::from_fn(|_| u128::from(next()) | (u128::from(next()) << 64));
                assert_slots(&domain, &words);
            }
        }
    }

    #[test]
    fn limb_vbmi_fallback_leaves_output_and_tables_unchanged() {
        for backend in backend::available_backends()
            .into_iter()
            .filter(|&backend| backend != Backend::Avx512VbmiBatch)
        {
            for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
                let mut domain = TrinomialLimbDomain::new(prime).unwrap();
                domain.backend = backend;
                let mut output = domain.zero_slots();
                output.values.fill(17);
                let original = output.clone();
                assert!(!domain
                    .try_forward_interleaved_words_batch(
                        &[[u128::MAX; 4]],
                        std::slice::from_mut(&mut output)
                    )
                    .unwrap());
                assert_eq!(output.values, original.values);
                assert_eq!(output.prime, original.prime);
                assert!(domain.interleaved_first_level.get().is_none());
            }
        }
    }

    #[test]
    fn limb_vbmi_rejects_mismatched_output_prime_before_initialization() {
        for backend in backend::available_backends() {
            for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
                let mut domain = TrinomialLimbDomain::new(prime).unwrap();
                domain.backend = backend;
                let other = TrinomialLimbDomain::ADMITTED_PRIMES
                    .into_iter()
                    .find(|&p| p != prime)
                    .unwrap();
                let mut output = TrinomialLimbDomain::new(other).unwrap().zero_slots();
                output.values.fill(17);
                let original = output.clone();
                assert!(domain
                    .try_forward_interleaved_words_batch(
                        &[[u128::MAX; 4]],
                        std::slice::from_mut(&mut output)
                    )
                    .is_err());
                assert_eq!(output.values, original.values);
                assert_eq!(output.prime, original.prime);
                assert!(domain.interleaved_first_level.get().is_none());
            }
        }
    }
}
