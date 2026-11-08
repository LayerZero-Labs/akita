//! Batch source validation and all source/output tail lengths on every backend.

use super::*;

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

#[test]
fn limb_source_batch_tails_and_fallback_match_all_backends() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        for backend in available_backends() {
            let supported = {
                #[cfg(target_arch = "x86_64")]
                {
                    backend == Backend::Avx512VbmiBatch
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    false
                }
            };
            for count in 0..=9 {
                let mut domain = TrinomialLimbDomain::new(prime).unwrap();
                domain.backend = backend;
                let words: Vec<[u128; 4]> = (0..count)
                    .map(|source| {
                        std::array::from_fn(|component| {
                            let value = 0x31a7_b6d5_a290_e4c3_5b8f_610d_27c9_fa02u128;
                            value.rotate_left((19 * source + 7 * component) as u32)
                        })
                    })
                    .collect();
                let mut outputs = vec![domain.zero_slots(); count];
                for (i, output) in outputs.iter_mut().enumerate() {
                    output.values.fill(i as i32 + 17);
                }
                let before = outputs.clone();
                assert_eq!(
                    domain
                        .try_forward_interleaved_words_batch(&words, &mut outputs)
                        .unwrap(),
                    supported
                );
                if supported {
                    for (word, output) in words.iter().zip(&outputs) {
                        let mut expected = domain.zero_slots();
                        domain
                            .forward_interleaved_bits_backend(
                                &packed(word),
                                &mut expected,
                                Backend::Scalar,
                            )
                            .unwrap();
                        assert_eq!(output.values, expected.values);
                    }
                    if count == 0 {
                        assert_eq!(domain.table_storage_bytes(), 0);
                    }
                } else {
                    for (output, before) in outputs.iter().zip(&before) {
                        assert_eq!(output.values, before.values);
                        assert_eq!(output.prime, before.prime);
                    }
                    assert_eq!(domain.table_storage_bytes(), 0);
                }
            }
        }
    }
}

#[test]
fn limb_source_batch_rejects_all_bad_shapes_and_tags_atomically() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let other = TrinomialLimbDomain::ADMITTED_PRIMES
            .into_iter()
            .find(|&candidate| candidate != prime)
            .unwrap();
        for backend in available_backends() {
            for count in (0..=9).filter(|&count| count != 4) {
                let mut domain = TrinomialLimbDomain::new(prime).unwrap();
                domain.backend = backend;
                let mut outputs = vec![domain.zero_slots(); count];
                for (i, output) in outputs.iter_mut().enumerate() {
                    output.values.fill(i as i32 + 17);
                }
                let before = outputs.clone();
                assert!(domain
                    .try_forward_interleaved_words_batch(&[[u128::MAX; 4]; 4], &mut outputs)
                    .is_err());
                for (output, before) in outputs.iter().zip(&before) {
                    assert_eq!(output.values, before.values);
                    assert_eq!(output.prime, before.prime);
                }
                assert_eq!(domain.table_storage_bytes(), 0);
            }
            for invalid in 0..9 {
                let mut domain = TrinomialLimbDomain::new(prime).unwrap();
                domain.backend = backend;
                let mut outputs = vec![domain.zero_slots(); 9];
                for (i, output) in outputs.iter_mut().enumerate() {
                    output.values.fill(i as i32 + 17);
                }
                outputs[invalid].prime = other;
                let before = outputs.clone();
                assert!(domain
                    .try_forward_interleaved_words_batch(&[[u128::MAX; 4]; 9], &mut outputs)
                    .is_err());
                for (output, before) in outputs.iter().zip(&before) {
                    assert_eq!(output.values, before.values);
                    assert_eq!(output.prime, before.prime);
                }
                assert_eq!(domain.table_storage_bytes(), 0);
            }
        }
    }
}
