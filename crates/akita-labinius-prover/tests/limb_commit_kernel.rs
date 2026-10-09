#![cfg(feature = "labinius")]

use akita_algebra::{
    binary::{field_switch::SwitchField, BinaryField128, BinaryField192},
    MinusTrinomial, TrinomialRing, TrinomialWideLimbDomain, TrinomialWideLimbSlots,
};
use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use akita_error::AkitaError;
use akita_labinius_prover::{
    commit_binary_clear_limb_prepared, commit_binary_clear_prepared,
    commit_kernel::pack_binary_element_i8, limb_commit_kernel::pack_binary_element_bits,
    PreparedCommitMatrix, PreparedLimbCommitMatrix,
};
use akita_labinius_verifier::BinaryClearSetup;
use akita_params::sis::labinius::{LabiniusCoefficientPrime, LabiniusRingDegree};
use jolt_field::{CanonicalEncoding, Prime128OffsetA7F7, Ring};
use rand::{rngs::StdRng, RngCore, SeedableRng};

const D: usize = 648;
const K: usize = 4;
const Q: u32 = 268_433_353;

fn random_matrix(rng: &mut StdRng, prime: u32, rank: usize, width: usize) -> Vec<u32> {
    (0..rank * width * D)
        .map(|_| (rng.next_u64() % u64::from(prime)) as u32)
        .collect()
}

fn random_source(rng: &mut StdRng, words: usize) -> Vec<u128> {
    (0..words)
        .map(|_| u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64))
        .collect()
}

/// Decode the defining signed interleaving directly, independently of either
/// kernel's source packer or transform tables. Reduce the integer convolution
/// in descending degree using Y^648 = Y^324 - 1, before reducing modulo q.
fn integer_oracle<H: SwitchField>(
    matrix: &[u32],
    rank: usize,
    width: usize,
    columns: usize,
    source: &[H::Source],
) -> Vec<i128> {
    let mut output = Vec::with_capacity(columns * rank * D);
    for column in 0..columns {
        for row in 0..rank {
            let mut convolution = [0i128; 2 * D - 1];
            for element in 0..width {
                let words = &source[(column * width + element) * K..][..K];
                let coefficients = &matrix[(row * width + element) * D..][..D];
                for (component, &word) in words.iter().enumerate() {
                    let word: u128 = word.into();
                    for scalar in 0..128 {
                        if (word >> scalar) & 1 == 0 {
                            continue;
                        }
                        let degree = scalar * K + component;
                        let sign = if scalar % 2 == 0 { 1 } else { -1 };
                        for (i, &coefficient) in coefficients.iter().enumerate() {
                            convolution[i + degree] += sign * i128::from(coefficient);
                        }
                    }
                }
            }
            for degree in (D..convolution.len()).rev() {
                let value = convolution[degree];
                convolution[degree - D / 2] += value;
                convolution[degree - D] -= value;
            }
            output.extend_from_slice(&convolution[..D]);
        }
    }
    output
}

fn residue(value: i128, prime: u32) -> u32 {
    value.rem_euclid(i128::from(prime)) as u32
}

fn reference<H: SwitchField>(
    matrix: &[u32],
    rank: usize,
    width: usize,
    columns: usize,
    source: &[H::Source],
) -> Vec<i128>
where
    H::Source: Sync,
{
    // Explicit p128 setups admit only power-of-two widths. Zero padding every
    // row and each source column embeds the same integer matrix product.
    let padded_width = width.next_power_of_two();
    let rings = (0..rank)
        .flat_map(|row| {
            (0..padded_width).map(move |element| {
                TrinomialRing::<Prime128OffsetA7F7, D, MinusTrinomial>::from_coefficients(
                    std::array::from_fn(|coefficient| {
                        if element < width {
                            Prime128OffsetA7F7::from_u64(u64::from(
                                matrix[(row * width + element) * D + coefficient],
                            ))
                        } else {
                            Prime128OffsetA7F7::from_u64(0)
                        }
                    }),
                )
                .unwrap()
            })
        })
        .collect();
    let setup = BinaryClearSetup::new(
        rings,
        rank,
        padded_width,
        columns,
        -1024,
        1024,
        128,
        BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 47).unwrap(),
        LabiniusCoefficientPrime::P128OffsetA7F7,
        LabiniusRingDegree::D648,
    )
    .unwrap();
    let mut padded_source = vec![H::Source::default(); columns * padded_width * K];
    for (input, output) in source
        .chunks_exact(width * K)
        .zip(padded_source.chunks_exact_mut(padded_width * K))
    {
        output[..input.len()].copy_from_slice(input);
    }
    let prepared = PreparedCommitMatrix::prepare(&setup).unwrap();
    let result = commit_binary_clear_prepared::<H, Prime128OffsetA7F7, D, MinusTrinomial>(
        &prepared,
        &setup,
        &padded_source,
    )
    .unwrap();
    let modulus = LabiniusCoefficientPrime::P128OffsetA7F7.modulus();
    result
        .images
        .iter()
        .flat_map(|image| image.coefficients())
        .map(|coefficient| {
            let value = coefficient.to_u128_checked().unwrap();
            if value > modulus / 2 {
                -i128::try_from(modulus - value).unwrap()
            } else {
                i128::try_from(value).unwrap()
            }
        })
        .collect()
}

fn check_case<H: SwitchField>(
    prime: u32,
    rank: usize,
    width: usize,
    columns: usize,
    matrix: &[u32],
    source: &[H::Source],
) where
    H::Source: Sync,
{
    let bound = 3u128 * width as u128 * D as u128 * u128::from(prime - 1);
    let modulus = LabiniusCoefficientPrime::P128OffsetA7F7.modulus();
    // Each reduced coefficient combines at most three unreduced coefficients,
    // each containing at most width * 648 signed terms of size at most q-1.
    assert!(bound < modulus / 2, "p128 centered lift must be exact");
    let expected_integer = integer_oracle::<H>(matrix, rank, width, columns, source);
    assert!(expected_integer
        .iter()
        .all(|coefficient| coefficient.unsigned_abs() <= bound));
    let prepared = PreparedLimbCommitMatrix::prepare(prime, D, rank, width, matrix).unwrap();
    let actual = commit_binary_clear_limb_prepared::<H>(&prepared, columns, source).unwrap();
    let expected: Vec<_> = expected_integer
        .iter()
        .map(|&coefficient| residue(coefficient, prime))
        .collect();
    assert_eq!(actual.len(), columns * rank * D);
    assert!(actual.iter().all(|&coefficient| coefficient < prime));
    assert_eq!(
        actual, expected,
        "prime={prime}, rank={rank}, width={width}"
    );
    let lifted = reference::<H>(matrix, rank, width, columns, source);
    assert_eq!(lifted, expected_integer, "exact centered p128 lift");
    let reduced_reference: Vec<_> = lifted
        .into_iter()
        .map(|coefficient| residue(coefficient, prime))
        .collect();
    assert_eq!(actual, reduced_reference);
    // The future root profile serializes the reduced coefficients. Equality is
    // also explicit at the byte boundary, independent of native endianness.
    let actual_bytes: Vec<_> = actual
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let reference_bytes: Vec<_> = reduced_reference
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(actual_bytes, reference_bytes);
}

#[test]
fn all_wide_primes_random_ranks_and_widths_match_integer_and_p128() {
    let mut rng = StdRng::seed_from_u64(0x011b_c064_8000);
    for prime in TrinomialWideLimbDomain::ADMITTED_PRIMES {
        for rank in 1..=4 {
            for width in [1, 2, 3] {
                let matrix = random_matrix(&mut rng, prime, rank, width);
                let source = random_source(&mut rng, 2 * width * K);
                check_case::<BinaryField128>(prime, rank, width, 2, &matrix, &source);
            }
        }
        // More than two 128-term checkpoints at every rank, with random
        // dense sources and an extremal matrix/source pair at the largest rank.
        let width = 257;
        for rank in 1..=4 {
            let matrix = random_matrix(&mut rng, prime, rank, width);
            let source = random_source(&mut rng, width * K);
            check_case::<BinaryField128>(prime, rank, width, 1, &matrix, &source);
        }
        check_case::<BinaryField128>(
            prime,
            4,
            width,
            1,
            &vec![prime - 1; 4 * width * D],
            &vec![u128::MAX; width * K],
        );
    }
}

#[test]
fn extreme_coefficients_and_edge_sources_match_integer_and_p128() {
    for prime in TrinomialWideLimbDomain::ADMITTED_PRIMES {
        for rank in 1..=4 {
            for width in [1, 2, 3] {
                let matrix = vec![prime - 1; rank * width * D];
                for word in [0, u128::MAX] {
                    check_case::<BinaryField128>(
                        prime,
                        rank,
                        width,
                        1,
                        &matrix,
                        &vec![word; width * K],
                    );
                }
            }
        }
        let mut rng = StdRng::seed_from_u64(0x000e_d648);
        let matrix = random_matrix(&mut rng, prime, 2, 3);
        // Positions straddle 324, alternate odd/even scalar signs, and include
        // the first and last live bits. Coefficients 512..648 are zero padding.
        for position in [0, 3, 4, 7, 319, 320, 323, 324, 327, 328, 507, 508, 511] {
            for element in [0, 2] {
                let mut source = vec![0u128; 3 * K];
                source[element * K + position % K] = 1u128 << (position / K);
                check_case::<BinaryField128>(prime, 2, 3, 1, &matrix, &source);
            }
        }
    }
}

#[test]
fn u64_source_host_matches_integer_and_p128() {
    let mut rng = StdRng::seed_from_u64(0x0064_0192);
    for prime in TrinomialWideLimbDomain::ADMITTED_PRIMES {
        let matrix = random_matrix(&mut rng, prime, 3, 3);
        let random: Vec<u64> = (0..2 * 3 * K).map(|_| rng.next_u64()).collect();
        for source in [random, vec![u64::MAX; 2 * 3 * K]] {
            check_case::<BinaryField192>(prime, 3, 3, 2, &matrix, &source);
        }
    }
}

fn check_bit_packing<H: SwitchField>(words: &[H::Source]) {
    let signed = pack_binary_element_i8::<H, D>(words).unwrap();
    let bits = pack_binary_element_bits::<H>(words).unwrap();
    for (degree, &coefficient) in signed.iter().enumerate() {
        let bit = ((bits[degree / 64] >> (degree % 64)) & 1) as i8;
        let expected = if (degree / K).is_multiple_of(2) {
            bit
        } else {
            -bit
        };
        assert_eq!(coefficient, expected, "degree={degree}");
    }
    assert!(bits[8..].iter().all(|&word| word == 0));
    for prime in TrinomialWideLimbDomain::ADMITTED_PRIMES {
        let domain = TrinomialWideLimbDomain::new(prime).unwrap();
        let mut actual = domain.zero_slots();
        let mut expected = domain.zero_slots();
        domain.forward_interleaved_bits(&bits, &mut actual).unwrap();
        domain
            .forward_centered(&signed.map(i32::from), &mut expected)
            .unwrap();
        let mut coefficients = [0; D];
        let mut expected_coefficients = [0; D];
        domain.inverse_centered(&actual, &mut coefficients).unwrap();
        domain
            .inverse_centered(&expected, &mut expected_coefficients)
            .unwrap();
        assert_eq!(coefficients, expected_coefficients);
        assert_eq!(coefficients, signed.map(i32::from));
    }
}

#[test]
fn bit_source_packing_matches_signed_packer_and_centered_transform() {
    let mut rng = StdRng::seed_from_u64(0x0b17_0648);
    check_bit_packing::<BinaryField128>(&random_source(&mut rng, K));
    check_bit_packing::<BinaryField128>(&[u128::MAX; K]);
    check_bit_packing::<BinaryField128>(&[0; K]);
    check_bit_packing::<BinaryField192>(&[u64::MAX; K]);
    check_bit_packing::<BinaryField192>(&std::array::from_fn::<_, K, _>(|_| rng.next_u64()));
    for component in 0..K {
        for scalar in [0, 1, 63, 64, 79, 80, 81, 82, 126, 127] {
            let mut words = [0u128; K];
            words[component] = 1u128 << scalar;
            check_bit_packing::<BinaryField128>(&words);
            if scalar < 64 {
                let mut words = [0u64; K];
                words[component] = 1u64 << scalar;
                check_bit_packing::<BinaryField192>(&words);
            }
        }
    }
    for words in [3, 5] {
        assert_eq!(
            pack_binary_element_bits::<BinaryField128>(&vec![0; words]).unwrap_err(),
            pack_binary_element_i8::<BinaryField128, D>(&vec![0; words]).unwrap_err(),
        );
    }
}

#[test]
fn prepared_matrix_has_tagged_slots_and_canonical_transform() {
    let real_width_bound = 3u128 * 4096 * D as u128 * u128::from(Q - 1);
    assert!(real_width_bound < LabiniusCoefficientPrime::P128OffsetA7F7.modulus() / 2);
    let mut rng = StdRng::seed_from_u64(0x0000_b648);
    let matrix = random_matrix(&mut rng, Q, 3, 2);
    let prepared = PreparedLimbCommitMatrix::prepare(Q, D, 3, 2, &matrix).unwrap();
    assert_eq!(
        prepared.matrix_bytes(),
        3 * 2 * size_of::<TrinomialWideLimbSlots>()
    );
    assert_eq!(
        prepared.prepared_bytes(),
        prepared.matrix_bytes() + prepared.domain().table_storage_bytes()
    );
    for (coefficients, expected) in matrix.chunks_exact(D).zip(prepared.matrix_ntt()) {
        let centered: Vec<_> = coefficients
            .iter()
            .map(|&coefficient| {
                if coefficient > Q / 2 {
                    (i64::from(coefficient) - i64::from(Q)) as i32
                } else {
                    coefficient as i32
                }
            })
            .collect();
        let mut actual = prepared.domain().zero_slots();
        prepared
            .domain()
            .forward_centered(&centered, &mut actual)
            .unwrap();
        assert_eq!(&actual, expected);
    }
}

#[test]
fn malformed_matrix_and_source_inputs_return_specific_errors() {
    let coefficients = vec![0; D];
    let error = |prime, degree, rank, width, coefficients: &[u32]| {
        PreparedLimbCommitMatrix::prepare(prime, degree, rank, width, coefficients)
            .err()
            .unwrap()
    };
    assert_eq!(
        error(Q, 324, 1, 1, &coefficients),
        AkitaError::InvalidSetup("limb commitment degree must be 648".into())
    );
    for (rank, width) in [(0, 1), (1, 0)] {
        assert_eq!(
            error(Q, D, rank, width, &[]),
            AkitaError::InvalidSetup("limb commitment rank and width must be nonzero".into())
        );
    }
    let mut invalid = coefficients.clone();
    invalid[D - 1] = Q;
    assert_eq!(
        error(Q, D, 1, 1, &invalid),
        AkitaError::InvalidInput("matrix coefficient must be below the limb prime".into())
    );
    for actual in [0, D - 1, D + 1] {
        assert_eq!(
            error(Q, D, 1, 1, &vec![0; actual]),
            AkitaError::InvalidSize {
                expected: D,
                actual
            }
        );
    }
    let unadmitted = 268_435_399;
    let expected = TrinomialWideLimbDomain::new(unadmitted).unwrap_err();
    assert_eq!(
        error(unadmitted, D, 1, 1, &coefficients),
        AkitaError::InvalidSetup(expected.to_string())
    );
    assert_eq!(
        error(Q, D, usize::MAX, 2, &[]),
        AkitaError::InvalidSetup("limb matrix size overflow".into()),
    );
    let prepared = PreparedLimbCommitMatrix::prepare(Q, D, 1, 1, &coefficients).unwrap();
    for actual in [0, 7, 9] {
        assert_eq!(
            commit_binary_clear_limb_prepared::<BinaryField128>(&prepared, 2, &vec![0; actual])
                .unwrap_err(),
            AkitaError::InvalidSize {
                expected: 8,
                actual
            }
        );
    }
    assert_eq!(
        commit_binary_clear_limb_prepared::<BinaryField128>(&prepared, usize::MAX, &[])
            .unwrap_err(),
        AkitaError::InvalidInput("limb source size overflow".into()),
    );
    assert!(
        commit_binary_clear_limb_prepared::<BinaryField128>(&prepared, 0, &[])
            .unwrap()
            .is_empty()
    );
}

#[cfg(feature = "parallel")]
#[test]
fn one_and_three_thread_pools_have_identical_column_order_and_values() {
    let mut rng = StdRng::seed_from_u64(0x0000_0031);
    let matrix = random_matrix(&mut rng, Q, 4, 3);
    let source = random_source(&mut rng, 8 * 3 * K);
    let prepared = PreparedLimbCommitMatrix::prepare(Q, D, 4, 3, &matrix).unwrap();
    let results: Vec<_> = [1, 3]
        .map(|threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    commit_binary_clear_limb_prepared::<BinaryField128>(&prepared, 8, &source)
                        .unwrap()
                })
        })
        .into_iter()
        .collect();
    assert_eq!(results[0], results[1]);
    let expected: Vec<_> = integer_oracle::<BinaryField128>(&matrix, 4, 3, 8, &source)
        .into_iter()
        .map(|value| residue(value, Q))
        .collect();
    assert_eq!(results[0], expected);
}

#[cfg(feature = "parallel")]
#[test]
#[ignore = "full 679477248-bit table thread-scaling measurement; run in release mode"]
fn full_table_thread_scaling() {
    use std::{process::Command, time::Instant};

    if cfg!(debug_assertions) {
        panic!("measurement requires --release");
    }
    const RANK: usize = 3;
    const WIDTH: usize = 4096;
    const COLUMNS: usize = 256;
    const BITS: usize = COLUMNS * WIDTH * D;
    let load = |when| {
        if let Ok(output) = Command::new("uptime").output() {
            println!(
                "load {when}: {}",
                String::from_utf8_lossy(&output.stdout).trim()
            );
        }
    };
    load("before");
    let mut rng = StdRng::seed_from_u64(0x0256_4096_0648);
    let matrix = random_matrix(&mut rng, Q, RANK, WIDTH);
    let source = random_source(&mut rng, COLUMNS * WIDTH * K);
    let start = Instant::now();
    let prepared = PreparedLimbCommitMatrix::prepare(Q, D, RANK, WIDTH, &matrix).unwrap();
    println!(
        "prepared: build={:.6}s, lane_bytes={}, tagged_slot_bytes={}, prepared_payload_bytes={}, workspace_bytes_per_thread={}, committed_bits={BITS}, live_source_bits={}",
        start.elapsed().as_secs_f64(),
        RANK * WIDTH * D * size_of::<u32>(),
        prepared.matrix_bytes(),
        prepared.prepared_bytes(),
        prepared.workspace_bytes(),
        source.len() * 128,
    );
    drop(matrix);
    let mut one_thread_min = 0.0;
    let mut canonical = None;
    println!("threads repetition wall_seconds");
    for threads in [1, 2, 4, 8, 12, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut minimum = f64::INFINITY;
        for repetition in 1..=3 {
            let start = Instant::now();
            let actual = pool.install(|| {
                commit_binary_clear_limb_prepared::<BinaryField128>(&prepared, COLUMNS, &source)
                    .unwrap()
            });
            let seconds = start.elapsed().as_secs_f64();
            println!("run {threads} {repetition} {seconds:.6}");
            minimum = minimum.min(seconds);
            if let Some(ref expected) = canonical {
                assert_eq!(&actual, expected, "thread count {threads}");
            } else {
                canonical = Some(actual);
            }
        }
        if threads == 1 {
            one_thread_min = minimum;
        }
        println!(
            "minimum threads={threads} seconds={minimum:.6} ns_per_bit={:.6} speedup={:.3}",
            minimum * 1e9 / BITS as f64,
            one_thread_min / minimum,
        );
    }
    load("after");
}
