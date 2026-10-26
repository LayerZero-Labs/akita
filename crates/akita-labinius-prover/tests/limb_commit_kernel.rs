#![cfg(feature = "labinius")]

mod common;

use akita_algebra::{
    binary::{field_switch::SwitchField, BinaryField128, BinaryField192},
    MinusTrinomial, TrinomialLimbDomain,
};
use akita_error::AkitaError;
use akita_labinius_prover::{
    commit_binary_clear, commit_binary_clear_limb_prepared, commit_binary_clear_prepared,
    limb_commit_kernel::pack_binary_element_bits, PreparedLimbCommitMatrix,
};
use common::{clear_setup, clear_setup_with, TestHost, Q};
use rand::{rngs::StdRng, RngCore, SeedableRng};

const D: usize = 648;
const K: usize = 4;

#[test]
fn source_packer_preserves_every_bit_and_rejects_wrong_lengths() {
    for component in 0..K {
        for bit in 0..128 {
            let mut words = [0u128; K];
            words[component] = 1 << bit;
            let mut expected = [0u64; 11];
            let position = K * bit + component;
            expected[position / 64] = 1 << (position % 64);
            assert_eq!(
                pack_binary_element_bits::<BinaryField128>(&words).unwrap(),
                expected,
                "component={component}, bit={bit}"
            );
        }
    }
    for length in [0, 3, 5] {
        assert!(matches!(
            pack_binary_element_bits::<BinaryField128>(&vec![0; length]),
            Err(AkitaError::InvalidInput(_))
        ));
    }
}

/// Decode the defining signed interleaving directly, independently of the
/// kernel's source packer and transform tables. Reduce the integer convolution
/// in descending degree using Y^648 = Y^324 - 1, then reduce modulo the prime.
fn integer_definition<H: SwitchField>(
    prime: u32,
    matrix: &[u32],
    rank: usize,
    width: usize,
    columns: usize,
    source: &[H::Source],
) -> Vec<u32> {
    let mut output = Vec::with_capacity(columns * rank * D);
    for column in 0..columns {
        for row in 0..rank {
            let mut convolution = [0i128; 2 * D - 1];
            for element in 0..width {
                let words = &source[(column * width + element) * K..][..K];
                let coefficients = &matrix[(row * width + element) * D..][..D];
                for (component, &word) in words.iter().enumerate() {
                    let word: u128 = word.into();
                    for scalar in (0..128).filter(|scalar| (word >> scalar) & 1 == 1) {
                        let sign = if scalar % 2 == 0 { 1 } else { -1 };
                        for (i, &coefficient) in coefficients.iter().enumerate() {
                            convolution[i + scalar * K + component] +=
                                sign * i128::from(coefficient);
                        }
                    }
                }
            }
            for degree in (D..convolution.len()).rev() {
                let value = convolution[degree];
                convolution[degree - D / 2] += value;
                convolution[degree - D] -= value;
            }
            output.extend(
                convolution[..D]
                    .iter()
                    .map(|value| value.rem_euclid(i128::from(prime)) as u32),
            );
        }
    }
    output
}

fn check<H: TestHost>(prime: u32, rank: usize, width: usize, columns: usize, extremal: bool)
where
    H::Source: core::ops::Not<Output = H::Source>,
{
    let mut rng = StdRng::seed_from_u64(0x011b_c064_8000 ^ (rank * 31 + width) as u64);
    let matrix: Vec<u32> = (0..rank * width * D)
        .map(|_| {
            if extremal {
                prime - 1
            } else {
                (rng.next_u64() % u64::from(prime)) as u32
            }
        })
        .collect();
    let mut source: Vec<H::Source> = (0..columns * width * K)
        .map(|_| H::random_source(&mut rng))
        .collect();
    if extremal {
        // Every admitted source bit set: the largest accumulator inputs.
        for word in &mut source {
            *word = !H::Source::default();
        }
    }
    let prepared = PreparedLimbCommitMatrix::prepare(prime, D, rank, width, &matrix).unwrap();
    assert_eq!(
        commit_binary_clear_limb_prepared::<H>(&prepared, columns, &source).unwrap(),
        integer_definition::<H>(prime, &matrix, rank, width, columns, &source),
        "prime={prime}, rank={rank}, width={width}, extremal={extremal}"
    );
}

#[test]
fn every_limb_prime_matches_the_integer_definition() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        for (rank, width) in [(1, 1), (2, 3), (3, 3), (4, 2)] {
            check::<BinaryField128>(prime, rank, width, 2, false);
        }
        check::<BinaryField192>(prime, 2, 2, 2, false);
        // More than two 128-term accumulator checkpoints, random and extremal.
        check::<BinaryField128>(prime, 1, 257, 1, false);
        check::<BinaryField128>(prime, 1, 257, 1, true);
    }
}

#[test]
fn rows_wider_than_one_matrix_page_match_the_integer_definition() {
    // Rank 3 pairs two rows and runs the third alone; width 520 puts the last
    // element batch of every row on its second page.
    check::<BinaryField128>(Q, 3, 520, 1, false);
}

#[test]
fn binary_field192_ragged_columns_match_the_integer_definition() {
    let run = || check::<BinaryField192>(Q, 1, 3, 9, false);
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(run);
    #[cfg(not(feature = "parallel"))]
    run();
}

#[test]
fn ragged_column_blocks_match_unblocked_and_integer_definitions() {
    const WIDTH: usize = 129;
    const COLUMNS: usize = 9;
    // Rank three exercises a two-row tile followed by the odd-row fallback.
    for rank in [2, 3] {
        let mut rng = StdRng::seed_from_u64(0x09c0_1129);
        let matrix: Vec<u32> = (0..rank * WIDTH * D)
            .map(|_| (rng.next_u64() % u64::from(Q)) as u32)
            .collect();
        // Sparse, independently positioned bits keep the integer convolution cheap
        // while exercising every component, both signs, and the last source bit.
        let source: Vec<u128> = (0..COLUMNS * WIDTH * K)
            .map(|index| 1 | (1 << (1 + index % 126)) | (1 << 127))
            .collect();
        let prepared = PreparedLimbCommitMatrix::prepare(Q, D, rank, WIDTH, &matrix).unwrap();
        let expected =
            integer_definition::<BinaryField128>(Q, &matrix, rank, WIDTH, COLUMNS, &source);
        // A single column cannot share its matrix stream with another column and
        // provides the unblocked reference across the 128-term checkpoint.
        let unblocked: Vec<u32> = source
            .chunks_exact(WIDTH * K)
            .flat_map(|column| {
                commit_binary_clear_limb_prepared::<BinaryField128>(&prepared, 1, column).unwrap()
            })
            .collect();
        assert_eq!(unblocked, expected);
        let commit = || {
            commit_binary_clear_limb_prepared::<BinaryField128>(&prepared, COLUMNS, &source)
                .unwrap()
        };
        assert_eq!(commit(), expected);
        #[cfg(feature = "parallel")]
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert_eq!(pool.install(commit), expected, "threads={threads}");
        }
    }
}

#[test]
fn setup_bound_and_reference_commitments_match_the_integer_definition() {
    let setup = clear_setup::<D, MinusTrinomial>(2, 2, 4);
    let (source, _, _) = common::data::<BinaryField128>(setup.source_len(), setup.num_vars());
    let expected = integer_definition::<BinaryField128>(
        Q,
        &setup.matrix_coefficients().copied().collect::<Vec<_>>(),
        2,
        2,
        4,
        &source,
    );
    let prepared = PreparedLimbCommitMatrix::prepare_for_setup(&setup).unwrap();
    let commit = |prepared, setup, source: &[u128]| {
        commit_binary_clear_prepared::<BinaryField128>(prepared, setup, source)
    };
    assert_eq!(commit(&prepared, &setup, &source).unwrap().images, expected);
    assert_eq!(
        commit_binary_clear::<BinaryField128, D, MinusTrinomial>(&setup, &source)
            .unwrap()
            .images,
        expected
    );
    // A raw cache, and the cache of another matrix, cannot serve this setup.
    let raw = PreparedLimbCommitMatrix::prepare(
        Q,
        D,
        2,
        2,
        &setup.matrix_coefficients().copied().collect::<Vec<_>>(),
    )
    .unwrap();
    let mut other_matrix = setup.matrix_coefficients().copied().collect::<Vec<_>>();
    other_matrix[0] ^= 1;
    let other = clear_setup_with::<D, MinusTrinomial>(other_matrix, 2, 2, 4).unwrap();
    for foreign in [&raw, &prepared] {
        let target = if std::ptr::eq(foreign, &raw) {
            &setup
        } else {
            &other
        };
        assert!(matches!(
            commit(foreign, target, &source),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
    assert_eq!(
        commit(&prepared, &setup, &source[1..]).unwrap_err(),
        AkitaError::InvalidSize {
            expected: source.len(),
            actual: source.len() - 1
        }
    );
}

#[test]
fn malformed_matrices_and_sources_are_rejected() {
    let coefficients = vec![0; D];
    let error = |prime, degree, rank, width, coefficients: &[u32]| {
        PreparedLimbCommitMatrix::prepare(prime, degree, rank, width, coefficients).unwrap_err()
    };
    assert!(matches!(
        error(Q, 324, 1, 1, &coefficients),
        AkitaError::InvalidSetup(_)
    ));
    assert!(matches!(
        error(Q, D, 0, 1, &[]),
        AkitaError::InvalidSetup(_)
    ));
    assert!(matches!(
        error(268_435_399, D, 1, 1, &coefficients),
        AkitaError::InvalidSetup(_)
    ));
    assert!(matches!(
        error(Q, D, usize::MAX, 2, &[]),
        AkitaError::InvalidSetup(_)
    ));
    assert_eq!(
        error(Q, D, 1, 1, &coefficients[1..]),
        AkitaError::InvalidSize {
            expected: D,
            actual: D - 1
        }
    );
    let mut unreduced = coefficients.clone();
    unreduced[D - 1] = Q;
    assert!(matches!(
        error(Q, D, 1, 1, &unreduced),
        AkitaError::InvalidInput(_)
    ));
    let prepared = PreparedLimbCommitMatrix::prepare(Q, D, 1, 1, &coefficients).unwrap();
    assert!(
        commit_binary_clear_limb_prepared::<BinaryField128>(&prepared, 0, &[])
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        commit_binary_clear_limb_prepared::<BinaryField128>(&prepared, 0, &[0]).unwrap_err(),
        AkitaError::InvalidSize {
            expected: 0,
            actual: 1
        }
    );
    assert_eq!(
        commit_binary_clear_limb_prepared::<BinaryField128>(&prepared, 2, &[0; 7]).unwrap_err(),
        AkitaError::InvalidSize {
            expected: 8,
            actual: 7
        }
    );
    assert!(
        commit_binary_clear_limb_prepared::<BinaryField128>(&prepared, usize::MAX, &[]).is_err()
    );
}

#[test]
fn prepared_pages_preserve_first_last_and_boundary_elements() {
    let width = 513;
    let rank = 2;
    let matrix: Vec<u32> = (0..rank * width * D)
        .map(|index| index as u32 % Q)
        .collect();
    let prepared = PreparedLimbCommitMatrix::prepare(Q, D, rank, width, &matrix).unwrap();
    assert_eq!(prepared.matrix_page_count(), 4);
    assert_eq!(prepared.matrix_maximum_page_bytes(), 512 * (D * 4 + 4));
    assert_eq!(prepared.matrix_bytes(), rank * width * (D * 4 + 4));
    for row in 0..rank {
        for element in [0, 511, 512] {
            let mut coefficients = [0i32; D];
            prepared
                .domain()
                .inverse_centered(
                    prepared.matrix_ntt_element(row, element).unwrap(),
                    &mut coefficients,
                )
                .unwrap();
            let start = (row * width + element) * D;
            for (&actual, &expected) in coefficients.iter().zip(&matrix[start..start + D]) {
                assert_eq!(
                    i64::from(actual).rem_euclid(i64::from(Q)),
                    i64::from(expected)
                );
            }
        }
    }
    assert!(prepared.matrix_ntt_element(0, width).is_none());
    assert!(prepared.matrix_ntt_element(rank, 0).is_none());
}
