#![cfg(feature = "labinius")]

mod common;

use akita_algebra::{
    binary::{field_switch::SwitchField, BinaryField128, BinaryField192},
    MinusTrinomial, TrinomialLimbDomain,
};
use akita_error::AkitaError;
use akita_labinius_prover::{
    commit_binary_clear, commit_binary_clear_limb_prepared, commit_binary_clear_prepared,
    PreparedLimbCommitMatrix,
};
use common::{clear_setup, clear_setup_with, TestHost, Q};
use rand::{rngs::StdRng, RngCore, SeedableRng};

const D: usize = 648;
const K: usize = 4;

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
        for (rank, width) in [(1, 1), (2, 3), (4, 2)] {
            check::<BinaryField128>(prime, rank, width, 2, false);
        }
        check::<BinaryField192>(prime, 2, 2, 2, false);
        // More than two 128-term accumulator checkpoints, random and extremal.
        check::<BinaryField128>(prime, 1, 257, 1, false);
        check::<BinaryField128>(prime, 1, 257, 1, true);
    }
}

#[test]
fn setup_bound_and_reference_commitments_match_the_integer_definition() {
    let setup = clear_setup::<D, MinusTrinomial>(2, 2, 4);
    let (source, _, _) = common::data::<BinaryField128>(setup.source_len(), setup.num_vars());
    let expected = integer_definition::<BinaryField128>(Q, setup.matrix(), 2, 2, 4, &source);
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
    let raw = PreparedLimbCommitMatrix::prepare(Q, D, 2, 2, setup.matrix()).unwrap();
    let mut other_matrix = setup.matrix().to_vec();
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
