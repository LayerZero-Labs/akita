//! The packed q24 commitment path against the unpacked accumulator path.
//!
//! The reference transforms each source element with the domain and adds
//! products of the prepared (unpacked) matrix with `TrinomialLimbAccumulator`,
//! exactly as the commitment kernel does without the feature.

#![cfg(feature = "q24-packed-commit")]

use akita_algebra::{
    binary::{field_switch::SwitchField, BinaryField128, BinaryField192},
    ring::trinomial::{image::Q24, Q24_REDUCTION_TERMS},
    TrinomialLimbAccumulator, TrinomialLimbDomain,
};
use akita_labinius_prover::{
    commit_binary_clear_limb_prepared, limb_commit_kernel::pack_binary_element_bits,
    PreparedLimbCommitMatrix,
};
use rand::{rngs::StdRng, RngCore, SeedableRng};

const D: usize = 648;
const K: usize = 4;
const H: u32 = (Q24 - 1) / 2;

/// The unpacked path: one `TrinomialLimbAccumulator` per row and column.
fn unpacked<S: SwitchField>(
    prepared: &PreparedLimbCommitMatrix,
    width: usize,
    source: &[S::Source],
) -> Vec<u32> {
    let domain = prepared.domain();
    let prime = domain.prime() as i32;
    let mut slots = domain.zero_slots();
    let mut coefficients = [0i32; D];
    let mut output = Vec::new();
    for column in source.chunks_exact(width * K) {
        let transformed: Vec<_> = column
            .chunks_exact(K)
            .map(|words| {
                let bits = pack_binary_element_bits::<S>(words).unwrap();
                let mut out = domain.zero_slots();
                domain.forward_interleaved_bits(&bits, &mut out).unwrap();
                out
            })
            .collect();
        for row in prepared.matrix_ntt().chunks_exact(width) {
            let mut accumulator = TrinomialLimbAccumulator::new(domain);
            for (matrix, source) in row.iter().zip(&transformed) {
                accumulator.add_product(matrix, source).unwrap();
            }
            accumulator.finish(&mut slots).unwrap();
            domain.inverse_centered(&slots, &mut coefficients).unwrap();
            output.extend(coefficients.iter().map(|&v| v.rem_euclid(prime) as u32));
        }
    }
    output
}

#[derive(Clone, Copy, Debug)]
enum Matrix {
    Random,
    Constant(u32),
}

#[derive(Clone, Copy, Debug)]
enum Source {
    Random,
    Zero,
    Ones,
}

fn check(rank: usize, width: usize, columns: usize, matrix: Matrix, source: Source) {
    let mut rng = StdRng::seed_from_u64(0x0024_c0de ^ (rank * 977 + width * 31 + columns) as u64);
    let coefficients: Vec<u32> = (0..rank * width * D)
        .map(|_| match matrix {
            Matrix::Random => (rng.next_u64() % u64::from(Q24)) as u32,
            Matrix::Constant(value) => value,
        })
        .collect();
    let words: Vec<u128> = (0..columns * width * K)
        .map(|_| match source {
            Source::Random => u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64),
            Source::Zero => 0,
            Source::Ones => u128::MAX,
        })
        .collect();
    let prepared = PreparedLimbCommitMatrix::prepare(Q24, D, rank, width, &coefficients).unwrap();
    assert!(prepared.uses_q24_packed_slots());
    let expected = unpacked::<BinaryField128>(&prepared, width, &words);
    let commit =
        || commit_binary_clear_limb_prepared::<BinaryField128>(&prepared, columns, &words).unwrap();
    let case = format!("rank={rank} width={width} columns={columns} {matrix:?} {source:?}");
    assert_eq!(commit(), expected, "{case}");
    #[cfg(feature = "parallel")]
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(pool.install(commit), expected, "{case} threads={threads}");
    }
}

#[test]
fn q24_packed_path_is_selected_only_for_the_q24_prime() {
    for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
        let prepared = PreparedLimbCommitMatrix::prepare(prime, D, 1, 1, &[0; D]).unwrap();
        assert_eq!(prepared.uses_q24_packed_slots(), prime == Q24, "{prime}");
    }
}

/// Widths around the eight-element batch, ragged column blocks and two-row
/// pairs, on random data and on every extreme matrix value.
#[test]
fn q24_packed_commitments_match_the_unpacked_path() {
    let shapes = [
        (1, 1, 1),
        (2, 7, 3),
        (3, 8, 5),
        (2, 9, 9),
        (4, 16, 4),
        (1, 17, 2),
        (3, 33, 7),
    ];
    for (rank, width, columns) in shapes {
        check(rank, width, columns, Matrix::Random, Source::Random);
    }
    // Zero, one, the centered extremes (q-1)/2 and (q+1)/2, and q - 1.
    for value in [0, 1, H, H + 1, Q24 - 1] {
        for source in [Source::Random, Source::Zero, Source::Ones] {
            check(2, 9, 3, Matrix::Constant(value), source);
        }
    }
}

#[test]
fn q24_packed_commitments_match_for_binary_field192_sources() {
    let (rank, width, columns) = (2, 5, 3);
    let mut rng = StdRng::seed_from_u64(0x0192_0024);
    let coefficients: Vec<u32> = (0..rank * width * D)
        .map(|_| (rng.next_u64() % u64::from(Q24)) as u32)
        .collect();
    // BinaryField192 sources are 64-bit words.
    let words: Vec<u64> = (0..columns * width * K).map(|_| rng.next_u64()).collect();
    let prepared = PreparedLimbCommitMatrix::prepare(Q24, D, rank, width, &coefficients).unwrap();
    assert_eq!(
        commit_binary_clear_limb_prepared::<BinaryField192>(&prepared, columns, &words).unwrap(),
        unpacked::<BinaryField192>(&prepared, width, &words)
    );
}

/// Exactly `2^15` products per accumulator, then one more, so the second case
/// crosses the q24 reduction with the largest matrix residue and every
/// source bit set.
#[test]
fn q24_packed_commitments_cross_the_reduction_interval() {
    let terms = Q24_REDUCTION_TERMS as usize;
    check(1, terms, 1, Matrix::Random, Source::Random);
    check(1, terms + 1, 1, Matrix::Constant(Q24 - 1), Source::Ones);
}
