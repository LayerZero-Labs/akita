#![cfg(feature = "labinius")]

use akita_algebra::binary::{
    field_switch::{embed_source, SwitchField},
    BinaryField128, BinaryField162, BinaryField192,
};
use akita_labinius_verifier::{endpoint::left_expansion, source::equality_weights};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn compare<H: SwitchField>() {
    let mut rng = StdRng::seed_from_u64(0x5e21a1);
    for (row_vars, col_vars) in [(0, 0), (1, 0), (0, 1), (3, 3), (8, 2), (2, 8)] {
        let rows = 1 << row_vars;
        let columns = 1 << col_vars;
        let source: Vec<H::Source> = (0..rows * columns)
            .map(|_| {
                let bits = if H::ROWS == 128 {
                    rng.gen::<u128>()
                } else {
                    u128::from(rng.gen::<u64>())
                };
                H::Source::try_from(bits).ok().unwrap()
            })
            .collect();
        let point: Vec<_> = (0..row_vars + col_vars)
            .map(|_| {
                BinaryField162::from_words([rng.gen(), rng.gen(), rng.gen::<u64>() >> 30]).unwrap()
            })
            .collect();
        let weights = equality_weights(&point[..row_vars]).unwrap();
        let expected: Vec<_> = source
            .chunks_exact(rows)
            .map(|words| {
                weights
                    .iter()
                    .zip(words)
                    .fold(BinaryField162::ZERO, |value, (&weight, &word)| {
                        value + weight * embed_source::<H>(word)
                    })
            })
            .collect();
        let check = || {
            assert_eq!(
                left_expansion::<H>(&source, &point, rows, columns).unwrap(),
                expected
            );
            assert!(
                left_expansion::<H>(&source[..source.len() - 1], &point, rows, columns).is_err()
            );
            let mut wrong_point = point.clone();
            wrong_point.push(BinaryField162::ONE);
            assert!(left_expansion::<H>(&source, &wrong_point, rows, columns).is_err());
        };
        check();
        #[cfg(feature = "parallel")]
        for threads in [1, 3] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(check);
        }
    }
}

#[test]
fn left_expansion_matches_serial_definition_for_both_hosts() {
    compare::<BinaryField128>();
    compare::<BinaryField192>();
}
