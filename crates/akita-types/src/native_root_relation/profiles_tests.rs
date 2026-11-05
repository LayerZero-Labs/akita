use crate::binary_source::{root_weight_fixtures, RootFixtureField as F};
use akita_algebra::poly::fold_evals_in_place;
use jolt_algebra::Ring;

/// Every block width, including widths that send the 256-stride lanes
/// through the per-address fallback, folds like the materialized table.
#[test]
fn profiles_fold_like_the_materialized_table() {
    for fixture in root_weight_fixtures() {
        let dense = fixture.weights.materialize().unwrap();
        let num_vars = dense.len().trailing_zeros() as usize;
        for block_bits in [0, 3, 8, 10].into_iter().filter(|&bits| bits <= num_vars) {
            let mut profiles = fixture.weights.profiles(block_bits).unwrap();
            assert_eq!(profiles.len(), dense.len());
            assert_eq!(profiles.clone().take_dense(), dense, "{}", fixture.name);
            let mut folded = dense.clone();
            for round in 0..block_bits {
                for left in (0..folded.len()).step_by(2 * 37 + 2) {
                    assert_eq!(
                        profiles.pair(left),
                        (folded[left], folded[left + 1]),
                        "{} bits {block_bits} round {round}",
                        fixture.name
                    );
                }
                let challenge = F::from_u64(1_000_003 + 7 * round as u64);
                profiles.bind(challenge);
                fold_evals_in_place(&mut folded, challenge);
                assert_eq!(profiles.len(), folded.len());
            }
            assert_eq!(
                profiles.take_dense(),
                folded,
                "{} bits {block_bits}",
                fixture.name
            );
            assert!(profiles.is_empty());
        }
    }
}

#[test]
fn profiles_reject_blocks_wider_than_the_table() {
    let fixture = root_weight_fixtures().swap_remove(0);
    let num_vars = fixture.weights.table_len().trailing_zeros() as usize;
    assert!(fixture.weights.profiles(num_vars + 1).is_err());
}
