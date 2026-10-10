//! LS18 partial-splitting check for `2^128 - 275`, the modulus of
//! `SisModulusProfileId::Q128Offset275`.
//!
//! The unit tests of `SparseChallengeConfig` check the same two conditions
//! for the primes the presets use. No preset uses this prime.

#![allow(missing_docs)]

use akita_challenges::{SparseChallengeConfig, PRODUCTION_FOLD_CHALLENGE_RING_DIMS};
use jolt_field::{pseudo_mersenne_modulus, Prime128Offset275, PseudoMersenne};

/// Number of irreducible factors of `X^d + 1` modulo the prime.
const SPLIT_COUNT: u128 = 2;

fn modulus<F: PseudoMersenne>() -> u128 {
    pseudo_mersenne_modulus(F::MODULUS_BITS, F::OFFSET).expect("pseudo-Mersenne modulus")
}

#[test]
fn prime128_offset275_meets_ls18_shortness_at_two_splits() {
    let modulus = modulus::<Prime128Offset275>();
    assert_eq!(modulus, u128::MAX - 274);

    // `q = 2 ell + 1 (mod 4 ell)`: here `q = 5 (mod 8)`.
    assert_eq!(modulus % (4 * SPLIT_COUNT), 2 * SPLIT_COUNT + 1);
    for &ring_dimension in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
        assert!(ring_dimension.is_power_of_two());
        assert!(ring_dimension >= SPLIT_COUNT as usize);
    }

    let max_challenge_coefficient = PRODUCTION_FOLD_CHALLENGE_RING_DIMS
        .iter()
        .map(|&ring_dimension| {
            SparseChallengeConfig::production_for_ring_dim(ring_dimension)
                .expect("production challenge")
                .infinity_norm()
        })
        .max()
        .expect("production challenge ladder");
    assert!(max_challenge_coefficient <= 2);

    // A challenge difference has coefficients of magnitude at most `2 c_max`.
    // For even `ell`, raising `2 c_max < q^(1/ell) / sqrt(ell)` to `ell`
    // gives this integer check.
    let split_count = u32::try_from(SPLIT_COUNT).expect("small split count");
    let shortness_left_hand_side = (2 * u128::from(max_challenge_coefficient)).pow(split_count)
        * SPLIT_COUNT.pow(split_count / 2);
    assert!(shortness_left_hand_side < modulus);
}
