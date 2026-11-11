//! Main's LS18 field-switch regression, exercised against preview's shared sampler.
use jolt_algebra::{
    pseudo_mersenne_modulus, Prime128Offset6A23801F, Prime32Offset99, Prime64Offset59,
    PseudoMersenne,
};
use jolt_crypto::challenge::{SparseChallengeConfig, PRODUCTION_FOLD_CHALLENGE_RING_DIMS};

fn assert_ls18_short_difference_bound<F: PseudoMersenne>(split_count: u32) {
    let modulus = pseudo_mersenne_modulus(F::MODULUS_BITS, F::OFFSET).expect("production modulus");
    let split_count = u128::from(split_count);
    for &carrier_dimension in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
        assert!(carrier_dimension.is_power_of_two());
        assert!(carrier_dimension >= split_count as usize);
    }
    assert_eq!(
        modulus % (4 * split_count),
        2 * split_count + 1,
        "production prime must meet the LS18 partial splitting congruence"
    );

    let max_challenge_coefficient = PRODUCTION_FOLD_CHALLENGE_RING_DIMS
        .iter()
        .map(|&ring_dim| {
            SparseChallengeConfig::production_for_ring_dim(ring_dim)
                .expect("production challenge")
                .infinity_norm()
        })
        .max()
        .expect("production challenge ladder");
    assert!(max_challenge_coefficient <= 2);

    let max_difference_coefficient = 2 * u128::from(max_challenge_coefficient);
    let split_count_exponent = u32::try_from(split_count).expect("small production split count");
    // For even `ell`, raising
    // `2 c_max < q^(1/ell) / sqrt(ell)` to `ell` gives this integer check.
    let exact_shortness_lhs = max_difference_coefficient.pow(split_count_exponent)
        * split_count.pow(split_count_exponent / 2);
    assert!(
        exact_shortness_lhs < modulus,
        "raising the strict LS18 shortness bound to split_count must remain below q"
    );
}

#[test]
fn production_primes_and_challenges_meet_ls18_shortness() {
    assert_ls18_short_difference_bound::<Prime32Offset99>(2);
    assert_ls18_short_difference_bound::<Prime64Offset59>(2);
    assert_ls18_short_difference_bound::<Prime128Offset6A23801F>(16);
}
