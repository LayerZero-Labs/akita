//! `SisModulusProfileId::Q128Offset275` owns no generated SIS rows:
//! `akita-params` prices it from the rows generated at the `Q128OffsetA7F7`
//! modulus. That is sound exactly while the estimator cannot tell the two
//! moduli apart, and these tests pin the reason rather than a sample of
//! outputs.
//!
//! The estimator reads a modulus `q` in three ways:
//!
//! - its bit length (`half_q_minus_one`, `sis_trivially_easy`);
//! - `log2_biguint(q)`, a function of the bit length and the leading 64 bits;
//! - exact comparisons of a squared length with `q^2`: the small-box fallback
//!   in `lattice::infinity_uses_small_box`, taken only within `2^(1e-8)` of
//!   equality, and the Euclidean triviality check.
//!
//! The two moduli share the first two. The third cannot separate them on any
//! input the Akita constructors produce, because those squared lengths stay
//! below `2^192` while both `q^2` exceed `2^255`.
//!
//! If one of these tests fails, generate rows for `Q128Offset275` instead of
//! weakening it.

use akita_params::sis::SisModulusProfileId;
use akita_sis_estimator::{
    estimate,
    math::{half_q_minus_one, log2_biguint},
    scalar_sis_from_ring_euclidean, scalar_sis_from_ring_wide, AkitaModulusProfileId, Bound,
    EstimateConfig, SisParameters,
};
use num_bigint::BigUint;

const OWNER: AkitaModulusProfileId = AkitaModulusProfileId::Q128OffsetA7F7;
const ADDED: AkitaModulusProfileId = AkitaModulusProfileId::Q128Offset275;

fn two_pow(exponent: u32) -> BigUint {
    BigUint::from(1u8) << exponent
}

/// `(d, rank, coeff_linf_bound, max_width)`, each at `max_width` and one
/// column past it. The first three are q128 rows of `policy_audit.csv`: the
/// two compression cells and a row of the largest covered coefficient bound.
/// The last replaces that bound with the largest one the constructor accepts.
fn infinity_cases(profile: AkitaModulusProfileId) -> Vec<SisParameters> {
    [
        (8, 1, 1, 508),
        (16, 1, 1, 7_077),
        (64, 9, 1_821_066_133_292, 33),
        (64, 9, u64::MAX, 33),
    ]
    .into_iter()
    .flat_map(|(d, rank, bound, max_width)| {
        [max_width, max_width + 1].map(|width| {
            scalar_sis_from_ring_wide(profile, d, rank, width, bound).expect("infinity cell")
        })
    })
    .collect()
}

/// `(d, rank, width, collision_l2_sq)`, ending at the largest squared
/// collision bound the constructor accepts.
fn euclidean_cases(profile: AkitaModulusProfileId) -> Vec<SisParameters> {
    [
        (64, 2, 3_000, 1 << 40),
        (512, 1, 64, 1 << 84),
        (512, 1, 64, u128::MAX),
    ]
    .into_iter()
    .map(|(d, rank, width, collision_l2_sq)| {
        scalar_sis_from_ring_euclidean(profile, d, rank, width, collision_l2_sq)
            .expect("Euclidean cell")
    })
    .collect()
}

#[test]
fn the_two_128_bit_moduli_agree_on_what_the_estimator_reads() {
    assert_eq!(
        AkitaModulusProfileId::from(SisModulusProfileId::Q128Offset275),
        ADDED
    );
    assert_eq!(AkitaModulusProfileId::parse(ADDED.label()).unwrap(), ADDED);
    assert_eq!(ADDED.modulus(), two_pow(128) - BigUint::from(275u32));
    assert_ne!(ADDED.modulus(), OWNER.modulus());

    for profile in [OWNER, ADDED] {
        let q = profile.modulus();
        assert_eq!(q.bits(), 128);
        assert_eq!(&q >> 64usize, BigUint::from(u64::MAX));
    }
    assert_eq!(
        log2_biguint(&ADDED.modulus()).to_bits(),
        log2_biguint(&OWNER.modulus()).to_bits()
    );
    assert_eq!(
        half_q_minus_one(&ADDED.modulus()).to_bits(),
        half_q_minus_one(&OWNER.modulus()).to_bits()
    );
}

#[test]
fn akita_inputs_cannot_reach_an_exact_comparison_with_q() {
    // The Akita constructors produce a `u64` integer bound for the infinity
    // norm and the square root of a `u128` for the Euclidean norm. A wider
    // bound type would reopen the exact comparisons and must fail here.
    let mut largest_squared_length = BigUint::from(0u8);
    for params in infinity_cases(ADDED) {
        let Bound::Integer(bound) = &params.length_bound else {
            panic!("infinity cells use an integer bound");
        };
        assert!(bound.bits() <= 64);
        largest_squared_length = largest_squared_length.max(bound * bound);
    }
    for params in euclidean_cases(ADDED) {
        let Bound::SqrtInteger(radicand) = &params.length_bound else {
            panic!("Euclidean cells use a square-root bound");
        };
        assert!(radicand.bits() <= 128);
        largest_squared_length = largest_squared_length.max(radicand.clone());
    }
    assert_eq!(largest_squared_length, BigUint::from(u128::MAX));

    // The small-box test compares `dimension * length^2` with `q^2`, and the
    // dimension is a `u64`. A factor of `2^63` is far outside the fallback
    // window of `2^(1e-8)`, so the test is decided by `log2_biguint` alone,
    // and the Euclidean triviality check `length^2 >= q^2` is false.
    let largest_left_hand_side = BigUint::from(u64::MAX) * &largest_squared_length;
    for profile in [OWNER, ADDED] {
        let q = profile.modulus();
        assert!((&largest_left_hand_side << 63usize) < &q * &q);
    }
}

#[test]
fn pricing_is_constant_across_moduli_that_share_those_quantities() {
    // The two profiles, then the smallest and the largest integer with bit
    // length 128 and leading 64 bits all ones. The two profiles differ only
    // below those leading bits, so the argument above predicts one cost for
    // all four.
    let shared = [
        OWNER.modulus(),
        ADDED.modulus(),
        two_pow(128) - two_pow(64),
        two_pow(128) - BigUint::from(1u8),
    ];
    // Same bit length, different leading bits: pricing must notice.
    let other_leading_bits = two_pow(127) + BigUint::from(1u8);

    let infinity = EstimateConfig::akita_infinity_table();
    let euclidean = EstimateConfig::akita_euclidean_table();
    let cases = infinity_cases(OWNER)
        .into_iter()
        .map(|params| (params, &infinity))
        .chain(
            euclidean_cases(OWNER)
                .into_iter()
                .map(|params| (params, &euclidean)),
        );

    let mut distinguished = 0usize;
    for (params, config) in cases {
        let cost_at = |q: &BigUint| {
            let mut params = params.clone();
            params.q = q.clone();
            estimate(&params, config).expect("estimate")
        };
        let owner_cost = cost_at(&shared[0]);
        for q in &shared[1..] {
            assert_eq!(cost_at(q), owner_cost, "q={q} {params:?}");
        }
        distinguished += usize::from(cost_at(&other_leading_bits) != owner_cost);
    }
    assert!(distinguished > 0, "the comparison never depends on q");
}
