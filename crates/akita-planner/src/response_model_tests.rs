use super::*;

#[test]
fn field_plane_moments_include_the_residual_top_plane() {
    let energy = field_digit_energy(1_000_000, 64, 6, 11).unwrap();
    let expected = 1_000_000.0 * (10.0 * 341.5 + 21.5);
    assert_eq!(energy, expected as u128);
}

#[test]
fn tensor_pack_moments_match_supported_extension_factors() {
    assert_eq!(tensor_packed_moments(400, 100, 1), Some((400, 100, 100)));
    assert_eq!(tensor_packed_moments(400, 100, 2), Some((600, 150, 200)));
    assert_eq!(tensor_packed_moments(400, 100, 4), Some((700, 175, 200)));
    assert_eq!(tensor_packed_moments(400, 100, 8), Some((750, 188, 200)));
}

#[test]
fn peak_column_shares_capacity_across_disjoint_components() {
    const PEAK: u128 = 1 << 24;
    let component = SourceMomentComponent {
        mean_l2_sq: 1024,
        full_ring_peak_second_moment_ppm: PEAK,
        local_peak_second_moment_ppm: 2 * PEAK,
    };
    let source = SourceMomentEstimate::from_components(
        [
            component,
            component,
            component,
            component,
            Default::default(),
        ],
        8,
    )
    .unwrap();

    assert_eq!(
        source.peak_column_second_moment_ppm(8, 1),
        Some(8 * PEAK),
        "four disjoint component classes must share one eight-coefficient column"
    );
    assert_eq!(
        source.peak_column_second_moment_ppm(4, 2),
        Some(16 * PEAK),
        "a strict subring retains the local two-coordinate packing bound"
    );
}

#[test]
fn gaussian_z_model_matches_measured_cross_field_states() {
    // These independent measurements test the rounded-normal digit transform.
    // Current schedule calibration is checked by the profile report pipeline.
    let rows = [
        (21_319_133_492, 524_288, 3, 4, 8_570_345),
        (352_065_629, 65_536, 4, 3, 2_447_776),
        (3_847_283_483, 262_144, 3, 4, 3_767_203),
        (473_967_459, 65_536, 4, 3, 2_593_330),
        (234_370_171, 32_768, 5, 2, 3_041_573),
        (9_985_694_564, 262_144, 4, 3, 11_458_186),
        (483_233_512, 32_768, 6, 2, 11_379_250),
        (2_853_063_371, 16_384, 6, 2, 6_333_831),
    ];
    for (response, count, log_basis, digits, observed) in rows {
        let predicted = gaussian_response_digit_energy(response, count, log_basis, digits).unwrap();
        let relative_error = (predicted as f64 / observed as f64 - 1.0).abs();
        assert!(
            relative_error <= 0.02,
            "response={response} basis={log_basis}: predicted={predicted}, observed={observed}, error={relative_error}"
        );
    }
}

#[test]
fn cap_multiplier_has_markov_grinding_budget() {
    let source = SourceMomentEstimate::new(1_048_576).unwrap();
    assert_eq!(source.response_l2_sq_cap(75), Some(83_079_484));
    assert_eq!(
        83_079_484u128,
        (78_643_200u128 * 1_030_000u128 * 40).div_ceil(1_000_000u128 * 39)
    );
}

#[test]
fn gaussian_slab_quantile_meets_joint_grinding_target() {
    let count = 16_384;
    let quantile = whole_response_normal_quantile(count).unwrap();
    let marginal = 1.0 - libm::erfc(quantile / core::f64::consts::SQRT_2);
    let joint_lower_bound = libm::exp(count as f64 * libm::log(marginal));
    assert!((joint_lower_bound * 40.0 - 1.0).abs() <= 1e-9);
}

#[test]
fn source_moment_bucketing_is_conservative_and_below_one_over_sixty_four() {
    for value in [1, 127, 128, 129, 1_000_000, u64::MAX as u128] {
        let bucketed = SourceMomentEstimate::new(value).unwrap().mean_l2_sq();
        assert!(bucketed >= value);
        assert!(bucketed - value < value.div_ceil(64).max(1));
    }
}
