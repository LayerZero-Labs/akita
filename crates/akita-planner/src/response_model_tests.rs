use super::*;

fn response_geometry_params(opening_method: akita_types::OpeningMethod) -> CommittedGroupParams {
    let mut params = CommittedGroupParams::params_only(
        akita_types::SisModulusProfileId::Q128OffsetA7F7,
        256,
        2,
        2,
        2,
        2,
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(64)
            .expect("D64 challenge"),
    )
    .with_decomp(4, 8, 2, 2, 2)
    .expect("response geometry params");
    params.payload_mode = akita_types::CommitmentPayloadMode::Raw;
    params.own_group_mut().opening.opening_method = opening_method;
    let opening = params.open().matrix;
    params.open_matrix = akita_types::OpenCommitMatrixParams::new_unchecked(
        opening.security_policy(),
        opening.sis_table_key().table_digest,
        opening.sis_modulus_profile(),
        opening.output_rank(),
        opening.input_width(),
        opening.coeff_linf_bound(),
        128,
    );
    params
}

#[test]
fn next_source_moment_prices_packing_e_and_r_but_keeps_ambient_t() {
    let opening = OpeningClaimsLayout::new(0, 1).expect("opening batch");
    let source = SourceMomentEstimate::new(1 << 16).expect("source moment");
    let packing = response_geometry_params(akita_types::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    });
    let trace = response_geometry_params(akita_types::OpeningMethod::EvaluationTrace);
    let packing_moment =
        next_source_moment(&packing, &opening, &[source], 128, 2).expect("packing source moment");
    let trace_moment =
        next_source_moment(&trace, &opening, &[source], 128, 2).expect("trace source moment");
    assert!(packing_moment.mean_l2_sq() < trace_moment.mean_l2_sq());
    assert_eq!(
        packing_moment.components[T_COMPONENT],
        trace_moment.components[T_COMPONENT]
    );
    assert!(
        packing_moment.components[E_COMPONENT].mean_l2_sq
            < trace_moment.components[E_COMPONENT].mean_l2_sq
    );
    assert!(
        packing_moment.components[R_COMPONENT].mean_l2_sq
            < trace_moment.components[R_COMPONENT].mean_l2_sq
    );
}

#[test]
fn reduced_evaluation_source_moment_omits_only_quotient_rows() {
    let opening = OpeningClaimsLayout::new(0, 1).expect("opening batch");
    let source = SourceMomentEstimate::new(1 << 16).expect("source moment");
    for payload_mode in [
        akita_types::CommitmentPayloadMode::Raw,
        akita_types::CommitmentPayloadMode::Compressed,
    ] {
        let mut lifted = response_geometry_params(akita_types::OpeningMethod::EvaluationTrace);
        lifted.payload_mode = payload_mode;
        lifted.ring_relation_mode = akita_types::RingRelationMode::QuotientLift;
        let mut reduced = lifted.clone();
        reduced.ring_relation_mode = akita_types::RingRelationMode::ReducedEvaluation;

        let lifted_moment =
            next_source_moment(&lifted, &opening, &[source], 128, 2).expect("lifted moment");
        let reduced_moment =
            next_source_moment(&reduced, &opening, &[source], 128, 2).expect("reduced moment");

        assert!(lifted_moment.components[R_COMPONENT].mean_l2_sq > 0);
        assert_eq!(
            reduced_moment.components[R_COMPONENT],
            SourceMomentComponent::default()
        );
        for component in [Z_COMPONENT, E_COMPONENT, T_COMPONENT, COMPRESSION_COMPONENT] {
            assert_eq!(
                reduced_moment.components[component],
                lifted_moment.components[component]
            );
        }
        assert!(reduced_moment.mean_l2_sq() < lifted_moment.mean_l2_sq());
    }
}

#[test]
fn field_plane_moments_include_the_residual_top_plane() {
    let energy = field_digit_energy(1_000_000, 64, 6, 11).unwrap();
    let expected = 1_000_000.0 * (10.0 * 341.5 + 21.5);
    assert_eq!(energy, expected as u128);
}

/// A bounded source charges each plane the range its bound leaves, and charges
/// the carry plane past the bound rather than dropping it.
///
/// The canonical depth adds that extra plane exactly when `log_basis` divides the
/// bound, and balanced extraction can push `±1` into it
/// (`|c_p| <= |v| / b^p + b / (2·(b - 1))`). Dropping it would make this an
/// under-estimate rather than the deterministic maximum the L2 caps are priced
/// from.
#[test]
fn bounded_source_charges_the_carry_plane_past_its_bound() {
    let per_scalar = |bound, log_basis, digits| {
        bounded_field_source_moment(1, bound, log_basis, digits)
            .unwrap()
            .mean_l2_sq()
    };
    // `mean_l2_sq` is bucketed conservatively upward, so the expectation goes
    // through the same bucketing. That keeps the assertion exact about the plane
    // arithmetic without restating the bucket rule.
    let plane_energy = |full_planes: u128, log_basis: u32, top_plane_bits: u32| {
        let raw = full_planes * (1u128 << (log_basis - 1)).pow(2)
            + (1u128 << (top_plane_bits - 1)).pow(2);
        SourceMomentEstimate::new(raw).unwrap().mean_l2_sq()
    };

    // No overshoot: 13 base-2^5 planes under a 64-bit bound consume 60 bits, so
    // every plane is a real one and the top plane is charged the 4 bits left.
    assert_eq!(per_scalar(64, 5, 13), plane_energy(12, 5, 4));

    // Overshoot by exactly one plane: `log_basis = 8` divides `bound = 64`, so
    // `compute_num_digits` returns 9 and plane 8 sits entirely past the bound.
    // It is charged `1` (a `plane_bits = 1` carry), not dropped.
    assert_eq!(per_scalar(64, 8, 9), plane_energy(8, 8, 1));

    // The same geometry one digit shallower has no carry plane at all, so it is
    // strictly cheaper — the carry charge is load-bearing, not rounding noise.
    assert!(per_scalar(64, 8, 8) < per_scalar(64, 8, 9));

    // A `u64` workload (`log_commit_bound = 65`) at the base the nv=14 row picks:
    // 14 base-2^5 planes span 70 bits, so plane 13 is the carry plane.
    assert_eq!(per_scalar(65, 5, 14), plane_energy(13, 5, 1));

    // A full-field source never overshoots by a whole plane, so the carry rule
    // cannot fire for it: `ceil(128 / 5) = 26` planes consume at most 125 bits.
    assert_eq!(per_scalar(128, 5, 26), plane_energy(25, 5, 3));
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
fn rounded_normal_digit_moment_matches_two_sided_reference() {
    fn reference(sigma: f64, basis: i64) -> f64 {
        let radius = (8.0 * sigma + 0.5).ceil() as i64;
        let mut moment = 0.0;
        let mut lower_cdf = normal_cdf((-radius as f64 - 0.5) / sigma);
        for value in -radius..=radius {
            let upper_cdf = normal_cdf((value as f64 + 0.5) / sigma);
            let probability = upper_cdf - lower_cdf;
            let digit = centered_residue(value, basis) as f64;
            moment += probability * digit * digit;
            lower_cdf = upper_cdf;
        }
        moment
    }

    for basis in [2, 4, 8, 16, 64] {
        for sigma in [0.1, 0.75, 1.5, basis as f64 * 0.9] {
            let expected = reference(sigma, basis);
            let actual = rounded_normal_digit_second_moment(sigma, basis);
            let tolerance = expected.abs().max(1.0) * 1e-12;
            assert!(
                (actual - expected).abs() <= tolerance,
                "sigma={sigma} basis={basis}: actual={actual}, expected={expected}"
            );
        }
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
