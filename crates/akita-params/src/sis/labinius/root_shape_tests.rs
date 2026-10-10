use num_bigint::BigUint;

use super::*;

const PROFILE: LabiniusRootProfile = LabiniusRootProfile::D648Q25BoundedW46;
const P64_OFFSET_59: u128 = (1 << 64) - 59;
const P128_OFFSET_275: u128 = u128::MAX - 274;

fn rejection<T: std::fmt::Debug>(result: Result<T, AkitaError>, message: &str) {
    match result {
        Err(AkitaError::InvalidSetup(actual)) => assert!(actual.contains(message), "{actual}"),
        other => panic!("expected InvalidSetup containing {message:?}, got {other:?}"),
    }
}

/// The sample geometry of the design: every number is derived, and the
/// derived value is compared with the design's figure.
#[test]
fn sample_geometry_derives_the_design_numbers() {
    let shape = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    assert_eq!(shape.ring_elements_per_column(), 4_096);
    assert_eq!(shape.fold_width(), 256);
    assert_eq!(shape.scalars_per_column(), 16_384);
    assert_eq!((shape.packing_degree(), shape.scalar_degree()), (4, 162));
    assert_eq!(shape.response().honest_cap(), 1_266);
    assert_eq!(shape.response().interval(), (-2_184, 1_911));
    assert_eq!(shape.response().diameter(), 4_095);
    assert_eq!(shape.response().eta_a(), 1_506_960);
    assert_eq!(shape.rank_a(), 2);
    assert_eq!(shape.honest_a_carry_bound(), 17_390_406_144);
    // H = d * M * B_z + C * w, B_Q = 2 * H, B_K = floor(3 * H / 2).
    let residual = 162 * 16_384 * 2_184 + 256 * 46;
    assert_eq!(shape.parity_residual_bound(), residual);
    assert_eq!(shape.honest_quotient_bound(), 2 * residual);
    assert_eq!(shape.honest_carry_bound(), 3 * residual / 2);
}

#[test]
fn geometry_budget_and_missing_cell_rejections() {
    LabiniusRootShape::derive(PROFILE, 2, 0, 128).unwrap();
    for (log_num_cells, log_fold_width) in [(1, 0), (2, 3)] {
        rejection(
            LabiniusRootShape::derive(PROFILE, log_num_cells, log_fold_width, 128),
            "M must be a positive multiple of k",
        );
    }
    for (log_num_cells, log_fold_width) in [(usize::BITS, 0), (22, usize::BITS), (u32::MAX, 0)] {
        rejection(
            LabiniusRootShape::derive(PROFILE, log_num_cells, log_fold_width, 128),
            "geometry size overflow",
        );
    }
    // 256 columns is the widest fold meeting 128 bits.
    rejection(
        LabiniusRootShape::derive(PROFILE, 22, 8, 129),
        "challenge profile does not meet fold budget",
    );
    rejection(
        LabiniusRootShape::derive(PROFILE, 22, 9, 128),
        "challenge profile does not meet fold budget",
    );
    // The certified cells cover three response digits and widths to 6.4e12.
    // A wider fold needs a fourth digit; a longer column leaves the width cap.
    let widest = (9..=22)
        .map(|log| (log, LabiniusRootShape::derive(PROFILE, 22, log, 0)))
        .find(|(_, shape)| shape.is_err())
        .unwrap();
    let (first_uncovered, uncovered) = widest;
    rejection(uncovered, "no certified SIS cell");
    let covered = LabiniusRootShape::derive(PROFILE, 22, first_uncovered - 1, 0).unwrap();
    assert_eq!(covered.response().digit_count(), 3);
    let fourth_digit = PROFILE.challenge_profile().unwrap();
    let response = LabiniusFoldResponse::derive(
        &fourth_digit,
        1 << first_uncovered,
        (1 << (22 - first_uncovered)) / 4,
        PROFILE.ring_degree(),
    )
    .unwrap();
    assert_eq!(response.digit_count(), 4);
    LabiniusRootShape::derive(PROFILE, 44, 0, 128).unwrap();
    rejection(
        LabiniusRootShape::derive(PROFILE, 45, 0, 128),
        "no certified SIS cell",
    );
}

/// The check is `n_A * m * D * q * 2^64 <= 4 * P`, evaluated here in big
/// integers. A 64-bit stream is rejected at every geometry.
#[test]
fn derivation_bias_check_agrees_with_big_integer_evaluation() {
    let q = PROFILE.commitment_modulus().modulus();
    let (mut accepted, mut rejected) = (0, 0);
    for (log_num_cells, log_fold_width) in [(2, 0), (12, 4), (22, 8), (30, 8), (44, 0)] {
        let shape = LabiniusRootShape::derive(PROFILE, log_num_cells, log_fold_width, 128).unwrap();
        let coefficients = BigUint::from(shape.rank_a())
            * shape.ring_elements_per_column()
            * shape.commitment_degree()
            * q;
        // The least admitted stream modulus, when one exists below 2^128.
        let boundary = u128::try_from(&coefficients << 62usize).map_or(vec![], |first_admitted| {
            vec![first_admitted - 1, first_admitted]
        });
        for stream_prime in [vec![P64_OFFSET_59, P128_OFFSET_275], boundary].concat() {
            let expected = (&coefficients << 64usize) <= BigUint::from(stream_prime) * 4u32;
            let result = shape.check_derivation_bias(stream_prime);
            assert_eq!(result.is_ok(), expected);
            if expected {
                accepted += 1;
            } else {
                rejection(result, "derivation bias is below policy floor");
                rejected += 1;
            }
        }
        assert!(shape.check_derivation_bias(P64_OFFSET_59).is_err());
    }
    assert!(accepted > 0 && rejected > 0);
    let sample = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    assert!(sample.check_derivation_bias(P128_OFFSET_275).is_ok());
}
