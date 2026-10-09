use super::*;
const PROFILE: LabiniusRootProfile = LabiniusRootProfile::D648P128BoundedW46Delta16;

fn rejection(result: Result<LabiniusRootShape, AkitaError>, message: &str) {
    match result {
        Err(AkitaError::InvalidSetup(actual)) => assert!(actual.contains(message), "{actual}"),
        other => panic!("expected InvalidSetup containing {message:?}, got {other:?}"),
    }
}

#[test]
fn golden_shape_and_largest_fold_width() {
    let shape = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    assert_eq!(
        shape,
        LabiniusRootShape {
            profile: PROFILE,
            coefficient_prime: LabiniusCoefficientPrime::P128OffsetA7F7,
            response_interval: (-32_768, 32_767),
            num_cells: 4_194_304,
            fold_width: 256,
            scalars_per_column: 16_384,
            ring_elements_per_column: 4096,
            packing_degree: 4,
            scalar_degree: 162,
            commitment_degree: 648,
            eta_a: 24_116_880,
            rank_a: 1,
            image_len: 165_888,
            a_quotient_len: 647,
            parity_residual_bound: 86_973_099_520,
            honest_quotient_bound: 173_946_199_040,
            honest_carry_bound: 130_459_649_280,
            foreign_modulus_lift: None,
        }
    );
    assert_eq!(shape.profile(), PROFILE);
    assert_eq!(shape.num_cells(), 1 << 22);
    assert_eq!(shape.fold_width(), 256);
    assert_eq!(shape.scalars_per_column(), 1 << 14);
    assert_eq!(shape.ring_elements_per_column(), 1 << 12);
    assert_eq!(shape.packing_degree(), 4);
    assert_eq!(shape.scalar_degree(), 162);
    assert_eq!(shape.commitment_degree(), 648);
    assert_eq!(shape.rank_a(), 1);
    assert_eq!(shape.eta_a(), 24_116_880);
    assert_eq!(shape.image_len(), 165_888);
    assert_eq!(shape.a_quotient_len(), 647);
    assert_eq!(shape.parity_residual_bound(), 86_973_099_520);
    assert_eq!(shape.honest_quotient_bound(), 173_946_199_040);
    assert_eq!(shape.honest_carry_bound(), 130_459_649_280);
    let maximum = (0..=22)
        .filter(|&log| LabiniusRootShape::derive(PROFILE, 22, log, 128).is_ok())
        .max()
        .unwrap();
    assert!(maximum >= 8);
    assert_eq!(maximum, 8);
    rejection(
        LabiniusRootShape::derive(PROFILE, 22, maximum + 1, 128),
        "challenge profile does not meet fold budget",
    );
}

#[test]
fn caller_geometry_budget_response_and_missing_cell_rejections() {
    LabiniusRootShape::derive(PROFILE, 2, 0, 128).unwrap();
    rejection(
        LabiniusRootShape::derive(PROFILE, 1, 0, 128),
        "M must be a positive multiple of k",
    );
    rejection(
        LabiniusRootShape::derive(PROFILE, 2, 3, 128),
        "M must be a positive multiple of k",
    );
    rejection(
        LabiniusRootShape::derive(PROFILE, usize::BITS, 0, 128),
        "geometry size overflow",
    );
    rejection(
        LabiniusRootShape::derive(PROFILE, 22, usize::BITS, 128),
        "geometry size overflow",
    );
    rejection(
        LabiniusRootShape::derive(PROFILE, u32::MAX, 0, 128),
        "geometry size overflow",
    );
    rejection(
        LabiniusRootShape::derive(PROFILE, 22, 8, 129),
        "challenge profile does not meet fold budget",
    );
    LabiniusRootShape::derive(PROFILE, 22, 8, 127).unwrap();
    rejection(
        LabiniusRootShape::derive(PROFILE, 22, 9, 127),
        "honest response exceeds accepted interval",
    );
    LabiniusRootShape::derive(PROFILE, 32, 0, 128).unwrap();
    rejection(
        LabiniusRootShape::derive(PROFILE, 33, 0, 128),
        "no certified SIS cell",
    );
}

#[test]
fn ledger_no_wrap_boundary_is_exercised_through_the_canonical_derivation() {
    // The sole public profile always has eta=24,116,880<P. Perturb the internal
    // accepted interval to exercise the ledger rejection without a public profile seam.
    let challenge = PROFILE.challenge_profile().unwrap();
    let prime = LabiniusCoefficientPrime::P64Offset23703;
    let gamma = u128::from(challenge.multiplication_linf_operator_bound());
    let first_diameter = prime.modulus().div_ceil(4 * gamma);
    let upper = i128::try_from(first_diameter / 2).unwrap();
    let lower = upper - i128::try_from(first_diameter).unwrap();
    rejection(
        derive_shape(
            PROFILE,
            prime,
            None,
            PROFILE.ring_degree(),
            &challenge,
            (lower, upper),
            2,
            0,
            128,
        ),
        "does not satisfy eta < P",
    );
    let occurrence = SourceOccurrenceBound::binary_extracted(gamma, first_diameter - 1).unwrap();
    assert!(checked_source_comparison_class_bound(
        LabiniusSourceComparisonId {
            coefficient_prime: prime,
            ring_degree: PROFILE.ring_degree(),
            matrix_view_digest: [0; 32],
        },
        &[occurrence]
    )
    .is_ok());
}

#[test]
fn parity_envelopes_check_honesty_and_first_no_wrap_boundary() {
    let shape = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    let q = shape.honest_quotient_bound();
    let k = shape.honest_carry_bound();
    shape.check_parity_no_wrap(q, k).unwrap();
    for (eq, ek, text) in [
        (q - 1, k, "quotient envelope"),
        (q, k - 1, "carry envelope"),
    ] {
        match shape.check_parity_no_wrap(eq, ek) {
            Err(AkitaError::InvalidSetup(message)) => assert!(message.contains(text)),
            other => panic!("unexpected result {other:?}"),
        }
    }
    let remaining = PROFILE.coefficient_prime().modulus() - shape.parity_residual_bound() - 2 * k;
    let first_q = remaining.div_ceil(3);
    shape.check_parity_no_wrap(first_q - 1, k).unwrap();
    assert!(shape.check_parity_no_wrap(first_q, k).is_err());
    assert!(shape.check_parity_no_wrap(u128::MAX, k).is_err());
    assert!(shape.check_parity_no_wrap(q, u128::MAX).is_err());
}
