use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};

use super::super::SourceOccurrenceBound;
use super::*;
use crate::sis::source_comparison_inf_norm;

fn profile(ring: BinaryScalarRing, weight: usize) -> BinaryChallengeProfile {
    BinaryChallengeProfile::bounded_weight(ring, weight).unwrap()
}

/// Largest value of `digits` balanced base-16 digits, summed digit by digit.
fn positive_reach(digits: usize) -> u128 {
    (0..digits).map(|i| 7u128 << (4 * i)).sum()
}

#[test]
fn sample_fold_matches_the_design_anchors() {
    let challenge = profile(BinaryScalarRing::Cyclotomic243, 46);
    let response =
        LabiniusFoldResponse::derive(&challenge, 256, 4_096, LabiniusRingDegree::D648, 128)
            .unwrap();
    assert_eq!(response.honest_cap(), 1_266);
    assert_eq!(response.tail_threshold(), 1_266);
    assert_eq!(response.digit_count(), 3);
    assert_eq!(response.interval(), (-2_184, 1_911));
    assert_eq!(response.diameter(), 4_095);
    assert_eq!(response.eta_a(), 1_506_960);
    assert_eq!(response.abort_probability_bound(), (7, 8));
}

#[test]
fn folds_outside_the_challenge_budget_are_rejected() {
    let singleton =
        BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 162).unwrap();
    let sample = profile(BinaryScalarRing::Cyclotomic243, 46);
    LabiniusFoldResponse::derive(&sample, 256, 4_096, LabiniusRingDegree::D648, 128).unwrap();
    for (challenge, columns, width, degree, lambda_fold) in [
        (&singleton, 65_536, 1, LabiniusRingDegree::D162, 0),
        (&sample, 256, 4_096, LabiniusRingDegree::D648, 129),
        (&sample, 512, 4_096, LabiniusRingDegree::D648, 128),
    ] {
        match LabiniusFoldResponse::derive(challenge, columns, width, degree, lambda_fold) {
            Err(AkitaError::InvalidSetup(message)) => assert!(
                message.contains("challenge profile does not meet fold budget"),
                "{message}"
            ),
            other => panic!("expected InvalidSetup naming the fold budget, got {other:?}"),
        }
    }
}

/// The cap is compared with the real Hoeffding threshold
/// `sqrt(2 * C * w * 2^2 * ln(2 * m * D / (7/8)))` computed in floating point,
/// the digit data with a digit-by-digit reach, and `eta_A` with the binary
/// source-comparison ledger.
#[test]
fn derivation_agrees_with_independent_definitions() {
    let rings = [
        (BinaryScalarRing::Cyclotomic243, LabiniusRingDegree::D162),
        (BinaryScalarRing::Cyclotomic243, LabiniusRingDegree::D648),
        (BinaryScalarRing::Cyclotomic729, LabiniusRingDegree::D486),
        (BinaryScalarRing::Cyclotomic729, LabiniusRingDegree::D1944),
    ];
    for (ring, degree) in rings {
        for weight in [2usize, 16, 46, 81] {
            let challenge = profile(ring, weight);
            for columns in [1usize, 16, 256, 4_096] {
                for width in [1usize, 64, 4_096, 1 << 20] {
                    let response =
                        LabiniusFoldResponse::derive(&challenge, columns, width, degree, 0)
                            .unwrap();
                    let coefficients = (width * degree.degree() as usize) as f64;
                    let variance = (columns * weight * 4) as f64;
                    let hoeffding = (2.0 * variance * (16.0 * coefficients / 7.0).ln()).sqrt();
                    let tail = response.tail_threshold() as f64;
                    assert!(hoeffding <= tail && tail <= 1.15 * hoeffding + 1.0);

                    let worst_case = (columns * 2 * weight) as u128;
                    let cap = response.honest_cap();
                    assert_eq!(cap, worst_case.min(response.tail_threshold()));
                    let deterministic = worst_case < response.tail_threshold();
                    assert_eq!(response.abort_probability_bound().0 == 0, deterministic);

                    let digits = response.digit_count();
                    assert!(positive_reach(digits - 1) < cap && cap <= positive_reach(digits));
                    let (lower, upper) = response.interval();
                    assert_eq!(upper, positive_reach(digits) as i128);
                    assert_eq!(lower, -upper - (upper / 7));
                    assert_eq!(response.diameter(), (upper - lower) as u128);

                    let gamma = u128::from(challenge.multiplication_linf_operator_bound());
                    let occurrence =
                        SourceOccurrenceBound::binary_extracted(gamma, response.diameter())
                            .unwrap();
                    let ledger = source_comparison_inf_norm(
                        occurrence.numerator_bound(),
                        occurrence.slack_operator_bound(),
                        occurrence.numerator_bound(),
                        occurrence.slack_operator_bound(),
                    );
                    assert_eq!(Some(response.eta_a()), ledger);
                }
            }
        }
    }
}

#[test]
fn degenerate_and_mismatched_folds_are_rejected() {
    let challenge = profile(BinaryScalarRing::Cyclotomic243, 46);
    let degree = LabiniusRingDegree::D648;
    assert!(LabiniusFoldResponse::derive(&challenge, 0, 4_096, degree, 0).is_err());
    assert!(LabiniusFoldResponse::derive(&challenge, 256, 0, degree, 0).is_err());
    assert!(LabiniusFoldResponse::derive(&challenge, 256, usize::MAX, degree, 0).is_err());
    let zero = profile(BinaryScalarRing::Cyclotomic243, 0);
    assert!(LabiniusFoldResponse::derive(&zero, 256, 4_096, degree, 0).is_err());
    let other_ring = profile(BinaryScalarRing::Cyclotomic729, 46);
    assert!(LabiniusFoldResponse::derive(&other_ring, 256, 4_096, degree, 0).is_err());
}
