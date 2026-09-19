use super::*;

#[test]
fn current_capacity_prices_exact_nominal_loss_bits() {
    for (loss, expected) in [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (u64::MAX, 64)] {
        let actual = if expected > u32::from(MAX_GRINDING_BITS) {
            grind_bits_for_loss(loss, 128).expect_err("oversized target")
        } else {
            let actual = grind_bits_for_loss(loss, 128).expect("supported target");
            assert_eq!(u32::from(actual), expected);
            continue;
        };
        assert!(matches!(actual, AkitaError::InvalidSetup(_)));
    }
}

#[test]
fn nominal_security_inequality_holds_for_every_supported_target() {
    let losses = [
        1,
        2,
        3,
        4,
        5,
        (1u64 << (MAX_GRINDING_BITS - 1)) - 1,
        1u64 << (MAX_GRINDING_BITS - 1),
        (1u64 << MAX_GRINDING_BITS) - 1,
        1u64 << MAX_GRINDING_BITS,
    ];
    for loss in losses {
        let grind = grind_bits_for_loss(loss, 128).expect("supported loss");
        assert!(u128::from(loss) <= (1u128 << grind));
    }
}

#[test]
fn nonce_slack_provisions_exactly_128_expected_trials() {
    for grind in 1..=MAX_GRINDING_BITS {
        let nonce_bits = grind + GRINDING_NONCE_SLACK_BITS;
        assert_eq!((1u64 << nonce_bits) / (1u64 << grind), 128);
        let failure = (1.0 - 2f64.powi(-i32::from(grind))).powf(2f64.powi(i32::from(nonce_bits)));
        assert!(failure <= (-128f64).exp());
    }
}

#[test]
fn plan_encoding_covers_every_discriminator() {
    let capacity = 128;
    let sites = [
        GrindingSite::EvaluationBatch { level: 0 },
        GrindingSite::ExtensionOpeningPoint { level: 0 },
        GrindingSite::ExtensionOpeningClaimBatch { level: 0 },
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::ExtensionOpeningReduction,
            level: 0,
            stage: 1,
            round: 2,
        },
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::Stage1,
            level: 3,
            stage: 4,
            round: 5,
        },
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::PhysicalL2,
            level: 6,
            stage: 7,
            round: 8,
        },
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::Stage2,
            level: 9,
            stage: 10,
            round: 11,
        },
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::Stage3,
            level: 12,
            stage: 13,
            round: 14,
        },
        GrindingSite::RingSwitchAlpha { level: 1 },
        GrindingSite::Tau0Point { level: 1 },
        GrindingSite::Tau1Point { level: 1 },
        GrindingSite::Stage1InterstageBatch { level: 1, stage: 2 },
        GrindingSite::L2SubclaimBatch { level: 1 },
        GrindingSite::L2NormMerge { level: 1 },
        GrindingSite::L2VirtualBatch { level: 1 },
        GrindingSite::CompressionBinary { level: 1 },
        GrindingSite::Stage2Batch { level: 1 },
    ];
    let mut runs = sites
        .into_iter()
        .map(|site| GrindingRun::proof_of_work(site, 3, capacity).unwrap())
        .collect::<Vec<_>>();
    runs.push(GrindingRun::fold_response(2));
    runs.push(GrindingRun::fold_challenge_group(2, 3, 4).unwrap());
    let plan = GrindingPlan::new(runs, capacity).unwrap();
    let bytes = plan.canonical_bytes().unwrap();
    assert!(bytes.starts_with(GRINDING_PLAN_DOMAIN));
    assert_eq!(plan.expanded_query_count(), 23);
    assert_eq!(plan.total_nonce_bits(), 17 * 9 + 12);
    assert_eq!(
        plan.digest().unwrap(),
        [
            179, 255, 114, 159, 154, 171, 9, 43, 28, 145, 136, 184, 170, 219, 207, 226, 194, 253,
            244, 145, 170, 157, 89, 76, 55, 219, 113, 80, 49, 167, 153, 126,
        ]
    );
}

#[test]
fn ring_switch_loss_uses_the_opening_polynomial_dimension() {
    assert_eq!(
        ring_switch_alpha_loss_factor(OpeningMethod::EvaluationTrace, 64).unwrap(),
        127
    );
    assert_eq!(
        ring_switch_alpha_loss_factor(
            OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 16,
            },
            64,
        )
        .unwrap(),
        31
    );
}

#[test]
fn special_proof_of_work_site_and_reserved_sentinel_are_rejected() {
    assert!(GrindingRun::proof_of_work(GrindingSite::FoldResponse { level: 0 }, 1, 128).is_err());

    let mut underpriced =
        GrindingRun::proof_of_work(GrindingSite::RingSwitchAlpha { level: 0 }, 3, 128).unwrap();
    underpriced.grind_bits = 1;
    underpriced.nonce_bits = 8;
    assert!(GrindingPlan::new(vec![underpriced], 128).is_err());

    let reserved = GrindingRun::proof_of_work(
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::Stage2,
            level: u32::MAX,
            stage: 0,
            round: 0,
        },
        3,
        128,
    )
    .unwrap();
    assert!(GrindingPlan::new(vec![reserved], 128).is_err());
}

#[test]
fn public_plan_rejects_query_limit_without_expanding_runs() {
    let accepted =
        GrindingRun::fold_challenge_group(0, 0, TRANSCRIPT_GRINDING_QUERY_LIMIT - 2).unwrap();
    assert_eq!(
        GrindingPlan::new(vec![accepted], 128)
            .unwrap()
            .expanded_query_count(),
        TRANSCRIPT_GRINDING_QUERY_LIMIT - 1
    );

    let excessive =
        GrindingRun::fold_challenge_group(0, 0, TRANSCRIPT_GRINDING_QUERY_LIMIT - 1).unwrap();
    assert!(matches!(
        GrindingPlan::new(vec![excessive], 128),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn planner_accumulator_prices_an_oversized_edge_without_plan_validation() {
    let run = GrindingRun::fold_challenge_group(0, 0, TRANSCRIPT_GRINDING_QUERY_LIMIT - 1).unwrap();
    let mut accumulator = GrindingPlanAccumulator::new(128).unwrap();
    accumulator.push(run).unwrap();
    assert_eq!(
        accumulator.cost(),
        TranscriptGrindingCost {
            total_nonce_bits: 0,
            native_nonce_bytes: 0,
            expanded_query_count: TRANSCRIPT_GRINDING_QUERY_LIMIT,
        }
    );
}
