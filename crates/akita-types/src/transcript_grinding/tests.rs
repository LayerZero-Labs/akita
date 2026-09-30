use super::*;

#[test]
fn full_capacity_prices_exact_loss_bits() {
    let order = ChallengeFieldOrder::from_full_capacity(128).unwrap();
    for (loss, expected) in [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (u64::MAX, 64)] {
        let actual = if expected > u32::from(MAX_GRINDING_BITS) {
            grind_bits_for_loss(loss, order).expect_err("oversized target")
        } else {
            let actual = grind_bits_for_loss(loss, order).expect("supported target");
            assert_eq!(u32::from(actual), expected);
            continue;
        };
        assert!(matches!(actual, AkitaError::InvalidSetup(_)));
    }
}

#[test]
fn full_capacity_security_inequality_holds_for_every_supported_target() {
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
    let order = ChallengeFieldOrder::from_full_capacity(128).unwrap();
    for loss in losses {
        let grind = grind_bits_for_loss(loss, order).expect("supported loss");
        assert!(u128::from(loss) <= (1u128 << grind));
    }
}

#[test]
fn exact_pseudo_mersenne_orders_meet_the_security_bound_with_minimal_bits() {
    const FP128_MODULUS: u128 = 340_282_366_920_938_463_463_374_607_427_473_266_697;
    const FP64_MODULUS: u128 = 18_446_744_073_709_551_557;
    const FP32_MODULUS: u128 = 4_294_967_197;

    fn assert_exact_price(field_modulus: u128, modulus_bits: u32, extension_degree: u32) {
        let order = field_modulus
            .checked_pow(extension_degree)
            .expect("selected exact challenge order fits in u128");
        for (loss, expected_bits) in [(1u64, 1u8), (2, 2), (3, 2), (4, 3), (5, 3), (7, 3), (8, 4)] {
            let challenge_order = ChallengeFieldOrder::from_field(
                modulus_bits,
                usize::try_from(extension_degree).unwrap(),
                field_modulus,
            )
            .expect("valid exact field order");
            let bits = grind_bits_for_loss(loss, challenge_order).expect("supported exact price");
            assert_eq!(
                bits, expected_bits,
                "loss={loss}, degree={extension_degree}"
            );

            let security_shift = TRANSCRIPT_SECURITY_BITS - u16::from(bits);
            let exact_threshold = if u32::from(security_shift) >= u128::BITS {
                None
            } else {
                let shift = u32::from(security_shift);
                u128::from(loss)
                    .checked_shl(shift)
                    .filter(|threshold| *threshold >> shift == u128::from(loss))
            };
            assert!(
                exact_threshold.is_some_and(|threshold| threshold <= order),
                "priced inequality must hold for loss={loss}, bits={bits}"
            );
            if bits > 0 {
                let previous_shift = TRANSCRIPT_SECURITY_BITS - u16::from(bits - 1);
                let previous_threshold = if u32::from(previous_shift) >= u128::BITS {
                    None
                } else {
                    let shift = u32::from(previous_shift);
                    u128::from(loss)
                        .checked_shl(shift)
                        .filter(|threshold| *threshold >> shift == u128::from(loss))
                };
                assert!(
                    previous_threshold.is_none_or(|threshold| threshold > order),
                    "one fewer bit must fail for loss={loss}, bits={bits}"
                );
            }
        }
    }

    assert_exact_price(FP128_MODULUS, 128, 1);
    assert_exact_price(FP64_MODULUS, 64, 2);
    assert_exact_price(FP32_MODULUS, 32, 4);
}

#[test]
fn exact_orders_above_the_comparison_width_require_no_small_loss_grinding() {
    const FP128_MODULUS: u128 = 340_282_366_920_938_463_463_374_607_427_473_266_697;
    let quadratic = ChallengeFieldOrder::from_field(128, 2, FP128_MODULUS).unwrap();
    assert_eq!(grind_bits_for_loss(2, quadratic).unwrap(), 0);
    assert_eq!(grind_bits_for_loss(u64::MAX, quadratic).unwrap(), 0);

    let quartic = ChallengeFieldOrder::from_field(128, 4, FP128_MODULUS).unwrap();
    assert_eq!(grind_bits_for_loss(u64::MAX, quartic).unwrap(), 0);
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
        .map(|site| {
            GrindingRun::proof_of_work(
                site,
                3,
                ChallengeFieldOrder::from_full_capacity(capacity).unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    runs.push(GrindingRun::fold_response(2));
    runs.push(GrindingRun::fold_challenge_group(2, 3, 4).unwrap());
    let plan = GrindingPlan::new(
        runs,
        ChallengeFieldOrder::from_full_capacity(capacity).unwrap(),
    )
    .unwrap();
    let bytes = plan.canonical_bytes().unwrap();
    assert!(bytes.starts_with(GRINDING_PLAN_DOMAIN));
    assert_eq!(plan.expanded_query_count(), 23);
    assert_eq!(plan.total_nonce_bits(), 17 * 9 + 12);
    assert_eq!(
        plan.digest().unwrap(),
        [
            23, 236, 97, 122, 19, 64, 167, 147, 169, 143, 37, 147, 216, 175, 71, 42, 8, 224, 38,
            33, 26, 236, 47, 127, 109, 150, 186, 198, 151, 118, 193, 12,
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
    let order = ChallengeFieldOrder::from_full_capacity(128).unwrap();
    assert!(GrindingRun::proof_of_work(GrindingSite::FoldResponse { level: 0 }, 1, order).is_err());

    let mut underpriced =
        GrindingRun::proof_of_work(GrindingSite::RingSwitchAlpha { level: 0 }, 3, order).unwrap();
    underpriced.grind_bits = 1;
    underpriced.nonce_bits = 8;
    assert!(GrindingPlan::new(vec![underpriced], order).is_err());

    let reserved = GrindingRun::proof_of_work(
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::Stage2,
            level: u32::MAX,
            stage: 0,
            round: 0,
        },
        3,
        order,
    )
    .unwrap();
    assert!(GrindingPlan::new(vec![reserved], order).is_err());
}

#[test]
fn public_plan_rejects_query_limit_without_expanding_runs() {
    let order = ChallengeFieldOrder::from_full_capacity(128).unwrap();
    let accepted =
        GrindingRun::fold_challenge_group(0, 0, TRANSCRIPT_GRINDING_QUERY_LIMIT - 2).unwrap();
    assert_eq!(
        GrindingPlan::new(vec![accepted], order)
            .unwrap()
            .expanded_query_count(),
        TRANSCRIPT_GRINDING_QUERY_LIMIT - 1
    );

    let excessive =
        GrindingRun::fold_challenge_group(0, 0, TRANSCRIPT_GRINDING_QUERY_LIMIT - 1).unwrap();
    assert!(matches!(
        GrindingPlan::new(vec![excessive], order),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn planner_accumulator_prices_an_oversized_edge_without_plan_validation() {
    let run = GrindingRun::fold_challenge_group(0, 0, TRANSCRIPT_GRINDING_QUERY_LIMIT - 1).unwrap();
    let mut accumulator =
        GrindingPlanAccumulator::new(ChallengeFieldOrder::from_full_capacity(128).unwrap());
    accumulator.push(run).unwrap();
    assert_eq!(
        accumulator.cost(),
        TranscriptGrindingCost {
            total_nonce_bits: 0,
            native_nonce_max_bytes: 0,
            expanded_query_count: TRANSCRIPT_GRINDING_QUERY_LIMIT,
        }
    );
}
