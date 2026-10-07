use akita_sis_estimator::{
    akita_q32, cost_infinity, math::log2_erf_from_log2_arg, probability::log2_amplify, Adps16Mode,
    Bound, EstimateConfig, ReductionCostModel, SisNorm, SisParameters,
};

#[test]
fn log_erf_is_accurate_and_conservative_against_independent_decimal_oracle() {
    // Generated with 140-digit Decimal arithmetic and an independent power
    // series; covers approximation branch boundaries and small/large tails.
    for row in include_str!("../../../scripts/sis_golden/probability_oracle.csv")
        .lines()
        .skip(1)
    {
        let (arg, expected) = row.split_once(',').unwrap();
        let arg: f64 = arg.parse().unwrap();
        let expected: f64 = expected.parse().unwrap();
        let actual = log2_erf_from_log2_arg(arg);
        assert!(actual.is_finite(), "log2_arg={arg}");
        assert!(actual <= 0.0, "log2_arg={arg}");
        assert!(actual >= expected, "log2_arg={arg}: {actual} < {expected}");
        assert!(
            actual - expected <= expected.abs() * 1e-10,
            "log2_arg={arg}: {actual} vs {expected}"
        );
    }
}

#[test]
fn numerical_allowance_never_overcharges_at_repetition_discontinuities() {
    let mut rounded_in_attackers_favor = false;
    for row in include_str!("../../../scripts/sis_golden/probability_repetition_boundaries.csv")
        .lines()
        .skip(1)
    {
        let fields: Vec<f64> = row.split(',').map(|value| value.parse().unwrap()).collect();
        let log_probability = fields[1] * log2_erf_from_log2_arg(fields[0]) + fields[2];
        let actual = log2_amplify(0.99, log_probability);
        let expected = fields[3].log2();
        assert!(actual.is_finite() && actual >= 0.0);
        assert!(actual <= expected, "{row}: {actual} > {expected}");
        rounded_in_attackers_favor |= actual < expected;
    }
    assert!(rounded_in_attackers_favor);
}

#[test]
fn reported_cells_and_neighbors_match_high_precision_repetition_decisions() {
    for row in include_str!("../../../scripts/sis_golden/probability_boundaries.csv")
        .lines()
        .skip(1)
    {
        let fields: Vec<_> = row.split(',').collect();
        let params = SisParameters::try_new(
            fields[0].parse().unwrap(),
            akita_q32(),
            Some(fields[1].parse().unwrap()),
            Bound::from_u64(fields[2].parse().unwrap()),
            SisNorm::Infinity,
        )
        .unwrap();
        let beta = fields[3].parse().unwrap();
        let probability: f64 = fields[5].parse().unwrap();
        let repetitions: f64 = fields[6].parse().unwrap();
        let quantum_cost: f64 = fields[7].parse().unwrap();
        for mode in [Adps16Mode::Quantum, Adps16Mode::Classical] {
            let config = EstimateConfig {
                red_cost_model: ReductionCostModel::Adps16 { mode },
                ..EstimateConfig::default()
            };
            let cost = cost_infinity(beta, &params, 0, &config).unwrap();
            let actual_probability = cost.prob.unwrap().get();
            assert!(
                (actual_probability / probability - 1.0).abs() < 1e-7,
                "{row}: {actual_probability}"
            );
            if repetitions <= 6.0 {
                assert_eq!(cost.repetitions.unwrap().log2(), Some(repetitions.log2()));
            }
            let expected_cost = quantum_cost
                + if mode == Adps16Mode::Classical {
                    0.027 * f64::from(beta)
                } else {
                    0.0
                };
            assert!(
                (cost.rop.log2().unwrap() - expected_cost).abs() < 1e-7,
                "{row}: {:?}",
                cost.rop
            );
            if mode == Adps16Mode::Quantum {
                assert_eq!(cost.rop.log2().unwrap() >= 128.0, quantum_cost >= 128.0);
            }
        }
    }
}
