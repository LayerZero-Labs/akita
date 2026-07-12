use akita_sis_estimator::{
    estimate, scalar_sis_from_ring, AkitaModulusFamily, CostValue, EstimateConfig,
    SisSecurityPolicy,
};

#[test]
fn policy_models_are_independently_optimized_on_a_representative_row() {
    let policy = SisSecurityPolicy::Classical138Quantum128WithIdealizedBcssV1;
    let models = [
        policy.classical_constraint().reduction_model,
        policy.conventional_quantum_constraint().reduction_model,
        policy.idealized_bcss_diagnostic().reduction_model,
    ];
    let params = scalar_sis_from_ring(AkitaModulusFamily::Q32, 32, 2, 12, 15).unwrap();
    let costs = models.map(|red_cost_model| {
        estimate(
            &params,
            &EstimateConfig {
                red_cost_model,
                ..EstimateConfig::lattice_estimator_parity()
            },
        )
        .unwrap()
    });

    let scores = costs
        .iter()
        .map(|cost| match cost.rop {
            CostValue::Finite(cost) => cost.log2,
            CostValue::Infinity => panic!("representative row unexpectedly had infinite cost"),
        })
        .collect::<Vec<_>>();
    assert!((scores[0] - 150.143_670_698_131_4).abs() < 1e-9);
    assert!((scores[1] - 139.802_670_698_131_38).abs() < 1e-9);
    assert!((scores[2] - 136.470_570_698_131_38).abs() < 1e-9);
    assert!(costs.iter().all(|cost| cost.beta == Some(383)));
    assert!(costs.iter().all(|cost| cost.zeta == Some(0)));

    // The full cost includes amplification, so scaling the classical optimum
    // by an exponent ratio is observably not the independently optimized score.
    assert!((scores[1] - scores[0] * 0.2650 / 0.2920).abs() > 1.0);
    assert!((scores[2] - scores[0] * 0.2563 / 0.2920).abs() > 1.0);
}
