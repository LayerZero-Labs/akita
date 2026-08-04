use akita_sis_estimator::{
    estimate, scalar_sis_from_ring, Adps16Mode, AkitaModulusProfileId, CostValue, EstimateConfig,
    ReductionCostModel, SisSecurityPolicy,
};

#[test]
fn policy_uses_only_the_adps16_quantum_model() {
    let policy = SisSecurityPolicy::Quantum128BitADPS16;
    assert_eq!(
        policy.adps16_quantum_constraint().reduction_model,
        ReductionCostModel::Adps16 {
            mode: Adps16Mode::Quantum,
        }
    );
    let params = scalar_sis_from_ring(AkitaModulusProfileId::Q32Offset99, 32, 2, 12, 15).unwrap();
    let cost = estimate(&params, &EstimateConfig::akita_infinity_table()).unwrap();
    assert!(matches!(cost.rop, CostValue::Finite(value) if value.log2.is_finite()));
}
