//! The stage-1 leaf and stage-2 oracle checks in the clear domain: honest
//! values pass, a shifted claim or input fails.

use super::{check_range_leaf, check_stage2_oracle, RangeLeaf, Stage2OracleWeights};
use akita_error::AkitaError;
use akita_params::DigitRangePlan;
use jolt_algebra::{One, Prime128OffsetA7F7, Ring};
use jolt_transcript::{ClaimMismatch, ClearDomain};

type F = Prime128OffsetA7F7;

#[test]
fn range_leaf_check_accepts_honest_values_and_rejects_shifted_ones() {
    for basis in [4, 8, 16, 32, 64] {
        let plan = DigitRangePlan::new(basis).unwrap();
        let leaf_coeffs = plan.leaf_coeffs::<F>();
        let batching_weights = (1..=leaf_coeffs.len() as u64)
            .map(|index| F::from_u64(0x9e37_79b9 * index))
            .collect::<Vec<_>>();
        let coefficients = plan
            .batch_leaf_polynomials(&batching_weights, &leaf_coeffs)
            .unwrap();
        let leaf = RangeLeaf::new(plan, &coefficients).unwrap();
        let range_image = F::from_u64(0x1234_5678_9abc_def1);
        let claim = plan.evaluate_leaf_polynomial(&coefficients, range_image);

        assert_eq!(
            check_range_leaf(&mut ClearDomain, &leaf, range_image, claim),
            Ok(()),
            "basis {basis}"
        );
        assert_eq!(
            check_range_leaf(&mut ClearDomain, &leaf, range_image, claim + F::one()),
            Err(ClaimMismatch),
            "basis {basis}"
        );
        assert_eq!(
            check_range_leaf(&mut ClearDomain, &leaf, range_image + F::one(), claim),
            Err(ClaimMismatch),
            "basis {basis}"
        );
    }
}

#[test]
fn range_leaf_rejects_coefficients_above_the_plan_degree() {
    let plan = DigitRangePlan::new(4).unwrap();
    let coefficients = vec![F::one(); plan.leaf_degree() + 2];
    assert!(matches!(
        RangeLeaf::new(plan, &coefficients),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn stage2_oracle_check_accepts_honest_values_and_rejects_a_shifted_claim() {
    let weights = Stage2OracleWeights {
        binary: F::from_u64(0x0bad_cafe_f00d),
        linear: F::from_u64(0x00de_ad00_beef),
    };
    let witness_eval = F::from_u64(0x7777_1111_3333);
    // The oracle term by term: B·w(w+1) collects the virtual and binary
    // terms, R·w the linear ones.
    let claim =
        weights.binary * witness_eval * (witness_eval + F::one()) + weights.linear * witness_eval;
    assert_eq!(
        check_stage2_oracle(&mut ClearDomain, &weights, witness_eval, claim),
        Ok(())
    );
    assert_eq!(
        check_stage2_oracle(&mut ClearDomain, &weights, witness_eval, claim + F::one()),
        Err(ClaimMismatch)
    );
}
