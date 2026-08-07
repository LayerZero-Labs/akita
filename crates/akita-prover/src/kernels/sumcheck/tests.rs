use super::SumcheckKernelPlan;
use akita_field::{Ext2, ExtField, FpExt4, Prime32Offset99, Prime64Offset59};
use akita_sumcheck::EvaluationTable;

type F = Prime32Offset99;
type E = FpExt4<F>;
type F64 = Prime64Offset59;
type E64 = Ext2<F64>;

fn value(row: usize) -> E {
    E::from_base_fn(|coefficient| F::from_u64((row as u64 + 3) * (coefficient as u64 + 11)))
}

fn value64(row: usize) -> E64 {
    E64::from_base_fn(|coefficient| F64::from_u64((row as u64 + 5) * (coefficient as u64 + 13)))
}

fn compare_plan(plan: SumcheckKernelPlan) {
    for len in [2, 4, 8, 16, 32, 64, 256] {
        let source: Vec<_> = (0..len).map(value).collect();
        let challenge = value(len + 17);
        let mut expected = EvaluationTable::<F, E>::from_multilinear_evaluations(&source)
            .expect("valid table length");
        let mut actual = expected.clone();
        SumcheckKernelPlan::SCALAR.fold_first_variable_fp32(&mut expected, challenge);
        plan.fold_first_variable_fp32(&mut actual, challenge);
        assert_eq!(actual, expected, "fold mismatch at len={len}");
    }
}

fn compare_product_round_plan(plan: SumcheckKernelPlan) {
    for len in [2, 4, 8, 16, 32, 64, 256] {
        let witness = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(len, value)
            .expect("valid witness table");
        let factor = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(len, |row| {
            value(row + len + 31)
        })
        .expect("valid factor table");
        let expected = SumcheckKernelPlan::SCALAR.compute_product_round_fp32(&witness, &factor);
        let actual = plan.compute_product_round_fp32(&witness, &factor);
        assert_eq!(actual, expected, "product round mismatch at len={len}");
    }
}

fn compare_fused_product_round_plan(plan: SumcheckKernelPlan) {
    for len in [4, 8, 16, 32, 64, 256] {
        let witness = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(len, value)
            .expect("valid witness table");
        let factor = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(len, |row| {
            value(row + len + 31)
        })
        .expect("valid factor table");
        let challenge = value(len + 73);
        let mut expected_witness = witness.clone();
        let mut expected_factor = factor.clone();
        let expected = SumcheckKernelPlan::SCALAR.fold_and_compute_product_round_fp32(
            &mut expected_witness,
            &mut expected_factor,
            challenge,
        );
        let mut actual_witness = witness;
        let mut actual_factor = factor;
        let actual = plan.fold_and_compute_product_round_fp32(
            &mut actual_witness,
            &mut actual_factor,
            challenge,
        );
        assert_eq!(actual, expected, "fused round mismatch at len={len}");
        assert_eq!(
            actual_witness, expected_witness,
            "fused witness mismatch at len={len}"
        );
        assert_eq!(
            actual_factor, expected_factor,
            "fused factor mismatch at len={len}"
        );
    }
}

fn compare_fp64_plan(plan: SumcheckKernelPlan) {
    for len in [2, 4, 8, 16, 32, 64, 256] {
        let source: Vec<_> = (0..len).map(value64).collect();
        let challenge = value64(len + 17);
        let mut expected = EvaluationTable::<F64, E64>::from_multilinear_evaluations(&source)
            .expect("valid table length");
        let mut actual = expected.clone();
        SumcheckKernelPlan::SCALAR.fold_first_variable_fp64(&mut expected, challenge);
        plan.fold_first_variable_fp64(&mut actual, challenge);
        assert_eq!(actual, expected, "fp64 fold mismatch at len={len}");

        let witness = EvaluationTable::<F64, E64>::from_multilinear_evaluation_fn(len, value64)
            .expect("valid witness table");
        let factor = EvaluationTable::<F64, E64>::from_multilinear_evaluation_fn(len, |row| {
            value64(row + len + 31)
        })
        .expect("valid factor table");
        let expected_round =
            SumcheckKernelPlan::SCALAR.compute_product_round_fp64(&witness, &factor);
        let actual_round = plan.compute_product_round_fp64(&witness, &factor);
        assert_eq!(
            actual_round, expected_round,
            "fp64 product round mismatch at len={len}"
        );

        if len >= 4 {
            let mut expected_witness = witness.clone();
            let mut expected_factor = factor.clone();
            let expected_round = SumcheckKernelPlan::SCALAR.fold_and_compute_product_round_fp64(
                &mut expected_witness,
                &mut expected_factor,
                challenge,
            );
            let mut actual_witness = witness;
            let mut actual_factor = factor;
            let actual_round = plan.fold_and_compute_product_round_fp64(
                &mut actual_witness,
                &mut actual_factor,
                challenge,
            );
            assert_eq!(
                actual_round, expected_round,
                "fp64 fused round mismatch at len={len}"
            );
            assert_eq!(actual_witness, expected_witness);
            assert_eq!(actual_factor, expected_factor);
        }
    }
}

#[test]
fn detected_fp32_fold_matches_scalar() {
    compare_plan(SumcheckKernelPlan::detect());
    compare_product_round_plan(SumcheckKernelPlan::detect());
    compare_fused_product_round_plan(SumcheckKernelPlan::detect());
    compare_fp64_plan(SumcheckKernelPlan::detect());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn supported_x86_fp32_folds_match_scalar() {
    if std::is_x86_feature_detected!("avx2") {
        compare_plan(SumcheckKernelPlan::AVX2);
        compare_product_round_plan(SumcheckKernelPlan::AVX2);
        compare_fused_product_round_plan(SumcheckKernelPlan::AVX2);
        compare_fp64_plan(SumcheckKernelPlan::AVX2);
    }
    if std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512ifma")
    {
        compare_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_product_round_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_fused_product_round_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_fp64_plan(SumcheckKernelPlan::AVX512_IFMA);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn supported_neon_fp32_folds_match_scalar() {
    if std::arch::is_aarch64_feature_detected!("neon") {
        compare_plan(SumcheckKernelPlan::NEON);
        compare_product_round_plan(SumcheckKernelPlan::NEON);
        compare_fused_product_round_plan(SumcheckKernelPlan::NEON);
        compare_fp64_plan(SumcheckKernelPlan::NEON);
    }
}
