use super::SumcheckKernelPlan;
use akita_field::{Ext2, ExtField, FpExt4, Prime32Offset99, Prime64Offset59};
use akita_sumcheck::{batched_affine_product_coefficients, EvaluationTable};

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

fn compare_weighted_affine_product_plan<const LANES: usize>(
    plan: SumcheckKernelPlan,
    arity: usize,
) {
    for len in [2, 4, 8, 16, 32, 64, 256] {
        let lanes: [EvaluationTable<F, E>; LANES] = std::array::from_fn(|lane| {
            EvaluationTable::<F, E>::from_multilinear_evaluation_fn(len, |row| {
                value(row + lane * len + 7)
            })
            .expect("valid lane table")
        });
        let equality = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(len / 2, |row| {
            value(row + LANES * len + 19)
        })
        .expect("valid equality table");
        let parent_weights = (0..LANES / arity)
            .map(|parent| value(parent + 101))
            .collect::<Vec<_>>();
        let expected = SumcheckKernelPlan::SCALAR.compute_weighted_affine_product_round_fp32(
            &lanes,
            &equality,
            arity,
            &parent_weights,
        );
        let actual = plan.compute_weighted_affine_product_round_fp32(
            &lanes,
            &equality,
            arity,
            &parent_weights,
        );
        assert_eq!(actual, expected, "weighted product mismatch at len={len}");
    }
}

fn compare_weighted_affine_polynomial_plan(plan: SumcheckKernelPlan) {
    for len in [2, 4, 8, 16, 32, 64, 256] {
        let values = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(len, value)
            .expect("valid value table");
        let equality = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(len / 2, |row| {
            value(row + len + 37)
        })
        .expect("valid equality table");
        for degree in 0..=4 {
            let coefficients = (0..=degree)
                .map(|coefficient| value(coefficient + 503))
                .collect::<Vec<_>>();
            let expected = SumcheckKernelPlan::SCALAR
                .compute_weighted_affine_polynomial_round_fp32(&values, &equality, &coefficients);
            let actual = plan.compute_weighted_affine_polynomial_round_fp32(
                &values,
                &equality,
                &coefficients,
            );
            assert_eq!(
                actual, expected,
                "weighted polynomial mismatch at len={len}, degree={degree}"
            );
        }
    }
}

fn compare_compact_affine_product_plan<const LANES: usize>(plan: SumcheckKernelPlan, arity: usize) {
    let rows: Vec<[E; LANES]> = (0..64)
        .map(|row| std::array::from_fn(|lane| value(row * LANES + lane + 7)))
        .collect();
    let ordered_pair_indices = (0..128)
        .map(|index| u16::try_from((index * 17 + 3) % rows.len()).unwrap())
        .collect::<Vec<_>>();
    let first_equality = (0..32).map(|row| value(row + 2_001)).collect::<Vec<_>>();
    let second_equality = (0..2).map(|row| value(row + 3_001)).collect::<Vec<_>>();
    let parent_weights = (0..LANES / arity)
        .map(|parent| value(parent + 4_001))
        .collect::<Vec<_>>();
    let mut expected = [E::zero(); 5];
    for quartet in 0..ordered_pair_indices.len() / 2 {
        let left = rows[usize::from(ordered_pair_indices[2 * quartet])];
        let right = rows[usize::from(ordered_pair_indices[2 * quartet + 1])];
        let coefficients =
            batched_affine_product_coefficients(&left, &right, arity, &parent_weights);
        let weight = first_equality[quartet % first_equality.len()]
            * second_equality[quartet / first_equality.len()];
        for degree in 0..=arity {
            expected[degree] += weight * coefficients[degree];
        }
    }
    if let Some(actual) = plan.try_compute_compact_affine_product_round_fp32(
        &ordered_pair_indices,
        &rows,
        &first_equality,
        &second_equality,
        arity,
        &parent_weights,
    ) {
        assert_eq!(actual, expected);
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
    compare_weighted_affine_product_plan::<2>(SumcheckKernelPlan::detect(), 2);
    compare_weighted_affine_product_plan::<4>(SumcheckKernelPlan::detect(), 4);
    compare_weighted_affine_product_plan::<8>(SumcheckKernelPlan::detect(), 4);
    compare_weighted_affine_polynomial_plan(SumcheckKernelPlan::detect());
    compare_compact_affine_product_plan::<2>(SumcheckKernelPlan::detect(), 2);
    compare_compact_affine_product_plan::<4>(SumcheckKernelPlan::detect(), 4);
    compare_compact_affine_product_plan::<8>(SumcheckKernelPlan::detect(), 4);
    compare_fp64_plan(SumcheckKernelPlan::detect());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn supported_x86_fp32_folds_match_scalar() {
    if std::is_x86_feature_detected!("avx2") {
        compare_plan(SumcheckKernelPlan::AVX2);
        compare_product_round_plan(SumcheckKernelPlan::AVX2);
        compare_fused_product_round_plan(SumcheckKernelPlan::AVX2);
        compare_weighted_affine_product_plan::<2>(SumcheckKernelPlan::AVX2, 2);
        compare_weighted_affine_product_plan::<4>(SumcheckKernelPlan::AVX2, 4);
        compare_weighted_affine_product_plan::<8>(SumcheckKernelPlan::AVX2, 4);
        compare_weighted_affine_polynomial_plan(SumcheckKernelPlan::AVX2);
        compare_compact_affine_product_plan::<2>(SumcheckKernelPlan::AVX2, 2);
        compare_compact_affine_product_plan::<4>(SumcheckKernelPlan::AVX2, 4);
        compare_compact_affine_product_plan::<8>(SumcheckKernelPlan::AVX2, 4);
        compare_fp64_plan(SumcheckKernelPlan::AVX2);
    }
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512ifma") {
        compare_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_product_round_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_fused_product_round_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_weighted_affine_product_plan::<2>(SumcheckKernelPlan::AVX512_IFMA, 2);
        compare_weighted_affine_product_plan::<4>(SumcheckKernelPlan::AVX512_IFMA, 4);
        compare_weighted_affine_product_plan::<8>(SumcheckKernelPlan::AVX512_IFMA, 4);
        compare_weighted_affine_polynomial_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_compact_affine_product_plan::<2>(SumcheckKernelPlan::AVX512_IFMA, 2);
        compare_compact_affine_product_plan::<4>(SumcheckKernelPlan::AVX512_IFMA, 4);
        compare_compact_affine_product_plan::<8>(SumcheckKernelPlan::AVX512_IFMA, 4);
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
        compare_weighted_affine_product_plan::<2>(SumcheckKernelPlan::NEON, 2);
        compare_weighted_affine_product_plan::<4>(SumcheckKernelPlan::NEON, 4);
        compare_weighted_affine_product_plan::<8>(SumcheckKernelPlan::NEON, 4);
        compare_weighted_affine_polynomial_plan(SumcheckKernelPlan::NEON);
        compare_compact_affine_product_plan::<2>(SumcheckKernelPlan::NEON, 2);
        compare_compact_affine_product_plan::<4>(SumcheckKernelPlan::NEON, 4);
        compare_compact_affine_product_plan::<8>(SumcheckKernelPlan::NEON, 4);
        compare_fp64_plan(SumcheckKernelPlan::NEON);
    }
}
