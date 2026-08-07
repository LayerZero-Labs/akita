use super::SumcheckKernelPlan;
use akita_field::{Ext2, ExtField, FpExt4, Prime32Offset99, Prime64Offset59};
use akita_sumcheck::{
    batched_affine_product_coefficients, compose_polynomial_with_affine, EvaluationTable,
};
use akita_types::TensorFactorProjection;

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

fn compare_tensor_factor_round_plan(plan: SumcheckKernelPlan) {
    let projection = TensorFactorProjection::<F, E>::new(&[value(701), value(702)])
        .expect("valid quartic projection");
    for num_vars in [1usize, 4, 7, 10, 12] {
        let len = 1usize << num_vars;
        let witness =
            EvaluationTable::<F, E>::from_multilinear_evaluation_fn(len, |row| value(row + 811))
                .expect("valid witness table");
        let tail_point = (0..num_vars)
            .map(|coordinate| value(coordinate + 907))
            .collect::<Vec<_>>();
        let expected = SumcheckKernelPlan::SCALAR
            .materialize_tensor_factor_and_compute_product_round_fp32(
                &witness,
                &tail_point,
                &projection,
            )
            .expect("valid scalar tensor factor");
        let actual = plan
            .materialize_tensor_factor_and_compute_product_round_fp32(
                &witness,
                &tail_point,
                &projection,
            )
            .expect("valid packed tensor factor");
        assert_eq!(actual, expected, "tensor factor mismatch at len={len}");
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

fn direct_range_polynomial(degree: usize) -> Vec<E> {
    match degree {
        2 => vec![E::zero(), E::zero() - E::from_u64(2), E::from_u64(1)],
        4 => vec![
            E::zero(),
            E::zero() - E::from_u64(144),
            E::from_u64(108),
            E::zero() - E::from_u64(20),
            E::from_u64(1),
        ],
        _ => unreachable!(),
    }
}

fn compare_class_coded_affine_polynomial_plan(plan: SumcheckKernelPlan) {
    let class_values = (0..64).map(|row| value(row + 5_001)).collect::<Vec<_>>();
    let class_codes = (0..272)
        .map(|index| u16::try_from((index * 29 + 7) % class_values.len()).unwrap())
        .collect::<Vec<_>>();
    let first_equality = (0..32).map(|row| value(row + 6_001)).collect::<Vec<_>>();
    let second_equality = (0..8).map(|row| value(row + 7_001)).collect::<Vec<_>>();

    for degree in [2, 4] {
        let polynomial = direct_range_polynomial(degree);
        let class_taylor_coefficients = class_values
            .iter()
            .map(|&class_value| {
                let coefficients =
                    compose_polynomial_with_affine(&polynomial, class_value, E::from_u64(1));
                std::array::from_fn(|index| coefficients[index])
            })
            .collect::<Vec<_>>();
        let mut expected = [E::zero(); 5];
        for pair in 0..class_codes.len() / 2 {
            let left = class_values[usize::from(class_codes[2 * pair])];
            let right = class_values[usize::from(class_codes[2 * pair + 1])];
            let coefficients = compose_polynomial_with_affine(&polynomial, left, right - left);
            let equality = first_equality[pair % first_equality.len()]
                * second_equality[pair / first_equality.len()];
            for coefficient in 0..=degree {
                expected[coefficient] += equality * coefficients[coefficient];
            }
        }
        if let Some(actual) = plan.try_compute_class_coded_affine_polynomial_round_fp32(
            &class_codes,
            &class_values,
            &class_taylor_coefficients,
            &first_equality,
            &second_equality,
            degree,
        ) {
            assert_eq!(actual, expected, "class-coded degree {degree}");
        }
    }
}

fn expected_sparse_affine_polynomial_fold(
    values: &[E],
    first_equality: &[E],
    second_equality: &[E],
    challenge: E,
    degree: usize,
) -> (Vec<E>, [E; 5]) {
    let polynomial = direct_range_polynomial(degree);
    let mut folded_values = vec![E::zero(); values.len() / 2];
    let mut coefficients_sum = [E::zero(); 5];
    for pair in 0..values.len() / 4 {
        let left = values[4 * pair] + challenge * (values[4 * pair + 1] - values[4 * pair]);
        let right =
            values[4 * pair + 2] + challenge * (values[4 * pair + 3] - values[4 * pair + 2]);
        folded_values[2 * pair] = left;
        folded_values[2 * pair + 1] = right;
        let coefficients = compose_polynomial_with_affine(&polynomial, left, right - left);
        let equality = first_equality[pair % first_equality.len()]
            * second_equality[pair / first_equality.len()];
        for coefficient in 0..=degree {
            coefficients_sum[coefficient] += equality * coefficients[coefficient];
        }
    }
    (folded_values, coefficients_sum)
}

fn compare_sparse_affine_polynomial_fold_plan(plan: SumcheckKernelPlan) {
    let values = (0..544).map(|row| value(row + 8_001)).collect::<Vec<_>>();
    let class_values = (0..64).map(|row| value(row + 12_001)).collect::<Vec<_>>();
    let class_codes = (0..544)
        .map(|index| u16::try_from((index * 31 + 9) % class_values.len()).unwrap())
        .collect::<Vec<_>>();
    let class_coded_values = class_codes
        .iter()
        .map(|&class| class_values[usize::from(class)])
        .collect::<Vec<_>>();
    let first_equality = (0..32).map(|row| value(row + 9_001)).collect::<Vec<_>>();
    let second_equality = (0..8).map(|row| value(row + 10_001)).collect::<Vec<_>>();
    let challenge = value(11_001);

    for degree in [2, 4] {
        let (expected_values, expected) = expected_sparse_affine_polynomial_fold(
            &values,
            &first_equality,
            &second_equality,
            challenge,
            degree,
        );
        let mut actual_values = vec![E::zero(); values.len() / 2];
        if let Some(actual) = plan.try_fold_and_compute_sparse_affine_polynomial_round_fp32(
            &values,
            &mut actual_values,
            &first_equality,
            &second_equality,
            challenge,
            degree,
        ) {
            assert_eq!(actual_values, expected_values, "folded degree {degree}");
            assert_eq!(actual, expected, "fused sparse degree {degree}");
        }

        let (expected_values, expected) = expected_sparse_affine_polynomial_fold(
            &class_coded_values,
            &first_equality,
            &second_equality,
            challenge,
            degree,
        );
        let mut actual_values = vec![E::zero(); class_codes.len() / 2];
        if let Some(actual) = plan
            .try_fold_class_coded_and_compute_sparse_affine_polynomial_round_fp32(
                &class_codes,
                &class_values,
                &mut actual_values,
                (&first_equality, &second_equality),
                challenge,
                degree,
            )
        {
            assert_eq!(actual_values, expected_values, "class fold degree {degree}");
            assert_eq!(actual, expected, "class fused sparse degree {degree}");
        }
    }
}

fn compare_stage2_coefficient_round_plan(plan: SumcheckKernelPlan) {
    for old_coefficient_count in [4usize, 8, 16] {
        for live_lane_count in [1usize, 3, 7] {
            let len = live_lane_count * old_coefficient_count;
            let witness = EvaluationTable::<F, E>::from_evaluation_fn(len, value);
            let next_coefficient_count = old_coefficient_count / 2;
            let pair_count = live_lane_count * (next_coefficient_count / 2);
            let first_equality_len = pair_count.next_power_of_two().min(4);
            let second_equality_len = pair_count.div_ceil(first_equality_len).next_power_of_two();
            let next_alpha_factor = (0..next_coefficient_count)
                .map(|row| value(len + row + 17))
                .collect::<Vec<_>>();
            let relation_lane_weights = (0..live_lane_count.next_power_of_two())
                .map(|row| value(len + row + 41))
                .collect::<Vec<_>>();
            let first_equality = (0..first_equality_len)
                .map(|row| value(len + row + 73))
                .collect::<Vec<_>>();
            let second_equality = (0..second_equality_len)
                .map(|row| value(len + row + 101))
                .collect::<Vec<_>>();
            let challenge = value(len + 131);

            for include_norm_linear in [false, true] {
                let mut expected_witness = witness.clone();
                let expected = SumcheckKernelPlan::SCALAR
                    .fold_and_compute_stage2_coefficient_round_fp32(
                        &mut expected_witness,
                        live_lane_count,
                        old_coefficient_count,
                        &next_alpha_factor,
                        &relation_lane_weights,
                        &first_equality,
                        &second_equality,
                        challenge,
                        include_norm_linear,
                    );
                let mut actual_witness = witness.clone();
                let actual = plan.fold_and_compute_stage2_coefficient_round_fp32(
                    &mut actual_witness,
                    live_lane_count,
                    old_coefficient_count,
                    &next_alpha_factor,
                    &relation_lane_weights,
                    &first_equality,
                    &second_equality,
                    challenge,
                    include_norm_linear,
                );
                assert_eq!(actual_witness, expected_witness);
                assert_eq!(actual, expected);
            }
        }
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
    compare_tensor_factor_round_plan(SumcheckKernelPlan::detect());
    compare_weighted_affine_product_plan::<2>(SumcheckKernelPlan::detect(), 2);
    compare_weighted_affine_product_plan::<4>(SumcheckKernelPlan::detect(), 4);
    compare_weighted_affine_product_plan::<8>(SumcheckKernelPlan::detect(), 4);
    compare_weighted_affine_polynomial_plan(SumcheckKernelPlan::detect());
    compare_compact_affine_product_plan::<2>(SumcheckKernelPlan::detect(), 2);
    compare_compact_affine_product_plan::<4>(SumcheckKernelPlan::detect(), 4);
    compare_compact_affine_product_plan::<8>(SumcheckKernelPlan::detect(), 4);
    compare_class_coded_affine_polynomial_plan(SumcheckKernelPlan::detect());
    compare_sparse_affine_polynomial_fold_plan(SumcheckKernelPlan::detect());
    compare_stage2_coefficient_round_plan(SumcheckKernelPlan::detect());
    compare_fp64_plan(SumcheckKernelPlan::detect());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn supported_x86_fp32_folds_match_scalar() {
    if std::is_x86_feature_detected!("avx2") {
        compare_plan(SumcheckKernelPlan::AVX2);
        compare_product_round_plan(SumcheckKernelPlan::AVX2);
        compare_fused_product_round_plan(SumcheckKernelPlan::AVX2);
        compare_tensor_factor_round_plan(SumcheckKernelPlan::AVX2);
        compare_weighted_affine_product_plan::<2>(SumcheckKernelPlan::AVX2, 2);
        compare_weighted_affine_product_plan::<4>(SumcheckKernelPlan::AVX2, 4);
        compare_weighted_affine_product_plan::<8>(SumcheckKernelPlan::AVX2, 4);
        compare_weighted_affine_polynomial_plan(SumcheckKernelPlan::AVX2);
        compare_compact_affine_product_plan::<2>(SumcheckKernelPlan::AVX2, 2);
        compare_compact_affine_product_plan::<4>(SumcheckKernelPlan::AVX2, 4);
        compare_compact_affine_product_plan::<8>(SumcheckKernelPlan::AVX2, 4);
        compare_class_coded_affine_polynomial_plan(SumcheckKernelPlan::AVX2);
        compare_sparse_affine_polynomial_fold_plan(SumcheckKernelPlan::AVX2);
        compare_stage2_coefficient_round_plan(SumcheckKernelPlan::AVX2);
        compare_fp64_plan(SumcheckKernelPlan::AVX2);
    }
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512ifma") {
        compare_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_product_round_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_fused_product_round_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_tensor_factor_round_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_weighted_affine_product_plan::<2>(SumcheckKernelPlan::AVX512_IFMA, 2);
        compare_weighted_affine_product_plan::<4>(SumcheckKernelPlan::AVX512_IFMA, 4);
        compare_weighted_affine_product_plan::<8>(SumcheckKernelPlan::AVX512_IFMA, 4);
        compare_weighted_affine_polynomial_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_compact_affine_product_plan::<2>(SumcheckKernelPlan::AVX512_IFMA, 2);
        compare_compact_affine_product_plan::<4>(SumcheckKernelPlan::AVX512_IFMA, 4);
        compare_compact_affine_product_plan::<8>(SumcheckKernelPlan::AVX512_IFMA, 4);
        compare_class_coded_affine_polynomial_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_sparse_affine_polynomial_fold_plan(SumcheckKernelPlan::AVX512_IFMA);
        compare_stage2_coefficient_round_plan(SumcheckKernelPlan::AVX512_IFMA);
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
        compare_tensor_factor_round_plan(SumcheckKernelPlan::NEON);
        compare_weighted_affine_product_plan::<2>(SumcheckKernelPlan::NEON, 2);
        compare_weighted_affine_product_plan::<4>(SumcheckKernelPlan::NEON, 4);
        compare_weighted_affine_product_plan::<8>(SumcheckKernelPlan::NEON, 4);
        compare_weighted_affine_polynomial_plan(SumcheckKernelPlan::NEON);
        compare_compact_affine_product_plan::<2>(SumcheckKernelPlan::NEON, 2);
        compare_compact_affine_product_plan::<4>(SumcheckKernelPlan::NEON, 4);
        compare_compact_affine_product_plan::<8>(SumcheckKernelPlan::NEON, 4);
        compare_class_coded_affine_polynomial_plan(SumcheckKernelPlan::NEON);
        compare_sparse_affine_polynomial_fold_plan(SumcheckKernelPlan::NEON);
        compare_stage2_coefficient_round_plan(SumcheckKernelPlan::NEON);
        compare_fp64_plan(SumcheckKernelPlan::NEON);
    }
}
