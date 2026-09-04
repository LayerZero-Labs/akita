use super::*;
use crate::EqPolynomial;
use jolt_field::{Fp64, Ring, Zero};

type F = Fp64<4294967197>;

fn matrix_from_entries(entries: &[Vec<i8>]) -> TernaryProjectionMatrix {
    let rows = entries.len();
    let cols = entries.first().unwrap().len();
    let shape = TernaryProjectionShape::new(rows, cols).unwrap();
    let mut first = vec![0u8; shape.plane_len()];
    let mut second = vec![0u8; shape.plane_len()];
    for (row, values) in entries.iter().enumerate() {
        assert_eq!(values.len(), cols);
        for (col, &value) in values.iter().enumerate() {
            let index = (col >> 2) * shape.row_pairs() + (row >> 1);
            let bit = 1u8 << (((row & 1) << 2) | (col & 3));
            match value {
                -1 => {}
                0 => first[index] |= bit,
                1 => {
                    first[index] |= bit;
                    second[index] |= bit;
                }
                _ => panic!("test matrix entry is not ternary"),
            }
        }
    }
    TernaryProjectionMatrix::from_rademacher_bitplanes(shape, first, second).unwrap()
}

fn fixture() -> TernaryProjectionMatrix {
    matrix_from_entries(&[
        vec![1, 0, -1, 1, 0],
        vec![-1, -1, 0, 0, 1],
        vec![0, 1, 1, -1, -1],
    ])
}

fn field_from_i128(value: i128) -> F {
    F::from_i128(value)
}

#[test]
fn shape_checks_empty_overflow_and_materialization_budget() {
    assert!(TernaryProjectionShape::new(0, 1).is_err());
    assert!(TernaryProjectionShape::new(1, 0).is_err());
    assert!(TernaryProjectionShape::new(usize::MAX, usize::MAX).is_err());
    assert!(TernaryProjectionShape::new(1 << 20, 1 << 20).is_err());

    let shape = TernaryProjectionShape::new(3, 9).unwrap();
    assert_eq!(shape.col_groups(), 3);
    assert_eq!(shape.row_pairs(), 2);
    assert_eq!(shape.plane_len(), 6);
    assert_eq!(shape.packed_len(), 12);
    assert_eq!(shape.dense_len(), 27);
    assert_eq!(shape.materialized_len(), 39);
    assert_eq!(shape.row_num_vars().unwrap(), 2);
    assert_eq!(shape.col_num_vars().unwrap(), 4);

    let oversized_u128_len = MAX_MATERIALIZED_JL_BYTES / std::mem::size_of::<u128>() + 1;
    assert!(try_zeroed_vec(oversized_u128_len, 0u128).is_err());
}

#[test]
fn rademacher_bitplanes_are_canonical_and_decode_balanced_ternary() {
    let matrix = fixture();
    let expected = [[1, 0, -1, 1, 0], [-1, -1, 0, 0, 1], [0, 1, 1, -1, -1]];
    for (row, entries) in expected.iter().enumerate() {
        for (col, &entry) in entries.iter().enumerate() {
            assert_eq!(matrix.entry(row, col).unwrap(), entry);
        }
    }
    assert!(matrix.entry(3, 0).is_err());
    assert!(matrix.entry(0, 5).is_err());

    let shape = TernaryProjectionShape::new(1, 5).unwrap();
    assert!(
        TernaryProjectionMatrix::from_rademacher_bitplanes(shape, vec![0xf0, 0], vec![0, 0],)
            .is_err()
    );
    assert!(
        TernaryProjectionMatrix::from_rademacher_bitplanes(shape, vec![0, 0b0010], vec![0, 0],)
            .is_err()
    );
}

#[test]
fn dense_compute_plane_is_lazy_cached_and_not_matrix_identity() {
    let matrix = fixture();
    let canonical_clone = matrix.clone();
    assert!(matrix.dense.get().is_none());
    assert_eq!(matrix, canonical_clone);

    let first = matrix.dense_rows().unwrap().as_ptr();
    let second = matrix.dense_rows().unwrap().as_ptr();
    assert_eq!(first, second);
    assert!(matrix.dense.get().is_some());
    assert_eq!(matrix, canonical_clone);
    let materialized_clone = matrix.clone();
    assert!(materialized_clone.dense.get().is_none());
    assert_eq!(matrix, materialized_clone);
}

#[test]
fn integer_and_field_projection_match_dense_reference() {
    let matrix = fixture();
    let input = [2, -3, 5, 7, 11];
    let expected = [4i128, 12, -16];
    let expected_i64 = [4i64, 12, -16];
    assert_eq!(matrix.project_i128(&input).unwrap(), expected);
    assert_eq!(
        matrix.project(&input.map(|value| value as i8)).unwrap(),
        expected_i64
    );
    assert_eq!(
        matrix.project(&input.map(|value| value as i16)).unwrap(),
        expected_i64
    );
    assert_eq!(
        matrix.project(&input.map(|value| value as i32)).unwrap(),
        expected_i64
    );
    assert_eq!(
        matrix.project(&input.map(|value| value as i64)).unwrap(),
        expected_i64
    );

    let field_input = input.map(field_from_i128);
    let field_expected = expected.map(field_from_i128);
    assert_eq!(matrix.project_field(&field_input).unwrap(), field_expected);
    assert!(matrix.project_i128(&input[..4]).is_err());

    let overflow_matrix = matrix_from_entries(&[vec![1, 1]]);
    assert!(overflow_matrix.project_i128(&[i128::MAX, 1]).is_err());
    assert!(overflow_matrix.project(&[i64::MAX, 1]).is_err());

    let cancellation_matrix = matrix_from_entries(&[vec![1, -1]]);
    assert_eq!(
        cancellation_matrix.project(&[i64::MAX, 1]).unwrap(),
        [i64::MAX - 1]
    );
    let intermediate_overflow_matrix = matrix_from_entries(&[vec![1, 1, -1]]);
    assert_eq!(
        intermediate_overflow_matrix
            .project(&[i64::MAX, 1, 1])
            .unwrap(),
        [i64::MAX]
    );
}

#[test]
fn repeated_block_projection_reuses_the_same_matrix() {
    let matrix = fixture();
    let input = [2, -3, 5, 7, 11, -2, 3, -5, -7, -11];
    assert_eq!(
        matrix.project_i128_blocks(&input).unwrap(),
        [4i128, 12, -16, -4, -12, 16]
    );
    assert_eq!(
        matrix
            .project_blocks(&input.map(|value| value as i16))
            .unwrap(),
        [4i64, 12, -16, -4, -12, 16]
    );
    assert!(matrix.project_i128_blocks(&[]).is_err());
    assert!(matrix.project_i128_blocks(&input[..9]).is_err());
}

#[test]
fn native_width_kernels_match_exact_reference_with_simd_tails() {
    let shape = TernaryProjectionShape::new(17, 259).unwrap();
    let mut entries = vec![vec![0i8; shape.cols()]; shape.rows()];
    for (row, row_entries) in entries.iter_mut().enumerate() {
        for (col, entry) in row_entries.iter_mut().enumerate() {
            let code = (row * 37 + col * 19 + row * col) % 4;
            *entry = match code {
                0 => 0,
                1 | 3 => 1,
                _ => -1,
            };
        }
    }
    let matrix = matrix_from_entries(&entries);
    let input_i8: Vec<i8> = (0..shape.cols())
        .map(|index| (index % 127) as i8 - 63)
        .collect();
    let expected = matrix
        .project_i128(&input_i8.iter().copied().map(i128::from).collect::<Vec<_>>())
        .unwrap();
    let expected_i64: Vec<i64> = expected.iter().map(|value| *value as i64).collect();
    assert_eq!(matrix.project(&input_i8).unwrap(), expected_i64);
    assert_eq!(
        projection::tests::dense_scalar_i8(&matrix, &input_i8),
        expected_i64
    );
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        assert_eq!(
            projection::tests::avx512_small(&matrix, &input_i8),
            expected_i64
        );
    }
    let input_i16: Vec<i16> = input_i8
        .iter()
        .copied()
        .map(|value| i16::from(value) * 101)
        .collect();
    assert_eq!(
        matrix.project(&input_i16).unwrap(),
        expected
            .iter()
            .map(|value| *value as i64 * 101)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        projection::tests::dense_scalar_i16(&matrix, &input_i16),
        expected_i64
            .iter()
            .map(|value| value * 101)
            .collect::<Vec<_>>()
    );
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        assert_eq!(
            projection::tests::avx512_small(&matrix, &input_i16),
            expected_i64
                .iter()
                .map(|value| value * 101)
                .collect::<Vec<_>>()
        );
    }
    let input_i32: Vec<i32> = input_i8
        .iter()
        .copied()
        .map(|value| i32::from(value) * 100_003)
        .collect();
    assert_eq!(
        matrix.project(&input_i32).unwrap(),
        expected
            .iter()
            .map(|value| *value as i64 * 100_003)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        projection::tests::dense_scalar_i32(&matrix, &input_i32),
        expected_i64
            .iter()
            .map(|value| value * 100_003)
            .collect::<Vec<_>>()
    );
    let input_i64: Vec<i64> = input_i8
        .iter()
        .copied()
        .map(|value| i64::from(value) * 10_000_019)
        .collect();
    assert_eq!(
        matrix.project(&input_i64).unwrap(),
        expected
            .iter()
            .map(|value| *value as i64 * 10_000_019)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        projection::tests::dense_scalar_i64(&matrix, &input_i64),
        expected_i64
            .iter()
            .map(|value| value * 10_000_019)
            .collect::<Vec<_>>()
    );
}

#[test]
fn exact_squared_norm_rejects_overflow() {
    assert_eq!(squared_l2_i128(&[4, 12, -16]).unwrap(), 416);
    assert!(squared_l2_i128(&[i128::MAX]).is_err());
    assert!(squared_l2_i128(&[i128::MIN]).is_err());
}

#[test]
fn matrix_mle_matches_zero_padded_dense_evaluation() {
    let matrix = fixture();
    let row_point = [F::from_u64(2), F::from_u64(3)];
    let col_point = [F::from_u64(5), F::from_u64(7), F::from_u64(11)];
    let row_eq = EqPolynomial::evals(&row_point).unwrap();
    let col_eq = EqPolynomial::evals(&col_point).unwrap();
    let mut expected = F::zero();
    for (row, &row_weight) in row_eq.iter().enumerate() {
        for (col, &col_weight) in col_eq.iter().enumerate() {
            let entry = if row < 3 && col < 5 {
                matrix.entry(row, col).unwrap()
            } else {
                0
            };
            expected += row_weight * col_weight * field_from_i128(i128::from(entry));
        }
    }
    assert_eq!(
        eval_ternary_matrix_mle(&matrix, &row_point, &col_point).unwrap(),
        expected
    );
    assert_eq!(
        eval_ternary_matrix_mle_from_eq_tables(&matrix, &row_eq, &col_eq).unwrap(),
        expected
    );
    assert!(eval_ternary_matrix_mle_from_eq_tables(&matrix, &row_eq[..3], &col_eq).is_err());
    assert!(eval_ternary_matrix_mle(&matrix, &row_point[..1], &col_point).is_err());
}

#[test]
fn partial_row_evaluation_matches_joint_mle() {
    let matrix = fixture();
    let row_point = [F::from_u64(13), F::from_u64(17)];
    let col_point = [F::from_u64(19), F::from_u64(23), F::from_u64(29)];
    let weights = build_ternary_column_weights(&matrix, &row_point).unwrap();
    let col_eq = EqPolynomial::evals(&col_point).unwrap();
    let expected = weights
        .iter()
        .zip(col_eq)
        .fold(F::zero(), |sum, (&weight, eq)| sum + weight * eq);
    assert_eq!(
        eval_ternary_matrix_mle(&matrix, &row_point, &col_point).unwrap(),
        expected
    );
    assert_eq!(weights[5..], [F::zero(), F::zero(), F::zero()]);
}

#[test]
fn power_of_two_block_mle_factorizes_without_repeated_matrix_scan() {
    let matrix = fixture();
    let output_block = [F::from_u64(2), F::from_u64(3)];
    let input_block = [F::from_u64(5), F::from_u64(7)];
    let row_point = [F::from_u64(11), F::from_u64(13)];
    let col_point = [F::from_u64(17), F::from_u64(19), F::from_u64(23)];
    let got = eval_power_of_two_block_diagonal_mle(
        &matrix,
        4,
        &output_block,
        &row_point,
        &input_block,
        &col_point,
    )
    .unwrap();
    let expected = EqPolynomial::mle(&output_block, &input_block).unwrap()
        * eval_ternary_matrix_mle(&matrix, &row_point, &col_point).unwrap();
    assert_eq!(got, expected);
    assert!(eval_power_of_two_block_diagonal_mle(
        &matrix,
        3,
        &output_block,
        &row_point,
        &input_block,
        &col_point,
    )
    .is_err());
}
