use super::*;
use crate::EqPolynomial;
use jolt_field::{Fp64, Ring, Zero};

type F = Fp64<4294967197>;

fn fixture() -> TernaryProjectionMatrix {
    let shape = TernaryProjectionShape::new(3, 5).unwrap();
    TernaryProjectionMatrix::from_bitplanes(
        shape,
        vec![0b0_01101, 0b0_10011, 0b0_11110],
        vec![0b0_01001, 0b0_10000, 0b0_00110],
    )
    .unwrap()
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
    assert_eq!(shape.row_bytes(), 2);
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
fn bitplanes_are_canonical_and_decode_balanced_ternary() {
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
    assert!(TernaryProjectionMatrix::from_bitplanes(shape, vec![0], vec![1]).is_err());
    assert!(TernaryProjectionMatrix::from_bitplanes(shape, vec![0b1000_0000], vec![0]).is_err());
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

    let overflow_matrix = TernaryProjectionMatrix::from_bitplanes(
        TernaryProjectionShape::new(1, 2).unwrap(),
        vec![0b11],
        vec![0b11],
    )
    .unwrap();
    assert!(overflow_matrix.project_i128(&[i128::MAX, 1]).is_err());
    assert!(overflow_matrix.project(&[i64::MAX, 1]).is_err());

    let cancellation_matrix = TernaryProjectionMatrix::from_bitplanes(
        TernaryProjectionShape::new(1, 2).unwrap(),
        vec![0b11],
        vec![0b01],
    )
    .unwrap();
    assert_eq!(
        cancellation_matrix.project(&[i64::MAX, 1]).unwrap(),
        [i64::MAX - 1]
    );
    let intermediate_overflow_matrix = TernaryProjectionMatrix::from_bitplanes(
        TernaryProjectionShape::new(1, 3).unwrap(),
        vec![0b111],
        vec![0b011],
    )
    .unwrap();
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
    let mut nonzero = vec![0u8; shape.plane_len()];
    let mut positive = vec![0u8; shape.plane_len()];
    for row in 0..shape.rows() {
        for col in 0..shape.cols() {
            let code = (row * 37 + col * 19 + row * col) % 4;
            if code != 0 {
                let index = row * shape.row_bytes() + col / 8;
                let bit = 1 << (col & 7);
                nonzero[index] |= bit;
                if code & 1 == 1 {
                    positive[index] |= bit;
                }
            }
        }
    }
    let matrix = TernaryProjectionMatrix::from_bitplanes(shape, nonzero, positive).unwrap();
    let input_i8: Vec<i8> = (0..shape.cols())
        .map(|index| (index % 127) as i8 - 63)
        .collect();
    let expected = matrix
        .project_i128(&input_i8.iter().copied().map(i128::from).collect::<Vec<_>>())
        .unwrap();
    let expected_i64: Vec<i64> = expected.iter().map(|value| *value as i64).collect();
    assert_eq!(matrix.project(&input_i8).unwrap(), expected_i64);
    let first_row = &matrix.dense_rows()[..shape.cols()];
    assert_eq!(
        projection::tests::scalar_i8(first_row, &input_i8),
        expected_i64[0]
    );
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
        projection::tests::scalar_i16(first_row, &input_i16),
        expected_i64[0] * 101
    );
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
        projection::tests::scalar_i32(first_row, &input_i32),
        expected_i64[0] * 100_003
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
        projection::tests::scalar_i64(first_row, &input_i64),
        expected_i64[0] * 10_000_019
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
