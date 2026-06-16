use super::{
    build_jl_row_weights, build_jl_row_weights_reference, eval_jl_mle_at, eval_jl_mle_at_reference,
    eval_mle_from_weights,
};
use crate::jl::{JlProjectionMatrix, DEFAULT_JL_ROWS};
use akita_field::{FieldCore, Fp64, FromPrimitiveInt, Prime128Offset275, Prime32Offset99};

type F32 = Prime32Offset99;
type F64 = Fp64<4294967197>;
type F128 = Prime128Offset275;

fn challenge_point<L: FieldCore + FromPrimitiveInt>(bits: usize, seed: u64) -> Vec<L> {
    (0..bits)
        .map(|i| {
            L::from_u64(
                seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(i as u64),
            )
        })
        .collect()
}

fn sample_sign_matrix(n_rows: usize, cols: usize) -> JlProjectionMatrix {
    let signs: Vec<Vec<i8>> = (0..n_rows)
        .map(|r| {
            (0..cols)
                .map(|c| (((r * 17 + c * 31) % 3) as i8) - 1)
                .collect()
        })
        .collect();
    JlProjectionMatrix::from_sign_rows(&signs).unwrap()
}

fn mle_roundtrip_for<L: FieldCore + FromPrimitiveInt>() {
    let matrix = sample_sign_matrix(DEFAULT_JL_ROWS, 1023);
    let row_bits = matrix.n_rows().next_power_of_two().trailing_zeros() as usize;
    let col_bits = matrix.cols().next_power_of_two().trailing_zeros() as usize;
    let r_J = challenge_point::<L>(row_bits, 0x4A4A_4A4A);
    let r_w = challenge_point::<L>(col_bits, 0xB5B5_B5B5);

    let fused = eval_jl_mle_at(&matrix, &r_J, &r_w).expect("fused eval");
    let reference = eval_jl_mle_at_reference(&matrix, &r_J, &r_w).expect("ref eval");
    assert_eq!(fused, reference);

    let g = build_jl_row_weights(&matrix, &r_J).expect("row weights");
    let g_ref = build_jl_row_weights_reference(&matrix, &r_J).expect("ref row weights");
    assert_eq!(g, g_ref);

    let from_g = eval_mle_from_weights(&g, &r_w).expect("eval from weights");
    assert_eq!(from_g, fused);
}

#[test]
fn split_eq_matches_reference_fp32() {
    mle_roundtrip_for::<F32>();
}

#[test]
fn split_eq_matches_reference_fp64() {
    mle_roundtrip_for::<F64>();
}

#[test]
fn split_eq_matches_reference_fp128() {
    mle_roundtrip_for::<F128>();
}

#[test]
fn split_eq_matches_reference_small_matrix() {
    let signs: Vec<Vec<i8>> = (0..5)
        .map(|r| (0..7).map(|c| ((r + c) % 3) as i8 - 1).collect())
        .collect();
    let matrix = JlProjectionMatrix::from_sign_rows(&signs).unwrap();
    let r_J = vec![F64::from_u64(3), F64::from_u64(5), F64::from_u64(7)];
    let r_w = vec![F64::from_u64(11), F64::from_u64(13), F64::from_u64(17)];

    let fused = eval_jl_mle_at(&matrix, &r_J, &r_w).unwrap();
    let reference = eval_jl_mle_at_reference(&matrix, &r_J, &r_w).unwrap();
    assert_eq!(fused, reference);
}

#[test]
fn malformed_point_length_returns_error() {
    let matrix = sample_sign_matrix(DEFAULT_JL_ROWS, 64);
    let err = eval_jl_mle_at(&matrix, &[F64::one()], &[F64::one(); 6]).unwrap_err();
    assert!(matches!(err, akita_field::AkitaError::InvalidSize { .. }));
}

#[test]
fn accumulate_row_weight_range_matches_entrywise() {
    use super::common::{accum_sign_weight, accumulate_row_weight_range, entry_sign};

    let signs: Vec<Vec<i8>> = vec![(0..17).map(|c| ((c * 2) % 3) as i8 - 1).collect()];
    let matrix = JlProjectionMatrix::from_sign_rows(&signs).unwrap();
    let row = matrix.row_bytes_slice(0);
    let col0 = 3;
    let n_cols = 10;
    let weights: Vec<F64> = (0..n_cols).map(|i| F64::from_u64(i as u64 + 1)).collect();

    let fast = accumulate_row_weight_range::<F64>(row, col0, n_cols, &weights);
    let mut slow = F64::zero();
    for (k, &weight) in weights.iter().enumerate() {
        let sign = entry_sign(&matrix, 0, col0 + k);
        slow = accum_sign_weight(slow, sign, weight);
    }
    assert_eq!(fast, slow);
}
