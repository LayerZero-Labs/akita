use super::schoolbook_digit_mat_vec;
use crate::kernels::linear::{mat_vec_mul_ntt_digits_i8, validate_compression_batch_shape};
use akita_algebra::CyclotomicRing;
use akita_field::{
    CanonicalField, FieldCore, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59,
};
use akita_types::layout::FlatMatrix;
use akita_types::prepare_compression_ntt_cache;

fn assert_compression_batch<F: FieldCore + CanonicalField, const D: usize>() {
    let column_count = 3;
    let matrix = (0..column_count)
        .map(|index| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                F::from_i64(((index * 7 + coefficient * 3) % 17) as i64 - 8)
            }))
        })
        .collect::<Vec<_>>();
    let digit_vectors = vec![
        (0..column_count)
            .map(|column| std::array::from_fn(|coefficient| -(((column + coefficient) % 2) as i8)))
            .collect::<Vec<_>>(),
        (0..column_count)
            .map(|column| {
                std::array::from_fn(|coefficient| -(((2 * column + coefficient) % 2) as i8))
            })
            .collect::<Vec<_>>(),
    ];
    let flat = FlatMatrix::from_ring_slice(&matrix);
    let slot = prepare_compression_ntt_cache(
        flat.ring_view::<D>(1, column_count)
            .expect("compression matrix view"),
        column_count,
    )
    .expect("compression NTT profile");
    let views = digit_vectors.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let actual = mat_vec_mul_ntt_digits_i8::<F, D>(&slot, 1, column_count, &views, 1)
        .expect("compression batch rows");
    let expected_matrix = vec![matrix];
    let expected = schoolbook_digit_mat_vec::<F, D>(&expected_matrix, &digit_vectors);
    assert_eq!(actual, expected);
}

#[test]
fn compression_batch_matches_schoolbook_across_the_rank_one_ladders() {
    assert_compression_batch::<Prime128OffsetA7F7, 8>();
    assert_compression_batch::<Prime128OffsetA7F7, 16>();
    assert_compression_batch::<Prime128OffsetA7F7, 32>();
    assert_compression_batch::<Prime64Offset59, 16>();
    assert_compression_batch::<Prime64Offset59, 32>();
    assert_compression_batch::<Prime64Offset59, 64>();
    assert_compression_batch::<Prime32Offset99, 32>();
    assert_compression_batch::<Prime32Offset99, 64>();
    assert_compression_batch::<Prime32Offset99, 128>();
}

#[test]
fn compression_batch_rejects_mixed_shapes_and_non_binary_digits() {
    type F = Prime128OffsetA7F7;
    const D: usize = 8;
    let flat = FlatMatrix::from_ring_slice(&[CyclotomicRing::<F, D>::one(); 4]);
    let slot = prepare_compression_ntt_cache(flat.ring_view::<D>(1, 4).expect("matrix"), 4)
        .expect("compression NTT profile");
    let short = [[0i8; D]; 3];
    let full = [[0i8; D]; 4];
    assert!(validate_compression_batch_shape(&[&short, &full]).is_err());

    let outside_binary = [[2i8; D]; 4];
    assert!(mat_vec_mul_ntt_digits_i8::<F, D>(&slot, 1, 4, &[&outside_binary], 1).is_err());
}
