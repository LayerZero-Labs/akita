use super::*;

#[test]
fn aligned_i8_tile_width_keeps_full_tiles_on_digit_boundaries() {
    assert_eq!(aligned_i8_tile_width(130, 512, 64), 128);
    assert_eq!(aligned_i8_tile_width(63, 512, 64), 64);
    assert_eq!(aligned_i8_tile_width(1024, 65, 64), 64);
    assert_eq!(aligned_i8_tile_width(1024, 48, 64), 48);
}

#[test]
fn base_tile_width_accounts_for_every_matrix_row() {
    let single_row = base_tile_width::<i32, 2, 128>(1);
    let eleven_rows = base_tile_width::<i32, 2, 128>(11);

    assert_eq!(eleven_rows, single_row / 11);
}

#[test]
fn predecomposed_digit_api_rejects_digits_outside_log_basis_range() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let row = CyclotomicRing::<F, D>::one();
    let flat = FlatMatrix::from_ring_slice(&[row]);
    let slot = prepare_both_transforms(flat.ring_view::<D>(1, 1).expect("valid matrix"))
        .expect("Q32 dispatch should support this field and ring dimension");
    let bad_digits = vec![[4i8; D]];
    let blocks: Vec<&[[i8; D]]> = vec![bad_digits.as_slice()];

    assert!(matches!(
        mat_vec_mul_ntt_digits_i8::<F, D>(&slot, 1, 1, &blocks, 3),
        Err(akita_error::AkitaError::InvalidInput(_))
    ));
}

#[test]
fn cyclic_kernel_rejects_negacyclic_only_prepared_slot() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let flat = FlatMatrix::from_ring_slice(&[CyclotomicRing::<F, D>::one()]);
    let slot = build_negacyclic_ntt_slot(flat.ring_view::<D>(1, 1).expect("valid matrix"))
        .expect("Q32 dispatch should support this field and ring dimension");

    assert!(matches!(
        mat_vec_mul_ntt_single_i8_cyclic::<F, D>(&slot, 1, 1, &[[1; D]], 2),
        Err(akita_error::AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn i8_kernels_reject_crt_parameters_too_small_for_the_field() {
    use crate::kernels::linear::single_cyclic::{
        mat_vec_mul_single_i8_cyclic_with_params, mat_vec_mul_single_i8_with_params,
    };
    use akita_algebra::ntt::tables::Q32_PRIMES;
    use akita_error::AkitaError;

    type F = Prime128Offset275;
    const D: usize = 64;
    // One 32-bit prime cannot hold a single product term of a 128-bit field.
    let params = CrtNttParamSet::<i32, 1, D>::new([Q32_PRIMES[0]]);
    let row = [CyclotomicCrtNtt::<i32, 1, D>::zero()];
    let matrix: [&[CyclotomicCrtNtt<i32, 1, D>]; 1] = [&row];
    let ring_block = [CyclotomicRing::<F, D>::one()];
    let ring_blocks: [&[CyclotomicRing<F, D>]; 1] = [&ring_block];
    let digits = [[0i8; D]];

    let rejected = |result: Result<(), AkitaError>, expected: &str| {
        assert!(matches!(result, Err(AkitaError::InvalidSetup(message)) if message == expected));
    };
    rejected(
        mat_vec_mul_i8_with_params::<F, _, 1, D>(&matrix, &ring_blocks, 1, 4, &params).map(drop),
        "i8 matvec CRT capacity cannot fit a single term",
    );
    rejected(
        mat_vec_mul_i8_dense_with_params::<F, _, 1, D>(&matrix, &ring_blocks, 1, 4, &params)
            .map(drop),
        "i8 matvec CRT capacity cannot fit a single term",
    );
    rejected(
        mat_vec_mul_i8_dense_single_row_with_params::<F, _, 1, D>(
            &matrix,
            &ring_blocks,
            1,
            4,
            &params,
        )
        .map(drop),
        "single-row i8 CRT capacity cannot fit a single term",
    );
    let digit_blocks: [&[[i8; D]]; 1] = [&digits];
    rejected(
        mat_vec_mul_digits_i8_with_params::<F, _, 1, D>(&matrix, &digit_blocks, 4, &params)
            .map(drop),
        "digit matvec CRT capacity cannot fit a single term",
    );
    rejected(
        crate::kernels::linear::digits::mat_vec_mul_dense_digits_i8_with_params::<F, _, 1, D>(
            &matrix,
            &digit_blocks,
            4,
            &params,
        )
        .map(drop),
        "digit matvec CRT capacity cannot fit a single term",
    );
    // A call with no blocks returns before it needs any CRT capacity.
    assert_eq!(
        mat_vec_mul_digits_i8_with_params::<F, _, 1, D>(&matrix, &[], 4, &params).unwrap(),
        Vec::<Vec<CyclotomicRing<F, D>>>::new(),
    );
    rejected(
        mat_vec_mul_single_i8_with_params::<F, _, 1, D>(&matrix, &digits, 4, &params).map(drop),
        "single i8 CRT capacity cannot fit a single term",
    );
    rejected(
        mat_vec_mul_single_i8_cyclic_with_params::<F, _, 1, D>(&matrix, &digits, 4, &params)
            .map(drop),
        "cyclic i8 CRT capacity cannot fit a single term",
    );
}
