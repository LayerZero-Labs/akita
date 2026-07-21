use super::*;

#[test]
fn aligned_i8_tile_width_keeps_full_tiles_on_digit_boundaries() {
    assert_eq!(aligned_i8_tile_width(130, 512, 64), 128);
    assert_eq!(aligned_i8_tile_width(63, 512, 64), 64);
    assert_eq!(aligned_i8_tile_width(1024, 65, 64), 64);
    assert_eq!(aligned_i8_tile_width(1024, 48, 64), 48);
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
        Err(akita_field::AkitaError::InvalidInput(_))
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
        Err(akita_field::AkitaError::InvalidSetup(_))
    ));
}
