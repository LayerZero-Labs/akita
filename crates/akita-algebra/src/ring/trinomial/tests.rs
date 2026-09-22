use jolt_field::{Field, Prime128OffsetA7F7, Ring, Zero};

use super::*;
use crate::fft::{field_pow, primitive_nth_root, Prime64Offset23703};

fn sample_ring<F: Field, const D: usize, M: TrinomialModulus>(seed: u64) -> TrinomialRing<F, D, M> {
    TrinomialRing::from_coefficients(std::array::from_fn(|index| {
        let value = seed
            .wrapping_add(index as u64 * 0x9e37_79b9)
            .rotate_left((index % 61) as u32);
        F::from_u64(value)
    }))
    .expect("test degree is valid")
}

fn assert_roundtrip_and_product<F, const D: usize, M>()
where
    F: SmoothFftField + fmt::Debug,
    M: TrinomialModulus,
{
    let domain = TrinomialNttDomain::<F, D, M>::new().expect("test shape must fully split");
    let lhs = sample_ring::<F, D, M>(0x1234_5678);
    let rhs = sample_ring::<F, D, M>(0xfedc_ba98);
    let mut workspace = domain.workspace();
    let transformed = domain.forward_with_workspace(&lhs, &mut workspace);
    assert_eq!(
        domain.inverse_with_workspace(&transformed, &mut workspace),
        lhs
    );
    assert_eq!(
        domain.multiply_with_workspace(&lhs, &rhs, &mut workspace),
        lhs.schoolbook_mul(&rhs).expect("schoolbook multiplication")
    );
}

fn assert_i8_lut_matches_field_transform<F, const D: usize, M>(log_basis: u32)
where
    F: SmoothFftField + fmt::Debug,
    M: TrinomialModulus,
{
    let domain = TrinomialNttDomain::<F, D, M>::new().expect("test shape must fully split");
    let digit_count = 1usize << log_basis;
    let offset = digit_count / 2;
    let digits: [i8; D] =
        std::array::from_fn(|index| ((index * 29 + 7) % digit_count) as i16 - offset as i16)
            .map(|digit| digit as i8);
    let coefficients = digits.map(|digit| F::from_i64(i64::from(digit)));
    let ring = TrinomialRing::from_coefficients(coefficients).unwrap();
    let expected = domain.forward(&ring);
    let lut = domain.prepare_i8_lut(log_basis).unwrap();
    assert_eq!(
        lut.table_bytes(),
        2 * D * digit_count * core::mem::size_of::<F>()
    );

    let mut workspace = domain.workspace();
    let mut actual = domain.forward(&sample_ring::<F, D, M>(73));
    domain
        .forward_i8_with_lut_into_workspace(&digits, &lut, &mut actual, &mut workspace)
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(
        domain
            .forward_i8_with_lut_workspace(&digits, &lut, &mut workspace)
            .unwrap(),
        expected
    );
}

fn assert_packed_accumulation_matches_scalar<F, const D: usize, M>()
where
    F: SmoothFftField + jolt_field::WithPacking + fmt::Debug,
    M: TrinomialModulus,
{
    let domain = TrinomialNttDomain::<F, D, M>::new().expect("test shape must fully split");
    let lhs = domain.forward(&sample_ring::<F, D, M>(79));
    let rhs = domain.forward(&sample_ring::<F, D, M>(83));
    let mut scalar = domain.forward(&sample_ring::<F, D, M>(89));
    let mut packed = scalar.clone();
    scalar.add_assign_pointwise_mul(&lhs, &rhs);
    packed.add_assign_pointwise_mul_packed(&lhs, &rhs);
    assert_eq!(packed, scalar);
}

fn assert_transform_slots_are_direct_evaluations<F, const D: usize, M>()
where
    F: SmoothFftField + fmt::Debug,
    M: TrinomialModulus,
{
    let domain = TrinomialNttDomain::<F, D, M>::new().expect("test shape must fully split");
    let value = sample_ring::<F, D, M>(41);
    let transformed = domain.forward(&value);
    let half = D / 2;
    let order = half * M::ROOT_ORDER_STRIDE;
    let positive_shift = primitive_nth_root::<F>(order);
    let negative_shift = positive_shift.inverse().expect("root is nonzero");
    let omega = field_pow(positive_shift, M::ROOT_ORDER_STRIDE as u64);

    for index in 0..half {
        for (slot, shift) in [(index, positive_shift), (index + half, negative_shift)] {
            let point = shift * field_pow(omega, index as u64);
            let mut expected = F::zero();
            let mut power = F::one();
            for &coefficient in value.coefficients() {
                expected += coefficient * power;
                power *= point;
            }
            assert_eq!(transformed.slots()[slot], expected, "slot {slot}");
        }
    }
}

fn assert_quotient_identity<F, const D: usize, M>()
where
    F: Field + fmt::Debug,
    M: TrinomialModulus,
{
    let lhs = sample_ring::<F, D, M>(7);
    let rhs = sample_ring::<F, D, M>(11);
    let product = lhs
        .schoolbook_product_coefficients(&rhs)
        .expect("schoolbook product");
    let (remainder, quotient) =
        TrinomialRing::<F, D, M>::reduce_product_with_quotient(&product).expect("bounded product");

    let mut recomposed = remainder.coefficients().to_vec();
    recomposed.resize(2 * D - 1, F::zero());
    for (degree, &coefficient) in quotient.iter().enumerate() {
        recomposed[degree] += coefficient;
        if M::MIDDLE_COEFFICIENT > 0 {
            recomposed[degree + D / 2] += coefficient;
        } else {
            recomposed[degree + D / 2] -= coefficient;
        }
        recomposed[degree + D] += coefficient;
    }
    assert_eq!(recomposed, product);
}

#[test]
fn p64_initial_tower_matches_schoolbook() {
    assert_roundtrip_and_product::<Prime64Offset23703, 162, PlusTrinomial>();
    assert_roundtrip_and_product::<Prime64Offset23703, 324, MinusTrinomial>();
    assert_roundtrip_and_product::<Prime64Offset23703, 648, MinusTrinomial>();
}

#[test]
fn p128_initial_tower_matches_schoolbook() {
    assert_roundtrip_and_product::<Prime128OffsetA7F7, 162, PlusTrinomial>();
    assert_roundtrip_and_product::<Prime128OffsetA7F7, 324, MinusTrinomial>();
    assert_roundtrip_and_product::<Prime128OffsetA7F7, 648, MinusTrinomial>();
}

#[test]
fn p128_degree_486_scalar_profile_is_generic() {
    assert_roundtrip_and_product::<Prime128OffsetA7F7, 486, PlusTrinomial>();
}

#[test]
fn prepared_i8_transforms_match_field_conversion_at_small_and_full_ranges() {
    assert_i8_lut_matches_field_transform::<Prime64Offset23703, 162, PlusTrinomial>(4);
    assert_i8_lut_matches_field_transform::<Prime64Offset23703, 648, MinusTrinomial>(8);
    assert_i8_lut_matches_field_transform::<Prime128OffsetA7F7, 324, MinusTrinomial>(7);
}

#[test]
fn prepared_i8_transform_rejects_invalid_basis_and_digits_before_output_mutation() {
    type F = Prime64Offset23703;
    let domain = TrinomialNttDomain::<F, 162, PlusTrinomial>::new().unwrap();
    assert!(matches!(
        domain.prepare_i8_lut(0),
        Err(TrinomialError::InvalidDigitBasis { log_basis: 0 })
    ));
    assert!(matches!(
        domain.prepare_i8_lut(9),
        Err(TrinomialError::InvalidDigitBasis { log_basis: 9 })
    ));

    let lut = domain.prepare_i8_lut(4).unwrap();
    let mut digits = [0i8; 162];
    digits[81] = 8;
    let mut output = domain.forward(&sample_ring::<F, 162, PlusTrinomial>(97));
    let original = output.clone();
    let mut workspace = domain.workspace();
    assert!(matches!(
        domain.forward_i8_with_lut_into_workspace(&digits, &lut, &mut output, &mut workspace),
        Err(TrinomialError::DigitOutOfRange {
            digit: 8,
            log_basis: 4
        })
    ));
    assert_eq!(output, original);
}

#[test]
fn packed_pointwise_accumulation_matches_fused_scalar_kernel() {
    assert_packed_accumulation_matches_scalar::<Prime64Offset23703, 162, PlusTrinomial>();
    assert_packed_accumulation_matches_scalar::<Prime128OffsetA7F7, 162, PlusTrinomial>();
}

#[test]
fn p64_rejects_unsplit_degree_486_scalar_profile() {
    assert!(matches!(
        TrinomialNttDomain::<Prime64Offset23703, 486, PlusTrinomial>::new(),
        Err(TrinomialError::UnsupportedRootOrder {
            required: 729,
            available: 1_944
        })
    ));
}

#[test]
fn slot_layout_matches_direct_polynomial_evaluation() {
    assert_transform_slots_are_direct_evaluations::<Prime64Offset23703, 162, PlusTrinomial>();
    assert_transform_slots_are_direct_evaluations::<Prime64Offset23703, 324, MinusTrinomial>();
}

#[test]
fn quotient_identity_covers_both_modulus_signs() {
    assert_quotient_identity::<Prime64Offset23703, 162, PlusTrinomial>();
    assert_quotient_identity::<Prime64Offset23703, 324, MinusTrinomial>();
}

#[test]
fn quotient_reduction_covers_half_degree_and_top_wraps() {
    type F = Prime64Offset23703;
    let mut polynomial = vec![F::zero(); 323];
    polynomial[162] = F::from_u64(7);
    polynomial[242] = F::from_u64(11);
    polynomial[322] = F::from_u64(13);
    let (remainder, quotient) =
        TrinomialRing::<F, 162, PlusTrinomial>::reduce_product_with_quotient(&polynomial)
            .expect("bounded product");
    assert_eq!(quotient[0], F::from_u64(7));
    assert_eq!(quotient[80], F::from_u64(11));
    assert_eq!(quotient[160], F::from_u64(13));

    let mut recomposed = remainder.coefficients().to_vec();
    recomposed.resize(323, F::zero());
    for (degree, coefficient) in quotient.into_iter().enumerate() {
        recomposed[degree] += coefficient;
        recomposed[degree + 81] += coefficient;
        recomposed[degree + 162] += coefficient;
    }
    assert_eq!(recomposed, polynomial);
}

#[test]
fn signed_packing_roundtrips_rank_two_and_four() {
    type F = Prime64Offset23703;
    let components_two = [
        sample_ring::<F, 162, PlusTrinomial>(1),
        sample_ring::<F, 162, PlusTrinomial>(2),
    ];
    let packed_two =
        pack_scalar_components::<F, 162, 324, MinusTrinomial>(&components_two).unwrap();
    assert_eq!(
        unpack_scalar_components::<F, 162, 324, MinusTrinomial>(&packed_two).unwrap(),
        components_two
    );

    let components_four = [
        sample_ring::<F, 162, PlusTrinomial>(3),
        sample_ring::<F, 162, PlusTrinomial>(4),
        sample_ring::<F, 162, PlusTrinomial>(5),
        sample_ring::<F, 162, PlusTrinomial>(6),
    ];
    let packed_four =
        pack_scalar_components::<F, 162, 648, MinusTrinomial>(&components_four).unwrap();
    assert_eq!(
        unpack_scalar_components::<F, 162, 648, MinusTrinomial>(&packed_four).unwrap(),
        components_four
    );

    for index in 0..162 {
        let expected = if index % 2 == 0 {
            components_four[3].coefficients()[index]
        } else {
            -components_four[3].coefficients()[index]
        };
        assert_eq!(packed_four.coefficients()[4 * index + 3], expected);
    }
}

#[test]
fn scalar_embedding_commutes_with_multiplication() {
    type F = Prime64Offset23703;
    let lhs = sample_ring::<F, 162, PlusTrinomial>(17);
    let rhs = sample_ring::<F, 162, PlusTrinomial>(29);
    let scalar_product = lhs.schoolbook_mul(&rhs).unwrap();

    let embedded_lhs = embed_scalar::<F, 162, 648, MinusTrinomial>(&lhs).unwrap();
    let embedded_rhs = embed_scalar::<F, 162, 648, MinusTrinomial>(&rhs).unwrap();
    let expected = embed_scalar::<F, 162, 648, MinusTrinomial>(&scalar_product).unwrap();
    let domain = TrinomialNttDomain::<F, 648, MinusTrinomial>::new().unwrap();
    assert_eq!(domain.multiply(&embedded_lhs, &embedded_rhs), expected);
}

#[test]
fn rank_one_plus_packing_is_the_identity_ring_map() {
    type F = Prime64Offset23703;
    let lhs = sample_ring::<F, 162, PlusTrinomial>(31);
    let rhs = sample_ring::<F, 162, PlusTrinomial>(47);

    let packed = pack_scalar_components::<F, 162, 162, PlusTrinomial>(&[lhs]).unwrap();
    let embedded_lhs = embed_scalar::<F, 162, 162, PlusTrinomial>(&lhs).unwrap();
    let embedded_rhs = embed_scalar::<F, 162, 162, PlusTrinomial>(&rhs).unwrap();
    assert_eq!(packed, lhs);
    assert_eq!(embedded_lhs, lhs);
    assert_eq!(embedded_rhs, rhs);
    assert_eq!(
        unpack_scalar_components::<F, 162, 162, PlusTrinomial>(&packed).unwrap(),
        vec![lhs]
    );

    let scalar_product = lhs.schoolbook_mul(&rhs).unwrap();
    assert_eq!(
        embedded_lhs.schoolbook_mul(&embedded_rhs).unwrap(),
        embed_scalar::<F, 162, 162, PlusTrinomial>(&scalar_product).unwrap()
    );
}

#[test]
fn packed_scalar_action_is_componentwise() {
    type F = Prime64Offset23703;
    let challenge = sample_ring::<F, 162, PlusTrinomial>(13);
    let components = [
        sample_ring::<F, 162, PlusTrinomial>(21),
        sample_ring::<F, 162, PlusTrinomial>(22),
        sample_ring::<F, 162, PlusTrinomial>(23),
        sample_ring::<F, 162, PlusTrinomial>(24),
    ];
    let acted = components.map(|component| component.schoolbook_mul(&challenge).unwrap());
    let packed = pack_scalar_components::<F, 162, 648, MinusTrinomial>(&components).unwrap();
    let expected = pack_scalar_components::<F, 162, 648, MinusTrinomial>(&acted).unwrap();
    let embedded = embed_scalar::<F, 162, 648, MinusTrinomial>(&challenge).unwrap();
    let domain = TrinomialNttDomain::<F, 648, MinusTrinomial>::new().unwrap();
    let mut workspace = domain.workspace();
    assert_eq!(
        domain.multiply_with_workspace(&packed, &embedded, &mut workspace),
        expected
    );
}

#[test]
fn packing_rejects_wrong_sign_rank_and_dimensions() {
    type F = Prime64Offset23703;
    let components = [
        sample_ring::<F, 162, PlusTrinomial>(1),
        sample_ring::<F, 162, PlusTrinomial>(2),
    ];
    assert!(pack_scalar_components::<F, 162, 324, PlusTrinomial>(&components).is_err());
    assert!(pack_scalar_components::<F, 162, 648, MinusTrinomial>(&components).is_err());
    assert!(TrinomialRing::<F, 3, PlusTrinomial>::from_coefficients([F::zero(); 3]).is_err());
    let scalar = sample_ring::<F, 162, PlusTrinomial>(3);
    assert!(unpack_scalar_components::<F, 0, 162, PlusTrinomial>(&scalar).is_err());
}

#[test]
fn pointwise_accumulation_matches_sum_of_products() {
    type F = Prime64Offset23703;
    let domain = TrinomialNttDomain::<F, 324, MinusTrinomial>::new().unwrap();
    let lhs = domain.forward(&sample_ring::<F, 324, MinusTrinomial>(1));
    let rhs = domain.forward(&sample_ring::<F, 324, MinusTrinomial>(2));
    let other_lhs = domain.forward(&sample_ring::<F, 324, MinusTrinomial>(3));
    let other_rhs = domain.forward(&sample_ring::<F, 324, MinusTrinomial>(4));
    let mut accumulated = lhs.pointwise_mul(&rhs);
    accumulated.add_assign_pointwise_mul(&other_lhs, &other_rhs);
    for index in 0..324 {
        assert_eq!(
            accumulated.slots()[index],
            lhs.slots()[index] * rhs.slots()[index]
                + other_lhs.slots()[index] * other_rhs.slots()[index]
        );
    }
}

#[test]
fn coefficient_arithmetic_and_scalar_accumulation_agree() {
    type F = Prime64Offset23703;
    let lhs = sample_ring::<F, 162, PlusTrinomial>(31);
    let rhs = sample_ring::<F, 162, PlusTrinomial>(37);
    let scalar = F::from_u64(43);
    let mut accumulated = lhs;
    rhs.scale_accumulate_into(&mut accumulated, scalar);
    assert_eq!(accumulated, lhs + rhs.scale(scalar));
    assert_eq!((lhs + rhs) - rhs, lhs);
    assert_eq!(lhs + (-lhs), TrinomialRing::zero().unwrap());
    assert_eq!(
        lhs.schoolbook_mul(&TrinomialRing::one().unwrap()).unwrap(),
        lhs
    );
}
