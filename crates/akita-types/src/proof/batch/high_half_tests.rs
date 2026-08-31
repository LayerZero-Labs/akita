use super::*;
use crate::RingOpeningPoint;
use jolt_field::{Ext2, Fp32, FpExt4, FpExt8, MulBaseUnreduced, One, Ring, Zero};

type F = Fp32<251>;

fn ordinary_product_high_half<const D: usize>(
    lhs: &CyclotomicRing<F, D>,
    rhs: &CyclotomicRing<F, D>,
    mut output: Vec<F>,
) -> Vec<F> {
    for (lhs_index, &lhs_coeff) in lhs.coefficients().iter().enumerate() {
        for (rhs_index, &rhs_coeff) in rhs.coefficients().iter().enumerate() {
            let degree = lhs_index + rhs_index;
            if degree >= D {
                output[degree - D] += lhs_coeff * rhs_coeff;
            }
        }
    }
    output
}

fn deterministic_coeff(seed: usize) -> F {
    let value = (seed as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    F::from_u64(value)
}

fn check_base_multiplier_high_half<const D: usize>() {
    let point = RingMultiplierOpeningPoint::Base(RingOpeningPoint {
        position_weights: vec![F::zero(), F::from_u64(7)],
        live_block_weights: vec![F::one()],
    });
    let rhs: CyclotomicRing<F, D> =
        CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
            deterministic_coeff(index + D)
        }));
    let initial = (0..D)
        .map(|index| deterministic_coeff(index + 2 * D))
        .collect::<Vec<_>>();
    for index in 0..2 {
        let mut actual = initial.clone();
        point
            .accumulate_position_product_high_half(index, &rhs, &mut actual)
            .expect("base multiplier high half");
        assert_eq!(actual, initial);
    }
    assert!(point
        .accumulate_position_product_high_half(2, &rhs, &mut vec![F::zero(); D])
        .is_err());
    assert!(point
        .accumulate_position_product_high_half(0, &rhs, &mut vec![F::zero(); D - 1])
        .is_err());
}

fn check_subfield_multiplier_high_half<L, const D: usize>()
where
    L: FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    let k = L::DEGREE;
    let mut coordinate_cases = vec![vec![F::zero(); k]];
    let mut constant = vec![F::zero(); k];
    constant[0] = F::from_u64(9);
    coordinate_cases.push(constant);
    let mut boundary_shift = vec![F::zero(); k];
    boundary_shift[k - 1] = F::from_u64(11);
    coordinate_cases.push(boundary_shift);
    coordinate_cases.push(
        (0..k)
            .map(|index| deterministic_coeff(index + D + k))
            .collect(),
    );

    let mut rhs_cases = vec![CyclotomicRing::<F, D>::zero()];
    rhs_cases.push(CyclotomicRing::from_coefficients(std::array::from_fn(
        |index| {
            if index == 0 {
                F::from_u64(13)
            } else {
                F::zero()
            }
        },
    )));
    rhs_cases.push(CyclotomicRing::from_coefficients(std::array::from_fn(
        |index| {
            if index == D - 1 {
                F::from_u64(17)
            } else {
                F::zero()
            }
        },
    )));
    rhs_cases.push(CyclotomicRing::from_coefficients(std::array::from_fn(
        |index| deterministic_coeff(3 * D + index),
    )));

    for coordinates in coordinate_cases {
        let value = L::from_base_slice(&coordinates);
        let point = SubfieldMultiplierOpeningPoint::new::<L, D>(
            &[value],
            &[value],
            AkitaError::InvalidProof,
        )
        .map(RingMultiplierOpeningPoint::Subfield)
        .expect("valid compact subfield multiplier");
        let multiplier =
            crate::embed_ring_subfield_scalar::<F, L, D>(value, AkitaError::InvalidProof)
                .expect("ordinary-product oracle multiplier");

        for rhs in &rhs_cases {
            let initial = (0..D)
                .map(|index| deterministic_coeff(4 * D + index))
                .collect::<Vec<_>>();
            let expected = ordinary_product_high_half(&multiplier, rhs, initial.clone());
            let mut actual = initial;
            point
                .accumulate_position_product_high_half(0, rhs, &mut actual)
                .expect("compact high-half product");
            assert_eq!(actual, expected, "D={D} K={k}");
        }
    }
}

#[test]
fn multiplier_high_half_matches_ordinary_polynomial_oracle() {
    check_base_multiplier_high_half::<64>();
    check_base_multiplier_high_half::<128>();
    check_base_multiplier_high_half::<256>();
    check_base_multiplier_high_half::<512>();

    check_subfield_multiplier_high_half::<Ext2<F>, 64>();
    check_subfield_multiplier_high_half::<Ext2<F>, 128>();
    check_subfield_multiplier_high_half::<Ext2<F>, 256>();
    check_subfield_multiplier_high_half::<Ext2<F>, 512>();
    check_subfield_multiplier_high_half::<FpExt4<F>, 64>();
    check_subfield_multiplier_high_half::<FpExt4<F>, 128>();
    check_subfield_multiplier_high_half::<FpExt4<F>, 256>();
    check_subfield_multiplier_high_half::<FpExt4<F>, 512>();
    check_subfield_multiplier_high_half::<FpExt4<F>, 1024>();
    check_subfield_multiplier_high_half::<FpExt8<F>, 64>();
    check_subfield_multiplier_high_half::<FpExt8<F>, 128>();
    check_subfield_multiplier_high_half::<FpExt8<F>, 256>();
}
