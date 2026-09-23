use super::*;
use akita_algebra::{MinusTrinomial, PlusTrinomial, Prime64Offset23703};
use jolt_field::{Field, Prime128OffsetA7F7};

fn sample<F: Field, const D: usize, M: TrinomialModulus>(seed: u64) -> TrinomialRing<F, D, M> {
    TrinomialRing::from_coefficients(std::array::from_fn(|i| {
        F::from_i64(((i as u64 * 29 + seed) % 37) as i64 - 18)
    }))
    .unwrap()
}

fn check<F: SmoothFftField + Debug, const D: usize, M: TrinomialModulus>() {
    let a = [sample::<F, D, M>(3), sample::<F, D, M>(21)];
    let v = [sample::<F, D, M>(7), sample::<F, D, M>(16)];
    let one = TrinomialRing::from_coefficients(std::array::from_fn(|i| {
        if i == 0 {
            F::one()
        } else {
            F::zero()
        }
    }))
    .unwrap();
    let challenges = [sample::<F, D, M>(9), one];
    let first_image = sample::<F, D, M>(15);
    let images = [
        first_image,
        a[0].schoolbook_mul(&v[0]).unwrap() + a[1].schoolbook_mul(&v[1]).unwrap()
            - first_image.schoolbook_mul(&challenges[0]).unwrap(),
    ];
    let mut builder = TrinomialRelationQuotientBuilder::new(&v, &challenges).unwrap();
    let quotient = builder.build(&a, &images).unwrap();

    // Independent convolution and explicit recomposition, without sharing
    // either the FFT or the production polynomial-division algorithm.
    let mut expected = vec![F::zero(); 2 * D - 1];
    for (a, v) in a.iter().zip(&v) {
        for (i, &a) in a.coefficients().iter().enumerate() {
            for (j, &v) in v.coefficients().iter().enumerate() {
                expected[i + j] += a * v;
            }
        }
    }
    for (image, challenge) in images.iter().zip(&challenges) {
        for (i, &image) in image.coefficients().iter().enumerate() {
            for (j, &challenge) in challenge.coefficients().iter().enumerate() {
                expected[i + j] -= image * challenge;
            }
        }
    }
    let mut recomposed = vec![F::zero(); 2 * D - 1];
    assert_eq!(quotient.len(), D - 1);
    for (i, &q) in quotient.iter().enumerate() {
        recomposed[i] += q;
        recomposed[i + D / 2] += F::from_i64(i64::from(M::MIDDLE_COEFFICIENT)) * q;
        recomposed[i + D] += q;
    }
    assert_eq!(recomposed, expected);
    let eval = |coefficients: &[F], alpha: F| {
        coefficients
            .iter()
            .rev()
            .fold(F::zero(), |acc, &x| acc * alpha + x)
    };
    let modulus_root = primitive_nth_root::<F>(D / 2 * M::ROOT_ORDER_STRIDE);
    assert!(builder
        .polynomial()
        .evaluate_modulus_at(modulus_root)
        .unwrap()
        .is_zero());
    for alpha in [F::zero(), F::one(), F::from_u64(11), modulus_root] {
        assert_eq!(
            eval(&expected, alpha),
            builder.polynomial().evaluate_modulus_at(alpha).unwrap() * eval(&quotient, alpha),
        );
    }
    let wrong_images = [images[0] + one, images[1]];
    assert!(builder.build(&a, &wrong_images).is_err());
    assert!(builder.build(&[], &images).is_err());
    assert!(builder.build(&a, &[]).is_err());
    // A failed row must not leave an accumulator contribution in the next row.
    assert_eq!(
        builder
            .build(&a.map(|x| x + x), &images.map(|x| x + x))
            .unwrap(),
        quotient.iter().map(|&q| q + q).collect::<Vec<_>>()
    );
}

#[test]
fn trinomial_quotients_match_independent_coefficients() {
    check::<Prime64Offset23703, 162, PlusTrinomial>();
    check::<Prime64Offset23703, 324, MinusTrinomial>();
    check::<Prime128OffsetA7F7, 648, MinusTrinomial>();
}

#[test]
fn invalid_convolution_domains_reject_before_allocation() {
    assert!(
        TrinomialRelationQuotientBuilder::<Prime64Offset23703, 0, PlusTrinomial>::new(&[], &[])
            .is_err()
    );
    assert!(
        TrinomialRelationQuotientBuilder::<Prime64Offset23703, 5, PlusTrinomial>::new(&[], &[])
            .is_err()
    );
    assert!(
        TrinomialRelationQuotientBuilder::<Prime64Offset23703, 1296, MinusTrinomial>::new(&[], &[])
            .is_err()
    );
}
