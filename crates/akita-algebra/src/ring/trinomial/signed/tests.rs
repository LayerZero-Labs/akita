use rand::{rngs::StdRng, Rng, SeedableRng};

use super::{signed_trinomial_product, signed_trinomial_remainder};
use crate::ring::trinomial::{MinusTrinomial, PlusTrinomial, TrinomialModulus};

const D: usize = 648;
const MATRIX_BOUND: i128 = 7_126_704;

// Independent long division, rather than the closed monomial folding used
// by the production reducer. Test inputs keep every operation in range.
fn reference_remainder<M: TrinomialModulus>(input: &[i128], degree: usize) -> Vec<i128> {
    let mut work = vec![0i128; (2 * degree - 1).max(input.len())];
    work[..input.len()].copy_from_slice(input);
    for exponent in (degree..work.len()).rev() {
        let value = work[exponent];
        work[exponent] = 0;
        work[exponent - degree] -= value;
        work[exponent - degree / 2] -= i128::from(M::MIDDLE_COEFFICIENT) * value;
    }
    work.truncate(degree);
    work
}

fn reference_product<M: TrinomialModulus>(left: &[i128], right: &[i128]) -> Vec<i128> {
    let mut product = vec![0; left.len() + right.len() - 1];
    for (u, &a) in left.iter().enumerate() {
        for (v, &b) in right.iter().enumerate() {
            product[u + v] += a * b;
        }
    }
    reference_remainder::<M>(&product, left.len())
}

#[test]
fn exact_centered_products_match_independent_integer_long_division() {
    let mut rng = StdRng::seed_from_u64(0x0051_676e_6564);
    for _ in 0..8 {
        let a: [i32; D] =
            std::array::from_fn(|_| rng.gen_range(-(MATRIX_BOUND as i32)..=MATRIX_BOUND as i32));
        let digits: [i8; D] = std::array::from_fn(|_| rng.gen_range(-4..=3));
        let sources: [i64; D] = std::array::from_fn(|_| rng.gen_range(-2340..=2340));
        let signed_bits: [i8; D] = std::array::from_fn(|t| {
            if rng.gen() {
                if (t / 4).is_multiple_of(2) {
                    1
                } else {
                    -1
                }
            } else {
                0
            }
        });
        let a128 = a.map(i128::from);
        for rhs in [
            digits.map(i128::from),
            sources.map(i128::from),
            signed_bits.map(i128::from),
        ] {
            assert_eq!(
                signed_trinomial_product::<D, MinusTrinomial>(&a, &rhs)
                    .unwrap()
                    .as_slice(),
                reference_product::<MinusTrinomial>(&a128, &rhs)
            );
        }
        // Exercise the actual narrow signed digit input, not only its i128 lift.
        assert_eq!(
            signed_trinomial_product::<D, MinusTrinomial>(&a, &digits).unwrap(),
            signed_trinomial_product::<D, MinusTrinomial>(&a128, &digits.map(i128::from)).unwrap()
        );
    }
    for a in [-MATRIX_BOUND, MATRIX_BOUND] {
        for z in [-2340i64, 2340] {
            assert_eq!(
                signed_trinomial_product::<D, MinusTrinomial>(&[a; D], &[z; D])
                    .unwrap()
                    .as_slice(),
                reference_product::<MinusTrinomial>(&[a; D], &[i128::from(z); D])
            );
        }
    }
    let small = [7i128; 162];
    assert_eq!(
        signed_trinomial_product::<162, PlusTrinomial>(&small, &small)
            .unwrap()
            .as_slice(),
        reference_product::<PlusTrinomial>(&small, &small)
    );
}

#[test]
fn every_product_monomial_and_dense_polynomial_matches_long_division() {
    for exponent in 0..2 * D - 1 {
        let mut monomial = vec![0; exponent + 1];
        monomial[exponent] = 1;
        assert_eq!(
            signed_trinomial_remainder::<D, MinusTrinomial>(&monomial)
                .unwrap()
                .as_slice(),
            reference_remainder::<MinusTrinomial>(&monomial, D)
        );
        assert_eq!(
            signed_trinomial_remainder::<D, PlusTrinomial>(&monomial)
                .unwrap()
                .as_slice(),
            reference_remainder::<PlusTrinomial>(&monomial, D)
        );
    }
    let dense: Vec<_> = (0..2 * D - 1).map(|i| i as i128 - 647).collect();
    assert_eq!(
        signed_trinomial_remainder::<D, MinusTrinomial>(&dense)
            .unwrap()
            .as_slice(),
        reference_remainder::<MinusTrinomial>(&dense, D)
    );
}

#[test]
fn exhaustive_psi_coefficient_kernel_has_exact_row_weight_972() {
    // Precompute rem(Y^k) independently. Each (u,v) below enumerates its
    // actual K[t,u,v]; sparse storage only omits known zero coefficients.
    let kernels: Vec<Vec<_>> = (0..2 * D - 1)
        .map(|exponent| {
            let mut monomial = vec![0; exponent + 1];
            monomial[exponent] = 1;
            reference_remainder::<MinusTrinomial>(&monomial, D)
                .into_iter()
                .enumerate()
                .filter(|(_, coefficient)| *coefficient != 0)
                .collect()
        })
        .collect();
    let mut weights = [0i128; D];
    let mut maximum_shift_weight = 0;
    for u in 0..D {
        let mut shift_weights = [0i128; D];
        for v in 0..D {
            for &(t, coefficient) in &kernels[u + v] {
                weights[t] += coefficient.abs();
                shift_weights[t] += coefficient.abs();
            }
        }
        maximum_shift_weight = maximum_shift_weight.max(shift_weights.into_iter().max().unwrap());
    }
    assert_eq!(weights.into_iter().max(), Some(972));
    // Each signed monomial shift has infinity operator norm at most two;
    // weight-46 signed challenges therefore have the stated Gamma = 92.
    assert_eq!(maximum_shift_weight, 2);
}

#[test]
fn q24_residual_and_projection_bounds_are_exact_and_fit_accumulators() {
    const Q: u128 = 14_253_409;
    const P: u128 = u128::MAX - 274; // 2^128 - 275.
    const BZ: i128 = 2340;
    const BT: i128 = 9_586_980;
    const GAMMA: i128 = 92;
    const C: i128 = 256;
    assert_eq!(MATRIX_BOUND, ((Q - 1) / 2) as i128);
    assert_eq!(BT, 4 * ((8i128.pow(8) - 1) / 7));
    for (m, expected_lr, expected_uj) in [
        (
            4096,
            66_394_525_036_769_280i128,
            1_839_701_006_388_918_996_628_930_560u128,
        ),
        (
            8192,
            132_788_824_280_985_600,
            3_679_395_756_376_625_911_509_811_200,
        ),
        (
            16384,
            265_577_422_769_418_240,
            7_358_785_256_352_039_741_271_572_480,
        ),
    ] {
        let lr = 972 * m * MATRIX_BOUND * BZ + C * GAMMA * BT;
        assert_eq!(lr, expected_lr);
        // Even an unreduced convolution accumulated across all m products,
        // the reduced sum, and the full residual fit signed 64-bit storage.
        // The exact integer kernels themselves use checked signed 128 bits.
        assert!(D as i128 * m * MATRIX_BOUND * BZ < i128::from(i64::MAX));
        assert!(lr < i128::from(i64::MAX));
        assert!(lr < i128::MAX);
        let uj = (Q - 1) * 1944 * u128::try_from(lr).unwrap();
        assert_eq!(uj, expected_uj);
        for rank in [3u128, 4] {
            let projection = (Q - 1) * rank * D as u128 * lr as u128;
            assert!(2 * projection < P);
            assert!(projection < 1 << 95); // Twelve-byte signed encoding.
            assert!(projection < i128::MAX as u128);
        }
    }
}

#[test]
fn malformed_shapes_and_integer_overflows_return_errors() {
    assert!(signed_trinomial_remainder::<0, MinusTrinomial>(&[]).is_err());
    assert!(signed_trinomial_remainder::<3, MinusTrinomial>(&[]).is_err());
    assert!(signed_trinomial_remainder::<D, MinusTrinomial>(&[0; 2 * D]).is_err());
    assert!(signed_trinomial_product::<D, MinusTrinomial>(&[1i8; D - 1], &[1i8; D]).is_err());
    assert!(signed_trinomial_product::<0, MinusTrinomial>(&[] as &[i8], &[] as &[i8]).is_err());
    assert!(signed_trinomial_product::<2, MinusTrinomial>(&[i128::MAX; 2], &[2i128; 2]).is_err());
    assert!(signed_trinomial_product::<2, MinusTrinomial>(&[i128::MAX; 2], &[1i128; 2]).is_err());
    assert!(signed_trinomial_remainder::<2, MinusTrinomial>(&[i128::MIN, 0, 1]).is_err());
    assert!(signed_trinomial_remainder::<2, PlusTrinomial>(&[0, 0, i128::MIN]).is_err());
    assert!(
        signed_trinomial_remainder::<4, MinusTrinomial>(&[0, 0, 0, 0, 0, 0, i128::MIN]).is_err()
    );
}
