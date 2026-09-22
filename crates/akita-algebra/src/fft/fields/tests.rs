use super::*;
use crate::fft::{distinct_prime_factors, primitive_nth_root};
use jolt_field::{CanonicalBytes, CanonicalEncoding, Field, One, Zero};

type F = Prime64Offset23703;

fn mul_mod(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((u128::from(lhs) * u128::from(rhs)) % u128::from(modulus)) as u64
}

fn pow_mod(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = mul_mod(result, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exponent >>= 1;
    }
    result
}

// Recursive Lucas primality certificate: each tuple is a prime factor q of
// n - 1, its multiplicity, and a Fermat/order witness. This fixed fixture is
// replayed entirely with exact u128 arithmetic, independently of jolt-field.
fn verify_lucas_certificate(candidate: u64) {
    if candidate == 2 {
        return;
    }
    let factors: &[(u64, u32, u64)] = match candidate {
        3 => &[(2, 1, 2)],
        5 => &[(2, 2, 2)],
        7 => &[(2, 1, 3), (3, 1, 2)],
        11 => &[(2, 1, 2), (5, 1, 2)],
        13 => &[(2, 2, 2), (3, 1, 2)],
        23 => &[(2, 1, 5), (11, 1, 2)],
        29 => &[(2, 2, 2), (7, 1, 2)],
        53 => &[(2, 2, 2), (13, 1, 2)],
        59 => &[(2, 1, 2), (29, 1, 2)],
        181 => &[(2, 2, 2), (3, 2, 2), (5, 1, 2)],
        827 => &[(2, 1, 2), (7, 1, 2), (59, 1, 2)],
        967 => &[(2, 1, 3), (3, 1, 2), (7, 1, 2), (23, 1, 2)],
        2897 => &[(2, 4, 3), (181, 1, 2)],
        7618704421 => &[
            (2, 2, 2),
            (3, 1, 2),
            (5, 1, 2),
            (53, 1, 2),
            (827, 1, 2),
            (2897, 1, 2),
        ],
        1355580840219689 => &[(2, 3, 3), (23, 1, 2), (967, 1, 2), (7618704421, 1, 2)],
        18446744073709527913 => &[(2, 3, 5), (3, 5, 2), (7, 1, 2), (1355580840219689, 1, 2)],
        _ => panic!("missing Lucas certificate for {candidate}"),
    };
    let mut factor_product = 1u128;
    for &(prime, exponent, witness) in factors {
        verify_lucas_certificate(prime);
        factor_product = factor_product
            .checked_mul(u128::from(prime).checked_pow(exponent).unwrap())
            .unwrap();
        assert_eq!(pow_mod(witness, candidate - 1, candidate), 1);
        let residue = pow_mod(witness, (candidate - 1) / prime, candidate);
        let mut a = if residue == 0 {
            candidate - 1
        } else {
            residue - 1
        };
        let mut b = candidate;
        while b != 0 {
            (a, b) = (b, a % b);
        }
        assert_eq!(a, 1, "Lucas order witness for factor {prime}");
    }
    assert_eq!(factor_product, u128::from(candidate - 1));
}

#[test]
fn modulus_is_prime() {
    verify_lucas_certificate(PRIME64_OFFSET_23703_MODULUS);
}

#[test]
fn declared_root_has_exact_order_1944_by_u128_arithmetic() {
    let root = F::SMOOTH_OMEGA as u64;
    let modulus = PRIME64_OFFSET_23703_MODULUS;
    assert!(root < modulus);
    assert_eq!(pow_mod(root, 1_944, modulus), 1);
    assert_ne!(pow_mod(root, 1_944 / 2, modulus), 1);
    assert_ne!(pow_mod(root, 1_944 / 3, modulus), 1);
}

#[test]
fn encoding_boundaries_are_canonical() {
    let largest = PRIME64_OFFSET_23703_MODULUS - 1;
    let field_largest = F::from_u128_checked(u128::from(largest)).expect("p - 1 is canonical");
    assert_eq!(field_largest.to_u128_checked(), Some(u128::from(largest)));
    assert!(F::from_u128_checked(u128::from(PRIME64_OFFSET_23703_MODULUS)).is_none());

    let mut encoded = [0u8; 8];
    field_largest.to_bytes_le(&mut encoded);
    assert_eq!(encoded, largest.to_le_bytes());
    assert_eq!(F::from_bytes_le_checked(&encoded), Some(field_largest));
    assert!(F::from_bytes_le_checked(&PRIME64_OFFSET_23703_MODULUS.to_le_bytes()).is_none());
}

#[test]
fn five_is_a_quadratic_non_residue_by_u128_arithmetic() {
    assert_eq!(
        pow_mod(
            5,
            (PRIME64_OFFSET_23703_MODULUS - 1) / 2,
            PRIME64_OFFSET_23703_MODULUS,
        ),
        PRIME64_OFFSET_23703_MODULUS - 1
    );
}

#[test]
fn nr5_extension_has_expected_basis_norm_and_inverses() {
    type E = Prime64Offset23703Ext2;
    let u = E::new(F::zero(), F::one());
    assert_eq!(u * u, E::new(F::from_u64(5), F::zero()));

    for a in 0..8 {
        for b in 0..8 {
            let value = E::new(F::from_u64(a), F::from_u64(b));
            let expected_norm = F::from_u64(a * a) - F::from_u64(5 * b * b);
            assert_eq!(value.norm(), expected_norm);
            assert_eq!(value.c0(), F::from_u64(a));
            assert_eq!(value.c1(), F::from_u64(b));
            if a == 0 && b == 0 {
                assert!(value.inverse().is_none());
            } else {
                assert_eq!(
                    value * value.inverse().expect("nonzero extension value"),
                    E::one()
                );
            }
        }
    }

    // The nontrivial Frobenius action on the power-basis generator is
    // conjugation because five is a non-residue.
    let mut frobenius_u = u;
    let mut exponent = PRIME64_OFFSET_23703_MODULUS - 1;
    let mut base = u;
    while exponent != 0 {
        if exponent & 1 == 1 {
            frobenius_u *= base;
        }
        base *= base;
        exponent >>= 1;
    }
    assert_eq!(frobenius_u, -u);
}

#[test]
fn transform_root_orders_cover_the_initial_tower() {
    for n in [81usize, 162, 243, 324, 648, 972, 1_944] {
        let root = primitive_nth_root::<F>(n);
        let expected = pow_mod(
            F::SMOOTH_OMEGA as u64,
            F::SMOOTH_SUBGROUP_ORDER as u64 / n as u64,
            PRIME64_OFFSET_23703_MODULUS,
        );
        assert_eq!(root.to_u128_checked(), Some(u128::from(expected)));
        assert_eq!(pow_mod(expected, n as u64, PRIME64_OFFSET_23703_MODULUS), 1);
        for factor in distinct_prime_factors(n) {
            assert_ne!(
                pow_mod(expected, (n / factor) as u64, PRIME64_OFFSET_23703_MODULUS),
                1,
            );
        }
    }
}
