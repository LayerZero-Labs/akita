use super::*;
use crate::fft::{distinct_prime_factors, field_pow, primitive_nth_root};
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

// Deterministic Miller-Rabin for the entire u64 range. These seven bases
// are independent of the field implementation under test.
fn is_prime_u64(candidate: u64) -> bool {
    if candidate < 2 {
        return false;
    }
    for prime in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if candidate == prime {
            return true;
        }
        if candidate.is_multiple_of(prime) {
            return false;
        }
    }

    let trailing = (candidate - 1).trailing_zeros();
    let odd_part = (candidate - 1) >> trailing;
    for witness in [2u64, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022] {
        let witness = witness % candidate;
        if witness == 0 {
            continue;
        }
        let mut power = pow_mod(witness, odd_part, candidate);
        if power == 1 || power == candidate - 1 {
            continue;
        }
        let mut composite = true;
        for _ in 1..trailing {
            power = mul_mod(power, power, candidate);
            if power == candidate - 1 {
                composite = false;
                break;
            }
        }
        if composite {
            return false;
        }
    }
    true
}

#[test]
fn modulus_is_prime() {
    assert!(is_prime_u64(PRIME64_OFFSET_23703_MODULUS));
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
        assert_eq!(field_pow(root, n as u64), F::one());
        for factor in distinct_prime_factors(n) {
            assert_ne!(field_pow(root, (n / factor) as u64), F::one());
        }
    }
}
