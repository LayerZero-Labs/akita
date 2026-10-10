use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use num_bigint::BigUint;

use super::*;

const P64_OFFSET_59: u128 = (1 << 64) - 59;
const P128_OFFSET_275: u128 = u128::MAX - 274;
/// `2^64 - 23703` and `2^128 - 2^32 + 22537`: both are one modulo 243.
const P64_OFFSET_23703: u128 = (1 << 64) - 23_703;
const P128_OFFSET_A7F7: u128 = u128::MAX - (1 << 32) + 22_538;

fn profile(ring: BinaryScalarRing, weight: usize) -> BinaryChallengeProfile {
    BinaryChallengeProfile::bounded_weight(ring, weight).unwrap()
}

#[test]
fn named_proof_primes_are_admitted_only_with_a_large_residue_degree() {
    let ring = BinaryScalarRing::Cyclotomic243;
    let challenge = profile(ring, 46);
    for (prime, residue_degree) in [(P64_OFFSET_59, 162), (P128_OFFSET_275, 6)] {
        assert_eq!(
            labinius_proof_prime_residue_degree(prime, ring),
            Some(residue_degree)
        );
        assert!(check_labinius_proof_prime_units(prime, &challenge).is_ok());
    }
    for prime in [P64_OFFSET_23703, P128_OFFSET_A7F7] {
        assert_eq!(labinius_proof_prime_residue_degree(prime, ring), Some(1));
        assert!(check_labinius_proof_prime_units(prime, &challenge).is_err());
    }
    for multiple_of_three in [0, 3, 243, P64_OFFSET_59 * 3] {
        assert_eq!(
            labinius_proof_prime_residue_degree(multiple_of_three, ring),
            None
        );
        assert!(check_labinius_proof_prime_units(multiple_of_three, &challenge).is_err());
    }
}

/// Acceptance implies the exact inequality `P^f > (6w)^(deg/2)`, and rejection
/// implies it fails by less than the bit-length slack `2^(f + deg/2 - 1)`. The
/// order is recomputed by modular exponentiation.
#[test]
fn bit_length_check_brackets_the_exact_norm_inequality() {
    let primes = [
        2,
        5,
        7,
        (1 << 31) - 1,
        (1 << 61) - 1,
        P64_OFFSET_59,
        P64_OFFSET_23703,
        (1 << 127) - 1,
        P128_OFFSET_275,
        P128_OFFSET_A7F7,
    ];
    let one = BigUint::from(1u8);
    let (mut accepted, mut rejected) = (0, 0);
    for ring in [
        BinaryScalarRing::Cyclotomic243,
        BinaryScalarRing::Cyclotomic729,
    ] {
        let degree = ring.degree() as u32;
        let conductor = BigUint::from(3 * degree / 2);
        for prime in primes {
            let big_prime = BigUint::from(prime);
            let order = (1..=degree)
                .find(|&k| big_prime.modpow(&BigUint::from(k), &conductor) == one)
                .unwrap();
            assert_eq!(
                labinius_proof_prime_residue_degree(prime, ring),
                Some(order)
            );
            for weight in [1usize, 8, 46, 81, 162] {
                let norm_bound = BigUint::from(6 * weight).pow(degree / 2);
                let prime_power = big_prime.pow(order);
                if check_labinius_proof_prime_units(prime, &profile(ring, weight)).is_ok() {
                    assert!(prime_power > norm_bound);
                    accepted += 1;
                } else {
                    assert!(prime_power < (norm_bound << (order + degree / 2 - 1)));
                    rejected += 1;
                }
            }
        }
    }
    assert!(accepted > 0 && rejected > 0);
}
