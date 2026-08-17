#![allow(missing_docs)]

use akita_field::{pseudo_mersenne_modulus, PrimeOffsetSpec, PRIME_OFFSET_SPECS};

// Strong probable-prime test using multiple fixed bases.
// This is not a formal primality certificate, but is sufficient as a
// practical regression guard for the current registered prime-offset profiles.
fn is_probable_prime_miller_rabin(n: u128) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }

    const SMALL_PRIMES: [u128; 11] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for p in SMALL_PRIMES {
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
    }

    let (d, s) = decompose_pow2(n - 1);
    const BASES: [u128; 24] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
    ];

    'outer: for a in BASES {
        if a >= n {
            continue;
        }
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }

    true
}

fn decompose_pow2(mut d: u128) -> (u128, u32) {
    let mut s = 0u32;
    while d.is_multiple_of(2) {
        d >>= 1;
        s += 1;
    }
    (d, s)
}

fn pow_mod(mut base: u128, mut exp: u128, modulus: u128) -> u128 {
    let mut result = 1u128;
    base %= modulus;
    while exp > 0 {
        if (exp & 1) == 1 {
            result = mul_mod(result, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exp >>= 1;
    }
    result
}

fn mul_mod(mut a: u128, mut b: u128, modulus: u128) -> u128 {
    let mut result = 0u128;
    a %= modulus;
    b %= modulus;
    while b > 0 {
        if (b & 1) == 1 {
            result = add_mod(result, a, modulus);
        }
        a = add_mod(a, a, modulus);
        b >>= 1;
    }
    result
}

fn add_mod(a: u128, b: u128, modulus: u128) -> u128 {
    if a >= modulus - b {
        a - (modulus - b)
    } else {
        a + b
    }
}

#[test]
fn prime_offset_profiles_are_probable_primes() {
    for PrimeOffsetSpec {
        bits,
        offset,
        modulus,
    } in PRIME_OFFSET_SPECS
    {
        assert_eq!(
            Some(modulus),
            pseudo_mersenne_modulus(bits, offset as u128),
            "profile formula mismatch for bits={bits}, offset={offset}"
        );
        assert!(
            is_probable_prime_miller_rabin(modulus),
            "Miller-Rabin rejected bits={bits}, offset={offset}, q={modulus}"
        );
    }
}

#[test]
fn selected_subword_primes_are_largest_admissible_candidates() {
    let selected31 = pseudo_mersenne_modulus(31, 19).expect("valid modulus");
    assert_eq!(selected31 % 8, 5);
    assert!(is_probable_prime_miller_rabin(selected31));
    for offset in 1..19 {
        let candidate = pseudo_mersenne_modulus(31, offset).expect("valid modulus");
        if candidate % 8 == 5 {
            assert!(!is_probable_prime_miller_rabin(candidate));
        }
    }

    let selected63 = pseudo_mersenne_modulus(63, 259).expect("valid modulus");
    assert_eq!(selected63 % 8, 5);
    assert!(is_probable_prime_miller_rabin(selected63));
    for offset in 1..259 {
        let candidate = pseudo_mersenne_modulus(63, offset).expect("valid modulus");
        let admissible = candidate % 8 == 5 || candidate % 16 == 9 || candidate % 32 == 17;
        if admissible {
            assert!(
                !is_probable_prime_miller_rabin(candidate),
                "found larger admissible 63-bit prime: {candidate}"
            );
        }
    }
}

#[test]
fn miller_rabin_rejects_known_composites() {
    let composites: [u128; 9] = [4, 9, 15, 21, 341, 561, 645, 1105, 1729];
    for n in composites {
        assert!(
            !is_probable_prime_miller_rabin(n),
            "composite unexpectedly accepted: {n}"
        );
    }
}
