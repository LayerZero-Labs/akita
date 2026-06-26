//! Deterministic parameter presets for small-prime CRT arithmetic.
//!
//! Q32: `logq = 32` with two `i32` NTT-friendly primes.
//! Q64: `logq = 64` with three `i32` NTT-friendly primes.
//! Q128: `logq = 128` with five `i32` NTT-friendly primes.

use super::crt::GarnerData;
use super::prime::NttPrime;

/// Polynomial degree for the base ring `Z_q[X]/(X^d + 1)`.
pub const RING_DEGREE: usize = 64;
/// Maximum ring degree covered by the CRT parameter sets.
pub const MAX_CRT_RING_DEGREE: usize = 256;

/// Number of CRT primes for the `logq = 32` parameter set.
pub const Q32_NUM_PRIMES: usize = 2;

/// The modulus `q = 2^32 - 99`.
pub const Q32_MODULUS: u64 = (1u64 << 32) - 99;

/// Number of CRT primes for the `logq = 64` reduced profile.
pub const Q64_NUM_PRIMES: usize = 3;

/// The modulus `q = 2^64 - 59`.
pub const Q64_MODULUS: u64 = u64::MAX - 58;

/// Number of CRT primes for the `logq = 128` parameter set.
pub const Q128_NUM_PRIMES: usize = 5;

/// Protocol modulus `q = 2^128 - 275`.
pub const Q128_MODULUS: u128 = u128::MAX - 274;

/// Raw 30-bit primes for the supported i32 profiles.
///
/// They are ordered descending by value.
pub const I32_RAW_PRIMES: [i32; Q128_NUM_PRIMES] =
    [1073707009, 1073698817, 1073692673, 1073682433, 1073668097];

/// CRT primes and per-prime Montgomery constants for Q32 measured `2xi32` profile.
pub const Q32_PRIMES: [NttPrime<i32>; Q32_NUM_PRIMES] = [
    NttPrime {
        p: 1073707009,
        pinv: 138446849,
        mont: 139260,
        montsq: 66621438,
    },
    NttPrime {
        p: 1073698817,
        pinv: 775989249,
        mont: 172028,
        montsq: -469934092,
    },
];

/// Garner CRT reconstruction constants for Q32 measured `2xi32` profile.
pub fn q32_garner() -> GarnerData<i32, Q32_NUM_PRIMES> {
    GarnerData::compute(&Q32_PRIMES)
}

/// Raw 30-bit primes for Q128.
pub const Q128_RAW_PRIMES: [i32; Q128_NUM_PRIMES] = I32_RAW_PRIMES;

/// CRT primes and per-prime Montgomery constants for `logq = 64` reduced profile.
pub const Q64_PRIMES: [NttPrime<i32>; Q64_NUM_PRIMES] = [
    NttPrime {
        p: 1073707009,
        pinv: 138446849,
        mont: 139260,
        montsq: 66621438,
    },
    NttPrime {
        p: 1073698817,
        pinv: 775989249,
        mont: 172028,
        montsq: -469934092,
    },
    NttPrime {
        p: 1073692673,
        pinv: 1342226433,
        mont: 196604,
        montsq: 196588,
    },
];

/// Garner CRT reconstruction constants for Q64 reduced profile.
pub fn q64_garner() -> GarnerData<i32, Q64_NUM_PRIMES> {
    GarnerData::compute(&Q64_PRIMES)
}

/// CRT primes and per-prime Montgomery constants for `logq = 128`.
pub fn q128_primes() -> [NttPrime<i32>; Q128_NUM_PRIMES] {
    std::array::from_fn(|k| NttPrime::compute(Q128_RAW_PRIMES[k]))
}

/// Garner CRT reconstruction constants for Q128.
pub fn q128_garner() -> GarnerData<i32, Q128_NUM_PRIMES> {
    let primes = q128_primes();
    GarnerData::compute(&primes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_garner_profile<W, const K: usize>(
        label: &str,
        primes: &[NttPrime<W>; K],
        garner: GarnerData<W, K>,
    ) where
        W: super::super::prime::PrimeWidth,
    {
        for (i, prime_i) in primes.iter().enumerate().skip(1) {
            let pi = prime_i.p.to_i64();
            for (j, prime_j) in primes[..i].iter().enumerate() {
                let pj = prime_j.p.to_i64();
                let g = garner.gamma[i][j].to_i64();
                assert_eq!(
                    (pj * g) % pi,
                    1,
                    "{label} garner gamma[{i}][{j}] not inverse of p_{j} mod p_{i}"
                );
            }
        }
    }

    fn is_prime(n: i64) -> bool {
        if n <= 1 {
            return false;
        }
        let mut divisor = 2;
        while divisor * divisor <= n {
            if n % divisor == 0 {
                return false;
            }
            divisor += 1;
        }
        true
    }

    fn assert_i16_prime_profile(primes: &[NttPrime<i16>]) {
        for prime in primes {
            let p = prime.p as i64;
            assert!(is_prime(p), "p={p} must be prime");
            assert!(p < (1 << 14), "p={p} must fit the i16 profile bound");
            assert_eq!(
                (p - 1) % 512,
                0,
                "512 must divide p-1 for D=256 NTT (p={p})"
            );
            let recomputed = NttPrime::compute(prime.p);
            assert_eq!(
                prime.pinv, recomputed.pinv,
                "pinv mismatch for p={}",
                prime.p
            );
            assert_eq!(
                prime.mont, recomputed.mont,
                "mont mismatch for p={}",
                prime.p
            );
            assert_eq!(
                prime.montsq, recomputed.montsq,
                "montsq mismatch for p={}",
                prime.p
            );
        }
    }

    fn assert_i32_prime_profile(primes: &[NttPrime<i32>]) {
        for prime in primes {
            let p = prime.p as i64;
            assert!(is_prime(p), "p={p} must be prime");
            assert_eq!(
                (p - 1) % 512,
                0,
                "512 must divide p-1 for D=256 NTT (p={p})"
            );
            let recomputed = NttPrime::compute(prime.p);
            assert_eq!(
                prime.pinv, recomputed.pinv,
                "pinv mismatch for p={}",
                prime.p
            );
            assert_eq!(
                prime.mont, recomputed.mont,
                "mont mismatch for p={}",
                prime.p
            );
            assert_eq!(
                prime.montsq, recomputed.montsq,
                "montsq mismatch for p={}",
                prime.p
            );
        }
    }

    #[test]
    fn verify_synthetic_i16_prime_derived_constants() {
        let primes = [
            NttPrime::compute(15361_i16),
            NttPrime::compute(13313_i16),
            NttPrime::compute(12289_i16),
        ];
        assert_i16_prime_profile(&primes);
    }

    #[test]
    fn verify_q32_prime_derived_constants() {
        assert_i32_prime_profile(&Q32_PRIMES);
    }

    #[test]
    fn verify_q128_primes_are_valid() {
        assert_i32_prime_profile(&q128_primes());
    }

    #[test]
    fn verify_q64_primes_are_valid() {
        assert_i32_prime_profile(&Q64_PRIMES);
    }

    #[test]
    fn garner_data_is_consistent() {
        assert_garner_profile("Q32", &Q32_PRIMES, q32_garner());
        assert_garner_profile("Q64", &Q64_PRIMES, q64_garner());
        let q128_primes = q128_primes();
        assert_garner_profile("Q128", &q128_primes, q128_garner());
    }
}
