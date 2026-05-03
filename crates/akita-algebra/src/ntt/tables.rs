//! Deterministic parameter presets for small-prime CRT arithmetic.
//!
//! Q32: `logq = 32` with six `i16` NTT-friendly primes (D ≤ 64).
//! Q64: `logq = 64` with `i32` NTT-friendly primes (D ≤ 1024).
//! Q128: `logq = 128` with five `i32` NTT-friendly primes (D ≤ 1024).

use super::crt::GarnerData;
use super::prime::NttPrime;

/// Polynomial degree for the base ring `Z_q[X]/(X^d + 1)`.
pub const RING_DEGREE: usize = 64;
/// Maximum ring degree covered by the i32 CRT parameter sets.
pub const MAX_CRT_RING_DEGREE: usize = 1024;

/// Number of CRT primes for the `logq = 32` parameter set.
pub const Q32_NUM_PRIMES: usize = 6;

/// The modulus `q = 2^32 - 99`.
pub const Q32_MODULUS: u64 = (1u64 << 32) - 99;

/// CRT primes and per-prime Montgomery constants for `logq = 32`.
///
/// All constants are for `R = 2^16` (i16 width).
pub const Q32_PRIMES: [NttPrime<i16>; Q32_NUM_PRIMES] = [
    NttPrime {
        p: 13697,
        pinv: 2689,
        mont: -2949,
        montsq: -994,
    },
    NttPrime {
        p: 13441,
        pinv: 2945,
        mont: -1669,
        montsq: 3274,
    },
    NttPrime {
        p: 13313,
        pinv: -13311,
        mont: -1029,
        montsq: -6199,
    },
    NttPrime {
        p: 12289,
        pinv: -12287,
        mont: 4091,
        montsq: -1337,
    },
    NttPrime {
        p: 12161,
        pinv: 4225,
        mont: 4731,
        montsq: -6040,
    },
    NttPrime {
        p: 11777,
        pinv: -11775,
        mont: -5126,
        montsq: 1389,
    },
];

/// Garner CRT reconstruction constants for Q32.
pub fn q32_garner() -> GarnerData<i16, Q32_NUM_PRIMES> {
    GarnerData::compute(&Q32_PRIMES)
}

/// Number of CRT primes for the `logq = 64` fast profile (`P > q`).
pub const Q64_NUM_PRIMES_FAST: usize = 3;
/// Number of CRT primes for the `logq = 64` conservative profile (`P > 128*q^2`).
pub const Q64_NUM_PRIMES: usize = 5;

/// The modulus `q = 2^64 - 59`.
pub const Q64_MODULUS: u64 = u64::MAX - 58;

/// Number of CRT primes for the `logq = 128` parameter set.
pub const Q128_NUM_PRIMES: usize = 5;

/// Protocol modulus `q = 2^128 - 275`.
pub const Q128_MODULUS: u128 = u128::MAX - 274;

/// Raw 30-bit primes for D≤1024, each satisfying `2048 | (p - 1)`.
///
/// They are ordered descending by value.
pub const D1024_RAW_PRIMES: [i32; Q128_NUM_PRIMES] =
    [1073707009, 1073698817, 1073692673, 1073682433, 1073668097];

/// Raw 30-bit primes for Q64 fast profile (`K=3`, `P > q`).
pub const Q64_RAW_PRIMES_FAST: [i32; Q64_NUM_PRIMES_FAST] = [
    D1024_RAW_PRIMES[0],
    D1024_RAW_PRIMES[1],
    D1024_RAW_PRIMES[2],
];

/// Raw 30-bit primes for Q64 conservative profile (`K=5`, `P > 128*q^2`).
pub const Q64_RAW_PRIMES: [i32; Q64_NUM_PRIMES] = D1024_RAW_PRIMES;

/// Raw 30-bit primes for Q128, each satisfying `2048 | (p - 1)`.
pub const Q128_RAW_PRIMES: [i32; Q128_NUM_PRIMES] = D1024_RAW_PRIMES;

/// CRT primes and per-prime Montgomery constants for `logq = 64` fast profile.
pub fn q64_primes_fast() -> [NttPrime<i32>; Q64_NUM_PRIMES_FAST] {
    std::array::from_fn(|k| NttPrime::compute(Q64_RAW_PRIMES_FAST[k]))
}

/// Garner CRT reconstruction constants for Q64 fast profile.
pub fn q64_garner_fast() -> GarnerData<i32, Q64_NUM_PRIMES_FAST> {
    let primes = q64_primes_fast();
    GarnerData::compute(&primes)
}

/// CRT primes and per-prime Montgomery constants for `logq = 64` conservative profile.
pub fn q64_primes() -> [NttPrime<i32>; Q64_NUM_PRIMES] {
    std::array::from_fn(|k| NttPrime::compute(Q64_RAW_PRIMES[k]))
}

/// Garner CRT reconstruction constants for Q64 conservative profile.
pub fn q64_garner() -> GarnerData<i32, Q64_NUM_PRIMES> {
    let primes = q64_primes();
    GarnerData::compute(&primes)
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

    #[test]
    fn verify_q32_prime_derived_constants() {
        for prime in &Q32_PRIMES {
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
    fn verify_q128_primes_are_valid() {
        let primes = q128_primes();
        for np in &primes {
            let p = np.p as i64;
            assert!(p > 1 && p % 2 == 1, "prime must be odd and > 1");
            assert_eq!(
                (p - 1) % 2048,
                0,
                "2048 must divide p-1 for D=1024 NTT (p={p})"
            );
            // Verify pinv: p * pinv ≡ 1 (mod 2^32)
            assert_eq!(
                np.p.wrapping_mul(np.pinv),
                1,
                "pinv verification failed for p={p}"
            );
        }
    }

    #[test]
    fn verify_q64_primes_are_valid() {
        let primes = q64_primes();
        for np in &primes {
            let p = np.p as i64;
            assert!(p > 1 && p % 2 == 1, "prime must be odd and > 1");
            assert_eq!(
                (p - 1) % 2048,
                0,
                "2048 must divide p-1 for D=1024 NTT (p={p})"
            );
            assert_eq!(
                np.p.wrapping_mul(np.pinv),
                1,
                "pinv verification failed for p={p}"
            );
        }
    }

    #[test]
    fn garner_data_is_consistent() {
        let garner = q32_garner();
        for (i, &prime_i) in Q32_PRIMES.iter().enumerate().skip(1) {
            let pi = prime_i.p as i64;
            for (j, &prime_j) in Q32_PRIMES[..i].iter().enumerate() {
                let pj = prime_j.p as i64;
                let g = garner.gamma[i][j] as i64;
                assert_eq!(
                    (pj * g) % pi,
                    1,
                    "garner gamma[{i}][{j}] not inverse of p_{j} mod p_{i}"
                );
            }
        }
    }
}
