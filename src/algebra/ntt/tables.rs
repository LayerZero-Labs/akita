//! Deterministic parameter presets for small-prime CRT arithmetic.
//!
//! Initial scope: `logq = 32` preset with six NTT-friendly primes.

use super::crt::{LimbQ, QData};
use super::prime::NttPrime;

/// Polynomial degree for the base ring `Z_q[X]/(X^d + 1)`.
pub const RING_DEGREE: usize = 64;

/// Number of CRT primes for the `logq = 32` parameter set.
pub const Q32_NUM_PRIMES: usize = 6;

/// Number of radix-`2^14` limbs for `logq = 32`.
pub const Q32_NUM_LIMBS: usize = 3;

/// Offset `c` in `q = 2^32 - c`.
pub const Q32_OFFSET: u16 = 99;

/// `log2(q)` for the `logq = 32` parameter set.
pub const Q32_LOG_MODULUS: u32 = 32;

/// The modulus `q = 2^32 - 99`.
pub const Q32_MODULUS: u64 = (1u64 << 32) - (Q32_OFFSET as u64);

/// CRT primes and per-prime Montgomery constants for `logq = 32`.
pub const Q32_PRIMES: [NttPrime; Q32_NUM_PRIMES] = [
    NttPrime {
        p: 13697,
        pinv: 2689,
        v: 9799,
        mont: -2949,
        montsq: -994,
        s: 4705,
        f: 3540,
        t: -5758,
    },
    NttPrime {
        p: 13441,
        pinv: 2945,
        v: 9986,
        mont: -1669,
        montsq: 3274,
        s: 3777,
        f: -5468,
        t: -1680,
    },
    NttPrime {
        p: 13313,
        pinv: -13311,
        v: 10082,
        mont: -1029,
        montsq: -6199,
        s: 3325,
        f: 4553,
        t: -948,
    },
    NttPrime {
        p: 12289,
        pinv: -12287,
        v: 10922,
        mont: 4091,
        montsq: -1337,
        s: -3,
        f: 354,
        t: 4472,
    },
    NttPrime {
        p: 12161,
        pinv: 4225,
        v: 11037,
        mont: 4731,
        montsq: -6040,
        s: -383,
        f: 5993,
        t: 1653,
    },
    NttPrime {
        p: 11777,
        pinv: -11775,
        v: 11397,
        mont: -5126,
        montsq: 1389,
        s: -1475,
        f: -3812,
        t: 1191,
    },
];

/// CRT limb constants for the `logq = 32` parameter set.
pub const Q32_DATA: QData<Q32_NUM_PRIMES, Q32_NUM_LIMBS> = QData {
    q: LimbQ::from_limbs([16285, 16383, 15]),
    pmq: LimbQ::from_limbs([8747, 999, 12]),
    xvec: [
        LimbQ::from_limbs([9406, 14930, 2]),
        LimbQ::from_limbs([9295, 15936, 2]),
        LimbQ::from_limbs([16336, 1061, 5]),
        LimbQ::from_limbs([1006, 14584, 9]),
        LimbQ::from_limbs([1315, 5273, 10]),
        LimbQ::from_limbs([7927, 10169, 6]),
    ],
    logq: Q32_LOG_MODULUS,
    qoff: Q32_OFFSET,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that derived constants (`pinv`, `v`, `mont`, `montsq`) are
    /// consistent with the prime `p`. Turns the magic numbers into auditable
    /// values.
    #[test]
    fn verify_prime_derived_constants() {
        for prime in &Q32_PRIMES {
            let p = prime.p;

            // pinv: p * pinv ≡ 1 (mod 2^16)
            assert_eq!(p.wrapping_mul(prime.pinv), 1, "pinv failed for p={p}");

            // v: round(2^27 / p)
            let expected_v = ((1i64 << 27) + (p as i64 / 2)) / (p as i64);
            assert_eq!(prime.v as i64, expected_v, "v failed for p={p}");

            // mont: 2^16 mod p, centered to [-p/2, p/2)
            let raw_mont = ((1i32 << 16) % (p as i32)) as i16;
            let centered_mont = if raw_mont > p / 2 {
                raw_mont - p
            } else {
                raw_mont
            };
            assert_eq!(prime.mont, centered_mont, "mont failed for p={p}");

            // montsq: 2^32 mod p, centered to [-p/2, p/2)
            let raw_montsq = ((1i64 << 32) % (p as i64)) as i16;
            let centered_montsq = if raw_montsq > p / 2 {
                raw_montsq - p
            } else {
                raw_montsq
            };
            assert_eq!(prime.montsq, centered_montsq, "montsq failed for p={p}");
        }
    }

    #[test]
    fn verify_q_value() {
        assert_eq!(Q32_DATA.q_u128().unwrap(), Q32_MODULUS as u128);
    }
}
