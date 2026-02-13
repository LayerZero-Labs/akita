//! Deterministic parameter presets for small-prime CRT arithmetic.
//!
//! Initial scope: `logq = 32` preset with six NTT-friendly primes.

use super::crt::{LimbQ, QData};
use super::prime::NttPrime;

/// Polynomial degree used by Labrador's base ring (`X^64 + 1`).
pub const LABRADOR_N: usize = 64;
/// Number of CRT primes for `logq=32`.
pub const LABRADOR32_K: usize = 6;
/// Number of radix-`2^14` limbs for `logq=32`.
pub const LABRADOR32_L: usize = 3;
/// `q = 2^32 - 99`.
pub const LABRADOR32_QOFF: u16 = 99;
/// `log2(q)` target for the preset.
pub const LABRADOR32_LOGQ: u32 = 32;

/// Small CRT primes and arithmetic constants (`p`, `pinv`, `v`, `mont`, `montsq`, `s`, `f`, `t`).
pub const LABRADOR32_PRIMES: [NttPrime; LABRADOR32_K] = [
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

/// Limb constants for the `logq = 32` parameter set.
pub const LABRADOR32_QDATA: QData<LABRADOR32_K, LABRADOR32_L> = QData {
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
    logq: LABRADOR32_LOGQ,
    qoff: LABRADOR32_QOFF,
};

/// `q` value as `u64` for this preset.
#[inline]
pub const fn labrador32_q_u64() -> u64 {
    (1u64 << 32) - (LABRADOR32_QOFF as u64)
}
