//! `2^k - offset` pseudo-Mersenne registry and field aliases.
//!
//! This module models the specific flavor where each modulus is the smallest
//! prime below `2^k` with `q % 8 == 5`, written as `q = 2^k - offset`.

use super::{Fp128, Fp32, Fp64};
use crate::PseudoMersenneField;

/// Offset table (`q = 2^k - offset[k]`) imported from `labrador/data.py`.
pub const POW2_OFFSET_TABLE: [i16; 256] = [
    -1, -1, -1, 3, 3, 3, 3, 19, 27, 3, 3, 19, 3, 75, 3, 19, 99, 91, 11, 19, 3, 19, 3, 27, 3, 91,
    27, 115, 299, 3, 35, 19, 99, 355, 131, 451, 243, 123, 107, 19, 195, 75, 11, 67, 539, 139, 635,
    115, 59, 123, 27, 139, 395, 315, 131, 67, 27, 195, 27, 99, 107, 259, 171, 259, 59, 115, 203,
    19, 83, 19, 35, 411, 107, 475, 35, 427, 123, 43, 11, 67, 1307, 51, 315, 139, 35, 19, 35, 67,
    299, 99, 75, 315, 83, 51, 3, 211, 147, 595, 51, 115, 99, 99, 483, 339, 395, 139, 1187, 171, 59,
    91, 195, 835, 75, 211, 11, 67, 3, 451, 563, 867, 395, 531, 3, 67, 59, 579, 203, 507, 275, 315,
    27, 315, 347, 99, 603, 795, 243, 339, 203, 187, 27, 171, 1491, 355, 83, 355, 1371, 387, 347,
    99, 3, 195, 539, 171, 243, 499, 195, 19, 155, 91, 75, 1011, 627, 867, 155, 115, 1811, 771,
    1467, 643, 195, 19, 155, 531, 3, 267, 563, 339, 563, 507, 107, 283, 267, 147, 59, 339, 371,
    1411, 363, 819, 11, 19, 915, 123, 75, 915, 459, 75, 627, 459, 75, 1035, 195, 187, 1515, 1219,
    1443, 91, 299, 451, 171, 1099, 99, 3, 395, 1147, 683, 675, 243, 355, 395, 3, 875, 235, 363,
    1131, 155, 835, 723, 91, 27, 235, 875, 3, 83, 259, 875, 1515, 731, 531, 467, 819, 267, 475,
    1923, 163, 107, 411, 387, 75, 2331, 355, 1515, 1723, 1427, 19,
];

/// Maximum supported offset in this `2^k - offset` specialization.
pub const POW2_OFFSET_MAX: u128 = 1u128 << 16;

/// Current active bit-size bound for concrete field aliases in this phase.
pub const POW2_OFFSET_IMPLEMENTED_MAX_BITS: u32 = 128;

/// Metadata describing a `2^k - offset` pseudo-Mersenne modulus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pow2OffsetPrimeSpec {
    /// `k` in `2^k - offset`.
    pub bits: u32,
    /// `offset` in `2^k - offset`.
    pub offset: u16,
    /// Modulus value.
    pub modulus: u128,
}

/// Return table offset for `q = 2^k - offset` when available and positive.
pub const fn pow2_offset(bits: u32) -> Option<u16> {
    if bits as usize >= POW2_OFFSET_TABLE.len() {
        return None;
    }
    let offset = POW2_OFFSET_TABLE[bits as usize];
    if offset <= 0 {
        None
    } else {
        Some(offset as u16)
    }
}

/// Compute `2^k - offset` for `k <= 128`.
pub const fn pseudo_mersenne_modulus(bits: u32, offset: u128) -> Option<u128> {
    if bits == 0 || bits > 128 || offset == 0 {
        return None;
    }
    if bits == 128 {
        Some(u128::MAX - (offset - 1))
    } else {
        Some((1u128 << bits) - offset)
    }
}

/// Check whether `(k, offset)` is accepted by the `2^k - offset` policy.
pub const fn is_pow2_offset(bits: u32, offset: u128) -> bool {
    if bits > POW2_OFFSET_IMPLEMENTED_MAX_BITS || offset > POW2_OFFSET_MAX {
        return false;
    }
    match pow2_offset(bits) {
        Some(qoff) => (qoff as u128) == offset,
        None => false,
    }
}

/// `offset` for `k = 24`.
pub const POW2_OFFSET_24: u16 = 3;
/// `offset` for `k = 32`.
pub const POW2_OFFSET_32: u16 = 99;
/// `offset` for `k = 40`.
pub const POW2_OFFSET_40: u16 = 195;
/// `offset` for `k = 48`.
pub const POW2_OFFSET_48: u16 = 59;
/// `offset` for `k = 56`.
pub const POW2_OFFSET_56: u16 = 27;
/// `offset` for `k = 64`.
pub const POW2_OFFSET_64: u16 = 59;
/// `offset` for `k = 128`.
pub const POW2_OFFSET_128: u16 = 275;

/// `2^24 - 3`.
pub const POW2_OFFSET_MODULUS_24: u32 = ((1u128 << 24) - (POW2_OFFSET_24 as u128)) as u32;
/// `2^32 - 99`.
pub const POW2_OFFSET_MODULUS_32: u32 = ((1u128 << 32) - (POW2_OFFSET_32 as u128)) as u32;
/// `2^40 - 195`.
pub const POW2_OFFSET_MODULUS_40: u64 = ((1u128 << 40) - (POW2_OFFSET_40 as u128)) as u64;
/// `2^48 - 59`.
pub const POW2_OFFSET_MODULUS_48: u64 = ((1u128 << 48) - (POW2_OFFSET_48 as u128)) as u64;
/// `2^56 - 27`.
pub const POW2_OFFSET_MODULUS_56: u64 = ((1u128 << 56) - (POW2_OFFSET_56 as u128)) as u64;
/// `2^64 - 59`.
pub const POW2_OFFSET_MODULUS_64: u64 = u64::MAX - ((POW2_OFFSET_64 as u64) - 1);
/// `2^128 - 275`.
pub const POW2_OFFSET_MODULUS_128: u128 = u128::MAX - (POW2_OFFSET_128 as u128 - 1);

/// Alias for `2^24 - offset`.
pub type Pow2Offset24Field = Fp32<POW2_OFFSET_MODULUS_24>;
/// Alias for `2^32 - offset`.
pub type Pow2Offset32Field = Fp32<POW2_OFFSET_MODULUS_32>;
/// Alias for `2^40 - offset`.
pub type Pow2Offset40Field = Fp64<POW2_OFFSET_MODULUS_40>;
/// Alias for `2^48 - offset`.
pub type Pow2Offset48Field = Fp64<POW2_OFFSET_MODULUS_48>;
/// Alias for `2^56 - offset`.
pub type Pow2Offset56Field = Fp64<POW2_OFFSET_MODULUS_56>;
/// Alias for `2^64 - offset`.
pub type Pow2Offset64Field = Fp64<POW2_OFFSET_MODULUS_64>;
/// Alias for `2^128 - offset`.
pub type Pow2Offset128Field = Fp128<POW2_OFFSET_MODULUS_128>;

/// `2^k - offset` profiles currently enabled in-code.
///
/// Each listed modulus satisfies `q % 8 == 5`.
pub const POW2_OFFSET_PRIMES: [Pow2OffsetPrimeSpec; 7] = [
    Pow2OffsetPrimeSpec {
        bits: 24,
        offset: POW2_OFFSET_24,
        modulus: POW2_OFFSET_MODULUS_24 as u128,
    },
    Pow2OffsetPrimeSpec {
        bits: 32,
        offset: POW2_OFFSET_32,
        modulus: POW2_OFFSET_MODULUS_32 as u128,
    },
    Pow2OffsetPrimeSpec {
        bits: 40,
        offset: POW2_OFFSET_40,
        modulus: POW2_OFFSET_MODULUS_40 as u128,
    },
    Pow2OffsetPrimeSpec {
        bits: 48,
        offset: POW2_OFFSET_48,
        modulus: POW2_OFFSET_MODULUS_48 as u128,
    },
    Pow2OffsetPrimeSpec {
        bits: 56,
        offset: POW2_OFFSET_56,
        modulus: POW2_OFFSET_MODULUS_56 as u128,
    },
    Pow2OffsetPrimeSpec {
        bits: 64,
        offset: POW2_OFFSET_64,
        modulus: POW2_OFFSET_MODULUS_64 as u128,
    },
    Pow2OffsetPrimeSpec {
        bits: 128,
        offset: POW2_OFFSET_128,
        modulus: POW2_OFFSET_MODULUS_128,
    },
];

impl PseudoMersenneField for Fp32<POW2_OFFSET_MODULUS_24> {
    const MODULUS_BITS: u32 = 24;
    const MODULUS_OFFSET: u128 = POW2_OFFSET_24 as u128;
}

impl PseudoMersenneField for Fp32<POW2_OFFSET_MODULUS_32> {
    const MODULUS_BITS: u32 = 32;
    const MODULUS_OFFSET: u128 = POW2_OFFSET_32 as u128;
}

impl PseudoMersenneField for Fp64<POW2_OFFSET_MODULUS_40> {
    const MODULUS_BITS: u32 = 40;
    const MODULUS_OFFSET: u128 = POW2_OFFSET_40 as u128;
}

impl PseudoMersenneField for Fp64<POW2_OFFSET_MODULUS_48> {
    const MODULUS_BITS: u32 = 48;
    const MODULUS_OFFSET: u128 = POW2_OFFSET_48 as u128;
}

impl PseudoMersenneField for Fp64<POW2_OFFSET_MODULUS_56> {
    const MODULUS_BITS: u32 = 56;
    const MODULUS_OFFSET: u128 = POW2_OFFSET_56 as u128;
}

impl PseudoMersenneField for Fp64<POW2_OFFSET_MODULUS_64> {
    const MODULUS_BITS: u32 = 64;
    const MODULUS_OFFSET: u128 = POW2_OFFSET_64 as u128;
}

impl PseudoMersenneField for Fp128<POW2_OFFSET_MODULUS_128> {
    const MODULUS_BITS: u32 = 128;
    const MODULUS_OFFSET: u128 = POW2_OFFSET_128 as u128;
}
