//! Stable symbols used to inspect the production A7F7 addition and subtraction paths.

use akita_field::Prime128OffsetA7F7;

/// Two 64-bit limbs returned through the AArch64 C calling convention.
#[repr(C)]
pub struct Fp128Result {
    lo: u64,
    hi: u64,
}

/// Invoke public field addition for the proved A7F7 modulus.
///
/// The formal verification check requires this optimized symbol to contain the
/// expected modulus setup, the exact eight-instruction body proved by HOL
/// Light, and the return instruction.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn akita_fp128_add_production_witness(
    a_lo: u64,
    a_hi: u64,
    b_lo: u64,
    b_hi: u64,
) -> Fp128Result {
    let a = Prime128OffsetA7F7::from_canonical_u128((a_hi as u128) << 64 | a_lo as u128);
    let b = Prime128OffsetA7F7::from_canonical_u128((b_hi as u128) << 64 | b_lo as u128);
    let result = a + b;
    let [lo, hi] = result.to_limbs();
    Fp128Result { lo, hi }
}

/// Invoke public field subtraction for the proved A7F7 modulus.
///
/// The formal verification check requires this optimized symbol to contain the
/// expected modulus setup, the exact five-instruction body proved by HOL Light,
/// and the return instruction.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn akita_fp128_sub_production_witness(
    a_lo: u64,
    a_hi: u64,
    b_lo: u64,
    b_hi: u64,
) -> Fp128Result {
    let a = Prime128OffsetA7F7::from_canonical_u128((a_hi as u128) << 64 | a_lo as u128);
    let b = Prime128OffsetA7F7::from_canonical_u128((b_hi as u128) << 64 | b_lo as u128);
    let result = a - b;
    let [lo, hi] = result.to_limbs();
    Fp128Result { lo, hi }
}

fn main() {
    let add = std::hint::black_box(akita_fp128_add_production_witness(1, 0, 1, 0));
    let sub = std::hint::black_box(akita_fp128_sub_production_witness(1, 0, 1, 0));
    std::hint::black_box((add, sub));
}
