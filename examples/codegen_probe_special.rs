#![allow(missing_docs)]

//! Codegen probe for packed/scalar Fp64 multiply kernels.
//!
//! Build with:
//! `cargo rustc --example codegen_probe_special --release -- --emit=asm`

use hachi_pcs::algebra::fields::pseudo_mersenne::{POW2_OFFSET_MODULUS_40, POW2_OFFSET_MODULUS_64};
use hachi_pcs::algebra::{Fp64, Fp64Packing, PackedValue};
use hachi_pcs::CanonicalField;

const MASK40: u64 = (1u64 << 40) - 1;
const P40: u64 = POW2_OFFSET_MODULUS_40;
const C40: u64 = (1u64 << 40) - P40; // 195
const P64: u64 = POW2_OFFSET_MODULUS_64;
const C64: u64 = 0u64.wrapping_sub(P64); // 59

#[inline(always)]
fn mul_c40_split(x: u64) -> u64 {
    let c = C40 as u32;
    let x_lo = x as u32;
    let x_hi = (x >> 32) as u32;
    (c as u64 * x_lo as u64).wrapping_add((c as u64 * x_hi as u64) << 32)
}

#[inline(always)]
fn mul_c40_shiftadd(x: u64) -> u64 {
    // 195x = (128 + 64 + 2 + 1) * x
    (x << 7)
        .wrapping_add(x << 6)
        .wrapping_add(x << 1)
        .wrapping_add(x)
}

#[inline(always)]
fn reduce40_with_mulc(lo: u64, hi: u64, mulc: fn(u64) -> u64) -> u64 {
    let high = (lo >> 40) | (hi << 24);
    let f1 = (lo & MASK40).wrapping_add(mulc(high));
    let f2 = (f1 & MASK40).wrapping_add(mulc(f1 >> 40));
    let reduced = f2.wrapping_sub(P40);
    let borrow = reduced >> 63;
    reduced.wrapping_add(borrow.wrapping_neg() & P40)
}

#[inline(always)]
fn reduce64(lo: u64, hi: u64) -> u64 {
    let f1 = (lo as u128) + (C64 as u128) * (hi as u128);
    let f2 = (f1 as u64 as u128) + (C64 as u128) * ((f1 >> 64) as u64 as u128);
    let reduced = f2.wrapping_sub(P64 as u128);
    let borrow = reduced >> 127;
    reduced.wrapping_add(borrow.wrapping_neg() & (P64 as u128)) as u64
}

#[inline(never)]
#[no_mangle]
pub extern "C" fn probe_reduce40_split(lo: u64, hi: u64) -> u64 {
    reduce40_with_mulc(lo, hi, mul_c40_split)
}

#[inline(never)]
#[no_mangle]
pub extern "C" fn probe_reduce40_shiftadd(lo: u64, hi: u64) -> u64 {
    reduce40_with_mulc(lo, hi, mul_c40_shiftadd)
}

#[inline(never)]
#[no_mangle]
pub extern "C" fn probe_reduce64(lo: u64, hi: u64) -> u64 {
    reduce64(lo, hi)
}

#[inline(never)]
#[no_mangle]
pub extern "C" fn probe_packed_fp64_40_mul(a0: u64, a1: u64, b0: u64, b1: u64) -> u64 {
    type F = Fp64<{ POW2_OFFSET_MODULUS_40 }>;
    type PF = Fp64Packing<{ POW2_OFFSET_MODULUS_40 }>;

    let a = PF::from_fn(|i| {
        if i == 0 {
            F::from_canonical_u64(a0)
        } else {
            F::from_canonical_u64(a1)
        }
    });
    let b = PF::from_fn(|i| {
        if i == 0 {
            F::from_canonical_u64(b0)
        } else {
            F::from_canonical_u64(b1)
        }
    });
    let c = a * b;
    (c.extract(0).to_canonical_u128() as u64) ^ (c.extract(1).to_canonical_u128() as u64)
}

#[inline(never)]
#[no_mangle]
pub extern "C" fn probe_packed_fp64_64_mul(a0: u64, a1: u64, b0: u64, b1: u64) -> u64 {
    type F = Fp64<{ POW2_OFFSET_MODULUS_64 }>;
    type PF = Fp64Packing<{ POW2_OFFSET_MODULUS_64 }>;

    let a = PF::from_fn(|i| {
        if i == 0 {
            F::from_canonical_u64(a0)
        } else {
            F::from_canonical_u64(a1)
        }
    });
    let b = PF::from_fn(|i| {
        if i == 0 {
            F::from_canonical_u64(b0)
        } else {
            F::from_canonical_u64(b1)
        }
    });
    let c = a * b;
    (c.extract(0).to_canonical_u128() as u64) ^ (c.extract(1).to_canonical_u128() as u64)
}

#[inline(never)]
#[no_mangle]
pub extern "C" fn probe_scalar_fp64_40_mul(a: u64, b: u64) -> u64 {
    type F = Fp64<{ POW2_OFFSET_MODULUS_40 }>;
    (F::from_canonical_u64(a) * F::from_canonical_u64(b)).to_canonical_u128() as u64
}

#[inline(never)]
#[no_mangle]
pub extern "C" fn probe_scalar_fp64_64_mul(a: u64, b: u64) -> u64 {
    type F = Fp64<{ POW2_OFFSET_MODULUS_64 }>;
    (F::from_canonical_u64(a) * F::from_canonical_u64(b)).to_canonical_u128() as u64
}

fn main() {
    let x = probe_packed_fp64_40_mul(1, 2, 3, 4)
        ^ probe_packed_fp64_64_mul(5, 6, 7, 8)
        ^ probe_scalar_fp64_40_mul(9, 10)
        ^ probe_scalar_fp64_64_mul(11, 12)
        ^ probe_reduce40_split(13, 14)
        ^ probe_reduce40_shiftadd(15, 16)
        ^ probe_reduce64(17, 18);
    std::hint::black_box(x);
}
