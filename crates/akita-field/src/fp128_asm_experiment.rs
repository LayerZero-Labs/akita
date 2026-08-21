//! Benchmark only access to the proof linkage experiment.

use core::arch::asm;

use crate::Fp128;

#[repr(C)]
struct Fp128AsmResult {
    lo: u64,
    hi: u64,
}

unsafe extern "C" {
    fn akita_fp128_add_asm(a_lo: u64, a_hi: u64, b_lo: u64, b_hi: u64, c: u64) -> Fp128AsmResult;
    fn akita_fp128_sub_asm(a_lo: u64, a_hi: u64, b_lo: u64, b_hi: u64, c: u64) -> Fp128AsmResult;
    fn akita_fp128_mul_asm(a_lo: u64, a_hi: u64, b_lo: u64, b_hi: u64, c: u64) -> Fp128AsmResult;
}

/// Add through the standalone AArch64 assembly symbol.
#[inline(always)]
pub fn add_linked<const P: u128>(a: Fp128<P>, b: Fp128<P>) -> Fp128<P> {
    let out = unsafe { akita_fp128_add_asm(a.0[0], a.0[1], b.0[0], b.0[1], Fp128::<P>::C_LO) };
    Fp128([out.lo, out.hi])
}

/// Subtract through the standalone AArch64 assembly symbol.
#[inline(always)]
pub fn sub_linked<const P: u128>(a: Fp128<P>, b: Fp128<P>) -> Fp128<P> {
    let out = unsafe { akita_fp128_sub_asm(a.0[0], a.0[1], b.0[0], b.0[1], Fp128::<P>::C_LO) };
    Fp128([out.lo, out.hi])
}

/// Multiply through the standalone AArch64 assembly symbol.
#[inline(always)]
pub fn mul_linked<const P: u128>(a: Fp128<P>, b: Fp128<P>) -> Fp128<P> {
    let out = unsafe { akita_fp128_mul_asm(a.0[0], a.0[1], b.0[0], b.0[1], Fp128::<P>::C_LO) };
    Fp128([out.lo, out.hi])
}

/// Add with the fixed register instruction body used by the standalone object.
#[inline(always)]
pub fn add_inline<const P: u128>(a: Fp128<P>, b: Fp128<P>) -> Fp128<P> {
    let [mut a_lo, mut a_hi] = a.0;
    let [b_lo, b_hi] = b.0;
    unsafe {
        asm!(
            include_str!("../asm/aarch64/fp128_add_body.inc"),
            inout("x0") a_lo,
            inout("x1") a_hi,
            in("x2") b_lo,
            in("x3") b_hi,
            in("x4") Fp128::<P>::C_LO,
            out("x5") _,
            out("x6") _,
            out("x7") _,
            out("x8") _,
            out("x9") _,
            options(pure, nomem, nostack),
        );
    }
    Fp128([a_lo, a_hi])
}

/// Subtract with the fixed register instruction body used by the standalone object.
#[inline(always)]
pub fn sub_inline<const P: u128>(a: Fp128<P>, b: Fp128<P>) -> Fp128<P> {
    let [mut a_lo, mut a_hi] = a.0;
    let [b_lo, b_hi] = b.0;
    unsafe {
        asm!(
            include_str!("../asm/aarch64/fp128_sub_body.inc"),
            inout("x0") a_lo,
            inout("x1") a_hi,
            in("x2") b_lo,
            in("x3") b_hi,
            in("x4") Fp128::<P>::C_LO,
            out("x5") _,
            out("x6") _,
            out("x7") _,
            options(pure, nomem, nostack),
        );
    }
    Fp128([a_lo, a_hi])
}

/// Multiply with the fixed register instruction body used by the standalone object.
#[inline(always)]
pub fn mul_inline<const P: u128>(a: Fp128<P>, b: Fp128<P>) -> Fp128<P> {
    let [mut a_lo, mut a_hi] = a.0;
    let [b_lo, b_hi] = b.0;
    unsafe {
        asm!(
            include_str!("../asm/aarch64/fp128_mul_body.inc"),
            inout("x0") a_lo,
            inout("x1") a_hi,
            in("x2") b_lo,
            in("x3") b_hi,
            in("x4") Fp128::<P>::C_LO,
            out("x5") _,
            out("x6") _,
            out("x7") _,
            out("x8") _,
            out("x9") _,
            out("x10") _,
            out("x11") _,
            out("x12") _,
            options(pure, nomem, nostack),
        );
    }
    Fp128([a_lo, a_hi])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Prime128OffsetA7F7 as F;

    const P: u128 = 0u128.wrapping_sub(F::C);
    const VALUES: [u128; 8] = [0, 1, 2, u64::MAX as u128, 1u128 << 64, P / 2, P - 2, P - 1];

    #[test]
    fn linked_and_fixed_register_kernels_match_production() {
        for a in VALUES {
            for b in VALUES {
                let a = F::from_canonical_u128(a);
                let b = F::from_canonical_u128(b);

                assert_eq!(add_linked(a, b), a + b);
                assert_eq!(sub_linked(a, b), a - b);
                assert_eq!(mul_linked(a, b), a * b);

                assert_eq!(add_inline(a, b), a + b);
                assert_eq!(sub_inline(a, b), a - b);
                assert_eq!(mul_inline(a, b), a * b);
            }
        }
    }
}
