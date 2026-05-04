//! 128-bit prime field for primes of the form `p = 2^128 − c` with `c < 2^32`.
//!
//! Uses Solinas-style two-fold reduction: no Montgomery form, ~23 cycles/mul
//! on both AArch64 and x86-64.  The offset `c` is computed at compile time
//! from the const-generic modulus `P`.
//!
//! ## Built-in primes
//!
//! Two built-in protocol primes are exposed:
//!
//! - `Prime128OffsetA7F7` (`p = 2^128 − 2^32 + 22537`, `C = 0xFFFFA7F7`),
//!   whose multiplicative group has a smooth subgroup of order
//!   `2^3 · 3^7 = 17 496` (with a clean radix-3 substructure of order
//!   `3^7 = 2187`). This is the default protocol prime.
//! - `Prime128Offset2355` (`p = 2^128 − 2355`), with smooth subgroup
//!   `2² · 3 · 5² · 7² = 14 700`, supported as a peer prime.
//!
//! A secondary split-NTT-only prime `Prime128Offset159`
//! (`p = 2^128 − 159`, `p ≡ 33 mod 64`) is kept for the algebra benchmark/test
//! path that only needs 32-way roots of unity.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use core::arch::asm;
use std::io::{Read, Write};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand_core::RngCore;

use crate::{
    AdditiveGroup, CanonicalField, FieldCore, FieldSampling, FromSmallInt, Invertible,
    PseudoMersenneField, SmoothFftField,
};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};

/// Pack two u64 limbs into `[lo, hi]`.
#[inline(always)]
const fn pack(lo: u64, hi: u64) -> [u64; 2] {
    [lo, hi]
}

/// Convert `u128` → `[u64; 2]`.
#[inline(always)]
const fn from_u128(x: u128) -> [u64; 2] {
    [x as u64, (x >> 64) as u64]
}

/// Convert `[u64; 2]` → `u128`.
#[inline(always)]
const fn to_u128(x: [u64; 2]) -> u128 {
    x[0] as u128 | (x[1] as u128) << 64
}

use super::util::{is_pow2_u64, log2_pow2_u64, mul64_wide};

/// 128-bit prime field element for primes `p = 2^128 − c` with `c < 2^32`.
///
/// Stored as `[u64; 2]` (lo, hi) for 8-byte alignment and direct limb access.
///
/// The offset `c = 2^128 − p` and all derived constants are computed at
/// compile time from the const-generic `P`.  Instantiating `Fp128` with a
/// modulus that is not of this form is a compile-time error.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fp128<const P: u128>(pub(crate) [u64; 2]);

impl<const P: u128> PartialEq for Fp128<P> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const P: u128> Eq for Fp128<P> {}

impl<const P: u128> Fp128<P> {
    /// Offset `c = 2^128 − p`.  Validated at compile time.
    pub const C: u128 = {
        let c = 0u128.wrapping_sub(P);
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        assert!(
            c < (1u128 << 32),
            "C must be < 2^32 (asm fold-2 uses single mul)"
        );
        assert!(
            c * (c + 1) < P,
            "C(C+1) < P required for fused canonicalize"
        );
        c
    };
    /// Low 64 bits of `C` (always equals `C` since `C < 2^32`).
    pub const C_LO: u64 = Self::C as u64;
    /// +1 means `C = 2^a + 1`, -1 means `C = 2^a - 1`, 0 means generic.
    const C_SHIFT_KIND: i8 = {
        let c = Self::C_LO;
        if c > 1 && is_pow2_u64(c - 1) {
            1
        } else if c == u64::MAX || is_pow2_u64(c + 1) {
            -1
        } else {
            0
        }
    };
    const C_SHIFT: u32 = {
        let c = Self::C_LO;
        if Self::C_SHIFT_KIND == 1 {
            log2_pow2_u64(c - 1)
        } else if Self::C_SHIFT_KIND == -1 {
            if c == u64::MAX {
                64
            } else {
                log2_pow2_u64(c + 1)
            }
        } else {
            0
        }
    };

    /// Multiply by `C = 2^128 - P`. For `C = 2^a ± 1`, this is shift/add or
    /// shift/sub only; otherwise it falls back to generic widening multiply.
    #[inline(always)]
    fn mul_c_wide(x: u64) -> (u64, u64) {
        if Self::C_SHIFT_KIND == 1 {
            let v = ((x as u128) << Self::C_SHIFT) + x as u128;
            (v as u64, (v >> 64) as u64)
        } else if Self::C_SHIFT_KIND == -1 {
            let v = ((x as u128) << Self::C_SHIFT) - x as u128;
            (v as u64, (v >> 64) as u64)
        } else {
            mul64_wide(Self::C_LO, x)
        }
    }

    /// Create from a canonical representative in `[0, p)`.
    #[inline]
    pub fn from_canonical_u128(x: u128) -> Self {
        debug_assert!(x < P);
        Self(from_u128(x))
    }

    /// Return the canonical representative in `[0, p)`.
    #[inline]
    pub fn to_canonical_u128(self) -> u128 {
        to_u128(self.0)
    }

    /// Const-evaluable `from_i64`. Embeds a small signed integer into `Fp`.
    pub const fn from_i64_const(val: i64) -> Self {
        if val >= 0 {
            Self(from_u128(val as u128))
        } else {
            Self(Self::sub_raw_portable(
                pack(0, 0),
                from_u128(val.unsigned_abs() as u128),
            ))
        }
    }

    /// Const-evaluable lookup table for balanced digits in `[-b/2, b/2)`
    /// where `b = 2^log_basis`. Requires `log_basis <= 6`.
    ///
    /// # Panics
    ///
    /// Panics if `log_basis` is outside `1..=6`.
    pub const fn digit_lut(log_basis: u32) -> [Self; 64] {
        assert!(log_basis > 0 && log_basis <= 6);
        let b = 1u32 << log_basis;
        let half_b = (b / 2) as i64;
        let mut lut = [Self(pack(0, 0)); 64];
        let mut i = 0u32;
        while i < b {
            lut[i as usize] = Self::from_i64_const(i as i64 - half_b);
            i += 1;
        }
        lut
    }

    #[inline(always)]
    fn add_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(target_arch = "aarch64")]
        {
            // On AArch64 we can keep the reduction predicate in flags via `ccmp`,
            // which is materially better than the generic `u128` lowering.
            Self::add_raw_aarch64_dispatch(a, b)
        }

        #[cfg(target_arch = "x86_64")]
        {
            // On x86-64, `sbb reg, reg` turns carry1 into a 0/-1 mask without
            // leaving flags. After computing `s + C`, one more `adc mask, mask`
            // makes ZF encode "need reduction", so the final select stays on
            // the flag path via `cmovne`.
            Self::add_raw_x86_64_dispatch(a, b)
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::add_raw_portable(a, b)
        }
    }

    #[cfg_attr(any(target_arch = "aarch64", target_arch = "x86_64"), allow(dead_code))]
    #[inline(always)]
    fn add_raw_portable(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // Compute s = a + b as two limbs.
        let (s0, carry0) = a[0].overflowing_add(b[0]);
        let (s1a, carry1a) = a[1].overflowing_add(b[1]);
        let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
        let overflow = carry1a | carry1b;

        // Since p = 2^128 - C and C < 2^64, reducing s modulo p is just
        // adding C into the low limb and propagating that carry.
        let (r0, carry2) = s0.overflowing_add(Self::C_LO);
        let (r1, carry3) = s1.overflowing_add(carry2 as u64);

        pack(
            if overflow | carry3 { r0 } else { s0 },
            if overflow | carry3 { r1 } else { s1 },
        )
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn add_raw_aarch64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // The immediate form is best when C < 4096 (the AArch64 add-immediate
        // encoding limit). Stable Rust does not let us feed `Self::C_LO`
        // directly into an `asm!(..., const ...)` operand, so the known
        // built-in offsets are spelled out here and everything else uses the
        // register form.
        match Self::C_LO {
            275 => Self::add_raw_aarch64_imm::<275>(a, b),
            159 => Self::add_raw_aarch64_imm::<159>(a, b),
            2355 => Self::add_raw_aarch64_imm::<2355>(a, b),
            _ => Self::add_raw_aarch64_reg(a, b, Self::C_LO),
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn add_raw_aarch64_imm<const C: u64>(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        unsafe {
            // carry1 is the overflow bit from a + b.
            // carry2 is the overflow bit from s + C, equivalently s >= p.
            // `ccmp` folds `carry1 | carry2` back into flags so the final
            // select stays branchless and never round-trips through GPR logic.
            asm!(
                "adds {s_lo}, {a_lo}, {b_lo}",
                "adcs {s_hi}, {a_hi}, {b_hi}",
                "cset {carry1:w}, hs",
                "adds {t_lo}, {s_lo}, #{c}",
                "adcs {t_hi}, {s_hi}, xzr",
                "ccmp {carry1:w}, #0, #0, lo",
                "csel {out_lo}, {t_lo}, {s_lo}, ne",
                "csel {out_hi}, {t_hi}, {s_hi}, ne",
                c = const C,
                a_lo = in(reg) a[0],
                a_hi = in(reg) a[1],
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                s_lo = out(reg) _,
                s_hi = out(reg) _,
                t_lo = out(reg) _,
                t_hi = out(reg) _,
                carry1 = out(reg) _,
                out_lo = lateout(reg) out_lo,
                out_hi = lateout(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn add_raw_aarch64_reg(a: [u64; 2], b: [u64; 2], c: u64) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        unsafe {
            // Same flag flow as the immediate path above, but with C supplied in
            // a register for offsets that are not encodable as add immediates.
            asm!(
                "adds {s_lo}, {a_lo}, {b_lo}",
                "adcs {s_hi}, {a_hi}, {b_hi}",
                "cset {carry1:w}, hs",
                "adds {t_lo}, {s_lo}, {c}",
                "adcs {t_hi}, {s_hi}, xzr",
                "ccmp {carry1:w}, #0, #0, lo",
                "csel {out_lo}, {t_lo}, {s_lo}, ne",
                "csel {out_hi}, {t_hi}, {s_hi}, ne",
                c = in(reg) c,
                a_lo = in(reg) a[0],
                a_hi = in(reg) a[1],
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                s_lo = out(reg) _,
                s_hi = out(reg) _,
                t_lo = out(reg) _,
                t_hi = out(reg) _,
                carry1 = out(reg) _,
                out_lo = lateout(reg) out_lo,
                out_hi = lateout(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn add_raw_x86_64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // As on AArch64, stable Rust does not let us feed `Self::C_LO`
        // directly into a const asm operand. The built-in offsets get the
        // immediate form and everything else uses the register form.
        match Self::C_LO {
            275 => Self::add_raw_x86_64_imm::<275>(a, b),
            159 => Self::add_raw_x86_64_imm::<159>(a, b),
            2355 => Self::add_raw_x86_64_imm::<2355>(a, b),
            // For C >= 2^31 the i32 immediate form is unusable: `add r64,
            // imm32` sign-extends the immediate, which would silently
            // corrupt the high limb. Such offsets fall through to the
            // register form below (`Prime128OffsetA7F7` lands here).
            _ => Self::add_raw_x86_64_reg(a, b, Self::C_LO),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn add_raw_x86_64_imm<const C: i32>(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let mut out_lo = a[0];
        let mut out_hi = a[1];
        let _mask: u64;
        let _t_lo: u64;
        let _t_hi: u64;
        unsafe {
            // After `s = a + b`, `sbb mask, mask` materializes carry1 as 0/-1.
            // After `t = s + C`, `adc mask, mask` leaves ZF=1 iff neither
            // carry1 nor carry2 was set. `cmovne` then picks `t` exactly when
            // reduction is needed.
            asm!(
                "add {out_lo}, {b_lo}",
                "adc {out_hi}, {b_hi}",
                "sbb {mask}, {mask}",
                "mov {t_lo}, {out_lo}",
                "mov {t_hi}, {out_hi}",
                "add {t_lo}, {c}",
                "adc {t_hi}, 0",
                "adc {mask}, {mask}",
                "cmovne {out_lo}, {t_lo}",
                "cmovne {out_hi}, {t_hi}",
                out_lo = inout(reg) out_lo,
                out_hi = inout(reg) out_hi,
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                mask = out(reg) _mask,
                t_lo = out(reg) _t_lo,
                t_hi = out(reg) _t_hi,
                c = const C,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn add_raw_x86_64_reg(a: [u64; 2], b: [u64; 2], c: u64) -> [u64; 2] {
        let mut out_lo = a[0];
        let mut out_hi = a[1];
        let _mask: u64;
        let _t_lo: u64;
        let _t_hi: u64;
        unsafe {
            asm!(
                "add {out_lo}, {b_lo}",
                "adc {out_hi}, {b_hi}",
                "sbb {mask}, {mask}",
                "mov {t_lo}, {out_lo}",
                "mov {t_hi}, {out_hi}",
                "add {t_lo}, {c}",
                "adc {t_hi}, 0",
                "adc {mask}, {mask}",
                "cmovne {out_lo}, {t_lo}",
                "cmovne {out_hi}, {t_hi}",
                out_lo = inout(reg) out_lo,
                out_hi = inout(reg) out_hi,
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                c = in(reg) c,
                mask = out(reg) _mask,
                t_lo = out(reg) _t_lo,
                t_hi = out(reg) _t_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[inline(always)]
    fn sub_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(target_arch = "aarch64")]
        {
            // The const path still uses `sub_raw_portable`, but at runtime on
            // AArch64 we can keep subtraction in limbs and reduce with `-C`
            // instead of materializing `P = 2^128 - C`.
            Self::sub_raw_aarch64_dispatch(a, b)
        }

        #[cfg(target_arch = "x86_64")]
        {
            // On x86-64, `sbb reg, reg` turns the final borrow into a 0/-1 mask.
            // Masking that with C lets us keep the same "select 0 or C, then do
            // one final subtract" structure that worked well on AArch64.
            Self::sub_raw_x86_64_dispatch(a, b)
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::sub_raw_portable(a, b)
        }
    }

    #[inline(always)]
    const fn sub_raw_portable(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let (diff, borrow) = to_u128(a).overflowing_sub(to_u128(b));
        from_u128(if borrow { diff.wrapping_add(P) } else { diff })
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn sub_raw_aarch64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // As in add_raw, stable Rust cannot feed `Self::C_LO` directly into a
        // `const` asm operand, so the built-in offsets get immediate forms and
        // everything else falls back to the register form.
        match Self::C_LO {
            275 => Self::sub_raw_aarch64_imm::<275>(a, b),
            159 => Self::sub_raw_aarch64_imm::<159>(a, b),
            2355 => Self::sub_raw_aarch64_imm::<2355>(a, b),
            _ => Self::sub_raw_aarch64_reg(a, b, Self::C_LO),
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn sub_raw_aarch64_imm<const C: u64>(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        unsafe {
            // If `a - b` borrows, then modulo `p = 2^128 - C` we need
            // `diff + p = diff - C (mod 2^128)`. Instead of round-tripping the
            // borrow bit through a GPR with `cset`/`cmp`, select the subtrahend
            // (`0` or `C`) directly from flags and do one final subtract.
            asm!(
                "mov {c_tmp}, #{c}",
                "subs {out_lo}, {a_lo}, {b_lo}",
                "sbcs {out_hi}, {a_hi}, {b_hi}",
                "csel {c_tmp}, xzr, {c_tmp}, hs",
                "subs {out_lo}, {out_lo}, {c_tmp}",
                "sbc {out_hi}, {out_hi}, xzr",
                c = const C,
                a_lo = in(reg) a[0],
                a_hi = in(reg) a[1],
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                c_tmp = out(reg) _,
                out_lo = out(reg) out_lo,
                out_hi = out(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn sub_raw_aarch64_reg(a: [u64; 2], b: [u64; 2], c: u64) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        unsafe {
            asm!(
                "subs {out_lo}, {a_lo}, {b_lo}",
                "sbcs {out_hi}, {a_hi}, {b_hi}",
                "csel {c_tmp}, xzr, {c}, hs",
                "subs {out_lo}, {out_lo}, {c_tmp}",
                "sbc {out_hi}, {out_hi}, xzr",
                c = in(reg) c,
                a_lo = in(reg) a[0],
                a_hi = in(reg) a[1],
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                c_tmp = out(reg) _,
                out_lo = out(reg) out_lo,
                out_hi = out(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn sub_raw_x86_64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // The immediate form keeps C out of the input register set for the
        // built-in offsets. Stable Rust does not let us pass `Self::C_LO`
        // directly as a const asm operand, so the known built-ins are spelled
        // out here and everything else uses the register form.
        match Self::C_LO {
            275 => Self::sub_raw_x86_64_imm::<275>(a, b),
            159 => Self::sub_raw_x86_64_imm::<159>(a, b),
            2355 => Self::sub_raw_x86_64_imm::<2355>(a, b),
            // See the matching note in `add_raw_x86_64_dispatch`: offsets
            // with C >= 2^31 cannot use the i32 immediate form because the
            // sign-extended `and r64, imm32` would corrupt the mask, so
            // they fall through to the register path here.
            _ => Self::sub_raw_x86_64_reg(a, b, Self::C_LO),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn sub_raw_x86_64_imm<const C: i32>(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let mut out_lo = a[0];
        let mut out_hi = a[1];
        unsafe {
            asm!(
                "sub {out_lo}, {b_lo}",
                "sbb {out_hi}, {b_hi}",
                "sbb {mask}, {mask}",
                "and {mask}, {c}",
                "sub {out_lo}, {mask}",
                "sbb {out_hi}, 0",
                out_lo = inout(reg) out_lo,
                out_hi = inout(reg) out_hi,
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                mask = out(reg) _,
                c = const C,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn sub_raw_x86_64_reg(a: [u64; 2], b: [u64; 2], c: u64) -> [u64; 2] {
        let mut out_lo = a[0];
        let mut out_hi = a[1];
        unsafe {
            asm!(
                "sub {out_lo}, {b_lo}",
                "sbb {out_hi}, {b_hi}",
                "sbb {mask}, {mask}",
                "and {mask}, {c}",
                "sub {out_lo}, {mask}",
                "sbb {out_hi}, 0",
                out_lo = inout(reg) out_lo,
                out_hi = inout(reg) out_hi,
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                c = in(reg) c,
                mask = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    /// Fold 2 + canonicalize: reduce `[t0, t1] + t2·2^128` into `[0, p)`.
    ///
    /// Correctness argument for the fused overflow+canonicalize:
    ///
    /// Let `v = base + C·t2` (mathematical, not mod 2^128).
    /// From the fold-1 mac chain, `t2 ≤ C`, so `C·t2 ≤ C²`.
    ///
    /// - **No overflow** (`v < 2^128`): `s = v`, and the standard
    ///   canonicalize applies — `s + C` carries iff `s ≥ P`.
    /// - **Overflow** (`v ≥ 2^128`): `s = v − 2^128`, so `s < C·t2 ≤ C²`.
    ///   The correct reduced value is `s + C` (since `2^128 ≡ C mod P`).
    ///   Because `s + C < C² + C = C(C+1)` and `C(C+1) < P` for all
    ///   `C < 2^64`, the value `s + C` is already in `[0, P)` — no
    ///   further canonicalization is needed, and `s + C < 2^128` so the
    ///   add does NOT carry.
    ///
    /// Therefore `if (overflow | carry) { s + C } else { s }` is correct
    /// in both cases, fusing the overflow correction with canonicalization.
    #[inline(always)]
    fn fold2_canonicalize(t0: u64, t1: u64, t2: u64) -> [u64; 2] {
        let (ct2_lo, ct2_hi) = Self::mul_c_wide(t2);

        let (s0, carry0) = t0.overflowing_add(ct2_lo);
        let (s1a, carry1a) = t1.overflowing_add(ct2_hi);
        let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
        let overflow = carry1a | carry1b;

        let (r0, carry2) = s0.overflowing_add(Self::C_LO);
        let (r1, carry3) = s1.overflowing_add(carry2 as u64);

        pack(
            if overflow | carry3 { r0 } else { s0 },
            if overflow | carry3 { r1 } else { s1 },
        )
    }

    /// Solinas fold for exactly 4 limbs: `[r0,r1] + C·[r2,r3]` → 3 limbs,
    /// then `fold2_canonicalize`.
    #[inline(always)]
    fn reduce_4(r0: u64, r1: u64, r2: u64, r3: u64) -> [u64; 2] {
        let (cr2_lo, cr2_hi) = Self::mul_c_wide(r2);
        let (cr3_lo, cr3_hi) = Self::mul_c_wide(r3);

        let t0_sum = r0 as u128 + cr2_lo as u128;
        let t0 = t0_sum as u64;
        let carryf = (t0_sum >> 64) as u64;

        let t1_sum = r1 as u128 + cr2_hi as u128 + cr3_lo as u128 + carryf as u128;
        let t1 = t1_sum as u64;

        let t2_sum = cr3_hi as u128 + (t1_sum >> 64);
        let t2 = t2_sum as u64;
        debug_assert_eq!(t2_sum >> 64, 0);

        Self::fold2_canonicalize(t0, t1, t2)
    }

    /// Add a canonical 128-bit value into a 256-bit little-endian limb array.
    ///
    /// Since both multiplicands and addends are canonical field elements,
    /// `a * b + c < 2^256`, so the top carry is guaranteed to be zero.
    #[inline(always)]
    fn add_128_into_256(prod: [u64; 4], addend: [u64; 2]) -> [u64; 4] {
        let (s0, carry0) = prod[0].overflowing_add(addend[0]);
        let (s1a, carry1a) = prod[1].overflowing_add(addend[1]);
        let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
        let carry1 = carry1a | carry1b;
        let (s2, carry2) = prod[2].overflowing_add(carry1 as u64);
        let (s3, carry3) = prod[3].overflowing_add(carry2 as u64);
        debug_assert!(!carry3);
        [s0, s1, s2, s3]
    }

    #[inline(always)]
    fn mul_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(target_arch = "aarch64")]
        {
            Self::mul_raw_aarch64(a, b)
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            Self::mul_raw_portable(a, b)
        }
    }

    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[inline(always)]
    fn mul_raw_portable(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let [r0, r1, r2, r3] = Self(a).mul_wide(Self(b));
        Self::reduce_4(r0, r1, r2, r3)
    }

    #[inline(always)]
    fn mul_add_raw(a: [u64; 2], b: [u64; 2], addend: [u64; 2]) -> [u64; 2] {
        #[cfg(target_arch = "aarch64")]
        {
            Self::mul_add_raw_aarch64(a, b, addend)
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            Self::mul_add_raw_portable(a, b, addend)
        }
    }

    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[inline(always)]
    fn mul_add_raw_portable(a: [u64; 2], b: [u64; 2], addend: [u64; 2]) -> [u64; 2] {
        let prod = Self(a).mul_wide(Self(b));
        let [s0, s1, s2, s3] = Self::add_128_into_256(prod, addend);
        Self::reduce_4(s0, s1, s2, s3)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn mul_add_raw_aarch64(a: [u64; 2], b: [u64; 2], addend: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        unsafe {
            asm!(
                // Schoolbook 2×2 → 256-bit product [r0,r1,r2,r3]
                "mul     {p00l}, {a0}, {b0}",
                "umulh   {p00h}, {a0}, {b0}",
                "mul     {p01l}, {a0}, {b1}",
                "umulh   {p01h}, {a0}, {b1}",
                "mul     {p10l}, {a1}, {b0}",
                "umulh   {p10h}, {a1}, {b0}",
                "mul     {p11l}, {a1}, {b1}",
                "umulh   {p11h}, {a1}, {b1}",

                // Carry accumulation into [r0=p00l, r1=p00h, r2=p01h, r3=p11h]
                "adds   {p00h}, {p00h}, {p01l}",
                "cset   {p01l:w}, hs",
                "adds   {p01h}, {p01h}, {p10h}",
                "cset   {p10h:w}, hs",
                "adds   {p01h}, {p01h}, {p11l}",
                "cinc   {p10h}, {p10h}, hs",
                "adds   {p00h}, {p00h}, {p10l}",
                "adcs   {p01h}, {p01h}, {p01l}",
                "adc    {p11h}, {p11h}, {p10h}",

                // Fuse the addend into the low 128 bits before the Solinas fold.
                "adds   {p00l}, {p00l}, {add_lo}",
                "adcs   {p00h}, {p00h}, {add_hi}",
                "adcs   {p01h}, {p01h}, xzr",
                "adc    {p11h}, {p11h}, xzr",

                // Fold-1: [t0,t1,t2] = [r0,r1] + C·[r2,r3]
                "mul    {p01l}, {p01h}, {c}",
                "umulh  {p10l}, {p01h}, {c}",
                "mul    {p10h}, {p11h}, {c}",
                "umulh  {p11l}, {p11h}, {c}",

                "adds   {p00l}, {p00l}, {p01l}",
                "adcs   {p00h}, {p00h}, {p10l}",
                "cset   {p01h:w}, hs",
                "adds   {p00h}, {p00h}, {p10h}",
                "adc    {p11h}, {p11l}, {p01h}",

                // Fold-2 + canonicalize via ccmp
                "mul    {p01l}, {p11h}, {c}",
                "adds   {p00l}, {p00l}, {p01l}",
                "adcs   {p00h}, {p00h}, xzr",
                "cset   {p01l:w}, hs",
                "adds   {p10l}, {p00l}, {c}",
                "adcs   {p10h}, {p00h}, xzr",
                "ccmp   {p01l:w}, #0, #0, lo",
                "csel   {out_lo}, {p10l}, {p00l}, ne",
                "csel   {out_hi}, {p10h}, {p00h}, ne",

                a0 = in(reg) a[0],
                a1 = in(reg) a[1],
                b0 = in(reg) b[0],
                b1 = in(reg) b[1],
                add_lo = in(reg) addend[0],
                add_hi = in(reg) addend[1],
                c = in(reg) Self::C_LO,
                p00l = out(reg) _,
                p00h = out(reg) _,
                p01l = out(reg) _,
                p01h = out(reg) _,
                p10l = out(reg) _,
                p10h = out(reg) _,
                p11l = out(reg) _,
                p11h = out(reg) _,
                out_lo = lateout(reg) out_lo,
                out_hi = lateout(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    /// 35-instruction AArch64 inline-asm multiply with Solinas reduction.
    ///
    /// Saves 6 instructions vs LLVM's codegen by:
    ///   - Fold-1 carry chain: direct adds/adcs/adc (5 vs 8 instructions),
    ///     avoiding intermediate cset/cinc shuttling of carries.
    ///   - Fold-2 + canonicalize: `ccmp` folds the overflow predicate with
    ///     the ≥p check (8 vs 10 instructions).
    ///
    /// Benchmarked at 1.29x throughput improvement on Apple M4.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn mul_raw_aarch64(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        unsafe {
            asm!(
                // Schoolbook 2×2 → 256-bit product [r0,r1,r2,r3]
                "mul     {p00l}, {a0}, {b0}",
                "umulh   {p00h}, {a0}, {b0}",
                "mul     {p01l}, {a0}, {b1}",
                "umulh   {p01h}, {a0}, {b1}",
                "mul     {p10l}, {a1}, {b0}",
                "umulh   {p10h}, {a1}, {b0}",
                "mul     {p11l}, {a1}, {b1}",
                "umulh   {p11h}, {a1}, {b1}",

                // Carry accumulation into [r0=p00l, r1=p00h, r2=p01h, r3=p11h]
                "adds   {p00h}, {p00h}, {p01l}",
                "cset   {p01l:w}, hs",
                "adds   {p01h}, {p01h}, {p10h}",
                "cset   {p10h:w}, hs",
                "adds   {p01h}, {p01h}, {p11l}",
                "cinc   {p10h}, {p10h}, hs",
                "adds   {p00h}, {p00h}, {p10l}",
                "adcs   {p01h}, {p01h}, {p01l}",
                "adc    {p11h}, {p11h}, {p10h}",

                // Fold-1: [t0,t1,t2] = [r0,r1] + C·[r2,r3]
                "mul    {p01l}, {p01h}, {c}",
                "umulh  {p10l}, {p01h}, {c}",
                "mul    {p10h}, {p11h}, {c}",
                "umulh  {p11l}, {p11h}, {c}",

                "adds   {p00l}, {p00l}, {p01l}",
                "adcs   {p00h}, {p00h}, {p10l}",
                "cset   {p01h:w}, hs",
                "adds   {p00h}, {p00h}, {p10h}",
                "adc    {p11h}, {p11l}, {p01h}",

                // Fold-2 + canonicalize via ccmp (C < 2^32 ⇒ C·t2 fits in 64 bits)
                "mul    {p01l}, {p11h}, {c}",
                "adds   {p00l}, {p00l}, {p01l}",
                "adcs   {p00h}, {p00h}, xzr",
                "cset   {p01l:w}, hs",
                "adds   {p10l}, {p00l}, {c}",
                "adcs   {p10h}, {p00h}, xzr",
                "ccmp   {p01l:w}, #0, #0, lo",
                "csel   {out_lo}, {p10l}, {p00l}, ne",
                "csel   {out_hi}, {p10h}, {p00h}, ne",

                a0 = in(reg) a[0],
                a1 = in(reg) a[1],
                b0 = in(reg) b[0],
                b1 = in(reg) b[1],
                c = in(reg) Self::C_LO,
                p00l = out(reg) _,
                p00h = out(reg) _,
                p01l = out(reg) _,
                p01h = out(reg) _,
                p10l = out(reg) _,
                p10h = out(reg) _,
                p11l = out(reg) _,
                p11h = out(reg) _,
                out_lo = lateout(reg) out_lo,
                out_hi = lateout(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[inline(always)]
    fn sqr_wide(self) -> [u64; 4] {
        let (a0, a1) = (self.0[0], self.0[1]);
        let (p00_lo, p00_hi) = mul64_wide(a0, a0);
        let (p01_lo, p01_hi) = mul64_wide(a0, a1);
        let (p11_lo, p11_hi) = mul64_wide(a1, a1);

        let row1 = p00_hi as u128 + (p01_lo as u128) * 2;
        let r0 = p00_lo;
        let r1 = row1 as u64;
        let carry1 = (row1 >> 64) as u64;

        let row2 = (p01_hi as u128) * 2 + p11_lo as u128 + carry1 as u128;
        let r2 = row2 as u64;
        let carry2 = (row2 >> 64) as u64;

        let row3 = p11_hi as u128 + carry2 as u128;
        let r3 = row3 as u64;
        debug_assert_eq!(row3 >> 64, 0);

        [r0, r1, r2, r3]
    }

    #[inline(always)]
    fn sqr_raw(a: [u64; 2]) -> [u64; 2] {
        #[cfg(target_arch = "aarch64")]
        {
            Self::sqr_raw_aarch64(a)
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            Self::sqr_raw_portable(a)
        }
    }

    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[inline(always)]
    fn sqr_raw_portable(a: [u64; 2]) -> [u64; 2] {
        let [r0, r1, r2, r3] = Self(a).sqr_wide();
        Self::reduce_4(r0, r1, r2, r3)
    }

    /// 31-instruction AArch64 inline-asm squaring with Solinas reduction.
    ///
    /// Uses 3 widening multiplies (vs 4 for general mul) and doubles the
    /// cross term via shifted-register operands. Same fold-1 + ccmp
    /// canonicalize as `mul_raw_aarch64`.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn sqr_raw_aarch64(a: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        unsafe {
            asm!(
                // Squaring schoolbook: 3 widening muls
                "mul     {p00l}, {a0}, {a0}",
                "umulh   {p00h}, {a0}, {a0}",
                "mul     {p01l}, {a0}, {a1}",
                "umulh   {p01h}, {a0}, {a1}",
                "mul     {p11l}, {a1}, {a1}",
                "umulh   {p11h}, {a1}, {a1}",

                // Carry accumulation with doubled cross term
                // row1 = p00h + 2*p01l, row2 = 2*p01h + p11l, r3 = p11h + carries
                "lsr    {t0}, {p01l}, #63",
                "lsr    {t1}, {p01h}, #63",
                "adds   {p01h}, {p11l}, {p01h}, lsl #1",
                "cinc   {t1}, {t1}, hs",
                "adds   {p00h}, {p00h}, {p01l}, lsl #1",
                "adcs   {p01h}, {p01h}, {t0}",
                "adc    {p11h}, {p11h}, {t1}",

                // At this point: r0=p00l, r1=p00h, r2=p01h, r3=p11h

                // Fold-1: [t0,t1,t2] = [r0,r1] + C·[r2,r3]
                "mul    {t0}, {p01h}, {c}",
                "umulh  {t1}, {p01h}, {c}",
                "mul    {p01l}, {p11h}, {c}",
                "umulh  {p11l}, {p11h}, {c}",

                "adds   {p00l}, {p00l}, {t0}",
                "adcs   {p00h}, {p00h}, {t1}",
                "cset   {t0:w}, hs",
                "adds   {p00h}, {p00h}, {p01l}",
                "adc    {p11h}, {p11l}, {t0}",

                // Fold-2 + canonicalize via ccmp (C < 2^32 ⇒ C·t2 fits in 64 bits)
                "mul    {t0}, {p11h}, {c}",
                "adds   {p00l}, {p00l}, {t0}",
                "adcs   {p00h}, {p00h}, xzr",
                "cset   {t0:w}, hs",
                "adds   {t1}, {p00l}, {c}",
                "adcs   {p01l}, {p00h}, xzr",
                "ccmp   {t0:w}, #0, #0, lo",
                "csel   {out_lo}, {t1}, {p00l}, ne",
                "csel   {out_hi}, {p01l}, {p00h}, ne",

                a0 = in(reg) a[0],
                a1 = in(reg) a[1],
                c = in(reg) Self::C_LO,
                p00l = out(reg) _,
                p00h = out(reg) _,
                p01l = out(reg) _,
                p01h = out(reg) _,
                p11l = out(reg) _,
                p11h = out(reg) _,
                t0 = out(reg) _,
                t1 = out(reg) _,
                out_lo = lateout(reg) out_lo,
                out_hi = lateout(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    /// Squaring, equivalent to `self * self`.
    #[inline(always)]
    pub fn square(self) -> Self {
        Self(Self::sqr_raw(self.0))
    }

    /// Fused multiply-add, equivalent to `self * rhs + addend`.
    ///
    /// This widens the product, adds the canonical addend before reduction,
    /// and performs a single final Solinas reduction.
    #[inline(always)]
    pub fn mul_add(self, rhs: Self, addend: Self) -> Self {
        Self(Self::mul_add_raw(self.0, rhs.0, addend.0))
    }

    fn pow_u128(self, mut exp: u128) -> Self {
        let mut base = self;
        let mut acc = Self::one();
        while exp > 0 {
            if (exp & 1) == 1 {
                acc *= base;
            }
            base = Self(Self::sqr_raw(base.0));
            exp >>= 1;
        }
        acc
    }

    /// Extract the canonical `[lo, hi]` limb representation.
    #[inline(always)]
    pub fn to_limbs(self) -> [u64; 2] {
        self.0
    }

    /// 128×64 → 192-bit widening multiply, **no reduction**.
    ///
    /// Returns `[lo, mid, hi]` representing `self · other` as a 192-bit
    /// integer.  Cost: 2 widening `mul64`.
    #[inline(always)]
    pub fn mul_wide_u64(self, other: u64) -> [u64; 3] {
        let (a0, a1) = (self.0[0], self.0[1]);
        let (p0_lo, p0_hi) = mul64_wide(a0, other);
        let (p1_lo, p1_hi) = mul64_wide(a1, other);
        let mid = p0_hi as u128 + p1_lo as u128;
        let hi = p1_hi + (mid >> 64) as u64;
        [p0_lo, mid as u64, hi]
    }

    /// 128×128 → 256-bit widening multiply, **no reduction**.
    ///
    /// Returns `[r0, r1, r2, r3]` representing `self · other` as a 256-bit
    /// integer.  This is the schoolbook 2×2 portion of the Solinas multiply,
    /// without the reduction fold.  Cost: 4 widening `mul64`.
    #[inline(always)]
    pub fn mul_wide(self, other: Self) -> [u64; 4] {
        let (a0, a1) = (self.0[0], self.0[1]);
        let (b0, b1) = (other.0[0], other.0[1]);
        let (p00_lo, p00_hi) = mul64_wide(a0, b0);
        let (p01_lo, p01_hi) = mul64_wide(a0, b1);
        let (p10_lo, p10_hi) = mul64_wide(a1, b0);
        let (p11_lo, p11_hi) = mul64_wide(a1, b1);

        let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
        let r0 = p00_lo;
        let r1 = row1 as u64;
        let carry1 = (row1 >> 64) as u64;

        let row2 = p01_hi as u128 + p10_hi as u128 + p11_lo as u128 + carry1 as u128;
        let r2 = row2 as u64;
        let carry2 = (row2 >> 64) as u64;

        let row3 = p11_hi as u128 + carry2 as u128;
        let r3 = row3 as u64;
        debug_assert_eq!(row3 >> 64, 0);

        [r0, r1, r2, r3]
    }

    /// 128×128 → 256-bit widening multiply with a raw `u128` operand,
    /// **no reduction**.
    #[inline(always)]
    pub fn mul_wide_u128(self, other: u128) -> [u64; 4] {
        self.mul_wide(Self(from_u128(other)))
    }

    /// 128×(64*M) → (64*OUT) widening multiply, **no reduction**.
    ///
    /// Multiplies a canonical Fp128 value (`[u64; 2]`) by an arbitrary
    /// little-endian limb array and returns the little-endian product
    /// truncated/extended to `OUT` limbs.
    #[inline(always)]
    pub fn mul_wide_limbs<const M: usize, const OUT: usize>(self, other: [u64; M]) -> [u64; OUT] {
        let (a0, a1) = (self.0[0], self.0[1]);

        // Hot-path specializations used by Jolt (M in {3,4}, OUT in {4,5}).
        // These avoid loop/control-flow overhead in tight sumcheck FMAs.
        if M == 3 && OUT == 5 {
            let b0 = other[0];
            let b1 = other[1];
            let b2 = other[2];

            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let (p12_lo, p12_hi) = mul64_wide(a1, b2);

            let r0 = p00_lo;

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let r1 = row1 as u64;
            let carry1 = row1 >> 64;

            let row2 = p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + carry1;
            let r2 = row2 as u64;
            let carry2 = row2 >> 64;

            let row3 = p02_hi as u128 + p11_hi as u128 + p12_lo as u128 + carry2;
            let r3 = row3 as u64;
            let carry3 = row3 >> 64;

            let row4 = p12_hi as u128 + carry3;
            let r4 = row4 as u64;
            debug_assert_eq!(row4 >> 64, 0);

            let mut out = [0u64; OUT];
            out[0] = r0;
            out[1] = r1;
            out[2] = r2;
            out[3] = r3;
            out[4] = r4;
            return out;
        }
        if M == 3 && OUT == 4 {
            let b0 = other[0];
            let b1 = other[1];
            let b2 = other[2];

            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let p12_lo = a1.wrapping_mul(b2);

            let r0 = p00_lo;

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let r1 = row1 as u64;
            let carry1 = row1 >> 64;

            let row2 = p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + carry1;
            let r2 = row2 as u64;
            let carry2 = row2 >> 64;

            let row3 = p02_hi as u128 + p11_hi as u128 + p12_lo as u128 + carry2;
            let r3 = row3 as u64;

            let mut out = [0u64; OUT];
            out[0] = r0;
            out[1] = r1;
            out[2] = r2;
            out[3] = r3;
            return out;
        }
        if M == 4 && OUT == 6 {
            let b0 = other[0];
            let b1 = other[1];
            let b2 = other[2];
            let b3 = other[3];

            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let (p03_lo, p03_hi) = mul64_wide(a0, b3);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let (p12_lo, p12_hi) = mul64_wide(a1, b2);
            let (p13_lo, p13_hi) = mul64_wide(a1, b3);

            let r0 = p00_lo;

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let r1 = row1 as u64;
            let carry1 = row1 >> 64;

            let row2 = p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + carry1;
            let r2 = row2 as u64;
            let carry2 = row2 >> 64;

            let row3 = p02_hi as u128 + p03_lo as u128 + p11_hi as u128 + p12_lo as u128 + carry2;
            let r3 = row3 as u64;
            let carry3 = row3 >> 64;

            let row4 = p03_hi as u128 + p12_hi as u128 + p13_lo as u128 + carry3;
            let r4 = row4 as u64;
            let carry4 = row4 >> 64;

            let row5 = p13_hi as u128 + carry4;
            let r5 = row5 as u64;
            debug_assert_eq!(row5 >> 64, 0);

            let mut out = [0u64; OUT];
            out[0] = r0;
            out[1] = r1;
            out[2] = r2;
            out[3] = r3;
            out[4] = r4;
            out[5] = r5;
            return out;
        }
        if M == 4 && OUT == 5 {
            let b0 = other[0];
            let b1 = other[1];
            let b2 = other[2];
            let b3 = other[3];

            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let (p03_lo, p03_hi) = mul64_wide(a0, b3);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let (p12_lo, p12_hi) = mul64_wide(a1, b2);
            let p13_lo = a1.wrapping_mul(b3);

            let r0 = p00_lo;

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let r1 = row1 as u64;
            let carry1 = row1 >> 64;

            let row2 = p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + carry1;
            let r2 = row2 as u64;
            let carry2 = row2 >> 64;

            let row3 = p02_hi as u128 + p03_lo as u128 + p11_hi as u128 + p12_lo as u128 + carry2;
            let r3 = row3 as u64;
            let carry3 = row3 >> 64;

            let row4 = p03_hi as u128 + p12_hi as u128 + p13_lo as u128 + carry3;
            let r4 = row4 as u64;

            let mut out = [0u64; OUT];
            out[0] = r0;
            out[1] = r1;
            out[2] = r2;
            out[3] = r3;
            out[4] = r4;
            return out;
        }
        if M == 4 && OUT == 4 {
            let b0 = other[0];
            let b1 = other[1];
            let b2 = other[2];
            let b3 = other[3];

            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let p03_lo = a0.wrapping_mul(b3);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let p12_lo = a1.wrapping_mul(b2);

            let r0 = p00_lo;

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let r1 = row1 as u64;
            let carry1 = row1 >> 64;

            let row2 = p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + carry1;
            let r2 = row2 as u64;
            let carry2 = row2 >> 64;

            let row3 = p02_hi as u128 + p03_lo as u128 + p11_hi as u128 + p12_lo as u128 + carry2;
            let r3 = row3 as u64;

            let mut out = [0u64; OUT];
            out[0] = r0;
            out[1] = r1;
            out[2] = r2;
            out[3] = r3;
            return out;
        }

        let mut out = [0u64; OUT];

        for (i, &b) in other.iter().enumerate() {
            if i >= OUT {
                break;
            }

            let (p0_lo, p0_hi) = mul64_wide(a0, b);
            let (p1_lo, p1_hi) = mul64_wide(a1, b);

            let s0 = out[i] as u128 + p0_lo as u128;
            out[i] = s0 as u64;
            let mut carry = s0 >> 64;

            if i + 1 >= OUT {
                continue;
            }
            let s1 = out[i + 1] as u128 + p0_hi as u128 + p1_lo as u128 + carry;
            out[i + 1] = s1 as u64;
            carry = s1 >> 64;

            if i + 2 >= OUT {
                continue;
            }
            let s2 = out[i + 2] as u128 + p1_hi as u128 + carry;
            out[i + 2] = s2 as u64;

            let mut carry_hi = s2 >> 64;
            let mut j = i + 3;
            while carry_hi != 0 && j < OUT {
                let sj = out[j] as u128 + carry_hi;
                out[j] = sj as u64;
                carry_hi = sj >> 64;
                j += 1;
            }
        }

        out
    }

    /// Reduce an arbitrary-width little-endian limb array to a canonical
    /// field element via iterated Solinas folding.
    ///
    /// Each fold splits at the 128-bit boundary and replaces
    /// `hi · 2^128` with `hi · C`, reducing width by one limb per
    /// iteration.  Supports 0–10 input limbs (up to 640 bits).
    ///
    /// # Panics
    ///
    /// Panics if `limbs.len() > 10`.
    #[inline(always)]
    pub fn solinas_reduce(limbs: &[u64]) -> Self {
        match limbs.len() {
            0 => Self::zero(),
            1 => Self(pack(limbs[0], 0)),
            2 => Self::from_canonical_u128_reduced(to_u128([limbs[0], limbs[1]])),
            3 => Self(Self::fold2_canonicalize(limbs[0], limbs[1], limbs[2])),
            4 => Self(Self::reduce_4(limbs[0], limbs[1], limbs[2], limbs[3])),
            5 => {
                let (l0, l1, l2, l3, l4) = (limbs[0], limbs[1], limbs[2], limbs[3], limbs[4]);
                let (c2_lo, c2_hi) = Self::mul_c_wide(l2);
                let (c3_lo, c3_hi) = Self::mul_c_wide(l3);
                let (c4_lo, c4_hi) = Self::mul_c_wide(l4);

                let s0 = l0 as u128 + c2_lo as u128;
                let s1 = l1 as u128 + c2_hi as u128 + c3_lo as u128 + (s0 >> 64);
                let s2 = c3_hi as u128 + c4_lo as u128 + (s1 >> 64);
                let s3 = c4_hi as u128 + (s2 >> 64);
                debug_assert_eq!(s3 >> 64, 0);

                Self(Self::reduce_4(s0 as u64, s1 as u64, s2 as u64, s3 as u64))
            }
            n => {
                assert!(n <= 10, "solinas_reduce supports at most 10 limbs");
                let mut buf = [0u64; 11];
                buf[..n].copy_from_slice(limbs);
                let mut len = n;
                let c = Self::C_LO;

                while len > 5 {
                    let high_len = len - 2;
                    let mut next = [0u64; 11];

                    let mut carry: u64 = 0;
                    for i in 0..high_len {
                        let wide = c as u128 * buf[i + 2] as u128 + carry as u128;
                        next[i] = wide as u64;
                        carry = (wide >> 64) as u64;
                    }
                    next[high_len] = carry;

                    let s0 = next[0] as u128 + buf[0] as u128;
                    next[0] = s0 as u64;
                    let s1 = next[1] as u128 + buf[1] as u128 + (s0 >> 64);
                    next[1] = s1 as u64;
                    let mut c_out = (s1 >> 64) as u64;
                    for limb in &mut next[2..=high_len] {
                        if c_out == 0 {
                            break;
                        }
                        let s = *limb as u128 + c_out as u128;
                        *limb = s as u64;
                        c_out = (s >> 64) as u64;
                    }
                    debug_assert_eq!(c_out, 0);

                    buf = next;
                    len -= 1;
                    while len > 5 && buf[len - 1] == 0 {
                        len -= 1;
                    }
                }

                Self::solinas_reduce(&buf[..len])
            }
        }
    }
}

impl<const P: u128> Add for Fp128<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(Self::add_raw(self.0, rhs.0))
    }
}

impl<const P: u128> Sub for Fp128<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(Self::sub_raw(self.0, rhs.0))
    }
}

impl<const P: u128> Mul for Fp128<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(Self::mul_raw(self.0, rhs.0))
    }
}

impl<const P: u128> Neg for Fp128<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(Self::sub_raw(pack(0, 0), self.0))
    }
}

impl<const P: u128> AddAssign for Fp128<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u128> SubAssign for Fp128<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u128> MulAssign for Fp128<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, const P: u128> Add<&'a Self> for Fp128<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, const P: u128> Sub<&'a Self> for Fp128<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, const P: u128> Mul<&'a Self> for Fp128<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<const P: u128> Valid for Fp128<P> {
    fn check(&self) -> Result<(), SerializationError> {
        if to_u128(self.0) < P {
            Ok(())
        } else {
            Err(SerializationError::InvalidData("Fp128 out of range".into()))
        }
    }
}

impl<const P: u128> AkitaSerialize for Fp128<P> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        to_u128(self.0).serialize_with_mode(&mut writer, Compress::No)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        16
    }
}

impl<const P: u128> AkitaDeserialize for Fp128<P> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let x = u128::deserialize_with_mode(&mut reader, Compress::No, validate, &())?;
        if matches!(validate, Validate::Yes) && x >= P {
            return Err(SerializationError::InvalidData(
                "Fp128 out of range".to_string(),
            ));
        }

        // Without validation, reduce without division.
        // For `p = 2^128 − c` with `c < 2^64` we have `p > 2^127`,
        // so any `u128` is in `[0, 2p)` and one conditional subtract suffices.
        let out = if matches!(validate, Validate::Yes) {
            x
        } else {
            let (sub, borrow) = x.overflowing_sub(P);
            if borrow {
                x
            } else {
                sub
            }
        };
        Ok(Self(from_u128(out)))
    }
}

impl<const P: u128> AdditiveGroup for Fp128<P> {
    const ZERO: Self = Self(pack(0, 0));
}

impl<const P: u128> FieldCore for Fp128<P> {
    fn one() -> Self {
        Self(pack(1, 0))
    }

    fn is_zero(&self) -> bool {
        self.0 == [0, 0]
    }

    fn inv(self) -> Option<Self> {
        let inv = self.inv_or_zero();
        if self.is_zero() {
            None
        } else {
            Some(inv)
        }
    }

    const TWO_INV: Self = {
        let v = (P >> 1) + 1;
        Self(pack(v as u64, (v >> 64) as u64))
    };
}

impl<const P: u128> Invertible for Fp128<P> {
    fn inv_or_zero(self) -> Self {
        let candidate = self.pow_u128(P.wrapping_sub(2));
        let v = to_u128(self.0);
        let nz = ((v | v.wrapping_neg()) >> 127) & 1;
        let mask = 0u128.wrapping_sub(nz);
        let masked = to_u128(candidate.0) & mask;
        Self(from_u128(masked))
    }
}

impl<const P: u128> FieldSampling for Fp128<P> {
    fn sample<R: RngCore>(rng: &mut R) -> Self {
        loop {
            let lo = rng.next_u64();
            let hi = rng.next_u64();
            let x = lo as u128 | (hi as u128) << 64;
            if x < P {
                return Self(pack(lo, hi));
            }
        }
    }
}

impl<const P: u128> FromSmallInt for Fp128<P> {
    fn from_u64(val: u64) -> Self {
        // For Fp128 pseudo-Mersenne primes, p = 2^128 - c with c < 2^64.
        // Therefore any u64 is always canonical (< p), so this can be a
        // direct limb construction with no reduction path.
        Self(from_u128(val as u128))
    }

    fn from_i64(val: i64) -> Self {
        Self::from_i64_const(val)
    }

    fn digit_lut(log_basis: u32) -> [Self; 64] {
        Self::digit_lut(log_basis)
    }
}

impl<const P: u128> CanonicalField for Fp128<P> {
    fn to_canonical_u128(self) -> u128 {
        to_u128(self.0)
    }

    fn from_canonical_u128_checked(val: u128) -> Option<Self> {
        if val < P {
            Some(Self(from_u128(val)))
        } else {
            None
        }
    }

    fn from_canonical_u128_reduced(val: u128) -> Self {
        let (sub, borrow) = val.overflowing_sub(P);
        Self(from_u128(if borrow { val } else { sub }))
    }
}

impl<const P: u128> PseudoMersenneField for Fp128<P> {
    const MODULUS_BITS: u32 = 128;
    const MODULUS_OFFSET: u128 = Self::C;
}

/// `p = 2^128 − 275`  (C = 275).
pub type Prime128Offset275 = Fp128<0xfffffffffffffffffffffffffffffeed>;
/// `p = 2^128 − 159`  (C = 159). Split-NTT-only helper prime.
pub type Prime128Offset159 = Fp128<0xffffffffffffffffffffffffffffff61>;
/// `p = 2^128 − 2355`  (C = 2355, p ≡ 5 mod 8).
///
/// Smooth multiplicative subgroup of order 14700 = 2² × 3 × 5² × 7²,
/// supporting mixed-radix FFT up to size 14700 (e.g. 1470 = 2·3·5·7²
/// for RS encoding with 256+1024 ≥ 1280 evaluations).
///
/// Factorization: `p − 1 = 2² · 3 · 5² · 7² · 701 · 2955365183 · 11173595356596918495491`.
pub type Prime128Offset2355 = Fp128<0xfffffffffffffffffffffffffffff6cd>;

impl SmoothFftField for Prime128Offset2355 {
    const SMOOTH_SUBGROUP_ORDER: usize = 14_700;
    /// `2 ^ ((p − 1) / 14_700)` where `g = 2` is a primitive root of `p`.
    /// Verified by `prime_2355_tests::smooth_omega_matches_search` in
    /// `src/algebra/fields/fft.rs`.
    const SMOOTH_OMEGA: u128 = 0x2ecd_18d0_8238_2c0c_818c_c05f_446a_8075;
}

/// `p = 2^128 − 2^32 + 22537`  (C = 2^32 − 22537 = 0xFFFFA7F7).
///
/// Solinas-form prime sharing the same CPU reduction cost as
/// `Prime128Offset2355` on x86_64 / AArch64 (both go through the generic
/// 32-bit-C `mul_c_wide` path; neither C is of the form `2^a ± 1`). The
/// multiplicative group contains a smooth subgroup of order
/// `2^3 · 3^7 = 17 496` with a pure radix-3 subgroup of order
/// `3^7 = 2187`, enabling a low-mul mixed-radix FFT.
///
/// Factorization of `p − 1` includes `2^3 · 3^7 · 19 · 41 · 459 647 · …`.
///
/// Subgroup sizes available for FFT-based RS encoding include
/// `1458 = 2 · 3^6`, `2187 = 3^7`, `4374 = 2 · 3^7`, `8748 = 2^2 · 3^7`,
/// and the full `17 496 = 2^3 · 3^7`.
pub type Prime128OffsetA7F7 = Fp128<0xffffffffffffffffffffffff00005809>;

impl SmoothFftField for Prime128OffsetA7F7 {
    const SMOOTH_SUBGROUP_ORDER: usize = 17_496;
    /// `g ^ ((p − 1) / 17_496)` where `g` is the smallest primitive root
    /// found by `find_primitive_nth_root` (note: `g = 2` is a quadratic
    /// residue mod `p` and therefore *not* a primitive root, so the
    /// scanner falls through to the next candidate). Verified by
    /// `prime_a7f7_tests::smooth_omega_matches_search` in
    /// `src/algebra/fields/fft.rs`.
    const SMOOTH_OMEGA: u128 = 0x4e9f_650b_7003_d201_9945_e1da_c47c_8b18;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FieldSampling, PseudoMersenneField};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_core::RngCore;

    type F = Prime128Offset275;

    #[test]
    fn to_limbs_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0xdead_beef_cafe_1234);
        for _ in 0..1000 {
            let a: F = FieldSampling::sample(&mut rng);
            assert_eq!(Fp128(a.to_limbs()), a);
        }
    }

    #[test]
    fn mul_wide_u64_matches_full_mul() {
        let mut rng = StdRng::seed_from_u64(0x1122_3344_5566_7788);
        for _ in 0..1000 {
            let a: F = FieldSampling::sample(&mut rng);
            let b = rng.next_u64();
            let expected = a * F::from_u64(b);
            let reduced = F::solinas_reduce(&a.mul_wide_u64(b));
            assert_eq!(reduced, expected);
        }
    }

    #[test]
    fn mul_wide_matches_full_mul() {
        let mut rng = StdRng::seed_from_u64(0xaabb_ccdd_eeff_0011);
        for _ in 0..1000 {
            let a: F = FieldSampling::sample(&mut rng);
            let b: F = FieldSampling::sample(&mut rng);
            let expected = a * b;
            let reduced = F::solinas_reduce(&a.mul_wide(b));
            assert_eq!(reduced, expected);
        }
    }

    #[test]
    fn mul_add_matches_mul_then_add() {
        let mut rng = StdRng::seed_from_u64(0x3141_5926_5358_9793);
        for _ in 0..1000 {
            let a: F = FieldSampling::sample(&mut rng);
            let b: F = FieldSampling::sample(&mut rng);
            let c: F = FieldSampling::sample(&mut rng);
            assert_eq!(a.mul_add(b, c), a * b + c);
        }

        let near = -F::one();
        assert_eq!(near.mul_add(near, near), near * near + near);
    }

    #[test]
    fn mul_wide_u128_matches_full_mul() {
        let mut rng = StdRng::seed_from_u64(0x9988_7766_5544_3322);
        for _ in 0..1000 {
            let a: F = FieldSampling::sample(&mut rng);
            let b = rng.next_u64() as u128 | ((rng.next_u64() as u128) << 64);
            let expected = a * F::from_canonical_u128_reduced(b);
            let reduced = F::solinas_reduce(&a.mul_wide_u128(b));
            assert_eq!(reduced, expected);
        }
    }

    #[test]
    fn mul_wide_limbs_roundtrips_through_reduction() {
        let mut rng = StdRng::seed_from_u64(0x1bad_f00d_0ddc_afe1);
        for _ in 0..1000 {
            let a: F = FieldSampling::sample(&mut rng);
            let b3 = [rng.next_u64(), rng.next_u64(), rng.next_u64()];
            let b4 = [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ];

            let got3_full = a.mul_wide_limbs::<3, 5>(b3);
            let got3_trunc = a.mul_wide_limbs::<3, 4>(b3);
            assert_eq!(
                got3_trunc,
                [got3_full[0], got3_full[1], got3_full[2], got3_full[3]]
            );
            let exp3 = a * F::solinas_reduce(&b3);
            assert_eq!(F::solinas_reduce(&got3_full), exp3);

            let got4_full = a.mul_wide_limbs::<4, 6>(b4);
            let got4_trunc = a.mul_wide_limbs::<4, 4>(b4);
            assert_eq!(
                got4_trunc,
                [got4_full[0], got4_full[1], got4_full[2], got4_full[3]]
            );
            let exp4 = a * F::solinas_reduce(&b4);
            assert_eq!(F::solinas_reduce(&got4_full), exp4);
        }
    }

    #[test]
    fn solinas_reduce_small_inputs() {
        assert_eq!(F::solinas_reduce(&[]), F::zero());
        assert_eq!(F::solinas_reduce(&[42]), F::from_u64(42));
        let one_shifted = F::from_canonical_u128_reduced(1u128 << 64);
        assert_eq!(F::solinas_reduce(&[0, 1]), one_shifted);
    }

    #[test]
    fn solinas_reduce_4_limbs_max() {
        // 2^256 - 1 ≡ C² - 1 (mod P), since 2^128 ≡ C
        let c = F::from_canonical_u128_reduced(<F as PseudoMersenneField>::MODULUS_OFFSET);
        let expected = c * c - F::one();
        assert_eq!(F::solinas_reduce(&[u64::MAX; 4]), expected);
    }

    #[test]
    fn solinas_reduce_9_limbs() {
        // 1 + 2^512 = 1 + (2^128)^4 ≡ 1 + C^4
        let c = F::from_canonical_u128_reduced(<F as PseudoMersenneField>::MODULUS_OFFSET);
        let expected = F::one() + c * c * c * c;
        assert_eq!(F::solinas_reduce(&[1, 0, 0, 0, 0, 0, 0, 0, 1]), expected);
    }

    #[test]
    fn solinas_reduce_accumulated_products() {
        let mut rng = StdRng::seed_from_u64(0xfeed_face_0bad_c0de);
        let mut acc = [0u64; 5];
        let mut expected = F::zero();

        for _ in 0..200 {
            let a: F = FieldSampling::sample(&mut rng);
            let b = rng.next_u64();
            let wide = a.mul_wide_u64(b);

            let mut carry: u64 = 0;
            for j in 0..5 {
                let addend = if j < 3 { wide[j] } else { 0 };
                let sum = acc[j] as u128 + addend as u128 + carry as u128;
                acc[j] = sum as u64;
                carry = (sum >> 64) as u64;
            }
            assert_eq!(carry, 0);
            expected += a * F::from_u64(b);
        }

        assert_eq!(F::solinas_reduce(&acc), expected);
    }

    #[test]
    fn solinas_reduce_cross_prime() {
        type G = Prime128Offset275;
        let c = G::from_canonical_u128_reduced(<G as PseudoMersenneField>::MODULUS_OFFSET);
        let expected = c * c - G::one();
        assert_eq!(G::solinas_reduce(&[u64::MAX; 4]), expected);
    }

    #[test]
    fn from_i64_handles_min_without_overflow() {
        let x = F::from_i64(i64::MIN);
        let y = F::from_u64(i64::MIN.unsigned_abs());
        assert_eq!(x + y, F::zero());
    }

    #[test]
    fn prime128_offset_a7f7_constants() {
        // p = 2^128 − 2^32 + 22537, so C = 2^32 − 22537 = 0xFFFFA7F7.
        assert_eq!(
            <Prime128OffsetA7F7 as PseudoMersenneField>::MODULUS_OFFSET,
            0xFFFFA7F7,
        );
        assert_eq!(Prime128OffsetA7F7::C, 0xFFFFA7F7);
        assert_eq!(Prime128OffsetA7F7::C_LO, 0xFFFFA7F7);
        // Round-trip through the field arithmetic: p ≡ 0 (mod p), so
        // Fp(2^128 − C) + Fp(C) = 0.
        let neg_c = -Prime128OffsetA7F7::from_canonical_u128_reduced(0xFFFFA7F7);
        assert_eq!(
            neg_c + Prime128OffsetA7F7::from_canonical_u128_reduced(0xFFFFA7F7),
            Prime128OffsetA7F7::zero()
        );
    }

    #[test]
    fn prime128_offset_a7f7_mul_wide_matches_full_mul() {
        type G = Prime128OffsetA7F7;
        let mut rng = StdRng::seed_from_u64(0xa7f7_a7f7_a7f7_a7f7);
        for _ in 0..1000 {
            let a: G = FieldSampling::sample(&mut rng);
            let b: G = FieldSampling::sample(&mut rng);
            let expected = a * b;
            let reduced = G::solinas_reduce(&a.mul_wide(b));
            assert_eq!(reduced, expected);
        }
    }
}
