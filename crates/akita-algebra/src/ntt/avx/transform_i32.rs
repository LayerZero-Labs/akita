//! x86 negacyclic and cyclic NTTs for one `i32` CRT limb.
//!
//! The forward transform is Gentleman–Sande DIF and the inverse is
//! Cooley–Tukey DIT, both over bit-reversed evaluations, exactly as in the
//! scalar `butterfly` module. The passes are:
//!
//! - Forward: one radix-4 pass per two stages from `D/2` down, the first of
//!   which also applies the negacyclic twist (or converts signed digits), then
//!   a 256-bit tail that runs the last four stages (`log2 D` even) or five
//!   stages (`log2 D` odd) in registers.
//! - Inverse: the mirror-image 256-bit head, then radix-4 passes, the last of
//!   which also applies the untwist and `D^{-1}` scaling.
//!
//! Every radix-4 pass therefore has a butterfly half-length of at least 16, so
//! the passes are generic over [`Lanes`] and run at 256 or 512 bits. Stored
//! values stay in `[0, 2p)` between passes; see [`super::lanes`]. Constant
//! multiplies use the precomputed quotients in [`super::twiddles`], and the
//! stage-1 and stage-2 multiplies by `fwd_twiddles[0..2]` and
//! `inv_twiddles[0..2]`, which are the Montgomery one, are skipped.
//!
//! Range contracts: the negacyclic forward transform accepts any `i32`, the
//! cyclic forward transform and both inverses accept `(-p, p)`. Forward output
//! is canonical `[0, p)`; inverse output lies in `(-p, p)`. Degrees below 64
//! use the scalar transforms.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::lanes::{Lanes, Modulus, Multiplier};
use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::prime::{MontCoeff, NttPrime};

/// A constant table with its Montgomery quotients.
#[derive(Clone, Copy)]
struct Table {
    values: *const i32,
    quotients: *const i32,
}

impl Table {
    fn new<T, const D: usize>(values: &[T; D], quotients: &[i32; D]) -> Self {
        Self {
            values: values.as_ptr().cast(),
            quotients: quotients.as_ptr(),
        }
    }

    #[inline(always)]
    unsafe fn load<V: Lanes>(self, index: usize) -> Multiplier<V> {
        // SAFETY: the caller keeps `index + LANES` within the table.
        unsafe { Multiplier::load(self.values.add(index), self.quotients.add(index)) }
    }
}

/// Stages `len` and `len / 2` of the forward DIF transform on four quarters
/// at offsets `0, len/2, len, 3len/2`, returning values in `[0, 2p)`.
///
/// `SIGNED` inputs lie in `(-p, p)`; otherwise inputs lie in `[0, 2p)`.
#[inline(always)]
unsafe fn forward_radix4<V: Lanes, const SIGNED: bool>(
    q: [V; 4],
    outer0: Multiplier<V>,
    outer1: Multiplier<V>,
    inner: Multiplier<V>,
    m: Modulus<V>,
) -> [V; 4] {
    // SAFETY: the caller's target features cover `V`.
    unsafe {
        let e0 = outer0.mul(q[0].sub(q[2]), m);
        let e1 = outer1.mul(q[1].sub(q[3]), m);
        let (a, b) = if SIGNED {
            (m.fold(q[0].add(q[2])), m.fold(q[1].add(q[3])))
        } else {
            (m.reduce_4p(q[0].add(q[2])), m.reduce_4p(q[1].add(q[3])))
        };
        [
            m.reduce_4p(a.add(b)),
            inner.mul(a.sub(b), m).add(m.p),
            m.fold(e0.add(e1)),
            inner.mul(e0.sub(e1), m).add(m.p),
        ]
    }
}

/// One forward radix-4 pass over stages `len` and `len / 2`. `load(i)` reads
/// the input vector at index `i`, which lets the first pass twist or convert
/// its inputs on the way in.
#[inline(always)]
unsafe fn forward_pass<V: Lanes, const D: usize, const SIGNED: bool>(
    a: *mut i32,
    len: usize,
    fwd: Table,
    m: Modulus<V>,
    load: impl Fn(usize) -> V,
) {
    let half = len / 2;
    let mut j = 0;
    while j < half {
        // SAFETY: `len <= D/2` is a stage length and `half >= LANES`, so every
        // twiddle and data index below is in bounds.
        unsafe {
            let outer0 = fwd.load(len - 1 + j);
            let outer1 = fwd.load(len - 1 + half + j);
            let inner = fwd.load(half - 1 + j);
            let mut start = j;
            while start < D {
                let q = [
                    load(start),
                    load(start + half),
                    load(start + len),
                    load(start + len + half),
                ];
                let out = forward_radix4::<V, SIGNED>(q, outer0, outer1, inner, m);
                out[0].store(a.add(start));
                out[1].store(a.add(start + half));
                out[2].store(a.add(start + len));
                out[3].store(a.add(start + len + half));
                start += 2 * len;
            }
        }
        j += V::LANES;
    }
}

/// Transpose the 4×4 `i32` blocks held in each 128-bit lane of four vectors.
#[inline(always)]
unsafe fn transpose4(y: [__m256i; 4]) -> [__m256i; 4] {
    // SAFETY: AVX2 is enabled by every caller.
    unsafe {
        let t0 = _mm256_unpacklo_epi32(y[0], y[1]);
        let t1 = _mm256_unpacklo_epi32(y[2], y[3]);
        let t2 = _mm256_unpackhi_epi32(y[0], y[1]);
        let t3 = _mm256_unpackhi_epi32(y[2], y[3]);
        [
            _mm256_unpacklo_epi64(t0, t1),
            _mm256_unpackhi_epi64(t0, t1),
            _mm256_unpacklo_epi64(t2, t3),
            _mm256_unpackhi_epi64(t2, t3),
        ]
    }
}

/// Split two vectors of eight into their low and high 128-bit halves:
/// `([x.lo, y.lo], [x.hi, y.hi])`. The map is its own inverse.
#[inline(always)]
unsafe fn swap_halves(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
    // SAFETY: AVX2 is enabled by every caller.
    unsafe {
        (
            _mm256_permute2x128_si256::<0x20>(x, y),
            _mm256_permute2x128_si256::<0x31>(x, y),
        )
    }
}

/// Broadcast four consecutive table entries to both 128-bit lanes.
#[inline(always)]
unsafe fn load_broadcast4(table: Table, index: usize) -> Multiplier<__m256i> {
    // SAFETY: the caller keeps `index + 4` within the table; AVX2 is enabled.
    unsafe {
        Multiplier::from_vectors(
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.values.add(index).cast())),
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.quotients.add(index).cast())),
        )
    }
}

/// Radix-2 DIF butterfly on `[0, 2p)` inputs with `[0, 2p)` outputs.
#[inline(always)]
unsafe fn dif<V: Lanes>(u: V, v: V, w: Multiplier<V>, m: Modulus<V>) -> (V, V) {
    // SAFETY: the caller's target features cover `V`.
    unsafe { (m.reduce_4p(u.add(v)), w.mul(u.sub(v), m).add(m.p)) }
}

/// Radix-2 DIT butterfly on `[0, 2p)` inputs with `[0, 2p)` outputs.
#[inline(always)]
unsafe fn dit<V: Lanes>(u: V, v: V, w: Multiplier<V>, m: Modulus<V>) -> (V, V) {
    // SAFETY: the caller's target features cover `V`.
    unsafe {
        let u = u.sub(m.p);
        let v = w.mul(v, m);
        (m.fold(u.add(v)), m.fold(u.sub(v)))
    }
}

/// Last four (or, with `FIVE`, five) forward stages, 32 values at a time,
/// from `[0, 2p)` to canonical `[0, p)`.
#[inline(always)]
unsafe fn forward_tail<const D: usize, const FIVE: bool>(
    a: *mut i32,
    fwd: Table,
    m: Modulus<__m256i>,
) {
    // SAFETY: `D >= 64`, so the stage-16 twiddles at 15..31 exist, and every
    // block of 32 values is in bounds.
    unsafe {
        let t16 = [fwd.load(15), fwd.load(23)];
        let t8 = fwd.load(7);
        let t4 = load_broadcast4(fwd, 3);
        let t2 = Multiplier::splat(*fwd.values.add(2), *fwd.quotients.add(2));
        let mut base = 0;
        while base < D {
            let p = a.add(base);
            let mut y = [
                __m256i::load(p),
                __m256i::load(p.add(8)),
                __m256i::load(p.add(16)),
                __m256i::load(p.add(24)),
            ];
            if FIVE {
                (y[0], y[2]) = dif(y[0], y[2], t16[0], m);
                (y[1], y[3]) = dif(y[1], y[3], t16[1], m);
            }
            (y[0], y[1]) = dif(y[0], y[1], t8, m);
            (y[2], y[3]) = dif(y[2], y[3], t8, m);
            for k in [0, 2] {
                let (u, v) = swap_halves(y[k], y[k + 1]);
                let (u, v) = dif(u, v, t4, m);
                (y[k], y[k + 1]) = swap_halves(u, v);
            }

            // Each lane position now holds one 4-point group `r0..r3`.
            let [r0, r1, r2, r3] = transpose4(y);
            let s0 = m.canonical(m.reduce_4p(r0.add(r2)));
            let s1 = m.canonical(m.reduce_4p(r1.add(r3)));
            let d0 = m.canonical(m.fold(r0.sub(r2)));
            let d1 = m.canonical_signed(t2.mul(r1.sub(r3), m));
            let out = transpose4([
                m.canonical(s0.add(s1)),
                m.canonical_signed(s0.sub(s1)),
                m.canonical(d0.add(d1)),
                m.canonical_signed(d0.sub(d1)),
            ]);
            for (k, value) in out.into_iter().enumerate() {
                value.store(p.add(8 * k));
            }
            base += 32;
        }
    }
}

/// First four (or, with `FIVE`, five) inverse stages, 32 values at a time,
/// from `(-p, p)` to `[0, 2p)`.
#[inline(always)]
unsafe fn inverse_head<const D: usize, const FIVE: bool>(
    a: *mut i32,
    inv: Table,
    m: Modulus<__m256i>,
) {
    // SAFETY: `D >= 64`, so the stage-16 twiddles at 15..31 exist, and every
    // block of 32 values is in bounds.
    unsafe {
        let t2 = Multiplier::splat(*inv.values.add(2), *inv.quotients.add(2));
        let t4 = load_broadcast4(inv, 3);
        let t8 = inv.load(7);
        let t16 = [inv.load(15), inv.load(23)];
        let mut base = 0;
        while base < D {
            let p = a.add(base);
            let [r0, r1, r2, r3] = transpose4([
                __m256i::load(p),
                __m256i::load(p.add(8)),
                __m256i::load(p.add(16)),
                __m256i::load(p.add(24)),
            ]);
            let a0 = m.fold(r0.add(r1));
            let a1 = m.fold(r0.sub(r1)).sub(m.p);
            let a2 = m.fold(r2.add(r3));
            let v = t2.mul(r2.sub(r3), m);
            let mut y = transpose4([
                m.reduce_4p(a0.add(a2)),
                m.fold(a1.add(v)),
                m.fold(a0.sub(a2)),
                m.fold(a1.sub(v)),
            ]);
            for k in [0, 2] {
                let (u, v) = swap_halves(y[k], y[k + 1]);
                let (u, v) = dit(u, v, t4, m);
                (y[k], y[k + 1]) = swap_halves(u, v);
            }
            (y[0], y[1]) = dit(y[0], y[1], t8, m);
            (y[2], y[3]) = dit(y[2], y[3], t8, m);
            if FIVE {
                (y[0], y[2]) = dit(y[0], y[2], t16[0], m);
                (y[1], y[3]) = dit(y[1], y[3], t16[1], m);
            }
            for (k, value) in y.into_iter().enumerate() {
                value.store(p.add(8 * k));
            }
            base += 32;
        }
    }
}

/// One inverse radix-4 pass over stages `len` and `2 len`, from `[0, 2p)`.
/// Each output in `(-2p, 2p)` is written as `finish(index, output)`.
#[inline(always)]
unsafe fn inverse_pass<V: Lanes, const D: usize>(
    a: *mut i32,
    len: usize,
    inv: Table,
    m: Modulus<V>,
    finish: impl Fn(usize, V) -> V,
) {
    let mut j = 0;
    while j < len {
        // SAFETY: `len <= D/4` is a stage length and `len >= LANES`, so every
        // twiddle and data index below is in bounds.
        unsafe {
            let inner = inv.load(len - 1 + j);
            let outer0 = inv.load(2 * len - 1 + j);
            let outer1 = inv.load(3 * len - 1 + j);
            let mut start = j;
            while start < D {
                let p = a.add(start);
                let q0 = V::load(p).add(m.p);
                let q1 = inner.mul(V::load(p.add(len)), m);
                let q2 = V::load(p.add(2 * len)).sub(m.p);
                let q3 = inner.mul(V::load(p.add(3 * len)), m);
                let sum0 = m.reduce_4p(q0.add(q1)).sub(m.p);
                let diff0 = m.reduce_4p(q0.sub(q1)).sub(m.p);
                let sum1 = outer0.mul(q2.add(q3), m);
                let diff1 = outer1.mul(q2.sub(q3), m);
                finish(start, sum0.add(sum1)).store(p);
                finish(start + len, diff0.add(diff1)).store(p.add(len));
                finish(start + 2 * len, sum0.sub(sum1)).store(p.add(2 * len));
                finish(start + 3 * len, diff0.sub(diff1)).store(p.add(3 * len));
                start += 4 * len;
            }
        }
        j += V::LANES;
    }
}

/// Forward DIF transform whose first pass reads its inputs through
/// `load_first`, which must return values in `(-p, p)`.
#[inline(always)]
unsafe fn forward<V: Lanes, const D: usize>(
    a: *mut i32,
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
    load_first: impl Fn(usize) -> V,
) {
    debug_assert!(D >= 64 && D.is_power_of_two());
    // SAFETY: the caller's target features cover `V` and AVX2, and `a` holds
    // `D` values.
    unsafe {
        let m = Modulus::<V>::splat(prime.p);
        let fwd = Table::new(&tw.fwd_twiddles, &tw.quotients.fwd);
        forward_pass::<V, D, true>(a, D / 2, fwd, m, load_first);
        let five = D.trailing_zeros() % 2 == 1;
        let tail_len = if five { 16 } else { 8 };
        let mut len = D / 8;
        while len > tail_len {
            forward_pass::<V, D, false>(
                a,
                len,
                fwd,
                m,
                #[inline(always)]
                |i| V::load(a.add(i)),
            );
            len /= 4;
        }
        let m = Modulus::<__m256i>::splat(prime.p);
        if five {
            forward_tail::<D, true>(a, fwd, m);
        } else {
            forward_tail::<D, false>(a, fwd, m);
        }
    }
}

/// Inverse DIT transform whose last pass writes `finish(index, output)`.
#[inline(always)]
unsafe fn inverse<V: Lanes, const D: usize>(
    a: *mut i32,
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
    finish: impl Fn(usize, V) -> V,
) {
    debug_assert!(D >= 64 && D.is_power_of_two());
    // SAFETY: the caller's target features cover `V` and AVX2, and `a` holds
    // `D` values.
    unsafe {
        let inv = Table::new(&tw.inv_twiddles, &tw.quotients.inv);
        let five = D.trailing_zeros() % 2 == 1;
        let m256 = Modulus::<__m256i>::splat(prime.p);
        if five {
            inverse_head::<D, true>(a, inv, m256);
        } else {
            inverse_head::<D, false>(a, inv, m256);
        }
        let m = Modulus::<V>::splat(prime.p);
        let mut len = if five { 32 } else { 16 };
        while len < D / 4 {
            inverse_pass::<V, D>(
                a,
                len,
                inv,
                m,
                #[inline(always)]
                |_, x| m.fold(x),
            );
            len *= 4;
        }
        inverse_pass::<V, D>(a, len, inv, m, finish);
    }
}

#[inline(always)]
unsafe fn forward_negacyclic<V: Lanes, const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let psi = Table::new(&tw.psi_pows, &tw.quotients.psi);
    let m = Modulus::<V>::splat(prime.p);
    let a = a.as_mut_ptr().cast::<i32>();
    // SAFETY: inherited from the caller; `psi` has `D` entries.
    unsafe {
        forward::<V, D>(
            a,
            prime,
            tw,
            #[inline(always)]
            |i| psi.load(i).mul(V::load(a.add(i)), m),
        )
    }
}

#[inline(always)]
unsafe fn forward_digits<V: Lanes, const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    digits: &[i8; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let psi_r2 = Table::new(&tw.psi_pows_r2, &tw.quotients.psi_r2);
    let m = Modulus::<V>::splat(prime.p);
    let digits = digits.as_ptr();
    // SAFETY: inherited from the caller; `digits` and `psi_r2` have `D` entries.
    unsafe {
        forward::<V, D>(
            a.as_mut_ptr().cast(),
            prime,
            tw,
            #[inline(always)]
            |i| psi_r2.load(i).mul(V::load_i8(digits.add(i)), m),
        )
    }
}

#[inline(always)]
unsafe fn forward_cyclic<V: Lanes, const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let a = a.as_mut_ptr().cast::<i32>();
    // SAFETY: inherited from the caller.
    unsafe {
        forward::<V, D>(
            a,
            prime,
            tw,
            #[inline(always)]
            |i| V::load(a.add(i)),
        )
    }
}

#[inline(always)]
unsafe fn inverse_negacyclic<V: Lanes, const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let untwist = Table::new(&tw.d_inv_psi_inv, &tw.quotients.d_inv_psi_inv);
    let m = Modulus::<V>::splat(prime.p);
    // SAFETY: inherited from the caller; `untwist` has `D` entries.
    unsafe {
        inverse::<V, D>(
            a.as_mut_ptr().cast(),
            prime,
            tw,
            #[inline(always)]
            |i, x| untwist.load(i).mul(x, m),
        )
    }
}

#[inline(always)]
unsafe fn inverse_cyclic<V: Lanes, const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let m = Modulus::<V>::splat(prime.p);
    // SAFETY: inherited from the caller.
    unsafe {
        let d_inv = Multiplier::<V>::splat(tw.d_inv.raw(), tw.quotients.d_inv);
        inverse::<V, D>(
            a.as_mut_ptr().cast(),
            prime,
            tw,
            #[inline(always)]
            |_, x| d_inv.mul(x, m),
        )
    }
}

macro_rules! x86_transforms {
    ($(
        $(#[$doc:meta])*
        $name:ident, $avx512:ident, $generic:ident, ($($arg:ident: $ty:ty),*);
    )*) => {$(
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// The caller must ensure AVX2 is available, and AVX-512F/DQ/BW when
        /// `use_avx512` is set. `D` must be a power of two of at least 64.
        #[target_feature(enable = "avx2")]
        pub(crate) unsafe fn $name<const D: usize>(
            a: &mut [MontCoeff<i32>; D],
            $($arg: $ty,)*
            prime: NttPrime<i32>,
            tw: &NttTwiddles<i32, D>,
            use_avx512: bool,
        ) {
            if use_avx512 {
                // SAFETY: forwarded from this function's contract.
                unsafe { $avx512(a, $($arg,)* prime, tw) }
            } else {
                // SAFETY: forwarded from this function's contract.
                unsafe { $generic::<__m256i, D>(a, $($arg,)* prime, tw) }
            }
        }

        #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx2")]
        unsafe fn $avx512<const D: usize>(
            a: &mut [MontCoeff<i32>; D],
            $($arg: $ty,)*
            prime: NttPrime<i32>,
            tw: &NttTwiddles<i32, D>,
        ) {
            // SAFETY: forwarded from the caller's contract.
            unsafe { $generic::<__m512i, D>(a, $($arg,)* prime, tw) }
        }
    )*};
}

x86_transforms! {
    /// Forward negacyclic NTT.
    forward_ntt_i32, forward_ntt_i32_avx512, forward_negacyclic, ();
    /// Signed-digit conversion and forward negacyclic NTT. The `psi^i R^2`
    /// twist enters Montgomery form and twists in one product.
    forward_ntt_i8_i32, forward_ntt_i8_i32_avx512, forward_digits, (digits: &[i8; D]);
    /// Inverse negacyclic NTT.
    inverse_ntt_i32, inverse_ntt_i32_avx512, inverse_negacyclic, ();
    /// Forward cyclic NTT.
    forward_ntt_cyclic_i32, forward_ntt_cyclic_i32_avx512, forward_cyclic, ();
    /// Inverse cyclic NTT.
    inverse_ntt_cyclic_i32, inverse_ntt_cyclic_i32_avx512, inverse_cyclic, ();
}
