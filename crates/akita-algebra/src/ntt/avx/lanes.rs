//! Width-generic `i32` lane arithmetic for the x86 transforms.
//!
//! The transform passes are written once over [`Lanes`] and instantiated for
//! `__m256i` inside `#[target_feature(enable = "avx2")]` entry points and for
//! `__m512i` inside AVX-512 entry points. Every method is `#[inline(always)]`,
//! so the intrinsics are inlined into the target-feature caller.
//!
//! Stored transform values stay in `[0, 2p)` between passes. The helpers on
//! [`Modulus`] move values between that range and the signed ranges the
//! Montgomery product returns, each with one add or subtract and one unsigned
//! minimum. Every x86 NTT prime is below `2^30`, so `4p < 2^32` and no
//! intermediate below wraps as an unsigned value.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// A vector of `i32` lanes with the operations the transforms need.
pub(super) trait Lanes: Copy {
    /// Number of `i32` lanes.
    const LANES: usize;

    /// Load `LANES` values from `ptr`.
    unsafe fn load(ptr: *const i32) -> Self;
    /// Store `LANES` values to `ptr`.
    unsafe fn store(self, ptr: *mut i32);
    /// Load `LANES` signed bytes from `ptr`, sign-extended to `i32`.
    unsafe fn load_i8(ptr: *const i8) -> Self;
    /// Broadcast one value.
    unsafe fn splat(value: i32) -> Self;
    /// Lane-wise wrapping addition.
    unsafe fn add(self, rhs: Self) -> Self;
    /// Lane-wise wrapping subtraction.
    unsafe fn sub(self, rhs: Self) -> Self;
    /// Lane-wise unsigned minimum.
    unsafe fn min_u32(self, rhs: Self) -> Self;
    /// Signed 64-bit products of the even lanes.
    unsafe fn mul_even_i64(self, rhs: Self) -> Self;
    /// Unsigned 64-bit products of the even lanes.
    unsafe fn mul_even_u64(self, rhs: Self) -> Self;
    /// Copy every odd lane into the even lane below it.
    unsafe fn odd_to_even(self) -> Self;
    /// Gather the high halves of the 64-bit lanes of `even` and `odd` into
    /// the even and odd `i32` lanes of one vector.
    unsafe fn merge_high_halves(even: Self, odd: Self) -> Self;
}

impl Lanes for __m256i {
    const LANES: usize = 8;

    #[inline(always)]
    unsafe fn load(ptr: *const i32) -> Self {
        // SAFETY: the caller guarantees `ptr` is valid for eight reads.
        unsafe { _mm256_loadu_si256(ptr.cast()) }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut i32) {
        // SAFETY: the caller guarantees `ptr` is valid for eight writes.
        unsafe { _mm256_storeu_si256(ptr.cast(), self) }
    }

    #[inline(always)]
    unsafe fn load_i8(ptr: *const i8) -> Self {
        // SAFETY: the caller guarantees `ptr` is valid for eight reads.
        unsafe { _mm256_cvtepi8_epi32(_mm_loadl_epi64(ptr.cast())) }
    }

    #[inline(always)]
    unsafe fn splat(value: i32) -> Self {
        _mm256_set1_epi32(value)
    }

    #[inline(always)]
    unsafe fn add(self, rhs: Self) -> Self {
        _mm256_add_epi32(self, rhs)
    }

    #[inline(always)]
    unsafe fn sub(self, rhs: Self) -> Self {
        _mm256_sub_epi32(self, rhs)
    }

    #[inline(always)]
    unsafe fn min_u32(self, rhs: Self) -> Self {
        _mm256_min_epu32(self, rhs)
    }

    #[inline(always)]
    unsafe fn mul_even_i64(self, rhs: Self) -> Self {
        _mm256_mul_epi32(self, rhs)
    }

    #[inline(always)]
    unsafe fn mul_even_u64(self, rhs: Self) -> Self {
        _mm256_mul_epu32(self, rhs)
    }

    #[inline(always)]
    unsafe fn odd_to_even(self) -> Self {
        _mm256_shuffle_epi32::<0xF5>(self)
    }

    #[inline(always)]
    unsafe fn merge_high_halves(even: Self, odd: Self) -> Self {
        _mm256_blend_epi32::<0xAA>(_mm256_shuffle_epi32::<0xF5>(even), odd)
    }
}

impl Lanes for __m512i {
    const LANES: usize = 16;

    #[inline(always)]
    unsafe fn load(ptr: *const i32) -> Self {
        // SAFETY: the caller guarantees `ptr` is valid for sixteen reads.
        unsafe { _mm512_loadu_si512(ptr.cast()) }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut i32) {
        // SAFETY: the caller guarantees `ptr` is valid for sixteen writes.
        unsafe { _mm512_storeu_si512(ptr.cast(), self) }
    }

    #[inline(always)]
    unsafe fn load_i8(ptr: *const i8) -> Self {
        // SAFETY: the caller guarantees `ptr` is valid for sixteen reads.
        unsafe { _mm512_cvtepi8_epi32(_mm_loadu_si128(ptr.cast())) }
    }

    #[inline(always)]
    unsafe fn splat(value: i32) -> Self {
        _mm512_set1_epi32(value)
    }

    #[inline(always)]
    unsafe fn add(self, rhs: Self) -> Self {
        _mm512_add_epi32(self, rhs)
    }

    #[inline(always)]
    unsafe fn sub(self, rhs: Self) -> Self {
        _mm512_sub_epi32(self, rhs)
    }

    #[inline(always)]
    unsafe fn min_u32(self, rhs: Self) -> Self {
        _mm512_min_epu32(self, rhs)
    }

    #[inline(always)]
    unsafe fn mul_even_i64(self, rhs: Self) -> Self {
        _mm512_mul_epi32(self, rhs)
    }

    #[inline(always)]
    unsafe fn mul_even_u64(self, rhs: Self) -> Self {
        _mm512_mul_epu32(self, rhs)
    }

    #[inline(always)]
    unsafe fn odd_to_even(self) -> Self {
        _mm512_shuffle_epi32::<0xF5>(self)
    }

    #[inline(always)]
    unsafe fn merge_high_halves(even: Self, odd: Self) -> Self {
        // Even lanes take the shuffled `even` high halves; odd lanes keep `odd`.
        _mm512_mask_shuffle_epi32::<0xF5>(odd, 0x5555, even)
    }
}

/// The prime and its double, broadcast.
#[derive(Clone, Copy)]
pub(super) struct Modulus<V> {
    pub(super) p: V,
    two_p: V,
}

impl<V: Lanes> Modulus<V> {
    #[inline(always)]
    pub(super) unsafe fn splat(p: i32) -> Self {
        // SAFETY: the caller's target features cover `V`.
        unsafe {
            Self {
                p: V::splat(p),
                two_p: V::splat(2 * p),
            }
        }
    }

    /// `[0, 4p) -> [0, 2p)`.
    #[inline(always)]
    pub(super) unsafe fn reduce_4p(self, x: V) -> V {
        // SAFETY: the caller's target features cover `V`.
        unsafe { x.min_u32(x.sub(self.two_p)) }
    }

    /// `(-2p, 2p) -> [0, 2p)`.
    #[inline(always)]
    pub(super) unsafe fn fold(self, x: V) -> V {
        // SAFETY: the caller's target features cover `V`.
        unsafe { x.min_u32(x.add(self.two_p)) }
    }

    /// `[0, 2p) -> [0, p)`.
    #[inline(always)]
    pub(super) unsafe fn canonical(self, x: V) -> V {
        // SAFETY: the caller's target features cover `V`.
        unsafe { x.min_u32(x.sub(self.p)) }
    }

    /// `(-p, p) -> [0, p)`.
    #[inline(always)]
    pub(super) unsafe fn canonical_signed(self, x: V) -> V {
        // SAFETY: the caller's target features cover `V`.
        unsafe { x.min_u32(x.add(self.p)) }
    }
}

/// A constant multiplier `w` with its Montgomery quotient `w * p^{-1}`, and
/// both with their odd lanes copied down for the odd-lane products.
#[derive(Clone, Copy)]
pub(super) struct Multiplier<V> {
    value: V,
    value_odd: V,
    quotient: V,
    quotient_odd: V,
}

impl<V: Lanes> Multiplier<V> {
    /// Load `LANES` multipliers from parallel value and quotient tables.
    #[inline(always)]
    pub(super) unsafe fn load(values: *const i32, quotients: *const i32) -> Self {
        // SAFETY: the caller guarantees both pointers are valid for `LANES`
        // reads and that its target features cover `V`.
        unsafe { Self::from_vectors(V::load(values), V::load(quotients)) }
    }

    /// Broadcast one multiplier.
    #[inline(always)]
    pub(super) unsafe fn splat(value: i32, quotient: i32) -> Self {
        // SAFETY: the caller's target features cover `V`.
        unsafe { Self::from_vectors(V::splat(value), V::splat(quotient)) }
    }

    #[inline(always)]
    pub(super) unsafe fn from_vectors(value: V, quotient: V) -> Self {
        // SAFETY: the caller's target features cover `V`.
        unsafe {
            Self {
                value,
                value_odd: value.odd_to_even(),
                quotient,
                quotient_odd: quotient.odd_to_even(),
            }
        }
    }

    /// Montgomery product `x * w * 2^-32 mod p`, bit-identical to
    /// `NttPrime::mont_mul_raw`. The result lies in `(-p, p)` for every `i32`
    /// `x` because `|w| < p`.
    #[inline(always)]
    pub(super) unsafe fn mul(self, x: V, m: Modulus<V>) -> V {
        // SAFETY: the caller's target features cover `V`.
        unsafe {
            let x_odd = x.odd_to_even();
            let even = x.mul_even_i64(self.value);
            let odd = x_odd.mul_even_i64(self.value_odd);
            let even_t = x.mul_even_u64(self.quotient);
            let odd_t = x_odd.mul_even_u64(self.quotient_odd);
            // `x*w - t*p` is 0 mod 2^32, so a 32-bit subtract leaves the exact
            // high halves in the odd `i32` lanes.
            let even = even.sub(even_t.mul_even_i64(m.p));
            let odd = odd.sub(odd_t.mul_even_i64(m.p));
            V::merge_high_halves(even, odd)
        }
    }
}

/// Montgomery product of two variable vectors, bit-identical to
/// `NttPrime::mont_mul_raw`, with `p` and `pinv = p^{-1} mod 2^32` broadcast.
#[inline(always)]
pub(super) unsafe fn mont_mul<V: Lanes>(a: V, b: V, p: V, pinv: V) -> V {
    // SAFETY: the caller's target features cover `V`.
    unsafe {
        let even = a.mul_even_i64(b);
        let odd = a.odd_to_even().mul_even_i64(b.odd_to_even());
        let even = even.sub(even.mul_even_u64(pinv).mul_even_i64(p));
        let odd = odd.sub(odd.mul_even_u64(pinv).mul_even_i64(p));
        V::merge_high_halves(even, odd)
    }
}
