//! Carryless field arithmetic with one process-wide backend selection.

use std::sync::OnceLock;

use super::BinaryField162 as F;

type MulKernel = fn(F, F) -> F;
type SquareKernel = fn(F) -> F;
type DotKernel = fn(&[F], &[F]) -> F;
type InverseKernel = fn(F) -> F;

/// The selected backend. Keeping this handle lets compound operations hoist
/// runtime feature detection and the `OnceLock` load out of their inner loops.
pub(super) struct Kernels {
    multiply: MulKernel,
    square: SquareKernel,
    dot_product: DotKernel,
    inverse: InverseKernel,
}

impl Kernels {
    #[inline]
    pub(super) fn multiply(&self, a: F, b: F) -> F {
        (self.multiply)(a, b)
    }

    #[inline]
    pub(super) fn square(&self, a: F) -> F {
        (self.square)(a)
    }

    #[inline]
    pub(super) fn dot_product(&self, a: &[F], b: &[F]) -> F {
        (self.dot_product)(a, b)
    }

    #[inline]
    pub(super) fn inverse(&self, a: F) -> F {
        (self.inverse)(a)
    }
}

pub(super) fn kernels() -> &'static Kernels {
    static KERNELS: OnceLock<Kernels> = OnceLock::new();
    KERNELS.get_or_init(detect)
}

fn detect() -> Kernels {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes")
        && std::arch::is_aarch64_feature_detected!("pmull")
    {
        return Kernels {
            multiply: |a, b| {
                // SAFETY: this closure is installed only after detecting PMULL.
                unsafe { arm_multiply(a, b) }
            },
            square: |a| {
                // SAFETY: this closure is installed only after detecting PMULL.
                unsafe { arm_square(a) }
            },
            dot_product: |a, b| {
                // SAFETY: this closure is installed only after detecting PMULL.
                unsafe { arm_dot_product(a, b) }
            },
            inverse: |a| {
                // SAFETY: this closure is installed only after detecting PMULL.
                unsafe { arm_inverse(a) }
            },
        };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("pclmulqdq") {
        let dot_product: DotKernel = if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("vpclmulqdq")
        {
            |a: &[F], b: &[F]| {
                // SAFETY: this closure is installed only after detecting both features.
                unsafe { x86_dot_product_vec2(a, b) }
            }
        } else {
            |a: &[F], b: &[F]| {
                // SAFETY: this closure is installed only after detecting PCLMUL.
                unsafe { x86_dot_product(a, b) }
            }
        };
        return Kernels {
            multiply: |a, b| {
                // SAFETY: this closure is installed only after detecting PCLMUL.
                unsafe { x86_multiply(a, b) }
            },
            square: |a| {
                // SAFETY: this closure is installed only after detecting PCLMUL.
                unsafe { x86_square(a) }
            },
            dot_product,
            inverse: |a| {
                // SAFETY: this closure is installed only after detecting PCLMUL.
                unsafe { x86_inverse(a) }
            },
        };
    }
    Kernels {
        multiply: portable_multiply,
        square: portable_square,
        dot_product: portable_dot_product,
        inverse: portable_inverse,
    }
}

/// Reduce a polynomial of degree at most 322 modulo `X^162 + X^81 + 1`.
#[inline(always)]
pub(super) fn reduce(p: [u64; 6]) -> F {
    // Write P=L+X^162 H, then use X^162=X^81+1 twice.
    // H has degree <=160, so the second high part has degree <=79.
    let h = [
        (p[2] >> 34) | (p[3] << 30),
        (p[3] >> 34) | (p[4] << 30),
        (p[4] >> 34) | (p[5] << 30),
    ];
    let j = [
        h[0] ^ (h[1] >> 17) ^ (h[2] << 47),
        h[1] ^ (h[2] >> 17),
        h[2],
    ];
    F([
        p[0] ^ j[0],
        p[1] ^ j[1] ^ (j[0] << 17),
        (p[2] ^ j[2] ^ (j[0] >> 47) ^ (j[1] << 17)) & F::TOP_MASK,
    ])
}

/// Recombine the six three-term Karatsuba products into polynomial words.
#[inline(always)]
pub(super) fn karatsuba_product(
    d0: u128,
    d1: u128,
    d2: u128,
    m01: u128,
    m02: u128,
    m12: u128,
) -> [u64; 6] {
    let c01 = m01 ^ d0 ^ d1;
    let c02 = m02 ^ d0 ^ d2;
    let c12 = m12 ^ d1 ^ d2;
    [
        d0 as u64,
        (d0 >> 64) as u64 ^ c01 as u64,
        d1 as u64 ^ (c01 >> 64) as u64 ^ c02 as u64,
        (d1 >> 64) as u64 ^ (c02 >> 64) as u64 ^ c12 as u64,
        d2 as u64 ^ (c12 >> 64) as u64,
        (d2 >> 64) as u64,
    ]
}

/// Recombine and reduce without crossing an unreduced-product ABI boundary.
#[inline(always)]
fn reduce_karatsuba(d0: u128, d1: u128, d2: u128, m01: u128, m02: u128, m12: u128) -> F {
    reduce(karatsuba_product(d0, d1, d2, m01, m02, m12))
}

#[inline]
pub(super) fn portable_clmul(a: u64, b: u64) -> u128 {
    let mut product = 0u128;
    for i in 0..64 {
        product ^= (u128::from(a) << i) & 0u128.wrapping_sub(u128::from((b >> i) & 1));
    }
    product
}

#[inline]
pub(super) fn portable_product(a: F, b: F) -> [u64; 6] {
    let [a0, a1, a2] = a.0;
    let [b0, b1, b2] = b.0;
    let d0 = portable_clmul(a0, b0);
    let d1 = portable_clmul(a1, b1);
    let d2 = portable_clmul(a2, b2);
    karatsuba_product(
        d0,
        d1,
        d2,
        portable_clmul(a0 ^ a1, b0 ^ b1),
        portable_clmul(a0 ^ a2, b0 ^ b2),
        portable_clmul(a1 ^ a2, b1 ^ b2),
    )
}

#[inline]
pub(super) fn portable_multiply(a: F, b: F) -> F {
    reduce(portable_product(a, b))
}

#[inline]
pub(super) fn portable_square(a: F) -> F {
    let mut product = [0; 6];
    for (i, word) in a.0.into_iter().enumerate() {
        product[2 * i] = spread(word as u32);
        product[2 * i + 1] = spread((word >> 32) as u32);
    }
    reduce(product)
}

#[inline]
fn spread(value: u32) -> u64 {
    let mut x = u64::from(value);
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    (x | (x << 1)) & 0x5555_5555_5555_5555
}

pub(super) fn portable_dot_product(a: &[F], b: &[F]) -> F {
    let mut product = [0; 6];
    for (&a, &b) in a.iter().zip(b) {
        for (sum, term) in product.iter_mut().zip(portable_product(a, b)) {
            *sum ^= term;
        }
    }
    reduce(product)
}

#[inline(always)]
fn inverse_with(a: F, multiply: impl Fn(F, F) -> F, square: impl Fn(F) -> F) -> F {
    #[inline(always)]
    fn square_n(mut value: F, count: usize, square: &impl Fn(F) -> F) -> F {
        for _ in 0..count {
            value = square(value);
        }
        value
    }

    let mut power = a;
    let mut power32 = a;
    for width in [1, 2, 4, 8, 16, 32, 64] {
        let previous = power;
        power = multiply(square_n(previous, width, &square), previous);
        if width == 16 {
            power32 = power;
        }
    }
    power = multiply(square_n(power, 32, &square), power32);
    power = multiply(square(power), a);
    square(power)
}

#[inline]
fn portable_inverse(a: F) -> F {
    inverse_with(a, portable_multiply, portable_square)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(super) unsafe fn arm_multiply(a: F, b: F) -> F {
    let [a0, a1, a2] = a.0;
    let [b0, b1, b2] = b.0;
    reduce_karatsuba(
        std::arch::aarch64::vmull_p64(a0, b0),
        std::arch::aarch64::vmull_p64(a1, b1),
        std::arch::aarch64::vmull_p64(a2, b2),
        std::arch::aarch64::vmull_p64(a0 ^ a1, b0 ^ b1),
        std::arch::aarch64::vmull_p64(a0 ^ a2, b0 ^ b2),
        std::arch::aarch64::vmull_p64(a1 ^ a2, b1 ^ b2),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(super) unsafe fn arm_square(a: F) -> F {
    let [a0, a1, a2] = a.0;
    let p0 = std::arch::aarch64::vmull_p64(a0, a0);
    let p1 = std::arch::aarch64::vmull_p64(a1, a1);
    let p2 = std::arch::aarch64::vmull_p64(a2, a2);
    reduce([
        p0 as u64,
        (p0 >> 64) as u64,
        p1 as u64,
        (p1 >> 64) as u64,
        p2 as u64,
        (p2 >> 64) as u64,
    ])
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn arm_inverse(a: F) -> F {
    inverse_with(
        a,
        |x, y| {
            // SAFETY: this function carries the required target feature.
            unsafe { arm_multiply(x, y) }
        },
        |x| {
            // SAFETY: this function carries the required target feature.
            unsafe { arm_square(x) }
        },
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(super) unsafe fn arm_dot_product(a: &[F], b: &[F]) -> F {
    use std::arch::aarch64::{uint64x2_t, vdupq_n_u64, veorq_u64, vmull_p64};

    #[inline]
    #[target_feature(enable = "aes")]
    unsafe fn vector_product(a: u64, b: u64) -> uint64x2_t {
        // Every bit pattern is valid in both representations. Keeping the
        // product in this form avoids two lane extracts per loop iteration.
        unsafe { std::mem::transmute::<u128, uint64x2_t>(vmull_p64(a, b)) }
    }

    let zero = vdupq_n_u64(0);
    let (mut d0, mut d1, mut d2, mut m01, mut m02, mut m12) = (zero, zero, zero, zero, zero, zero);
    for (&a, &b) in a.iter().zip(b) {
        let [a0, a1, a2] = a.0;
        let [b0, b1, b2] = b.0;
        // SAFETY: the enclosing function carries the same target feature.
        unsafe {
            d0 = veorq_u64(d0, vector_product(a0, b0));
            d1 = veorq_u64(d1, vector_product(a1, b1));
            d2 = veorq_u64(d2, vector_product(a2, b2));
            m01 = veorq_u64(m01, vector_product(a0 ^ a1, b0 ^ b1));
            m02 = veorq_u64(m02, vector_product(a0 ^ a2, b0 ^ b2));
            m12 = veorq_u64(m12, vector_product(a1 ^ a2, b1 ^ b2));
        }
    }
    // SAFETY: the accumulated vectors contain exactly six 128-bit products.
    unsafe {
        reduce_karatsuba(
            std::mem::transmute::<uint64x2_t, u128>(d0),
            std::mem::transmute::<uint64x2_t, u128>(d1),
            std::mem::transmute::<uint64x2_t, u128>(d2),
            std::mem::transmute::<uint64x2_t, u128>(m01),
            std::mem::transmute::<uint64x2_t, u128>(m02),
            std::mem::transmute::<uint64x2_t, u128>(m12),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "pclmulqdq")]
pub(super) unsafe fn x86_clmul(a: u64, b: u64) -> u128 {
    use std::arch::x86_64::{__m128i, _mm_clmulepi64_si128, _mm_cvtsi64_si128};
    // SAFETY: the function's target feature guarantees PCLMUL support, and the
    // transmute preserves all 128 product bits.
    unsafe {
        std::mem::transmute::<__m128i, u128>(_mm_clmulepi64_si128::<0>(
            _mm_cvtsi64_si128(a as i64),
            _mm_cvtsi64_si128(b as i64),
        ))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq")]
pub(super) unsafe fn x86_multiply(a: F, b: F) -> F {
    let [a0, a1, a2] = a.0;
    let [b0, b1, b2] = b.0;
    // SAFETY: the function's target feature guarantees PCLMUL support.
    unsafe {
        reduce_karatsuba(
            x86_clmul(a0, b0),
            x86_clmul(a1, b1),
            x86_clmul(a2, b2),
            x86_clmul(a0 ^ a1, b0 ^ b1),
            x86_clmul(a0 ^ a2, b0 ^ b2),
            x86_clmul(a1 ^ a2, b1 ^ b2),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq")]
pub(super) unsafe fn x86_square(a: F) -> F {
    let [a0, a1, a2] = a.0;
    // SAFETY: the function's target feature guarantees PCLMUL support.
    unsafe {
        let p0 = x86_clmul(a0, a0);
        let p1 = x86_clmul(a1, a1);
        let p2 = x86_clmul(a2, a2);
        reduce([
            p0 as u64,
            (p0 >> 64) as u64,
            p1 as u64,
            (p1 >> 64) as u64,
            p2 as u64,
            (p2 >> 64) as u64,
        ])
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq")]
unsafe fn x86_inverse(a: F) -> F {
    inverse_with(
        a,
        |x, y| {
            // SAFETY: this function carries the required target feature.
            unsafe { x86_multiply(x, y) }
        },
        |x| {
            // SAFETY: this function carries the required target feature.
            unsafe { x86_square(x) }
        },
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq")]
pub(super) unsafe fn x86_dot_product(a: &[F], b: &[F]) -> F {
    let (mut d0, mut d1, mut d2, mut m01, mut m02, mut m12) = (0, 0, 0, 0, 0, 0);
    // SAFETY: the function's target feature guarantees PCLMUL support.
    unsafe {
        for (&a, &b) in a.iter().zip(b) {
            let [a0, a1, a2] = a.0;
            let [b0, b1, b2] = b.0;
            d0 ^= x86_clmul(a0, b0);
            d1 ^= x86_clmul(a1, b1);
            d2 ^= x86_clmul(a2, b2);
            m01 ^= x86_clmul(a0 ^ a1, b0 ^ b1);
            m02 ^= x86_clmul(a0 ^ a2, b0 ^ b2);
            m12 ^= x86_clmul(a1 ^ a2, b1 ^ b2);
        }
    }
    reduce_karatsuba(d0, d1, d2, m01, m02, m12)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,pclmulqdq,vpclmulqdq")]
pub(super) unsafe fn x86_dot_product_vec2(a: &[F], b: &[F]) -> F {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_castsi256_si128, _mm256_clmulepi64_epi128,
        _mm256_extracti128_si256, _mm256_set_epi64x, _mm256_setzero_si256, _mm256_xor_si256,
        _mm_xor_si128,
    };

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn pack(x0: u64, x1: u64) -> __m256i {
        _mm256_set_epi64x(0, x1 as i64, 0, x0 as i64)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn fold_lanes(value: __m256i) -> u128 {
        // SAFETY: AVX2 is established by the caller; XOR combines the two
        // independent 128-bit lanes before conversion to a polynomial product.
        unsafe {
            std::mem::transmute::<__m128i, u128>(_mm_xor_si128(
                _mm256_castsi256_si128(value),
                _mm256_extracti128_si256::<1>(value),
            ))
        }
    }

    // SAFETY: the function's target features guarantee every intrinsic below.
    unsafe {
        let zero = _mm256_setzero_si256();
        let (mut d0, mut d1, mut d2, mut m01, mut m02, mut m12) =
            (zero, zero, zero, zero, zero, zero);
        let mut i = 0;
        while i + 1 < a.len() {
            let [a0, a1, a2] = a[i].0;
            let [c0, c1, c2] = a[i + 1].0;
            let [b0, b1, b2] = b[i].0;
            let [e0, e1, e2] = b[i + 1].0;
            let mul = |x, y| _mm256_clmulepi64_epi128::<0>(x, y);
            d0 = _mm256_xor_si256(d0, mul(pack(a0, c0), pack(b0, e0)));
            d1 = _mm256_xor_si256(d1, mul(pack(a1, c1), pack(b1, e1)));
            d2 = _mm256_xor_si256(d2, mul(pack(a2, c2), pack(b2, e2)));
            m01 = _mm256_xor_si256(m01, mul(pack(a0 ^ a1, c0 ^ c1), pack(b0 ^ b1, e0 ^ e1)));
            m02 = _mm256_xor_si256(m02, mul(pack(a0 ^ a2, c0 ^ c2), pack(b0 ^ b2, e0 ^ e2)));
            m12 = _mm256_xor_si256(m12, mul(pack(a1 ^ a2, c1 ^ c2), pack(b1 ^ b2, e1 ^ e2)));
            i += 2;
        }

        let (mut d0, mut d1, mut d2, mut m01, mut m02, mut m12) = (
            fold_lanes(d0),
            fold_lanes(d1),
            fold_lanes(d2),
            fold_lanes(m01),
            fold_lanes(m02),
            fold_lanes(m12),
        );
        if i < a.len() {
            let [a0, a1, a2] = a[i].0;
            let [b0, b1, b2] = b[i].0;
            d0 ^= x86_clmul(a0, b0);
            d1 ^= x86_clmul(a1, b1);
            d2 ^= x86_clmul(a2, b2);
            m01 ^= x86_clmul(a0 ^ a1, b0 ^ b1);
            m02 ^= x86_clmul(a0 ^ a2, b0 ^ b2);
            m12 ^= x86_clmul(a1 ^ a2, b1 ^ b2);
        }
        reduce_karatsuba(d0, d1, d2, m01, m02, m12)
    }
}
