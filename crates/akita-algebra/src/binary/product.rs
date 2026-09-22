//! Carryless products, dispatched once per process.

use std::sync::OnceLock;

type Kernel = fn([u64; 3], [u64; 3]) -> [u64; 6];

pub(super) fn multiply(a: [u64; 3], b: [u64; 3]) -> [u64; 6] {
    static KERNEL: OnceLock<Kernel> = OnceLock::new();
    KERNEL.get_or_init(detect)(a, b)
}

fn detect() -> Kernel {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return |a, b| {
            // SAFETY: this function is selected only after detecting PMULL.
            unsafe { arm_product(a, b) }
        };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("pclmulqdq") {
        return |a, b| {
            // SAFETY: this function is selected only after detecting PCLMUL.
            unsafe { x86_product(a, b) }
        };
    }
    portable_product
}

// Three diagonal and three cross products replace nine word products.
#[inline]
fn karatsuba(a: [u64; 3], b: [u64; 3], mul: impl Fn(u64, u64) -> u128) -> [u64; 6] {
    let d0 = mul(a[0], b[0]);
    let d1 = mul(a[1], b[1]);
    let d2 = mul(a[2], b[2]);
    let c01 = mul(a[0] ^ a[1], b[0] ^ b[1]) ^ d0 ^ d1;
    let c02 = mul(a[0] ^ a[2], b[0] ^ b[2]) ^ d0 ^ d2;
    let c12 = mul(a[1] ^ a[2], b[1] ^ b[2]) ^ d1 ^ d2;
    [
        d0 as u64,
        (d0 >> 64) as u64 ^ c01 as u64,
        d1 as u64 ^ (c01 >> 64) as u64 ^ c02 as u64,
        (d1 >> 64) as u64 ^ (c02 >> 64) as u64 ^ c12 as u64,
        d2 as u64 ^ (c12 >> 64) as u64,
        (d2 >> 64) as u64,
    ]
}

pub(super) fn portable_product(a: [u64; 3], b: [u64; 3]) -> [u64; 6] {
    karatsuba(a, b, |a, b| {
        let mut product = 0u128;
        for i in 0..64 {
            product ^= (u128::from(a) << i) & 0u128.wrapping_sub(u128::from((b >> i) & 1));
        }
        product
    })
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn arm_product(a: [u64; 3], b: [u64; 3]) -> [u64; 6] {
    karatsuba(a, b, |a, b| std::arch::aarch64::vmull_p64(a, b))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq")]
unsafe fn x86_product(a: [u64; 3], b: [u64; 3]) -> [u64; 6] {
    use std::arch::x86_64::{_mm_clmulepi64_si128, _mm_cvtsi64_si128};
    karatsuba(a, b, |a, b| {
        // SAFETY: x86_64 guarantees SSE2 and the caller established PCLMUL.
        // The transmute copies the complete 128-bit polynomial product.
        unsafe {
            std::mem::transmute(_mm_clmulepi64_si128::<0>(
                _mm_cvtsi64_si128(a as i64),
                _mm_cvtsi64_si128(b as i64),
            ))
        }
    })
}
