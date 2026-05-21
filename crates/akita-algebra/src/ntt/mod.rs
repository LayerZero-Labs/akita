//! NTT-friendly small-prime arithmetic and CRT helpers.

use std::sync::OnceLock;

pub mod butterfly;
pub mod crt;
pub mod prime;
pub mod tables;

/// Whether the active SIMD NTT backend is enabled. Cached on first call.
///
/// Set `AKITA_SCALAR_NTT=1` to force the scalar fallback for A/B performance
/// comparison. The function is backend-agnostic (it only reads an env var),
/// so it lives at the module level rather than being duplicated in each
/// backend submodule.
pub fn use_simd_ntt() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("AKITA_SCALAR_NTT").map_or(true, |v| v != "1"))
}

// SIMD backends. Each backend's `pub mod` declaration and unified `simd`
// alias share a cfg gate. Precedence mirrors `fields::packed`: AVX-512 wins
// on x86 if all required features are present, then AVX2, then NEON on
// aarch64. Dispatch sites refer to `super::simd::*` regardless of arch.
#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "aarch64")]
pub use neon as simd;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512bw",
))]
pub mod avx512;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512bw",
))]
pub use avx512 as simd;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512bw"
    ))
))]
pub mod avx2;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512bw"
    ))
))]
pub use avx2 as simd;

#[cfg(all(test, not(feature = "zk")))]
mod simd_tests;

pub use butterfly::NttTwiddles;
pub use crt::{GarnerData, LimbQ, RADIX_BITS};
pub use prime::{MontCoeff, NttPrime, PrimeWidth};
