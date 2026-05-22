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

// SIMD NTT backend. Only NEON is currently wired into the dispatch sites
// (`butterfly.rs`, `crt_ntt_repr.rs`, `kernels/linear.rs`). A draft AVX2 /
// AVX-512 NTT port was reverted because it regressed `commit` and `setup`
// at the e2e level — LLVM's auto-vectorization of the simple scalar
// butterfly / pointwise-mul-acc loops turned out to be competitive with
// hand-written intrinsics for the typical small `D ≤ 64` NTT sizes. The
// `pub use ... as simd` alias is kept arch-agnostic so future SIMD
// backends can plug in without touching every dispatch site.
#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "aarch64")]
pub use neon as simd;

#[cfg(all(test, not(feature = "zk")))]
mod simd_tests;

pub use butterfly::NttTwiddles;
pub use crt::{GarnerData, LimbQ, RADIX_BITS};
pub use prime::{MontCoeff, NttPrime, PrimeWidth};
