//! AArch64 NEON SIMD kernels for NTT butterfly, Montgomery conversion,
//! and pointwise operations.
//!
//! Width-specific implementations live in [`i16_kernels`] and
//! [`i32_kernels`]. Set `AKITA_SCALAR_NTT=1` to force scalar kernels for A/B
//! measurements.

use std::sync::OnceLock;

mod i16_kernels;
mod i32_kernels;

#[cfg(feature = "parallel")]
pub use i16_kernels::add_reduce_i16;
pub(crate) use i16_kernels::{
    centered_i16_to_mont_i16, forward_ntt_cyclic_i16, forward_ntt_i16, inverse_ntt_cyclic_i16,
    inverse_ntt_i16, pointwise_mul_acc_i16,
};
#[cfg(feature = "parallel")]
pub use i32_kernels::add_reduce_i32;
pub(crate) use i32_kernels::{
    centered_i16_to_mont_i32, forward_ntt_cyclic_i32, forward_ntt_i32, inverse_ntt_cyclic_i32,
    inverse_ntt_i32, pointwise_mul_acc_i32,
};

/// Whether the NEON NTT path is active. Cached on first call.
/// Set `AKITA_SCALAR_NTT=1` to force scalar fallback.
pub fn use_neon_ntt() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("AKITA_SCALAR_NTT").map_or(true, |v| v != "1"))
}

#[cfg(test)]
mod tests;
