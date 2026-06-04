//! x86 runtime dispatch helpers for CRT NTT SIMD kernels.
//!
//! `AKITA_SCALAR_NTT=1` forces the scalar fallback for all CRT NTT SIMD.
//! `AKITA_AVX_NTT=0` disables only x86 CRT NTT SIMD. AVX-512 kernels are the
//! default when the host advertises the required features; `AKITA_AVX512_NTT=0`
//! forces the AVX2 path for A/B testing or hosts with AVX-512 frequency
//! penalties.

mod d32;
mod montgomery;
mod pointwise;
mod runtime;
#[cfg(test)]
mod tests;
mod transform_i32;
mod wide512;

pub use runtime::{avx_ntt_mode, use_avx2_transform_ntt, AvxNttMode};

use montgomery::{
    mont_mul_16x_i32_avx512, mont_mul_4x_i32_avx2, mont_mul_8x_i32_avx2,
    reduce_range_16x_i32_avx512, reduce_range_4x_i32_avx2, reduce_range_8x_i32_avx2,
};
pub use pointwise::{add_reduce_i16, add_reduce_i32, add_reduce_i32_avx512};
pub(crate) use pointwise::{
    pointwise_mul_acc_i16, pointwise_mul_acc_i32, pointwise_mul_acc_i32_avx512,
};
#[cfg(test)]
use runtime::{select_avx_ntt_mode, AvxCpuFeatures};
pub(crate) use transform_i32::{
    forward_ntt_cyclic_i32, forward_ntt_i32, inverse_ntt_cyclic_i32, inverse_ntt_i32,
};
