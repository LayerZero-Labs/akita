//! Shared compile-time gate for the vectorized forward-DIF NTT tail (NEON / AVX2).
//!
//! The SIMD forward NTT finishes its last two DIF stages in a dedicated tail that
//! transposes 4×4 coefficient blocks and runs them 4-wide. Each tail iteration
//! covers **16** `i32` coefficients (`base += 16` in `neon.rs` / `avx/montgomery.rs`),
//! so the vectorized path is only safe when `D` is a multiple of 16; otherwise
//! callers use the scalar butterfly loop for those stages.

/// `true` when ring degree `D` is a multiple of 16 and the vectorized tail may run.
#[inline]
pub const fn forward_dif_tail_eligible<const D: usize>() -> bool {
    D.is_multiple_of(16)
}
