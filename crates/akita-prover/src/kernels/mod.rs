//! Low-level NTT and digit-decomposition kernels.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) mod decompose_fold_avx;
#[cfg(target_arch = "aarch64")]
pub(crate) mod decompose_fold_neon;
pub mod linear;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) use decompose_fold_avx as avx_decompose_fold;
#[cfg(target_arch = "aarch64")]
pub(crate) use decompose_fold_neon as neon_decompose_fold;
