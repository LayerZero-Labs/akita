//! Low-level NTT, matrix, and digit-decomposition kernels.

pub mod crt_ntt;
#[cfg(target_arch = "aarch64")]
pub(crate) mod decompose_fold_neon;
pub mod linear;
pub mod ntt_cache;

pub use crt_ntt::{build_ntt_slot, select_crt_ntt_params, NttSlotCache, ProtocolCrtNttParams};
pub use ntt_cache::MultiDNttCaches;

#[cfg(target_arch = "aarch64")]
pub(crate) use decompose_fold_neon as neon_decompose_fold;
