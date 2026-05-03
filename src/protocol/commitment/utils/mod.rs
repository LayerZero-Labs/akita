//! Utility helpers for commitment internals.

pub mod linear;
pub(crate) mod matrix;
#[cfg(feature = "disk-persistence")]
pub(crate) mod norm;
pub mod ntt_cache;
