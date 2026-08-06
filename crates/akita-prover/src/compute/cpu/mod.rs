//! CPU compute backend and its prepared setup caches.

mod commitment;
mod compression;
mod compression_cache;
mod cyclic_rows;
mod digit_rows;
mod exact_i16;
#[cfg(test)]
mod exact_i16_tests;
#[cfg(test)]
mod kernel_tests;
mod prepared;
#[cfg(test)]
mod prepared_tests;
mod ring_switch;

pub use prepared::{CpuPreparedSetup, PreparedCrtNttProfile, PreparedNttCacheMetric};

/// CPU backend using the existing Rust/Rayon kernels.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;
