//! CPU compute backend and its prepared setup caches.

mod commitment;
mod commitment_stage;
#[cfg(test)]
mod commitment_tests;
mod compression;
mod compression_cache;
mod compression_stage;
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
#[cfg(test)]
mod streamed_tests;

pub use commitment_stage::{CpuInnerCommitOperation, CpuOuterCommitOperation};
pub use compression_stage::CpuCompressionOperation;
pub use prepared::{CpuPreparedSetup, PreparedCrtNttProfile, PreparedNttCacheMetric};

/// CPU backend using the existing Rust/Rayon kernels.
///
/// The ring-switch cache limit chooses equivalent execution paths without
/// affecting protocol parameters or proof bytes. Commitment scratch is sized
/// automatically for each operation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct CpuBackend {
    max_cached_ring_switch_elements: usize,
}

impl CpuBackend {
    /// Default maximum cached extent for a ring-switch NTT operation.
    pub const DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS: usize = 1 << 21;

    /// CPU backend with the default ring-switch cache limit.
    pub const DEFAULT: Self = Self {
        max_cached_ring_switch_elements: Self::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS,
    };

    /// Create a CPU backend with a ring-switch cache limit.
    /// Zero streams every supported operation; `usize::MAX` retains all of them.
    pub const fn with_ring_switch_cache_limit(max_cached_ring_switch_elements: usize) -> Self {
        Self {
            max_cached_ring_switch_elements,
        }
    }

    /// Largest ring-switch operation extent retained as an NTT cache.
    pub const fn max_cached_ring_switch_elements(&self) -> usize {
        self.max_cached_ring_switch_elements
    }

    #[inline]
    pub(crate) fn ntt_operation_uses_cache(
        &self,
        cluster: crate::compute::requirements::NttOperationCluster,
        num_ring_elements: usize,
    ) -> bool {
        let cached = cluster != crate::compute::requirements::NttOperationCluster::RingSwitch
            || num_ring_elements <= self.max_cached_ring_switch_elements;
        tracing::debug!(
            ?cluster,
            num_ring_elements,
            max_cached_ring_switch_elements = self.max_cached_ring_switch_elements,
            cached,
            "CPU NTT execution policy"
        );
        cached
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::DEFAULT
    }
}
