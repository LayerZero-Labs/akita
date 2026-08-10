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
#[cfg(test)]
mod streamed_tests;

pub use prepared::{CpuPreparedSetup, PreparedCrtNttProfile, PreparedNttCacheMetric};

/// CPU backend using the existing Rust/Rayon kernels.
///
/// These deployment resource limits choose equivalent execution paths. They
/// do not affect protocol parameters or proof bytes.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct CpuBackend {
    max_cached_ring_switch_elements: usize,
    onehot_scratch_bytes_per_worker: usize,
}

impl CpuBackend {
    /// Default maximum cached extent for a ring-switch NTT operation.
    pub const DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS: usize = 1 << 21;

    /// Default temporary one-hot commitment memory per worker.
    pub const DEFAULT_ONEHOT_SCRATCH_BYTES_PER_WORKER: usize = 8 << 20;

    /// CPU backend with the default resource limits.
    pub const DEFAULT: Self = Self {
        max_cached_ring_switch_elements: Self::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS,
        onehot_scratch_bytes_per_worker: Self::DEFAULT_ONEHOT_SCRATCH_BYTES_PER_WORKER,
    };

    /// Create a CPU backend with explicit resource limits.
    pub fn with_resource_limits(
        max_cached_ring_switch_elements: usize,
        onehot_scratch_bytes_per_worker: usize,
    ) -> Result<Self, akita_field::AkitaError> {
        if onehot_scratch_bytes_per_worker == 0 {
            return Err(akita_field::AkitaError::InvalidSetup(
                "CPU one-hot scratch bytes per worker must be nonzero".into(),
            ));
        }
        Ok(Self {
            max_cached_ring_switch_elements,
            onehot_scratch_bytes_per_worker,
        })
    }

    /// Largest ring-switch operation extent retained as an NTT cache.
    pub const fn max_cached_ring_switch_elements(&self) -> usize {
        self.max_cached_ring_switch_elements
    }

    /// Temporary one-hot commitment memory allowed per worker.
    pub const fn onehot_scratch_bytes_per_worker(&self) -> usize {
        self.onehot_scratch_bytes_per_worker
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
