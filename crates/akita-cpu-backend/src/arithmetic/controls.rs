//! Application-facing cache controls and public resource diagnostics.
use super::{CpuBackend, PreparedCrtNttProfile, PreparedNttCacheMetric};
use crate::opaque::NttExecutionRequirements;
use akita_error::AkitaError;
use akita_types::FoldSchedule;
use jolt_field::{CanonicalEncoding, Field};

impl CpuBackend {
    /// Prepare the CPU cache union for a complete commit-and-prove workload.
    pub fn prewarm<F: Field + CanonicalEncoding>(
        &self,
        schedule: &FoldSchedule,
    ) -> Result<(), AkitaError> {
        let prepared = self.prepared::<F>()?;
        let planned = NttExecutionRequirements::from_commit_and_prove_schedule(schedule)?;
        super::requirements::warm_joined_ntt_requirements(self, prepared, &planned)
    }

    /// Initialized setup transforms for application profiling.
    pub fn shared_ntt_cache_metrics<F: Field + CanonicalEncoding>(
        &self,
    ) -> Result<Vec<PreparedNttCacheMetric>, AkitaError> {
        self.prepared::<F>()?.shared_ntt_cache_metrics()
    }

    /// Arithmetic capacity profile for an initialized ring degree.
    pub fn shared_ntt_profile<F: Field + CanonicalEncoding>(
        &self,
        ring_d: usize,
    ) -> Result<PreparedCrtNttProfile, AkitaError> {
        self.prepared::<F>()?.shared_ntt_profile(ring_d)
    }
}
