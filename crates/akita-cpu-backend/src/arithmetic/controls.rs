//! Application-facing cache controls and public resource diagnostics.
use super::{CpuBackend, PreparedCrtNttProfile, PreparedNttCacheMetric};
use crate::opaque::NttExecutionRequirements;
use akita_error::AkitaError;
use akita_types::FoldSchedule;

impl<F: jolt_field::Field + jolt_field::CanonicalEncoding, E> CpuBackend<F, E> {
    /// Prepare the CPU cache union for a complete commit-and-prove workload.
    pub fn prewarm(&self, schedule: &FoldSchedule) -> Result<(), AkitaError> {
        let prepared = self.prepared()?;
        let planned = NttExecutionRequirements::from_commit_and_prove_schedule(schedule)?;
        super::requirements::warm_joined_ntt_requirements(self, prepared, &planned)
    }

    /// Initialized setup transforms for application profiling.
    pub fn shared_ntt_cache_metrics(&self) -> Result<Vec<PreparedNttCacheMetric>, AkitaError> {
        self.prepared()?.shared_ntt_cache_metrics()
    }

    /// Arithmetic capacity profile for an initialized ring degree.
    pub fn shared_ntt_profile(&self, ring_d: usize) -> Result<PreparedCrtNttProfile, AkitaError> {
        self.prepared()?.shared_ntt_profile(ring_d)
    }
}
