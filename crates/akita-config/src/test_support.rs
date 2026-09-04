//! Test-only layout helpers shared by the workspace's integration tests and
//! unit tests.
//!
//! Everything in this module is gated behind tests or the `test-support`
//! feature, which production builds never enable. Production callers load
//! artifact bytes at an application boundary and pass the resulting catalog
//! explicitly.
//!
use akita_error::AkitaError;

use crate::CommitmentConfig;

/// Path to this config's checked-in workspace schedule artifact.
pub fn workspace_schedule_artifact_path<Cfg: CommitmentConfig>() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("artifacts/schedules")
        .join(format!("{}.aks", Cfg::schedule_family_name()))
}

/// Load this config's checked-in workspace schedule artifact.
pub fn workspace_schedule_catalog<Cfg: CommitmentConfig>(
) -> Result<crate::TrustedScheduleCatalog, AkitaError> {
    let path = workspace_schedule_artifact_path::<Cfg>();
    let bytes = std::fs::read(&path).map_err(|error| {
        AkitaError::InvalidSetup(format!(
            "failed to read workspace schedule artifact {}: {error}",
            path.display()
        ))
    })?;
    crate::trusted_schedule_catalog_from_bytes::<Cfg>(&bytes)
}

/// Minimal setup seed for schedule ring-dimension integration tests.
#[must_use]
pub fn ring_plan_test_seed() -> akita_types::AkitaSetupDescriptor {
    akita_types::AkitaSetupDescriptor {
        max_num_vars: 20,
        max_num_batched_polys: 1,
        num_field_elements: 1 << 20,
        setup_seed: [0u8; 32].into(),
    }
}
