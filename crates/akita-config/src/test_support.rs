//! Test-only layout helpers shared by the workspace's integration tests and
//! unit tests.
//!
//! Everything in this module is gated behind tests or the `test-support`
//! feature, which production builds never enable. Production callers load
//! artifact bytes at an application boundary and pass the resulting catalog
//! explicitly.
//!
use akita_error::AkitaError;
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupBatchProfile, CommittedGroupParams, OpeningClaimsLayout,
    OpeningScheduleSelection, PolynomialGroupLayout, SetupMatrixCapacity,
};

use crate::CommitmentConfig;

/// Explicit test fixture for schedule lookup and setup sizing.
///
/// Production configuration types deliberately do not own schedules. Tests
/// opt into this separate fixture trait so synthetic catalogs cannot leak into
/// the production protocol surface.
pub trait TestScheduleProvider: CommitmentConfig {
    /// Size setup through this fixture's catalog.
    fn setup_matrix_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixCapacity, AkitaError> {
        let catalog = workspace_schedule_catalog::<Self>()?;
        crate::trusted_setup_matrix_capacity::<Self>(&catalog, max_num_vars, max_num_batched_polys)
    }

    /// Resolve one exact runtime key through this fixture's catalog.
    fn resolve_catalog_row_for_key(
        key: &AkitaScheduleLookupKey,
    ) -> Result<crate::ResolvedScheduleRow, AkitaError> {
        workspace_schedule_catalog::<Self>()?.resolve_key(key)
    }

    /// Resolve a scalar opening layout through this fixture's catalog.
    fn resolve_catalog_row_for_opening(
        layout: &OpeningClaimsLayout,
    ) -> Result<crate::ResolvedScheduleRow, AkitaError> {
        layout.check()?;
        if layout.num_groups() != 1 {
            return Err(AkitaError::InvalidInput(
                "grouped schedule selection requires exact committed-group descriptors".to_string(),
            ));
        }
        Self::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
            layout.root_final_group_layout()?,
        ))
    }

    /// Resolve the independently committed profile for one group.
    fn profile_without_precommitted_groups(
        group: PolynomialGroupLayout,
    ) -> Result<akita_types::GroupCommitPhaseParams, AkitaError> {
        Ok(
            Self::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(group))?
                .profiles()
                .final_group,
        )
    }

    /// Resolve exact committed profiles through this fixture's catalog.
    fn resolve_catalog_row_for_profiles(
        profiles: &CommittedGroupBatchProfile,
    ) -> Result<crate::ResolvedScheduleRow, AkitaError> {
        workspace_schedule_catalog::<Self>()?.resolve_profiles(profiles)
    }

    /// Resolve a public row selection through this fixture's catalog.
    fn resolve_schedule_selection(
        selection: OpeningScheduleSelection,
    ) -> Result<crate::ResolvedScheduleRow, AkitaError> {
        workspace_schedule_catalog::<Self>()?.resolve_selection(selection)
    }
}

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

/// Derive the per-polynomial commitment layout optimized for a batch of
/// `num_polynomials` polynomials with `num_vars` variables.
///
/// First reads the runtime schedule. When the schedule is a root fold it
/// returns that root layout; for a direct-only schedule it derives the batched
/// root commit layout
/// `Cfg::resolve_catalog_row_for_opening` derives for the same
/// `num_polynomials` (so the fallback layout is sized for the requested batch,
/// not a singleton).
///
/// Tests, benches, and the `profile` example use this to pre-size per-poly
/// inputs (e.g. `OneHotPoly`) so the `num_positions_per_block` / `num_live_blocks` line up with
/// what `Scheme::commit` will use under the batched layout. Production
/// callers always go through
/// `Cfg::resolve_catalog_row_for_opening(&opening_batch)` and ask the resolved
/// root for its final group.
/// instead.
///
/// # Errors
///
/// Returns an error if the layout parameters overflow or are invalid.
pub fn akita_batched_root_layout<Cfg>(
    num_vars: usize,
    num_polynomials: usize,
) -> Result<CommittedGroupParams, AkitaError>
where
    Cfg: CommitmentConfig + TestScheduleProvider,
{
    let lookup_key = PolynomialGroupLayout::new(num_vars, num_polynomials);
    let schedule = Cfg::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(lookup_key))?;
    let layout = schedule.schedule().root.params.clone();
    tracing::info!(
        num_vars,
        num_polynomials,
        root_m = layout.position_index_bits(),
        root_r = layout.block_index_bits(),
        root_lb_inner = layout.inner().digits.log_basis,
        root_lb_outer = layout.outer().digits.log_basis,
        root_lb_open = layout.open().digits.log_basis,
        "batched root split: read from runtime schedule"
    );
    Ok(layout)
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
