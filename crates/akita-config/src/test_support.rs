//! Test-only layout helpers shared by the workspace's integration tests and
//! unit tests.
//!
//! Everything in this module is gated behind the `test-support` Cargo
//! feature, which production builds never enable. Production callers size
//! their per-poly inputs through
//! [`CommitmentConfig::resolve_catalog_row_for_opening`] directly and never
//! need this module.
//!
use akita_error::AkitaError;
use akita_types::{AkitaScheduleLookupKey, CommittedGroupParams, PolynomialGroupLayout};

use crate::CommitmentConfig;

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
    Cfg: CommitmentConfig,
{
    let lookup_key = PolynomialGroupLayout::new(num_vars, num_polynomials);
    let schedule = Cfg::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(lookup_key))?;
    let layout = schedule.schedule().root.params.clone();
    tracing::info!(
        num_vars,
        num_polynomials,
        root_m = layout.position_index_bits(),
        root_r = layout.block_index_bits(),
        root_lb_inner = layout.log_basis_inner,
        root_lb_outer = layout.log_basis_outer,
        root_lb_open = layout.log_basis_open,
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
