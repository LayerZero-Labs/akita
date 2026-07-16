//! Test-only layout helpers shared by the workspace's integration tests,
//! unit tests, and the `profile` example.
//!
//! Everything in this module is gated behind the `test-support` Cargo
//! feature, which production builds never enable: it is switched on only
//! through the dev-dependency edge of `akita-pcs`, so the helpers here are
//! compiled for test/example/bench targets and are
//! absent from every shipped artifact. Production callers size their
//! per-poly inputs through [`CommitmentConfig::get_params_for_batched_commitment`]
//! directly and never need this module.

use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleLookupKey, LevelParams, OpeningClaimsLayout, PolynomialGroupLayout,
};

use crate::CommitmentConfig;

/// Derive the per-polynomial commitment layout optimized for a batch of
/// `num_polynomials` polynomials with `num_vars` variables.
///
/// First reads the runtime schedule (table hit or DP fallback). When the
/// schedule is a root fold it returns that root layout; for a direct-only
/// schedule it falls back to the batched root commit layout
/// `Cfg::get_params_for_batched_commitment` derives for the same
/// `num_polynomials` (so the fallback layout is sized for the requested batch,
/// not a singleton).
///
/// Tests, benches, and the `profile` example use this to pre-size per-poly
/// inputs (e.g. `OneHotPoly`) so the `positions_per_block` / `live_block_count` line up with
/// what `Scheme::commit` will use under the batched layout. Production
/// callers always go through `Cfg::get_params_for_batched_commitment(&opening_batch)`
/// instead.
///
/// # Errors
///
/// Returns an error if the layout parameters overflow or are invalid.
pub fn akita_batched_root_layout<Cfg>(
    num_vars: usize,
    num_polynomials: usize,
) -> Result<LevelParams, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let lookup_key = PolynomialGroupLayout::new(num_vars, num_polynomials);
    let schedule = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(lookup_key))?;
    if let Some(root) = akita_types::schedule_root_fold_step(&schedule) {
        let layout = root.params.clone();
        tracing::info!(
            num_vars,
            num_polynomials,
            total_bytes = schedule.total_bytes,
            root_m = layout.position_index_bits(),
            root_r = layout.block_index_bits(),
            root_lb = layout.log_basis,
            "batched root split: read from runtime schedule"
        );
        return Ok(layout);
    }
    tracing::info!(
        num_vars,
        num_polynomials,
        "batched root split: schedule is direct-only, falling back to config root layout"
    );
    // Size the fallback for the requested batch (`num_polynomials`), not a
    // singleton — otherwise the per-poly inputs would be smaller than the
    // batched commit layout `Scheme::commit` actually uses.
    Cfg::get_params_for_batched_commitment(&OpeningClaimsLayout::new(num_vars, num_polynomials)?)
}
/// Minimal setup seed for schedule ring-dimension integration tests.
#[must_use]
pub fn ring_plan_test_seed(gen_ring_dim: usize) -> akita_types::AkitaSetupSeed {
    akita_types::AkitaSetupSeed {
        max_num_vars: 20,
        max_num_batched_polys: 1,
        gen_ring_dim,
        max_setup_len: 1 << 20,
        public_matrix_seed: [0u8; 32],
    }
}
