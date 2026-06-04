//! Test-only layout helpers shared by the workspace's integration tests,
//! unit tests, and the `profile` example.
//!
//! Everything in this module is gated behind the `test-support` Cargo
//! feature, which production builds never enable: it is switched on only
//! through the dev-dependency edges of `akita-pcs` and `akita-scheme`, so
//! the helpers here are compiled for test/example/bench targets and are
//! absent from every shipped artifact. Production callers size their
//! per-poly inputs through [`CommitmentConfig::get_params_for_batched_commitment`]
//! directly and never need this module.

use akita_field::AkitaError;
use akita_types::{AkitaScheduleLookupKey, ClaimIncidenceSummary, LevelParams, Schedule};

use crate::{policy_of, CommitmentConfig};

/// Derive the per-polynomial commitment layout optimized for a batch of
/// `num_claims` polynomials with `num_vars` variables.
///
/// First reads the runtime schedule (table hit or DP fallback). When the
/// schedule is a root fold it returns that root layout; for a direct-only
/// schedule it falls back to the batched root commit layout
/// `Cfg::get_params_for_batched_commitment` derives for the same
/// `num_claims` (so the fallback layout is sized for the requested batch,
/// not a singleton).
///
/// Tests, benches, and the `profile` example use this to pre-size per-poly
/// inputs (e.g. `OneHotPoly`) so the `block_len` / `num_blocks` line up with
/// what `Scheme::commit` will use under the batched layout. Production
/// callers always go through `Cfg::get_params_for_batched_commitment(&incidence)`
/// instead.
///
/// # Errors
///
/// Returns an error if the layout parameters overflow or are invalid.
pub fn akita_batched_root_layout<Cfg>(
    num_vars: usize,
    num_claims: usize,
) -> Result<LevelParams, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let lookup_key = AkitaScheduleLookupKey::new(num_vars, num_claims, num_claims, 1);
    let schedule = Cfg::runtime_schedule(lookup_key)?;
    if let Some(root) = akita_types::schedule_root_fold_step(&schedule) {
        let layout = root.params.clone();
        tracing::info!(
            num_vars,
            num_claims,
            total_bytes = schedule.total_bytes,
            root_m = layout.log_block_len(),
            root_r = layout.log_num_blocks(),
            root_lb = layout.log_basis,
            "batched root split: read from runtime schedule"
        );
        return Ok(layout);
    }
    tracing::info!(
        num_vars,
        num_claims,
        "batched root split: schedule is direct-only, falling back to config root layout"
    );
    // Size the fallback for the requested batch (`num_claims`), not a
    // singleton — otherwise the per-poly inputs would be smaller than the
    // batched commit layout `Scheme::commit` actually uses.
    Cfg::get_params_for_batched_commitment(&ClaimIncidenceSummary::same_point(
        num_vars, num_claims,
    )?)
}

/// Test-only helper for constructing a recursive carried-opening suffix from
/// an already committed recursive state.
pub fn recursive_carried_suffix_schedule<Cfg>(
    key: AkitaScheduleLookupKey,
    start_level: usize,
    current_w_len: usize,
    current_log_basis: u32,
) -> Result<Schedule, AkitaError>
where
    Cfg: CommitmentConfig,
{
    akita_planner::find_recursive_carried_suffix_schedule(
        key,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        start_level,
        current_w_len,
        current_log_basis,
    )
}
