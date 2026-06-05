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

use std::marker::PhantomData;

use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleLookupKey, ClaimIncidenceSummary, LevelParams, SetupMatrixEnvelope,
    TerminalProofMode,
};

use crate::CommitmentConfig;

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

/// `Cfg` wrapper that selects [`TerminalProofMode::DirectRingRelations`] while
/// delegating every other policy hook to the inner config.
#[derive(Clone, Copy, Debug, Default)]
pub struct DirectTerminalCfg<Cfg>(PhantomData<Cfg>);

#[allow(clippy::expl_impl_clone_on_copy)]
impl<Cfg> DirectTerminalCfg<Cfg> {
    /// Construct the marker value.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Cfg: CommitmentConfig> CommitmentConfig for DirectTerminalCfg<Cfg> {
    type Field = Cfg::Field;
    type ClaimField = Cfg::ClaimField;
    type ChallengeField = Cfg::ChallengeField;

    const D: usize = Cfg::D;

    fn decomposition() -> akita_types::DecompositionParams {
        Cfg::decomposition()
    }

    fn fold_challenge_shape_at_level(
        inputs: akita_types::AkitaScheduleInputs,
    ) -> akita_challenges::TensorChallengeShape {
        Cfg::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_family() -> akita_types::SisModulusFamily {
        Cfg::sis_modulus_family()
    }

    fn terminal_proof_mode() -> TerminalProofMode {
        TerminalProofMode::DirectRingRelations
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Cfg::ring_subfield_embedding_norm_bound()
    }

    fn onehot_chunk_size() -> usize {
        Cfg::onehot_chunk_size()
    }

    fn basis_range() -> (u32, u32) {
        Cfg::basis_range()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        Cfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)
    }

    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        Cfg::ring_challenge_config(d)
    }
}
