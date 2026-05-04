//! Config adapters for shared SIS-derivation primitives.
//!
//! The policy-free derivation math lives in `akita-types`; this module keeps
//! only the adapters that need the root-owned [`CommitmentConfig`] trait.

use super::schedule::hachi_recursive_level_layout_from_params;
use crate::protocol::config::CommitmentConfig;
use akita_algebra::SparseChallengeConfig;
use akita_field::HachiError;
pub(crate) use akita_types::decomp_depths;
use akita_types::CommitmentEnvelope;
use akita_types::{HachiScheduleInputs, LevelParams};

/// Derive SIS-secure recursive (level > 0) params from the active envelope.
pub(crate) fn sis_derived_recursive_params<Cfg: CommitmentConfig>(
    d: usize,
    log_basis: u32,
    current_w_len: usize,
    stage1_config: &SparseChallengeConfig,
    envelope: &CommitmentEnvelope,
) -> Option<LevelParams> {
    let tentative =
        LevelParams::params_only(d, log_basis, envelope.max_n_a, 1, 1, stage1_config.clone());
    let layout = hachi_recursive_level_layout_from_params::<Cfg>(&tentative, current_w_len).ok()?;
    akita_types::sis_derived_recursive_params_for_layout(
        d,
        log_basis,
        stage1_config,
        envelope,
        &layout,
    )
}

/// Derive SIS-secure root params for a concrete root layout.
pub(crate) fn sis_derived_root_params_for_layout<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, HachiError> {
    akita_types::sis_derived_root_params_for_layout(
        Cfg::D,
        Cfg::decomposition(),
        Cfg::stage1_challenge_config(Cfg::D),
        inputs,
        lp,
    )
}

/// Build a root `LevelParams` from a candidate parameter set by splitting
/// `max_num_vars` into outer (`m`) and inner (`r`) variables.
pub(crate) fn derived_root_commitment_layout_from_params<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    params: &LevelParams,
    allow_zero_outer: bool,
) -> Result<LevelParams, HachiError> {
    akita_types::derived_root_commitment_layout_from_params(
        inputs,
        Cfg::decomposition(),
        params,
        allow_zero_outer,
    )
}
