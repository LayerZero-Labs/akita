//! Config adapters for shared SIS-derivation primitives.
//!
//! The policy-free derivation math lives in `akita-types`; this module keeps
//! only the adapters that need [`super::CommitmentConfig`].

use crate::CommitmentConfig;
use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::CommitmentEnvelope;
use akita_types::{AkitaScheduleInputs, LevelParams};

/// Derive SIS-secure recursive (level > 0) params from the active envelope.
pub(crate) fn sis_derived_recursive_params<Cfg: CommitmentConfig>(
    d: usize,
    log_basis: u32,
    current_w_len: usize,
    stage1_config: &SparseChallengeConfig,
    envelope: &CommitmentEnvelope,
) -> Option<LevelParams> {
    let tentative = LevelParams::params_only(
        Cfg::sis_modulus_family(),
        d,
        log_basis,
        envelope.max_n_a,
        1,
        1,
        stage1_config.clone(),
    );
    let layout = akita_types::recursive_level_layout_from_params(
        &tentative,
        current_w_len,
        Cfg::decomposition(),
    )
    .ok()?;
    akita_types::sis_derived_recursive_params_for_layout(
        Cfg::sis_modulus_family(),
        d,
        log_basis,
        stage1_config,
        envelope,
        &layout,
    )
}

/// Derive SIS-secure root params for a concrete root layout.
pub(crate) fn sis_derived_root_params_for_layout<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, AkitaError> {
    akita_types::sis_derived_root_params_for_layout(
        Cfg::sis_modulus_family(),
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
    inputs: AkitaScheduleInputs,
    params: &LevelParams,
    allow_zero_outer: bool,
) -> Result<LevelParams, AkitaError> {
    akita_types::derived_root_commitment_layout_from_params(
        inputs,
        Cfg::decomposition(),
        params,
        allow_zero_outer,
    )
}
