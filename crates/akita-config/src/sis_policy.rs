//! Config adapters for shared SIS-derivation primitives.
//!
//! The policy-free derivation math lives in `akita-types`; this module keeps
//! only the adapters that need [`super::CommitmentConfig`].

use crate::CommitmentConfig;
use akita_challenges::{SparseChallengeConfig, Stage1ChallengeShape};
use akita_field::AkitaError;
use akita_types::generated::sis_floor::MAX_RANK;
use akita_types::CommitmentEnvelope;
use akita_types::{AkitaScheduleInputs, LevelParams};

/// Pick the production stage-1 challenge shape for a given sparse challenge
/// family. Matches `proof_optimized::stage1_challenge_shape_for_config` so the
/// recursive derivation sees the same shape the runtime will use.
fn stage1_challenge_shape_for_config(config: &SparseChallengeConfig) -> Stage1ChallengeShape {
    match config {
        SparseChallengeConfig::BoundedL1Norm => Stage1ChallengeShape::Flat,
        SparseChallengeConfig::Uniform { .. } | SparseChallengeConfig::ExactShell { .. } => {
            Stage1ChallengeShape::Tensor
        }
    }
}

/// Derive SIS-secure recursive (level > 0) params from the active envelope.
///
/// Mirrors the root-level fixed-point pattern in
/// `proof_optimized_root_level_layout_with_log_basis`: at every iteration the
/// tentative layout carries the production [`Stage1ChallengeShape`] so the
/// SIS extraction collision bucket reflects the *runtime* shape (including
/// the `4ω` tensor extraction-degradation), and the iteration continues until
/// the rank derived from `sis_derived_recursive_params_for_layout` matches
/// the rank used to lay out the level.
///
/// This avoids the historic two-step inconsistency where:
///   1. the tentative layout was built with `params_only` (shape defaults to
///      `Flat`), so the SIS lookup happened at the *flat* extraction collision
///      even when the production shape is `Tensor`; and
///   2. after the rank was bumped, `optimal_m_r_split` could re-pick a
///      different `(m_vars, r_vars)` whose `inner_width` no longer fit at
///      the bumped rank — and the rank was never re-validated.
///
/// Each iteration produces a `(rank, layout)` pair where `layout` is the
/// layout for the *current* `candidate_n_a` and the SIS lookup at `layout`'s
/// `inner_width` returns the same `candidate_n_a` (i.e. the rank is a
/// fixed point under `(rank -> layout -> rank)`).
///
/// Returns `None` if the iteration fails to converge within
/// [`MAX_RANK`](akita_types::generated::sis_floor::MAX_RANK) tries.
pub(crate) fn sis_derived_recursive_params<Cfg: CommitmentConfig>(
    d: usize,
    log_basis: u32,
    current_w_len: usize,
    stage1_config: &SparseChallengeConfig,
    envelope: &CommitmentEnvelope,
) -> Option<LevelParams> {
    let production_shape = stage1_challenge_shape_for_config(stage1_config);
    let mut candidate_n_a = envelope.max_n_a.max(1);
    for _ in 0..(MAX_RANK + 1) {
        let mut tentative = LevelParams::params_only(
            d,
            log_basis,
            candidate_n_a,
            envelope.max_n_b.max(1),
            envelope.max_n_d.max(1),
            stage1_config.clone(),
        );
        tentative.stage1_challenge_shape = production_shape;
        let layout = akita_types::recursive_level_layout_from_params(
            &tentative,
            current_w_len,
            Cfg::decomposition(),
        )
        .ok()?;
        let derived = akita_types::sis_derived_recursive_params_for_layout(
            d,
            log_basis,
            stage1_config,
            envelope,
            &layout,
        )?;
        if derived.a_key.row_len() <= candidate_n_a {
            // Fixed point: the *candidate*'s layout (built at `candidate_n_a`)
            // is secure at rank `derived.a_key.row_len() <= candidate_n_a`,
            // so it is also secure at `candidate_n_a` itself. Return the
            // candidate's layout with the (possibly over-provisioned) rank
            // we used to lay it out.
            let mut params = derived;
            params.a_key = akita_types::AjtaiKeyParams::try_new(
                candidate_n_a,
                params.a_key.col_len(),
                params.a_key.collision_inf(),
                d,
            )
            .ok()?;
            return Some(params);
        }
        // The derived rank is *larger* than the tentative rank, meaning the
        // tentative layout was under-secure. Bump candidate_n_a and try
        // again. Bounded by `MAX_RANK + 1` iterations.
        candidate_n_a = derived.a_key.row_len();
    }
    None
}

/// Derive SIS-secure root params for a concrete root layout.
pub(crate) fn sis_derived_root_params_for_layout<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, AkitaError> {
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
