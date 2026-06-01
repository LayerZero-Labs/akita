//! Compute width (col) and rank (row) length of each Ajtai commitment key.
//!
//! In Akita, these are the A, B, and D matrices.
//!
//! Procedure: given inputs, each function computes the width of its key
//! and, against the pre-computed secure SIS ranks, the corresponding
//! rank, and returns an `AjtaiKeyParams`.
//!
//! These helpers are `Cfg`-free: every per-preset input is carried by the
//! plain-value [`PlannerPolicy`] plus the `stage1` challenge-config closure,
//! matching the shape `akita_types::schedule_from_entry_bits` already uses.

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};
use akita_types::layout::digit_math::num_digits_for_bound;
use akita_types::AjtaiKeyParams;

use crate::PlannerPolicy;

/// Stage-1 sparse-challenge closure shared by the planner entry points.
pub(crate) type Stage1Fn<'a> = &'a dyn Fn(usize) -> Result<SparseChallengeConfig, AkitaError>;

/// Per-witness decomposition / binding-norm rule.
pub(crate) enum WitnessType {
    /// Decomposed witness `s_i`. Committed via the A matrix.
    S,
    /// Decomposed `t_i = A · s_i`. Committed via the B matrix.
    T,
    /// Decomposed `w_i = a · s_i`. Committed via the D matrix.
    W,
}

impl WitnessType {
    /// Witness infinity norm to satisfy weak binding (Hachi paper, Lemma 7).
    ///
    /// The S (witness `s`) base norm is level-dependent: the root commits
    /// the balanced-decomposed witness, bounded per coefficient by `2·β`
    /// with `β = 2^(lb−1) − 1` (or `1` when `log_commit_bound == 1`); a
    /// recursive level commits the full digit-range witness, bounded by
    /// `2^lb − 1`.
    pub(crate) fn binding_norm(
        self,
        policy: &PlannerPolicy,
        stage1: Stage1Fn<'_>,
        log_basis: u32,
        is_root_level: bool,
    ) -> Result<u32, AkitaError> {
        match self {
            Self::S => {
                let base = if is_root_level {
                    let beta = if policy.decomposition.log_commit_bound == 1 {
                        1
                    } else {
                        (1u32 << (log_basis - 1)) - 1
                    };
                    2 * beta
                } else {
                    (1u32 << log_basis) - 1
                };
                Ok(base
                    * stage1(policy.ring_dimension)?.infinity_norm()
                    * policy.ring_subfield_norm_bound)
            }
            Self::T | Self::W => Ok((1u32 << log_basis) - 1),
        }
    }

    /// Number of `log_basis`-bit digits per coefficient under this
    /// witness's decomposition rule.
    ///
    /// The S commit bound is level-dependent: the root commits the
    /// witness against its configured `log_commit_bound`, while a
    /// recursive level commits the balanced-digit witness, whose commit
    /// bound collapses to `log_basis`.
    pub(crate) fn decomposed_num_digits(
        self,
        policy: &PlannerPolicy,
        log_basis: u32,
        is_root_level: bool,
    ) -> usize {
        let field_bits = policy.decomposition.field_bits();
        let bound = match self {
            Self::S => {
                if is_root_level {
                    policy.decomposition.log_commit_bound
                } else {
                    log_basis
                }
            }
            Self::T | Self::W => policy
                .decomposition
                .log_open_bound
                .unwrap_or(policy.decomposition.log_commit_bound),
        };
        num_digits_for_bound(bound, field_bits, log_basis)
    }
}

pub(crate) fn compute_ajtai_key_params_a(
    policy: &PlannerPolicy,
    stage1: Stage1Fn<'_>,
    block_len: usize,
    log_basis: u32,
    is_root_level: bool,
) -> Result<Option<AjtaiKeyParams>, AkitaError> {
    let inf_norm = WitnessType::S.binding_norm(policy, stage1, log_basis, is_root_level)?;
    let num_digits = WitnessType::S.decomposed_num_digits(policy, log_basis, is_root_level);
    let Some(width) = block_len.checked_mul(num_digits) else {
        return Ok(None);
    };
    let d = policy.ring_dimension as u32;
    let Some(ceil_inf_norm) = ceil_supported_collision(policy.sis_family, d, inf_norm) else {
        return Ok(None);
    };
    let Some(rank) = min_rank_for_secure_width(policy.sis_family, d, ceil_inf_norm, width as u64)
    else {
        return Ok(None);
    };
    AjtaiKeyParams::try_new(
        policy.sis_family,
        rank,
        width,
        ceil_inf_norm,
        policy.ring_dimension,
    )
    .map(Some)
}

pub(crate) fn compute_ajtai_key_params_b(
    policy: &PlannerPolicy,
    stage1: Stage1Fn<'_>,
    matrix_a_rank: usize,
    num_blocks: usize,
    t_vectors: usize,
    log_basis: u32,
) -> Result<Option<AjtaiKeyParams>, AkitaError> {
    let inf_norm = WitnessType::T.binding_norm(policy, stage1, log_basis, true)?;
    let num_digits = WitnessType::T.decomposed_num_digits(policy, log_basis, true);
    let Some(width) = matrix_a_rank
        .checked_mul(num_digits)
        .and_then(|w| w.checked_mul(num_blocks))
        .and_then(|w| w.checked_mul(t_vectors))
    else {
        return Ok(None);
    };
    let d = policy.ring_dimension as u32;
    let Some(ceil_inf_norm) = ceil_supported_collision(policy.sis_family, d, inf_norm) else {
        return Ok(None);
    };
    let Some(rank) = min_rank_for_secure_width(policy.sis_family, d, ceil_inf_norm, width as u64)
    else {
        return Ok(None);
    };
    AjtaiKeyParams::try_new(
        policy.sis_family,
        rank,
        width,
        ceil_inf_norm,
        policy.ring_dimension,
    )
    .map(Some)
}

/// The A, B, and D Ajtai keys for one fold level.
pub(crate) type AjtaiKeysParams = (AjtaiKeyParams, AjtaiKeyParams, AjtaiKeyParams);

/// Compute all three Ajtai keys (A, B, D) for one fold level in one shot.
pub(crate) fn compute_all_ajtai_keys_params(
    policy: &PlannerPolicy,
    stage1: Stage1Fn<'_>,
    block_len: usize,
    num_blocks: usize,
    t_vectors: usize,
    log_basis: u32,
    is_root_level: bool,
) -> Result<Option<AjtaiKeysParams>, AkitaError> {
    let Some(a_key) =
        compute_ajtai_key_params_a(policy, stage1, block_len, log_basis, is_root_level)?
    else {
        return Ok(None);
    };
    let Some(b_key) = compute_ajtai_key_params_b(
        policy,
        stage1,
        a_key.row_len(),
        num_blocks,
        t_vectors,
        log_basis,
    )?
    else {
        return Ok(None);
    };
    let Some(d_key) = compute_ajtai_key_params_d(policy, stage1, num_blocks, t_vectors, log_basis)?
    else {
        return Ok(None);
    };
    Ok(Some((a_key, b_key, d_key)))
}

pub(crate) fn compute_ajtai_key_params_d(
    policy: &PlannerPolicy,
    stage1: Stage1Fn<'_>,
    num_blocks: usize,
    t_vectors: usize,
    log_basis: u32,
) -> Result<Option<AjtaiKeyParams>, AkitaError> {
    let inf_norm = WitnessType::W.binding_norm(policy, stage1, log_basis, true)?;
    let num_digits_open = WitnessType::W.decomposed_num_digits(policy, log_basis, true);
    let Some(width) = num_digits_open
        .checked_mul(num_blocks)
        .and_then(|w| w.checked_mul(t_vectors))
    else {
        return Ok(None);
    };
    let d = policy.ring_dimension as u32;
    let Some(ceil_inf_norm) = ceil_supported_collision(policy.sis_family, d, inf_norm) else {
        return Ok(None);
    };
    let Some(rank) = min_rank_for_secure_width(policy.sis_family, d, ceil_inf_norm, width as u64)
    else {
        return Ok(None);
    };
    AjtaiKeyParams::try_new(
        policy.sis_family,
        rank,
        width,
        ceil_inf_norm,
        policy.ring_dimension,
    )
    .map(Some)
}
