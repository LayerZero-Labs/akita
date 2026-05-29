//! Compute width (col) and rank (row) length of each Ajtai commitment key.
//!
//! In Akita, these are the A, B, and D matrices.
//!
//! Procedure: given inputs, each function computes the width of its key
//! and, against the pre-computed secure SIS ranks, the corresponding
//! rank, and returns an `AjtaiKeyParams`.

use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};
use akita_types::layout::digit_math::{compute_num_digits_fold_with_claims, num_digits_for_bound};
use akita_types::AjtaiKeyParams;

/// Per-witness decomposition / binding-norm rule.
pub(crate) enum WitnessType {
    /// Decomposed witness `s_i`. Committed via the A matrix.
    S,
    /// Decomposed `t_i = A · s_i`. Committed via the B matrix.
    T,
    /// Decomposed `w_i = a · s_i`. Committed via the D matrix.
    W,
    /// Decomposed z = \sum_i c_i . s_i
    Z,
}

impl WitnessType {
    /// Witness infinity norm to satisfy weak binding (Hachi paper, Lemma 7).
    ///
    /// The S (witness `s`) base norm is level-dependent: the root commits
    /// the balanced-decomposed witness, bounded per coefficient by `2·β`
    /// with `β = 2^(lb−1) − 1` (or `1` when `log_commit_bound == 1`); a
    /// recursive level commits the full digit-range witness, bounded by
    /// `2^lb − 1`.
    pub(crate) fn binding_norm<Cfg: CommitmentConfig>(
        self,
        log_basis: u32,
        is_root_level: bool,
    ) -> Result<u32, AkitaError> {
        match self {
            Self::S => {
                let base = if is_root_level {
                    let beta = if Cfg::decomposition().log_commit_bound == 1 {
                        1
                    } else {
                        (1u32 << (log_basis - 1)) - 1
                    };
                    2 * beta
                } else {
                    (1u32 << log_basis) - 1
                };
                Ok(base
                    * Cfg::stage1_challenge_config(Cfg::D)?.infinity_norm()
                    * Cfg::ring_subfield_embedding_norm_bound())
            }
            Self::T | Self::W => Ok((1u32 << log_basis) - 1),
            Self::Z => unreachable!("Z has no SIS binding norm: not committed via A/B/D"),
        }
    }

    /// Number of `log_basis`-bit digits per coefficient under this
    /// witness's decomposition rule. Valid for S / T / W; Z goes through
    /// [`Self::decomposed_fold_num_digits`].
    ///
    /// The S commit bound is level-dependent: the root commits the
    /// witness against its configured `log_commit_bound`, while a
    /// recursive level commits the balanced-digit witness, whose commit
    /// bound collapses to `log_basis`.
    pub(crate) fn decomposed_num_digits<Cfg: CommitmentConfig>(
        self,
        log_basis: u32,
        is_root_level: bool,
    ) -> usize {
        let field_bits = Cfg::decomposition().field_bits();
        let bound = match self {
            Self::S => {
                if is_root_level {
                    Cfg::decomposition().log_commit_bound
                } else {
                    log_basis
                }
            }
            Self::T | Self::W => Cfg::decomposition()
                .log_open_bound
                .unwrap_or(Cfg::decomposition().log_commit_bound),
            Self::Z => {
                unreachable!("Z digit count is computed via decomposed_fold_num_digits")
            }
        };
        num_digits_for_bound(bound, field_bits, log_basis)
    }

    /// Number of `log_basis`-bit digits per Z-row coefficient after folding.
    /// Only valid for `Z`; S / T / W go through [`Self::decomposed_num_digits`].
    pub(crate) fn decomposed_fold_num_digits<Cfg: CommitmentConfig>(
        self,
        log_basis: u32,
        r_vars: usize,
        challenge_l1_mass: usize,
        num_claims: usize,
    ) -> usize {
        match self {
            Self::S | Self::T | Self::W => {
                unreachable!("decomposed_fold_num_digits is only valid for Z (z-pre rows)")
            }
            Self::Z => compute_num_digits_fold_with_claims(
                r_vars,
                challenge_l1_mass,
                log_basis,
                num_claims,
                Cfg::decomposition().field_bits(),
            ),
        }
    }
}

pub(crate) fn compute_ajtai_key_params_a<Cfg: CommitmentConfig>(
    block_len: usize,
    log_basis: u32,
    is_root_level: bool,
) -> Result<Option<AjtaiKeyParams>, AkitaError> {
    let inf_norm = WitnessType::S.binding_norm::<Cfg>(log_basis, is_root_level)?;
    let num_digits = WitnessType::S.decomposed_num_digits::<Cfg>(log_basis, is_root_level);
    let Some(width) = block_len.checked_mul(num_digits) else {
        return Ok(None);
    };
    let Some(ceil_inf_norm) =
        ceil_supported_collision(Cfg::sis_modulus_family(), Cfg::D as u32, inf_norm)
    else {
        return Ok(None);
    };
    let Some(rank) = min_rank_for_secure_width(
        Cfg::sis_modulus_family(),
        Cfg::D as u32,
        ceil_inf_norm,
        width as u64,
    ) else {
        return Ok(None);
    };
    AjtaiKeyParams::try_new(
        Cfg::sis_modulus_family(),
        rank,
        width,
        ceil_inf_norm,
        Cfg::D,
    )
    .map(Some)
}

pub(crate) fn compute_ajtai_key_params_b<Cfg: CommitmentConfig>(
    matrix_a_rank: usize,
    num_blocks: usize,
    t_vectors: usize,
    log_basis: u32,
) -> Result<Option<AjtaiKeyParams>, AkitaError> {
    let inf_norm = WitnessType::T.binding_norm::<Cfg>(log_basis, true)?;
    let num_digits = WitnessType::T.decomposed_num_digits::<Cfg>(log_basis, true);
    let Some(width) = matrix_a_rank
        .checked_mul(num_digits)
        .and_then(|w| w.checked_mul(num_blocks))
        .and_then(|w| w.checked_mul(t_vectors))
    else {
        return Ok(None);
    };
    let Some(ceil_inf_norm) =
        ceil_supported_collision(Cfg::sis_modulus_family(), Cfg::D as u32, inf_norm)
    else {
        return Ok(None);
    };
    let Some(rank) = min_rank_for_secure_width(
        Cfg::sis_modulus_family(),
        Cfg::D as u32,
        ceil_inf_norm,
        width as u64,
    ) else {
        return Ok(None);
    };
    AjtaiKeyParams::try_new(
        Cfg::sis_modulus_family(),
        rank,
        width,
        ceil_inf_norm,
        Cfg::D,
    )
    .map(Some)
}

/// The A, B, and D Ajtai keys for one fold level.
pub(crate) type AjtaiKeysParams = (AjtaiKeyParams, AjtaiKeyParams, AjtaiKeyParams);

/// Compute all three Ajtai keys (A, B, D) for one fold level in one shot.
pub(crate) fn compute_all_ajtai_keys_params<Cfg: CommitmentConfig>(
    block_len: usize,
    num_blocks: usize,
    t_vectors: usize,
    log_basis: u32,
    is_root_level: bool,
) -> Result<Option<AjtaiKeysParams>, AkitaError> {
    let Some(a_key) = compute_ajtai_key_params_a::<Cfg>(block_len, log_basis, is_root_level)?
    else {
        return Ok(None);
    };
    let Some(b_key) =
        compute_ajtai_key_params_b::<Cfg>(a_key.row_len(), num_blocks, t_vectors, log_basis)?
    else {
        return Ok(None);
    };
    let Some(d_key) = compute_ajtai_key_params_d::<Cfg>(num_blocks, t_vectors, log_basis)? else {
        return Ok(None);
    };
    Ok(Some((a_key, b_key, d_key)))
}

pub(crate) fn compute_ajtai_key_params_d<Cfg: CommitmentConfig>(
    num_blocks: usize,
    t_vectors: usize,
    log_basis: u32,
) -> Result<Option<AjtaiKeyParams>, AkitaError> {
    let inf_norm = WitnessType::W.binding_norm::<Cfg>(log_basis, true)?;
    let num_digits_open = WitnessType::W.decomposed_num_digits::<Cfg>(log_basis, true);
    let Some(width) = num_digits_open
        .checked_mul(num_blocks)
        .and_then(|w| w.checked_mul(t_vectors))
    else {
        return Ok(None);
    };
    let Some(ceil_inf_norm) =
        ceil_supported_collision(Cfg::sis_modulus_family(), Cfg::D as u32, inf_norm)
    else {
        return Ok(None);
    };
    let Some(rank) = min_rank_for_secure_width(
        Cfg::sis_modulus_family(),
        Cfg::D as u32,
        ceil_inf_norm,
        width as u64,
    ) else {
        return Ok(None);
    };
    AjtaiKeyParams::try_new(
        Cfg::sis_modulus_family(),
        rank,
        width,
        ceil_inf_norm,
        Cfg::D,
    )
    .map(Some)
}
