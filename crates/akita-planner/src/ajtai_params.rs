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
use akita_types::layout::digit_math::num_digits_for_bound;
use akita_types::AjtaiKeyParams;

/// Per-key decomposition / binding-norm rule.
enum AjtaiKeyType {
    /// A-role: Commit to decomposed witness s_i.
    A,
    /// B-role: Commit to decomposed t_i, where t_i = A s_i.
    B,
    /// D-role: Commit to decomposed w_i, where w_i = a s_i
    D,
}

impl AjtaiKeyType {
    /// Witness infinity norm to satisfy weak binding (Hachi paper, Lemma 7).
    fn binding_norm<Cfg: CommitmentConfig>(self, log_basis: u32) -> Result<u32, AkitaError> {
        match self {
            Self::A => {
                let beta = if Cfg::decomposition().log_commit_bound == 1 {
                    1
                } else {
                    (1u32 << (log_basis - 1)) - 1
                };
                Ok(2 * beta
                    * Cfg::stage1_challenge_config(Cfg::D)?.infinity_norm()
                    * Cfg::ring_subfield_embedding_norm_bound())
            }
            Self::B | Self::D => Ok((1u32 << log_basis) - 1),
        }
    }

    /// Number of `log_basis`-bit digits needed for one coefficient
    /// under this decomposition rule.
    fn decomposed_num_digits<Cfg: CommitmentConfig>(self, log_basis: u32) -> usize {
        let bound = match self {
            Self::A => Cfg::decomposition().log_commit_bound,
            Self::B | Self::D => Cfg::decomposition()
                .log_open_bound
                .unwrap_or(Cfg::decomposition().log_commit_bound),
        };
        num_digits_for_bound(bound, Cfg::decomposition().field_bits(), log_basis)
    }
}

pub(crate) fn compute_ajtai_key_params_a<Cfg: CommitmentConfig>(
    block_len: usize,
    log_basis: u32,
) -> Result<Option<AjtaiKeyParams>, AkitaError> {
    let inf_norm = AjtaiKeyType::A.binding_norm::<Cfg>(log_basis)?;
    let num_digits = AjtaiKeyType::A.decomposed_num_digits::<Cfg>(log_basis);
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
    let inf_norm = AjtaiKeyType::B.binding_norm::<Cfg>(log_basis)?;
    let num_digits = AjtaiKeyType::B.decomposed_num_digits::<Cfg>(log_basis);
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

pub(crate) fn compute_ajtai_key_params_d<Cfg: CommitmentConfig>(
    num_blocks: usize,
    t_vectors: usize,
    log_basis: u32,
) -> Result<Option<AjtaiKeyParams>, AkitaError> {
    let inf_norm = AjtaiKeyType::D.binding_norm::<Cfg>(log_basis)?;
    let num_digits_open = AjtaiKeyType::D.decomposed_num_digits::<Cfg>(log_basis);
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
