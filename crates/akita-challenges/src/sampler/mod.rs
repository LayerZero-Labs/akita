//! Sparse challenge sampling via Fiat-Shamir with PRG expansion.
//!
//! Challenges are derived by absorbing context into the transcript once,
//! drawing a 32-byte PRG seed, and expanding it via SHAKE256 XOF
//! ([`xof::XofCursor`]) into all per-challenge randomness. This replaces the
//! previous per-challenge hash chain with a single seed derivation followed
//! by fast XOF expansion, providing ~6x speedup for large batch sizes (e.g.
//! 4096 challenges).
//!
//! Position and shell sampling use bitmask rejection sampling to achieve
//! zero modulo bias, ensuring ≥128-bit security in the Fiat-Shamir challenge
//! distribution.
//!
//! The dispatcher in [`parse_challenge`] routes each [`SparseChallengeConfig`]
//! variant to its dedicated submodule:
//!
//! - [`SparseChallengeConfig::Uniform`] → [`uniform::sample_uniform_sparse`]
//! - [`SparseChallengeConfig::ExactShell`] → [`exact_shell::sample_exact_shell_sparse`]
//! - [`SparseChallengeConfig::BoundedL1Ball`] → [`bounded_l1::sample_bounded_l1_into`]

mod bounded_l1;
mod exact_shell;
mod uniform;
mod xof;

use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::Transcript;

use crate::{SparseChallenge, SparseChallengeConfig};

use bounded_l1::{sample_bounded_l1_into, PRESET_B, PRESET_D, PRESET_M};
use exact_shell::sample_exact_shell_sparse;
use uniform::{sample_uniform_sparse, MAX_STACK_RING_DIM};
use xof::XofCursor;

/// Validate that a `BoundedL1Ball` config matches the only supported preset
/// `(D=32, M=8, B=121)`. There is no runtime DP path: any other triple is
/// rejected before the dispatcher runs.
fn check_bounded_l1_preset<const D: usize>(
    max_abs_coeff: u8,
    l1_bound: u16,
) -> Result<(), AkitaError> {
    if D == PRESET_D && max_abs_coeff as usize == PRESET_M && l1_bound as usize == PRESET_B {
        Ok(())
    } else {
        Err(AkitaError::InvalidInput(format!(
            "BoundedL1Ball: only the preset (D={PRESET_D}, M={PRESET_M}, B={PRESET_B}) is supported, \
             got (D={D}, M={max_abs_coeff}, B={l1_bound})"
        )))
    }
}

/// Parse a single sparse challenge from a streaming XOF cursor.
fn parse_challenge<const D: usize>(
    cursor: &mut XofCursor,
    cfg: &SparseChallengeConfig,
) -> SparseChallenge {
    match cfg {
        SparseChallengeConfig::Uniform {
            weight,
            nonzero_coeffs,
        } => sample_uniform_sparse(cursor, D, *weight, nonzero_coeffs),
        SparseChallengeConfig::ExactShell {
            count_mag1,
            count_mag2,
        } => sample_exact_shell_sparse(cursor, D, *count_mag1, *count_mag2),
        SparseChallengeConfig::BoundedL1Ball { .. } => {
            // The output `SparseChallenge` owns its `Vec`s, so each call
            // ultimately needs its own allocation. Sizing both buffers to the
            // tight upper bound `D.min(B)` once lets `push` grow into the
            // reserved capacity without further reallocs.
            let cap = D.min(PRESET_B);
            let mut positions: Vec<u32> = Vec::with_capacity(cap);
            let mut coeffs: Vec<i16> = Vec::with_capacity(cap);
            sample_bounded_l1_into(cursor, &mut positions, &mut coeffs);
            SparseChallenge { positions, coeffs }
        }
    }
}

#[inline]
fn sparse_challenge_absorb_buf<const D: usize>(
    label: &[u8],
    instance_tag: u64,
    cfg: &SparseChallengeConfig,
) -> Vec<u8> {
    let domain_sep = cfg.domain_separator_bytes();
    let mut absorb_buf = Vec::with_capacity(label.len() + 8 + 8 + domain_sep.len());
    absorb_buf.extend_from_slice(label);
    absorb_buf.extend_from_slice(&instance_tag.to_le_bytes());
    absorb_buf.extend_from_slice(&(D as u64).to_le_bytes());
    absorb_buf.extend_from_slice(&domain_sep);
    absorb_buf
}

/// Absorb context into the transcript, derive a PRG seed, and create a
/// streaming XOF cursor for challenge randomness.
fn derive_xof_cursor<F, T>(transcript: &mut T, absorb_data: &[u8]) -> XofCursor
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, absorb_data);
    let seed = transcript.challenge_bytes(CHALLENGE_SPARSE_CHALLENGE, 32);
    XofCursor::from_seed(&seed)
}

/// Sample `n` sparse challenges from a transcript, returning the sparse
/// representation directly.
///
/// Absorbs the context (label, count, ring degree, config) into the
/// transcript once, derives a single 32-byte PRG seed, and expands it
/// via SHAKE256 XOF into all per-challenge randomness in one streaming
/// pass.
///
/// # Errors
///
/// Returns an error if challenge sampling fails.
#[tracing::instrument(skip_all, name = "sample_sparse_challenges")]
pub fn sample_sparse_challenges<F, T, const D: usize>(
    transcript: &mut T,
    label: &[u8],
    n: usize,
    cfg: &SparseChallengeConfig,
) -> Result<Vec<SparseChallenge>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    if D > MAX_STACK_RING_DIM {
        return Err(AkitaError::InvalidInput(format!(
            "ring dimension {D} exceeds sampling stack-buffer limit ({MAX_STACK_RING_DIM})"
        )));
    }
    cfg.validate::<D>()
        .map_err(|e| AkitaError::InvalidInput(format!("invalid sparse challenge config: {e}")))?;

    if let SparseChallengeConfig::BoundedL1Ball {
        max_abs_coeff,
        l1_bound,
    } = cfg
    {
        check_bounded_l1_preset::<D>(*max_abs_coeff, *l1_bound)?;
    }

    let absorb_buf = sparse_challenge_absorb_buf::<D>(label, n as u64, cfg);
    let mut cursor = derive_xof_cursor::<F, T>(transcript, &absorb_buf);
    let mut challenges = Vec::with_capacity(n);
    for _ in 0..n {
        challenges.push(parse_challenge::<D>(&mut cursor, cfg));
    }
    Ok(challenges)
}
