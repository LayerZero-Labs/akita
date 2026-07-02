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
//! - [`SparseChallengeConfig::Uniform`] → [`uniform::sample_uniform_challenge`]
//! - [`SparseChallengeConfig::ExactShell`] → [`exact_shell::sample_exact_shell_challenge`]
//! - [`SparseChallengeConfig::BoundedL1Norm`] → [`bounded_l1::sample_bounded_l1_challenge`]

pub(crate) mod bounded_l1;
#[cfg(test)]
mod bounded_l1_support;
mod exact_shell;
mod uniform;
mod xof;

pub(crate) use xof::XofCursor;

use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::Transcript;

use crate::{SparseChallenge, SparseChallengeConfig};

use bounded_l1::{sample_bounded_l1_challenge, D_32};
use exact_shell::{sample_exact_shell_challenge, ExactShellScratch};
use uniform::{sample_uniform_challenge, MAX_STACK_RING_DIM};

/// Expand sparse challenges from an already-derived XOF cursor.
pub(crate) fn sparse_challenges_from_xof_cursor(
    cursor: &mut XofCursor,
    ring_d: usize,
    n: usize,
    cfg: &SparseChallengeConfig,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    let mut challenges = Vec::with_capacity(n);
    if let SparseChallengeConfig::ExactShell {
        count_mag1,
        count_mag2,
    } = cfg
    {
        let mut scratch = ExactShellScratch::new(*count_mag1, *count_mag2);
        for _ in 0..n {
            scratch.sample(cursor, ring_d, *count_mag1, *count_mag2);
            challenges.push(scratch.take_challenge());
        }
    } else {
        for _ in 0..n {
            challenges.push(parse_challenge(cursor, ring_d, cfg));
        }
    }
    Ok(challenges)
}

/// Reject sparse draws that exceed stack-sampler limits or fail config validation.
pub(crate) fn validate_sparse_challenge_draw(
    ring_d: usize,
    cfg: &SparseChallengeConfig,
) -> Result<(), AkitaError> {
    if ring_d > MAX_STACK_RING_DIM {
        return Err(AkitaError::InvalidInput(format!(
            "ring dimension {ring_d} exceeds supported stack sampler limit ({MAX_STACK_RING_DIM})"
        )));
    }
    cfg.validate_dyn(ring_d)
        .map_err(|e| AkitaError::InvalidInput(format!("invalid sparse challenge config: {e}")))
}

/// Expand sparse challenges from a fixed 32-byte PRG seed (fold-grind preview path).
pub fn sparse_challenges_from_seed(
    seed: &[u8],
    ring_d: usize,
    n: usize,
    cfg: &SparseChallengeConfig,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    let mut cursor = XofCursor::from_seed(seed);
    sparse_challenges_from_xof_cursor(&mut cursor, ring_d, n, cfg)
}

/// Parse a single sparse challenge from a streaming XOF cursor.
fn parse_challenge(
    cursor: &mut XofCursor,
    ring_d: usize,
    cfg: &SparseChallengeConfig,
) -> SparseChallenge {
    match cfg {
        SparseChallengeConfig::Uniform {
            weight,
            nonzero_coeffs,
        } => sample_uniform_challenge(cursor, ring_d, *weight, nonzero_coeffs),
        SparseChallengeConfig::ExactShell {
            count_mag1,
            count_mag2,
        } => sample_exact_shell_challenge(cursor, ring_d, *count_mag1, *count_mag2),
        SparseChallengeConfig::BoundedL1Norm => {
            debug_assert_eq!(ring_d, D_32);
            sample_bounded_l1_challenge(cursor)
        }
    }
}

/// Build the absorb buffer for one sparse-challenge Fiat–Shamir draw.
pub fn sparse_challenge_absorb_buf(
    label: &[u8],
    instance_tag: u64,
    ring_d: usize,
    cfg: &SparseChallengeConfig,
    grind_nonce: u32,
) -> Vec<u8> {
    let domain_sep = cfg.domain_separator_bytes();
    let mut absorb_buf = Vec::with_capacity(label.len() + 8 + 8 + domain_sep.len() + 4);
    absorb_buf.extend_from_slice(label);
    absorb_buf.extend_from_slice(&instance_tag.to_le_bytes());
    // Byte-critical: same little-endian u64 encoding of the ring dimension as
    // the former `(D as u64)`; identical bytes for equal values.
    absorb_buf.extend_from_slice(&(ring_d as u64).to_le_bytes());
    absorb_buf.extend_from_slice(&domain_sep);
    absorb_buf.extend_from_slice(&grind_nonce.to_le_bytes());
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
pub fn sample_sparse_challenges<F, T>(
    transcript: &mut T,
    label: &[u8],
    ring_d: usize,
    n: usize,
    cfg: &SparseChallengeConfig,
    grind_nonce: u32,
) -> Result<Vec<SparseChallenge>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    validate_sparse_challenge_draw(ring_d, cfg)?;

    let absorb_buf = sparse_challenge_absorb_buf(label, n as u64, ring_d, cfg, grind_nonce);
    let mut cursor = derive_xof_cursor::<F, T>(transcript, &absorb_buf);
    sparse_challenges_from_xof_cursor(&mut cursor, ring_d, n, cfg)
}
