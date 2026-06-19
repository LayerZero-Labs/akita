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
#[cfg(all(test, not(feature = "zk")))]
mod bounded_l1_support;
mod exact_shell;
pub(crate) mod op_norm;
mod uniform;
mod xof;

pub(crate) use xof::XofCursor;

use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::Transcript;

use crate::{SparseChallenge, SparseChallengeConfig};

use bounded_l1::{sample_bounded_l1_challenge, D_32};
use exact_shell::sample_exact_shell_challenge;
use op_norm::OpNormTable;
use uniform::{sample_uniform_challenge, MAX_STACK_RING_DIM};

/// Fixed-point scale for the certified operator-norm predicate tables built
/// during rejection sampling. `q = 48` keeps the predicate's `i128`
/// accumulators within range for every shipping shell (`||c||_1 <= 2D`,
/// `T <= 2D`, `D <= MAX_STACK_RING_DIM`) while leaving the certified
/// uncertainty band negligible.
const OP_NORM_PREDICATE_SCALE: u32 = 48;

/// Liveness cap on operator-norm rejection attempts per challenge slot.
///
/// Prover and verifier replay the identical transcript-derived XOF stream and
/// the identical certified predicate, so this bound is reached (or not)
/// identically on both sides: rejection sampling can only fail closed, never
/// diverge. At the shipping acceptance probabilities (`p >= ~0.5`) exceeding
/// even a few dozen attempts is astronomically unlikely; the cap exists only to
/// keep sampling a terminating, no-panic operation under a pathological threshold.
const MAX_OP_NORM_ATTEMPTS: usize = 4096;

/// Expand sparse challenges from an already-derived XOF cursor.
pub(crate) fn sparse_challenges_from_xof_cursor<const D: usize>(
    cursor: &mut XofCursor,
    n: usize,
    cfg: &SparseChallengeConfig,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    let mut challenges = Vec::with_capacity(n);
    match op_norm_rejection_oracle::<D>(cfg)? {
        Some((table, t)) => {
            for _ in 0..n {
                challenges.push(sample_with_op_norm_rejection::<D>(cursor, cfg, &table, t)?);
            }
        }
        None => {
            for _ in 0..n {
                challenges.push(parse_challenge::<D>(cursor, cfg));
            }
        }
    }
    Ok(challenges)
}

/// Reject sparse draws that exceed stack-sampler limits or fail config validation.
pub(crate) fn validate_sparse_challenge_draw<const D: usize>(
    cfg: &SparseChallengeConfig,
) -> Result<(), AkitaError> {
    if D > MAX_STACK_RING_DIM {
        return Err(AkitaError::InvalidInput(format!(
            "ring dimension {D} exceeds supported stack sampler limit ({MAX_STACK_RING_DIM})"
        )));
    }
    cfg.validate::<D>()
        .map_err(|e| AkitaError::InvalidInput(format!("invalid sparse challenge config: {e}")))
}

/// Expand sparse challenges from a fixed 32-byte PRG seed (fold-grind preview path).
pub fn sparse_challenges_from_seed<const D: usize>(
    seed: &[u8],
    n: usize,
    cfg: &SparseChallengeConfig,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    let mut cursor = XofCursor::from_seed(seed);
    sparse_challenges_from_xof_cursor::<D>(&mut cursor, n, cfg)
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
        } => sample_uniform_challenge(cursor, D, *weight, nonzero_coeffs),
        SparseChallengeConfig::ExactShell {
            count_mag1,
            count_mag2,
            ..
        } => sample_exact_shell_challenge(cursor, D, *count_mag1, *count_mag2),
        SparseChallengeConfig::BoundedL1Norm => {
            debug_assert_eq!(D, D_32);
            sample_bounded_l1_challenge(cursor)
        }
    }
}

/// Build the absorb buffer for one sparse-challenge Fiat–Shamir draw.
pub fn sparse_challenge_absorb_buf<const D: usize>(
    label: &[u8],
    instance_tag: u64,
    cfg: &SparseChallengeConfig,
    grind_nonce: u32,
) -> Vec<u8> {
    let domain_sep = cfg.domain_separator_bytes();
    let mut absorb_buf = Vec::with_capacity(label.len() + 8 + 8 + domain_sep.len() + 4);
    absorb_buf.extend_from_slice(label);
    absorb_buf.extend_from_slice(&instance_tag.to_le_bytes());
    absorb_buf.extend_from_slice(&(D as u64).to_le_bytes());
    absorb_buf.extend_from_slice(&domain_sep);
    absorb_buf.extend_from_slice(&grind_nonce.to_le_bytes());
    absorb_buf
}

/// Build the operator-norm rejection oracle for `cfg`, when the family rejects.
///
/// Returns `Some((table, t))` for an [`SparseChallengeConfig::ExactShell`]
/// whose threshold is strictly below `||c||_1` (so rejection actually fires),
/// and `None` for every other family (and for a non-binding threshold
/// `T >= ||c||_1`, where `gamma_D(c) <= ||c||_1 <= T` always holds and the
/// predicate would accept every candidate). The certified table is built once
/// per [`sample_sparse_challenges`] call and shared across all `n` slots.
fn op_norm_rejection_oracle<const D: usize>(
    cfg: &SparseChallengeConfig,
) -> Result<Option<(OpNormTable, u64)>, AkitaError> {
    if !cfg.operator_norm_rejection_binds() {
        return Ok(None);
    }
    let SparseChallengeConfig::ExactShell {
        count_mag1,
        count_mag2,
        operator_norm_threshold,
    } = cfg
    else {
        return Ok(None);
    };
    let l1 = (count_mag1 + 2 * count_mag2) as u64;
    let t = u64::from(*operator_norm_threshold);
    let table = OpNormTable::new(D, OP_NORM_PREDICATE_SCALE, l1, t)?;
    Ok(Some((table, t)))
}

/// Draw candidates from `cursor` until one passes the certified operator-norm
/// predicate `gamma_D(c) <= t`, bounded by [`MAX_OP_NORM_ATTEMPTS`].
///
/// Each rejected candidate advances the shared XOF cursor identically for
/// prover and verifier, so the accepted challenge (and the cursor position the
/// next slot starts from) is a deterministic function of the transcript.
fn sample_with_op_norm_rejection<const D: usize>(
    cursor: &mut XofCursor,
    cfg: &SparseChallengeConfig,
    table: &OpNormTable,
    t: u64,
) -> Result<SparseChallenge, AkitaError> {
    for _ in 0..MAX_OP_NORM_ATTEMPTS {
        let candidate = parse_challenge::<D>(cursor, cfg);
        if table.accept_strict(&candidate, t)? {
            return Ok(candidate);
        }
    }
    Err(AkitaError::InvalidInput(format!(
        "operator-norm rejection sampling exceeded {MAX_OP_NORM_ATTEMPTS} attempts: \
         threshold T = {t} is too tight for the configured shell"
    )))
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
    grind_nonce: u32,
) -> Result<Vec<SparseChallenge>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    validate_sparse_challenge_draw::<D>(cfg)?;

    let absorb_buf = sparse_challenge_absorb_buf::<D>(label, n as u64, cfg, grind_nonce);
    let mut cursor = derive_xof_cursor::<F, T>(transcript, &absorb_buf);
    sparse_challenges_from_xof_cursor::<D>(&mut cursor, n, cfg)
}
