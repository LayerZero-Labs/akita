//! Sparse ring fold challenge sampling via Fiat-Shamir with PRG expansion.
//!
//! After the prover's folded witness message `v` is absorbed, the protocol
//! samples sparse ring elements `c` used to fold the witness toward the next
//! commitment. Every [`SparseChallengeConfig`] uses the signed-sparse sampler:
//! `count_pm1` coefficients at ±1 and `count_pm2` at ±2.

mod position_sample;
mod signed_sparse;
mod xof;

pub(crate) use xof::XofCursor;

use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::Transcript;

use crate::{SparseChallenge, SparseChallengeConfig};

use position_sample::MAX_STACK_RING_DIM;
use signed_sparse::SignedSparseScratch;

/// Expand sparse challenges from an already-derived XOF cursor.
pub(crate) fn sparse_challenges_from_xof_cursor(
    cursor: &mut XofCursor,
    ring_d: usize,
    n: usize,
    cfg: &SparseChallengeConfig,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    let mut challenges = Vec::with_capacity(n);
    let mut scratch = SignedSparseScratch::new(cfg.count_pm1, cfg.count_pm2);
    for _ in 0..n {
        scratch.sample(cursor, ring_d, cfg.count_pm1, cfg.count_pm2);
        challenges.push(scratch.take_challenge());
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
    ring_d: usize,
    seed: &[u8],
    n: usize,
    cfg: &SparseChallengeConfig,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    validate_sparse_challenge_draw(ring_d, cfg)?;
    let mut cursor = XofCursor::from_seed(seed);
    sparse_challenges_from_xof_cursor(&mut cursor, ring_d, n, cfg)
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

/// Sample `n` sparse ring fold challenges from a transcript.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::xof::XofCursor;

    #[test]
    fn pm1_only_matches_pm2_zero_sampler() {
        let ring_d = 128;
        let cfg = SparseChallengeConfig::pm1_only(31);
        let seed = [7u8; 32];
        let legacy = {
            let mut cursor = XofCursor::from_seed(&seed);
            let mut scratch = SignedSparseScratch::new(31, 0);
            scratch.sample(&mut cursor, ring_d, 31, 0);
            scratch.take_challenge()
        };
        let unified = sparse_challenges_from_seed(ring_d, &seed, 1, &cfg)
            .expect("sample")
            .pop()
            .expect("one challenge");
        assert_eq!(legacy.positions, unified.positions);
        assert_eq!(legacy.coeffs, unified.coeffs);
    }
}
