//! Sparse ring fold challenge sampling via Fiat-Shamir with PRG expansion.
//!
//! After the prover's folded witness message `v` is absorbed, the protocol
//! samples sparse ring elements `c` used to fold the witness toward the next
//! commitment. Every [`SparseChallengeConfig`] uses the signed-sparse sampler:
//! `count_pm1` coefficients at ±1 and `count_pm2` at ±2.

mod position_sample;
mod signed_sparse;
mod xof;

pub(crate) use position_sample::MAX_STACK_RING_DIM;
pub(crate) use signed_sparse::SignedSparseScratch;
pub(crate) use xof::XofCursor;

use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::Transcript;

use crate::{SparseChallenge, SparseChallengeConfig};

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
    if ring_d > MAX_STACK_RING_DIM {
        return Err(AkitaError::InvalidInput(format!(
            "ring dimension {ring_d} exceeds supported stack sampler limit ({MAX_STACK_RING_DIM})"
        )));
    }
    cfg.validate_dyn(ring_d)
        .map_err(|e| AkitaError::InvalidInput(format!("invalid sparse challenge config: {e}")))?;

    let domain_sep = cfg.domain_separator_bytes();
    let mut absorb_buf = Vec::with_capacity(label.len() + 8 + 8 + domain_sep.len() + 4);
    absorb_buf.extend_from_slice(label);
    absorb_buf.extend_from_slice(&(n as u64).to_le_bytes());
    absorb_buf.extend_from_slice(&(ring_d as u64).to_le_bytes());
    absorb_buf.extend_from_slice(&domain_sep);
    absorb_buf.extend_from_slice(&grind_nonce.to_le_bytes());

    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, &absorb_buf);
    let seed = transcript.challenge_bytes(CHALLENGE_SPARSE_CHALLENGE, 32);
    let mut cursor = XofCursor::from_seed(&seed);
    Ok(SignedSparseScratch::sample_challenges(
        &mut cursor,
        ring_d,
        n,
        cfg,
    ))
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
        let unified = {
            let mut cursor = XofCursor::from_seed(&seed);
            SignedSparseScratch::sample_challenges(&mut cursor, ring_d, 1, &cfg)
                .pop()
                .expect("one challenge")
        };
        assert_eq!(legacy.positions, unified.positions);
        assert_eq!(legacy.coeffs, unified.coeffs);
    }
}
