//! Sparse challenge sampling via Fiat–Shamir.

use crate::algebra::ring::{CyclotomicRing, SparseChallenge, SparseChallengeConfig};
use crate::error::HachiError;
use crate::protocol::transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};

/// Sample a sparse ring challenge (exact weight ω) from a transcript.
///
/// This is intentionally deterministic and label-aware:
/// - first we absorb the sampling context under `ABSORB_SPARSE_CHALLENGE`,
/// - then we derive as many `CHALLENGE_SPARSE_CHALLENGE` scalars as needed.
///
/// Notes:
/// - Indices are sampled with a simple `mod D` reduction. For the intended
///   regimes (small `D`, cryptographic transcript), any bias is negligible.
/// - Duplicate indices are rejected to enforce exact Hamming weight.
///
/// # Errors
///
/// Returns an error if the provided config is invalid for degree `D`.
pub fn sparse_challenge_from_transcript<F, T, const D: usize>(
    transcript: &mut T,
    label: &[u8],
    instance_idx: u64,
    cfg: &SparseChallengeConfig,
) -> Result<SparseChallenge, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    cfg.validate::<D>()
        .map_err(|e| HachiError::InvalidInput(format!("invalid sparse challenge config: {e}")))?;

    // Absorb domain-separating context so different call sites can't collide.
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, label);
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, &instance_idx.to_le_bytes());
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, &(D as u64).to_le_bytes());
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, &(cfg.weight as u64).to_le_bytes());
    // Include the coefficient alphabet (as little-endian i16 stream).
    let mut coeff_bytes = Vec::with_capacity(cfg.nonzero_coeffs.len() * 2);
    for &c in cfg.nonzero_coeffs.iter() {
        coeff_bytes.extend_from_slice(&c.to_le_bytes());
    }
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, &coeff_bytes);

    let mut seen = vec![false; D];
    let mut positions = Vec::with_capacity(cfg.weight);
    let mut coeffs = Vec::with_capacity(cfg.weight);

    while positions.len() < cfg.weight {
        let r = transcript
            .challenge_scalar(CHALLENGE_SPARSE_CHALLENGE)
            .to_canonical_u128();
        let lo = r as u64;
        let hi = (r >> 64) as u64;

        let pos = (lo % (D as u64)) as usize;
        if seen[pos] {
            continue;
        }
        seen[pos] = true;
        positions.push(pos as u32);

        let coeff_idx = (hi % (cfg.nonzero_coeffs.len() as u64)) as usize;
        let c = cfg.nonzero_coeffs[coeff_idx];
        debug_assert_ne!(c, 0);
        coeffs.push(c);
    }

    Ok(SparseChallenge { positions, coeffs })
}

/// Sample `n` sparse challenges from a transcript and convert them to dense
/// `CyclotomicRing` elements.
///
/// # Errors
///
/// Returns an error if challenge sampling or dense conversion fails.
pub fn sample_dense_challenges<F, T, const D: usize>(
    transcript: &mut T,
    label: &[u8],
    n: usize,
    cfg: &SparseChallengeConfig,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    (0..n)
        .map(|i| {
            let sparse =
                sparse_challenge_from_transcript::<F, T, D>(transcript, label, i as u64, cfg)?;
            sparse
                .to_dense::<F, D>()
                .map_err(|e| HachiError::InvalidInput(e.to_string()))
        })
        .collect()
}
