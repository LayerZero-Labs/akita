//! Sparse challenge sampling via Fiat–Shamir.

use crate::algebra::ring::{CyclotomicRing, SparseChallenge, SparseChallengeConfig};
use crate::error::HachiError;
use crate::protocol::transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};

#[inline]
fn next_challenge_u128<F, T>(transcript: &mut T) -> u128
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript
        .challenge_scalar(CHALLENGE_SPARSE_CHALLENGE)
        .to_canonical_u128()
}

fn sample_distinct_positions<F, T>(transcript: &mut T, universe: usize, count: usize) -> Vec<usize>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let mut seen = vec![false; universe];
    let mut positions = Vec::with_capacity(count);
    while positions.len() < count {
        let pos = (next_challenge_u128::<F, T>(transcript) as usize) % universe;
        if seen[pos] {
            continue;
        }
        seen[pos] = true;
        positions.push(pos);
    }
    positions
}

#[inline]
fn sample_sign<F, T>(transcript: &mut T) -> i16
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    if (next_challenge_u128::<F, T>(transcript) & 1) == 0 {
        1
    } else {
        -1
    }
}

fn sample_uniform_sparse<F, T, const D: usize>(
    transcript: &mut T,
    weight: usize,
    nonzero_coeffs: &[i16],
) -> SparseChallenge
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let positions = sample_distinct_positions::<F, T>(transcript, D, weight)
        .into_iter()
        .map(|pos| pos as u32)
        .collect();
    let coeffs = (0..weight)
        .map(|_| {
            let coeff_idx =
                (next_challenge_u128::<F, T>(transcript) as usize) % nonzero_coeffs.len();
            nonzero_coeffs[coeff_idx]
        })
        .collect();
    SparseChallenge { positions, coeffs }
}

fn sample_split_half<F, T>(
    transcript: &mut T,
    half_size: usize,
    half_weight: usize,
    max_mag2_per_half: usize,
    parity: usize,
) -> (Vec<u32>, Vec<i16>)
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let half_positions = sample_distinct_positions::<F, T>(transcript, half_size, half_weight);
    let mut coeffs: Vec<i16> = (0..half_weight)
        .map(|_| sample_sign::<F, T>(transcript))
        .collect();

    let num_mag2 = if max_mag2_per_half == 0 {
        0
    } else {
        (next_challenge_u128::<F, T>(transcript) as usize) % (max_mag2_per_half + 1)
    };
    for idx in sample_distinct_positions::<F, T>(transcript, half_weight, num_mag2) {
        coeffs[idx] *= 2;
    }

    let positions = half_positions
        .into_iter()
        .map(|pos| (2 * pos + parity) as u32)
        .collect();
    (positions, coeffs)
}

fn sample_split_ring_sparse<F, T, const D: usize>(
    transcript: &mut T,
    half_weight: usize,
    max_mag2_per_half: usize,
) -> SparseChallenge
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let half_size = D / 2;
    let (mut even_positions, mut even_coeffs) =
        sample_split_half::<F, T>(transcript, half_size, half_weight, max_mag2_per_half, 0);
    let (odd_positions, odd_coeffs) =
        sample_split_half::<F, T>(transcript, half_size, half_weight, max_mag2_per_half, 1);
    even_positions.extend(odd_positions);
    even_coeffs.extend(odd_coeffs);
    SparseChallenge {
        positions: even_positions,
        coeffs: even_coeffs,
    }
}

fn sample_exact_shell_sparse<F, T, const D: usize>(
    transcript: &mut T,
    count_mag1: usize,
    count_mag2: usize,
) -> SparseChallenge
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let total = count_mag1 + count_mag2;
    let positions = sample_distinct_positions::<F, T>(transcript, D, total)
        .into_iter()
        .map(|pos| pos as u32)
        .collect();
    let mut coeffs = Vec::with_capacity(total);
    for _ in 0..count_mag1 {
        coeffs.push(sample_sign::<F, T>(transcript));
    }
    for _ in 0..count_mag2 {
        coeffs.push(2 * sample_sign::<F, T>(transcript));
    }
    SparseChallenge { positions, coeffs }
}

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
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, &cfg.domain_separator_bytes());

    Ok(match cfg {
        SparseChallengeConfig::Uniform {
            weight,
            nonzero_coeffs,
        } => sample_uniform_sparse::<F, T, D>(transcript, *weight, nonzero_coeffs),
        SparseChallengeConfig::SplitRing {
            half_weight,
            max_mag2_per_half,
        } => sample_split_ring_sparse::<F, T, D>(transcript, *half_weight, *max_mag2_per_half),
        SparseChallengeConfig::ExactShell {
            count_mag1,
            count_mag2,
        } => sample_exact_shell_sparse::<F, T, D>(transcript, *count_mag1, *count_mag2),
    })
}

/// Sample `n` sparse challenges from a transcript, returning the sparse
/// representation directly.
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
) -> Result<Vec<SparseChallenge>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    (0..n)
        .map(|i| sparse_challenge_from_transcript::<F, T, D>(transcript, label, i as u64, cfg))
        .collect()
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
