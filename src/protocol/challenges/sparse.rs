//! Sparse challenge sampling via Fiat–Shamir.
//!
//! Performance: randomness is extracted via a single `challenge_bytes` call per
//! challenge (~5 Blake2b512 operations for the SplitRing D=64 config) rather
//! than ~118 individual `challenge_scalar` calls. Position sampling uses
//! Fisher-Yates partial shuffle, eliminating rejection-sampling waste.

use crate::algebra::ring::{CyclotomicRing, SparseChallenge, SparseChallengeConfig};
use crate::error::HachiError;
use crate::protocol::transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};

/// Cursor over a byte buffer for extracting random values.
struct ByteCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    #[inline]
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        let bytes: [u8; 4] = self.buf[self.pos..self.pos + 4].try_into().unwrap();
        self.pos += 4;
        u32::from_le_bytes(bytes)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let bytes: [u8; 8] = self.buf[self.pos..self.pos + 8].try_into().unwrap();
        self.pos += 8;
        u64::from_le_bytes(bytes)
    }

    #[inline]
    fn next_u8(&mut self) -> u8 {
        let b = self.buf[self.pos];
        self.pos += 1;
        b
    }

    #[inline]
    fn next_usize_mod(&mut self, modulus: usize) -> usize {
        (self.next_u32() as usize) % modulus
    }

    #[inline]
    fn next_sign(&mut self) -> i16 {
        if (self.next_u8() & 1) == 0 {
            1
        } else {
            -1
        }
    }
}

/// Upper bound on random bytes consumed when sampling one challenge.
fn max_challenge_bytes(cfg: &SparseChallengeConfig) -> usize {
    match cfg {
        SparseChallengeConfig::Uniform { weight, .. } => {
            // Fisher-Yates positions: weight × 4 bytes
            // Coefficient indices:    weight × 4 bytes
            weight * 8
        }
        SparseChallengeConfig::SplitRing {
            half_weight,
            max_mag2_per_half,
        } => {
            // Per half:
            //   Fisher-Yates positions over D/2:   half_weight × 4 bytes
            //   Signs:                             half_weight × 1 byte
            //   shell draw over j = 0..max:        8 bytes
            //   Fisher-Yates mag2 positions:       max_mag2_per_half × 4 bytes
            2 * (half_weight * 4 + half_weight + 8 + max_mag2_per_half * 4)
        }
        SparseChallengeConfig::ExactShell {
            count_mag1,
            count_mag2,
        } => {
            let total = count_mag1 + count_mag2;
            // Fisher-Yates positions: total × 4 bytes
            // Signs:                  total × 1 byte
            total * 5
        }
    }
}

/// Fisher-Yates partial shuffle: sample `count` distinct values from `0..universe`.
fn sample_distinct_positions(cursor: &mut ByteCursor, universe: usize, count: usize) -> Vec<usize> {
    debug_assert!(count <= universe);
    let mut perm: Vec<usize> = (0..universe).collect();
    for i in 0..count {
        let j = i + cursor.next_usize_mod(universe - i);
        perm.swap(i, j);
    }
    perm.truncate(count);
    perm
}

fn sample_uniform_sparse(
    cursor: &mut ByteCursor,
    d: usize,
    weight: usize,
    nonzero_coeffs: &[i16],
) -> SparseChallenge {
    let positions = sample_distinct_positions(cursor, d, weight)
        .into_iter()
        .map(|pos| pos as u32)
        .collect();
    let coeffs = (0..weight)
        .map(|_| {
            let coeff_idx = cursor.next_usize_mod(nonzero_coeffs.len());
            nonzero_coeffs[coeff_idx]
        })
        .collect();
    SparseChallenge { positions, coeffs }
}

#[inline]
fn binomial_u64(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut numer = 1u128;
    let mut denom = 1u128;
    for i in 0..k {
        numer *= (n - i) as u128;
        denom *= (i + 1) as u128;
    }
    (numer / denom)
        .try_into()
        .expect("split-ring shell size must fit into u64")
}

fn sample_split_shell_count(
    cursor: &mut ByteCursor,
    half_weight: usize,
    max_mag2_per_half: usize,
) -> usize {
    if max_mag2_per_half == 0 {
        return 0;
    }
    let total_shells: u64 = (0..=max_mag2_per_half)
        .map(|j| binomial_u64(half_weight, j))
        .sum();
    let mut draw = cursor.next_u64() % total_shells;
    for j in 0..=max_mag2_per_half {
        let shell_size = binomial_u64(half_weight, j);
        if draw < shell_size {
            return j;
        }
        draw -= shell_size;
    }
    unreachable!("split-ring shell sampler exhausted cumulative mass")
}

fn sample_split_half(
    cursor: &mut ByteCursor,
    half_size: usize,
    half_weight: usize,
    max_mag2_per_half: usize,
    parity: usize,
) -> (Vec<u32>, Vec<i16>) {
    let half_positions = sample_distinct_positions(cursor, half_size, half_weight);
    let mut coeffs: Vec<i16> = (0..half_weight).map(|_| cursor.next_sign()).collect();
    let num_mag2 = sample_split_shell_count(cursor, half_weight, max_mag2_per_half);
    for idx in sample_distinct_positions(cursor, half_weight, num_mag2) {
        coeffs[idx] *= 2;
    }

    let positions = half_positions
        .into_iter()
        .map(|pos| (2 * pos + parity) as u32)
        .collect();
    (positions, coeffs)
}

fn sample_split_ring_sparse(
    cursor: &mut ByteCursor,
    d: usize,
    half_weight: usize,
    max_mag2_per_half: usize,
) -> SparseChallenge {
    let half_size = d / 2;
    let (mut even_positions, mut even_coeffs) =
        sample_split_half(cursor, half_size, half_weight, max_mag2_per_half, 0);
    let (odd_positions, odd_coeffs) =
        sample_split_half(cursor, half_size, half_weight, max_mag2_per_half, 1);
    even_positions.extend(odd_positions);
    even_coeffs.extend(odd_coeffs);
    SparseChallenge {
        positions: even_positions,
        coeffs: even_coeffs,
    }
}

fn sample_exact_shell_sparse(
    cursor: &mut ByteCursor,
    d: usize,
    count_mag1: usize,
    count_mag2: usize,
) -> SparseChallenge {
    let total = count_mag1 + count_mag2;
    let positions = sample_distinct_positions(cursor, d, total)
        .into_iter()
        .map(|pos| pos as u32)
        .collect();
    let mut coeffs = Vec::with_capacity(total);
    for _ in 0..count_mag1 {
        coeffs.push(cursor.next_sign());
    }
    for _ in 0..count_mag2 {
        coeffs.push(2 * cursor.next_sign());
    }
    SparseChallenge { positions, coeffs }
}

/// Core sampling: absorbs domain separation, draws all randomness as a single
/// `challenge_bytes` call, and parses positions/signs from the byte buffer.
fn sample_one<F, T, const D: usize>(
    transcript: &mut T,
    label: &[u8],
    instance_idx: u64,
    cfg: &SparseChallengeConfig,
    domain_sep: &[u8],
) -> SparseChallenge
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, label);
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, &instance_idx.to_le_bytes());
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, &(D as u64).to_le_bytes());
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, domain_sep);

    let bytes = transcript.challenge_bytes(CHALLENGE_SPARSE_CHALLENGE, max_challenge_bytes(cfg));
    let mut cursor = ByteCursor::new(&bytes);

    match cfg {
        SparseChallengeConfig::Uniform {
            weight,
            nonzero_coeffs,
        } => sample_uniform_sparse(&mut cursor, D, *weight, nonzero_coeffs),
        SparseChallengeConfig::SplitRing {
            half_weight,
            max_mag2_per_half,
        } => sample_split_ring_sparse(&mut cursor, D, *half_weight, *max_mag2_per_half),
        SparseChallengeConfig::ExactShell {
            count_mag1,
            count_mag2,
        } => sample_exact_shell_sparse(&mut cursor, D, *count_mag1, *count_mag2),
    }
}

/// Sample a sparse ring challenge (exact weight ω) from a transcript.
///
/// This is intentionally deterministic and label-aware:
/// - first we absorb the sampling context under `ABSORB_SPARSE_CHALLENGE`,
/// - then we derive challenge bytes and extract positions/signs via
///   Fisher-Yates partial shuffle.
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

    let domain_sep = cfg.domain_separator_bytes();
    Ok(sample_one::<F, T, D>(
        transcript,
        label,
        instance_idx,
        cfg,
        &domain_sep,
    ))
}

/// Sample `n` sparse challenges from a transcript, returning the sparse
/// representation directly.
///
/// Validates the config once and pre-computes the domain separator, then
/// samples all `n` challenges sequentially.
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
    cfg.validate::<D>()
        .map_err(|e| HachiError::InvalidInput(format!("invalid sparse challenge config: {e}")))?;

    let domain_sep = cfg.domain_separator_bytes();
    Ok((0..n)
        .map(|i| sample_one::<F, T, D>(transcript, label, i as u64, cfg, &domain_sep))
        .collect())
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
    cfg.validate::<D>()
        .map_err(|e| HachiError::InvalidInput(format!("invalid sparse challenge config: {e}")))?;

    let domain_sep = cfg.domain_separator_bytes();
    (0..n)
        .map(|i| {
            let sparse = sample_one::<F, T, D>(transcript, label, i as u64, cfg, &domain_sep);
            sparse
                .to_dense::<F, D>()
                .map_err(|e| HachiError::InvalidInput(e.to_string()))
        })
        .collect()
}
