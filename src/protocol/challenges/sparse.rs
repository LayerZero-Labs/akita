//! Sparse challenge sampling via Fiat–Shamir with PRG expansion.
//!
//! Challenges are derived by absorbing context into the transcript once,
//! drawing a 32-byte PRG seed, and expanding it via SHAKE256 XOF into all
//! per-challenge randomness. This replaces the previous per-challenge hash
//! chain with a single seed derivation followed by fast XOF expansion,
//! providing ~6x speedup for large batch sizes (e.g. 4096 challenges).
//!
//! Position and shell sampling use bitmask rejection sampling to achieve
//! zero modulo bias, ensuring ≥128-bit security in the Fiat–Shamir
//! challenge distribution.

use crate::algebra::ring::{CyclotomicRing, SparseChallenge, SparseChallengeConfig};
use crate::error::HachiError;
use crate::protocol::transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

const SPARSE_PRG_DOMAIN: &[u8] = b"hachi/sparse-challenge-prg";

type ShakeReader = <Shake256 as ExtendableOutput>::Reader;

/// Streaming cursor backed by a SHAKE256 XOF for extracting random values
/// with zero modulo bias via bitmask rejection sampling.
struct XofCursor {
    reader: ShakeReader,
}

impl XofCursor {
    fn from_seed(seed: &[u8]) -> Self {
        let mut xof = Shake256::default();
        xof.update(SPARSE_PRG_DOMAIN);
        xof.update(seed);
        Self {
            reader: xof.finalize_xof(),
        }
    }

    #[inline]
    fn fill(&mut self, buf: &mut [u8]) {
        self.reader.read(buf);
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.fill(&mut buf);
        u32::from_le_bytes(buf)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill(&mut buf);
        u64::from_le_bytes(buf)
    }

    #[inline]
    fn next_u8(&mut self) -> u8 {
        let mut buf = [0u8; 1];
        self.fill(&mut buf);
        buf[0]
    }

    /// Uniformly sample from `0..modulus` using bitmask rejection sampling.
    /// Zero modulo bias; expected < 2 draws per call.
    #[inline]
    fn next_usize_mod(&mut self, modulus: usize) -> usize {
        debug_assert!(modulus > 0);
        if modulus == 1 {
            return 0;
        }
        let bits = usize::BITS - (modulus - 1).leading_zeros();
        let mask: usize = (1 << bits) - 1;
        loop {
            let val = (self.next_u32() as usize) & mask;
            if val < modulus {
                return val;
            }
        }
    }

    /// Uniformly sample from `0..modulus` (u64) using bitmask rejection sampling.
    #[inline]
    fn next_u64_mod(&mut self, modulus: u64) -> u64 {
        debug_assert!(modulus > 0);
        if modulus == 1 {
            return 0;
        }
        let bits = u64::BITS - (modulus - 1).leading_zeros();
        let mask: u64 = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        loop {
            let val = self.next_u64() & mask;
            if val < modulus {
                return val;
            }
        }
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

/// Fisher-Yates partial shuffle: sample `count` distinct values from `0..universe`.
fn sample_distinct_positions(cursor: &mut XofCursor, universe: usize, count: usize) -> Vec<usize> {
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
    cursor: &mut XofCursor,
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
    cursor: &mut XofCursor,
    half_weight: usize,
    max_mag2_per_half: usize,
) -> usize {
    if max_mag2_per_half == 0 {
        return 0;
    }
    let total_shells: u64 = (0..=max_mag2_per_half)
        .map(|j| binomial_u64(half_weight, j))
        .sum();
    let mut draw = cursor.next_u64_mod(total_shells);
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
    cursor: &mut XofCursor,
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
    cursor: &mut XofCursor,
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
    cursor: &mut XofCursor,
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
        SparseChallengeConfig::SplitRing {
            half_weight,
            max_mag2_per_half,
        } => sample_split_ring_sparse(cursor, D, *half_weight, *max_mag2_per_half),
        SparseChallengeConfig::ExactShell {
            count_mag1,
            count_mag2,
        } => sample_exact_shell_sparse(cursor, D, *count_mag1, *count_mag2),
    }
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

/// Sample a sparse ring challenge (exact weight ω) from a transcript.
///
/// Absorbs the sampling context, derives a PRG seed, and expands it via
/// SHAKE256 XOF to produce the challenge randomness. Deterministic given
/// the same transcript state and parameters.
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
    let mut absorb_buf = Vec::with_capacity(label.len() + 8 + 8 + domain_sep.len());
    absorb_buf.extend_from_slice(label);
    absorb_buf.extend_from_slice(&instance_idx.to_le_bytes());
    absorb_buf.extend_from_slice(&(D as u64).to_le_bytes());
    absorb_buf.extend_from_slice(&domain_sep);

    let mut cursor = derive_xof_cursor::<F, T>(transcript, &absorb_buf);
    Ok(parse_challenge::<D>(&mut cursor, cfg))
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
) -> Result<Vec<SparseChallenge>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    cfg.validate::<D>()
        .map_err(|e| HachiError::InvalidInput(format!("invalid sparse challenge config: {e}")))?;

    let domain_sep = cfg.domain_separator_bytes();
    let mut absorb_buf = Vec::with_capacity(label.len() + 8 + 8 + domain_sep.len());
    absorb_buf.extend_from_slice(label);
    absorb_buf.extend_from_slice(&(n as u64).to_le_bytes());
    absorb_buf.extend_from_slice(&(D as u64).to_le_bytes());
    absorb_buf.extend_from_slice(&domain_sep);

    let mut cursor = derive_xof_cursor::<F, T>(transcript, &absorb_buf);
    let mut challenges = Vec::with_capacity(n);
    for _ in 0..n {
        challenges.push(parse_challenge::<D>(&mut cursor, cfg));
    }
    Ok(challenges)
}

/// Sample `n` sparse challenges from a transcript and convert them to dense
/// `CyclotomicRing` elements.
///
/// Uses the same seed-then-expand protocol as [`sample_sparse_challenges`].
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
    sample_sparse_challenges::<F, T, D>(transcript, label, n, cfg)?
        .into_iter()
        .map(|sparse| {
            sparse
                .to_dense::<F, D>()
                .map_err(|e| HachiError::InvalidInput(e.to_string()))
        })
        .collect()
}
