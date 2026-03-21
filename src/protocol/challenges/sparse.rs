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

const XOF_BUF_SIZE: usize = 4096;

/// Streaming cursor backed by a SHAKE256 XOF with a 4 KB internal buffer
/// (~30 rate blocks) to amortize squeeze calls.
struct XofCursor {
    reader: ShakeReader,
    buf: Box<[u8; XOF_BUF_SIZE]>,
    pos: usize,
}

impl XofCursor {
    fn from_seed(seed: &[u8]) -> Self {
        let mut xof = Shake256::default();
        xof.update(SPARSE_PRG_DOMAIN);
        xof.update(seed);
        let mut cursor = Self {
            reader: xof.finalize_xof(),
            buf: Box::new([0u8; XOF_BUF_SIZE]),
            pos: XOF_BUF_SIZE,
        };
        cursor.refill();
        cursor
    }

    #[inline]
    fn refill(&mut self) {
        self.reader.read(self.buf.as_mut());
        self.pos = 0;
    }

    #[inline]
    fn next_u8(&mut self) -> u8 {
        if self.pos >= XOF_BUF_SIZE {
            self.refill();
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        b
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.pos + 4 <= XOF_BUF_SIZE {
            let val = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
            self.pos += 4;
            val
        } else {
            let mut tmp = [0u8; 4];
            for b in &mut tmp {
                *b = self.next_u8();
            }
            u32::from_le_bytes(tmp)
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        if self.pos + 8 <= XOF_BUF_SIZE {
            let val = u64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
            self.pos += 8;
            val
        } else {
            let mut tmp = [0u8; 8];
            for b in &mut tmp {
                *b = self.next_u8();
            }
            u64::from_le_bytes(tmp)
        }
    }

    /// Uniformly sample from `0..modulus` using bitmask rejection sampling
    /// with minimal XOF consumption. Uses 1-byte reads when the modulus
    /// fits in 8 bits, 2-byte reads for 16 bits, else 4 bytes.
    #[inline]
    fn next_usize_mod(&mut self, modulus: usize) -> usize {
        debug_assert!(modulus > 0);
        if modulus == 1 {
            return 0;
        }
        let bits = usize::BITS - (modulus - 1).leading_zeros();
        if bits <= 8 {
            let mask = ((1u16 << bits) - 1) as u8;
            loop {
                let val = (self.next_u8() & mask) as usize;
                if val < modulus {
                    return val;
                }
            }
        } else if bits <= 16 {
            let mask = (1usize << bits) - 1;
            loop {
                let lo = self.next_u8() as usize;
                let hi = self.next_u8() as usize;
                let val = (lo | (hi << 8)) & mask;
                if val < modulus {
                    return val;
                }
            }
        } else {
            let mask: usize = (1 << bits) - 1;
            loop {
                let val = (self.next_u32() as usize) & mask;
                if val < modulus {
                    return val;
                }
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

    /// Batch-draw random signs into a pre-allocated slice, packing 8 signs
    /// per XOF byte to minimize consumption.
    #[inline]
    fn fill_signs(&mut self, out: &mut [i16]) {
        let mut chunks = out.chunks_exact_mut(8);
        for chunk in &mut chunks {
            let byte = self.next_u8();
            for (i, s) in chunk.iter_mut().enumerate() {
                *s = if (byte >> i) & 1 == 0 { 1 } else { -1 };
            }
        }
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let byte = self.next_u8();
            for (i, s) in remainder.iter_mut().enumerate() {
                *s = if (byte >> i) & 1 == 0 { 1 } else { -1 };
            }
        }
    }
}

/// Fisher-Yates partial shuffle: sample `count` distinct values from `0..universe`.
///
/// Uses a stack buffer (universe ≤ 128, the max ring degree) to avoid
/// per-call heap allocation.
#[inline]
fn sample_distinct_positions(cursor: &mut XofCursor, universe: usize, count: usize) -> Vec<usize> {
    debug_assert!(count <= universe);
    debug_assert!(universe <= 128, "universe must fit stack buffer");
    let mut perm = [0usize; 128];
    for (i, slot) in perm[..universe].iter_mut().enumerate() {
        *slot = i;
    }
    for i in 0..count {
        let j = i + cursor.next_usize_mod(universe - i);
        perm.swap(i, j);
    }
    perm[..count].to_vec()
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

/// Sample one half of a split-ring challenge, writing positions and coeffs
/// into the provided output slices (must be `half_weight` long).
#[inline]
fn sample_split_half_into(
    cursor: &mut XofCursor,
    half_size: usize,
    half_weight: usize,
    max_mag2_per_half: usize,
    parity: usize,
    out_positions: &mut [u32],
    out_coeffs: &mut [i16],
) {
    debug_assert!(half_size <= 64);
    debug_assert!(half_weight <= 64);
    let mut perm = [0usize; 64];
    for (i, slot) in perm[..half_size].iter_mut().enumerate() {
        *slot = i;
    }
    for i in 0..half_weight {
        let j = i + cursor.next_usize_mod(half_size - i);
        perm.swap(i, j);
    }
    for (p, &perm_val) in out_positions.iter_mut().zip(perm.iter()) {
        *p = (2 * perm_val + parity) as u32;
    }
    cursor.fill_signs(out_coeffs);

    let num_mag2 = sample_split_shell_count(cursor, half_weight, max_mag2_per_half);
    if num_mag2 > 0 {
        let mut mag2_perm = [0usize; 64];
        for (i, slot) in mag2_perm[..half_weight].iter_mut().enumerate() {
            *slot = i;
        }
        for i in 0..num_mag2 {
            let j = i + cursor.next_usize_mod(half_weight - i);
            mag2_perm.swap(i, j);
        }
        for &idx in &mag2_perm[..num_mag2] {
            out_coeffs[idx] *= 2;
        }
    }
}

fn sample_split_ring_sparse(
    cursor: &mut XofCursor,
    d: usize,
    half_weight: usize,
    max_mag2_per_half: usize,
) -> SparseChallenge {
    let half_size = d / 2;
    let total_weight = 2 * half_weight;
    let mut positions = vec![0u32; total_weight];
    let mut coeffs = vec![0i16; total_weight];
    sample_split_half_into(
        cursor,
        half_size,
        half_weight,
        max_mag2_per_half,
        0,
        &mut positions[..half_weight],
        &mut coeffs[..half_weight],
    );
    sample_split_half_into(
        cursor,
        half_size,
        half_weight,
        max_mag2_per_half,
        1,
        &mut positions[half_weight..],
        &mut coeffs[half_weight..],
    );
    SparseChallenge { positions, coeffs }
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
