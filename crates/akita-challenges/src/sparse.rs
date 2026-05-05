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

use crate::{SparseChallenge, SparseChallengeConfig};
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::Transcript;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

use crate::bounded_l1::{
    build_ways_table, sample_bounded_l1_into, OwnedWaysTable, WaysTableRef, PRESET_D32_M8_B121_B,
    PRESET_D32_M8_B121_D, PRESET_D32_M8_B121_M, PRESET_D32_M8_B121_TABLE,
};

/// TODO(after this crate-decomposition PR): rename this byte domain to
/// `akita/...` in the dedicated transcript-domain cutover and refresh fixtures.
const SPARSE_PRG_DOMAIN: &[u8] = b"hachi/sparse-challenge-prg";

type ShakeReader = <Shake256 as ExtendableOutput>::Reader;

const XOF_BUF_SIZE: usize = 4096;

/// Streaming cursor backed by a SHAKE256 XOF with a 4 KB internal buffer
/// (~30 rate blocks) to amortize squeeze calls.
pub(crate) struct XofCursor {
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

    /// Read 16 little-endian bytes from the XOF and interpret them as an
    /// unsigned 128-bit integer.
    ///
    /// This is the canonical top-level draw for the truncated `2^128`
    /// bounded-`L1` sampler. There is no rejection loop and no modulo
    /// reduction: the realized distribution is uniform over `[0, 2^128)`,
    /// matching `read_u128_le` in `specs/bounded-l1-sparse-challenge.md`.
    #[inline]
    pub(crate) fn next_u128_le(&mut self) -> u128 {
        let mut bytes = [0u8; 16];
        for slot in bytes.iter_mut() {
            *slot = self.next_u8();
        }
        u128::from_le_bytes(bytes)
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

/// Max ring dimension supported by the stack-buffer sampling paths.
/// Public API functions reject `D` above this with an error before
/// reaching the sampling internals.
const MAX_STACK_RING_DIM: usize = 128;

/// Fisher-Yates partial shuffle: sample `out.len()` distinct values from
/// `0..universe` into `out`.
///
/// Uses a stack buffer (universe ≤ [`MAX_STACK_RING_DIM`]) to avoid
/// per-call heap allocation.
///
/// # Safety contract
///
/// Caller must ensure `universe <= MAX_STACK_RING_DIM`. The public API
/// enforces this via a fallible check that returns `Err` instead of
/// panicking.
#[inline]
fn sample_distinct_positions_into(cursor: &mut XofCursor, universe: usize, out: &mut [u32]) {
    debug_assert!(out.len() <= universe);
    debug_assert!(universe <= MAX_STACK_RING_DIM);
    let mut perm = [0usize; MAX_STACK_RING_DIM];
    for (i, slot) in perm[..universe].iter_mut().enumerate() {
        *slot = i;
    }
    for (i, dst) in out.iter_mut().enumerate() {
        let j = i + cursor.next_usize_mod(universe - i);
        perm.swap(i, j);
        *dst = perm[i] as u32;
    }
}

/// Heap-backed variant of [`sample_distinct_positions_into`] for ring
/// dimensions larger than [`MAX_STACK_RING_DIM`]. Not used on the
/// current hot path.
#[allow(dead_code)]
fn sample_distinct_positions_into_general(
    cursor: &mut XofCursor,
    universe: usize,
    out: &mut [u32],
) {
    debug_assert!(out.len() <= universe);
    let mut perm: Vec<usize> = (0..universe).collect();
    for (i, dst) in out.iter_mut().enumerate() {
        let j = i + cursor.next_usize_mod(universe - i);
        perm.swap(i, j);
        *dst = perm[i] as u32;
    }
}

fn sample_uniform_sparse(
    cursor: &mut XofCursor,
    d: usize,
    weight: usize,
    nonzero_coeffs: &[i16],
) -> SparseChallenge {
    let mut positions = vec![0u32; weight];
    sample_distinct_positions_into(cursor, d, &mut positions);
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
///
/// Uses stack buffers (half_size ≤ `MAX_STACK_RING_DIM / 2`,
/// half_weight ≤ `MAX_STACK_RING_DIM / 2`).
///
/// # Safety contract
///
/// Caller must ensure the size bounds. The public API enforces this
/// via a fallible `D <= MAX_STACK_RING_DIM` check.
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
    debug_assert!(half_size <= MAX_STACK_RING_DIM / 2);
    debug_assert!(half_weight <= MAX_STACK_RING_DIM / 2);
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
    let mut positions = vec![0u32; total];
    sample_distinct_positions_into(cursor, d, &mut positions);
    let mut coeffs = Vec::with_capacity(total);
    for _ in 0..count_mag1 {
        coeffs.push(cursor.next_sign());
    }
    for _ in 0..count_mag2 {
        coeffs.push(2 * cursor.next_sign());
    }
    SparseChallenge { positions, coeffs }
}

/// Per-batch precomputed state for the bounded-`L1` sampler.
///
/// For [`SparseChallengeConfig::BoundedL1Ball`] this holds the WAYS table
/// view: a borrowed `&'static` for the production `(D=32, M=8, B=121)` preset
/// (no table construction at all), or an owned `Vec`-backed table for any
/// other triple. For other variants the scratch carries no state.
struct SamplerScratch {
    /// Owned WAYS table for the runtime-built path. Held alongside
    /// `bounded_l1_view` so the view's borrow stays valid for the lifetime
    /// of `Self`.
    _bounded_l1_owned: Option<OwnedWaysTable>,
    /// `Some(view)` iff the active config is `BoundedL1Ball`. Borrows from
    /// either `_bounded_l1_owned` (runtime build) or the static
    /// [`PRESET_D32_M8_B121_TABLE`] (production preset).
    bounded_l1_view: Option<WaysTableRef<'static>>,
}

impl SamplerScratch {
    fn new<const D: usize>(cfg: &SparseChallengeConfig) -> Result<Self, AkitaError> {
        let (owned, view) = match cfg {
            SparseChallengeConfig::BoundedL1Ball {
                max_abs_coeff,
                l1_bound,
            } => {
                let m = *max_abs_coeff as usize;
                let b = *l1_bound as usize;
                if D == PRESET_D32_M8_B121_D
                    && m == PRESET_D32_M8_B121_M
                    && b == PRESET_D32_M8_B121_B
                {
                    // Production preset: skip table construction entirely;
                    // the static `PRESET_D32_M8_B121_TABLE` already lives in
                    // `.rodata` and the borrow is genuinely `'static`.
                    (None, Some(PRESET_D32_M8_B121_TABLE))
                } else {
                    let owned = build_ways_table(D, m, b)?;
                    // The truncated-`2^128` sampler requires
                    // `WAYS[D][B] >= 2^128` so that every top-level draw
                    // `r in [0, 2^128)` lands in some valid descent path. A
                    // `Wide` value is `>= 2^128` iff its high half is non-zero.
                    let total = owned.view().at(D, b);
                    if total.hi == 0 {
                        return Err(AkitaError::InvalidInput(format!(
                            "BoundedL1Ball: support |WAYS[{D}][{b}]| < 2^128 \
                             cannot drive the truncated-2^128 sampler; \
                             use a larger l1_bound or implement an exact-uniform fallback"
                        )));
                    }
                    // Safety: the owned table is stored in `Self` alongside
                    // the view; the borrow is alive for as long as `Self` is
                    // live. The `'static` is a uniform field shape with the
                    // const-preset case; we never expose this view past
                    // `Self`'s lifetime.
                    let view = unsafe {
                        std::mem::transmute::<WaysTableRef<'_>, WaysTableRef<'static>>(owned.view())
                    };
                    (Some(owned), Some(view))
                }
            }
            _ => (None, None),
        };
        Ok(Self {
            _bounded_l1_owned: owned,
            bounded_l1_view: view,
        })
    }
}

/// Parse a single sparse challenge from a streaming XOF cursor.
fn parse_challenge<const D: usize>(
    cursor: &mut XofCursor,
    cfg: &SparseChallengeConfig,
    scratch: &SamplerScratch,
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
        SparseChallengeConfig::BoundedL1Ball {
            max_abs_coeff,
            l1_bound,
        } => {
            let table = scratch
                .bounded_l1_view
                .expect("BoundedL1Ball requires a precomputed WAYS view in SamplerScratch");
            // The output `SparseChallenge` owns its `Vec`s, so each call
            // ultimately needs its own allocation. We still avoid the prior
            // 2-Vec-grow pattern (`Vec::with_capacity` inside the inner loop
            // followed by repeated `push`) by sizing both buffers to the
            // tight upper bound `D.min(B)` once and letting `push` grow into
            // the reserved capacity without further reallocs.
            let cap = D.min(*l1_bound as usize);
            let mut positions: Vec<u32> = Vec::with_capacity(cap);
            let mut coeffs: Vec<i16> = Vec::with_capacity(cap);
            sample_bounded_l1_into::<D>(
                cursor,
                table,
                *max_abs_coeff as usize,
                *l1_bound as usize,
                &mut positions,
                &mut coeffs,
            );
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
) -> Result<SparseChallenge, AkitaError>
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

    let scratch = SamplerScratch::new::<D>(cfg)?;
    let absorb_buf = sparse_challenge_absorb_buf::<D>(label, instance_idx, cfg);
    let mut cursor = derive_xof_cursor::<F, T>(transcript, &absorb_buf);
    Ok(parse_challenge::<D>(&mut cursor, cfg, &scratch))
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

    let scratch = SamplerScratch::new::<D>(cfg)?;
    let absorb_buf = sparse_challenge_absorb_buf::<D>(label, n as u64, cfg);
    let mut cursor = derive_xof_cursor::<F, T>(transcript, &absorb_buf);
    let mut challenges = Vec::with_capacity(n);
    for _ in 0..n {
        challenges.push(parse_challenge::<D>(&mut cursor, cfg, &scratch));
    }
    Ok(challenges)
}
