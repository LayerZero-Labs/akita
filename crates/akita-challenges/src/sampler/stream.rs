//! One transcript message per sparse fold group.
//!
//! A group's `count` challenges are read from one squeezed message of a fixed
//! byte length. The message is a little-endian bit stream: bit `i` is bit
//! `i % 8` of byte `i / 8`. Candidate `j` occupies bits `[j K, (j + 1) K)`.
//!
//! A candidate is a `K`-bit integer `R`. Its low `w = count_pm1 + count_pm2`
//! bits are the signs of the nonzero coefficients in output order (0 is `+`).
//! The remaining `K - w` bits are a position rank `r = r_a + C(d, a) r_b`:
//! `r_a` unranks to the ±1 positions and `r_b` to the ±2 positions among the
//! `d - a` remaining slots, both in colexicographic order. On `[0, |S|)` this
//! is a bijection onto the shell `S` of all such vectors.
//!
//! An unfiltered family uses `K = floor(log2 |S|)`, so every candidate is
//! accepted and the challenges are exactly uniform on the first `2^K` shell
//! elements. An operator-norm family uses `K = ceil(log2 |S|)` over the whole
//! shell: a candidate is accepted when its rank is in range and the certified
//! predicate accepts it, and the first `count` accepted candidates are the
//! challenges. Running out is an error, so the verifier rejects. Whether a
//! candidate is accepted does not depend on the accepted values, so the
//! challenges of a message that does not run out are exactly uniform on the
//! accepted set. The pool size only bounds how often an honest message runs
//! out: at most `2^-161` at the measured predicate rate, and at most `2^-32`
//! at the certified lower bound.

use super::ChallengePolicy;
use crate::{OperatorNormRejection, SparseChallenge, SparseChallengeConfig};
use akita_error::AkitaError;
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use std::sync::{Arc, Mutex, PoisonError};

/// Largest ring dimension the decoder supports.
const MAX_RING_DIM: usize = 2048;
/// Pool failure target: running out has probability at most `2^-161`.
const POOL_FAILURE_BITS: u128 = 161;
/// `224 >= 2 * 161 * ln 2`, the Chernoff exponent constant for `2^-161`.
const POOL_CHERNOFF_SCALE: u128 = 224;
/// Largest pool the sizing search considers.
const MAX_POOL_CANDIDATES: u128 = 1 << 32;
/// Largest message, matching Jolt's per-message squeeze limit.
const MAX_MESSAGE_BYTES: usize = 1 << 24;
/// Most binomial tables kept for reuse across decodes.
const MAX_CACHED_BINOMIAL_TABLES: usize = 8;
#[cfg(feature = "parallel")]
const PARALLEL_DECODE_WORK_THRESHOLD: usize = 1 << 12;

/// Lower bounds used to size an operator-norm pool.
///
/// `accept` is a rational lower bound `u / v` on the probability that one
/// candidate is accepted: the exact in-range rate times the measured predicate
/// rate lowered by more than ten standard errors. `fail_bits` is a rational
/// lower bound on `-log2(1 - u / v)`. A wrong bound changes only how often an
/// honest message runs out, never which challenges are accepted.
struct PoolRates {
    accept: (u128, u128),
    fail_bits: (u128, u128),
}

/// Pool rates and certified accepted-set exponent of one rejection policy.
fn operator_norm_rates(rejection: OperatorNormRejection) -> Result<(PoolRates, u32), AkitaError> {
    match rejection {
        // In-range rate 0.5557, predicate rate at least 0.65 (measured 0.6623).
        // The certificate retains 2^128.062 accepted challenges.
        OperatorNormRejection::D64_SELECTIVE_L2 => Ok((
            PoolRates {
                accept: (3611, 10_000),
                fail_bits: (64, 100),
            },
            128,
        )),
        // In-range rate 0.7692, predicate rate at least 0.78 (measured 0.7908).
        // The certificate retains 2^128.563 accepted challenges.
        OperatorNormRejection::D128_SELECTIVE_L2 => Ok((
            PoolRates {
                accept: (5999, 10_000),
                fail_bits: (132, 100),
            },
            128,
        )),
        _ => Err(AkitaError::InvalidSetup(
            "unsupported operator-norm rejection policy".into(),
        )),
    }
}

/// Fixed-length message layout and decoder for one group's sparse challenges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseChallengeStream {
    ring_d: usize,
    config: SparseChallengeConfig,
    rejection: Option<OperatorNormRejection>,
    count: usize,
    candidates: usize,
    candidate_bits: usize,
    rank_bits: u32,
    rank_space: u128,
    pm1_rank_space: u128,
    set_bits: u32,
}

impl SparseChallengeStream {
    /// Lay out `count` challenges of one family.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the family is invalid or
    /// outside the decoder's supported range, the rejection policy does not
    /// cover the family, or a size overflows.
    pub fn new(
        ring_d: usize,
        config: &SparseChallengeConfig,
        rejection: Option<OperatorNormRejection>,
        count: usize,
    ) -> Result<Self, AkitaError> {
        if ring_d == 0 || ring_d > MAX_RING_DIM {
            return Err(AkitaError::InvalidSetup(format!(
                "ring dimension {ring_d} is outside the sparse decoder range 1..={MAX_RING_DIM}"
            )));
        }
        config
            .validate_dyn(ring_d)
            .map_err(|error| AkitaError::InvalidSetup(error.into()))?;
        if count == 0 {
            return Err(AkitaError::InvalidSetup(
                "a sparse challenge message needs at least one challenge".into(),
            ));
        }
        let pm1_rank_space = binomial(ring_d, config.count_pm1)?;
        let pm2_rank_space = binomial(ring_d - config.count_pm1, config.count_pm2)?;
        let rank_space = pm1_rank_space.checked_mul(pm2_rank_space).ok_or_else(|| {
            AkitaError::InvalidSetup("sparse position rank space overflow".into())
        })?;
        let weight = u32::try_from(config.weight())
            .map_err(|_| AkitaError::InvalidSetup("sparse challenge weight exceeds u32".into()))?;
        let (rank_bits, candidates, set_bits) = match rejection {
            None => {
                let rank_bits = u128::BITS - 1 - rank_space.leading_zeros();
                let set_bits = rank_bits.checked_add(weight).ok_or_else(|| {
                    AkitaError::InvalidSetup("sparse challenge width overflow".into())
                })?;
                (rank_bits, count, set_bits)
            }
            Some(rejection) => {
                rejection
                    .validate(ring_d, config)
                    .map_err(|error| AkitaError::InvalidSetup(error.into()))?;
                let (rates, certified_bits) = operator_norm_rates(rejection)?;
                let rank_bits = u128::BITS - (rank_space - 1).leading_zeros();
                (rank_bits, pool_len(count, &rates)?, certified_bits)
            }
        };
        let candidate_bits = usize::try_from(rank_bits)
            .ok()
            .and_then(|bits| bits.checked_add(config.weight()))
            .filter(|&bits| bits > 0)
            .ok_or_else(|| AkitaError::InvalidSetup("sparse challenge width is invalid".into()))?;
        candidates
            .checked_mul(candidate_bits)
            .map(|bits| bits.div_ceil(8))
            .filter(|&bytes| bytes <= MAX_MESSAGE_BYTES)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "sparse challenge message exceeds {MAX_MESSAGE_BYTES} bytes"
                ))
            })?;
        Ok(Self {
            ring_d,
            config: *config,
            rejection,
            count,
            candidates,
            candidate_bits,
            rank_bits,
            rank_space,
            pm1_rank_space,
            set_bits,
        })
    }

    /// Exact byte length of the group's challenge message.
    #[must_use]
    pub fn message_len(&self) -> usize {
        // `new` checked that this product fits.
        (self.candidates * self.candidate_bits).div_ceil(8)
    }

    /// Guaranteed `log2` size of the set each challenge is uniform on: `K` for
    /// an unfiltered family, the certified exponent for an operator-norm one.
    #[must_use]
    pub const fn set_bits(&self) -> u32 {
        self.set_bits
    }

    /// Number of candidates in the message: `count` for an unfiltered family,
    /// the pool size for an operator-norm one.
    #[must_use]
    pub const fn candidates(&self) -> usize {
        self.candidates
    }

    /// Bits read per candidate.
    #[must_use]
    pub const fn candidate_bits(&self) -> usize {
        self.candidate_bits
    }

    /// Decode a message into `count` challenges, or `None` when an
    /// operator-norm pool runs out.
    ///
    /// A prover treats `None` as a failed candidate and moves on; a verifier
    /// rejects.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] when the message has the wrong
    /// length, and [`AkitaError::InvalidSetup`] when the predicate table is
    /// unavailable.
    pub fn decode(&self, message: &[u8]) -> Result<Option<Vec<SparseChallenge>>, AkitaError> {
        if message.len() != self.message_len() {
            return Err(AkitaError::InvalidInput(
                "sparse challenge message has the wrong length".into(),
            ));
        }
        let policy = ChallengePolicy::new(self.ring_d, &self.config, self.rejection)?;
        let binomials = binomial_table(
            self.ring_d,
            self.config.count_pm1.max(self.config.count_pm2),
        )?;
        let decoder = CandidateDecoder {
            stream: self,
            message,
            binomials: &binomials,
        };
        match policy {
            ChallengePolicy::Unrestricted => self.decode_all(&decoder).map(Some),
            ChallengePolicy::OperatorNorm { threshold, table } => {
                let accept = |index: usize| -> Result<Option<SparseChallenge>, AkitaError> {
                    let Some(challenge) = decoder.candidate(index)? else {
                        return Ok(None);
                    };
                    Ok(table
                        .accept_strict_parts(
                            &challenge.positions,
                            &challenge.coeffs,
                            u64::from(threshold),
                        )?
                        .then_some(challenge))
                };
                self.first_accepted(accept)
            }
        }
    }

    fn decode_all(
        &self,
        decoder: &CandidateDecoder<'_>,
    ) -> Result<Vec<SparseChallenge>, AkitaError> {
        let decode = |index: usize| {
            decoder.candidate(index)?.ok_or_else(|| {
                AkitaError::Internal("an unfiltered sparse candidate was out of range".into())
            })
        };
        #[cfg(feature = "parallel")]
        if self.parallel() {
            return (0..self.count).into_par_iter().map(decode).collect();
        }
        (0..self.count).map(decode).collect()
    }

    fn first_accepted(
        &self,
        accept: impl Fn(usize) -> Result<Option<SparseChallenge>, AkitaError> + Sync,
    ) -> Result<Option<Vec<SparseChallenge>>, AkitaError> {
        let mut accepted = Vec::new();
        accepted
            .try_reserve_exact(self.count)
            .map_err(|_| AkitaError::InvalidInput("sparse challenge allocation failed".into()))?;
        // In parallel, decode the pool in order in chunks of twice the
        // challenge count and stop after the chunk that fills the count.
        #[cfg(feature = "parallel")]
        if self.parallel() {
            let chunk = self.count.saturating_mul(2);
            let mut start = 0;
            while accepted.len() < self.count && start < self.candidates {
                let end = start.saturating_add(chunk).min(self.candidates);
                let decoded = (start..end)
                    .into_par_iter()
                    .map(&accept)
                    .collect::<Result<Vec<_>, _>>()?;
                let missing = self.count - accepted.len();
                accepted.extend(decoded.into_iter().flatten().take(missing));
                start = end;
            }
            return Ok(self.full(accepted));
        }
        for index in 0..self.candidates {
            if accepted.len() == self.count {
                break;
            }
            if let Some(challenge) = accept(index)? {
                accepted.push(challenge);
            }
        }
        Ok(self.full(accepted))
    }

    fn full(&self, accepted: Vec<SparseChallenge>) -> Option<Vec<SparseChallenge>> {
        (accepted.len() == self.count).then_some(accepted)
    }

    #[cfg(feature = "parallel")]
    fn parallel(&self) -> bool {
        self.candidates.saturating_mul(self.config.weight()) >= PARALLEL_DECODE_WORK_THRESHOLD
    }
}

/// Least pool size `B >= count` whose run-out probability is at most
/// `2^-161` by either the union bound
/// `(B - W + 1) r >= 161 + (W - 1) bitlen(B)` or the Chernoff bound
/// `(B u - W v)^2 >= 224 B u v` with `B u > W v`.
fn pool_len(count: usize, rates: &PoolRates) -> Result<usize, AkitaError> {
    let overflow = || AkitaError::InvalidSetup("operator-norm pool size overflow".into());
    let (u, v) = rates.accept;
    let (r_num, r_den) = rates.fail_bits;
    let w = u128::try_from(count).map_err(|_| overflow())?;
    let chernoff = |b: u128| -> Option<bool> {
        let bu = b.checked_mul(u)?;
        let wv = w.checked_mul(v)?;
        if bu <= wv {
            return Some(false);
        }
        let gap = bu - wv;
        Some(
            gap.checked_mul(gap)?
                >= POOL_CHERNOFF_SCALE
                    .checked_mul(b)?
                    .checked_mul(u)?
                    .checked_mul(v)?,
        )
    };
    // On `B u > W v` the Chernoff condition is a quadratic that holds exactly
    // from its larger root on, so a doubling and bisection find its least `B`.
    let mut high = w.max(1);
    while !chernoff(high).ok_or_else(overflow)? {
        high = high
            .checked_mul(2)
            .filter(|&b| b <= MAX_POOL_CANDIDATES)
            .ok_or_else(overflow)?;
    }
    let mut low = w;
    while low < high {
        let mid = low + (high - low) / 2;
        if chernoff(mid).ok_or_else(overflow)? {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    let union = |b: u128| -> Option<bool> {
        let tail = b.checked_sub(w)?.checked_add(1)?;
        let bitlen = u128::from(u128::BITS - b.leading_zeros());
        let rhs = w
            .checked_sub(1)?
            .checked_mul(bitlen)?
            .checked_add(POOL_FAILURE_BITS)?;
        Some(tail.checked_mul(r_num)? >= rhs.checked_mul(r_den)?)
    };
    let pool = (w..high)
        .find(|&b| union(b).unwrap_or(false))
        .unwrap_or(high);
    usize::try_from(pool).map_err(|_| overflow())
}

/// `C(n, k)`, or an error when it does not fit in `u128`.
fn binomial(n: usize, k: usize) -> Result<u128, AkitaError> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut value = 1u128;
    for i in 0..k {
        // `value * (n - i)` is divisible by `i + 1` at every step.
        value = value
            .checked_mul((n - i) as u128)
            .map(|product| product / (i as u128 + 1))
            .ok_or_else(|| AkitaError::InvalidSetup("sparse binomial overflow".into()))?;
    }
    Ok(value)
}

/// The binomial table for `n <= max_n`, `k <= max_k`, built once and then
/// shared by every decode of a family with that ring dimension and weight.
fn binomial_table(max_n: usize, max_k: usize) -> Result<Arc<BinomialTable>, AkitaError> {
    static TABLES: Mutex<Vec<Arc<BinomialTable>>> = Mutex::new(Vec::new());
    let mut tables = TABLES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(table) = tables
        .iter()
        .find(|table| table.max_n == max_n && table.max_k == max_k)
    {
        return Ok(Arc::clone(table));
    }
    let table = Arc::new(BinomialTable::new(max_n, max_k)?);
    if tables.len() < MAX_CACHED_BINOMIAL_TABLES {
        tables.push(Arc::clone(&table));
    }
    Ok(table)
}

/// Pascal's triangle `C(n, k)` for `n <= max_n`, `k <= max_k`, saturating at
/// `u128::MAX`, stored by column so that `C(0..=max_n, k)` is contiguous.
/// Every rank the decoder unranks is below `u128::MAX`, so a saturated entry
/// still compares correctly as "too large".
struct BinomialTable {
    max_n: usize,
    max_k: usize,
    entries: Vec<u128>,
}

impl BinomialTable {
    fn new(max_n: usize, max_k: usize) -> Result<Self, AkitaError> {
        let rows = max_n + 1;
        let len = rows
            .checked_mul(max_k + 1)
            .ok_or_else(|| AkitaError::InvalidSetup("sparse binomial table overflow".into()))?;
        let mut entries: Vec<u128> = Vec::new();
        entries.try_reserve_exact(len).map_err(|_| {
            AkitaError::InvalidInput("sparse binomial table allocation failed".into())
        })?;
        entries.resize(len, 0);
        let mut columns = entries.chunks_exact_mut(rows);
        if let Some(mut previous) = columns.next() {
            previous.fill(1);
            for column in columns {
                // `C(n, k) = C(n - 1, k) + C(n - 1, k - 1)`, with `C(0, k) = 0`.
                let mut above = 0u128;
                for (entry, &diagonal) in column.iter_mut().skip(1).zip(previous.iter()) {
                    above = above.saturating_add(diagonal);
                    *entry = above;
                }
                previous = column;
            }
        }
        Ok(Self {
            max_n,
            max_k,
            entries,
        })
    }

    /// `C(0..upper, k)`.
    fn column(&self, k: usize, upper: usize) -> Result<&[u128], AkitaError> {
        let rows = self.max_n + 1;
        (k <= self.max_k && upper <= rows)
            .then(|| self.entries.get(k * rows..k * rows + upper))
            .flatten()
            .ok_or_else(|| AkitaError::Internal("sparse binomial table is too small".into()))
    }

    /// Unrank `rank < C(n, k)` into a `k`-subset of `0..n`, colexicographic,
    /// written in increasing order into `out`.
    ///
    /// Element `i` (from 1) is the largest `c` below element `i + 1` with
    /// `C(c, i) <= rank`. The elements decrease, so one downward scan over all
    /// of `0..n` finds them all; it beats a binary search per element, whose
    /// comparisons the CPU cannot predict.
    fn unrank(
        &self,
        mut rank: u128,
        n: usize,
        k: usize,
        out: &mut [u32],
    ) -> Result<(), AkitaError> {
        let out = out
            .get_mut(..k)
            .ok_or_else(|| AkitaError::Internal("sparse unrank output is too short".into()))?;
        let mut upper = n;
        for (i, slot) in out.iter_mut().enumerate().rev() {
            let column = self.column(i + 1, upper)?;
            let (element, &value) = column
                .iter()
                .enumerate()
                .rfind(|&(_, &value)| value <= rank)
                .ok_or_else(|| AkitaError::Internal("sparse rank was out of range".into()))?;
            rank -= value;
            *slot = u32::try_from(element)
                .map_err(|_| AkitaError::Internal("sparse position exceeds u32".into()))?;
            upper = element;
        }
        if rank != 0 {
            return Err(AkitaError::Internal("sparse rank was out of range".into()));
        }
        Ok(())
    }
}

struct CandidateDecoder<'a> {
    stream: &'a SparseChallengeStream,
    message: &'a [u8],
    binomials: &'a BinomialTable,
}

impl CandidateDecoder<'_> {
    /// Candidate `index`, or `None` when its position rank is out of range.
    fn candidate(&self, index: usize) -> Result<Option<SparseChallenge>, AkitaError> {
        let stream = self.stream;
        let (a, b) = (stream.config.count_pm1, stream.config.count_pm2);
        let weight = a + b;
        let start = index
            .checked_mul(stream.candidate_bits)
            .ok_or_else(|| AkitaError::Internal("sparse candidate offset overflow".into()))?;
        let rank_start = start + weight;
        let rank = read_bits(self.message, rank_start, stream.rank_bits)?;
        if rank >= stream.rank_space {
            return Ok(None);
        }
        // One `u128` division, skipped for a ±1-only family.
        let pm2_rank = if b == 0 {
            0
        } else {
            rank / stream.pm1_rank_space
        };
        let pm1_rank = rank - pm2_rank * stream.pm1_rank_space;
        let mut positions = crate::SparseChallengePositions::from_elem(0, weight);
        let (pm1, pm2) = positions.split_at_mut(a);
        self.binomials.unrank(pm1_rank, stream.ring_d, a, pm1)?;
        if b > 0 {
            self.binomials.unrank(pm2_rank, stream.ring_d - a, b, pm2)?;
            map_to_complement(pm1, pm2, stream.ring_d)?;
        }
        let mut coeffs = crate::SparseChallengeCoefficients::with_capacity(weight);
        for offset in (0..weight).step_by(u64::BITS as usize) {
            let take = (weight - offset).min(u64::BITS as usize);
            let signs = read_bits(self.message, start + offset, take as u32)?;
            for bit in 0..take {
                // Branch-free: the sign bits are random, so a branch on them
                // mispredicts half the time.
                let magnitude = 1 + i8::from(offset + bit >= a);
                let negative = ((signs >> bit) & 1) as i8;
                coeffs.push(magnitude - 2 * negative * magnitude);
            }
        }
        Ok(Some(SparseChallenge { positions, coeffs }))
    }
}

/// Replace each increasing index in `slots` by the index-th position, in
/// increasing order, of `0..ring_d` that is not in the increasing set `taken`.
fn map_to_complement(taken: &[u32], slots: &mut [u32], ring_d: usize) -> Result<(), AkitaError> {
    // Positions of `taken` passed so far; the slots increase, so it only grows.
    let mut skipped = 0usize;
    for slot in slots.iter_mut() {
        let mut position = *slot as usize + skipped;
        while taken
            .get(skipped)
            .is_some_and(|&taken| taken as usize <= position)
        {
            skipped += 1;
            position += 1;
        }
        if position >= ring_d {
            return Err(AkitaError::Internal(
                "sparse complement slot out of range".into(),
            ));
        }
        *slot = u32::try_from(position)
            .map_err(|_| AkitaError::Internal("sparse position exceeds u32".into()))?;
    }
    Ok(())
}

/// Read `width <= 128` bits starting at bit `offset` of a little-endian bit
/// stream.
fn read_bits(message: &[u8], offset: usize, width: u32) -> Result<u128, AkitaError> {
    let shift = offset % 8;
    let bytes = message
        .get(offset / 8..(offset + width as usize).div_ceil(8))
        .filter(|_| width <= u128::BITS)
        .ok_or_else(|| AkitaError::Internal("sparse message read past its end".into()))?;
    let mut window = [0u8; 16];
    let low = bytes.len().min(16);
    window[..low].copy_from_slice(&bytes[..low]);
    let mut value = u128::from_le_bytes(window) >> shift;
    // A read of up to 128 bits spans 17 bytes only when it starts mid-byte.
    if let Some(&high) = bytes.get(16) {
        value |= u128::from(high) << (128 - shift);
    }
    if width < u128::BITS {
        value &= (1u128 << width) - 1;
    }
    Ok(value)
}

#[cfg(test)]
mod tests;
