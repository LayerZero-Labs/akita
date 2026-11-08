//! The response proof's two new sumchecks (Whiteout §4.3): the degree-2
//! product sumcheck `Σ_i U_i W_i` and the degree-3 fused range-and-relation
//! sumcheck `Σ_x A(x)E(x) + B(x)E(x)(E(x) + 1)`, and the prover's clear
//! round channel. Tables bind their lowest index bit first, as
//! `fold_evals_in_place` does, so challenge `k` is coordinate `k` of the
//! little-endian point.

use akita_error::AkitaError;
use akita_sumcheck::{SumcheckKernel, SumcheckProverChannel, SumcheckRole};
use akita_transcript::{ext_challenge, ChallengeCodec, ProtocolSiteId, ProverChannel};

use zeroize::{Zeroize, Zeroizing};

use super::rows::{eq_table, BatchWeights};
use super::{response_site, F};
use jolt_algebra::solinas::parallel::*;
use jolt_algebra::{
    cfg_fold_reduce, cfg_into_iter, lift_ct, Field, Fold, JoltField, One, Ring, Zero,
};
use jolt_poly::UnivariatePoly;

/// A table of values derived from `e`, zeroed over its whole capacity when
/// dropped. Folding builds the next table and drops the previous one, so no
/// freed buffer keeps `e`-derived values.
pub(crate) struct SecretTable<F: Copy + Send>(Vec<F>);

impl<F: Field> SecretTable<F> {
    /// `values` zero-padded to `len`, in a buffer of exactly that capacity;
    /// `values`' own buffer is wiped.
    pub(crate) fn padded(values: Vec<F>, len: usize) -> Self {
        let source = Self(values);
        let mut table = Vec::with_capacity(len.max(source.0.len()));
        table.extend_from_slice(&source.0);
        table.resize(len.max(source.0.len()), F::zero());
        Self(table)
    }

    /// Wraps a buffer about to hold `e`-derived values, so they are wiped
    /// with it.
    fn wrap(values: Vec<F>) -> Self {
        Self(values)
    }

    /// `low + r·(high − low)` over each pair.
    fn fold(&mut self, r: F) {
        let source = &self.0;
        let folded = cfg_into_iter!(0..source.len().div_ceil(2))
            .map(|i| {
                let low = source[2 * i];
                let high = source.get(2 * i + 1).copied().unwrap_or_default();
                low + r * (high - low)
            })
            .collect();
        *self = Self(folded);
    }
}

impl<F: Copy + Send> std::ops::Deref for SecretTable<F> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        &self.0
    }
}

impl<F: Copy + Send> Drop for SecretTable<F> {
    fn drop(&mut self) {
        self.0.clear();
        self.0
            .spare_capacity_mut()
            .par_chunks_mut(1 << 16)
            .for_each(|chunk| chunk.zeroize());
    }
}

/// Round evaluations at `0..=DEGREE` of `Σ_i f(t_i(X))`, where `t_i(X)` are
/// the tables' pairs `(2i, 2i + 1)` interpolated on the line.
fn round_evaluations<F: Field, const TABLES: usize, const POINTS: usize>(
    tables: [&[F]; TABLES],
    term: impl Fn(&[F; TABLES]) -> F + Sync,
) -> [F; POINTS] {
    let half = tables[0].len() / 2;
    cfg_fold_reduce!(
        0..half,
        || [F::zero(); POINTS],
        |mut sums: [F; POINTS], i: usize| {
            let low: [F; TABLES] = std::array::from_fn(|t| tables[t][2 * i]);
            let slope: [F; TABLES] =
                std::array::from_fn(|t| tables[t][2 * i + 1] - tables[t][2 * i]);
            let mut point = low;
            for sum in &mut sums {
                *sum += term(&point);
                for (value, step) in point.iter_mut().zip(&slope) {
                    *value += *step;
                }
            }
            sums
        },
        |mut a: [F; POINTS], b: [F; POINTS]| {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
            a
        }
    )
}

/// The round polynomial through `evals` at `0..N`. In tests and under
/// `test-support` it is shifted by half its gap to the running claim, so a
/// prover of a false statement (`UncheckedResponseProver`, the unit tests'
/// `Cheating`) still sends consistent rounds and only the verifier's checks
/// can reject it; an honest round has no gap, so honest proofs are
/// unchanged.
fn round_polynomial<F: Field, const N: usize>(
    #[cfg_attr(not(any(test, feature = "test-support")), expect(unused_mut))] mut evals: [F; N],
    #[cfg_attr(not(any(test, feature = "test-support")), expect(unused_variables))] claim: F,
) -> UnivariatePoly<F> {
    #[cfg(any(test, feature = "test-support"))]
    {
        let shift = (claim - evals[0] - evals[1]).half();
        for eval in &mut evals {
            *eval += shift;
        }
    }
    UnivariatePoly::from_evals(&evals)
}

/// `Σ_i U_i W_i` over `a` variables.
pub(crate) struct ProductKernel<F: Copy + Send> {
    pub(crate) u: SecretTable<F>,
    pub(crate) w: SecretTable<F>,
    /// `t_×`, the claimed sum.
    pub(crate) target: F,
}

impl<F: Field + Fold + Send> SumcheckKernel<F> for ProductKernel<F> {
    fn num_rounds(&self) -> usize {
        self.u.len().trailing_zeros() as usize
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> F {
        self.target
    }

    fn finish(&mut self) -> Result<(), AkitaError> {
        Ok(())
    }

    fn round_polynomial(
        &mut self,
        _round: usize,
        claim: F,
    ) -> Result<UnivariatePoly<F>, AkitaError> {
        let evals: [F; 3] = round_evaluations([&self.u, &self.w], |[u, w]| *u * *w);
        Ok(round_polynomial(evals, claim))
    }

    fn bind_challenge(&mut self, _round: usize, challenge: F) -> Result<(), AkitaError> {
        self.u.fold(challenge);
        self.w.fold(challenge);
        Ok(())
    }
}

/// `Σ_x A(x)E(x) + B(x)E(x)(E(x) + 1)` over `ν` variables, with
/// `B(x) = γ·eq(τ, x) + S(x)`: the dense part held as its point and the
/// product of its bound factors (split eq), the sparse part `S = ζ·eq(τ, ·)`
/// on `I_bin` as its nonzero entries.
///
/// For its first [`STREAMED_ROUNDS`] rounds the kernel holds neither `A`
/// nor `E`: each round writes `A` from its factors ([`BatchWeights`]) and
/// lifts `E` from `e`'s digits, one chunk at a time, folded through the
/// challenges bound so far. The bind that ends those rounds writes both,
/// bound, into one buffer of `2^{ν + 1 − STREAMED_ROUNDS}` entries, `A` in
/// `[0, len)` and `E` in `[half, half + len)`, which then fold in place
/// ([`fold_in_place`]); no table is allocated after it, and the buffer,
/// which holds `E`, is wiped when freed.
///
/// `E` is zero past the committed digits (the padding to `2^ν`), where the
/// summand vanishes whatever `A` and `B` are, so the dense sums and `E`'s
/// folds stop at `E`'s live length ([`Self::live`]); `A` is still folded
/// in full, since a pair that straddles the boundary reads `A`'s entry
/// past it.
///
/// `E` is constant time in `e`: every read is at a public index (the live
/// length is the committed length, which the statement's layout fixes),
/// the digits enter `F` through [`lift_ct`], which does not branch on
/// their sign, and every fold and summand is field arithmetic. The chunk
/// scratch that holds `E` is a [`SecretTable`] too.
pub(crate) struct FusedKernel {
    tables: FusedTables,
    /// The length of `A`, and of `E`, bound in the challenges so far.
    len: usize,
    /// The committed digits' count, bound in the challenges so far: `E` is
    /// zero from it on.
    live: usize,
    b: SplitB<F>,
    claim: F,
}

enum FusedTables {
    /// Fewer than [`STREAMED_ROUNDS`] challenges bound, in `bound`.
    Streamed {
        a: BatchWeights,
        digits: Zeroizing<Vec<i8>>,
        bound: Vec<F>,
    },
    /// `A` in `[0, len)` and `E` in `[half, half + len)`, `E` held exactly
    /// on its first [`FusedKernel::held_e`] entries.
    Held { tables: SecretTable<F>, half: usize },
}

/// Rounds for which [`FusedKernel`] streams `A` and `E` instead of holding
/// them: it holds `2^{1 − STREAMED_ROUNDS}` of `A`'s length afterwards
/// (256 MiB at `ν = 26`), which fits in what the digit-range session freed.
/// A streamed round writes `A` and `E` from scratch, a few field operations
/// per unbound entry, where a held round reads half the entries of the one
/// before: fewer streamed rounds hold more, more cost more time.
const STREAMED_ROUNDS: usize = 3;

/// Unbound entries per task of a streamed round.
const STREAM_CHUNK: usize = 1 << 12;

/// `B = γ·eq(τ, ·) + S`, `S` sparse.
pub(crate) struct SplitB<F> {
    /// `τ`'s coordinates still free, lowest first.
    point: Vec<F>,
    /// `γ·Π eq(τ_i, r_i)` over the bound coordinates.
    scale: F,
    /// `S`'s nonzero entries, by increasing index.
    sparse: Vec<(usize, F)>,
    /// The buffer the next fold of `sparse` writes.
    spare: Vec<(usize, F)>,
}

impl<F: JoltField> SplitB<F> {
    /// `γ·eq(τ, ·) + ζ·eq(τ, ·)·1_{I}` for sorted disjoint `intervals`.
    pub(crate) fn new(
        point: &[F],
        gamma: F,
        zeta: F,
        intervals: &[std::ops::Range<usize>],
    ) -> Self {
        let half = point.len() / 2;
        let (lo, hi) = (eq_table(&point[..half]), eq_table(&point[half..]));
        let mask = (1 << half) - 1;
        let sparse = intervals
            .iter()
            .flat_map(|interval| interval.clone())
            .map(|j| (j, zeta * lo[j & mask] * hi[j >> half]))
            .collect();
        Self {
            point: point.to_vec(),
            scale: gamma,
            sparse,
            spare: Vec::new(),
        }
    }

    /// `eq(τ_rest, ·)` for the coordinates after the current one, split in
    /// halves: `(lo, hi, bits of lo)`.
    fn rest(&self) -> (Vec<F>, Vec<F>, usize) {
        let rest = &self.point[1.min(self.point.len())..];
        let half = rest.len() / 2;
        (eq_table(&rest[..half]), eq_table(&rest[half..]), half)
    }
}

/// Tables shorter than this are folded serially: they cannot pay for a
/// parallel dispatch, as in `fold_evals_in_place`.
#[cfg(feature = "parallel")]
const PARALLEL_FOLD_MIN: usize = 1 << 12;

/// Sets `out[j] = value(j)` for every `j`.
fn fill<F: Send>(out: &mut [F], value: impl Fn(usize) -> F + Sync) {
    let set = |(j, slot): (usize, &mut F)| *slot = value(j);
    #[cfg(feature = "parallel")]
    if out.len() >= PARALLEL_FOLD_MIN {
        out.par_iter_mut().enumerate().for_each(set);
        return;
    }
    out.iter_mut().enumerate().for_each(set);
}

/// Folds `values`' pairs into its first `values.len().div_ceil(2)`
/// entries: entry `i` becomes `low + r·(high − low)` of pair `i` (`high`
/// zero past the end). Outputs `[s, 2s)` read only `[2s, 4s)`, so the
/// blocks `s = 1, 2, 4, …` run in that order, each in parallel, and every
/// input is read before it is overwritten.
fn fold_in_place<F: Field + Fold>(values: &mut [F], r: F) {
    let len = values.len();
    let outputs = len.div_ceil(2);
    if outputs == 0 {
        return;
    }
    let ctx = F::precompute(r);
    let high = |input: &[F], index: usize| input.get(index).copied().unwrap_or_default();
    values[0] = F::fold_one(&ctx, values[0], high(values, 1));
    let mut start = 1;
    while start < outputs {
        let end = (2 * start).min(outputs);
        let (written, read) = values.split_at_mut(2 * start);
        let input = &read[..(2 * end).min(len) - 2 * start];
        fill(&mut written[start..end], |j| {
            F::fold_one(&ctx, input[2 * j], high(input, 2 * j + 1))
        });
        start = end;
    }
}

/// `e·(a + b·(e + 1))`, the fused sumcheck's summand.
fn fused_term<F: Field>(a: F, b: F, e: F) -> F {
    e * (a + b * (e + F::one()))
}

/// `B`'s dense part in one round: `scale·eq(τ_k, t)·eq(τ_rest, i)` for
/// pair `i` at `t`, as `lo[i & mask]·hi[i >> bits]·(start + t·step)` with
/// the scale folded into `hi`.
struct RoundB {
    lo: Vec<F>,
    hi: Vec<F>,
    bits: usize,
    start: F,
    step: F,
}

impl RoundB {
    fn new(b: &SplitB<F>) -> Self {
        let tau = b.point.first().copied().unwrap_or_default();
        let (lo, hi, bits) = b.rest();
        Self {
            lo,
            hi: hi.iter().map(|value| *value * b.scale).collect(),
            bits,
            // eq(τ_k, t) = (1 − τ_k) + t·(2τ_k − 1).
            start: F::one() - tau,
            step: tau + tau - F::one(),
        }
    }

    /// `Σ_i Σ_t` over the pairs `i` of `a` and `e`, pair `first + i` of the
    /// round, of `e_t·(a_t + B_t·(e_t + 1))` at `t = 0..=3`.
    fn sums(&self, first: usize, a: &[F], e: &[F]) -> [F; 4] {
        let mask = (1 << self.bits) - 1;
        let mut sums = [F::zero(); 4];
        for (i, (a, e)) in a.chunks_exact(2).zip(e.chunks_exact(2)).enumerate() {
            let pair = first + i;
            let weight = self.lo[pair & mask] * self.hi[pair >> self.bits];
            let (mut b_t, b_slope) = (weight * self.start, weight * self.step);
            let (mut a_t, a_slope) = (a[0], a[1] - a[0]);
            let (mut e_t, e_slope) = (e[0], e[1] - e[0]);
            for sum in &mut sums {
                *sum += fused_term(a_t, b_t, e_t);
                a_t += a_slope;
                b_t += b_slope;
                e_t += e_slope;
            }
        }
        sums
    }
}

fn add_sums(mut x: [F; 4], y: [F; 4]) -> [F; 4] {
    for (x, y) in x.iter_mut().zip(y) {
        *x += y;
    }
    x
}

/// Writes `E`, bound in `bound`, on the unbound entries
/// `[start, start + out.len())` into the first `out.len() >> bound.len()`
/// entries of `out`: the digits lifted by [`lift_ct`], then folded.
fn bound_e(digits: &[i8], bound: &[F], start: usize, out: &mut [F]) -> Result<(), AkitaError> {
    let digits = digits
        .get(start..start + out.len())
        .ok_or_else(|| AkitaError::Internal("fused digits range".into()))?;
    for (dst, &digit) in out.iter_mut().zip(digits) {
        *dst = lift_ct(i64::from(digit));
    }
    fold_through(out, bound);
    Ok(())
}

/// Folds `values` through each of `bound` in turn, in place.
fn fold_through(values: &mut [F], bound: &[F]) {
    let mut len = values.len();
    for &challenge in bound {
        fold_in_place(&mut values[..len], challenge);
        len = len.div_ceil(2);
    }
}

/// One task's buffers for a chunk of a streamed round.
struct ChunkScratch {
    a: Vec<F>,
    e: SecretTable<F>,
}

impl ChunkScratch {
    fn new(len: usize) -> Self {
        Self {
            a: vec![F::zero(); len],
            e: SecretTable::wrap(vec![F::zero(); len]),
        }
    }

    /// `A` and `E` bound in `bound` on the unbound chunk starting at
    /// `start`: the first `len >> bound.len()` entries of each.
    fn stream(
        &mut self,
        weights: &BatchWeights,
        digits: &[i8],
        bound: &[F],
        start: usize,
    ) -> Result<(&[F], &[F]), AkitaError> {
        weights.fill(start, &mut self.a)?;
        fold_through(&mut self.a, bound);
        bound_e(digits, bound, start, &mut self.e.0)?;
        let len = self.a.len() >> bound.len();
        Ok((&self.a[..len], &self.e[..len]))
    }

    /// [`Self::stream`]'s `A` alone, for a chunk where `E` is zero.
    fn stream_a(
        &mut self,
        weights: &BatchWeights,
        bound: &[F],
        start: usize,
    ) -> Result<&[F], AkitaError> {
        weights.fill(start, &mut self.a)?;
        fold_through(&mut self.a, bound);
        Ok(&self.a[..self.a.len() >> bound.len()])
    }
}

/// `f(scratch, chunk)` summed over `chunks` chunks, each task with its own
/// scratch of `len` entries.
fn chunk_sums(
    chunks: usize,
    len: usize,
    f: impl Fn(&mut ChunkScratch, usize) -> Result<[F; 4], AkitaError> + Sync,
) -> Result<[F; 4], AkitaError> {
    #[cfg(feature = "parallel")]
    return (0..chunks)
        .into_par_iter()
        .map_init(|| ChunkScratch::new(len), &f)
        .try_reduce(|| [F::zero(); 4], |x, y| Ok(add_sums(x, y)));
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = ChunkScratch::new(len);
        (0..chunks).try_fold([F::zero(); 4], |sums, chunk| {
            Ok(add_sums(sums, f(&mut scratch, chunk)?))
        })
    }
}

impl FusedKernel {
    /// The kernel of the row weights `a`, `B` and `e`'s committed digits,
    /// one per weight, claiming the sum `claim`.
    /// `E` is zero from `live` on.
    pub(crate) fn new(
        a: BatchWeights,
        b: SplitB<F>,
        digits: Zeroizing<Vec<i8>>,
        live: usize,
        claim: F,
    ) -> Result<Self, AkitaError> {
        if digits.len() != a.len() || !a.len().is_power_of_two() || live > a.len() {
            return Err(AkitaError::Internal(
                "the fused sumcheck's tables differ in length".into(),
            ));
        }
        Ok(Self {
            len: a.len(),
            live,
            tables: FusedTables::Streamed {
                a,
                digits,
                bound: Vec::with_capacity(STREAMED_ROUNDS),
            },
            b,
            claim,
        })
    }

    /// The value `E` holds at its first entry, bound in the challenges so
    /// far: once bound in every variable, `E`'s value.
    pub(crate) fn e_first(&self) -> Option<F> {
        match &self.tables {
            FusedTables::Streamed { digits, bound, .. } => {
                let mut values = [F::zero(); 1 << STREAMED_ROUNDS];
                let values = values.get_mut(..1 << bound.len())?;
                bound_e(digits, bound, 0, values).ok()?;
                values.first().copied()
            }
            FusedTables::Held { tables, half } => tables.get(*half).copied(),
        }
    }

    /// `Σ_t S_t·e_t(e_t + 1)` over the pairs where `S` is nonzero, with
    /// `e_pair(i)` the pair's `(e_0, e_1 − e_0)`.
    fn sparse_evaluations(
        &self,
        e_pair: impl Fn(usize) -> Result<(F, F), AkitaError>,
    ) -> Result<[F; 4], AkitaError> {
        let mut sums = [F::zero(); 4];
        let mut entries = self.b.sparse.iter().peekable();
        while let Some(&(index, value)) = entries.next() {
            let pair = index / 2;
            let (mut low, mut high) = (F::zero(), F::zero());
            if index % 2 == 0 {
                low = value;
                if let Some(&&(next, next_value)) = entries.peek() {
                    if next == index + 1 {
                        high = next_value;
                        entries.next();
                    }
                }
            } else {
                high = value;
            }
            let (e_low, e_slope) = e_pair(pair)?;
            for (t, sum) in sums.iter_mut().enumerate() {
                let t = F::from_u64(t as u64);
                let s = low + t * (high - low);
                let e = e_low + t * e_slope;
                *sum += s * e * (e + F::one());
            }
        }
        Ok(sums)
    }

    /// The entries of `E` the held tables hold exactly: every pair that
    /// reaches a live entry. Past them `E` is zero, and the buffer stale.
    fn held_e(&self) -> usize {
        (2 * self.live.div_ceil(2)).min(self.len)
    }

    /// The round's dense sums while streaming: chunk by chunk over the
    /// unbound tables that reach a live entry.
    fn streamed_sums(
        round: &RoundB,
        weights: &BatchWeights,
        digits: &[i8],
        bound: &[F],
        live: usize,
    ) -> Result<[F; 4], AkitaError> {
        let chunk = STREAM_CHUNK.min(weights.len());
        chunk_sums(live.div_ceil(chunk), chunk, |scratch, index| {
            let start = index * chunk;
            let (a, e) = scratch.stream(weights, digits, bound, start)?;
            Ok(round.sums((start >> bound.len()) / 2, a, e))
        })
    }

    /// `A` and `E` bound in `bound`, written into one buffer: `A` in its
    /// lower half, `E` in its upper half. `E`'s chunks past `live` stay
    /// zero.
    fn hold(
        weights: &BatchWeights,
        digits: &[i8],
        bound: &[F],
        live: usize,
    ) -> Result<(SecretTable<F>, usize), AkitaError> {
        let chunk = STREAM_CHUNK.min(weights.len());
        let (half, out) = (weights.len() >> bound.len(), chunk >> bound.len());
        let mut tables = SecretTable::wrap(vec![F::zero(); 2 * half]);
        let (a, e) = tables.0.split_at_mut(half);
        let write = |scratch: &mut ChunkScratch,
                     (index, (a_out, e_out)): (usize, (&mut [F], &mut [F]))| {
            let start = index * chunk;
            if start < live {
                let (a, e) = scratch.stream(weights, digits, bound, start)?;
                a_out.copy_from_slice(a);
                e_out.copy_from_slice(e);
            } else {
                a_out.copy_from_slice(scratch.stream_a(weights, bound, start)?);
            }
            Ok::<(), AkitaError>(())
        };
        #[cfg(feature = "parallel")]
        a.par_chunks_mut(out)
            .zip(e.par_chunks_mut(out))
            .enumerate()
            .try_for_each_init(|| ChunkScratch::new(chunk), write)?;
        #[cfg(not(feature = "parallel"))]
        {
            let mut scratch = ChunkScratch::new(chunk);
            a.chunks_mut(out)
                .zip(e.chunks_mut(out))
                .enumerate()
                .try_for_each(|pair| write(&mut scratch, pair))?;
        }
        Ok((tables, half))
    }
}

impl SumcheckKernel<F> for FusedKernel {
    fn num_rounds(&self) -> usize {
        self.len.trailing_zeros() as usize
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> F {
        self.claim
    }

    fn finish(&mut self) -> Result<(), AkitaError> {
        Ok(())
    }

    fn round_polynomial(
        &mut self,
        _round: usize,
        claim: F,
    ) -> Result<UnivariatePoly<F>, AkitaError> {
        let round = RoundB::new(&self.b);
        let held_e = self.held_e();
        let (dense, sparse) = match &self.tables {
            FusedTables::Streamed { a, digits, bound } => (
                Self::streamed_sums(&round, a, digits, bound, self.live << bound.len())?,
                self.sparse_evaluations(|pair| {
                    let mut values = [F::zero(); 2 << STREAMED_ROUNDS];
                    let width = 2 << bound.len();
                    let values = values
                        .get_mut(..width)
                        .ok_or_else(|| AkitaError::Internal("fused pair width".into()))?;
                    bound_e(digits, bound, pair * width, values)?;
                    Ok((values[0], values[1] - values[0]))
                })?,
            ),
            FusedTables::Held { tables, half } => {
                let a = &tables[..held_e];
                let e = &tables[*half..*half + held_e];
                let chunk = STREAM_CHUNK.min(self.len);
                let dense = cfg_fold_reduce!(
                    0..held_e.div_ceil(chunk),
                    || [F::zero(); 4],
                    |sums: [F; 4], index: usize| {
                        let range = index * chunk..((index + 1) * chunk).min(held_e);
                        add_sums(
                            sums,
                            round.sums(range.start / 2, &a[range.clone()], &e[range]),
                        )
                    },
                    add_sums
                );
                (
                    dense,
                    self.sparse_evaluations(|pair| {
                        Ok(match (e.get(2 * pair), e.get(2 * pair + 1)) {
                            (Some(&low), Some(&high)) => (low, high - low),
                            _ => (F::zero(), F::zero()),
                        })
                    })?,
                )
            }
        };
        let evals: [F; 4] = std::array::from_fn(|t| dense[t] + sparse[t]);
        Ok(round_polynomial(evals, claim))
    }

    fn bind_challenge(&mut self, _round: usize, challenge: F) -> Result<(), AkitaError> {
        let len = self.len;
        let held_e = self.held_e();
        let live = self.live.div_ceil(2);
        match &mut self.tables {
            FusedTables::Streamed { a, digits, bound } => {
                // `live` rounded up to whole bound blocks: every chunk it
                // reaches holds a live entry.
                let unbound = self.live << bound.len();
                bound.push(challenge);
                if bound.len() == STREAMED_ROUNDS {
                    let (tables, half) = Self::hold(a, digits, bound, unbound)?;
                    self.tables = FusedTables::Held { tables, half };
                }
            }
            FusedTables::Held { tables, half } => {
                fold_in_place(&mut tables.0[..len], challenge);
                let e = &mut tables.0[*half..*half + len];
                fold_in_place(&mut e[..held_e], challenge);
                // The pair that straddles the new live length reads its
                // high entry, zero in `E` but stale in the buffer.
                if live < len.div_ceil(2) {
                    e[live] = F::zero();
                }
            }
        }
        self.len = len.div_ceil(2);
        self.live = live;
        let b = &mut self.b;
        let tau = b.point.first().copied().unwrap_or_default();
        b.scale *= (F::one() - tau) * (F::one() - challenge) + tau * challenge;
        b.point.remove(0);
        b.spare.clear();
        b.spare.reserve(b.sparse.len());
        for &(index, value) in &b.sparse {
            let weight = if index % 2 == 0 {
                F::one() - challenge
            } else {
                challenge
            };
            match b.spare.last_mut() {
                Some((pair, sum)) if *pair == index / 2 => *sum += weight * value,
                _ => b.spare.push((index / 2, weight * value)),
            }
        }
        std::mem::swap(&mut b.sparse, &mut b.spare);
        Ok(())
    }
}

/// Clear sumcheck rounds of response-proof `step` on the caller's prover
/// channel, without grinding.
pub(crate) struct ProverRounds<'a> {
    pub(crate) state: &'a mut ProverChannel,
    pub(crate) step: u32,
}

impl<E: ChallengeCodec> SumcheckProverChannel<E> for ProverRounds<'_> {
    fn state_mut(&mut self) -> &mut ProverChannel {
        self.state
    }

    fn sumcheck_site(&self, _invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        response_site(self.step, round, role as u32)
    }

    fn round_challenge(&mut self, _invocation: u32, round: u32) -> Result<E, AkitaError> {
        ext_challenge(
            self.state,
            response_site(self.step, round, SumcheckRole::Challenge as u32),
        )
    }
}
