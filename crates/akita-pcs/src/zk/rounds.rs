//! The response proof's two new sumchecks (Whiteout §4.3): the degree-2
//! product sumcheck `Σ_i U_i W_i` and the degree-3 fused range-and-relation
//! sumcheck `Σ_x A(x)E(x) + B(x)E(x)(E(x) + 1)`, and the clear round
//! contexts they run on. Tables bind their lowest index bit first, as
//! `fold_evals_in_place` does, so challenge `k` is coordinate `k` of the
//! little-endian point.

use akita_error::AkitaError;
use akita_sumcheck::SumcheckKernel;

use zeroize::{Zeroize, Zeroizing};

use super::rows::eq_table;
use jolt_algebra::solinas::parallel::*;
use jolt_algebra::{cfg_fold_reduce, cfg_into_iter, lift_ct, Field, Fold, JoltField};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProverRoundContext, SumcheckSite, VerifierRoundContext};
use jolt_transcript::{Channel, ProverTranscript, SiteId, Sponge, VerifierTranscript};

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
}

impl<F: Field + Fold + Send> SumcheckKernel<F> for ProductKernel<F> {
    fn num_rounds(&self) -> usize {
        self.u.len().trailing_zeros() as usize
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
/// `A` and `E` share `A`'s buffer and fold in place ([`fold_in_place`]):
/// `A` in `[0, len)` and, once the first challenge binds `e`'s digits, `E`
/// in `[half, half + len)`, where `half` is half the first `A`. No table is
/// allocated after the kernel is built, and the buffer, which holds `E`, is
/// wiped when freed.
///
/// `E` is constant time in `e`: every read is at a public index, and the
/// digits enter `F` through [`lift_ct`], which does not branch on their
/// sign. The first round reads one byte per entry instead of a field
/// element.
pub(crate) struct FusedKernel<F: Copy + Send> {
    tables: SecretTable<F>,
    /// The length of `A`, and of `E` once over `F`.
    len: usize,
    /// `e`'s committed digits, until the first challenge binds them.
    digits: Option<Zeroizing<Vec<i8>>>,
    b: SplitB<F>,
}

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

impl<F: JoltField + Fold> FusedKernel<F> {
    /// The kernel of the row weights `a`, `B` and `e`'s committed digits,
    /// one per weight.
    pub(crate) fn new(
        a: Vec<F>,
        b: SplitB<F>,
        digits: Zeroizing<Vec<i8>>,
    ) -> Result<Self, AkitaError> {
        if digits.len() != a.len() {
            return Err(AkitaError::Internal(
                "the fused sumcheck's tables differ in length".into(),
            ));
        }
        Ok(Self {
            len: a.len(),
            tables: SecretTable::wrap(a),
            digits: Some(digits),
            b,
        })
    }

    /// Where `E` starts in the shared buffer.
    fn half(&self) -> usize {
        self.tables.len() / 2
    }

    /// The value `E` holds once bound in every variable.
    pub(crate) fn e_first(&self) -> Option<F> {
        match &self.digits {
            Some(digits) => digits.first().map(|&digit| lift_ct(i64::from(digit))),
            None => self.tables.get(self.half()).copied(),
        }
    }

    /// `Σ_i Σ_t` over pairs `i` of `e_t·(a_t + B_t·(e_t + 1))` at
    /// `t = 0..=3`, with `B`'s dense part `scale·eq(τ_k, t)·eq(τ_rest, i)`
    /// and `e_pair(i)` the pair's `(e_0, e_1 − e_0)`.
    fn dense_evaluations(&self, a: &[F], e_pair: impl Fn(usize) -> (F, F) + Sync) -> [F; 4] {
        let tau = self.b.point.first().copied().unwrap_or_default();
        let (lo, hi, bits) = self.b.rest();
        // eq(τ_k, t)·scale = scale·(1 − τ_k) + t·scale·(2τ_k − 1), with the
        // scale folded into the high half once per round.
        let (start, step) = (F::one() - tau, tau + tau - F::one());
        let hi: Vec<F> = hi.iter().map(|value| *value * self.b.scale).collect();
        let mask = (1 << bits) - 1;
        cfg_fold_reduce!(
            0..a.len() / 2,
            || [F::zero(); 4],
            |mut sums: [F; 4], i: usize| {
                let weight = lo[i & mask] * hi[i >> bits];
                let (mut b_t, b_slope) = (weight * start, weight * step);
                let (mut a_t, a_slope) = (a[2 * i], a[2 * i + 1] - a[2 * i]);
                let (mut e_t, e_slope) = e_pair(i);
                for sum in &mut sums {
                    *sum += fused_term(a_t, b_t, e_t);
                    a_t += a_slope;
                    b_t += b_slope;
                    e_t += e_slope;
                }
                sums
            },
            |mut x: [F; 4], y: [F; 4]| {
                for (x, y) in x.iter_mut().zip(y) {
                    *x += y;
                }
                x
            }
        )
    }

    /// `Σ_t S_t·e_t(e_t + 1)` over the pairs where `S` is nonzero.
    fn sparse_evaluations(&self, e_at: impl Fn(usize, i64) -> F) -> [F; 4] {
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
            for (t, sum) in sums.iter_mut().enumerate() {
                let s = low + F::from_u64(t as u64) * (high - low);
                let e = e_at(pair, t as i64);
                *sum += s * e * (e + F::one());
            }
        }
        sums
    }
}

impl<F: JoltField + Fold> SumcheckKernel<F> for FusedKernel<F> {
    fn num_rounds(&self) -> usize {
        self.len.trailing_zeros() as usize
    }

    fn round_polynomial(
        &mut self,
        _round: usize,
        claim: F,
    ) -> Result<UnivariatePoly<F>, AkitaError> {
        let line = |(low, slope): (F, F), t: i64| low + F::from_i64(t) * slope;
        let a = &self.tables[..self.len];
        let (dense, sparse) = match &self.digits {
            None => {
                let e = &self.tables[self.half()..self.half() + self.len];
                let e_pair = |i: usize| (e[2 * i], e[2 * i + 1] - e[2 * i]);
                (
                    self.dense_evaluations(a, e_pair),
                    self.sparse_evaluations(|i, t| line(e_pair(i), t)),
                )
            }
            Some(digits) => {
                let e_pair = |i: usize| {
                    let low = i64::from(digits[2 * i]);
                    (
                        lift_ct::<F>(low),
                        lift_ct::<F>(i64::from(digits[2 * i + 1]) - low),
                    )
                };
                (
                    self.dense_evaluations(a, e_pair),
                    self.sparse_evaluations(|i, t| line(e_pair(i), t)),
                )
            }
        };
        let evals: [F; 4] = std::array::from_fn(|t| dense[t] + sparse[t]);
        Ok(round_polynomial(evals, claim))
    }

    fn bind_challenge(&mut self, _round: usize, challenge: F) -> Result<(), AkitaError> {
        let (half, len) = (self.half(), self.len);
        let tables = &mut self.tables.0;
        fold_in_place(&mut tables[..len], challenge);
        match self.digits.take() {
            None => fold_in_place(&mut tables[half..half + len], challenge),
            // `E`'s first table over `F` goes to the half of the buffer the
            // fold of `A` above left unread.
            Some(digits) => {
                let ctx = F::precompute(challenge);
                let digit =
                    |index: usize| lift_ct::<F>(digits.get(index).copied().map_or(0, i64::from));
                fill(&mut tables[half..half + len.div_ceil(2)], |i| {
                    F::fold_one(&ctx, digit(2 * i), digit(2 * i + 1))
                });
            }
        }
        self.len = len.div_ceil(2);
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

/// Clear rounds on the caller's transcript: no sites and no grinding.
pub(crate) struct ClearProverRounds<'a, H: Sponge>(pub(crate) &'a mut ProverTranscript<H>);

impl<F: Field, H: Sponge> ProverRoundContext<F> for ClearProverRounds<'_, H> {
    type Channel = ProverTranscript<H>;
    type Error = AkitaError;

    fn channel_mut(&mut self) -> &mut ProverTranscript<H> {
        self.0
    }

    fn site(&self, _site: SumcheckSite) -> Option<SiteId> {
        None
    }

    fn challenge(&mut self, _site: SumcheckSite) -> Result<F, AkitaError> {
        Ok(self.0.challenge())
    }
}

/// The verifier's [`ClearProverRounds`].
pub(crate) struct ClearVerifierRounds<'a, 'proof, H: Sponge>(
    pub(crate) &'a mut VerifierTranscript<'proof, H>,
);

impl<'proof, F: Field, H: Sponge> VerifierRoundContext<F> for ClearVerifierRounds<'_, 'proof, H> {
    type Channel = VerifierTranscript<'proof, H>;
    type Error = AkitaError;

    fn channel_mut(&mut self) -> &mut VerifierTranscript<'proof, H> {
        self.0
    }

    fn site(&self, _site: SumcheckSite) -> Option<SiteId> {
        None
    }

    fn challenge(&mut self, _site: SumcheckSite) -> Result<F, AkitaError> {
        Ok(self.0.challenge())
    }
}
