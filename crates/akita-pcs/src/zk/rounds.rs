//! The response proof's two new sumchecks (Whiteout §4.3): the degree-2
//! product sumcheck `Σ_i U_i W_i` and the degree-3 fused range-and-relation
//! sumcheck `Σ_x A(x)E(x) + B(x)E(x)(E(x) + 1)`, and the clear round
//! contexts they run on. Tables bind their lowest index bit first, as
//! `fold_evals_in_place` does, so challenge `k` is coordinate `k` of the
//! little-endian point.

use akita_error::AkitaError;
use akita_sumcheck::{fold_evals_in_place, SumcheckKernel};

use super::rows::eq_table;
use jolt_algebra::solinas::parallel::*;
use jolt_algebra::{cfg_fold_reduce, cfg_into_iter, Field, Fold, JoltField};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProverRoundContext, SumcheckSite, VerifierRoundContext};
use jolt_transcript::{Channel, ProverTranscript, SiteId, Sponge, VerifierTranscript};

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

/// The round polynomial through `evals` at `0..N`. Test-only: it is shifted
/// by half its gap to the running claim, so a prover of a false statement
/// still sends consistent rounds and only the verifier's final check can
/// reject it; an honest round has no gap.
fn round_polynomial<F: Field, const N: usize>(
    #[cfg_attr(not(test), expect(unused_mut))] mut evals: [F; N],
    #[cfg_attr(not(test), expect(unused_variables))] claim: F,
) -> UnivariatePoly<F> {
    #[cfg(test)]
    {
        let shift = (claim - evals[0] - evals[1]).half();
        for eval in &mut evals {
            *eval += shift;
        }
    }
    UnivariatePoly::from_evals(&evals)
}

/// `Σ_i U_i W_i` over `a` variables.
pub(crate) struct ProductKernel<F> {
    pub(crate) u: Vec<F>,
    pub(crate) w: Vec<F>,
}

impl<F: Field + Fold> SumcheckKernel<F> for ProductKernel<F> {
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
        fold_evals_in_place(&mut self.u, challenge);
        fold_evals_in_place(&mut self.w, challenge);
        Ok(())
    }
}

/// `Σ_x A(x)E(x) + B(x)E(x)(E(x) + 1)` over `ν` variables, with
/// `B(x) = γ·eq(τ, x) + S(x)`: the dense part held as its point and the
/// product of its bound factors (split eq), the sparse part `S = ζ·eq(τ, ·)`
/// on `I_bin` as its nonzero entries.
pub(crate) struct FusedKernel<F> {
    pub(crate) a: Vec<F>,
    pub(crate) e: FusedE<F>,
    pub(crate) b: SplitB<F>,
}

/// `B = γ·eq(τ, ·) + S`, `S` sparse.
pub(crate) struct SplitB<F> {
    /// `τ`'s coordinates still free, lowest first.
    point: Vec<F>,
    /// `γ·Π eq(τ_i, r_i)` over the bound coordinates.
    scale: F,
    /// `S`'s nonzero entries, by increasing index.
    sparse: Vec<(usize, F)>,
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

/// `E` as committed digits until the first challenge binds it, then over
/// `F`: the first round reads one byte per entry instead of a field element.
pub(crate) enum FusedE<F> {
    Digits(Vec<i8>),
    Field(Vec<F>),
}

impl<F: Field> FusedE<F> {
    /// The value a table bound in every variable holds.
    pub(crate) fn first(&self) -> Option<F> {
        match self {
            Self::Digits(digits) => digits.first().map(|&digit| F::from_i64(i64::from(digit))),
            Self::Field(values) => values.first().copied(),
        }
    }
}

/// `e·(a + b·(e + 1))`, the fused sumcheck's summand.
fn fused_term<F: Field>(a: F, b: F, e: F) -> F {
    e * (a + b * (e + F::one()))
}

impl<F: JoltField + Fold> FusedKernel<F> {
    /// `Σ_i Σ_t` over pairs `i` of `e_t·(a_t + B_t·(e_t + 1))` at
    /// `t = 0..=3`, with `B`'s dense part `scale·eq(τ_k, t)·eq(τ_rest, i)`.
    fn dense_evaluations(&self, e_at: impl Fn(usize, i64) -> F + Sync) -> [F; 4] {
        let tau = self.b.point.first().copied().unwrap_or_default();
        // eq(τ_k, t) = (1 − τ_k) + t·(2τ_k − 1).
        let eq_line: [F; 4] = std::array::from_fn(|t| {
            (F::one() - tau) + F::from_u64(t as u64) * (tau + tau - F::one())
        });
        let (lo, hi, bits) = self.b.rest();
        let mask = (1 << bits) - 1;
        let (a, scale) = (&self.a, self.b.scale);
        cfg_fold_reduce!(
            0..a.len() / 2,
            || [F::zero(); 4],
            |mut sums: [F; 4], i: usize| {
                let b_pair = scale * lo[i & mask] * hi[i >> bits];
                let (a_low, a_slope) = (a[2 * i], a[2 * i + 1] - a[2 * i]);
                let mut a_t = a_low;
                for (t, sum) in sums.iter_mut().enumerate() {
                    let e = e_at(i, t as i64);
                    *sum += fused_term(a_t, b_pair * eq_line[t], e);
                    a_t += a_slope;
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
        self.a.len().trailing_zeros() as usize
    }

    fn round_polynomial(
        &mut self,
        _round: usize,
        claim: F,
    ) -> Result<UnivariatePoly<F>, AkitaError> {
        let line = |low: F, high: F, t: i64| low + F::from_i64(t) * (high - low);
        let (dense, sparse) = match &self.e {
            FusedE::Field(e) => {
                let e_at = |i: usize, t: i64| line(e[2 * i], e[2 * i + 1], t);
                (self.dense_evaluations(e_at), self.sparse_evaluations(e_at))
            }
            FusedE::Digits(digits) => {
                let e_at = |i: usize, t: i64| {
                    let low = i64::from(digits[2 * i]);
                    F::from_i64(low + t * (i64::from(digits[2 * i + 1]) - low))
                };
                (self.dense_evaluations(e_at), self.sparse_evaluations(e_at))
            }
        };
        let evals: [F; 4] = std::array::from_fn(|t| dense[t] + sparse[t]);
        Ok(round_polynomial(evals, claim))
    }

    fn bind_challenge(&mut self, _round: usize, challenge: F) -> Result<(), AkitaError> {
        fold_evals_in_place(&mut self.a, challenge);
        match &mut self.e {
            FusedE::Field(e) => fold_evals_in_place(e, challenge),
            FusedE::Digits(digits) => {
                let folded = cfg_into_iter!(0..digits.len() / 2)
                    .map(|i| {
                        let low = i64::from(digits[2 * i]);
                        F::from_i64(low)
                            + challenge * F::from_i64(i64::from(digits[2 * i + 1]) - low)
                    })
                    .collect();
                self.e = FusedE::Field(folded);
            }
        }
        let b = &mut self.b;
        let tau = b.point.first().copied().unwrap_or_default();
        b.scale *= (F::one() - tau) * (F::one() - challenge) + tau * challenge;
        b.point.remove(0);
        let mut folded: Vec<(usize, F)> = Vec::with_capacity(b.sparse.len());
        for &(index, value) in &b.sparse {
            let weight = if index % 2 == 0 {
                F::one() - challenge
            } else {
                challenge
            };
            match folded.last_mut() {
                Some((pair, sum)) if *pair == index / 2 => *sum += weight * value,
                _ => folded.push((index / 2, weight * value)),
            }
        }
        b.sparse = folded;
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
