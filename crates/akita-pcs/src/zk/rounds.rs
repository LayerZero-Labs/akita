//! The response proof's two new sumchecks (Whiteout §4.3): the degree-2
//! product sumcheck `Σ_i U_i W_i` and the degree-3 fused range-and-relation
//! sumcheck `Σ_x A(x)E(x) + B(x)E(x)(E(x) + 1)`, and the clear round
//! contexts they run on. Tables bind their lowest index bit first, as
//! `fold_evals_in_place` does, so challenge `k` is coordinate `k` of the
//! little-endian point.

use akita_error::AkitaError;
use akita_sumcheck::{fold_evals_in_place, SumcheckKernel};
use jolt_algebra::solinas::parallel::*;
use jolt_algebra::{cfg_fold_reduce, Field, Fold};
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
        _claim: F,
    ) -> Result<UnivariatePoly<F>, AkitaError> {
        let evals: [F; 3] = round_evaluations([&self.u, &self.w], |[u, w]| *u * *w);
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn bind_challenge(&mut self, _round: usize, challenge: F) -> Result<(), AkitaError> {
        fold_evals_in_place(&mut self.u, challenge);
        fold_evals_in_place(&mut self.w, challenge);
        Ok(())
    }
}

/// `Σ_x A(x)E(x) + B(x)E(x)(E(x) + 1)` over `ν` variables.
pub(crate) struct FusedKernel<F> {
    pub(crate) a: Vec<F>,
    pub(crate) b: Vec<F>,
    pub(crate) e: Vec<F>,
}

impl<F: Field + Fold> SumcheckKernel<F> for FusedKernel<F> {
    fn num_rounds(&self) -> usize {
        self.e.len().trailing_zeros() as usize
    }

    fn round_polynomial(
        &mut self,
        _round: usize,
        claim: F,
    ) -> Result<UnivariatePoly<F>, AkitaError> {
        #[cfg_attr(not(test), expect(unused_mut))]
        let mut evals: [F; 4] = round_evaluations([&self.a, &self.b, &self.e], |[a, b, e]| {
            *e * (*a + *b * (*e + F::one()))
        });
        // Test-only: a prover of a false statement shifts each round by
        // half its gap to the running claim, so only the verifier's final
        // check can catch it.
        #[cfg(test)]
        {
            let shift = (claim - evals[0] - evals[1]).half();
            for eval in &mut evals {
                *eval += shift;
            }
        }
        #[cfg(not(test))]
        let _ = claim;
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn bind_challenge(&mut self, _round: usize, challenge: F) -> Result<(), AkitaError> {
        for table in [&mut self.a, &mut self.b, &mut self.e] {
            fold_evals_in_place(table, challenge);
        }
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
