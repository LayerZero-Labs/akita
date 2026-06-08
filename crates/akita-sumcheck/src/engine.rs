//! Descriptor-driven sumcheck prover.
//!
//! [`SumcheckEngine`] evaluates a [`SumcheckInstanceDescriptor`](crate::descriptor::SumcheckInstanceDescriptor)
//! directly: it walks the descriptor's weighted sub-claims over multilinear
//! witness and public oracles and emits the standard per-round univariate
//! messages. Any instance can be proven this way; hand-tuned provers in
//! [`crate::fast_path`] must match these round polynomials on the same layout.
//!
//! Each `Source::Opening` resolves to a borrowed [`PolynomialView`]. Each
//! `Source::Public` is either a multilinear view over the cube or a scalar
//! constant. Sub-claim weights resolve through `resolve_challenge`. Term bodies
//! are challenge-free by convention, but challenge factors in a body are still
//! resolved as scalars so the engine handles any summand shape.
//!
//! Each round binds the low bit of the hypercube: oracle tables fold adjacent
//! pairs `(2j, 2j+1) -> j`, matching [`akita_algebra::poly::fold_evals_in_place`].

use akita_algebra::uni_poly::UniPoly;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};
use akita_witness::PolynomialView;

use crate::descriptor::{Source, SumcheckInstanceDescriptor};
use crate::traits::SumcheckInstanceProver;

/// A resolved public oracle: either a multilinear view over the sumcheck
/// variables, or a scalar constant over the boolean hypercube.
pub enum PublicBinding<'a, E> {
    /// A multilinear public, given by its evaluations over the cube.
    Multilinear(PolynomialView<'a, E>),
    /// A constant public value that does not vary over the cube.
    Scalar(E),
}

/// One lowered term: the product of an integer coefficient with all scalar
/// factors (`constant`), times the multilinear oracle factors (`factors`, by
/// index into the engine's oracle list, with multiplicity so `W * W` lists the
/// same oracle twice and contributes degree 2 in the round variable).
#[derive(Debug)]
struct LoweredTerm<E> {
    constant: E,
    factors: Vec<usize>,
}

/// One lowered sub-claim: a Fiat-Shamir `weight` scaling a sum of lowered terms.
#[derive(Debug)]
struct LoweredSubClaim<E> {
    weight: E,
    terms: Vec<LoweredTerm<E>>,
}

/// Sumcheck prover that evaluates a descriptor's summand from witness oracles.
#[derive(Debug)]
pub struct SumcheckEngine<E: FieldCore> {
    num_rounds: usize,
    degree: usize,
    input_claim: E,
    /// Multilinear oracle tables, each `2^(num_rounds - rounds_bound)` long;
    /// every table is folded by the low bit on each ingested challenge.
    oracles: Vec<Vec<E>>,
    subclaims: Vec<LoweredSubClaim<E>>,
}

impl<E: FieldCore + FromPrimitiveInt> SumcheckEngine<E> {
    /// Lower a descriptor into a runnable prover over the supplied oracles.
    ///
    /// `input_claim` is the chained claim this instance proves (the verifier
    /// absorbs it before the rounds). Each `Source` in the summand is resolved
    /// once: openings to a multilinear [`PolynomialView`], publics to a
    /// [`PublicBinding`], and sub-claim weights through `resolve_challenge`.
    /// Repeated openings / multilinear publics share one oracle.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError`] when `num_rounds` is too large for the hypercube
    /// size to fit a `usize`, when a resolved multilinear view's `num_vars` does
    /// not equal `num_rounds`, when a resolver rejects a source, or when a
    /// multi-round summand references no multilinear oracle.
    pub fn new<'a, O, P, C, RO, RP, RC>(
        descriptor: &SumcheckInstanceDescriptor<O, P, C>,
        input_claim: E,
        mut resolve_opening: RO,
        mut resolve_public: RP,
        resolve_challenge: RC,
    ) -> Result<Self, AkitaError>
    where
        E: 'a,
        O: Clone + Eq,
        P: Clone + Eq,
        RO: FnMut(&O) -> Result<PolynomialView<'a, E>, AkitaError>,
        RP: FnMut(&P) -> Result<PublicBinding<'a, E>, AkitaError>,
        RC: Fn(&C) -> Result<E, AkitaError>,
    {
        let num_rounds = descriptor.num_rounds;
        // Bound num_rounds so a later `1 << num_rounds` cannot overflow; the view
        // length check below is the authoritative per-oracle shape gate.
        hypercube_len(num_rounds)?;

        let mut oracles: Vec<Vec<E>> = Vec::new();
        let mut opening_index: Vec<(O, usize)> = Vec::new();
        let mut public_index: Vec<(P, usize)> = Vec::new();

        let mut subclaims = Vec::with_capacity(descriptor.summand.subclaims.len());
        for subclaim in &descriptor.summand.subclaims {
            let weight = match &subclaim.weight {
                Some(challenge) => resolve_challenge(challenge)?,
                None => E::one(),
            };

            let mut terms = Vec::with_capacity(subclaim.body.terms.len());
            for term in &subclaim.body.terms {
                let mut constant = E::from_i64(term.coefficient);
                let mut factors = Vec::with_capacity(term.factors.len());

                for factor in &term.factors {
                    match factor {
                        Source::Opening(opening) => {
                            let idx = match opening_index.iter().find(|(o, _)| o == opening) {
                                Some((_, idx)) => *idx,
                                None => {
                                    let view = resolve_opening(opening)?;
                                    let idx = push_oracle(&mut oracles, view, num_rounds)?;
                                    opening_index.push((opening.clone(), idx));
                                    idx
                                }
                            };
                            factors.push(idx);
                        }
                        Source::Public(public) => {
                            if let Some((_, idx)) = public_index.iter().find(|(p, _)| p == public) {
                                factors.push(*idx);
                                continue;
                            }
                            match resolve_public(public)? {
                                PublicBinding::Multilinear(view) => {
                                    let idx = push_oracle(&mut oracles, view, num_rounds)?;
                                    public_index.push((public.clone(), idx));
                                    factors.push(idx);
                                }
                                PublicBinding::Scalar(value) => {
                                    constant *= value;
                                }
                            }
                        }
                        Source::Challenge(challenge) => {
                            constant *= resolve_challenge(challenge)?;
                        }
                    }
                }

                terms.push(LoweredTerm { constant, factors });
            }

            subclaims.push(LoweredSubClaim { weight, terms });
        }

        if num_rounds > 0 && oracles.is_empty() {
            return Err(AkitaError::InvalidInput(
                "sumcheck engine: a multi-round summand must reference at least one multilinear oracle"
                    .to_string(),
            ));
        }

        Ok(Self {
            num_rounds,
            degree: descriptor.degree,
            input_claim,
            oracles,
            subclaims,
        })
    }
}

impl<E: FieldCore + FromPrimitiveInt> SumcheckInstanceProver<E> for SumcheckEngine<E> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        self.degree
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let half = self.oracles.first().map_or(0, |table| table.len() / 2);

        // g(X) = Σ_subclaim weight · Σ_term constant · Σ_{rest j} Π_factor f_X(j),
        // where f_X(j) = oracle[2j] + X·(oracle[2j+1] - oracle[2j]) is the factor's
        // multilinear value with the low (current-round) bit set to X. Sampling at
        // X = 0..=degree and interpolating recovers the degree-bounded round poly.
        let mut evals = vec![E::zero(); self.degree + 1];
        for (k, eval) in evals.iter_mut().enumerate() {
            let x = E::from_u64(k as u64);
            let mut acc = E::zero();
            for subclaim in &self.subclaims {
                let mut subclaim_sum = E::zero();
                for term in &subclaim.terms {
                    let mut term_sum = E::zero();
                    for j in 0..half {
                        let mut product = term.constant;
                        for &idx in &term.factors {
                            let lo = self.oracles[idx][2 * j];
                            let hi = self.oracles[idx][2 * j + 1];
                            product *= lo + x * (hi - lo);
                        }
                        term_sum += product;
                    }
                    subclaim_sum += term_sum;
                }
                acc += subclaim.weight * subclaim_sum;
            }
            *eval = acc;
        }

        UniPoly::from_evals(&evals)
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        for table in &mut self.oracles {
            fold_low_bit(table, r);
        }
    }
}

/// Copy a multilinear view into a fresh owned oracle table, validating shape.
fn push_oracle<E: FieldCore>(
    oracles: &mut Vec<Vec<E>>,
    view: PolynomialView<'_, E>,
    num_rounds: usize,
) -> Result<usize, AkitaError> {
    if view.num_vars() != num_rounds {
        return Err(AkitaError::InvalidInput(format!(
            "sumcheck engine: oracle has {} vars, expected {num_rounds}",
            view.num_vars()
        )));
    }
    let idx = oracles.len();
    oracles.push(view.evals().to_vec());
    Ok(idx)
}

/// Fold a multilinear table by its low bit: `new[j] = lo + r·(hi - lo)` over the
/// adjacent pair `(2j, 2j+1)`, matching [`akita_algebra::poly::fold_evals_in_place`].
fn fold_low_bit<E: FieldCore>(table: &mut Vec<E>, r: E) {
    let half = table.len() / 2;
    for j in 0..half {
        let lo = table[2 * j];
        let hi = table[2 * j + 1];
        table[j] = lo + r * (hi - lo);
    }
    table.truncate(half);
}

/// `2^num_rounds`, or an error when it overflows `usize`.
fn hypercube_len(num_rounds: usize) -> Result<usize, AkitaError> {
    u32::try_from(num_rounds)
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift))
        .ok_or_else(|| {
            AkitaError::InvalidInput(format!(
                "sumcheck engine: num_rounds {num_rounds} too large: 2^num_rounds overflows usize"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{ClaimSlot, Expr, InstanceKind, SubClaim, Summand, Term};
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;

    // Minimal identifier types local to the test.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum O {
        W,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum P {
        A,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum C {
        Gamma,
    }

    fn f(v: u64) -> F {
        F::from_u64(v)
    }

    /// Multilinear evaluation of a 2-variable table at `(var0 = r0, var1 = r1)`
    /// using the low-bit-first convention (var0 is the low bit).
    fn mle2(table: &[F], r0: F, r1: F) -> F {
        let lo = table[0] + r0 * (table[1] - table[0]);
        let hi = table[2] + r0 * (table[3] - table[2]);
        lo + r1 * (hi - lo)
    }

    fn descriptor(
        weight: Option<C>,
        terms: Vec<Term<O, P, C>>,
        degree: usize,
    ) -> SumcheckInstanceDescriptor<O, P, C> {
        SumcheckInstanceDescriptor {
            label: "test",
            num_rounds: 2,
            degree,
            kind: InstanceKind::Regular,
            input_claim: ClaimSlot(0),
            output_claim: ClaimSlot(1),
            summand: Summand::new(vec![SubClaim::new("only", weight, Expr::new(terms))]),
        }
    }

    fn run_two_rounds(mut engine: SumcheckEngine<F>, claim: F, r0: F, r1: F) -> F {
        // Round 0: g(0)+g(1) must reproduce the input claim.
        let g0 = engine.compute_round_univariate(0, claim);
        assert_eq!(g0.evaluate(&F::zero()) + g0.evaluate(&F::one()), claim);
        let claim1 = g0.evaluate(&r0);
        engine.ingest_challenge(0, r0);

        // Round 1: g(0)+g(1) must reproduce the folded claim.
        let g1 = engine.compute_round_univariate(1, claim1);
        assert_eq!(g1.evaluate(&F::zero()) + g1.evaluate(&F::one()), claim1);
        engine.ingest_challenge(1, r1);

        g1.evaluate(&r1)
    }

    #[test]
    fn product_of_two_multilinears_matches_mle() {
        // summand = W * A (degree 2), W and A 2-variable tables.
        let w = [f(2), f(3), f(5), f(7)];
        let a = [f(11), f(13), f(17), f(19)];
        let claim: F = (0..4).map(|i| w[i] * a[i]).sum();

        let desc = descriptor(
            None,
            vec![Term::new(
                1,
                vec![Source::Opening(O::W), Source::Public(P::A)],
            )],
            2,
        );
        let engine = SumcheckEngine::new(
            &desc,
            claim,
            |_o| PolynomialView::new(2, &w),
            |_p| Ok(PublicBinding::Multilinear(PolynomialView::new(2, &a)?)),
            |_c| Err(AkitaError::InvalidInput("no challenge".to_string())),
        )
        .expect("engine builds");

        let (r0, r1) = (f(9), f(4));
        let final_claim = run_two_rounds(engine, claim, r0, r1);
        assert_eq!(final_claim, mle2(&w, r0, r1) * mle2(&a, r0, r1));
    }

    #[test]
    fn weighted_subclaim_scales_the_whole_sum() {
        // summand = gamma * (W * A); the weight is a sub-claim weight, not a body
        // factor.
        let w = [f(2), f(3), f(5), f(7)];
        let a = [f(11), f(13), f(17), f(19)];
        let gamma = f(6);
        let claim: F = gamma * (0..4).map(|i| w[i] * a[i]).sum::<F>();

        let desc = descriptor(
            Some(C::Gamma),
            vec![Term::new(
                1,
                vec![Source::Opening(O::W), Source::Public(P::A)],
            )],
            2,
        );
        let engine = SumcheckEngine::new(
            &desc,
            claim,
            |_o| PolynomialView::new(2, &w),
            |_p| Ok(PublicBinding::Multilinear(PolynomialView::new(2, &a)?)),
            |c| match c {
                C::Gamma => Ok(gamma),
            },
        )
        .expect("engine builds");

        let (r0, r1) = (f(9), f(4));
        let final_claim = run_two_rounds(engine, claim, r0, r1);
        assert_eq!(final_claim, gamma * mle2(&w, r0, r1) * mle2(&a, r0, r1));
    }

    #[test]
    fn repeated_oracle_factor_raises_degree() {
        // summand = W * W (degree 2 via a repeated oracle); a single shared oracle
        // is folded once and referenced twice.
        let w = [f(2), f(3), f(5), f(7)];
        let claim: F = (0..4).map(|i| w[i] * w[i]).sum();

        let desc = descriptor(
            None,
            vec![Term::new(
                1,
                vec![Source::Opening(O::W), Source::Opening(O::W)],
            )],
            2,
        );
        let engine = SumcheckEngine::new(
            &desc,
            claim,
            |_o| PolynomialView::new(2, &w),
            |_p| Ok(PublicBinding::Scalar(F::one())),
            |_c| Err(AkitaError::InvalidInput("no challenge".to_string())),
        )
        .expect("engine builds");

        // The round poly is genuinely quadratic in X.
        let mut probe = SumcheckEngine::new(
            &desc,
            claim,
            |_o| PolynomialView::new(2, &w),
            |_p| Ok(PublicBinding::Scalar(F::one())),
            |_c| Err(AkitaError::InvalidInput("no challenge".to_string())),
        )
        .expect("engine builds");
        assert_eq!(probe.compute_round_univariate(0, claim).degree(), 2);

        let (r0, r1) = (f(9), f(4));
        let mle = mle2(&w, r0, r1);
        assert_eq!(run_two_rounds(engine, claim, r0, r1), mle * mle);
    }

    #[test]
    fn rejects_view_with_wrong_num_vars() {
        // A 1-variable view for a 2-round descriptor must be rejected, not panic.
        let w = [f(2), f(3)];
        let a = [f(11), f(13), f(17), f(19)];
        let desc = descriptor(
            None,
            vec![Term::new(
                1,
                vec![Source::Opening(O::W), Source::Public(P::A)],
            )],
            2,
        );
        let err = SumcheckEngine::new(
            &desc,
            F::zero(),
            |_o| PolynomialView::new(1, &w),
            |_p| Ok(PublicBinding::Multilinear(PolynomialView::new(2, &a)?)),
            |_c| Err(AkitaError::InvalidInput("no challenge".to_string())),
        )
        .expect_err("mismatched oracle shape must be rejected");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }
}
