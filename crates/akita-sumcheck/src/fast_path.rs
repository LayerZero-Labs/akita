//! Registry for optimized sumcheck provers.
//!
//! Most sumcheck instances have a hand-tuned prover (compact witness scans,
//! split-eq folding, and similar) that is much faster than walking the
//! descriptor literally. This module selects such an optimized implementation
//! when one is registered for the instance, and otherwise uses
//! [`SumcheckEngine`], which evaluates the descriptor's summand directly from
//! witness and public oracles.
//!
//! The contract for every optimized prover: on the same witness layout it must
//! emit the same per-round univariate polynomials as [`SumcheckEngine`].
//! [`assert_same_round_polynomials`] checks that round-by-round.

use akita_algebra::uni_poly::UniPoly;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};

use crate::descriptor::SumcheckInstanceDescriptor;
use crate::engine::SumcheckEngine;
use crate::traits::SumcheckInstanceProver;

/// Hand-tuned sumcheck prover for a specific instance family.
///
/// Method names match [`SumcheckInstanceProver`]: compute the round polynomial,
/// ingest the verifier challenge, then optionally finalize. The optimized prover
/// must agree with [`SumcheckEngine`] on every round polynomial for the
/// descriptor it was built for.
pub trait OptimizedSumcheckProver<E: FieldCore>: Send + Sync {
    /// Compute the prover message `g_round(X)` given the previous running claim.
    fn compute_round_univariate(&mut self, round: usize, previous_claim: E) -> UniPoly<E>;

    /// Fold internal state after the verifier challenge `r_round`.
    fn ingest_challenge(&mut self, round: usize, r_round: E);

    /// Optional hook after the last challenge has been ingested.
    fn finalize(&mut self) {}

    /// Number of boolean-hypercube rounds for this instance.
    fn num_rounds(&self) -> usize;

    /// Maximum total degree of any round univariate polynomial.
    fn degree_bound(&self) -> usize;

    /// Initial claimed sum this instance proves.
    fn input_claim(&self) -> E;
}

/// Register an existing [`SumcheckInstanceProver`] as an optimized kernel.
///
/// Stage-specific provers such as `AkitaStage2Prover` already implement the
/// instance trait; this adapter exposes the optimized registry interface
/// without duplicating round logic.
pub struct InstanceProverAdapter<P> {
    prover: P,
}

impl<P> InstanceProverAdapter<P> {
    /// Wrap `prover` for the optimized registry.
    pub const fn new(prover: P) -> Self {
        Self { prover }
    }

    /// Consume the adapter and return the inner prover.
    pub fn into_inner(self) -> P {
        self.prover
    }

    /// Borrow the inner prover.
    pub const fn inner(&self) -> &P {
        &self.prover
    }

    /// Mutably borrow the inner prover.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.prover
    }
}

impl<P, E> OptimizedSumcheckProver<E> for InstanceProverAdapter<P>
where
    P: SumcheckInstanceProver<E>,
    E: FieldCore,
{
    fn compute_round_univariate(&mut self, round: usize, previous_claim: E) -> UniPoly<E> {
        self.prover.compute_round_univariate(round, previous_claim)
    }

    fn ingest_challenge(&mut self, round: usize, r_round: E) {
        self.prover.ingest_challenge(round, r_round);
    }

    fn finalize(&mut self) {
        self.prover.finalize();
    }

    fn num_rounds(&self) -> usize {
        self.prover.num_rounds()
    }

    fn degree_bound(&self) -> usize {
        self.prover.degree_bound()
    }

    fn input_claim(&self) -> E {
        self.prover.input_claim()
    }
}

impl<P, E> SumcheckInstanceProver<E> for InstanceProverAdapter<P>
where
    P: SumcheckInstanceProver<E>,
    E: FieldCore,
{
    fn num_rounds(&self) -> usize {
        OptimizedSumcheckProver::num_rounds(self)
    }

    fn degree_bound(&self) -> usize {
        OptimizedSumcheckProver::degree_bound(self)
    }

    fn input_claim(&self) -> E {
        OptimizedSumcheckProver::input_claim(self)
    }

    fn compute_round_univariate(&mut self, round: usize, previous_claim: E) -> UniPoly<E> {
        OptimizedSumcheckProver::compute_round_univariate(self, round, previous_claim)
    }

    fn ingest_challenge(&mut self, round: usize, r_round: E) {
        OptimizedSumcheckProver::ingest_challenge(self, round, r_round);
    }

    fn finalize(&mut self) {
        OptimizedSumcheckProver::finalize(self);
    }
}

/// Result of selecting a prover for one sumcheck instance.
pub enum ResolvedSumcheckProver<E: FieldCore> {
    /// Hand-tuned kernel chosen by the registry.
    Optimized(Box<dyn OptimizedSumcheckProver<E>>),
    /// Descriptor-driven [`SumcheckEngine`] (always available as fallback).
    DescriptorEngine(SumcheckEngine<E>),
}

impl<E> ResolvedSumcheckProver<E>
where
    E: FieldCore,
{
    /// Whether the registry supplied an optimized kernel instead of the engine.
    pub const fn uses_optimized_prover(&self) -> bool {
        matches!(self, Self::Optimized(_))
    }
}

impl<E> SumcheckInstanceProver<E> for ResolvedSumcheckProver<E>
where
    E: FieldCore + FromPrimitiveInt,
{
    fn num_rounds(&self) -> usize {
        match self {
            Self::Optimized(prover) => prover.num_rounds(),
            Self::DescriptorEngine(engine) => engine.num_rounds(),
        }
    }

    fn degree_bound(&self) -> usize {
        match self {
            Self::Optimized(prover) => prover.degree_bound(),
            Self::DescriptorEngine(engine) => engine.degree_bound(),
        }
    }

    fn input_claim(&self) -> E {
        match self {
            Self::Optimized(prover) => prover.input_claim(),
            Self::DescriptorEngine(engine) => engine.input_claim(),
        }
    }

    fn compute_round_univariate(&mut self, round: usize, previous_claim: E) -> UniPoly<E> {
        match self {
            Self::Optimized(prover) => prover.compute_round_univariate(round, previous_claim),
            Self::DescriptorEngine(engine) => {
                engine.compute_round_univariate(round, previous_claim)
            }
        }
    }

    fn ingest_challenge(&mut self, round: usize, r_round: E) {
        match self {
            Self::Optimized(prover) => prover.ingest_challenge(round, r_round),
            Self::DescriptorEngine(engine) => engine.ingest_challenge(round, r_round),
        }
    }

    fn finalize(&mut self) {
        match self {
            Self::Optimized(prover) => prover.finalize(),
            Self::DescriptorEngine(engine) => engine.finalize(),
        }
    }
}

/// Builds an optimized prover when a descriptor matches a known instance family.
pub trait OptimizedProverMatcher<E: FieldCore, O, P, C>: Send + Sync {
    /// Whether this matcher can build the optimized kernel for `descriptor`.
    fn matches(&self, descriptor: &SumcheckInstanceDescriptor<O, P, C>) -> bool;

    /// Materialize the optimized prover for `descriptor`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError`] when the descriptor matches but witness layout or
    /// oracle construction fails.
    fn build(
        &self,
        descriptor: &SumcheckInstanceDescriptor<O, P, C>,
    ) -> Result<Box<dyn OptimizedSumcheckProver<E>>, AkitaError>;
}

/// Ordered list of matchers; first successful build wins.
///
/// [`OptimizedProverRegistry::resolve`] tries each registered matcher in order.
/// If none match, or every matching build fails, it returns the supplied
/// [`SumcheckEngine`] unchanged.
pub struct OptimizedProverRegistry<E: FieldCore, O, P, C> {
    matchers: Vec<Box<dyn OptimizedProverMatcher<E, O, P, C>>>,
}

impl<E: FieldCore, O, P, C> Default for OptimizedProverRegistry<E, O, P, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: FieldCore, O, P, C> OptimizedProverRegistry<E, O, P, C> {
    /// Empty registry: every instance falls back to [`SumcheckEngine`].
    pub fn new() -> Self {
        Self {
            matchers: Vec::new(),
        }
    }

    /// Register a matcher. Earlier registrations take precedence.
    pub fn register<M>(&mut self, matcher: M)
    where
        M: OptimizedProverMatcher<E, O, P, C> + 'static,
    {
        self.matchers.push(Box::new(matcher));
    }

    /// Select an optimized prover for `descriptor`, or return `descriptor_engine`.
    pub fn resolve(
        &self,
        descriptor: &SumcheckInstanceDescriptor<O, P, C>,
        descriptor_engine: SumcheckEngine<E>,
    ) -> ResolvedSumcheckProver<E> {
        for matcher in &self.matchers {
            if !matcher.matches(descriptor) {
                continue;
            }
            if let Ok(optimized) = matcher.build(descriptor) {
                return ResolvedSumcheckProver::Optimized(optimized);
            }
        }
        ResolvedSumcheckProver::DescriptorEngine(descriptor_engine)
    }
}

/// Select optimized prover when provided, otherwise use the descriptor engine.
pub fn resolve_sumcheck_prover<E: FieldCore>(
    descriptor_engine: SumcheckEngine<E>,
    optimized: Option<Box<dyn OptimizedSumcheckProver<E>>>,
) -> ResolvedSumcheckProver<E> {
    match optimized {
        Some(prover) => ResolvedSumcheckProver::Optimized(prover),
        None => ResolvedSumcheckProver::DescriptorEngine(descriptor_engine),
    }
}

/// Check that two provers emit identical round polynomials, round by round.
///
/// Returns the final folded claim on success. Used to verify that an optimized
/// kernel matches [`SumcheckEngine`] on a fixed witness layout.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidInput`] when round count, degree bound, or
/// input claim disagree. Returns [`AkitaError::InvalidProof`] when any round
/// polynomial differs.
pub fn assert_same_round_polynomials<E, L, R, S>(
    left: &mut L,
    right: &mut R,
    mut sample_challenge: S,
) -> Result<E, AkitaError>
where
    E: FieldCore,
    L: SumcheckInstanceProver<E>,
    R: SumcheckInstanceProver<E>,
    S: FnMut(usize) -> E,
{
    if left.num_rounds() != right.num_rounds() {
        return Err(AkitaError::InvalidInput(format!(
            "round count mismatch: left {} vs right {}",
            left.num_rounds(),
            right.num_rounds()
        )));
    }
    if left.degree_bound() != right.degree_bound() {
        return Err(AkitaError::InvalidInput(format!(
            "degree bound mismatch: left {} vs right {}",
            left.degree_bound(),
            right.degree_bound()
        )));
    }
    if left.input_claim() != right.input_claim() {
        return Err(AkitaError::InvalidInput(
            "input claim mismatch between provers in round-polynomial check".to_string(),
        ));
    }

    let num_rounds = left.num_rounds();
    let mut claim = left.input_claim();

    for round in 0..num_rounds {
        let left_poly = left.compute_round_univariate(round, claim);
        let right_poly = right.compute_round_univariate(round, claim);
        if left_poly != right_poly {
            return Err(AkitaError::InvalidProof);
        }

        let r = sample_challenge(round);
        claim = left_poly.evaluate(&r);
        if claim != right_poly.evaluate(&r) {
            return Err(AkitaError::InvalidProof);
        }

        left.ingest_challenge(round, r);
        right.ingest_challenge(round, r);
    }

    left.finalize();
    right.finalize();
    Ok(claim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{ClaimSlot, Expr, InstanceKind, Source, SubClaim, Summand, Term};
    use akita_field::Prime128OffsetA7F7;
    use akita_witness::PolynomialView;

    type F = Prime128OffsetA7F7;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum O {
        W,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum P {
        A,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum C {}

    fn f(v: u64) -> F {
        F::from_u64(v)
    }

    fn sample_descriptor() -> SumcheckInstanceDescriptor<O, P, C> {
        SumcheckInstanceDescriptor {
            label: "registry-test",
            num_rounds: 2,
            degree: 2,
            kind: InstanceKind::Regular,
            input_claim: ClaimSlot(0),
            output_claim: ClaimSlot(1),
            summand: Summand::new(vec![SubClaim::new(
                "only",
                None,
                Expr::new(vec![Term::new(
                    1,
                    vec![Source::Opening(O::W), Source::Public(P::A)],
                )]),
            )]),
        }
    }

    fn build_engine(
        descriptor: &SumcheckInstanceDescriptor<O, P, C>,
        claim: F,
        w: &[F; 4],
        a: &[F; 4],
    ) -> SumcheckEngine<F> {
        SumcheckEngine::new(
            descriptor,
            claim,
            |_o| PolynomialView::new(2, w),
            |_p| {
                Ok(crate::engine::PublicBinding::Multilinear(
                    PolynomialView::new(2, a)?,
                ))
            },
            |_c| Err(AkitaError::InvalidInput("no challenge".to_string())),
        )
        .expect("engine builds")
    }

    struct AlwaysMatchMatcher;

    impl OptimizedProverMatcher<F, O, P, C> for AlwaysMatchMatcher {
        fn matches(&self, _descriptor: &SumcheckInstanceDescriptor<O, P, C>) -> bool {
            true
        }

        fn build(
            &self,
            descriptor: &SumcheckInstanceDescriptor<O, P, C>,
        ) -> Result<Box<dyn OptimizedSumcheckProver<F>>, AkitaError> {
            let w = [f(2), f(3), f(5), f(7)];
            let a = [f(11), f(13), f(17), f(19)];
            let claim: F = (0..4).map(|i| w[i] * a[i]).sum();
            let engine = build_engine(descriptor, claim, &w, &a);
            Ok(Box::new(InstanceProverAdapter::new(engine)))
        }
    }

    #[test]
    fn registry_selects_first_matching_optimized_prover() {
        let descriptor = sample_descriptor();
        let w = [f(2), f(3), f(5), f(7)];
        let a = [f(11), f(13), f(17), f(19)];
        let claim: F = (0..4).map(|i| w[i] * a[i]).sum();
        let descriptor_engine = build_engine(&descriptor, claim, &w, &a);

        let mut registry = OptimizedProverRegistry::new();
        registry.register(AlwaysMatchMatcher);
        let resolved = registry.resolve(&descriptor, descriptor_engine);
        assert!(resolved.uses_optimized_prover());
    }

    #[test]
    fn registry_falls_back_to_descriptor_engine_when_no_matcher() {
        let descriptor = sample_descriptor();
        let w = [f(2), f(3), f(5), f(7)];
        let a = [f(11), f(13), f(17), f(19)];
        let claim: F = (0..4).map(|i| w[i] * a[i]).sum();
        let descriptor_engine = build_engine(&descriptor, claim, &w, &a);

        let registry = OptimizedProverRegistry::<F, O, P, C>::new();
        let resolved = registry.resolve(&descriptor, descriptor_engine);
        assert!(!resolved.uses_optimized_prover());
    }

    #[test]
    fn descriptor_engine_agrees_with_itself_round_by_round() {
        let descriptor = sample_descriptor();
        let w = [f(2), f(3), f(5), f(7)];
        let a = [f(11), f(13), f(17), f(19)];
        let claim: F = (0..4).map(|i| w[i] * a[i]).sum();

        let mut left = build_engine(&descriptor, claim, &w, &a);
        let mut right = build_engine(&descriptor, claim, &w, &a);

        assert_same_round_polynomials(&mut left, &mut right, |round| f((round as u64) + 9))
            .expect("two descriptor engines must agree");
    }

    #[test]
    fn instance_prover_adapter_delegates_to_inner() {
        let descriptor = sample_descriptor();
        let w = [f(2), f(3), f(5), f(7)];
        let a = [f(11), f(13), f(17), f(19)];
        let claim: F = (0..4).map(|i| w[i] * a[i]).sum();

        let mut direct = build_engine(&descriptor, claim, &w, &a);
        let mut wrapped = InstanceProverAdapter::new(build_engine(&descriptor, claim, &w, &a));

        assert_same_round_polynomials(&mut direct, &mut wrapped, |round| f((round as u64) + 3))
            .expect("adapter must be transparent");
    }
}
