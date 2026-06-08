//! Tier-B fast-path registry for sumcheck proving.
//!
//! A [`SumcheckFastPath`] is an optimized prover admitted through a registry.
//! The driver selects a matching fast path or falls back to Tier A
//! ([`SumcheckEngine`]). The hard contract: for any instance a fast path
//! claims to match, its per-round polynomials equal Tier A's, enforced by
//! [`assert_round_polynomial_equivalence`].

use akita_algebra::uni_poly::UniPoly;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};

use crate::descriptor::SumcheckInstanceDescriptor;
use crate::engine::SumcheckEngine;
use crate::traits::SumcheckInstanceProver;

/// Optimized sumcheck prover (Tier B).
///
/// Matches the contract in `specs/akita-sumcheck-unification.md`: the fast
/// path emits the same per-round univariate messages as Tier A for the
/// descriptor it was registered for. Metadata accessors let the standard
/// sumcheck driver run without a parallel descriptor lookup.
pub trait SumcheckFastPath<E: FieldCore>: Send + Sync {
    /// Compute the prover message `g_round(X)` given the previous running claim.
    fn evaluate_round(&mut self, round: usize, previous_claim: E) -> UniPoly<E>;

    /// Fold/bind the current round variable after the verifier challenge `r`.
    fn bind(&mut self, round: usize, r: E);

    /// Optional end-of-protocol hook after the last challenge has been ingested.
    fn finalize(&mut self) {}

    /// Number of rounds (boolean variables bound by this instance).
    fn num_rounds(&self) -> usize;

    /// Maximum allowed degree for any round univariate polynomial.
    fn degree_bound(&self) -> usize;

    /// The initial claimed sum this instance proves.
    fn input_claim(&self) -> E;
}

/// Wrap any [`SumcheckInstanceProver`] as a [`SumcheckFastPath`].
///
/// This is the adapter used to register bespoke kernels such as
/// `AkitaStage2Prover` verbatim: the optimized prover already implements
/// [`SumcheckInstanceProver`], and this wrapper exposes the fast-path surface
/// without reimplementing round logic.
pub struct InstanceProverFastPath<P> {
    prover: P,
}

impl<P> InstanceProverFastPath<P> {
    /// Construct a fast path from an existing sumcheck instance prover.
    pub const fn new(prover: P) -> Self {
        Self { prover }
    }

    /// Consume the wrapper and return the inner prover.
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

impl<P, E> SumcheckFastPath<E> for InstanceProverFastPath<P>
where
    P: SumcheckInstanceProver<E>,
    E: FieldCore,
{
    fn evaluate_round(&mut self, round: usize, previous_claim: E) -> UniPoly<E> {
        self.prover.compute_round_univariate(round, previous_claim)
    }

    fn bind(&mut self, round: usize, r: E) {
        self.prover.ingest_challenge(round, r);
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

impl<P, E> SumcheckInstanceProver<E> for InstanceProverFastPath<P>
where
    P: SumcheckInstanceProver<E>,
    E: FieldCore,
{
    fn num_rounds(&self) -> usize {
        SumcheckFastPath::num_rounds(self)
    }

    fn degree_bound(&self) -> usize {
        SumcheckFastPath::degree_bound(self)
    }

    fn input_claim(&self) -> E {
        SumcheckFastPath::input_claim(self)
    }

    fn compute_round_univariate(&mut self, round: usize, previous_claim: E) -> UniPoly<E> {
        SumcheckFastPath::evaluate_round(self, round, previous_claim)
    }

    fn ingest_challenge(&mut self, round: usize, r: E) {
        SumcheckFastPath::bind(self, round, r);
    }

    fn finalize(&mut self) {
        SumcheckFastPath::finalize(self);
    }
}

/// Unified prover: a resolved fast path or the Tier-A reference engine.
pub enum ResolvedSumcheckProver<E: FieldCore> {
    /// Tier-B optimized prover selected by the registry.
    Fast(Box<dyn SumcheckFastPath<E>>),
    /// Tier-A descriptor-driven reference prover (always correct).
    TierA(SumcheckEngine<E>),
}

impl<E> ResolvedSumcheckProver<E>
where
    E: FieldCore,
{
    /// Whether this resolution selected a fast path rather than Tier A.
    pub const fn is_fast_path(&self) -> bool {
        matches!(self, Self::Fast(_))
    }
}

impl<E> SumcheckInstanceProver<E> for ResolvedSumcheckProver<E>
where
    E: FieldCore + FromPrimitiveInt,
{
    fn num_rounds(&self) -> usize {
        match self {
            Self::Fast(path) => path.num_rounds(),
            Self::TierA(engine) => engine.num_rounds(),
        }
    }

    fn degree_bound(&self) -> usize {
        match self {
            Self::Fast(path) => path.degree_bound(),
            Self::TierA(engine) => engine.degree_bound(),
        }
    }

    fn input_claim(&self) -> E {
        match self {
            Self::Fast(path) => path.input_claim(),
            Self::TierA(engine) => engine.input_claim(),
        }
    }

    fn compute_round_univariate(&mut self, round: usize, previous_claim: E) -> UniPoly<E> {
        match self {
            Self::Fast(path) => path.evaluate_round(round, previous_claim),
            Self::TierA(engine) => engine.compute_round_univariate(round, previous_claim),
        }
    }

    fn ingest_challenge(&mut self, round: usize, r: E) {
        match self {
            Self::Fast(path) => path.bind(round, r),
            Self::TierA(engine) => engine.ingest_challenge(round, r),
        }
    }

    fn finalize(&mut self) {
        match self {
            Self::Fast(path) => path.finalize(),
            Self::TierA(engine) => engine.finalize(),
        }
    }
}

/// Descriptor matcher that can materialize a fast path for a sumcheck instance.
pub trait SumcheckFastPathMatcher<E: FieldCore, O, P, C>: Send + Sync {
    /// Whether this matcher owns the optimized kernel for `descriptor`.
    fn matches(&self, descriptor: &SumcheckInstanceDescriptor<O, P, C>) -> bool;

    /// Build the fast path for `descriptor`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError`] when the descriptor matches but the prover-side
    /// witness layout cannot be constructed.
    fn build(
        &self,
        descriptor: &SumcheckInstanceDescriptor<O, P, C>,
    ) -> Result<Box<dyn SumcheckFastPath<E>>, AkitaError>;
}

/// Ordered registry of fast-path matchers.
///
/// [`SumcheckFastPathRegistry::resolve`] walks matchers in registration order and
/// returns the first successful [`ResolvedSumcheckProver::Fast`] build. On no
/// match, or when every matching build fails, it falls back to the supplied
/// Tier-A engine unchanged.
pub struct SumcheckFastPathRegistry<E: FieldCore, O, P, C> {
    matchers: Vec<Box<dyn SumcheckFastPathMatcher<E, O, P, C>>>,
}

impl<E: FieldCore, O, P, C> Default for SumcheckFastPathRegistry<E, O, P, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: FieldCore, O, P, C> SumcheckFastPathRegistry<E, O, P, C> {
    /// Construct an empty registry (Tier-A fallback only).
    pub fn new() -> Self {
        Self {
            matchers: Vec::new(),
        }
    }

    /// Register a matcher. Earlier registrations take precedence.
    pub fn register<M>(&mut self, matcher: M)
    where
        M: SumcheckFastPathMatcher<E, O, P, C> + 'static,
    {
        self.matchers.push(Box::new(matcher));
    }

    /// Select a fast path for `descriptor`, or return `tier_a` unchanged.
    pub fn resolve(
        &self,
        descriptor: &SumcheckInstanceDescriptor<O, P, C>,
        tier_a: SumcheckEngine<E>,
    ) -> ResolvedSumcheckProver<E> {
        for matcher in &self.matchers {
            if !matcher.matches(descriptor) {
                continue;
            }
            if let Ok(fast) = matcher.build(descriptor) {
                return ResolvedSumcheckProver::Fast(fast);
            }
        }
        ResolvedSumcheckProver::TierA(tier_a)
    }
}

/// Resolve a prover from an optional pre-built fast path and a Tier-A engine.
///
/// When `fast_path` is `Some`, the caller has already selected the optimized
/// kernel (typically after checking a matcher). Otherwise Tier A is used.
pub fn resolve_sumcheck_prover<E: FieldCore>(
    tier_a: SumcheckEngine<E>,
    fast_path: Option<Box<dyn SumcheckFastPath<E>>>,
) -> ResolvedSumcheckProver<E> {
    match fast_path {
        Some(path) => ResolvedSumcheckProver::Fast(path),
        None => ResolvedSumcheckProver::TierA(tier_a),
    }
}

/// Assert that two provers emit identical round polynomials round-by-round.
///
/// Used as the fast-path equivalence gate: Tier B must match Tier A on every
/// round for the same challenge sequence. Returns the final folded claim on
/// success.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidInput`] when metadata disagrees (round count,
/// degree bound, or input claim). Returns [`AkitaError::InvalidProof`] when any
/// round polynomial differs.
pub fn assert_round_polynomial_equivalence<E, L, R, S>(
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
            "input claim mismatch between provers under equivalence test".to_string(),
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
    use akita_field::Prime64Offset59;
    use akita_witness::PolynomialView;

    type F = Prime64Offset59;

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

    impl SumcheckFastPathMatcher<F, O, P, C> for AlwaysMatchMatcher {
        fn matches(&self, _descriptor: &SumcheckInstanceDescriptor<O, P, C>) -> bool {
            true
        }

        fn build(
            &self,
            descriptor: &SumcheckInstanceDescriptor<O, P, C>,
        ) -> Result<Box<dyn SumcheckFastPath<F>>, AkitaError> {
            let w = [f(2), f(3), f(5), f(7)];
            let a = [f(11), f(13), f(17), f(19)];
            let claim: F = (0..4).map(|i| w[i] * a[i]).sum();
            let engine = build_engine(descriptor, claim, &w, &a);
            Ok(Box::new(InstanceProverFastPath::new(engine)))
        }
    }

    #[test]
    fn registry_selects_first_matching_fast_path() {
        let descriptor = sample_descriptor();
        let w = [f(2), f(3), f(5), f(7)];
        let a = [f(11), f(13), f(17), f(19)];
        let claim: F = (0..4).map(|i| w[i] * a[i]).sum();
        let tier_a = build_engine(&descriptor, claim, &w, &a);

        let mut registry = SumcheckFastPathRegistry::new();
        registry.register(AlwaysMatchMatcher);
        let resolved = registry.resolve(&descriptor, tier_a);
        assert!(resolved.is_fast_path());
    }

    #[test]
    fn registry_falls_back_to_tier_a_when_no_matcher() {
        let descriptor = sample_descriptor();
        let w = [f(2), f(3), f(5), f(7)];
        let a = [f(11), f(13), f(17), f(19)];
        let claim: F = (0..4).map(|i| w[i] * a[i]).sum();
        let tier_a = build_engine(&descriptor, claim, &w, &a);

        let registry = SumcheckFastPathRegistry::<F, O, P, C>::new();
        let resolved = registry.resolve(&descriptor, tier_a);
        assert!(!resolved.is_fast_path());
    }

    #[test]
    fn tier_a_matches_itself_round_by_round() {
        let descriptor = sample_descriptor();
        let w = [f(2), f(3), f(5), f(7)];
        let a = [f(11), f(13), f(17), f(19)];
        let claim: F = (0..4).map(|i| w[i] * a[i]).sum();

        let mut left = build_engine(&descriptor, claim, &w, &a);
        let mut right = build_engine(&descriptor, claim, &w, &a);

        assert_round_polynomial_equivalence(&mut left, &mut right, |round| f((round as u64) + 9))
            .expect("two Tier-A engines must agree");
    }

    #[test]
    fn instance_prover_fast_path_delegates_to_inner() {
        let descriptor = sample_descriptor();
        let w = [f(2), f(3), f(5), f(7)];
        let a = [f(11), f(13), f(17), f(19)];
        let claim: F = (0..4).map(|i| w[i] * a[i]).sum();

        let mut direct = build_engine(&descriptor, claim, &w, &a);
        let mut wrapped = InstanceProverFastPath::new(build_engine(&descriptor, claim, &w, &a));

        assert_round_polynomial_equivalence(&mut direct, &mut wrapped, |round| {
            f((round as u64) + 3)
        })
        .expect("wrapper must be transparent");
    }
}
