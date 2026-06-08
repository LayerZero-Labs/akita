//! Declarative sumcheck descriptor algebra.
//!
//! A sumcheck instance's round-polynomial summand is a sum of products over
//! typed sources (openings, challenges, publics). The same descriptor is
//! consumed by both sides of the protocol: the prover resolves an opening to a
//! witness view, the verifier resolves it to a claimed evaluation. Keeping one
//! object for both sides means the verifier's expected-output equation and the
//! prover's round polynomial cannot drift apart.
//!
//! This module owns the generic, protocol-independent algebra. It is generic
//! over the identifier types and names no protocol-specific identifier or
//! equation; the concrete `Akita*Id` types and the per-stage formula
//! constructors live in `akita-protocol`.
//!
//! All sumcheck rounds here are over the boolean hypercube: there is no domain
//! field and no univariate-skip support. The evaluator
//! ([`Expr::try_evaluate`]) is fallible and panic-free so it can sit inside the
//! verifier no-panic boundary.

use akita_field::{AkitaError, RingCore};

/// A typed leaf of a sumcheck summand.
///
/// Generic over the protocol's identifier types `O`/`P`/`C` so the same algebra
/// serves both the prover and the verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source<O, P, C> {
    /// An MLE source. The prover resolves it to a witness view; the verifier
    /// resolves it to a claimed evaluation at the round point.
    Opening(O),
    /// A Fiat-Shamir scalar (a batching coefficient, a gamma power, ...).
    Challenge(C),
    /// A public, verifier-evaluable weight (a relation row, a trace weight, a
    /// range coefficient, an eq point, ...).
    Public(P),
}

/// A single monomial `coefficient * product(factors)`.
///
/// An empty `factors` list denotes the bare constant `coefficient`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Term<F, O, P, C> {
    /// Scalar multiplier applied to the product of `factors`.
    pub coefficient: F,
    /// Ordered factors whose product forms the monomial body.
    pub factors: Vec<Source<O, P, C>>,
}

impl<F, O, P, C> Term<F, O, P, C> {
    /// Construct a term from a coefficient and its ordered factors.
    pub fn new(coefficient: F, factors: Vec<Source<O, P, C>>) -> Self {
        Self {
            coefficient,
            factors,
        }
    }
}

/// A sum-of-products expression: the summand `g(x)` of a sumcheck instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expr<F, O, P, C> {
    /// Terms summed to form the expression.
    pub terms: Vec<Term<F, O, P, C>>,
}

/// Whether a sumcheck instance is batchable and which wire format it uses.
///
/// This is the proof-format / batchability axis. It is orthogonal to the
/// prover-compute axis (Gruen split-eq, compact-integer scan, ...): Gruen
/// split-eq is a compute optimization that an [`InstanceKind::EqFactored`]
/// instance keeps even when batching forces the regular wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstanceKind {
    /// Arbitrarily batchable; serializes in the regular compressed format.
    Regular,
    /// Not batchable. The eq-factored format (inner `q` with its linear term
    /// omitted) is valid only when the instance is proven standalone; once it
    /// is batched with any other instance it falls back to the regular format.
    EqFactored,
}

/// Identifier for a chained sumcheck claim (the split-sumcheck handoff).
///
/// An instance consumes the claim named by its `input_claim` slot and produces
/// the claim named by its `output_claim` slot; stages chain by matching one
/// instance's output slot to the next instance's input slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClaimSlot(pub usize);

/// A fully declarative sumcheck instance.
///
/// The same descriptor drives the verifier's expected-output equation (via
/// [`Self::try_evaluate`]) and the prover's round-polynomial computation; only
/// the resolution of each [`Source`] differs between the two sides.
///
/// Generic over the field `F` and the identifier types `O`/`P`/`C`. The
/// verifier instantiates `F` with the evaluation (extension) field so the
/// declared coefficients lift into the field that evaluation happens in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckInstanceDescriptor<F, O, P, C> {
    /// Human-readable diagnostic label. Diagnostics only: it is not bound into
    /// the transcript.
    pub label: &'static str,
    /// Number of boolean-hypercube rounds.
    pub num_rounds: usize,
    /// Total degree of `poly` in the round variable.
    pub degree: usize,
    /// Batchability / wire-format class.
    pub kind: InstanceKind,
    /// Chained input claim this instance consumes.
    pub input_claim: ClaimSlot,
    /// Chained output claim this instance produces.
    pub output_claim: ClaimSlot,
    /// The summand `g(x)` for this instance.
    pub poly: Expr<F, O, P, C>,
}

impl<F, O, P, C> Expr<F, O, P, C> {
    /// Construct an expression from its terms.
    pub fn new(terms: Vec<Term<F, O, P, C>>) -> Self {
        Self { terms }
    }
}

impl<F, O, P, C> Expr<F, O, P, C>
where
    F: RingCore,
{
    /// Evaluate the expression at a resolved point, fallibly and panic-free.
    ///
    /// Each resolver maps one identifier to its value in the evaluation field
    /// `F`: `resolve_opening` to a claimed/computed MLE evaluation,
    /// `resolve_challenge` to a Fiat-Shamir scalar, `resolve_public` to a
    /// verifier-evaluable public weight. A malformed, unknown, or
    /// dimension-mismatched source is reported by the resolver as `Err`, which
    /// short-circuits evaluation: this evaluator never panics, so it is safe on
    /// verifier-reachable paths.
    ///
    /// The result is `sum_terms coefficient * product_factors resolve(factor)`.
    /// A term with no factors contributes its bare `coefficient`.
    pub fn try_evaluate<RO, RC, RP>(
        &self,
        resolve_opening: RO,
        resolve_challenge: RC,
        resolve_public: RP,
    ) -> Result<F, AkitaError>
    where
        RO: Fn(&O) -> Result<F, AkitaError>,
        RC: Fn(&C) -> Result<F, AkitaError>,
        RP: Fn(&P) -> Result<F, AkitaError>,
    {
        let mut acc = F::zero();
        for term in &self.terms {
            let mut product = term.coefficient;
            for factor in &term.factors {
                let value = match factor {
                    Source::Opening(opening) => resolve_opening(opening)?,
                    Source::Challenge(challenge) => resolve_challenge(challenge)?,
                    Source::Public(public) => resolve_public(public)?,
                };
                product *= value;
            }
            acc += product;
        }
        Ok(acc)
    }
}

impl<F, O, P, C> SumcheckInstanceDescriptor<F, O, P, C>
where
    F: RingCore,
{
    /// Evaluate this instance's summand at a resolved point.
    ///
    /// This is the generic verifier descriptor-eval helper: a verifier computes
    /// its `expected_output_claim` by calling this with resolvers that close
    /// over the round challenges. It forwards to [`Expr::try_evaluate`] and is
    /// likewise fallible and panic-free.
    pub fn try_evaluate<RO, RC, RP>(
        &self,
        resolve_opening: RO,
        resolve_challenge: RC,
        resolve_public: RP,
    ) -> Result<F, AkitaError>
    where
        RO: Fn(&O) -> Result<F, AkitaError>,
        RC: Fn(&C) -> Result<F, AkitaError>,
        RP: Fn(&P) -> Result<F, AkitaError>,
    {
        self.poly
            .try_evaluate(resolve_opening, resolve_challenge, resolve_public)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime64Offset59;

    type F = Prime64Offset59;

    // Minimal identifier types local to the test so the generic algebra is
    // exercised without naming any protocol-specific identifier.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum O {
        W,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum P {
        A,
        B,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum C {
        Gamma,
    }

    fn f(v: u64) -> F {
        F::from_u64(v)
    }

    #[test]
    fn try_evaluate_sums_products_of_resolved_factors() {
        // gamma * A * W + 2 * B
        let expr: Expr<F, O, P, C> = Expr::new(vec![
            Term::new(
                F::one(),
                vec![
                    Source::Challenge(C::Gamma),
                    Source::Public(P::A),
                    Source::Opening(O::W),
                ],
            ),
            Term::new(f(2), vec![Source::Public(P::B)]),
        ]);

        let value = expr
            .try_evaluate(
                |o| match o {
                    O::W => Ok(f(5)),
                },
                |c| match c {
                    C::Gamma => Ok(f(3)),
                },
                |p| match p {
                    P::A => Ok(f(7)),
                    P::B => Ok(f(11)),
                },
            )
            .expect("all sources resolve");

        // 3 * 7 * 5 + 2 * 11 = 105 + 22 = 127
        assert_eq!(value, f(127));
    }

    #[test]
    fn try_evaluate_treats_empty_factor_list_as_bare_coefficient() {
        let expr: Expr<F, O, P, C> = Expr::new(vec![Term::new(f(9), Vec::new())]);
        let value = expr
            .try_evaluate(|_| Ok(f(0)), |_| Ok(f(0)), |_| Ok(f(0)))
            .expect("constant term");
        assert_eq!(value, f(9));
    }

    #[test]
    fn try_evaluate_of_empty_expr_is_zero() {
        let expr: Expr<F, O, P, C> = Expr::new(Vec::new());
        let value = expr
            .try_evaluate(|_| Ok(f(1)), |_| Ok(f(1)), |_| Ok(f(1)))
            .expect("empty expr");
        assert_eq!(value, F::zero());
    }

    #[test]
    fn try_evaluate_propagates_a_malformed_source_as_error() {
        // The opening resolver rejects the source instead of panicking.
        let expr: Expr<F, O, P, C> =
            Expr::new(vec![Term::new(F::one(), vec![Source::Opening(O::W)])]);
        let err = expr
            .try_evaluate(
                |_o| Err(AkitaError::InvalidProof),
                |_c| Ok(f(1)),
                |_p| Ok(f(1)),
            )
            .expect_err("malformed opening must be rejected, not panic");
        assert!(matches!(err, AkitaError::InvalidProof));
    }

    #[test]
    fn try_evaluate_short_circuits_before_evaluating_later_terms() {
        // A later term references a public the resolver rejects; the error must
        // surface even though the first term is fine.
        let expr: Expr<F, O, P, C> = Expr::new(vec![
            Term::new(F::one(), vec![Source::Opening(O::W)]),
            Term::new(F::one(), vec![Source::Public(P::A)]),
        ]);
        let err = expr
            .try_evaluate(
                |_o| Ok(f(4)),
                |_c| Ok(f(1)),
                |p| match p {
                    P::A => Err(AkitaError::InvalidInput("unknown public".to_string())),
                    P::B => Ok(f(1)),
                },
            )
            .expect_err("malformed public must be rejected");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }
}
