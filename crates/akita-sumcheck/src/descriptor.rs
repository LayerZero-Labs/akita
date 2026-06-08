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
//!
//! The descriptor structure is field-free: a term's `coefficient` is a small
//! integer, and the *evaluation* field is chosen at [`Expr::try_evaluate`]
//! time. This matches Akita's field reality, where base-field witness/setup
//! coefficients and extension-field weights only meet inside each
//! [`Source`]-resolver (the `base * extension` `mul_base` paths), while the
//! sum-of-products combination the descriptor describes is performed in the
//! single evaluation field the resolvers all return.

use akita_field::{AkitaError, FromPrimitiveInt, RingCore};

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
/// `coefficient` is a small integer lifted into the evaluation field at
/// [`Expr::try_evaluate`] time (every protocol coefficient is a structural
/// constant such as `1` or `-1`; field-valued weights are carried by
/// [`Source`] factors, not the coefficient). An empty `factors` list denotes
/// the bare constant `coefficient`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Term<O, P, C> {
    /// Integer multiplier applied to the product of `factors`.
    pub coefficient: i64,
    /// Ordered factors whose product forms the monomial body.
    pub factors: Vec<Source<O, P, C>>,
}

impl<O, P, C> Term<O, P, C> {
    /// Construct a term from a coefficient and its ordered factors.
    pub fn new(coefficient: i64, factors: Vec<Source<O, P, C>>) -> Self {
        Self {
            coefficient,
            factors,
        }
    }
}

/// A sum-of-products expression: the summand `g(x)` of a sumcheck instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expr<O, P, C> {
    /// Terms summed to form the expression.
    pub terms: Vec<Term<O, P, C>>,
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
/// Field-free: generic only over the identifier types `O`/`P`/`C`. The
/// evaluation field is chosen at [`Self::try_evaluate`] time (the verifier
/// supplies the extension/evaluation field its resolvers return), so the
/// descriptor itself names no field and cannot fix the wrong one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckInstanceDescriptor<O, P, C> {
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
    pub poly: Expr<O, P, C>,
}

impl<O, P, C> Expr<O, P, C> {
    /// Construct an expression from its terms.
    pub fn new(terms: Vec<Term<O, P, C>>) -> Self {
        Self { terms }
    }
}

impl<O, P, C> Expr<O, P, C> {
    /// Evaluate the expression at a resolved point, fallibly and panic-free.
    ///
    /// The evaluation field `F` is chosen by the caller (the verifier supplies
    /// the extension/evaluation field its resolvers return). Each resolver maps
    /// one identifier to its value in `F`: `resolve_opening` to a
    /// claimed/computed MLE evaluation, `resolve_challenge` to a Fiat-Shamir
    /// scalar, `resolve_public` to a verifier-evaluable public weight. A
    /// malformed, unknown, or dimension-mismatched source is reported by the
    /// resolver as `Err`, which short-circuits evaluation: this evaluator never
    /// panics, so it is safe on verifier-reachable paths.
    ///
    /// Each term's integer `coefficient` is lifted into `F`; the result is
    /// `sum_terms coefficient * product_factors resolve(factor)`. A term with
    /// no factors contributes its bare lifted `coefficient`.
    pub fn try_evaluate<F, RO, RC, RP>(
        &self,
        resolve_opening: RO,
        resolve_challenge: RC,
        resolve_public: RP,
    ) -> Result<F, AkitaError>
    where
        F: RingCore + FromPrimitiveInt,
        RO: Fn(&O) -> Result<F, AkitaError>,
        RC: Fn(&C) -> Result<F, AkitaError>,
        RP: Fn(&P) -> Result<F, AkitaError>,
    {
        let mut acc = F::zero();
        for term in &self.terms {
            let mut product = F::from_i64(term.coefficient);
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

impl<O, P, C> SumcheckInstanceDescriptor<O, P, C> {
    /// Evaluate this instance's summand at a resolved point.
    ///
    /// This is the generic verifier descriptor-eval helper: a verifier computes
    /// its `expected_output_claim` by calling this with resolvers that close
    /// over the round challenges. It forwards to [`Expr::try_evaluate`] and is
    /// likewise fallible and panic-free.
    pub fn try_evaluate<F, RO, RC, RP>(
        &self,
        resolve_opening: RO,
        resolve_challenge: RC,
        resolve_public: RP,
    ) -> Result<F, AkitaError>
    where
        F: RingCore + FromPrimitiveInt,
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
        let expr: Expr<O, P, C> = Expr::new(vec![
            Term::new(
                1,
                vec![
                    Source::Challenge(C::Gamma),
                    Source::Public(P::A),
                    Source::Opening(O::W),
                ],
            ),
            Term::new(2, vec![Source::Public(P::B)]),
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
        let expr: Expr<O, P, C> = Expr::new(vec![Term::new(9, Vec::new())]);
        let value = expr
            .try_evaluate(|_| Ok(f(0)), |_| Ok(f(0)), |_| Ok(f(0)))
            .expect("constant term");
        assert_eq!(value, f(9));
    }

    #[test]
    fn try_evaluate_lifts_negative_coefficient() {
        // -1 * A with A = 7 resolves to -7 in the evaluation field.
        let expr: Expr<O, P, C> = Expr::new(vec![Term::new(-1, vec![Source::Public(P::A)])]);
        let value = expr
            .try_evaluate(
                |_| Ok(f(0)),
                |_| Ok(f(0)),
                |p| match p {
                    P::A => Ok(f(7)),
                    P::B => Ok(f(0)),
                },
            )
            .expect("negative coefficient lifts");
        assert_eq!(value, -f(7));
    }

    #[test]
    fn try_evaluate_of_empty_expr_is_zero() {
        let expr: Expr<O, P, C> = Expr::new(Vec::new());
        let value = expr
            .try_evaluate(|_| Ok(f(1)), |_| Ok(f(1)), |_| Ok(f(1)))
            .expect("empty expr");
        assert_eq!(value, F::zero());
    }

    #[test]
    fn try_evaluate_propagates_a_malformed_source_as_error() {
        // The opening resolver rejects the source instead of panicking.
        let expr: Expr<O, P, C> = Expr::new(vec![Term::new(1, vec![Source::Opening(O::W)])]);
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
        let expr: Expr<O, P, C> = Expr::new(vec![
            Term::new(1, vec![Source::Opening(O::W)]),
            Term::new(1, vec![Source::Public(P::A)]),
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
