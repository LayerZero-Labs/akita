//! Sumcheck descriptor algebra.
//!
//! A sumcheck instance is described by what gets summed over the boolean
//! hypercube each round. Here that description has three layers:
//!
//! - [`Term`]: one monomial, `coefficient * product(sources)`.
//! - [`Expr`]: a sum of terms (the body of one sub-claim).
//! - [`Summand`]: a weighted sum of named [`SubClaim`]s,
//!   `g(x) = Σ weight_c · body_c(x)`.
//!
//! The same [`SumcheckInstanceDescriptor`] is used on both sides of the
//! protocol. The verifier calls [`SumcheckInstanceDescriptor::try_evaluate`] to
//! obtain the expected output claim; the prover walks the same summand over
//! witness oracles. Only how each [`Source`] is resolved
//! differs (claimed evaluation vs witness table).
//!
//! This module is generic over identifier types (`O`/`P`/`C`) and names no
//! protocol-specific opening or equation. Concrete Akita identifiers and
//! per-stage formulas live in `akita-protocol`.
//!
//! Types are field-free: integer monomial coefficients are lifted at evaluation
//! time, and the evaluation field is chosen by the caller. Every evaluator entry
//! point is fallible and panic-free for verifier-reachable use.
//!
//! Fiat-Shamir weights belong on [`SubClaim::weight`], not inside term bodies.
//! That keeps monomial integer coefficients ([`Term::coefficient`]) separate
//! from challenge scalars. A level's plan chooses which sub-claims are active
//! (for example fusing a carried norm/range claim with a relation claim, or
//! dropping the carried claim at a cleartext tail). The same weighting pattern
//! applies one level up when several instances are batched together.

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
/// `coefficient` is a small integer (typically `1` or `-1`) lifted into the
/// evaluation field at evaluation time. Fiat-Shamir weights belong on
/// [`SubClaim::weight`]; [`Source::Challenge`] in a body is reserved for the
/// rare case a challenge multiplies inside a monomial rather than scaling a
/// whole sub-claim. An empty `factors` list is the bare constant `coefficient`.
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

/// A sum-of-products expression: the body of a [`SubClaim`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expr<O, P, C> {
    /// Terms summed to form the expression.
    pub terms: Vec<Term<O, P, C>>,
}

/// A named, weighted sub-claim: a sum-of-products body scaled by an optional
/// Fiat-Shamir batching coefficient.
///
/// `weight` is the challenge that scales the whole `body`; `None` is weight `1`.
/// Pulling the coefficient out of the body keeps two roles distinct: a
/// monomial's structural integer multiplier stays in [`Term::coefficient`],
/// while a Fiat-Shamir weight is the sub-claim's `weight`. As a consequence
/// bodies carry no [`Source::Challenge`] factor; every challenge enters as a
/// sub-claim weight, allocated centrally by the protocol plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubClaim<O, P, C> {
    /// Human-readable diagnostic label (e.g. `"stage2-relation"`). Not bound
    /// into the transcript.
    pub label: &'static str,
    /// The Fiat-Shamir challenge scaling `body`, or `None` for weight `1`.
    pub weight: Option<C>,
    /// The sum-of-products body, with no weight factor included.
    pub body: Expr<O, P, C>,
}

/// A sumcheck instance's summand `g(x)`: a weighted sum of named sub-claims.
///
/// `g(x) = Σ_c weight_c · body_c(x)`. The active set of sub-claims is the
/// structural variable a level's plan controls: an intermediate fold level
/// fuses a virtual norm/range sub-claim onto the relation sub-claim, while a
/// terminal (cleartext-witness) level keeps only the relation sub-claim. This
/// is the same weighting cross-instance batching applies one level up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summand<O, P, C> {
    /// Sub-claims summed (each weighted) to form the summand.
    pub subclaims: Vec<SubClaim<O, P, C>>,
}

/// Whether a sumcheck instance may be batched with others and which proof wire
/// format it uses on the transcript.
///
/// Proof format is separate from how the prover computes round polynomials: an
/// eq-factored instance may still use split-eq internally even when batching
/// forces the regular compressed wire format on the transcript.
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
/// The same descriptor drives verifier evaluation ([`Self::try_evaluate`]) and
/// prover round polynomials; only how each [`Source`] is resolved differs
/// between the two sides.
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
    /// Total degree of the summand in the round variable.
    pub degree: usize,
    /// Batchability / wire-format class.
    pub kind: InstanceKind,
    /// Chained input claim this instance consumes.
    pub input_claim: ClaimSlot,
    /// Chained output claim this instance produces.
    pub output_claim: ClaimSlot,
    /// The summand `g(x)` for this instance, as a weighted sum of sub-claims.
    pub summand: Summand<O, P, C>,
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

impl<O, P, C> SubClaim<O, P, C> {
    /// Construct a sub-claim from its label, optional weight challenge, and body.
    pub fn new(label: &'static str, weight: Option<C>, body: Expr<O, P, C>) -> Self {
        Self {
            label,
            weight,
            body,
        }
    }
}

impl<O, P, C> Summand<O, P, C> {
    /// Construct a summand from its weighted sub-claims.
    pub fn new(subclaims: Vec<SubClaim<O, P, C>>) -> Self {
        Self { subclaims }
    }

    /// Evaluate the weighted sum of sub-claims at a resolved point, fallibly and
    /// panic-free.
    ///
    /// The result is `sum_subclaims weight * body`, where a `None` weight is `1`
    /// and a `Some(challenge)` weight is resolved through `resolve_challenge`.
    /// Each body is evaluated through [`Expr::try_evaluate`]; a malformed or
    /// unknown source short-circuits as `Err`, so this is safe on
    /// verifier-reachable paths.
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
        for subclaim in &self.subclaims {
            let weight = match &subclaim.weight {
                Some(challenge) => resolve_challenge(challenge)?,
                None => F::one(),
            };
            let body = subclaim.body.try_evaluate(
                &resolve_opening,
                &resolve_challenge,
                &resolve_public,
            )?;
            acc += weight * body;
        }
        Ok(acc)
    }
}

impl<O, P, C> SumcheckInstanceDescriptor<O, P, C> {
    /// Evaluate this instance's summand at a resolved point.
    ///
    /// This is the generic verifier descriptor-eval helper: a verifier computes
    /// its `expected_output_claim` by calling this with resolvers that close
    /// over the round challenges. It forwards to [`Summand::try_evaluate`] and is
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
        self.summand
            .try_evaluate(resolve_opening, resolve_challenge, resolve_public)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

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

    #[test]
    fn summand_weights_each_subclaim_by_its_challenge() {
        // gamma * (A * W)  +  1 * (2 * B), gamma = 3, A = 7, W = 5, B = 11.
        let summand: Summand<O, P, C> = Summand::new(vec![
            SubClaim::new(
                "weighted",
                Some(C::Gamma),
                Expr::new(vec![Term::new(
                    1,
                    vec![Source::Public(P::A), Source::Opening(O::W)],
                )]),
            ),
            SubClaim::new(
                "unweighted",
                None,
                Expr::new(vec![Term::new(2, vec![Source::Public(P::B)])]),
            ),
        ]);

        let value = summand
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

        // 3 * (7 * 5) + 1 * (2 * 11) = 105 + 22 = 127.
        assert_eq!(value, f(127));
    }

    #[test]
    fn summand_dropping_a_subclaim_drops_its_contribution() {
        // The same two sub-claims, but the weighted one is absent: only the
        // bare second sub-claim contributes (2 * 11 = 22). This is the terminal
        // vs intermediate distinction in miniature.
        let summand: Summand<O, P, C> = Summand::new(vec![SubClaim::new(
            "unweighted",
            None,
            Expr::new(vec![Term::new(2, vec![Source::Public(P::B)])]),
        )]);

        let value = summand
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

        assert_eq!(value, f(22));
    }

    #[test]
    fn summand_propagates_a_malformed_weight_as_error() {
        // A rejected weight challenge short-circuits instead of panicking.
        let summand: Summand<O, P, C> = Summand::new(vec![SubClaim::new(
            "weighted",
            Some(C::Gamma),
            Expr::new(vec![Term::new(1, vec![Source::Opening(O::W)])]),
        )]);
        let err = summand
            .try_evaluate(
                |_o| Ok(f(4)),
                |_c| Err(AkitaError::InvalidProof),
                |_p| Ok(f(1)),
            )
            .expect_err("malformed weight must be rejected, not panic");
        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
