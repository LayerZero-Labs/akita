//! Stage-2 sumcheck descriptor: a weighted sum of named sub-claims.
//!
//! This is the worked example of the descriptor abstraction. The verifier's
//! current hand-inlined `expected_output_claim` equation
//! (`crates/akita-verifier/src/stages/stage2.rs`) is
//!
//! ```text
//! g(r) = gamma * eq(stage1_point, r) * W(r) * (W(r) + 1)   // virtual sub-claim
//!       + W(r) * alpha(r_y) * row(r_x)                      // relation sub-claim
//! ```
//!
//! over the boolean hypercube, degree 3. The summand is a weighted sum of two
//! named sub-claims ([`stage2_summand`]):
//!
//! - the *virtual* norm/range sub-claim, body `eq * W * (W + 1)` (expanded to
//!   the two monomials `eq*W*W + eq*W`), weighted by the Fiat-Shamir batching
//!   coefficient `gamma` ([`stage2_virtual_subclaim`]);
//! - the *relation* sub-claim, body `W * alpha * row`, unweighted
//!   ([`stage2_relation_subclaim`]).
//!
//! Which sub-claims are active is the level's structural choice, carried by
//! [`LevelRole`]. An *intermediate* fold level keeps the witness committed and
//! fuses the virtual sub-claim onto the relation sub-claim. A *terminal* level
//! opens the witness in the clear, so there is no carried virtual claim and the
//! summand keeps only the relation sub-claim. The terminal summand is therefore
//! literally the intermediate summand with the virtual sub-claim dropped, not a
//! separate expression: the structural fact the verifier previously encoded
//! numerically by running the fused equation with `gamma = 0`, a zero
//! `s_claim`, and a fabricated zero `stage1_point`.

use akita_sumcheck::descriptor::{
    ClaimSlot, Expr, InstanceKind, Source, SubClaim, SumcheckInstanceDescriptor, Summand, Term,
};

use crate::ids::{AkitaChallengeId, AkitaOpeningId, AkitaPublicId};
use crate::plan::LevelRole;

/// A sumcheck descriptor over the Akita identifier types.
///
/// Field-free: the evaluation field is chosen when the descriptor is evaluated
/// (`SumcheckInstanceDescriptor::try_evaluate`).
pub type AkitaSumcheckDescriptor =
    SumcheckInstanceDescriptor<AkitaOpeningId, AkitaPublicId, AkitaChallengeId>;

/// A named sub-claim over the Akita identifier types.
pub type AkitaSubClaim = SubClaim<AkitaOpeningId, AkitaPublicId, AkitaChallengeId>;

/// A weighted-sum summand over the Akita identifier types.
pub type AkitaSummand = Summand<AkitaOpeningId, AkitaPublicId, AkitaChallengeId>;

/// The stage-2 *virtual* norm/range sub-claim.
///
/// Body `eq * W * (W + 1)`, expanded to the two monomials `eq*W*W + eq*W`,
/// weighted by the [`AkitaChallengeId::BatchingCoeff`] Fiat-Shamir scalar. The
/// weight is the sub-claim's batching coefficient, resolved at evaluation time
/// and allocated centrally by the protocol plan, never a hardcoded constant.
/// The body itself carries no challenge factor. Present only at intermediate
/// fold levels, where the witness stays committed.
pub fn stage2_virtual_subclaim() -> AkitaSubClaim {
    let eq = Source::Public(AkitaPublicId::EqStage1Point);
    let w = Source::Opening(AkitaOpeningId::Witness);

    SubClaim::new(
        "stage2-virtual-norm",
        Some(AkitaChallengeId::BatchingCoeff),
        Expr::new(vec![
            // eq * W * W   (the quadratic part of eq * W * (W + 1))
            Term::new(1, vec![eq, w, w]),
            // eq * W       (the linear part of eq * W * (W + 1))
            Term::new(1, vec![eq, w]),
        ]),
    )
}

/// The stage-2 *relation* sub-claim.
///
/// Body `W * alpha * row`, unweighted (weight `1`). Present at every level: it
/// is the whole summand at a terminal level and the second half of the fused
/// summand at an intermediate level.
pub fn stage2_relation_subclaim() -> AkitaSubClaim {
    let w = Source::Opening(AkitaOpeningId::Witness);
    let alpha = Source::Public(AkitaPublicId::Alpha);
    let row = Source::Public(AkitaPublicId::RelationRow);

    SubClaim::new(
        "stage2-relation",
        None,
        Expr::new(vec![Term::new(1, vec![w, alpha, row])]),
    )
}

/// The stage-2 summand for a level: a weighted sum of the active sub-claims.
///
/// An intermediate level fuses the virtual sub-claim onto the relation
/// sub-claim; a terminal level keeps only the relation sub-claim. The terminal
/// summand is the intermediate summand with the virtual sub-claim dropped.
pub fn stage2_summand(role: LevelRole) -> AkitaSummand {
    match role {
        LevelRole::Intermediate => {
            Summand::new(vec![stage2_virtual_subclaim(), stage2_relation_subclaim()])
        }
        LevelRole::Terminal => Summand::new(vec![stage2_relation_subclaim()]),
    }
}

/// Build the stage-2 sumcheck instance descriptor for a level.
///
/// `num_rounds` is `col_bits + ring_bits` for the level (derived by
/// `plan::plan_level`). The instance is [`InstanceKind::Regular`]: the stage-2
/// sumcheck is arbitrarily batchable and uses the regular compressed wire
/// format. The summand's active sub-claims follow `role` ([`stage2_summand`]):
/// fused (virtual + relation) at an intermediate level, relation-only at a
/// terminal level. The descriptor-level `degree` is `3` for both (three
/// multilinear factors in the longest body), matching the uniform stage-2
/// degree bound.
pub fn stage2_descriptor(num_rounds: usize, role: LevelRole) -> AkitaSumcheckDescriptor {
    let label = match role {
        LevelRole::Intermediate => "stage2-fused-virtual-relation",
        LevelRole::Terminal => "stage2-relation-only",
    };

    SumcheckInstanceDescriptor {
        label,
        num_rounds,
        degree: 3,
        kind: InstanceKind::Regular,
        input_claim: ClaimSlot(0),
        output_claim: ClaimSlot(1),
        summand: stage2_summand(role),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{AkitaError, Prime64Offset59};

    type F = Prime64Offset59;

    fn f(v: u64) -> F {
        F::from_u64(v)
    }

    fn eval_stage2(role: LevelRole, gamma: F, eq: F, w: F, alpha: F, row: F) -> F {
        stage2_descriptor(4, role)
            .try_evaluate(
                |opening| match opening {
                    AkitaOpeningId::Witness => Ok(w),
                },
                |challenge| match challenge {
                    AkitaChallengeId::BatchingCoeff => Ok(gamma),
                },
                |public| match public {
                    AkitaPublicId::EqStage1Point => Ok(eq),
                    AkitaPublicId::Alpha => Ok(alpha),
                    AkitaPublicId::RelationRow => Ok(row),
                },
            )
            .expect("all stage-2 sources resolve")
    }

    #[test]
    fn stage2_intermediate_evaluates_to_known_equation() {
        let (gamma, eq, w, alpha, row) = (f(3), f(5), f(7), f(11), f(13));
        let got = eval_stage2(LevelRole::Intermediate, gamma, eq, w, alpha, row);
        let expected = gamma * eq * w * (w + F::one()) + w * alpha * row;
        assert_eq!(got, expected);
    }

    // A terminal-level evaluation whose batching-coeff and eq-at-stage1-point
    // resolvers *reject*: a successful eval therefore proves the relation-only
    // summand never touches the dropped virtual sub-claim's sources.
    fn eval_terminal(w: F, alpha: F, row: F) -> F {
        stage2_descriptor(4, LevelRole::Terminal)
            .try_evaluate(
                |opening| match opening {
                    AkitaOpeningId::Witness => Ok(w),
                },
                |challenge| match challenge {
                    AkitaChallengeId::BatchingCoeff => Err(AkitaError::InvalidInput(
                        "no batching coeff at terminal".to_string(),
                    )),
                },
                |public| match public {
                    AkitaPublicId::EqStage1Point => {
                        Err(AkitaError::InvalidInput("no eq at terminal".to_string()))
                    }
                    AkitaPublicId::Alpha => Ok(alpha),
                    AkitaPublicId::RelationRow => Ok(row),
                },
            )
            .expect("relation-only sources resolve")
    }

    #[test]
    fn stage2_terminal_matches_fused_with_gamma_zero() {
        // Dropping the virtual sub-claim evaluates to exactly the fused summand
        // with gamma = 0: the byte-equality bridge. The terminal level dropped
        // the virtual sub-claim structurally rather than zeroing a challenge,
        // but computes the same value, so terminal proofs stay byte-identical.
        let (eq, w, alpha, row) = (f(5), f(7), f(11), f(13));
        let fused_gamma_zero = eval_stage2(LevelRole::Intermediate, F::zero(), eq, w, alpha, row);
        let terminal = eval_terminal(w, alpha, row);
        assert_eq!(fused_gamma_zero, terminal);
        assert_eq!(terminal, w * alpha * row);
    }

    #[test]
    fn stage2_terminal_resolves_no_gamma_or_eq() {
        let _ = eval_terminal(f(2), f(3), f(5));
    }

    #[test]
    fn stage2_intermediate_descriptor_shape() {
        let descriptor = stage2_descriptor(9, LevelRole::Intermediate);
        assert_eq!(descriptor.label, "stage2-fused-virtual-relation");
        assert_eq!(descriptor.num_rounds, 9);
        assert_eq!(descriptor.degree, 3);
        assert_eq!(descriptor.kind, InstanceKind::Regular);
        // Two sub-claims: the gamma-weighted virtual half and the relation half.
        assert_eq!(descriptor.summand.subclaims.len(), 2);
        let virtual_sc = &descriptor.summand.subclaims[0];
        assert_eq!(virtual_sc.weight, Some(AkitaChallengeId::BatchingCoeff));
        assert_eq!(virtual_sc.body.terms.len(), 2);
        let relation_sc = &descriptor.summand.subclaims[1];
        assert_eq!(relation_sc.weight, None);
        assert_eq!(relation_sc.body.terms.len(), 1);

        // The gamma is a sub-claim weight, never a body factor: bodies are
        // challenge-free.
        let body_has_challenge = descriptor.summand.subclaims.iter().any(|sc| {
            sc.body.terms.iter().any(|term| {
                term.factors
                    .iter()
                    .any(|f| matches!(f, Source::Challenge(_)))
            })
        });
        assert!(
            !body_has_challenge,
            "challenges are sub-claim weights, never body factors"
        );
    }

    #[test]
    fn stage2_terminal_descriptor_shape() {
        let descriptor = stage2_descriptor(9, LevelRole::Terminal);
        assert_eq!(descriptor.label, "stage2-relation-only");
        assert_eq!(descriptor.num_rounds, 9);
        assert_eq!(descriptor.degree, 3);
        assert_eq!(descriptor.kind, InstanceKind::Regular);
        // One sub-claim: the unweighted relation half (W * alpha * row).
        assert_eq!(descriptor.summand.subclaims.len(), 1);
        let relation_sc = &descriptor.summand.subclaims[0];
        assert_eq!(relation_sc.weight, None);
        assert_eq!(relation_sc.body.terms.len(), 1);
        assert_eq!(relation_sc.body.terms[0].factors.len(), 3);
    }

    #[test]
    fn stage2_descriptor_rejects_malformed_resolution() {
        let err = stage2_descriptor(4, LevelRole::Intermediate)
            .try_evaluate(
                |_opening| Err(AkitaError::InvalidProof),
                |_challenge| Ok(F::one()),
                |_public| Ok(F::one()),
            )
            .expect_err("opening resolver rejects -> error, no panic");
        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
