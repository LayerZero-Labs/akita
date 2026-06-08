//! Stage-2 fused virtual-claim + relation sumcheck descriptor.
//!
//! This is the worked example of the descriptor abstraction. The verifier's
//! current hand-inlined `expected_output_claim` equation
//! (`crates/akita-verifier/src/stages/stage2.rs`) is
//!
//! ```text
//! g(r) = gamma * eq(stage1_point, r) * W(r) * (W(r) + 1)   // virtual claim
//!       + W(r) * alpha(r_y) * row(r_x)                      // relation
//! ```
//!
//! over the boolean hypercube, degree 3, with input claim
//! `gamma * s_claim + relation_claim`. Expanding `W * (W + 1) = W*W + W`, the
//! virtual half is two monomials, so the summand is the three-term expression
//! built by [`stage2_expr`]. At a terminal level `gamma = 0`, which zeros both
//! virtual monomials, leaving only the relation term, matching the verifier's
//! terminal-level early return without any special case in the descriptor.

use akita_sumcheck::descriptor::{
    ClaimSlot, Expr, InstanceKind, Source, SumcheckInstanceDescriptor, Term,
};

use crate::ids::{AkitaChallengeId, AkitaOpeningId, AkitaPublicId};

/// A sumcheck descriptor over the Akita identifier types.
///
/// Field-free: the evaluation field is chosen when the descriptor is evaluated
/// (`SumcheckInstanceDescriptor::try_evaluate`).
pub type AkitaSumcheckDescriptor =
    SumcheckInstanceDescriptor<AkitaOpeningId, AkitaPublicId, AkitaChallengeId>;

/// The stage-2 fused summand `g(r)` as a sum-of-products expression.
///
/// All coefficients are `1`; the batching coefficient is the
/// [`AkitaChallengeId::BatchingCoeff`] source (a Fiat-Shamir scalar resolved at
/// evaluation time), never a hardcoded constant, so the central batching
/// allocation in the protocol plan controls it.
pub fn stage2_expr() -> Expr<AkitaOpeningId, AkitaPublicId, AkitaChallengeId> {
    let gamma = Source::Challenge(AkitaChallengeId::BatchingCoeff);
    let eq = Source::Public(AkitaPublicId::EqStage1Point);
    let w = Source::Opening(AkitaOpeningId::Witness);
    let alpha = Source::Public(AkitaPublicId::Alpha);
    let row = Source::Public(AkitaPublicId::RelationRow);

    Expr::new(vec![
        // gamma * eq * W * W   (the quadratic part of gamma * eq * W * (W + 1))
        Term::new(1, vec![gamma, eq, w, w]),
        // gamma * eq * W       (the linear part of gamma * eq * W * (W + 1))
        Term::new(1, vec![gamma, eq, w]),
        // W * alpha * row      (the relation term)
        Term::new(1, vec![w, alpha, row]),
    ])
}

/// Build the stage-2 sumcheck instance descriptor.
///
/// `num_rounds` is `col_bits + ring_bits` for the level (derived by
/// `plan::plan_level`). The instance is [`InstanceKind::Regular`]: the fused
/// stage-2 sumcheck is arbitrarily batchable and uses the regular compressed
/// wire format.
pub fn stage2_descriptor(num_rounds: usize) -> AkitaSumcheckDescriptor {
    SumcheckInstanceDescriptor {
        label: "stage2-fused-virtual-relation",
        num_rounds,
        degree: 3,
        kind: InstanceKind::Regular,
        input_claim: ClaimSlot(0),
        output_claim: ClaimSlot(1),
        poly: stage2_expr(),
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

    fn eval_stage2(gamma: F, eq: F, w: F, alpha: F, row: F) -> F {
        stage2_descriptor(4)
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
    fn stage2_descriptor_evaluates_to_known_equation() {
        let (gamma, eq, w, alpha, row) = (f(3), f(5), f(7), f(11), f(13));
        let got = eval_stage2(gamma, eq, w, alpha, row);
        let expected = gamma * eq * w * (w + F::one()) + w * alpha * row;
        assert_eq!(got, expected);
    }

    #[test]
    fn stage2_descriptor_terminal_gamma_zero_is_relation_only() {
        // With gamma = 0 the virtual monomials vanish; only W * alpha * row
        // remains, matching the verifier's terminal-level early return.
        let (eq, w, alpha, row) = (f(5), f(7), f(11), f(13));
        let got = eval_stage2(F::zero(), eq, w, alpha, row);
        assert_eq!(got, w * alpha * row);
    }

    #[test]
    fn stage2_descriptor_shape() {
        let descriptor = stage2_descriptor(9);
        assert_eq!(descriptor.num_rounds, 9);
        assert_eq!(descriptor.degree, 3);
        assert_eq!(descriptor.kind, InstanceKind::Regular);
        assert_eq!(descriptor.poly.terms.len(), 3);
    }

    #[test]
    fn stage2_descriptor_rejects_malformed_resolution() {
        let err = stage2_descriptor(4)
            .try_evaluate(
                |_opening| Err(AkitaError::InvalidProof),
                |_challenge| Ok(F::one()),
                |_public| Ok(F::one()),
            )
            .expect_err("opening resolver rejects -> error, no panic");
        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
