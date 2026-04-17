//! Claim-reduction sumcheck instance for the batched Stage 2.
//!
//! Fused s-virtualization + w-adaptation:
//!
//! `sum_z eq(r1,z) * [gamma_s * w(z)*(w(z)+1) + gamma_w * w(z)]`
//!
//! Degree 3 per round, shares a single witness table. Batched with
//! `SetupClaimProver` for the full Stage 2.

use super::{fold_evals_in_place, SumcheckInstanceProver, SumcheckInstanceVerifier, UniPoly};
use crate::algebra::eq_poly::EqPolynomial;
use crate::algebra::split_eq::GruenSplitEq;
use crate::error::HachiError;
use crate::{FieldCore, FromSmallInt};

/// Fused s-virtualization + w-adaptation claim reduction prover.
///
/// Inner polynomial: `q(z) = gamma_s * w(z)*(w(z)+1) + gamma_w * w(z)`,
/// multiplied by `eq(r1,z)` via `GruenSplitEq` to produce a degree-3 round
/// polynomial.
pub struct ClaimReductionProver<E: FieldCore> {
    w_table: Vec<E>,
    split_eq: GruenSplitEq<E>,
    num_vars: usize,
    claim: E,
    gamma_s: E,
    gamma_w: E,
}

impl<E: FieldCore + FromSmallInt> ClaimReductionProver<E> {
    /// Build from a padded witness table (full `2^num_vars` entries), the
    /// Stage 1 output point `r_stage1`, and batching coefficients.
    ///
    /// `claim = gamma_s * s_eval + gamma_w * w_eval`.
    pub fn new(w_table: Vec<E>, r_stage1: &[E], claim: E, gamma_s: E, gamma_w: E) -> Self {
        let num_vars = r_stage1.len();
        debug_assert_eq!(w_table.len(), 1 << num_vars);
        Self {
            w_table,
            split_eq: GruenSplitEq::new(r_stage1),
            num_vars,
            claim,
            gamma_s,
            gamma_w,
        }
    }

    /// Return `w(r2)` after all rounds (fully folded witness).
    pub fn final_w_eval(&self) -> E {
        debug_assert_eq!(self.w_table.len(), 1, "w_table not fully folded");
        self.w_table[0]
    }
}

impl<E: FieldCore + FromSmallInt> SumcheckInstanceProver<E> for ClaimReductionProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> E {
        self.claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();

        let mut q0_acc = E::zero();
        let mut q1_acc = E::zero();
        let mut q2_acc = E::zero();

        for (j_high, &e_out) in e_second.iter().enumerate() {
            let base = j_high * num_first;
            let mut inner0 = E::zero();
            let mut inner1 = E::zero();
            let mut inner2 = E::zero();
            for (j_low, &e_in) in e_first.iter().enumerate() {
                let j = base + j_low;
                let w0 = self.w_table[2 * j];
                let w1 = self.w_table[2 * j + 1];
                let dw = w1 - w0;

                let v0 = self.gamma_s * w0 * (w0 + E::one()) + self.gamma_w * w0;
                let v1 = self.gamma_s * w1 * (w1 + E::one()) + self.gamma_w * w1;
                let w2 = w0 + dw + dw;
                let v2 = self.gamma_s * w2 * (w2 + E::one()) + self.gamma_w * w2;

                inner0 += e_in * v0;
                inner1 += e_in * v1;
                inner2 += e_in * v2;
            }
            q0_acc += e_out * inner0;
            q1_acc += e_out * inner1;
            q2_acc += e_out * inner2;
        }

        let q = UniPoly::from_evals(&[q0_acc, q1_acc, q2_acc]);
        self.split_eq.gruen_mul(&q)
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        self.split_eq.bind(r);
        fold_evals_in_place(&mut self.w_table, r);
    }
}

/// Verifier for the fused claim-reduction sumcheck.
///
/// The expected output claim at `r2` is:
/// `eq(r1, r2) * [gamma_s * w(r2)*(w(r2)+1) + gamma_w * w(r2)]`
pub(crate) struct ClaimReductionVerifier<E: FieldCore> {
    r_stage1: Vec<E>,
    num_vars: usize,
    claim: E,
    gamma_s: E,
    gamma_w: E,
    w_eval_stage2: E,
}

impl<E: FieldCore + FromSmallInt> ClaimReductionVerifier<E> {
    /// `w_eval_stage2` is `w(r2)`, the witness MLE at the Stage 2 output point.
    pub(crate) fn new(
        r_stage1: Vec<E>,
        claim: E,
        gamma_s: E,
        gamma_w: E,
        w_eval_stage2: E,
    ) -> Self {
        let num_vars = r_stage1.len();
        Self {
            r_stage1,
            num_vars,
            claim,
            gamma_s,
            gamma_w,
            w_eval_stage2,
        }
    }
}

impl<E: FieldCore + FromSmallInt> SumcheckInstanceVerifier<E> for ClaimReductionVerifier<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> E {
        self.claim
    }

    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, HachiError> {
        let eq_val = EqPolynomial::mle(&self.r_stage1, challenges);
        let w = self.w_eval_stage2;
        let inner = self.gamma_s * w * (w + E::one()) + self.gamma_w * w;
        Ok(eq_val * inner)
    }
}
