//! Verifier for the Akita stage-2 fused sumcheck.

use crate::PreparedMEval;
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_sumcheck::{multilinear_eval, SumcheckInstanceVerifier};
use akita_types::{
    relation_claim_from_rows, AkitaExpandedSetup, DirectWitnessProof, PackedDigits,
    RingOpeningPoint,
};
use std::marker::PhantomData;

fn packed_witness_eval<F: FieldCore + FromPrimitiveInt>(
    packed_witness: &PackedDigits,
    challenges: &[F],
    col_bits: usize,
    ring_bits: usize,
) -> Result<F, AkitaError> {
    if challenges.len() != col_bits + ring_bits {
        return Err(AkitaError::InvalidSize {
            expected: col_bits + ring_bits,
            actual: challenges.len(),
        });
    }

    let d = 1usize << ring_bits;
    if !packed_witness.num_elems.is_multiple_of(d) {
        return Err(AkitaError::InvalidProof);
    }

    let (y_challenges, x_challenges) = challenges.split_at(ring_bits);
    let eq_y = EqPolynomial::evals(y_challenges);
    let eq_x = EqPolynomial::evals(x_challenges);
    let live_x_cols = packed_witness.num_elems / d;

    let mut acc = F::zero();
    for (x, &x_weight) in eq_x.iter().take(live_x_cols).enumerate() {
        let base = x << ring_bits;
        let mut y_eval = F::zero();
        for (y, &y_weight) in eq_y.iter().enumerate() {
            let digit = packed_witness
                .digit_at(base + y)
                .ok_or(AkitaError::InvalidProof)?;
            y_eval += y_weight * F::from_i64(digit as i64);
        }
        acc += x_weight * y_eval;
    }

    Ok(acc)
}

fn field_witness_eval<F: FieldCore>(
    field_witness: &[F],
    challenges: &[F],
    col_bits: usize,
    ring_bits: usize,
) -> Result<F, AkitaError> {
    if challenges.len() != col_bits + ring_bits {
        return Err(AkitaError::InvalidSize {
            expected: col_bits + ring_bits,
            actual: challenges.len(),
        });
    }

    let d = 1usize << ring_bits;
    if !field_witness.len().is_multiple_of(d) {
        return Err(AkitaError::InvalidProof);
    }

    let (y_challenges, x_challenges) = challenges.split_at(ring_bits);
    let eq_y = EqPolynomial::evals(y_challenges);
    let eq_x = EqPolynomial::evals(x_challenges);
    let live_x_cols = field_witness.len() / d;

    let mut acc = F::zero();
    for (x, &x_weight) in eq_x.iter().take(live_x_cols).enumerate() {
        let base = x << ring_bits;
        let mut y_eval = F::zero();
        for (y, &y_weight) in eq_y.iter().enumerate() {
            y_eval += field_witness[base + y] * y_weight;
        }
        acc += x_weight * y_eval;
    }

    Ok(acc)
}

fn direct_witness_eval<F: FieldCore + FromPrimitiveInt>(
    direct_witness: &DirectWitnessProof<F>,
    challenges: &[F],
    col_bits: usize,
    ring_bits: usize,
) -> Result<F, AkitaError> {
    match direct_witness {
        DirectWitnessProof::PackedDigits(packed_witness) => {
            packed_witness_eval(packed_witness, challenges, col_bits, ring_bits)
        }
        DirectWitnessProof::FieldElements(field_witness) => {
            field_witness_eval(field_witness.coeffs(), challenges, col_bits, ring_bits)
        }
    }
}

enum Stage2WitnessOracle<'a, F: FieldCore> {
    Direct(&'a DirectWitnessProof<F>),
    ClaimedEval(F),
}

/// Source of deferred M-table evaluations used by the stage-2 verifier.
pub struct Stage2MEvalSource<F: FieldCore> {
    prepared: PreparedMEval<F>,
}

impl<F: FieldCore> Stage2MEvalSource<F> {
    /// Construct a source from prepared M-eval state.
    pub fn new(prepared: PreparedMEval<F>) -> Self {
        Self { prepared }
    }
}

/// Verifier for the stage-2 fused virtual-claim and relation sumcheck.
///
/// Holds both batching coefficients per book §5.6 Figure 12 Round 8
/// line 912–919: `γ_range` scales the range-check / virtual-claim side
/// (`s_claim`, `eq(r_stage1) * w * (w+1)`) and `γ_rel` scales the
/// relation side (`relation_claim`, `w * α̃(r_y) * m̃(r_x)`).
pub struct AkitaStage2Verifier<'a, F: FieldCore, const D: usize> {
    gamma_range: F,
    gamma_rel: F,
    s_claim: F,
    witness_oracle: Stage2WitnessOracle<'a, F>,
    r_stage1: Vec<F>,
    alpha_evals_y: Vec<F>,
    m_eval_source: Stage2MEvalSource<F>,
    setup: &'a AkitaExpandedSetup<F>,
    opening_points: &'a [RingOpeningPoint<F>],
    alpha: F,
    col_bits: usize,
    ring_bits: usize,
    relation_claim: F,
    _marker: PhantomData<[F; D]>,
}

impl<'a, F: FieldCore + FromPrimitiveInt + CanonicalField, const D: usize>
    AkitaStage2Verifier<'a, F, D>
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        gamma_range: F,
        gamma_rel: F,
        s_claim: F,
        witness_oracle: Stage2WitnessOracle<'a, F>,
        r_stage1: Vec<F>,
        alpha_evals_y: Vec<F>,
        m_eval_source: Stage2MEvalSource<F>,
        setup: &'a AkitaExpandedSetup<F>,
        opening_points: &'a [RingOpeningPoint<F>],
        tau1: &[F],
        v: &[CyclotomicRing<F, D>],
        u: &[CyclotomicRing<F, D>],
        y_rings: &[CyclotomicRing<F, D>],
        alpha: F,
        col_bits: usize,
        ring_bits: usize,
    ) -> Self {
        let relation_claim = relation_claim_from_rows::<F, D>(tau1, alpha, v, u, y_rings);
        Self {
            gamma_range,
            gamma_rel,
            s_claim,
            witness_oracle,
            r_stage1,
            alpha_evals_y,
            m_eval_source,
            setup,
            opening_points,
            alpha,
            col_bits,
            ring_bits,
            relation_claim,
            _marker: PhantomData,
        }
    }

    /// Construct a verifier that evaluates the final direct witness locally.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "AkitaStage2Verifier::new_with_direct_witness")]
    pub fn new_with_direct_witness(
        gamma_range: F,
        gamma_rel: F,
        s_claim: F,
        direct_witness: &'a DirectWitnessProof<F>,
        r_stage1: Vec<F>,
        alpha_evals_y: Vec<F>,
        m_eval_source: Stage2MEvalSource<F>,
        setup: &'a AkitaExpandedSetup<F>,
        opening_points: &'a [RingOpeningPoint<F>],
        tau1: &[F],
        v: &[CyclotomicRing<F, D>],
        u: &[CyclotomicRing<F, D>],
        y_rings: &[CyclotomicRing<F, D>],
        alpha: F,
        col_bits: usize,
        ring_bits: usize,
    ) -> Self {
        Self::new(
            gamma_range,
            gamma_rel,
            s_claim,
            Stage2WitnessOracle::Direct(direct_witness),
            r_stage1,
            alpha_evals_y,
            m_eval_source,
            setup,
            opening_points,
            tau1,
            v,
            u,
            y_rings,
            alpha,
            col_bits,
            ring_bits,
        )
    }

    /// Construct a verifier that consumes an already claimed next-witness eval.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "AkitaStage2Verifier::new_with_claimed_w_eval")]
    pub fn new_with_claimed_w_eval(
        gamma_range: F,
        gamma_rel: F,
        s_claim: F,
        w_eval: F,
        r_stage1: Vec<F>,
        alpha_evals_y: Vec<F>,
        m_eval_source: Stage2MEvalSource<F>,
        setup: &'a AkitaExpandedSetup<F>,
        opening_points: &'a [RingOpeningPoint<F>],
        tau1: &[F],
        v: &[CyclotomicRing<F, D>],
        u: &[CyclotomicRing<F, D>],
        y_rings: &[CyclotomicRing<F, D>],
        alpha: F,
        col_bits: usize,
        ring_bits: usize,
    ) -> Self {
        Self::new(
            gamma_range,
            gamma_rel,
            s_claim,
            Stage2WitnessOracle::ClaimedEval(w_eval),
            r_stage1,
            alpha_evals_y,
            m_eval_source,
            setup,
            opening_points,
            tau1,
            v,
            u,
            y_rings,
            alpha,
            col_bits,
            ring_bits,
        )
    }

    /// Override the relation claim when the caller uses an explicit row layout.
    #[inline]
    pub fn with_relation_claim(mut self, relation_claim: F) -> Self {
        self.relation_claim = relation_claim;
        self
    }

    fn witness_eval(&self, challenges: &[F]) -> Result<F, AkitaError> {
        match &self.witness_oracle {
            Stage2WitnessOracle::Direct(direct_witness) => {
                direct_witness_eval(direct_witness, challenges, self.col_bits, self.ring_bits)
            }
            Stage2WitnessOracle::ClaimedEval(w_eval) => Ok(*w_eval),
        }
    }

    fn m_eval(&self, x_challenges: &[F]) -> Result<F, AkitaError> {
        self.m_eval_source.prepared.eval_at_point::<D>(
            x_challenges,
            self.setup,
            self.opening_points,
            self.alpha,
        )
    }

    /// Borrow the prepared M-eval state, used by the setup-side claim
    /// reduction to materialize structured weights and the setup polynomial.
    #[inline]
    pub fn prepared_m_eval(&self) -> &crate::PreparedMEval<F> {
        &self.m_eval_source.prepared
    }

    /// Algebraic part of `m(r_x)` only. This is what a claim-reduction-aware
    /// verifier can compute cheaply on its own: the setup-dependent residual
    /// is deferred to the claim-reduction sumcheck.
    ///
    /// Uses the algebraic-only fast path so the verifier avoids iterating over
    /// the shared setup matrix `S` when claim reduction is enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if the algebraic evaluation fails for the supplied
    /// challenges.
    pub fn m_alg_eval(&self, x_challenges: &[F]) -> Result<F, AkitaError> {
        self.m_eval_source.prepared.eval_algebraic_at_point::<D>(
            x_challenges,
            self.opening_points,
            self.alpha,
        )
    }

    /// Expected stage-2 closing oracle value when the setup-dependent residual
    /// is supplied externally.
    ///
    /// This mirrors [`Self::expected_output_claim`] except
    /// `scaled_m_setup_eval` supplies `w_eval * alpha(r_y) * m_setup(r_x)`.
    /// Callers verify the supplied scaled value separately via a
    /// claim-reduction sumcheck.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness oracle or alpha-power table evaluations
    /// fail for the supplied challenges.
    pub fn expected_output_claim_with_m_setup(
        &self,
        challenges: &[F],
        scaled_m_setup_eval: F,
    ) -> Result<F, AkitaError> {
        let eq_val = EqPolynomial::mle(&self.r_stage1, challenges);
        let w_eval = self.witness_eval(challenges)?;
        let virtual_oracle = eq_val * w_eval * (w_eval + F::one());

        let (y_challenges, x_challenges) = challenges.split_at(self.ring_bits);
        let alpha_val = multilinear_eval(&self.alpha_evals_y, y_challenges)?;
        let m_alg = self.m_alg_eval(x_challenges)?;
        let relation_oracle = w_eval * alpha_val * m_alg + scaled_m_setup_eval;
        Ok(self.gamma_range * virtual_oracle + self.gamma_rel * relation_oracle)
    }

    /// Scaling factor `lambda = w_eval * alpha(r_y)` used by the book §5.4
    /// setup-side claim reduction.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness oracle or alpha table evaluation fails
    /// at the supplied stage-2 point.
    pub fn setup_claim_scale(&self, challenges: &[F]) -> Result<F, AkitaError> {
        let w_eval = self.witness_eval(challenges)?;
        let (y_challenges, _) = challenges.split_at(self.ring_bits);
        let alpha_val = multilinear_eval(&self.alpha_evals_y, y_challenges)?;
        Ok(w_eval * alpha_val)
    }

    /// Expose the ring-coordinate variable count needed to split sumcheck
    /// challenges into `(y_challenges, x_challenges)`.
    #[inline]
    pub fn ring_bits(&self) -> usize {
        self.ring_bits
    }

    /// Expose the ring-switch challenge value for downstream verifiers.
    #[inline]
    pub fn alpha(&self) -> F {
        self.alpha
    }

    /// Borrow the verifier setup used for downstream setup-polynomial views.
    #[inline]
    pub fn setup(&self) -> &'a AkitaExpandedSetup<F> {
        self.setup
    }

    /// Borrow the opening points used by the stage-2 M-eval.
    #[inline]
    pub fn opening_points(&self) -> &'a [akita_types::RingOpeningPoint<F>] {
        self.opening_points
    }
}

impl<'a, F: FieldCore + FromPrimitiveInt + CanonicalField, const D: usize>
    SumcheckInstanceVerifier<F> for AkitaStage2Verifier<'a, F, D>
{
    fn num_rounds(&self) -> usize {
        self.col_bits + self.ring_bits
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> F {
        self.gamma_range * self.s_claim + self.gamma_rel * self.relation_claim
    }

    #[tracing::instrument(skip_all, name = "stage2_expected_output_claim")]
    fn expected_output_claim(&self, challenges: &[F]) -> Result<F, AkitaError> {
        let eq_val = EqPolynomial::mle(&self.r_stage1, challenges);
        let w_eval = {
            let _span = tracing::info_span!("stage2_witness_eval").entered();
            self.witness_eval(challenges)?
        };
        let virtual_oracle = eq_val * w_eval * (w_eval + F::one());

        let (y_challenges, x_challenges) = challenges.split_at(self.ring_bits);
        let alpha_val = multilinear_eval(&self.alpha_evals_y, y_challenges)?;
        let m_val = {
            let _span = tracing::info_span!("stage2_m_eval").entered();
            self.m_eval(x_challenges)?
        };
        let relation_oracle = w_eval * alpha_val * m_val;
        Ok(self.gamma_range * virtual_oracle + self.gamma_rel * relation_oracle)
    }
}

#[cfg(test)]
mod tests {
    use super::packed_witness_eval;
    use akita_field::Prime128Offset275;
    use akita_field::{AkitaError, FieldCore};
    use akita_sumcheck::multilinear_eval;
    use akita_types::PackedDigits;

    type F = Prime128Offset275;

    fn build_w_evals<F: FieldCore>(
        w: &[F],
        d: usize,
    ) -> Result<(Vec<F>, usize, usize), AkitaError> {
        if !w.len().is_multiple_of(d) {
            return Err(AkitaError::InvalidSize {
                expected: d,
                actual: w.len(),
            });
        }
        let ring_bits = d.trailing_zeros() as usize;
        let num_ring_elems = w.len() / d;
        let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
        let x_len = 1usize << col_bits;
        let n = x_len << ring_bits;

        let mut out = vec![F::zero(); n];
        out[..w.len()].copy_from_slice(w);
        Ok((out, col_bits, ring_bits))
    }

    #[test]
    fn packed_witness_eval_matches_materialized_table() {
        let d = 4usize;
        let w_digits = vec![3, -1, 2, 0, -2, 1, 4, -3, 1, 0, -4, 2];
        let packed = PackedDigits::from_i8_digits(&w_digits, 4);
        let w_field: Vec<F> = w_digits
            .iter()
            .map(|&digit| F::from_i64(digit as i64))
            .collect();
        let (w_evals, col_bits, ring_bits) =
            build_w_evals(&w_field, d).expect("valid witness shape");
        let challenges = vec![
            F::from_u64(2),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
        ];

        assert_eq!(col_bits + ring_bits, challenges.len());

        let expected = multilinear_eval(&w_evals, &challenges).expect("matching table shape");
        let actual = packed_witness_eval(&packed, &challenges, col_bits, ring_bits)
            .expect("valid packed witness");

        assert_eq!(actual, expected);
    }
}
