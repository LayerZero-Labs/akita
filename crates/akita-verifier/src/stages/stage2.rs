//! Verifier for the Akita stage-2 fused sumcheck.

use crate::RingSwitchDeferredRowEval;
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_sumcheck::{multilinear_eval, SumcheckInstanceVerifier};
use akita_types::{
    relation_claim_from_rows_extension, AkitaExpandedSetup, DirectWitnessProof, PackedDigits,
    RingOpeningPoint,
};
use std::marker::PhantomData;

fn packed_witness_eval<F, E>(
    packed_witness: &PackedDigits,
    challenges: &[E],
    col_bits: usize,
    ring_bits: usize,
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    if challenges.len() != col_bits + ring_bits {
        return Err(AkitaError::InvalidSize {
            expected: col_bits + ring_bits,
            actual: challenges.len(),
        });
    }

    let d = 1usize
        .checked_shl(
            u32::try_from(ring_bits).map_err(|_| AkitaError::InvalidSize {
                expected: usize::BITS as usize,
                actual: ring_bits,
            })?,
        )
        .ok_or(AkitaError::InvalidProof)?;
    if !packed_witness.num_elems.is_multiple_of(d) {
        return Err(AkitaError::InvalidProof);
    }

    let (y_challenges, x_challenges) = challenges.split_at(ring_bits);
    let eq_y = EqPolynomial::evals(y_challenges)?;
    let eq_x = EqPolynomial::evals(x_challenges)?;
    let live_x_cols = packed_witness.num_elems / d;

    let mut acc = E::zero();
    for (x, &x_weight) in eq_x.iter().take(live_x_cols).enumerate() {
        let base = x << ring_bits;
        let mut y_eval = E::zero();
        for (y, &y_weight) in eq_y.iter().enumerate() {
            let digit = packed_witness
                .digit_at(base + y)
                .ok_or(AkitaError::InvalidProof)?;
            y_eval += y_weight * E::from_i64(digit as i64);
        }
        acc += x_weight * y_eval;
    }

    Ok(acc)
}

fn field_witness_eval<F, E>(
    field_witness: &[F],
    challenges: &[E],
    col_bits: usize,
    ring_bits: usize,
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    if challenges.len() != col_bits + ring_bits {
        return Err(AkitaError::InvalidSize {
            expected: col_bits + ring_bits,
            actual: challenges.len(),
        });
    }

    let d = 1usize
        .checked_shl(
            u32::try_from(ring_bits).map_err(|_| AkitaError::InvalidSize {
                expected: usize::BITS as usize,
                actual: ring_bits,
            })?,
        )
        .ok_or(AkitaError::InvalidProof)?;
    if !field_witness.len().is_multiple_of(d) {
        return Err(AkitaError::InvalidProof);
    }

    let (y_challenges, x_challenges) = challenges.split_at(ring_bits);
    let eq_y = EqPolynomial::evals(y_challenges)?;
    let eq_x = EqPolynomial::evals(x_challenges)?;
    let live_x_cols = field_witness.len() / d;

    let mut acc = E::zero();
    for (x, &x_weight) in eq_x.iter().take(live_x_cols).enumerate() {
        let base = x << ring_bits;
        let mut y_eval = E::zero();
        for (y, &y_weight) in eq_y.iter().enumerate() {
            y_eval += y_weight.mul_base(field_witness[base + y]);
        }
        acc += x_weight * y_eval;
    }

    Ok(acc)
}

fn direct_witness_eval<F, E>(
    direct_witness: &DirectWitnessProof<F>,
    challenges: &[E],
    col_bits: usize,
    ring_bits: usize,
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    match direct_witness {
        DirectWitnessProof::PackedDigits(packed_witness) => {
            packed_witness_eval::<F, E>(packed_witness, challenges, col_bits, ring_bits)
        }
        DirectWitnessProof::FieldElements(field_witness) => {
            field_witness_eval::<F, E>(field_witness.coeffs(), challenges, col_bits, ring_bits)
        }
    }
}

enum Stage2WitnessOracle<'a, F: FieldCore, E: FieldCore> {
    Direct(&'a DirectWitnessProof<F>),
    ClaimedEval(E),
}

/// Source of deferred ring-switch row evaluations used by the stage-2 verifier.
pub struct Stage2RowEvalSource<F: FieldCore> {
    prepared: RingSwitchDeferredRowEval<F>,
}

impl<F: FieldCore> Stage2RowEvalSource<F> {
    /// Construct a source from prepared ring-switch row-eval state.
    pub fn new(prepared: RingSwitchDeferredRowEval<F>) -> Self {
        Self { prepared }
    }
}

/// Verifier for the stage-2 fused virtual-claim and relation sumcheck.
pub struct AkitaStage2Verifier<'a, F: FieldCore, E: FieldCore, const D: usize> {
    batching_coeff: E,
    s_claim: E,
    witness_oracle: Stage2WitnessOracle<'a, F, E>,
    r_stage1: Vec<E>,
    alpha_evals_y: Vec<E>,
    row_eval_source: Stage2RowEvalSource<E>,
    setup: &'a AkitaExpandedSetup<F>,
    opening_points: &'a [RingOpeningPoint<F>],
    alpha: E,
    col_bits: usize,
    ring_bits: usize,
    relation_claim: E,
    _marker: PhantomData<([F; D], E)>,
}

impl<'a, F, E, const D: usize> AkitaStage2Verifier<'a, F, E, D>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        batching_coeff: E,
        s_claim: E,
        witness_oracle: Stage2WitnessOracle<'a, F, E>,
        r_stage1: Vec<E>,
        alpha_evals_y: Vec<E>,
        row_eval_source: Stage2RowEvalSource<E>,
        setup: &'a AkitaExpandedSetup<F>,
        opening_points: &'a [RingOpeningPoint<F>],
        tau1: &[E],
        v: &[CyclotomicRing<F, D>],
        u: &[CyclotomicRing<F, D>],
        y_rings: &[CyclotomicRing<F, D>],
        alpha: E,
        col_bits: usize,
        ring_bits: usize,
    ) -> Result<Self, AkitaError> {
        let num_rounds = col_bits.checked_add(ring_bits).ok_or_else(|| {
            AkitaError::InvalidSetup("stage-2 variable count overflow".to_string())
        })?;
        if r_stage1.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: r_stage1.len(),
            });
        }
        let expected_alpha_len = 1usize
            .checked_shl(
                u32::try_from(ring_bits).map_err(|_| AkitaError::InvalidSize {
                    expected: usize::BITS as usize,
                    actual: ring_bits,
                })?,
            )
            .ok_or(AkitaError::InvalidProof)?;
        if alpha_evals_y.len() != expected_alpha_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_alpha_len,
                actual: alpha_evals_y.len(),
            });
        }
        let relation_claim =
            relation_claim_from_rows_extension::<F, E, D>(tau1, alpha, v, u, y_rings)?;
        Ok(Self {
            batching_coeff,
            s_claim,
            witness_oracle,
            r_stage1,
            alpha_evals_y,
            row_eval_source,
            setup,
            opening_points,
            alpha,
            col_bits,
            ring_bits,
            relation_claim,
            _marker: PhantomData,
        })
    }

    /// Construct a verifier that evaluates the final direct witness locally.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "AkitaStage2Verifier::new_with_direct_witness")]
    pub fn new_with_direct_witness(
        batching_coeff: E,
        s_claim: E,
        direct_witness: &'a DirectWitnessProof<F>,
        r_stage1: Vec<E>,
        alpha_evals_y: Vec<E>,
        row_eval_source: Stage2RowEvalSource<E>,
        setup: &'a AkitaExpandedSetup<F>,
        opening_points: &'a [RingOpeningPoint<F>],
        tau1: &[E],
        v: &[CyclotomicRing<F, D>],
        u: &[CyclotomicRing<F, D>],
        y_rings: &[CyclotomicRing<F, D>],
        alpha: E,
        col_bits: usize,
        ring_bits: usize,
    ) -> Result<Self, AkitaError> {
        Self::new(
            batching_coeff,
            s_claim,
            Stage2WitnessOracle::Direct(direct_witness),
            r_stage1,
            alpha_evals_y,
            row_eval_source,
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
        batching_coeff: E,
        s_claim: E,
        w_eval: E,
        r_stage1: Vec<E>,
        alpha_evals_y: Vec<E>,
        row_eval_source: Stage2RowEvalSource<E>,
        setup: &'a AkitaExpandedSetup<F>,
        opening_points: &'a [RingOpeningPoint<F>],
        tau1: &[E],
        v: &[CyclotomicRing<F, D>],
        u: &[CyclotomicRing<F, D>],
        y_rings: &[CyclotomicRing<F, D>],
        alpha: E,
        col_bits: usize,
        ring_bits: usize,
    ) -> Result<Self, AkitaError> {
        Self::new(
            batching_coeff,
            s_claim,
            Stage2WitnessOracle::ClaimedEval(w_eval),
            r_stage1,
            alpha_evals_y,
            row_eval_source,
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

    fn witness_eval(&self, challenges: &[E]) -> Result<E, AkitaError> {
        match &self.witness_oracle {
            Stage2WitnessOracle::Direct(direct_witness) => direct_witness_eval::<F, E>(
                direct_witness,
                challenges,
                self.col_bits,
                self.ring_bits,
            ),
            Stage2WitnessOracle::ClaimedEval(w_eval) => Ok(*w_eval),
        }
    }

    fn row_eval(&self, x_challenges: &[E]) -> Result<E, AkitaError> {
        self.row_eval_source.prepared.eval_at_point::<F, D>(
            x_challenges,
            self.setup,
            self.opening_points,
            self.alpha,
        )
    }
}

impl<'a, F, E, const D: usize> SumcheckInstanceVerifier<E> for AkitaStage2Verifier<'a, F, E, D>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    fn num_rounds(&self) -> usize {
        self.col_bits + self.ring_bits
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> E {
        self.batching_coeff * self.s_claim + self.relation_claim
    }

    #[tracing::instrument(skip_all, name = "stage2_expected_output_claim")]
    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        let eq_val = EqPolynomial::mle(&self.r_stage1, challenges)?;
        let w_eval = {
            let _span = tracing::info_span!("stage2_witness_eval").entered();
            self.witness_eval(challenges)?
        };
        let virtual_oracle = eq_val * w_eval * (w_eval + E::one());

        let (y_challenges, x_challenges) = challenges.split_at(self.ring_bits);
        let alpha_val = multilinear_eval(&self.alpha_evals_y, y_challenges)?;
        let row_val = {
            let _span = tracing::info_span!("stage2_ring_switch_row_eval").entered();
            self.row_eval(x_challenges)?
        };
        let relation_oracle = w_eval * alpha_val * row_val;
        Ok(self.batching_coeff * virtual_oracle + relation_oracle)
    }
}

#[cfg(test)]
mod tests {
    use super::{field_witness_eval, packed_witness_eval};
    use akita_field::{AkitaError, FieldCore};
    use akita_field::{Fp2, NegOneNr, Prime128Offset275};
    use akita_sumcheck::multilinear_eval;
    use akita_types::PackedDigits;

    type F = Prime128Offset275;
    type E = Fp2<F, NegOneNr>;

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

    #[test]
    fn field_witness_eval_lifts_base_witness_to_extension_challenges() {
        let field_witness = vec![
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
        ];
        let challenges = vec![
            E::new(F::from_u64(2), F::from_u64(3)),
            E::new(F::from_u64(5), F::from_u64(7)),
        ];

        let lifted_witness: Vec<E> = field_witness
            .iter()
            .copied()
            .map(|x| E::new(x, F::zero()))
            .collect();
        let expected =
            multilinear_eval(&lifted_witness, &challenges).expect("matching extension table shape");
        let actual =
            field_witness_eval::<F, E>(&field_witness, &challenges, 1, 1).expect("valid witness");

        assert_eq!(actual, expected);
    }

    #[test]
    fn packed_witness_eval_rejects_challenge_dimension_mismatch() {
        let packed = PackedDigits::from_i8_digits(&[1, -1, 0, 2], 3);
        let err =
            packed_witness_eval::<F, E>(&packed, &[E::zero()], 1, 1).expect_err("wrong arity");
        assert!(matches!(err, AkitaError::InvalidSize { .. }));
    }

    #[test]
    fn packed_witness_eval_rejects_truncated_data() {
        let packed = PackedDigits {
            num_elems: 4,
            bits_per_elem: 3,
            data: vec![],
        };
        let challenges = vec![E::zero(), E::zero()];
        let err = packed_witness_eval::<F, E>(&packed, &challenges, 1, 1)
            .expect_err("truncated packed witness");
        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
