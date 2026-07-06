//! Verifier for the Akita stage-2 fused sumcheck.

use crate::protocol::ring_switch::RingSwitchDeferredRowEval;
use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
};
use akita_sumcheck::{multilinear_eval, SumcheckInstanceVerifier};
use akita_types::{
    dispatch_for_field, eval_trace_terms_closed, AkitaExpandedSetup, CleartextWitnessProof,
    FpExtEncoding, RingMultiplierOpeningPoint, RingOpeningPoint, TraceClaim,
};
use std::borrow::Cow;
use std::marker::PhantomData;

fn witness_eval_by_index<E, V>(
    witness_len: usize,
    challenges: &[E],
    ring_bits: usize,
    y_len: usize,
    mut value_at: V,
) -> Result<E, AkitaError>
where
    E: FieldCore,
    V: FnMut(usize) -> Result<E, AkitaError>,
{
    if !witness_len.is_multiple_of(y_len) {
        return Err(AkitaError::InvalidProof);
    }

    let (y_challenges, x_challenges) = challenges.split_at(ring_bits);
    let eq_y = EqPolynomial::evals(y_challenges)?;
    let eq_x = EqPolynomial::evals(x_challenges)?;
    let live_x_cols = witness_len / y_len;
    if live_x_cols > eq_x.len() {
        return Err(AkitaError::InvalidProof);
    }

    let mut acc = E::zero();
    for (x, &x_weight) in eq_x.iter().take(live_x_cols).enumerate() {
        let base = x * y_len;
        let mut y_eval = E::zero();
        for (y, &y_weight) in eq_y.iter().enumerate() {
            y_eval += y_weight * value_at(base + y)?;
        }
        acc += x_weight * y_eval;
    }

    Ok(acc)
}

/// Stage-2 sumcheck operates on the logical witness hypercube, not the wire encoding.
pub(crate) enum Stage2CleartextSource<'a, F: FieldCore> {
    /// Expanded balanced digit planes (segment-typed terminal after decode).
    LogicalDigits(Cow<'a, [i8]>),
    /// Root-direct cleartext field coefficients.
    FieldElements(&'a [F]),
}

fn cleartext_source_eval<F, E, const D: usize>(
    physical_w_len: usize,
    source: &Stage2CleartextSource<'_, F>,
    challenges: &[E],
    col_bits: usize,
    ring_bits: usize,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let num_rounds = col_bits.checked_add(ring_bits).ok_or_else(|| {
        AkitaError::InvalidSetup("stage-2 witness variable count overflow".to_string())
    })?;
    if challenges.len() != num_rounds {
        return Err(AkitaError::InvalidSize {
            expected: num_rounds,
            actual: challenges.len(),
        });
    }
    let y_len = 1usize
        .checked_shl(
            u32::try_from(ring_bits).map_err(|_| AkitaError::InvalidSize {
                expected: usize::BITS as usize,
                actual: ring_bits,
            })?,
        )
        .ok_or(AkitaError::InvalidProof)?;
    match source {
        Stage2CleartextSource::LogicalDigits(digits) => {
            if digits.len() != physical_w_len || D == 0 || !physical_w_len.is_multiple_of(D) {
                return Err(AkitaError::InvalidProof);
            }
            witness_eval_by_index(physical_w_len, challenges, ring_bits, y_len, |idx| {
                Ok(E::from_i64(digits[idx] as i64))
            })
        }
        Stage2CleartextSource::FieldElements(field_witness) => {
            if field_witness.len() != physical_w_len {
                return Err(AkitaError::InvalidProof);
            }
            witness_eval_by_index(physical_w_len, challenges, ring_bits, y_len, |idx| {
                Ok(E::lift_base(field_witness[idx]))
            })
        }
    }
}

pub(crate) enum Stage2WitnessOracle<'a, F: FieldCore, E: FieldCore> {
    Cleartext {
        physical_w_len: usize,
        source: Stage2CleartextSource<'a, F>,
    },
    ClaimedEval {
        eval: E,
    },
}

/// Decode a terminal cleartext witness into the stage-2 digit oracle.
///
/// Segment-typed wire payloads expand to logical `i8` digits once here; stage-2
/// never sees the segment encoding again.
pub(crate) fn stage2_cleartext_oracle<'a, F, E>(
    witness: &'a CleartextWitnessProof<F>,
    physical_w_len: usize,
    lp: &akita_types::LevelParams,
    num_segments: usize,
) -> Result<Stage2WitnessOracle<'a, F, E>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: FieldCore,
{
    let d_a = lp.role_dims().d_a();
    let source = match witness {
        CleartextWitnessProof::SegmentTyped(_) => {
            let digits =
                dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
                    witness.logical_i8_digits::<D>(lp, num_segments)
                })?;
            if digits.len() != physical_w_len {
                return Err(AkitaError::InvalidProof);
            }
            Stage2CleartextSource::LogicalDigits(Cow::Owned(digits))
        }
        CleartextWitnessProof::FieldElements(field_elems) => {
            let coeffs = field_elems.coeffs();
            if coeffs.len() != physical_w_len {
                return Err(AkitaError::InvalidProof);
            }
            Stage2CleartextSource::FieldElements(coeffs)
        }
    };
    Ok(Stage2WitnessOracle::Cleartext {
        physical_w_len,
        source,
    })
}

/// Verifier for the stage-2 fused virtual-claim and relation sumcheck.
pub(crate) struct AkitaStage2Verifier<'a, F: FieldCore, E: FieldCore, const D: usize> {
    batching_coeff: E,
    s_claim: E,
    witness_oracle: Stage2WitnessOracle<'a, F, E>,
    stage1_point: Vec<E>,
    alpha_evals_y: Vec<E>,
    prepared_row_eval: RingSwitchDeferredRowEval<E>,
    setup_claim: Option<E>,
    setup: &'a AkitaExpandedSetup<F>,
    opening_point: &'a RingOpeningPoint<F>,
    ring_multiplier_point: &'a RingMultiplierOpeningPoint<F>,
    alpha: E,
    col_bits: usize,
    ring_bits: usize,
    relation_claim: E,
    trace: Option<TraceClaim<F, E, D>>,
    _marker: PhantomData<([F; D], E)>,
}

impl<'a, F, E, const D: usize> AkitaStage2Verifier<'a, F, E, D>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: ExtField<F> + FpExtEncoding<F> + FromPrimitiveInt,
{
    /// Construct a verifier from the shared stage-2 context and the witness
    /// oracle selected by the current proof level.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "AkitaStage2Verifier::new")]
    pub(crate) fn new(
        batching_coeff: E,
        s_claim: E,
        witness_oracle: Stage2WitnessOracle<'a, F, E>,
        stage1_point: Vec<E>,
        alpha_evals_y: Vec<E>,
        prepared_row_eval: RingSwitchDeferredRowEval<E>,
        setup_claim: Option<E>,
        setup: &'a AkitaExpandedSetup<F>,
        opening_point: &'a RingOpeningPoint<F>,
        ring_multiplier_point: &'a RingMultiplierOpeningPoint<F>,
        relation_claim: E,
        alpha: E,
        col_bits: usize,
        ring_bits: usize,
        trace: Option<TraceClaim<F, E, D>>,
    ) -> Result<Self, AkitaError> {
        let num_rounds = col_bits.checked_add(ring_bits).ok_or_else(|| {
            AkitaError::InvalidSetup("stage-2 variable count overflow".to_string())
        })?;
        if stage1_point.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: stage1_point.len(),
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
        Ok(Self {
            batching_coeff,
            s_claim,
            witness_oracle,
            stage1_point,
            alpha_evals_y,
            prepared_row_eval,
            setup_claim,
            setup,
            opening_point,
            ring_multiplier_point,
            alpha,
            col_bits,
            ring_bits,
            relation_claim,
            trace,
            _marker: PhantomData,
        })
    }

    fn witness_eval(&self, challenges: &[E]) -> Result<E, AkitaError> {
        match &self.witness_oracle {
            Stage2WitnessOracle::Cleartext {
                physical_w_len,
                source,
            } => cleartext_source_eval::<F, E, D>(
                *physical_w_len,
                source,
                challenges,
                self.col_bits,
                self.ring_bits,
            ),
            Stage2WitnessOracle::ClaimedEval { eval, .. } => Ok(*eval),
        }
    }

    fn row_eval(&self, x_challenges: &[E]) -> Result<E, AkitaError> {
        self.prepared_row_eval.eval_at_point::<F, D>(
            x_challenges,
            self.setup,
            self.opening_point,
            self.ring_multiplier_point,
            self.alpha,
            self.setup_claim,
        )
    }
}

impl<'a, F, E, const D: usize> SumcheckInstanceVerifier<E> for AkitaStage2Verifier<'a, F, E, D>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: ExtField<F> + FpExtEncoding<F> + FromPrimitiveInt,
{
    fn num_rounds(&self) -> usize {
        self.col_bits + self.ring_bits
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> E {
        let mut claim = self.batching_coeff * self.s_claim + self.relation_claim;
        if let Some(trace) = &self.trace {
            claim += trace.trace_opening_claim;
        }
        claim
    }

    #[tracing::instrument(skip_all, name = "stage2_expected_output_claim")]
    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        let w_eval = {
            let _span = tracing::info_span!("stage2_witness_eval").entered();
            self.witness_eval(challenges)?
        };

        let (y_challenges, x_challenges) = challenges.split_at(self.ring_bits);
        let alpha_val = multilinear_eval(&self.alpha_evals_y, y_challenges)?;
        let row_val = {
            let _span = tracing::info_span!("stage2_ring_switch_row_eval").entered();
            self.row_eval(x_challenges)?
        };
        let relation_oracle = w_eval * alpha_val * row_val;
        let trace_oracle = if let Some(trace) = &self.trace {
            let trace_weight = eval_trace_terms_closed::<F, E, D>(
                &trace.layout,
                y_challenges,
                x_challenges,
                &trace.trace_terms,
            )?;
            trace.trace_coeff * w_eval * trace_weight
        } else {
            E::zero()
        };

        // Terminal levels run with `batching_coeff = 0`, which zeros the
        // virtual half regardless of `stage1_point` / `w_eval`. Skip the
        // EqPolynomial eval and the `w * (w + 1)` round in that case.
        if self.batching_coeff.is_zero() {
            return Ok(relation_oracle + trace_oracle);
        }
        let eq_val = EqPolynomial::mle(&self.stage1_point, challenges)?;
        let virtual_oracle = eq_val * w_eval * (w_eval + E::one());
        Ok(self.batching_coeff * virtual_oracle + relation_oracle + trace_oracle)
    }
}

#[cfg(test)]
mod tests {
    use super::{cleartext_source_eval, Stage2CleartextSource};
    use akita_field::{AkitaError, FieldCore};
    use akita_field::{FpExt2, NegOneNr, Prime128Offset275};
    use akita_sumcheck::multilinear_eval;

    type F = Prime128Offset275;
    type E = FpExt2<F, NegOneNr>;
    const D: usize = 4;

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
    fn logical_digits_eval_matches_materialized_table() {
        let d = 4usize;
        let w_digits = vec![3, -1, 2, 0, -2, 1, 4, -3, 1, 0, -4, 2];
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
        let expected = multilinear_eval(&w_evals, &challenges).expect("matching table shape");
        let source = Stage2CleartextSource::LogicalDigits(std::borrow::Cow::Borrowed(&w_digits));
        let actual = cleartext_source_eval::<F, F, 4>(
            w_digits.len(),
            &source,
            &challenges,
            col_bits,
            ring_bits,
        )
        .expect("valid logical digits");
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
        let source = Stage2CleartextSource::FieldElements(&field_witness);
        let actual =
            cleartext_source_eval::<F, E, D>(field_witness.len(), &source, &challenges, 1, 1)
                .expect("valid witness");

        assert_eq!(actual, expected);
    }

    #[test]
    fn logical_digits_eval_rejects_challenge_dimension_mismatch() {
        let w_digits = vec![1i8, -1, 0, 2];
        let source = Stage2CleartextSource::LogicalDigits(std::borrow::Cow::Borrowed(&w_digits));
        let err = cleartext_source_eval::<F, E, D>(1, &source, &[E::zero()], 1, 1)
            .expect_err("wrong arity");
        assert!(matches!(err, AkitaError::InvalidSize { .. }));
    }

    #[test]
    fn logical_digits_eval_rejects_length_mismatch() {
        let w_digits = vec![1i8, -1, 0, 2];
        let challenges = vec![E::zero(), E::zero()];
        let source = Stage2CleartextSource::LogicalDigits(std::borrow::Cow::Borrowed(&w_digits));
        let err = cleartext_source_eval::<F, E, D>(8, &source, &challenges, 1, 1)
            .expect_err("witness length mismatch");
        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
