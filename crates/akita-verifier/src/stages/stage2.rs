//! Verifier for the Akita stage-2 fused sumcheck.

use crate::protocol::relation_weight::PreparedRelationWeightPolynomial;
use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
};
use akita_sumcheck::{multilinear_eval, SumcheckInstanceVerifier};
use akita_types::{
    AkitaExpandedSetup, CleartextWitnessProof, FpExtEncoding, RingMultiplierOpeningPoint,
    RingOpeningPoint,
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
            let digits = akita_types::dispatch_ring_dim_result!(d_a, |D| {
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
    prepared_relation_weight: PreparedRelationWeightPolynomial<F, E, D>,
    relation_weight_claim: E,
    setup_claim: Option<E>,
    setup: &'a AkitaExpandedSetup<F>,
    opening_point: &'a RingOpeningPoint<F>,
    ring_multiplier_point: &'a RingMultiplierOpeningPoint<F>,
    col_bits: usize,
    ring_bits: usize,
    _marker: PhantomData<E>,
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
        prepared_relation_weight: PreparedRelationWeightPolynomial<F, E, D>,
        relation_weight_claim: E,
        setup_claim: Option<E>,
        setup: &'a AkitaExpandedSetup<F>,
        opening_point: &'a RingOpeningPoint<F>,
        ring_multiplier_point: &'a RingMultiplierOpeningPoint<F>,
    ) -> Result<Self, AkitaError> {
        let col_bits = prepared_relation_weight.col_bits;
        let ring_bits = prepared_relation_weight.ring_bits;
        let num_rounds = col_bits.checked_add(ring_bits).ok_or_else(|| {
            AkitaError::InvalidSetup("stage-2 variable count overflow".to_string())
        })?;
        if stage1_point.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: stage1_point.len(),
            });
        }
        Ok(Self {
            batching_coeff,
            s_claim,
            witness_oracle,
            stage1_point,
            prepared_relation_weight,
            relation_weight_claim,
            setup_claim,
            setup,
            opening_point,
            ring_multiplier_point,
            col_bits,
            ring_bits,
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
        self.batching_coeff * self.s_claim + self.relation_weight_claim
    }

    #[tracing::instrument(skip_all, name = "stage2_expected_output_claim")]
    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        let w_eval = {
            let _span = tracing::info_span!("stage2_witness_eval").entered();
            self.witness_eval(challenges)?
        };
        let relation_weight_eval = {
            let _span = tracing::info_span!("stage2_relation_weight_eval").entered();
            self.prepared_relation_weight.eval_at_point(
                challenges,
                self.setup,
                self.opening_point,
                self.ring_multiplier_point,
                self.setup_claim,
            )?
        };
        let relation_term = w_eval * relation_weight_eval;

        if self.batching_coeff.is_zero() {
            return Ok(relation_term);
        }
        let eq_val = EqPolynomial::mle(&self.stage1_point, challenges)?;
        let virtual_oracle = eq_val * w_eval * (w_eval + E::one());
        Ok(self.batching_coeff * virtual_oracle + relation_term)
    }
}

#[cfg(test)]
mod tests {
    use super::{cleartext_source_eval, Stage2CleartextSource};
    use akita_field::{AkitaError, FieldCore, LiftBase};
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
        let live_x_cols = w.len() / d;
        let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
        let ring_bits = d.trailing_zeros() as usize;
        Ok((w.to_vec(), col_bits, ring_bits))
    }

    #[test]
    fn cleartext_field_elements_match_multilinear_eval() {
        let w: Vec<F> = (0..8).map(|i| F::from_u64(i as u64 + 1)).collect();
        let (evals, col_bits, ring_bits) = build_w_evals(&w, D).unwrap();
        let challenges: Vec<E> = (0..col_bits + ring_bits)
            .map(|i| E::from_u64(i as u64 + 3))
            .collect();
        let got = cleartext_source_eval::<F, E, D>(
            w.len(),
            &Stage2CleartextSource::FieldElements(&w),
            &challenges,
            col_bits,
            ring_bits,
        )
        .unwrap();
        let expected = multilinear_eval(
            &evals.iter().map(|x| E::lift_base(*x)).collect::<Vec<_>>(),
            &challenges,
        )
        .unwrap();
        assert_eq!(got, expected);
    }
}
