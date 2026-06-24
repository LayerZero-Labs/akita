//! Verifier for the Akita stage-2 fused sumcheck.

use crate::protocol::ring_switch::RingSwitchDeferredRowEval;
use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
};
use akita_protocol::ids::{AkitaChallengeId, AkitaOpeningId, AkitaPublicId};
use akita_protocol::{stage2_descriptor, LevelRole};
#[cfg(feature = "zk")]
use akita_r1cs::{ZkR1csLinearCombination, ZkRelationAccumulator};
#[cfg(feature = "zk")]
use akita_sumcheck::ZkSumcheckFinalRelation;
use akita_sumcheck::{multilinear_eval, SumcheckInstanceVerifier};
use akita_types::{
    eval_trace_terms_closed, AkitaExpandedSetup, CleartextWitnessProof, FpExtEncoding,
    PackedDigits, RingMultiplierOpeningPoint, RingOpeningPoint, TraceClaim,
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
    /// Lazily indexed packed digits (zk / legacy terminal).
    Packed(&'a PackedDigits),
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
        Stage2CleartextSource::Packed(packed_witness) => {
            if packed_witness.num_elems != physical_w_len
                || D == 0
                || !physical_w_len.is_multiple_of(D)
            {
                return Err(AkitaError::InvalidProof);
            }
            witness_eval_by_index(physical_w_len, challenges, ring_bits, y_len, |idx| {
                packed_witness
                    .digit_at(idx)
                    .map(|digit| E::from_i64(digit as i64))
                    .ok_or(AkitaError::InvalidProof)
            })
        }
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
        #[cfg(feature = "zk")]
        mask: ZkR1csLinearCombination<E>,
    },
}

/// Decode a terminal cleartext witness into the stage-2 digit oracle.
///
/// Segment-typed wire payloads expand to logical `i8` digits once here; stage-2
/// never sees the segment encoding again.
pub(crate) fn stage2_cleartext_oracle<'a, F, E, const D: usize>(
    witness: &'a CleartextWitnessProof<F>,
    physical_w_len: usize,
    lp: &'a akita_types::LevelParams,
    num_segments: usize,
) -> Result<Stage2WitnessOracle<'a, F, E>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: FieldCore,
{
    let source = match witness {
        CleartextWitnessProof::PackedDigits(packed) => {
            if packed.num_elems != physical_w_len {
                return Err(AkitaError::InvalidProof);
            }
            Stage2CleartextSource::Packed(packed)
        }
        CleartextWitnessProof::SegmentTyped(_) => {
            let digits = witness.logical_i8_digits::<D>(lp, num_segments)?;
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
    #[cfg(feature = "zk")]
    s_claim_mask: ZkR1csLinearCombination<E>,
    #[cfg(feature = "zk")]
    relation_claim_mask: ZkR1csLinearCombination<E>,
    #[cfg(feature = "zk")]
    trace_claim_mask: ZkR1csLinearCombination<E>,
    witness_oracle: Stage2WitnessOracle<'a, F, E>,
    stage1_point: Vec<E>,
    alpha_evals_y: Vec<E>,
    prepared_row_eval: RingSwitchDeferredRowEval<E>,
    setup_claim: Option<E>,
    setup: &'a AkitaExpandedSetup<F>,
    opening_point: &'a RingOpeningPoint<F>,
    ring_multiplier_point: &'a RingMultiplierOpeningPoint<F, D>,
    alpha: E,
    col_bits: usize,
    ring_bits: usize,
    relation_claim: E,
    level_role: LevelRole,
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
        #[cfg(feature = "zk")] s_claim_mask: ZkR1csLinearCombination<E>,
        #[cfg(feature = "zk")] relation_claim_mask: ZkR1csLinearCombination<E>,
        #[cfg(feature = "zk")] trace_claim_mask: ZkR1csLinearCombination<E>,
        witness_oracle: Stage2WitnessOracle<'a, F, E>,
        stage1_point: Vec<E>,
        alpha_evals_y: Vec<E>,
        prepared_row_eval: RingSwitchDeferredRowEval<E>,
        setup_claim: Option<E>,
        setup: &'a AkitaExpandedSetup<F>,
        opening_point: &'a RingOpeningPoint<F>,
        ring_multiplier_point: &'a RingMultiplierOpeningPoint<F, D>,
        relation_claim: E,
        alpha: E,
        col_bits: usize,
        ring_bits: usize,
        level_role: LevelRole,
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
            #[cfg(feature = "zk")]
            s_claim_mask,
            #[cfg(feature = "zk")]
            relation_claim_mask,
            #[cfg(feature = "zk")]
            trace_claim_mask,
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
            level_role,
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

    fn expected_output_from_descriptor(&self, challenges: &[E]) -> Result<E, AkitaError> {
        let descriptor = stage2_descriptor(self.num_rounds(), self.level_role);
        let (y_challenges, x_challenges) = challenges.split_at(self.ring_bits);

        descriptor.try_evaluate(
            |opening| match opening {
                AkitaOpeningId::Witness => self.witness_eval(challenges),
            },
            |challenge| match challenge {
                AkitaChallengeId::BatchingCoeff => Ok(self.batching_coeff),
            },
            |public| match public {
                AkitaPublicId::EqStage1Point => EqPolynomial::mle(&self.stage1_point, challenges),
                AkitaPublicId::Alpha => multilinear_eval(&self.alpha_evals_y, y_challenges),
                AkitaPublicId::RelationRow => self.row_eval(x_challenges),
            },
        )
    }

    fn trace_oracle_at(&self, challenges: &[E]) -> Result<E, AkitaError> {
        let Some(trace) = &self.trace else {
            return Ok(E::zero());
        };
        let (y_challenges, x_challenges) = challenges.split_at(self.ring_bits);
        let w_eval = self.witness_eval(challenges)?;
        let trace_weight = eval_trace_terms_closed::<F, E, D>(
            &trace.layout,
            y_challenges,
            x_challenges,
            &trace.trace_terms,
        )?;
        Ok(trace.trace_coeff * w_eval * trace_weight)
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
        let _span = tracing::info_span!("stage2_descriptor_eval").entered();
        let fused = self.expected_output_from_descriptor(challenges)?;
        let trace_oracle = self.trace_oracle_at(challenges)?;
        Ok(fused + trace_oracle)
    }
}

#[cfg(feature = "zk")]
impl<'a, F, E, const D: usize> ZkSumcheckFinalRelation<E> for AkitaStage2Verifier<'a, F, E, D>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: ExtField<F> + FpExtEncoding<F> + FromPrimitiveInt,
{
    /// Record the deferred relation tying the stage-2 masked input to the
    /// stage-1 masked `s_claim` handoff.
    fn initial_claim_mask(
        &self,
        _relations: &mut ZkRelationAccumulator<E>,
    ) -> Result<ZkR1csLinearCombination<E>, AkitaError> {
        let mut input_mask = ZkR1csLinearCombination::zero();
        input_mask.add_scaled(self.batching_coeff, &self.s_claim_mask);
        input_mask.add_scaled(E::one(), &self.relation_claim_mask);
        input_mask.add_scaled(E::one(), &self.trace_claim_mask);
        Ok(input_mask)
    }

    fn record_input_relation(
        &self,
        _masked_input_claim: E,
        _masked_round_sum: E,
        _round_sum_mask: &ZkR1csLinearCombination<E>,
        _relations: &mut ZkRelationAccumulator<E>,
    ) -> Result<(), AkitaError> {
        // Compressed sumcheck omits the linear term and reconstructs it from the
        // incoming masked claim, so the first-round chain equation has no
        // independent witness content to record here.
        Ok(())
    }

    fn record_final_relation(
        &self,
        challenges: &[E],
        final_claim: ZkR1csLinearCombination<E>,
        relations: &mut ZkRelationAccumulator<E>,
    ) -> Result<(), AkitaError> {
        let eq_val = EqPolynomial::mle(&self.stage1_point, challenges)?;
        let (y_challenges, x_challenges) = challenges.split_at(self.ring_bits);
        let alpha_val = multilinear_eval(&self.alpha_evals_y, y_challenges)?;
        let row_val = self.row_eval(x_challenges)?;
        let trace_val = if let Some(trace) = &self.trace {
            let trace_weight = eval_trace_terms_closed::<F, E, D>(
                &trace.layout,
                y_challenges,
                x_challenges,
                &trace.trace_terms,
            )?;
            trace.trace_coeff * trace_weight
        } else {
            E::zero()
        };

        // At the sampled point r = (r_y, r_x), the fused Stage-2 oracle is
        //
        //   gamma * eq(stage1_point, r) * w(r) * (w(r) + 1)
        //     + w(r) * alpha(r_y) * row(r_x).
        //
        // `final_claim` is already the unmasked final sumcheck claim as an LC.
        // If the next witness evaluation was public-masked, `w_lc` is
        // eval_masked - eval_mask; otherwise it is a constant direct witness
        // evaluation. The R1CS row below records the oracle equality as
        //
        //   w(r) * [gamma * eq(stage1_point, r) * w(r)
        //     + gamma * eq(stage1_point, r) + alpha(r_y) * row(r_x)]
        //     = final_claim.
        let w_lc = match &self.witness_oracle {
            Stage2WitnessOracle::Cleartext { .. } => {
                ZkR1csLinearCombination::constant(self.witness_eval(challenges)?)
            }
            Stage2WitnessOracle::ClaimedEval { eval, mask } => {
                ZkRelationAccumulator::unmask_lc(*eval, mask)
            }
        };
        let mut scaled_virtual = ZkR1csLinearCombination::zero();
        scaled_virtual.add_scaled(self.batching_coeff * eq_val, &w_lc);
        scaled_virtual.constant += self.batching_coeff * eq_val + alpha_val * row_val + trace_val;
        relations.push_r1cs("stage-2 final oracle", w_lc, scaled_virtual, final_claim)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{cleartext_source_eval, Stage2CleartextSource};
    use akita_field::{AkitaError, FieldCore};
    use akita_field::{FpExt2, NegOneNr, Prime128OffsetA7F7};
    use akita_protocol::ids::AkitaPublicId;
    use akita_protocol::{stage2_descriptor, LevelRole};
    use akita_sumcheck::multilinear_eval;
    use akita_types::PackedDigits;

    type F = Prime128OffsetA7F7;
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
        let source = Stage2CleartextSource::Packed(&packed);
        let actual = cleartext_source_eval::<F, F, 4>(
            w_digits.len(),
            &source,
            &challenges,
            col_bits,
            ring_bits,
        )
        .expect("valid packed witness");

        assert_eq!(actual, expected);
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
    fn packed_witness_eval_rejects_challenge_dimension_mismatch() {
        let packed = PackedDigits::from_i8_digits(&[1, -1, 0, 2], 3);
        let source = Stage2CleartextSource::Packed(&packed);
        let err = cleartext_source_eval::<F, E, D>(1, &source, &[E::zero()], 1, 1)
            .expect_err("wrong arity");
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
        let source = Stage2CleartextSource::Packed(&packed);
        let err = cleartext_source_eval::<F, E, D>(4, &source, &challenges, 1, 1)
            .expect_err("truncated packed witness");
        assert!(matches!(err, AkitaError::InvalidProof));
    }

    #[test]
    fn terminal_descriptor_evaluates_relation_only_without_virtual_sources() {
        let descriptor = stage2_descriptor(3, LevelRole::Terminal);
        let got = descriptor
            .try_evaluate(
                |_opening| Ok(F::from_u64(7)),
                |_challenge| {
                    Err(AkitaError::InvalidInput(
                        "batching coeff must not resolve at terminal".to_string(),
                    ))
                },
                |public| match public {
                    AkitaPublicId::EqStage1Point => Err(AkitaError::InvalidInput(
                        "eq must not resolve at terminal".to_string(),
                    )),
                    AkitaPublicId::Alpha => Ok(F::from_u64(11)),
                    AkitaPublicId::RelationRow => Ok(F::from_u64(13)),
                },
            )
            .expect("relation-only summand resolves");
        assert_eq!(got, F::from_u64(7) * F::from_u64(11) * F::from_u64(13));
    }

    #[test]
    fn intermediate_descriptor_matches_legacy_fused_equation() {
        let gamma = F::from_u64(17);
        let w = F::from_u64(7);
        let eq = F::from_u64(11);
        let alpha = F::from_u64(13);
        let row = F::from_u64(19);

        let descriptor = stage2_descriptor(3, LevelRole::Intermediate);
        let via_descriptor = descriptor
            .try_evaluate(
                |_opening| Ok(w),
                |_challenge| Ok(gamma),
                |public| match public {
                    AkitaPublicId::EqStage1Point => Ok(eq),
                    AkitaPublicId::Alpha => Ok(alpha),
                    AkitaPublicId::RelationRow => Ok(row),
                },
            )
            .expect("fused descriptor resolves");

        let legacy = gamma * eq * w * (w + F::one()) + w * alpha * row;
        assert_eq!(via_descriptor, legacy);
    }
}
