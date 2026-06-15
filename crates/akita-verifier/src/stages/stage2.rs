//! Verifier for the Akita stage-2 fused sumcheck.

use crate::protocol::ring_switch::RingSwitchDeferredRowEval;
use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
};
#[cfg(feature = "zk")]
use akita_r1cs::{ZkR1csLinearCombination, ZkRelationAccumulator};
#[cfg(feature = "zk")]
use akita_sumcheck::ZkSumcheckFinalRelation;
use akita_sumcheck::{multilinear_eval, SumcheckInstanceVerifier};
use akita_types::{
    eval_trace_terms_closed, AkitaExpandedSetup, CleartextWitnessProof, FpExtEncoding,
    RingMultiplierOpeningPoint, RingOpeningPoint, TraceClaim,
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

#[allow(clippy::too_many_arguments)]
fn cleartext_witness_eval<F, E, const D: usize>(
    cleartext_witness: &CleartextWitnessProof<F>,
    physical_w_len: usize,
    challenges: &[E],
    col_bits: usize,
    ring_bits: usize,
    lp: Option<&akita_types::LevelParams>,
    _num_claims: usize,
    num_commitment_groups: usize,
    preexpanded_digits: Option<&[i8]>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
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
    match cleartext_witness {
        CleartextWitnessProof::PackedDigits(packed_witness) => {
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
        CleartextWitnessProof::SegmentTyped(witness) => {
            let lp = lp.ok_or(AkitaError::InvalidProof)?;
            let digits: Cow<'_, [i8]> = if let Some(cached) = preexpanded_digits {
                Cow::Borrowed(cached)
            } else {
                let (num_w_vectors, num_t_vectors, num_public_rows) =
                    akita_types::tail_segment_multiplicities_from_layout(lp, &witness.layout)?;
                Cow::Owned(cleartext_witness.logical_i8_digits::<D>(
                    lp,
                    num_w_vectors,
                    num_t_vectors,
                    num_public_rows,
                    num_commitment_groups,
                )?)
            };
            if digits.len() != physical_w_len || D == 0 || !physical_w_len.is_multiple_of(D) {
                return Err(AkitaError::InvalidProof);
            }
            witness_eval_by_index(physical_w_len, challenges, ring_bits, y_len, |idx| {
                Ok(E::from_i64(digits[idx] as i64))
            })
        }
        CleartextWitnessProof::FieldElements(field_witness) => {
            let field_witness = field_witness.coeffs();
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
        witness: &'a CleartextWitnessProof<F>,
        physical_w_len: usize,
        lp: Option<&'a akita_types::LevelParams>,
        num_claims: usize,
        num_commitment_groups: usize,
        /// Pre-expanded digit hypercube for segment-typed terminal witnesses.
        /// Built once at oracle construction; packed digits stay lazily indexed.
        expanded_i8_digits: Option<Vec<i8>>,
    },
    ClaimedEval {
        eval: E,
        #[cfg(feature = "zk")]
        mask: ZkR1csLinearCombination<E>,
    },
}

/// Build a cleartext stage-2 oracle, expanding segment-typed witnesses once.
pub(crate) fn stage2_cleartext_oracle<'a, F, E, const D: usize>(
    witness: &'a CleartextWitnessProof<F>,
    physical_w_len: usize,
    lp: &'a akita_types::LevelParams,
    num_claims: usize,
    num_commitment_groups: usize,
) -> Result<Stage2WitnessOracle<'a, F, E>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: FieldCore,
{
    let expanded_i8_digits = if witness.as_segment_typed().is_some() {
        let (num_w_vectors, num_t_vectors, num_public_rows) =
            akita_types::tail_segment_multiplicities_from_layout(
                lp,
                &witness
                    .as_segment_typed()
                    .ok_or(AkitaError::InvalidProof)?
                    .layout,
            )?;
        let digits = witness.logical_i8_digits::<D>(
            lp,
            num_w_vectors,
            num_t_vectors,
            num_public_rows,
            num_commitment_groups,
        )?;
        if digits.len() != physical_w_len {
            return Err(AkitaError::InvalidProof);
        }
        Some(digits)
    } else {
        None
    };
    Ok(Stage2WitnessOracle::Cleartext {
        witness,
        physical_w_len,
        lp: Some(lp),
        num_claims,
        num_commitment_groups,
        expanded_i8_digits,
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
            trace,
            _marker: PhantomData,
        })
    }

    fn witness_eval(&self, challenges: &[E]) -> Result<E, AkitaError> {
        match &self.witness_oracle {
            Stage2WitnessOracle::Cleartext {
                witness,
                physical_w_len,
                lp,
                num_claims,
                num_commitment_groups,
                expanded_i8_digits,
            } => cleartext_witness_eval::<F, E, D>(
                witness,
                *physical_w_len,
                challenges,
                self.col_bits,
                self.ring_bits,
                *lp,
                *num_claims,
                *num_commitment_groups,
                expanded_i8_digits.as_deref(),
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
    use super::cleartext_witness_eval;
    use akita_field::{AkitaError, FieldCore};
    use akita_field::{FpExt2, NegOneNr, Prime128Offset275};
    use akita_sumcheck::multilinear_eval;
    use akita_types::{CleartextWitnessProof, FlatRingVec, PackedDigits};

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
        let actual = cleartext_witness_eval::<F, F, 4>(
            &CleartextWitnessProof::PackedDigits(packed),
            w_digits.len(),
            &challenges,
            col_bits,
            ring_bits,
            None,
            1,
            1,
            None,
        )
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
        let actual = cleartext_witness_eval::<F, E, D>(
            &CleartextWitnessProof::FieldElements(FlatRingVec::from_coeffs(field_witness.clone())),
            field_witness.len(),
            &challenges,
            1,
            1,
            None,
            1,
            1,
            None,
        )
        .expect("valid witness");

        assert_eq!(actual, expected);
    }

    #[test]
    fn packed_witness_eval_rejects_challenge_dimension_mismatch() {
        let packed = PackedDigits::from_i8_digits(&[1, -1, 0, 2], 3);
        let err = cleartext_witness_eval::<F, E, D>(
            &CleartextWitnessProof::PackedDigits(packed),
            1,
            &[E::zero()],
            1,
            1,
            None,
            1,
            1,
            None,
        )
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
        let err = cleartext_witness_eval::<F, E, D>(
            &CleartextWitnessProof::PackedDigits(packed),
            4,
            &challenges,
            1,
            1,
            None,
            1,
            1,
            None,
        )
        .expect_err("truncated packed witness");
        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
