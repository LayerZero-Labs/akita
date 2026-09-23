use super::handles_and_relation::ConsumerRelationWitness;
use crate::opaque::sumcheck::{digit_range, relation_range_image};
use akita_error::AkitaError;
use digit_range::DigitRangeProver;
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

pub struct CpuStage2SessionHandle<E: Field> {
    binding: crate::opaque::OperationBinding,
    lease: Option<crate::opaque::ScopeLease>,
    prover: relation_range_image::RelationRangeImageProver<E>,
    claim: E,
    next_round: usize,
    pub(super) pending: Option<akita_algebra::uni_poly::UniPoly<E>>,
}

#[cfg(test)]
impl<E: Field> CpuStage2SessionHandle<E> {
    pub(super) fn for_test(
        prover: relation_range_image::RelationRangeImageProver<E>,
        claim: E,
    ) -> Self {
        Self {
            binding: crate::opaque::OperationBinding::unbound(),
            lease: None,
            prover,
            claim,
            next_round: 0,
            pending: None,
        }
    }
}

pub struct CpuStage1SessionHandle<E: Field> {
    pub(crate) binding: crate::opaque::OperationBinding,
    pub(crate) lease: crate::opaque::ScopeLease,
    pub(crate) session_state: digit_range::DigitRangeSession<E>,
}

pub(crate) struct CpuExtensionOpeningSession<E: Field> {
    prover: crate::opaque::recursive::opening::ExtensionOpeningReductionProver<E>,
    claim: E,
    next_round: usize,
    pending: Option<akita_algebra::uni_poly::UniPoly<E>>,
    num_terms: usize,
}

pub(crate) type ConsumerStage2Session<E> = CpuStage2SessionHandle<E>;

impl<E: Field> CpuStage1SessionHandle<E> {
    pub(crate) fn operation_binding(&self) -> crate::opaque::OperationBinding {
        self.binding.clone()
    }

    pub(crate) const fn scope_lease(&self) -> &crate::opaque::ScopeLease {
        &self.lease
    }
}
impl<E: Field> CpuStage2SessionHandle<E> {
    pub(crate) fn operation_binding(&self) -> crate::opaque::OperationBinding {
        self.binding.clone()
    }

    pub(crate) fn set_operation_binding(
        &mut self,
        binding: crate::opaque::OperationBinding,
        lease: crate::opaque::ScopeLease,
    ) {
        self.binding = binding;
        self.lease = Some(lease);
    }

    pub(crate) fn scope_lease(&self) -> Result<&crate::opaque::ScopeLease, AkitaError> {
        self.lease.as_ref().ok_or_else(|| {
            AkitaError::InvalidInput("Stage 2 session has no proof-scope lease".into())
        })
    }
}

#[cfg(test)]
pub(crate) fn cpu_extension_opening_session<E>(
    group: crate::opaque::recursive::opening::ExtensionOpeningReductionGroup<E>,
    input_claim: E,
) -> Result<Box<dyn crate::opaque::eor::ExtensionOpeningSession<E>>, AkitaError>
where
    E: Field + jolt_field::Unreduced + jolt_field::Fold + 'static,
{
    Ok(Box::new(cpu_witness_eor_session(group, input_claim)?))
}

fn cpu_witness_eor_session<E>(
    group: crate::opaque::recursive::opening::ExtensionOpeningReductionGroup<E>,
    input_claim: E,
) -> Result<CpuExtensionOpeningSession<E>, AkitaError>
where
    E: Field + jolt_field::Unreduced + jolt_field::Fold + 'static,
{
    let num_terms = group.num_terms();
    Ok(CpuExtensionOpeningSession {
        prover: crate::opaque::recursive::opening::ExtensionOpeningReductionProver::new(
            vec![group],
            input_claim,
        )?,
        claim: input_claim,
        next_round: 0,
        pending: None,
        num_terms,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_extension_opening_session_from_witnesses<F, E>(
    witnesses: Vec<Vec<E>>,
    claim_coefficients: &[E],
    tail_point: &[E],
    eta: &[E],
    extra_point: Vec<E>,
    input_claim: E,
) -> Result<Box<dyn crate::opaque::eor::ExtensionOpeningSession<E>>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>
        + jolt_field::Unreduced
        + jolt_field::Fold
        + jolt_field::MulBaseUnreduced<F>
        + 'static,
{
    Ok(Box::new(cpu_witness_eor_session_from_witnesses::<F, E>(
        witnesses,
        claim_coefficients,
        tail_point,
        eta,
        extra_point,
        input_claim,
    )?))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn cpu_witness_eor_session_from_witnesses<F, E>(
    witnesses: Vec<Vec<E>>,
    claim_coefficients: &[E],
    tail_point: &[E],
    eta: &[E],
    extra_point: Vec<E>,
    input_claim: E,
) -> Result<CpuExtensionOpeningSession<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>
        + jolt_field::Unreduced
        + jolt_field::Fold
        + jolt_field::MulBaseUnreduced<F>
        + 'static,
{
    if witnesses.len() != claim_coefficients.len() {
        return Err(AkitaError::InvalidSize {
            expected: witnesses.len(),
            actual: claim_coefficients.len(),
        });
    }
    let terms = witnesses
        .into_iter()
        .zip(claim_coefficients)
        .map(|(witness, &coefficient)| {
            crate::opaque::recursive::opening::ExtensionOpeningReductionTerm::new(
                witness,
                coefficient,
            )
        })
        .collect();
    let factor = akita_types::tensor_equality_factor_evals::<F, E>(tail_point, eta)?;
    let group =
        crate::opaque::recursive::opening::ExtensionOpeningReductionGroup::new(terms, factor)?
            .extend_cylindrically(extra_point)?;
    cpu_witness_eor_session(group, input_claim)
}

impl<E> crate::opaque::eor::ExtensionOpeningSession<E> for CpuExtensionOpeningSession<E>
where
    E: Field + jolt_field::Unreduced + jolt_field::Fold,
{
    fn num_rounds(&self) -> usize {
        akita_sumcheck::SumcheckInstanceProver::num_rounds(&self.prover)
    }

    fn num_terms(&self) -> usize {
        self.num_terms
    }

    fn round_polynomial(
        &mut self,
        round: usize,
        previous_claim: E,
    ) -> Result<akita_algebra::uni_poly::UniPoly<E>, AkitaError> {
        if round != self.next_round
            || round >= self.num_rounds()
            || self.pending.is_some()
            || previous_claim != self.claim
        {
            return Err(AkitaError::InvalidInput(
                "extension-opening session round order or claim mismatch".into(),
            ));
        }
        let polynomial = akita_sumcheck::SumcheckInstanceProver::compute_round_univariate(
            &mut self.prover,
            round,
            previous_claim,
        );
        if polynomial.degree() > akita_types::EXTENSION_OPENING_REDUCTION_DEGREE
            || polynomial.evaluate(&E::zero()) + polynomial.evaluate(&E::one()) != previous_claim
        {
            return Err(AkitaError::InvalidInput(
                "extension-opening session returned an invalid round polynomial".into(),
            ));
        }
        self.pending = Some(polynomial.clone());
        Ok(polynomial)
    }

    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        if round != self.next_round || round >= self.num_rounds() {
            return Err(AkitaError::InvalidInput(
                "extension-opening session challenge order mismatch".into(),
            ));
        }
        let polynomial = self.pending.take().ok_or_else(|| {
            AkitaError::InvalidInput("extension-opening session challenge arrived early".into())
        })?;
        self.claim = polynomial.evaluate(&challenge);
        akita_sumcheck::SumcheckInstanceProver::ingest_challenge(
            &mut self.prover,
            round,
            challenge,
        );
        self.next_round += 1;
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<Vec<(E, E, E)>, AkitaError> {
        if self.pending.is_some() || self.next_round != self.num_rounds() {
            return Err(AkitaError::InvalidInput(
                "extension-opening session finished before all rounds".into(),
            ));
        }
        akita_sumcheck::SumcheckInstanceProver::finalize(&mut self.prover);
        self.prover.final_terms().ok_or_else(|| {
            AkitaError::InvalidInput("extension-opening session has no final claims".into())
        })
    }
}

impl<E> ConsumerStage2Session<E>
where
    E: Field + Ring + jolt_field::Unreduced + jolt_field::Fold + 'static,
{
    pub(crate) fn num_rounds(&self) -> usize {
        akita_sumcheck::SumcheckInstanceProver::num_rounds(&self.prover)
    }

    pub(crate) fn input_claim(&self) -> E {
        self.claim
    }

    pub(crate) fn round_polynomial(
        &mut self,
        round: usize,
        previous_claim: E,
    ) -> Result<akita_algebra::uni_poly::UniPoly<E>, AkitaError> {
        if round != self.next_round
            || round >= self.num_rounds()
            || self.pending.is_some()
            || previous_claim != self.claim
        {
            return Err(AkitaError::InvalidInput(
                "relation session round order or claim mismatch".into(),
            ));
        }
        let polynomial = akita_sumcheck::SumcheckInstanceProver::compute_round_univariate(
            &mut self.prover,
            round,
            previous_claim,
        );
        if polynomial.degree() > 3
            || polynomial.evaluate(&E::zero()) + polynomial.evaluate(&E::one()) != previous_claim
        {
            return Err(AkitaError::InvalidInput(
                "relation session returned an invalid round polynomial".into(),
            ));
        }
        self.pending = Some(polynomial.clone());
        Ok(polynomial)
    }

    pub(crate) fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        if round != self.next_round || round >= self.num_rounds() {
            return Err(AkitaError::InvalidInput(
                "relation session challenge order mismatch".into(),
            ));
        }
        let polynomial = self.pending.take().ok_or_else(|| {
            AkitaError::InvalidInput("relation session challenge arrived early".into())
        })?;
        self.claim = polynomial.evaluate(&challenge);
        akita_sumcheck::SumcheckInstanceProver::ingest_challenge(
            &mut self.prover,
            round,
            challenge,
        );
        self.next_round += 1;
        Ok(())
    }

    pub(crate) fn finish(
        mut self,
    ) -> Result<crate::opaque::RelationWitnessFinalClaims<E>, AkitaError> {
        if self.pending.is_some() || self.next_round != self.num_rounds() {
            return Err(AkitaError::InvalidInput(
                "relation session finished before all rounds".into(),
            ));
        }
        akita_sumcheck::SumcheckInstanceProver::finalize(&mut self.prover);
        if self.claim != self.prover.expected_final_claim()? {
            return Err(AkitaError::InvalidInput(
                "relation session final claim disagrees with its folded oracle".into(),
            ));
        }
        Ok(crate::opaque::RelationWitnessFinalClaims::new(
            self.prover.final_w_eval(),
            self.claim,
        ))
    }
}

impl<E> ConsumerStage2Session<E>
where
    E: Field + Ring + jolt_field::Unreduced + jolt_field::Fold + 'static,
{
    pub(crate) fn new<F>(
        prepared: &crate::opaque::CpuPreparedSetup<F>,
        witness: ConsumerRelationWitness,
        plan: crate::opaque::ValidatedRelationSessionPlan<'_, F, E>,
    ) -> Result<Self, AkitaError>
    where
        F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
        E: akita_types::FpExtEncoding<F> + jolt_field::MulBaseUnreduced<F>,
    {
        if witness.packed.len() != plan.witness_len() {
            return Err(AkitaError::InvalidInput(
                "Stage 2 plan disagrees with its witness manifest".into(),
            ));
        }
        let weights =
            crate::opaque::relation_weights::compile_stage2_weights(&prepared.expanded, &plan)?;
        let additional = (!weights.linear.is_empty() || !weights.binary_intervals.is_empty())
            .then(|| {
                relation_range_image::AdditionalRelationTerms::new(
                    &witness.packed,
                    plan.domain_len(),
                    weights.linear,
                    &weights.binary_intervals,
                    plan.stage1_point(),
                    plan.binary_batching(),
                )
            })
            .transpose()?;
        let ordinary_claim = plan.relation_claim() + plan.physical_l2_claim()
            - additional.as_ref().map_or_else(
                E::zero,
                relation_range_image::AdditionalRelationTerms::input_claim,
            );
        let relation_weights = match weights.ordinary {
            crate::opaque::RelationWeightDescription::QuotientFactored(weights) => {
                relation_range_image::RelationWeightOracle::QuotientFactored(weights)
            }
            crate::opaque::RelationWeightDescription::ReducedEvaluations {
                evaluations,
                live_len,
            } => relation_range_image::RelationWeightOracle::ReducedDense(
                relation_range_image::DenseRelationWeights::new(evaluations, live_len)?,
            ),
        };
        let batching_coefficient = plan.batching_coefficient();
        let stage1_point = plan.stage1_point().to_vec();
        let range_image_evaluation = plan.range_image_evaluation();
        let basis = plan.basis();
        let live_lane_count = plan.live_lane_count();
        let lane_bits = plan.lane_bits();
        let coefficient_bits = plan.coefficient_bits();
        let linear_opening_claim = plan.linear_opening_claim();
        let linear_terms = match plan.into_linear_terms() {
            crate::opaque::Stage2OpeningDescription::EvaluationTrace {
                trace,
                output_scale,
            } => {
                let coefficient_count = trace.relation_coefficient_block_len;
                let weights = relation_range_image::build_evaluation_trace_weights(trace)?;
                relation_range_image::PreparedProverLinearTerms::from_evaluation_trace(
                    &weights,
                    coefficient_count,
                    output_scale,
                )?
            }
            crate::opaque::Stage2OpeningDescription::CoefficientPacking(terms) => {
                let mut terms = terms.into_iter();
                let mut prepared =
                    relation_range_image::PreparedProverLinearTerms::from_coefficient_packing(
                        terms.next().ok_or(AkitaError::InvalidProof)?,
                    )?;
                for term in terms {
                    prepared.merge(
                        relation_range_image::PreparedProverLinearTerms::from_coefficient_packing(
                            term,
                        )?,
                    )?;
                }
                prepared
            }
        };
        let prover = relation_range_image::RelationRangeImageProver::new(
            batching_coefficient,
            witness.packed,
            &stage1_point,
            range_image_evaluation,
            basis,
            relation_weights,
            live_lane_count,
            lane_bits,
            coefficient_bits,
            ordinary_claim,
            linear_terms,
            linear_opening_claim,
            additional,
        )?;
        let claim = akita_sumcheck::SumcheckInstanceProver::input_claim(&prover);
        Ok(Self {
            binding: witness.binding.for_operation(0),
            lease: None,
            prover,
            claim,
            next_round: 0,
            pending: None,
        })
    }
}

impl<E> CpuStage1SessionHandle<E>
where
    E: Field + Ring + jolt_field::Unreduced + jolt_field::Fold + 'static,
{
    pub(crate) fn new(
        binding: crate::opaque::OperationBinding,
        lease: crate::opaque::ScopeLease,
        witness: &ConsumerRelationWitness,
        plan: &crate::opaque::ValidatedStage1Plan<E>,
    ) -> Result<Self, AkitaError> {
        if witness.len() != plan.witness_len() || plan.domain().live_len() != witness.len() {
            return Err(AkitaError::InvalidInput(
                "Stage 1 plan disagrees with its witness or operation context".into(),
            ));
        }
        let session_state = digit_range::DigitRangeSession::new(
            DigitRangeProver::from_packed_digits(
                witness.packed.clone(),
                plan.digit_range(),
                plan.domain(),
                plan.equality(),
            )?,
            plan.physical(),
        )?;
        Ok(Self {
            binding,
            lease,
            session_state,
        })
    }
}
