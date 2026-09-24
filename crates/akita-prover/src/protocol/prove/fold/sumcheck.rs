use super::*;
use jolt_poly::UnivariatePoly;

pub(super) fn prove_stage1<F, E, T, B>(
    ctx: &crate::backend::OperationCtx<'_, F, B>,
    transcript: &mut T,
    level: u32,
    rs: &mut RingSwitchOutput<E, B::RelationHandle>,
    lp: &CommittedGroupParams,
    plan: &RelationRangeImagePlan,
) -> Result<Stage1ProveOutput<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Send + Sync + 'static,
    E: ExtField<F> + Unreduced + Fold + Ring + AkitaSerialize + Send + Sync + 'static,
    T: akita_types::ProverTranscriptGrinding<F>,
    B: crate::backend::OpaqueStage1Kernel<F, E>,
{
    let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
    let domain = plan.digit_witness_domain();
    if plan.relation_address_geometry() != rs.relation_address_geometry
        || domain.live_len() != rs.witness_len
        || plan.digit_range_plan().basis() != rs.b
    {
        return Err(AkitaError::InvalidSetup(
            "ring-switch output disagrees with the relation/range-image plan".into(),
        ));
    }
    let digit_range_equality_col_bits = rs
        .tau0
        .len()
        .checked_sub(rs.digit_range_equality_low_variable_count)
        .ok_or_else(|| AkitaError::InvalidSetup("digit-range equality width overflow".into()))?;
    let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
        &rs.tau0,
        digit_range_equality_col_bits,
        rs.digit_range_equality_low_variable_count,
    )?;
    let physical_plan = PhysicalResponsePlan::new(lp, plan)?;
    let mut equality_coordinates = equality_point.clone().into_coordinates();
    let stage1_plan = crate::backend::ValidatedStage1Plan::new(
        plan.digit_range_plan(),
        domain,
        equality_point,
        physical_plan.clone(),
        rs.witness_len,
    );
    let mut session_handle = crate::backend::OpaqueStage1Kernel::begin_stage1(
        ctx.backend(),
        &rs.relation_handle,
        &stage1_plan,
    )?;
    let product_count = plan.digit_range_plan().product_stage_arities().len();
    let rounds = domain.num_vars();
    let mut stage_proofs = Vec::with_capacity(product_count + usize::from(physical_plan.is_none()));
    let mut claim = E::zero();
    for stage_index in 0..product_count {
        let shape = plan
            .digit_range_plan()
            .stage_shape(rounds, stage_index)
            .ok_or(AkitaError::InvalidProof)?;
        transcript.append_serde(akita_transcript::labels::ABSORB_SUMCHECK_CLAIM, &claim);
        let step = crate::backend::Stage1Step::Product(stage_index);
        let mut polynomials = Vec::with_capacity(rounds);
        let mut challenges = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let polynomial = crate::backend::OpaqueStage1Kernel::stage1_round_polynomial(
                ctx.backend(),
                &mut session_handle,
                step,
                round,
                claim,
            )?;
            let crate::backend::Stage1RoundPolynomial::EqFactored(polynomial) = polynomial else {
                return Err(AkitaError::InvalidProof);
            };
            if polynomial.degree() != shape.sumcheck_proof.1 {
                return Err(AkitaError::InvalidProof);
            }
            transcript.append_serde(akita_transcript::labels::ABSORB_SUMCHECK_ROUND, &polynomial);
            let challenge = akita_types::sample_grinded_sumcheck_challenge::<F, E, T>(
                transcript,
                akita_types::SumcheckProtocol::Stage1,
                level,
                u32::try_from(stage_index)
                    .map_err(|_| AkitaError::InvalidSetup("Stage 1 index overflow".into()))?,
                u32::try_from(round)
                    .map_err(|_| AkitaError::InvalidSetup("Stage 1 round overflow".into()))?,
            )?;
            claim = akita_sumcheck::advance_eq_factored_claim(
                claim,
                *equality_coordinates
                    .get(round)
                    .ok_or(AkitaError::InvalidProof)?,
                &polynomial,
                challenge,
            );
            crate::backend::OpaqueStage1Kernel::bind_stage1_challenge(
                ctx.backend(),
                &mut session_handle,
                step,
                round,
                challenge,
            )?;
            polynomials.push(polynomial);
            challenges.push(challenge);
        }
        let crate::backend::Stage1PublicTransition::ProductChildClaims(child_claims) =
            crate::backend::OpaqueStage1Kernel::stage1_public_transition(
                ctx.backend(),
                &mut session_handle,
                step,
            )?
        else {
            return Err(AkitaError::InvalidProof);
        };
        if child_claims.len() != shape.child_claims {
            return Err(AkitaError::InvalidProof);
        }
        akita_types::append_digit_range_child_claims::<F, E, T>(&child_claims, transcript);
        transcript.grind_query(akita_types::GrindingSite::Stage1InterstageBatch {
            level,
            stage: u32::try_from(stage_index)
                .map_err(|_| AkitaError::InvalidSetup("Stage 1 index overflow".into()))?,
        })?;
        let gamma = sample_ext_challenge::<F, E, T>(
            transcript,
            akita_transcript::labels::CHALLENGE_SUMCHECK_INTERSTAGE_BATCH,
        );
        let weights = plan
            .digit_range_plan()
            .interstage_batch_weights(gamma, child_claims.len());
        claim = plan
            .digit_range_plan()
            .batch_claims(&weights, &child_claims)?;
        crate::backend::OpaqueStage1Kernel::bind_stage1_batch_challenge(
            ctx.backend(),
            &mut session_handle,
            crate::backend::Stage1Transition::ProductBatch(stage_index),
            gamma,
        )?;
        equality_coordinates = challenges;
        stage_proofs.push(akita_types::AkitaStage1StageProof {
            sumcheck_proof: akita_sumcheck::EqFactoredSumcheckProof {
                round_polys: polynomials,
            },
            child_claims,
        });
    }

    let mut final_point = Vec::with_capacity(rounds);
    let (range_image_evaluation, norm_proof) = if physical_plan.is_some() {
        let step = crate::backend::Stage1Step::FusedRangeNorm;
        let crate::backend::Stage1PublicTransition::PhysicalL2Claims {
            response_l2_sq,
            subclaims,
        } = crate::backend::OpaqueStage1Kernel::stage1_public_transition(
            ctx.backend(),
            &mut session_handle,
            step,
        )?
        else {
            return Err(AkitaError::InvalidProof);
        };
        let physical = physical_plan.as_ref().ok_or(AkitaError::InvalidProof)?;
        if subclaims.len()
            != physical
                .shape()
                .subclaim_count()
                .ok_or(AkitaError::InvalidProof)?
        {
            return Err(AkitaError::InvalidProof);
        }
        transcript.append_serde(
            akita_transcript::labels::ABSORB_L2_NORM_INTEGER,
            &response_l2_sq,
        );
        for subclaim in &subclaims {
            transcript.append_serde(akita_transcript::labels::ABSORB_L2_NORM_SUBCLAIM, subclaim);
        }
        let norm_claim = if subclaims.is_empty() {
            E::from_u128(response_l2_sq)
        } else {
            transcript.grind_query(akita_types::GrindingSite::L2SubclaimBatch { level })?;
            let gamma = sample_ext_challenge::<F, E, T>(
                transcript,
                akita_transcript::labels::CHALLENGE_L2_NORM_BATCH,
            );
            crate::backend::OpaqueStage1Kernel::bind_stage1_batch_challenge(
                ctx.backend(),
                &mut session_handle,
                crate::backend::Stage1Transition::L2SubclaimBatch,
                gamma,
            )?;
            let mut power = E::one();
            subclaims.iter().fold(E::zero(), |sum, &subclaim| {
                let next = sum + power * subclaim;
                power *= gamma;
                next
            })
        };
        transcript.grind_query(akita_types::GrindingSite::L2NormMerge { level })?;
        let merge = sample_ext_challenge::<F, E, T>(
            transcript,
            akita_transcript::labels::CHALLENGE_L2_NORM_MERGE,
        );
        crate::backend::OpaqueStage1Kernel::bind_stage1_batch_challenge(
            ctx.backend(),
            &mut session_handle,
            crate::backend::Stage1Transition::RangeNormMerge,
            merge,
        )?;
        claim += merge * norm_claim;
        transcript.append_serde(akita_transcript::labels::ABSORB_SUMCHECK_CLAIM, &claim);
        let mut round_polys = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let polynomial = crate::backend::OpaqueStage1Kernel::stage1_round_polynomial(
                ctx.backend(),
                &mut session_handle,
                step,
                round,
                claim,
            )?;
            let crate::backend::Stage1RoundPolynomial::Standard(polynomial) = polynomial else {
                return Err(AkitaError::InvalidProof);
            };
            if polynomial.coefficients().len() != plan.digit_range_plan().leaf_degree() + 2
                || polynomial.evaluate(E::zero()) + polynomial.evaluate(E::one()) != claim
            {
                return Err(AkitaError::InvalidProof);
            }
            let compressed = polynomial.compress();
            transcript.append_serde(akita_transcript::labels::ABSORB_SUMCHECK_ROUND, &compressed);
            let challenge = akita_types::sample_grinded_sumcheck_challenge::<F, E, T>(
                transcript,
                akita_types::SumcheckProtocol::PhysicalL2,
                level,
                0,
                u32::try_from(round)
                    .map_err(|_| AkitaError::InvalidSetup("physical L2 round overflow".into()))?,
            )?;
            claim = compressed.eval_from_hint(&claim, &challenge);
            crate::backend::OpaqueStage1Kernel::bind_stage1_challenge(
                ctx.backend(),
                &mut session_handle,
                step,
                round,
                challenge,
            )?;
            round_polys.push(compressed);
            final_point.push(challenge);
        }
        let crate::backend::Stage1PublicTransition::Final {
            range_image_evaluation,
            virtual_evaluations,
        } = crate::backend::OpaqueStage1Kernel::stage1_public_transition(
            ctx.backend(),
            &mut session_handle,
            step,
        )?
        else {
            return Err(AkitaError::InvalidProof);
        };
        if virtual_evaluations.len() != physical.shape().virtual_evaluation_count() {
            return Err(AkitaError::InvalidProof);
        }
        for evaluation in &virtual_evaluations {
            transcript.append_serde(
                akita_transcript::labels::ABSORB_L2_VIRTUAL_EVALUATION,
                evaluation,
            );
        }
        (
            range_image_evaluation,
            Some(akita_types::PhysicalL2NormProof {
                response_l2_sq,
                subclaims,
                virtual_evaluations,
                sumcheck: akita_sumcheck::SumcheckProof { round_polys },
            }),
        )
    } else {
        let step = crate::backend::Stage1Step::RangeLeaf;
        transcript.append_serde(akita_transcript::labels::ABSORB_SUMCHECK_CLAIM, &claim);
        let mut polynomials = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let polynomial = crate::backend::OpaqueStage1Kernel::stage1_round_polynomial(
                ctx.backend(),
                &mut session_handle,
                step,
                round,
                claim,
            )?;
            let crate::backend::Stage1RoundPolynomial::EqFactored(polynomial) = polynomial else {
                return Err(AkitaError::InvalidProof);
            };
            if polynomial.degree() != plan.digit_range_plan().leaf_degree() {
                return Err(AkitaError::InvalidProof);
            }
            transcript.append_serde(akita_transcript::labels::ABSORB_SUMCHECK_ROUND, &polynomial);
            let challenge = akita_types::sample_grinded_sumcheck_challenge::<F, E, T>(
                transcript,
                akita_types::SumcheckProtocol::Stage1,
                level,
                u32::try_from(product_count)
                    .map_err(|_| AkitaError::InvalidSetup("Stage 1 index overflow".into()))?,
                u32::try_from(round)
                    .map_err(|_| AkitaError::InvalidSetup("Stage 1 round overflow".into()))?,
            )?;
            claim = akita_sumcheck::advance_eq_factored_claim(
                claim,
                *equality_coordinates
                    .get(round)
                    .ok_or(AkitaError::InvalidProof)?,
                &polynomial,
                challenge,
            );
            crate::backend::OpaqueStage1Kernel::bind_stage1_challenge(
                ctx.backend(),
                &mut session_handle,
                step,
                round,
                challenge,
            )?;
            polynomials.push(polynomial);
            final_point.push(challenge);
        }
        let crate::backend::Stage1PublicTransition::Final {
            range_image_evaluation,
            virtual_evaluations,
        } = crate::backend::OpaqueStage1Kernel::stage1_public_transition(
            ctx.backend(),
            &mut session_handle,
            step,
        )?
        else {
            return Err(AkitaError::InvalidProof);
        };
        if !virtual_evaluations.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        stage_proofs.push(akita_types::AkitaStage1StageProof {
            sumcheck_proof: akita_sumcheck::EqFactoredSumcheckProof {
                round_polys: polynomials,
            },
            child_claims: Vec::new(),
        });
        (range_image_evaluation, None)
    };
    let final_claims =
        crate::backend::OpaqueStage1Kernel::finish_stage1(ctx.backend(), session_handle)?;
    if final_claims.final_claim() != claim || final_claims.point() != final_point {
        return Err(AkitaError::InvalidProof);
    }
    let stage1_point = final_claims.point().to_vec();
    let stage1_proof = AkitaStage1Proof {
        stages: stage_proofs,
        range_image_evaluation,
        norm_proof,
    };
    let range_image_evaluation = stage1_proof.range_image_evaluation;
    let physical_l2 = match physical_plan {
        Some(physical_plan) => {
            let norm_proof = stage1_proof
                .norm_proof
                .as_ref()
                .ok_or(AkitaError::InvalidProof)?;
            let InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap, ..
            } = lp.inner().matrix.security_route()
            else {
                return Err(AkitaError::InvalidSetup(
                    "physical L2 plan disagrees with the A security route".into(),
                ));
            };
            if norm_proof.response_l2_sq > response_l2_sq_cap {
                return Err(AkitaError::InvalidInput(
                    "folded response exceeds the scheduled L2 cap".into(),
                ));
            }
            Some(PhysicalL2ProverReplay {
                plan: physical_plan,
                point: stage1_point.clone(),
                virtual_evaluations: norm_proof.virtual_evaluations.clone(),
                batching: Vec::new(),
                claim: E::zero(),
            })
        }
        None => {
            if stage1_proof.norm_proof.is_some() {
                return Err(AkitaError::InvalidInput(
                    "L-infinity route produced an L2 norm proof".into(),
                ));
            }
            None
        }
    };
    Ok(Stage1ProveOutput {
        proof: stage1_proof,
        point: stage1_point,
        range_image_evaluation,
        physical_l2,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_stage2<F, E, T, B>(
    ctx: &crate::backend::OperationCtx<'_, F, B>,
    level: usize,
    transcript: &mut T,
    batching_coeff: E,
    rs: RingSwitchOutput<E, B::RelationHandle>,
    stage1_point: &[E],
    range_image_evaluation: E,
    relation_claim: E,
    binary_batching: E,
    physical_l2: Option<PhysicalL2ProverReplay<E>>,
    linear_terms: Stage2OpeningDescription<'_, E>,
    trace_opening_claim: E,
    plan: &RelationRangeImagePlan,
    relation: crate::backend::RelationWeightRequest<'_, F, E>,
) -> Result<Stage2ProveOutput<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: ExtField<F> + Unreduced + Fold + Ring + AkitaSerialize,
    T: akita_types::ProverTranscriptGrinding<F>,
    B: crate::backend::OpaqueStage2Kernel<F, E>,
{
    let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
    let domain = plan.digit_witness_domain();
    let geometry = rs.relation_address_geometry;
    let live_relation_lane_count = geometry.live_relation_lane_count();
    let relation_lane_variable_count = geometry.relation_lane_variable_count();
    let relation_coefficient_variable_count = geometry.relation_coefficient_variable_count();
    if plan.relation_address_geometry() != geometry
        || domain.live_len() != rs.witness_len
        || plan.digit_range_plan().basis() != rs.b
    {
        return Err(AkitaError::InvalidSetup(
            "ring-switch output disagrees with the relation/range-image plan".into(),
        ));
    }
    let domain_len = domain.domain_len();
    let physical_l2_claim = physical_l2.as_ref().map_or_else(E::zero, |norm| norm.claim);
    let relation_plan = crate::backend::ValidatedRelationSessionPlan::new(
        batching_coeff,
        stage1_point,
        range_image_evaluation,
        plan.digit_range_plan().basis(),
        relation,
        live_relation_lane_count,
        relation_lane_variable_count,
        relation_coefficient_variable_count,
        relation_claim,
        physical_l2_claim,
        linear_terms,
        trace_opening_claim,
        domain_len,
        domain.live_len(),
        physical_l2
            .as_ref()
            .map(|norm| crate::backend::PhysicalL2WeightRequest {
                plan: &norm.plan,
                point: &norm.point,
                batching: &norm.batching,
            }),
        binary_batching,
    );
    let mut session_handle = crate::backend::OpaqueStage2Kernel::begin_stage2(
        ctx.backend(),
        rs.relation_handle,
        relation_plan,
    )
    .map_err(|err| {
        AkitaError::InvalidInput(format!(
            "stage-2 prover initialization failed at fold level {level}: {err}"
        ))
    })?;
    let level = u32::try_from(level)
        .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
    let claim =
        crate::backend::OpaqueStage2Kernel::stage2_input_claim(ctx.backend(), &session_handle)?;
    let num_rounds =
        crate::backend::OpaqueStage2Kernel::stage2_num_rounds(ctx.backend(), &session_handle)?;
    if num_rounds != domain.num_vars()
        || claim
            != batching_coeff * range_image_evaluation
                + relation_claim
                + trace_opening_claim
                + physical_l2_claim
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut kernel = Stage2Sumcheck::<F, E, B> {
        backend: ctx.backend(),
        session: &mut session_handle,
        claim,
        rounds: num_rounds,
        field: std::marker::PhantomData,
    };
    let mut round = 0u32;
    let (proof, sumcheck_challenges, claim) =
        akita_sumcheck::prove_sumcheck::<F, T, E, _, _>(&mut kernel, transcript, |tr| {
            let challenge = akita_types::sample_grinded_sumcheck_challenge::<F, E, T>(
                tr,
                akita_types::SumcheckProtocol::Stage2,
                level,
                0,
                round,
            )?;
            round = round.checked_add(1).ok_or(AkitaError::InvalidProof)?;
            Ok(challenge)
        })?;
    let final_output =
        crate::backend::OpaqueStage2Kernel::finish_stage2(ctx.backend(), session_handle)?;
    if claim != final_output.final_claim() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(Stage2ProveOutput {
        proof,
        challenges: sumcheck_challenges,
        witness_evaluation: final_output.witness_evaluation(),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_stage3<F, E, T, B>(
    backend: &B,
    session: &B::ProofSessionHandle,
    level: usize,
    setup_contribution_mode: SetupContributionMode,
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<F, B::CommitmentHandle>,
    lp: &CommittedGroupParams,
    next_level_params: &CommittedGroupParams,
    instance: &RingRelationInstance<F>,
    tau1: &[E],
    alpha: E,
    sumcheck_challenges: &[E],
    relation_address_geometry: akita_types::RelationAddressGeometry,
    transcript: &mut T,
) -> Result<Option<Stage3ProveOutput<E>>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F>
        + Ring
        + ExtField<F>
        + AkitaSerialize
        + jolt_field::Unreduced
        + jolt_field::MulBaseUnreduced<F>,
    T: akita_types::ProverTranscriptGrinding<F>,
    B: crate::backend::OpaqueStage3Kernel<F, E> + crate::backend::ProverHandleFamily<F, E>,
{
    match setup_contribution_mode {
        SetupContributionMode::Recursive => {
            let _stage3_span = tracing::info_span!(
                "stage3_sumcheck",
                level,
                stage2_rounds = sumcheck_challenges.len(),
                d_a = lp.d_a(),
            )
            .entered();
            let level = u32::try_from(level)
                .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
            let prefix_id = next_level_params
                .setup_prefix()
                .and_then(|group| group.slot_id())
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("Stage 3 requires a setup prefix".into())
                })?;
            let slot = prefix_slots.get(&prefix_id).ok_or_else(|| {
                AkitaError::InvalidSetup("planned setup prefix is missing".into())
            })?;
            let prefix_len = slot
                .public
                .id
                .natural_len
                .checked_next_power_of_two()
                .ok_or_else(|| AkitaError::InvalidSetup("setup prefix length overflow".into()))?;
            if slot.public.id != prefix_id || prefix_len > expanded.num_field_elements {
                return Err(AkitaError::InvalidSetup(
                    "invalid Stage 3 setup prefix".into(),
                ));
            }
            let (setup_product_claim, mut session) =
                backend.begin_stage3(crate::backend::Stage3Request {
                    session,
                    level,
                    prefix: &slot.public.id,
                    parameters: lp,
                    next_parameters: next_level_params,
                    relation: instance,
                    tau1,
                    alpha,
                    stage2_challenges: sumcheck_challenges,
                    address_geometry: relation_address_geometry,
                })?;
            transcript.append_serde(
                akita_transcript::labels::ABSORB_SETUP_PREFIX_SLOT,
                &slot.public.id,
            );
            let mut kernel = Stage3Sumcheck {
                backend,
                session: &mut session,
                claim: setup_product_claim,
                rounds: prefix_len.trailing_zeros() as usize,
                next_round: 0,
                field: std::marker::PhantomData,
            };
            let mut round = 0u32;
            let (sumcheck, setup_prefix_point, _) =
                akita_sumcheck::prove_sumcheck::<F, T, E, _, _>(&mut kernel, transcript, |tr| {
                    let challenge = akita_types::sample_grinded_sumcheck_challenge::<F, E, T>(
                        tr,
                        akita_types::SumcheckProtocol::Stage3,
                        level,
                        0,
                        round,
                    )?;
                    round = round
                        .checked_add(1)
                        .ok_or_else(|| AkitaError::InvalidSetup("Stage 3 round overflow".into()))?;
                    Ok(challenge)
                })?;
            let setup_prefix_eval = backend.finish_stage3(session)?;
            Ok(Some(Stage3ProveOutput {
                proof: SetupSumcheckProof {
                    claim: setup_product_claim,
                    setup_prefix_eval,
                    sumcheck,
                },
                setup_prefix_point,
            }))
        }
        SetupContributionMode::Direct => Ok(None),
    }
}

struct Stage3Sumcheck<'a, F: Field, E: Field, B: crate::backend::OpaqueStage3Kernel<F, E>> {
    backend: &'a B,
    session: &'a mut B::Stage3SessionHandle,
    claim: E,
    rounds: usize,
    next_round: usize,
    field: std::marker::PhantomData<F>,
}

impl<F: Field, E: Field, B: crate::backend::OpaqueStage3Kernel<F, E>>
    akita_sumcheck::SumcheckKernel<E> for Stage3Sumcheck<'_, F, E, B>
{
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn degree_bound(&self) -> usize {
        akita_types::SETUP_SUMCHECK_DEGREE
    }
    fn input_claim(&self) -> E {
        self.claim
    }
    fn round_polynomial(
        &mut self,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        self.backend
            .stage3_round_polynomial(self.session, round, claim)
    }
    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        self.backend
            .bind_stage3_challenge(self.session, round, challenge)?;
        self.next_round += 1;
        Ok(())
    }
    fn finish(&mut self) -> Result<(), AkitaError> {
        if self.next_round != self.rounds {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }
}

struct Stage2Sumcheck<
    'a,
    F: Field + CanonicalEncoding,
    E: Field,
    B: crate::backend::OpaqueStage2Kernel<F, E>,
> {
    backend: &'a B,
    session: &'a mut B::Stage2SessionHandle,
    claim: E,
    rounds: usize,
    field: std::marker::PhantomData<F>,
}
impl<F: Field + CanonicalEncoding, E: Field, B: crate::backend::OpaqueStage2Kernel<F, E>>
    akita_sumcheck::SumcheckKernel<E> for Stage2Sumcheck<'_, F, E, B>
{
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn degree_bound(&self) -> usize {
        3
    }
    fn input_claim(&self) -> E {
        self.claim
    }
    fn round_polynomial(
        &mut self,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        self.backend
            .stage2_round_polynomial(self.session, round, claim)
    }
    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        self.backend
            .bind_stage2_challenge(self.session, round, challenge)
    }
    fn finish(&mut self) -> Result<(), AkitaError> {
        Ok(())
    }
}
