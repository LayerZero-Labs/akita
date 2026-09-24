use super::*;

struct Stage1EqSumcheck<
    'a,
    F: Field + CanonicalEncoding,
    E: Field,
    B: crate::backend::OpaqueStage1Kernel<F, E>,
> {
    backend: &'a B,
    session: &'a mut B::Stage1SessionHandle,
    step: crate::backend::Stage1Step,
    equality_point: &'a [E],
    claim: E,
    degree: usize,
    next_round: usize,
    field: std::marker::PhantomData<F>,
}

impl<F: Field + CanonicalEncoding, E: Field, B: crate::backend::OpaqueStage1Kernel<F, E>>
    akita_sumcheck::EqFactoredSumcheckKernel<E> for Stage1EqSumcheck<'_, F, E, B>
{
    fn num_rounds(&self) -> usize {
        self.equality_point.len()
    }
    fn degree_bound(&self) -> usize {
        self.degree
    }
    fn input_claim(&self) -> E {
        self.claim
    }
    fn current_tau(&self) -> E {
        self.equality_point[self.next_round]
    }
    fn round_polynomial(
        &mut self,
        round: usize,
        claim: E,
    ) -> Result<akita_sumcheck::EqFactoredUniPoly<E>, AkitaError> {
        if round != self.next_round {
            return Err(AkitaError::InvalidProof);
        }
        let polynomial =
            self.backend
                .stage1_round_polynomial(self.session, self.step, round, claim)?;
        let crate::backend::Stage1RoundPolynomial::EqFactored(polynomial) = polynomial else {
            return Err(AkitaError::InvalidProof);
        };
        if polynomial.degree() != self.degree {
            return Err(AkitaError::InvalidProof);
        }
        Ok(polynomial)
    }
    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        if round != self.next_round {
            return Err(AkitaError::InvalidProof);
        }
        self.backend
            .bind_stage1_challenge(self.session, self.step, round, challenge)?;
        self.next_round += 1;
        Ok(())
    }
    fn finish(&mut self) -> Result<(), AkitaError> {
        if self.next_round != self.equality_point.len() {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }
}

struct Stage1StandardSumcheck<
    'a,
    F: Field + CanonicalEncoding,
    E: Field,
    B: crate::backend::OpaqueStage1Kernel<F, E>,
> {
    backend: &'a B,
    session: &'a mut B::Stage1SessionHandle,
    step: crate::backend::Stage1Step,
    claim: E,
    rounds: usize,
    degree: usize,
    next_round: usize,
    field: std::marker::PhantomData<F>,
}

impl<F: Field + CanonicalEncoding, E: Field, B: crate::backend::OpaqueStage1Kernel<F, E>>
    akita_sumcheck::SumcheckKernel<E> for Stage1StandardSumcheck<'_, F, E, B>
{
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn degree_bound(&self) -> usize {
        self.degree
    }
    fn input_claim(&self) -> E {
        self.claim
    }
    fn round_polynomial(
        &mut self,
        round: usize,
        claim: E,
    ) -> Result<akita_algebra::uni_poly::UniPoly<E>, AkitaError> {
        if round != self.next_round {
            return Err(AkitaError::InvalidProof);
        }
        let polynomial =
            self.backend
                .stage1_round_polynomial(self.session, self.step, round, claim)?;
        let crate::backend::Stage1RoundPolynomial::Standard(polynomial) = polynomial else {
            return Err(AkitaError::InvalidProof);
        };
        if polynomial.coeffs.len() != self.degree + 1 {
            return Err(AkitaError::InvalidProof);
        }
        Ok(polynomial)
    }
    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        if round != self.next_round {
            return Err(AkitaError::InvalidProof);
        }
        self.backend
            .bind_stage1_challenge(self.session, self.step, round, challenge)?;
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

pub(super) fn prove_stage1<F, E, B>(
    ctx: &crate::backend::OperationCtx<'_, F, B>,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: u32,
    rs: &mut RingSwitchOutput<E, B::RelationHandle>,
    lp: &CommittedGroupParams,
    plan: &RelationRangeImagePlan,
) -> Result<Stage1ProveOutput<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Send + Sync + 'static,
    E: ExtField<F> + Unreduced + Fold + Ring + AkitaSerialize + Send + Sync + 'static,
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
    let mut claim = E::zero();
    for stage_index in 0..product_count {
        let shape = plan
            .digit_range_plan()
            .stage_shape(rounds, stage_index)
            .ok_or(AkitaError::InvalidProof)?;
        let step = crate::backend::Stage1Step::Product(stage_index);
        let stage = u32::try_from(stage_index).map_err(|_| AkitaError::InvalidProof)?;
        let mut kernel = Stage1EqSumcheck {
            backend: ctx.backend(),
            session: &mut session_handle,
            step,
            equality_point: &equality_coordinates,
            claim,
            degree: shape.sumcheck_proof.1,
            next_round: 0,
            field: std::marker::PhantomData,
        };
        let mut channel = akita_types::NativeGrindingSumcheckProver::<F, E>::new(
            grinding,
            akita_types::SumcheckProtocol::Stage1,
            level,
            stage,
        );
        let (challenges, _output_claim) =
            akita_sumcheck::prove_eq_factored_sumcheck_native::<F, E, _, _>(
                &mut kernel,
                &mut channel,
                akita_sumcheck::NativeSumcheckShape::new(rounds, shape.sumcheck_proof.1)?,
                0,
            )?;
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
        akita_types::native_stage1_prover_child_claims::<F, E>(
            grinding,
            level,
            stage,
            &child_claims,
        )?;
        let gamma = grinding.grinded_ext_challenge::<F, E>(
            akita_types::GrindingSite::Stage1InterstageBatch { level, stage },
        )?;
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
    }

    let final_point;
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
        akita_types::native_l2_prover_prefix::<F, E>(grinding, level, response_l2_sq, &subclaims)?;
        let norm_claim = if subclaims.is_empty() {
            E::from_u128(response_l2_sq)
        } else {
            let gamma = grinding.grinded_ext_challenge::<F, E>(
                akita_types::GrindingSite::L2SubclaimBatch { level },
            )?;
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
        let merge = grinding
            .grinded_ext_challenge::<F, E>(akita_types::GrindingSite::L2NormMerge { level })?;
        crate::backend::OpaqueStage1Kernel::bind_stage1_batch_challenge(
            ctx.backend(),
            &mut session_handle,
            crate::backend::Stage1Transition::RangeNormMerge,
            merge,
        )?;
        claim += merge * norm_claim;
        let mut kernel = Stage1StandardSumcheck {
            backend: ctx.backend(),
            session: &mut session_handle,
            step,
            claim,
            rounds,
            degree: plan.digit_range_plan().leaf_degree() + 1,
            next_round: 0,
            field: std::marker::PhantomData,
        };
        let mut channel = akita_types::NativeGrindingSumcheckProver::<F, E>::new(
            grinding,
            akita_types::SumcheckProtocol::PhysicalL2,
            level,
            0,
        );
        let (point, output_claim) = akita_sumcheck::prove_sumcheck_native::<F, E, _, _>(
            &mut kernel,
            &mut channel,
            akita_sumcheck::NativeSumcheckShape::new(
                rounds,
                plan.digit_range_plan().leaf_degree() + 1,
            )?,
            0,
        )?;
        claim = output_claim;
        final_point = point;
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
        akita_types::native_l2_prover_virtual_evaluations::<F, E>(
            grinding,
            level,
            &virtual_evaluations,
        )?;
        (
            range_image_evaluation,
            Some((response_l2_sq, virtual_evaluations)),
        )
    } else {
        let step = crate::backend::Stage1Step::RangeLeaf;
        let stage = u32::try_from(product_count).map_err(|_| AkitaError::InvalidProof)?;
        let mut kernel = Stage1EqSumcheck {
            backend: ctx.backend(),
            session: &mut session_handle,
            step,
            equality_point: &equality_coordinates,
            claim,
            degree: plan.digit_range_plan().leaf_degree(),
            next_round: 0,
            field: std::marker::PhantomData,
        };
        let mut channel = akita_types::NativeGrindingSumcheckProver::<F, E>::new(
            grinding,
            akita_types::SumcheckProtocol::Stage1,
            level,
            stage,
        );
        let (point, output_claim) = akita_sumcheck::prove_eq_factored_sumcheck_native::<F, E, _, _>(
            &mut kernel,
            &mut channel,
            akita_sumcheck::NativeSumcheckShape::new(
                rounds,
                plan.digit_range_plan().leaf_degree(),
            )?,
            0,
        )?;
        claim = output_claim;
        final_point = point;
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
        (range_image_evaluation, None)
    };
    let final_claims =
        crate::backend::OpaqueStage1Kernel::finish_stage1(ctx.backend(), session_handle)?;
    if final_claims.final_claim() != claim || final_claims.point() != final_point {
        return Err(AkitaError::InvalidProof);
    }
    let stage1_point = final_claims.point().to_vec();
    akita_types::native_stage1_prover_range_image::<F, E>(
        grinding,
        level,
        u32::try_from(product_count).map_err(|_| AkitaError::InvalidProof)?,
        range_image_evaluation,
    )?;
    let physical_l2 = match physical_plan {
        Some(physical_plan) => {
            let (response_l2_sq, virtual_evaluations) =
                norm_proof.as_ref().ok_or(AkitaError::InvalidProof)?;
            let InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap, ..
            } = lp.inner().matrix.security_route()
            else {
                return Err(AkitaError::InvalidSetup(
                    "physical L2 plan disagrees with the A security route".into(),
                ));
            };
            if *response_l2_sq > response_l2_sq_cap {
                return Err(AkitaError::InvalidInput(
                    "folded response exceeds the scheduled L2 cap".into(),
                ));
            }
            Some(PhysicalL2ProverReplay {
                plan: physical_plan,
                point: stage1_point.clone(),
                virtual_evaluations: virtual_evaluations.clone(),
                batching: Vec::new(),
                claim: E::zero(),
            })
        }
        None => {
            if norm_proof.is_some() {
                return Err(AkitaError::InvalidInput(
                    "L-infinity route produced an L2 norm proof".into(),
                ));
            }
            None
        }
    };
    Ok(Stage1ProveOutput {
        point: stage1_point,
        range_image_evaluation,
        physical_l2,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_stage2<F, E, B>(
    ctx: &crate::backend::OperationCtx<'_, F, B>,
    level: usize,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
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
    let mut channel = akita_types::NativeGrindingSumcheckProver::<F, E>::new(
        grinding,
        akita_types::SumcheckProtocol::Stage2,
        level,
        0,
    );
    let (sumcheck_challenges, claim) = akita_sumcheck::prove_sumcheck_native::<F, E, _, _>(
        &mut kernel,
        &mut channel,
        akita_sumcheck::NativeSumcheckShape::new(num_rounds, 3)?,
        0,
    )?;
    let final_output =
        crate::backend::OpaqueStage2Kernel::finish_stage2(ctx.backend(), session_handle)?;
    if claim != final_output.final_claim() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(Stage2ProveOutput {
        challenges: sumcheck_challenges,
        witness_evaluation: final_output.witness_evaluation(),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_stage3<F, E, B>(
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
    grinding: &mut akita_types::NativeProverGrinding<'_>,
) -> Result<Option<Stage3ProveOutput<E>>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F>
        + Ring
        + ExtField<F>
        + AkitaSerialize
        + jolt_field::Unreduced
        + jolt_field::MulBaseUnreduced<F>,
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
            let mut encoded_slot = Vec::new();
            slot.public
                .id
                .serialize_compressed(&mut encoded_slot)
                .map_err(|_| AkitaError::InvalidProof)?;
            akita_types::native_stage3_public_slot_prover(grinding, level, &encoded_slot)?;
            akita_types::native_stage3_prover_claim::<F, E>(grinding, level, setup_product_claim)?;
            let mut kernel = Stage3Sumcheck {
                backend,
                session: &mut session,
                claim: setup_product_claim,
                rounds: prefix_len.trailing_zeros() as usize,
                next_round: 0,
                field: std::marker::PhantomData,
            };
            let mut channel = akita_types::NativeGrindingSumcheckProver::<F, E>::new(
                grinding,
                akita_types::SumcheckProtocol::Stage3,
                level,
                0,
            );
            let (setup_prefix_point, _) = akita_sumcheck::prove_sumcheck_native::<F, E, _, _>(
                &mut kernel,
                &mut channel,
                akita_sumcheck::NativeSumcheckShape::new(
                    prefix_len.trailing_zeros() as usize,
                    akita_types::SETUP_SUMCHECK_DEGREE,
                )?,
                0,
            )?;
            let setup_prefix_eval = backend.finish_stage3(session)?;
            akita_types::native_stage3_prover_prefix_eval::<F, E>(
                grinding,
                level,
                setup_prefix_eval,
            )?;
            Ok(Some(Stage3ProveOutput {
                setup_prefix_eval,
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
    ) -> Result<akita_algebra::uni_poly::UniPoly<E>, AkitaError> {
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
    ) -> Result<akita_algebra::uni_poly::UniPoly<E>, AkitaError> {
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
