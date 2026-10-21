use super::*;
use crate::backend::DigitRangeSteps;
use akita_params::DigitRangePlan;
use akita_types::GrindingReplay;
use jolt_poly::{Point, UnivariatePoly};
use jolt_sumcheck::{prove_single, SingleFormat, SingleInstance, SingleReduction};

/// The fold's Stage 1 session, stepped through its backend.
struct BackendStage1<
    'a,
    F: Field + CanonicalEncoding,
    E: Field,
    B: crate::backend::OpaqueStage1Kernel<F, E>,
> {
    backend: &'a B,
    session: &'a mut B::Stage1SessionHandle,
    field: std::marker::PhantomData<(F, E)>,
}

impl<F: Field + CanonicalEncoding, E: Field, B: crate::backend::OpaqueStage1Kernel<F, E>>
    DigitRangeSteps<E> for BackendStage1<'_, F, E, B>
{
    fn round_polynomial(
        &mut self,
        step: crate::backend::Stage1Step,
        round: usize,
        previous_claim: E,
    ) -> Result<crate::backend::Stage1RoundPolynomial<E>, AkitaError> {
        self.backend
            .stage1_round_polynomial(self.session, step, round, previous_claim)
    }

    fn bind_challenge(
        &mut self,
        step: crate::backend::Stage1Step,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.backend
            .bind_stage1_challenge(self.session, step, round, challenge)
    }

    fn public_transition(
        &mut self,
        step: crate::backend::Stage1Step,
    ) -> Result<crate::backend::Stage1PublicTransition<E>, AkitaError> {
        self.backend.stage1_public_transition(self.session, step)
    }

    fn bind_batch_challenge(
        &mut self,
        transition: crate::backend::Stage1Transition,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.backend
            .bind_stage1_batch_challenge(self.session, transition, challenge)
    }
}

struct Stage1EqSumcheck<'a, S> {
    session: &'a mut S,
    step: crate::backend::Stage1Step,
    rounds: usize,
    degree: usize,
    next_round: usize,
}

impl<E: Field, S: DigitRangeSteps<E>> akita_sumcheck::EqFactoredSumcheckKernel<E>
    for Stage1EqSumcheck<'_, S>
{
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn round_polynomial(
        &mut self,
        round: usize,
        claim: E,
    ) -> Result<jolt_poly::OmittedConstantPoly<E>, AkitaError> {
        if round != self.next_round {
            return Err(AkitaError::Internal(
                "stage 1 equality-factored round index differs from adapter state".into(),
            ));
        }
        let polynomial = self.session.round_polynomial(self.step, round, claim)?;
        let crate::backend::Stage1RoundPolynomial::EqFactored(polynomial) = polynomial else {
            return Err(AkitaError::Internal(
                "stage 1 equality-factored backend returned a standard polynomial".into(),
            ));
        };
        if polynomial.degree() != self.degree {
            return Err(AkitaError::Internal(
                "stage 1 equality-factored polynomial degree differs from plan".into(),
            ));
        }
        Ok(polynomial)
    }
    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        if round != self.next_round {
            return Err(AkitaError::Internal(
                "stage 1 equality-factored bind round differs from adapter state".into(),
            ));
        }
        self.session.bind_challenge(self.step, round, challenge)?;
        self.next_round += 1;
        Ok(())
    }
    fn finish(&mut self) -> Result<(), AkitaError> {
        if self.next_round != self.rounds {
            return Err(AkitaError::Internal(
                "stage 1 equality-factored sumcheck finished before all rounds were bound".into(),
            ));
        }
        Ok(())
    }
}

struct Stage1StandardSumcheck<'a, S> {
    session: &'a mut S,
    step: crate::backend::Stage1Step,
    rounds: usize,
    degree: usize,
    next_round: usize,
}

impl<E: Field, S: DigitRangeSteps<E>> akita_sumcheck::SumcheckKernel<E>
    for Stage1StandardSumcheck<'_, S>
{
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn round_polynomial(
        &mut self,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        if round != self.next_round {
            return Err(AkitaError::Internal(
                "stage 1 standard round index differs from adapter state".into(),
            ));
        }
        let polynomial = self.session.round_polynomial(self.step, round, claim)?;
        let crate::backend::Stage1RoundPolynomial::Standard(polynomial) = polynomial else {
            return Err(AkitaError::Internal(
                "stage 1 standard backend returned an equality-factored polynomial".into(),
            ));
        };
        if polynomial.coefficients().len() != self.degree + 1 {
            return Err(AkitaError::Internal(
                "stage 1 standard polynomial coefficient count differs from plan".into(),
            ));
        }
        Ok(polynomial)
    }
    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        if round != self.next_round {
            return Err(AkitaError::Internal(
                "stage 1 standard bind round differs from adapter state".into(),
            ));
        }
        self.session.bind_challenge(self.step, round, challenge)?;
        self.next_round += 1;
        Ok(())
    }
    fn finish(&mut self) -> Result<(), AkitaError> {
        if self.next_round != self.rounds {
            return Err(AkitaError::Internal(
                "stage 1 standard sumcheck finished before all rounds were bound".into(),
            ));
        }
        Ok(())
    }
}

pub(super) fn prove_stage1<F, E, B, H: Sponge>(
    ctx: &crate::backend::OperationCtx<'_, F, B>,
    grinding: &mut akita_types::ProverGrinding<'_, H>,
    level: u32,
    rs: &mut RingSwitchOutput<E, B::RelationHandle>,
    lp: &CommittedGroupParams,
    plan: &RelationRangeImagePlan,
) -> Result<Stage1ProveOutput<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Send + Sync + 'static,
    E: ExtField<F>
        + Unreduced
        + Fold
        + Ring
        + AkitaSerialize
        + Send
        + Sync
        + 'static
        + CanonicalDecode,
    B: crate::backend::OpaqueStage1Kernel<F, E>,
{
    let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
    let domain = plan.digit_witness_domain();
    if plan.relation_address_geometry() != rs.relation_address_geometry
        || domain.live_len() != rs.witness_len
        || plan.digit_range_plan().basis() != rs.b
    {
        return Err(AkitaError::Internal(
            "ring-switch output disagrees with the relation/range-image plan".into(),
        ));
    }
    let digit_range_equality_col_bits = rs
        .tau0
        .len()
        .checked_sub(rs.digit_range_equality_low_variable_count)
        .ok_or_else(|| AkitaError::Internal("digit-range equality width overflow".into()))?;
    let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
        &rs.tau0,
        digit_range_equality_col_bits,
        rs.digit_range_equality_low_variable_count,
    )?;
    let physical_plan = PhysicalResponsePlan::new(lp, plan)?;
    let equality_coordinates = equality_point.clone().into_coordinates();
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
    let mut steps = BackendStage1 {
        backend: ctx.backend(),
        session: &mut session_handle,
        field: std::marker::PhantomData,
    };
    let (mut claim, equality_coordinates) = prove_stage1_products::<F, E, _, _>(
        &mut steps,
        plan.digit_range_plan(),
        equality_coordinates,
        grinding,
        level,
    )?;

    let final_point;
    let (range_image_evaluation, norm_proof) = if physical_plan.is_some() {
        let step = crate::backend::Stage1Step::FusedRangeNorm;
        let crate::backend::Stage1PublicTransition::PhysicalL2Claims {
            response_l2_sq,
            mut subclaims,
        } = steps.public_transition(step)?
        else {
            return Err(AkitaError::Internal(
                "stage 1 fused range-norm transition did not return physical L2 claims".into(),
            ));
        };
        let physical = physical_plan.as_ref().ok_or_else(|| {
            AkitaError::Internal("stage 1 fused range-norm execution has no physical plan".into())
        })?;
        if subclaims.len()
            != physical.shape().subclaim_count().ok_or_else(|| {
                AkitaError::Internal("stage 1 physical L2 subclaim count overflow".into())
            })?
        {
            return Err(AkitaError::Internal(
                "stage 1 physical L2 subclaim count differs from plan".into(),
            ));
        }
        akita_types::l2_prefix::<F, E, _>(grinding, level, response_l2_sq, &mut subclaims)?;
        let norm_claim = if subclaims.is_empty() {
            E::from_u128(response_l2_sq)
        } else {
            let gamma = grinding.grinded_ext_challenge::<F, E>(
                akita_params::GrindingSite::L2SubclaimBatch { level },
            )?;
            steps.bind_batch_challenge(crate::backend::Stage1Transition::L2SubclaimBatch, gamma)?;
            let mut power = E::one();
            subclaims.iter().fold(E::zero(), |sum, &subclaim| {
                let next = sum + power * subclaim;
                power *= gamma;
                next
            })
        };
        let merge = grinding
            .grinded_ext_challenge::<F, E>(akita_params::GrindingSite::L2NormMerge { level })?;
        steps.bind_batch_challenge(crate::backend::Stage1Transition::RangeNormMerge, merge)?;
        claim += merge * norm_claim;
        let degree = plan.digit_range_plan().leaf_degree() + 1;
        let mut kernel = Stage1StandardSumcheck {
            session: &mut steps,
            step,
            rounds,
            degree,
            next_round: 0,
        };
        let instance = SingleInstance {
            input_claim: claim,
            rounds,
            degree,
            format: SingleFormat::Compressed,
            invocation: 0,
        };
        let mut channel = akita_types::GrindingSumcheckProver::<F, E, _>::new(
            grinding,
            akita_params::SumcheckProtocol::PhysicalL2,
            level,
            0,
        );
        let reduction = prove_single::<F, E, AkitaError, _>(
            &mut akita_sumcheck::FusedRounds(&mut kernel),
            &instance,
            &mut channel,
        )?;
        claim = reduction.output_claim;
        final_point = reduction.challenges;
        let crate::backend::Stage1PublicTransition::Final {
            range_image_evaluation,
            mut virtual_evaluations,
        } = steps.public_transition(step)?
        else {
            return Err(AkitaError::Internal(
                "stage 1 fused range-norm transition did not return final evaluations".into(),
            ));
        };
        if virtual_evaluations.len() != physical.shape().virtual_evaluation_count() {
            return Err(AkitaError::Internal(
                "stage 1 physical L2 virtual evaluation count differs from plan".into(),
            ));
        }
        akita_types::l2_virtual_evaluations::<F, E, _>(grinding, level, &mut virtual_evaluations)?;
        (
            range_image_evaluation,
            Some((response_l2_sq, virtual_evaluations)),
        )
    } else {
        let leaf = prove_stage1_leaf::<F, E, _, _>(
            &mut steps,
            plan.digit_range_plan(),
            claim,
            equality_coordinates,
            grinding,
            level,
        )?;
        claim = leaf.claim;
        final_point = leaf.point;
        (leaf.range_image, None)
    };
    let final_claims =
        crate::backend::OpaqueStage1Kernel::finish_stage1(ctx.backend(), session_handle)?;
    if final_claims.final_claim() != claim || final_claims.point() != final_point {
        return Err(AkitaError::Internal(
            "stage 1 backend final claim or point differs from sumcheck output".into(),
        ));
    }
    let stage1_point = final_claims.point().to_vec();
    akita_types::stage1_range_image::<F, E, _>(
        grinding,
        level,
        u32::try_from(product_count).map_err(|_| {
            AkitaError::Internal("stage 1 range-image stage index exceeds u32".into())
        })?,
        range_image_evaluation,
    )?;
    let physical_l2 = match physical_plan {
        Some(physical_plan) => {
            let (response_l2_sq, virtual_evaluations) = norm_proof.as_ref().ok_or_else(|| {
                AkitaError::Internal("stage 1 physical L2 replay has no norm proof".into())
            })?;
            let InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap, ..
            } = lp.inner().matrix.security_route()
            else {
                return Err(AkitaError::Internal(
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
                return Err(AkitaError::Internal(
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

/// Stage 1's product stages: each an eq-factored sumcheck, its child
/// claims, and their batch. Returns the leaf's input claim and equality
/// point.
fn prove_stage1_products<F, E, S, H: Sponge>(
    steps: &mut S,
    plan: DigitRangePlan,
    mut equality_coordinates: Vec<E>,
    grinding: &mut akita_types::ProverGrinding<'_, H>,
    level: u32,
) -> Result<(E, Vec<E>), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    S: DigitRangeSteps<E>,
{
    let rounds = equality_coordinates.len();
    let mut claim = E::zero();
    for stage_index in 0..plan.product_stage_arities().len() {
        let shape = plan
            .stage_shape(rounds, stage_index)
            .ok_or_else(|| AkitaError::Internal("stage 1 product stage shape is missing".into()))?;
        let step = crate::backend::Stage1Step::Product(stage_index);
        let stage = u32::try_from(stage_index)
            .map_err(|_| AkitaError::Internal("stage 1 product stage index exceeds u32".into()))?;
        let mut kernel = Stage1EqSumcheck {
            session: &mut *steps,
            step,
            rounds,
            degree: shape.sumcheck_proof.1,
            next_round: 0,
        };
        let instance = SingleInstance {
            input_claim: claim,
            rounds,
            degree: shape.sumcheck_proof.1,
            format: SingleFormat::EqFactored {
                equality_point: Point::low_to_high(equality_coordinates.clone()),
            },
            invocation: 0,
        };
        let mut channel = akita_types::GrindingSumcheckProver::<F, E, _>::new(
            grinding,
            akita_params::SumcheckProtocol::Stage1,
            level,
            stage,
        );
        let challenges = prove_single::<F, E, AkitaError, _>(
            &mut akita_sumcheck::FusedEqFactoredRounds::new(&mut kernel, &instance)?,
            &instance,
            &mut channel,
        )?
        .challenges;
        let crate::backend::Stage1PublicTransition::ProductChildClaims(mut child_claims) =
            steps.public_transition(step)?
        else {
            return Err(AkitaError::Internal(
                "stage 1 product transition did not return child claims".into(),
            ));
        };
        if child_claims.len() != shape.child_claims {
            return Err(AkitaError::Internal(
                "stage 1 product child claim count differs from stage shape".into(),
            ));
        }
        akita_types::stage1_child_claims::<F, E, _>(grinding, level, stage, &mut child_claims)?;
        let gamma = grinding.grinded_ext_challenge::<F, E>(
            akita_params::GrindingSite::Stage1InterstageBatch { level, stage },
        )?;
        let weights = plan.interstage_batch_weights(gamma, child_claims.len());
        claim = plan.batch_claims(&weights, &child_claims)?;
        steps.bind_batch_challenge(
            crate::backend::Stage1Transition::ProductBatch(stage_index),
            gamma,
        )?;
        equality_coordinates = challenges;
    }
    Ok((claim, equality_coordinates))
}

/// The L∞ range leaf's output claim, point and range image.
struct Stage1Leaf<E> {
    claim: E,
    point: Vec<E>,
    range_image: E,
}

/// Stage 1's L∞ range leaf: an eq-factored sumcheck from `claim` at
/// `equality_coordinates`, then the final transition.
fn prove_stage1_leaf<F, E, S, H: Sponge>(
    steps: &mut S,
    plan: DigitRangePlan,
    claim: E,
    equality_coordinates: Vec<E>,
    grinding: &mut akita_types::ProverGrinding<'_, H>,
    level: u32,
) -> Result<Stage1Leaf<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    S: DigitRangeSteps<E>,
{
    let step = crate::backend::Stage1Step::RangeLeaf;
    let stage = u32::try_from(plan.product_stage_arities().len())
        .map_err(|_| AkitaError::Internal("stage 1 leaf stage index exceeds u32".into()))?;
    let rounds = equality_coordinates.len();
    let degree = plan.leaf_degree();
    let mut kernel = Stage1EqSumcheck {
        session: &mut *steps,
        step,
        rounds,
        degree,
        next_round: 0,
    };
    let instance = SingleInstance {
        input_claim: claim,
        rounds,
        degree,
        format: SingleFormat::EqFactored {
            equality_point: Point::low_to_high(equality_coordinates),
        },
        invocation: 0,
    };
    let mut channel = akita_types::GrindingSumcheckProver::<F, E, _>::new(
        grinding,
        akita_params::SumcheckProtocol::Stage1,
        level,
        stage,
    );
    let reduction = prove_single::<F, E, AkitaError, _>(
        &mut akita_sumcheck::FusedEqFactoredRounds::new(&mut kernel, &instance)?,
        &instance,
        &mut channel,
    )?;
    let crate::backend::Stage1PublicTransition::Final {
        range_image_evaluation,
        virtual_evaluations,
    } = steps.public_transition(step)?
    else {
        return Err(AkitaError::Internal(
            "stage 1 range-leaf transition did not return final evaluations".into(),
        ));
    };
    if !virtual_evaluations.is_empty() {
        return Err(AkitaError::Internal(
            "stage 1 range-leaf transition returned physical L2 virtual evaluations".into(),
        ));
    }
    Ok(Stage1Leaf {
        claim: reduction.output_claim,
        point: reduction.challenges,
        range_image: range_image_evaluation,
    })
}

/// The output of [`prove_digit_range`]: the reduction's point and the range
/// image `Σ_j eq(point, j)·w_j(w_j + 1)` there.
pub struct DigitRangeOutput<E> {
    pub point: Vec<E>,
    pub range_image: E,
}

/// Akita's Stage 1 digit-range reduction outside a fold, on the L∞ route:
/// the product stages and the range leaf from `equality_point` (low to
/// high), then the range image, exactly as a fold's Stage 1 sends them.
/// `grinding`'s plan must schedule this reduction's sites at `level`.
pub fn prove_digit_range<F, E, S, H: Sponge>(
    steps: &mut S,
    plan: DigitRangePlan,
    equality_point: Vec<E>,
    grinding: &mut akita_types::ProverGrinding<'_, H>,
    level: u32,
) -> Result<DigitRangeOutput<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    S: DigitRangeSteps<E>,
{
    let (claim, equality_coordinates) =
        prove_stage1_products::<F, E, _, _>(steps, plan, equality_point, grinding, level)?;
    let leaf =
        prove_stage1_leaf::<F, E, _, _>(steps, plan, claim, equality_coordinates, grinding, level)?;
    let stage = u32::try_from(plan.product_stage_arities().len())
        .map_err(|_| AkitaError::Internal("stage 1 range-image stage index exceeds u32".into()))?;
    akita_types::stage1_range_image::<F, E, _>(grinding, level, stage, leaf.range_image)?;
    Ok(DigitRangeOutput {
        point: leaf.point,
        range_image: leaf.range_image,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_stage2<F, E, B, H: Sponge>(
    ctx: &crate::backend::OperationCtx<'_, F, B>,
    level: usize,
    grinding: &mut akita_types::ProverGrinding<'_, H>,
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
    E: ExtField<F> + Unreduced + Fold + Ring + AkitaSerialize + CanonicalDecode,
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
        return Err(AkitaError::Internal(
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
    )?;
    let level =
        u32::try_from(level).map_err(|_| AkitaError::Internal("fold level exceeds u32".into()))?;
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
        return Err(AkitaError::Internal(
            "stage 2 backend round count or input claim differs from relation plan".into(),
        ));
    }
    let mut kernel = Stage2Sumcheck::<F, E, B> {
        backend: ctx.backend(),
        session: &mut session_handle,
        rounds: num_rounds,
        field: std::marker::PhantomData,
    };
    let instance = SingleInstance {
        input_claim: claim,
        rounds: num_rounds,
        degree: akita_params::STAGE2_SUMCHECK_DEGREE,
        format: SingleFormat::Compressed,
        invocation: 0,
    };
    let mut channel = akita_types::GrindingSumcheckProver::<F, E, _>::new(
        grinding,
        akita_params::SumcheckProtocol::Stage2,
        level,
        0,
    );
    let SingleReduction {
        challenges: sumcheck_challenges,
        output_claim: claim,
    } = prove_single::<F, E, AkitaError, _>(
        &mut akita_sumcheck::FusedRounds(&mut kernel),
        &instance,
        &mut channel,
    )?;
    let final_output =
        crate::backend::OpaqueStage2Kernel::finish_stage2(ctx.backend(), session_handle)?;
    if claim != final_output.final_claim() {
        return Err(AkitaError::Internal(
            "stage 2 backend final claim differs from sumcheck output".into(),
        ));
    }
    Ok(Stage2ProveOutput {
        challenges: sumcheck_challenges,
        witness_evaluation: final_output.witness_evaluation(),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_stage3<F, E, B, H: Sponge>(
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
    relation_address_geometry: akita_params::RelationAddressGeometry,
    grinding: &mut akita_types::ProverGrinding<'_, H>,
) -> Result<Option<Stage3ProveOutput<E>>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F>
        + Ring
        + ExtField<F>
        + AkitaSerialize
        + jolt_algebra::Unreduced
        + jolt_algebra::MulBaseUnreduced<F>,
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
                .map_err(|_| AkitaError::Internal("fold level exceeds u32".into()))?;
            let prefix_id = next_level_params
                .setup_prefix()
                .and_then(|group| group.slot_id())
                .ok_or_else(|| AkitaError::Internal("Stage 3 requires a setup prefix".into()))?;
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
                .map_err(|error| {
                    AkitaError::Internal(format!("setup-prefix slot serialization failed: {error}"))
                })?;
            akita_types::stage3_public_slot(grinding, level, &encoded_slot);
            akita_types::stage3_claim::<F, E, _>(grinding, level, setup_product_claim)?;
            let rounds = prefix_len.trailing_zeros() as usize;
            let mut kernel = Stage3Sumcheck {
                backend,
                session: &mut session,
                rounds,
                next_round: 0,
                field: std::marker::PhantomData,
            };
            let instance = SingleInstance {
                input_claim: setup_product_claim,
                rounds,
                degree: akita_params::SETUP_SUMCHECK_DEGREE,
                format: SingleFormat::Compressed,
                invocation: 0,
            };
            let mut channel = akita_types::GrindingSumcheckProver::<F, E, _>::new(
                grinding,
                akita_params::SumcheckProtocol::Stage3,
                level,
                0,
            );
            let setup_prefix_point = prove_single::<F, E, AkitaError, _>(
                &mut akita_sumcheck::FusedRounds(&mut kernel),
                &instance,
                &mut channel,
            )?
            .challenges;
            let setup_prefix_eval = backend.finish_stage3(session)?;
            akita_types::stage3_prefix_eval::<F, E, _>(grinding, level, setup_prefix_eval)?;
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
            return Err(AkitaError::Internal(
                "stage 3 sumcheck finished before all rounds were bound".into(),
            ));
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
    rounds: usize,
    field: std::marker::PhantomData<F>,
}
impl<F: Field + CanonicalEncoding, E: Field, B: crate::backend::OpaqueStage2Kernel<F, E>>
    akita_sumcheck::SumcheckKernel<E> for Stage2Sumcheck<'_, F, E, B>
{
    fn num_rounds(&self) -> usize {
        self.rounds
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
