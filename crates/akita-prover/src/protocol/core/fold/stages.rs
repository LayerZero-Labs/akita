use super::*;

pub(super) enum Stage2Compression<E: Field> {
    Raw,
    QuotientLift {
        weights: akita_types::CompressionRelationWeights<E>,
        support: akita_types::NegativeBinarySupport,
        binary_batching: E,
    },
    ReducedEvaluation {
        support: akita_types::NegativeBinarySupport,
        binary_batching: E,
    },
}

trait Stage2ProverStream<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    type Proof;

    fn prove<P>(&mut self, prover: &mut P) -> Result<(Self::Proof, Vec<E>, E), AkitaError>
    where
        P: akita_sumcheck::SumcheckInstanceProver<E> + ?Sized;
}

struct LegacyStage2ProverStream<'a, T> {
    transcript: &'a mut T,
    level: u32,
}

impl<F, E, T> Stage2ProverStream<F, E> for LegacyStage2ProverStream<'_, T>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: ExtField<F> + AkitaSerialize,
    T: akita_types::ProverTranscriptGrinding<F>,
{
    type Proof = SumcheckProof<E>;

    fn prove<P>(&mut self, prover: &mut P) -> Result<(Self::Proof, Vec<E>, E), AkitaError>
    where
        P: akita_sumcheck::SumcheckInstanceProver<E> + ?Sized,
    {
        let mut round = 0u32;
        prove_sumcheck::<F, T, E, _, _>(prover, self.transcript, |tr| {
            let challenge = akita_types::sample_grinded_sumcheck_challenge::<F, E, T>(
                tr,
                akita_types::SumcheckProtocol::Stage2,
                self.level,
                0,
                round,
            )?;
            round = round
                .checked_add(1)
                .ok_or_else(|| AkitaError::InvalidSetup("Stage 2 round overflow".into()))?;
            Ok(challenge)
        })
    }
}

#[allow(dead_code)] // Constructed by the native fold driver during cutover.
struct NativeStage2ProverStream<'a, 'plan> {
    grinding: &'a mut akita_types::NativeProverGrinding<'plan>,
    level: u32,
}

impl<F, E> Stage2ProverStream<F, E> for NativeStage2ProverStream<'_, '_>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    type Proof = ();

    fn prove<P>(&mut self, prover: &mut P) -> Result<(Self::Proof, Vec<E>, E), AkitaError>
    where
        P: akita_sumcheck::SumcheckInstanceProver<E> + ?Sized,
    {
        let mut channel = akita_types::NativeGrindingSumcheckProver::<F, E>::new(
            self.grinding,
            akita_types::SumcheckProtocol::Stage2,
            self.level,
            0,
        );
        let (point, final_claim) =
            akita_sumcheck::prove_sumcheck_native::<F, E, _, _>(prover, &mut channel, 0)?;
        Ok(((), point, final_claim))
    }
}

pub(super) fn prove_stage1<F, E, T>(
    transcript: &mut T,
    level: u32,
    rs: &mut RingSwitchOutput<E>,
    lp: &CommittedGroupParams,
    plan: &RelationRangeImagePlan,
) -> Result<Stage1ProveOutput<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: ExtField<F> + Unreduced + Fold + Ring + AkitaSerialize,
    T: akita_types::ProverTranscriptGrinding<F>,
{
    let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
    let domain = plan.digit_witness_domain();
    if plan.relation_address_geometry() != rs.relation_address_geometry
        || domain.live_len() != rs.w_evals_compact.len()
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
    let stage1_prover = DigitRangeProver::from_packed_digits(
        rs.w_evals_compact.clone(),
        plan.digit_range_plan(),
        domain,
        equality_point,
    )?;
    let physical_plan = PhysicalResponsePlan::new(lp, plan)?;
    let (stage1_proof, stage1_point) =
        stage1_prover.prove::<F, T>(transcript, physical_plan.as_ref(), level)?;
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

#[allow(dead_code)] // Called by the native fold driver during cutover.
pub(super) fn prove_stage1_native<F, E>(
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: u32,
    rs: &mut RingSwitchOutput<E>,
    lp: &CommittedGroupParams,
    plan: &RelationRangeImagePlan,
) -> Result<NativeStage1ProveOutput<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: ExtField<F> + Unreduced + Fold + Ring + AkitaSerialize,
{
    let domain = plan.digit_witness_domain();
    if plan.relation_address_geometry() != rs.relation_address_geometry
        || domain.live_len() != rs.w_evals_compact.len()
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
    let stage1_prover = DigitRangeProver::from_packed_digits(
        rs.w_evals_compact.clone(),
        plan.digit_range_plan(),
        domain,
        equality_point,
    )?;
    let physical_plan = PhysicalResponsePlan::new(lp, plan)?;
    let output = stage1_prover.prove_native::<F>(grinding, physical_plan.as_ref(), level)?;
    let physical_l2 = match (physical_plan, output.physical_l2) {
        (Some(plan), Some(proof)) => {
            let InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap, ..
            } = lp.inner().matrix.security_route()
            else {
                return Err(AkitaError::InvalidSetup(
                    "physical L2 plan disagrees with the A security route".into(),
                ));
            };
            if proof.response_l2_sq > response_l2_sq_cap {
                return Err(AkitaError::InvalidInput(
                    "folded response exceeds the scheduled L2 cap".into(),
                ));
            }
            Some(PhysicalL2ProverReplay {
                plan,
                point: output.point.clone(),
                virtual_evaluations: proof.virtual_evaluations,
                batching: Vec::new(),
                claim: E::zero(),
            })
        }
        (None, None) => None,
        _ => return Err(AkitaError::InvalidProof),
    };
    Ok(NativeStage1ProveOutput {
        point: output.point,
        range_image_evaluation: output.range_image_evaluation,
        physical_l2,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_stage2<F, E, T>(
    level: usize,
    transcript: &mut T,
    batching_coeff: E,
    rs: RingSwitchOutput<E>,
    stage1_point: &[E],
    range_image_evaluation: E,
    relation_claim: E,
    compression: Stage2Compression<E>,
    physical_l2: Option<PhysicalL2ProverReplay<E>>,
    linear_terms: PreparedProverLinearTerms<E>,
    trace_opening_claim: E,
    plan: RelationRangeImagePlan,
) -> Result<Stage2ProveOutput<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: ExtField<F> + Unreduced + Fold + Ring + AkitaSerialize,
    T: akita_types::ProverTranscriptGrinding<F>,
{
    let level_u32 = u32::try_from(level)
        .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
    let mut stream = LegacyStage2ProverStream {
        transcript,
        level: level_u32,
    };
    prove_stage2_with_stream::<F, E, _>(
        level,
        &mut stream,
        batching_coeff,
        rs,
        stage1_point,
        range_image_evaluation,
        relation_claim,
        compression,
        physical_l2,
        linear_terms,
        trace_opening_claim,
        plan,
    )
}

#[allow(dead_code)] // Called by the native fold driver during production cutover.
#[allow(clippy::too_many_arguments)]
pub(super) fn prove_stage2_native<F, E>(
    level: usize,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    batching_coeff: E,
    rs: RingSwitchOutput<E>,
    stage1_point: &[E],
    range_image_evaluation: E,
    relation_claim: E,
    compression: Stage2Compression<E>,
    physical_l2: Option<PhysicalL2ProverReplay<E>>,
    linear_terms: PreparedProverLinearTerms<E>,
    trace_opening_claim: E,
    plan: RelationRangeImagePlan,
) -> Result<Stage2ProveOutput<E, ()>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: ExtField<F> + Unreduced + Fold + Ring + AkitaSerialize,
{
    let level_u32 = u32::try_from(level)
        .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
    let mut stream = NativeStage2ProverStream {
        grinding,
        level: level_u32,
    };
    prove_stage2_with_stream::<F, E, _>(
        level,
        &mut stream,
        batching_coeff,
        rs,
        stage1_point,
        range_image_evaluation,
        relation_claim,
        compression,
        physical_l2,
        linear_terms,
        trace_opening_claim,
        plan,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_stage2_with_stream<F, E, S>(
    level: usize,
    stream: &mut S,
    batching_coeff: E,
    rs: RingSwitchOutput<E>,
    stage1_point: &[E],
    range_image_evaluation: E,
    relation_claim: E,
    compression: Stage2Compression<E>,
    physical_l2: Option<PhysicalL2ProverReplay<E>>,
    linear_terms: PreparedProverLinearTerms<E>,
    trace_opening_claim: E,
    plan: RelationRangeImagePlan,
) -> Result<Stage2ProveOutput<E, S::Proof>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: ExtField<F> + Unreduced + Fold + Ring + AkitaSerialize,
    S: Stage2ProverStream<F, E>,
{
    let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
    let domain = plan.digit_witness_domain();
    let geometry = rs.relation_address_geometry;
    let live_relation_lane_count = geometry.live_relation_lane_count();
    let relation_lane_variable_count = geometry.relation_lane_variable_count();
    let relation_coefficient_variable_count = geometry.relation_coefficient_variable_count();
    if plan.relation_address_geometry() != geometry
        || domain.live_len() != rs.w_evals_compact.len()
        || plan.digit_range_plan().basis() != rs.b
    {
        return Err(AkitaError::InvalidSetup(
            "ring-switch output disagrees with the relation/range-image plan".into(),
        ));
    }
    let relation_weights = rs.relation_weights;
    let domain_len = domain.domain_len();
    let (mut linear_weights, binary_intervals, binary_batching) = match compression {
        Stage2Compression::Raw => (Vec::new(), Vec::new(), E::zero()),
        Stage2Compression::QuotientLift {
            weights,
            support,
            binary_batching,
        } => {
            if weights.physical_field_len() != domain_len {
                return Err(AkitaError::InvalidSetup(
                    "compression relation domain disagrees with Stage 2".into(),
                ));
            }
            (
                weights.into_sparse_entries()?,
                support.intervals().to_vec(),
                binary_batching,
            )
        }
        Stage2Compression::ReducedEvaluation {
            support,
            binary_batching,
        } => (Vec::new(), support.intervals().to_vec(), binary_batching),
    };
    let physical_l2_claim = physical_l2.as_ref().map_or_else(E::zero, |norm| norm.claim);
    if let Some(norm) = &physical_l2 {
        let families = norm.plan.virtualization_families(&norm.batching)?;
        let equality = OffsetEqWindow::new(&norm.point)?;
        linear_weights.extend(
            materialize_eq_tensor_left(&equality, &families, domain.live_len())?
                .into_iter()
                .enumerate()
                .filter(|(_, weight)| !weight.is_zero()),
        );
        linear_weights.sort_unstable_by_key(|(index, _)| *index);
    }
    let additional_relation_terms = (!linear_weights.is_empty() || !binary_intervals.is_empty())
        .then(|| {
            AdditionalRelationTerms::new(
                &rs.w_evals_compact,
                domain_len,
                linear_weights,
                &binary_intervals,
                stage1_point,
                binary_batching,
            )
        })
        .transpose()?;
    let ordinary_relation_claim = relation_claim + physical_l2_claim
        - additional_relation_terms
            .as_ref()
            .map_or_else(E::zero, AdditionalRelationTerms::input_claim);
    let mut stage2_prover = RelationRangeImageProver::new(
        batching_coeff,
        rs.w_evals_compact,
        stage1_point,
        range_image_evaluation,
        plan.digit_range_plan().basis(),
        relation_weights,
        live_relation_lane_count,
        relation_lane_variable_count,
        relation_coefficient_variable_count,
        ordinary_relation_claim,
        linear_terms,
        trace_opening_claim,
        additional_relation_terms,
    )
    .map_err(|err| {
        AkitaError::InvalidInput(format!(
            "stage-2 prover initialization failed at fold level {level}: {err}"
        ))
    })?;
    let (stage2_sumcheck_proof, sumcheck_challenges, final_claim) =
        stream.prove(&mut stage2_prover)?;
    if final_claim != stage2_prover.expected_final_claim()? {
        return Err(AkitaError::InvalidInput(
            "stage-2 prover final claim disagrees with its folded oracle".into(),
        ));
    }
    Ok(Stage2ProveOutput {
        proof: stage2_sumcheck_proof,
        challenges: sumcheck_challenges,
        prover: stage2_prover,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_stage3<F, E, T>(
    level: usize,
    setup_contribution_mode: SetupContributionMode,
    expanded: &AkitaExpandedSetup<F>,
    prefix_slots: &SetupPrefixProverRegistry<F>,
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
            let mut stage3_prover = {
                let _prepare_span = tracing::info_span!("stage3_prover_prepare").entered();
                AkitaStage3Prover::new::<T>(
                    expanded,
                    prefix_slots,
                    lp,
                    next_level_params,
                    instance,
                    tau1,
                    alpha,
                    sumcheck_challenges,
                    relation_address_geometry,
                    transcript,
                )?
            };
            let level = u32::try_from(level)
                .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
            let mut round = 0u32;
            let output = stage3_prover.prove::<T, _>(transcript, |tr| {
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
            Ok(Some(Stage3ProveOutput {
                proof: SetupSumcheckProof {
                    claim: output.setup_product_claim,
                    setup_prefix_eval: output.setup_prefix_eval,
                    sumcheck: output.sumcheck,
                },
                setup_prefix_point: output.setup_prefix_point,
            }))
        }
        SetupContributionMode::Direct => Ok(None),
    }
}

#[allow(dead_code)] // Called by the native fold driver during cutover.
#[allow(clippy::too_many_arguments)]
pub(super) fn prove_stage3_native<F, E>(
    level: usize,
    setup_contribution_mode: SetupContributionMode,
    expanded: &AkitaExpandedSetup<F>,
    prefix_slots: &SetupPrefixProverRegistry<F>,
    lp: &CommittedGroupParams,
    next_level_params: &CommittedGroupParams,
    instance: &RingRelationInstance<F>,
    tau1: &[E],
    alpha: E,
    sumcheck_challenges: &[E],
    relation_address_geometry: akita_types::RelationAddressGeometry,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
) -> Result<Option<crate::protocol::sumcheck::NativeAkitaStage3ProverOutput<E>>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F>
        + Ring
        + ExtField<F>
        + AkitaSerialize
        + jolt_field::Unreduced
        + jolt_field::MulBaseUnreduced<F>,
{
    match setup_contribution_mode {
        SetupContributionMode::Recursive => {
            let level = u32::try_from(level)
                .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
            let mut prover = AkitaStage3Prover::new_native(
                expanded,
                prefix_slots,
                lp,
                next_level_params,
                instance,
                tau1,
                alpha,
                sumcheck_challenges,
                relation_address_geometry,
                grinding,
                level,
            )?;
            Ok(Some(prover.prove_native(grinding, level)?))
        }
        SetupContributionMode::Direct => Ok(None),
    }
}
