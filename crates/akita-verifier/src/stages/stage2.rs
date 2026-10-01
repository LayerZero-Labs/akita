//! Verifier for the Akita stage-2 fused sumcheck.

use crate::coefficient_packing_relation::{
    CoefficientPackingVerifierBatchSemantics, CoefficientPackingVerifierGroupSemantics,
};
use crate::relation::evaluation_trace::PreparedEvaluationTrace;
use crate::relation::{PreparedRelationGroups, RelationMatrixEvaluator};
use crate::stages::ring_switch::RingSwitchVerifyOutput;
use crate::stages::stage1::Stage1Replay;
use akita_algebra::{
    eq_poly::EqPolynomial,
    offset_eq::{eval_boolean_pair_tensor_families, EqPairTensorFamily},
};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::AkitaVerifierSetup;
use akita_types::{
    AkitaExpandedSetup, CompressionRelationWeights, FpExtEncoding, NegativeBinarySupport,
    OpeningFamily, ReducedCompressionRelationWeights,
};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

pub(crate) struct EvaluationTraceStage2<E: Field> {
    pub(crate) trace: PreparedEvaluationTrace<E>,
    pub(crate) row_weight: E,
    pub(crate) opening_claim: E,
}

pub(crate) struct PackingStage2<'a, E: Field> {
    groups: &'a [CoefficientPackingVerifierGroupSemantics<E>],
    opening_claim: E,
}

pub(crate) struct Stage2OpeningSemantics<'a, E: Field>(
    OpeningFamily<EvaluationTraceStage2<E>, PackingStage2<'a, E>>,
);

impl<'a, E: Field> PackingStage2<'a, E> {
    pub(crate) fn new(
        batch: &'a CoefficientPackingVerifierBatchSemantics<E>,
        scalar_openings: &[(usize, E)],
    ) -> Result<Self, AkitaError> {
        let groups = batch.groups();
        if groups.is_empty() || scalar_openings.len() != groups.len() {
            return Err(AkitaError::InvalidProof);
        }
        let mut opening_claim = E::zero();
        for semantics in groups {
            let authenticated_opening = scalar_openings
                .iter()
                .find_map(|&(group, opening)| (group == semantics.group_index()).then_some(opening))
                .ok_or(AkitaError::InvalidProof)?;
            opening_claim += semantics.scalar_claim_weight() * authenticated_opening;
        }
        Ok(Self {
            groups,
            opening_claim,
        })
    }

    fn weight_at_point(&self, point: &[E]) -> Result<E, AkitaError> {
        let evaluate_group = |semantics: &CoefficientPackingVerifierGroupSemantics<E>| {
            let (relation, structured) = cfg_join!(
                || {
                    let _span =
                        tracing::info_span!("coefficient_packing_relation_weight").entered();
                    semantics
                        .compact_factors()
                        .evaluate_relation_at_point(point)
                },
                || {
                    let _span =
                        tracing::info_span!("coefficient_packing_structured_weight").entered();
                    semantics.compact_factors().evaluate_stage2_at_point(point)
                }
            );
            Ok::<_, AkitaError>(relation? + structured?)
        };
        if let [semantics] = self.groups {
            return evaluate_group(semantics);
        }
        #[cfg(feature = "parallel")]
        {
            self.groups
                .par_iter()
                .map(evaluate_group)
                .try_reduce(|| E::zero(), |left, right| Ok(left + right))
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.groups.iter().try_fold(E::zero(), |sum, semantics| {
                Ok(sum + evaluate_group(semantics)?)
            })
        }
    }
}

impl<'a, E: Field> Stage2OpeningSemantics<'a, E> {
    pub(crate) fn evaluation_trace(
        trace: PreparedEvaluationTrace<E>,
        row_weight: E,
        opening_claim: E,
    ) -> Self {
        Self(OpeningFamily::EvaluationTrace(EvaluationTraceStage2 {
            trace,
            row_weight,
            opening_claim,
        }))
    }

    pub(crate) fn packing(
        batch: &'a CoefficientPackingVerifierBatchSemantics<E>,
        scalar_openings: &[(usize, E)],
    ) -> Result<Self, AkitaError> {
        Ok(Self(OpeningFamily::SubringCoefficientPacking(
            PackingStage2::new(batch, scalar_openings)?,
        )))
    }

    pub(crate) fn opening_claim(&self) -> E {
        match &self.0 {
            OpeningFamily::EvaluationTrace(trace) => trace.opening_claim,
            OpeningFamily::SubringCoefficientPacking(packing) => packing.opening_claim,
        }
    }
}

/// Verifier for the stage-2 fused virtual-claim and relation sumcheck.
pub(crate) struct AkitaStage2Verifier<'a, F: Field, E: Field> {
    batching_coeff: E,
    witness_eval: E,
    stage1_point: Vec<E>,
    relation_matrix_evaluator: &'a RelationMatrixEvaluator<E>,
    compression: Stage2CompressionOracle<'a, E>,
    setup_claim: Option<E>,
    setup: &'a AkitaExpandedSetup<F>,
    alpha: E,
    opening_semantics: Stage2OpeningSemantics<'a, E>,
    physical_l2_families: Vec<EqPairTensorFamily<E>>,
    _marker: std::marker::PhantomData<F>,
}

/// Replayed inputs for [`AkitaStage2Verifier::new`].
pub(crate) struct Stage2VerifierInput<'a, F: Field, E: Field> {
    /// Stage 2 batching coefficient drawn after Stage 1.
    pub(crate) batching_coeff: E,
    /// Prover-sent witness evaluation at the Stage 2 point.
    pub(crate) witness_eval: E,
    /// Stage 1 sumcheck point.
    pub(crate) stage1_point: Vec<E>,
    /// Prepared relation-matrix evaluator from the ring switch.
    pub(crate) relation_matrix_evaluator: &'a RelationMatrixEvaluator<E>,
    /// Payload-mode compression oracle.
    pub(crate) compression: Stage2CompressionOracle<'a, E>,
    /// Expanded public setup.
    pub(crate) setup: &'a AkitaExpandedSetup<F>,
    /// Ring-switch alpha challenge.
    pub(crate) alpha: E,
    /// Stage 3 setup claim when the setup contribution is deferred.
    pub(crate) setup_claim: Option<E>,
    /// Relation lane variable count.
    pub(crate) col_bits: usize,
    /// Relation coefficient variable count.
    pub(crate) ring_bits: usize,
    /// Opening-claim semantics for this level.
    pub(crate) opening_semantics: Stage2OpeningSemantics<'a, E>,
    /// Batched physical L2 virtual claim, zero without a physical plan.
    pub(crate) physical_l2_claim: E,
    /// Physical L2 virtualization families.
    pub(crate) physical_l2_families: Vec<EqPairTensorFamily<E>>,
}

pub(crate) enum Stage2CompressionOracle<'a, E: Field> {
    Raw,
    QuotientLift {
        weights: &'a CompressionRelationWeights<E>,
        support: &'a NegativeBinarySupport,
        binary_batching: E,
    },
    ReducedEvaluation {
        weights: &'a ReducedCompressionRelationWeights<E>,
        support: &'a NegativeBinarySupport,
        binary_batching: E,
    },
}

impl<'a, F, E> AkitaStage2Verifier<'a, F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F> + Ring + MulBaseUnreduced<F>,
{
    /// Construct a verifier from the shared stage-2 context and the witness
    /// oracle selected by the current proof level.
    #[tracing::instrument(skip_all, name = "AkitaStage2Verifier::new")]
    pub(crate) fn new(input: Stage2VerifierInput<'a, F, E>) -> Result<Self, AkitaError> {
        let Stage2VerifierInput {
            batching_coeff,
            witness_eval,
            stage1_point,
            relation_matrix_evaluator,
            compression,
            setup,
            alpha,
            setup_claim,
            col_bits,
            ring_bits,
            opening_semantics,
            physical_l2_claim,
            physical_l2_families,
        } = input;
        let num_rounds = col_bits.checked_add(ring_bits).ok_or_else(|| {
            AkitaError::InvalidSetup("stage-2 variable count overflow".to_string())
        })?;
        if stage1_point.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: stage1_point.len(),
            });
        }
        if physical_l2_families.is_empty() && !physical_l2_claim.is_zero() {
            return Err(AkitaError::InvalidProof);
        }
        Ok(Self {
            batching_coeff,
            witness_eval,
            stage1_point,
            relation_matrix_evaluator,
            compression,
            setup_claim,
            setup,
            alpha,
            opening_semantics,
            physical_l2_families,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<'a, F, E> AkitaStage2Verifier<'a, F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F> + Ring + MulBaseUnreduced<F>,
{
    #[tracing::instrument(skip_all, name = "stage2_expected_output_claim")]
    pub(crate) fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        let w_eval = {
            let _span = tracing::info_span!("stage2_witness_eval").entered();
            self.witness_eval
        };

        let relation_is_reduced = matches!(
            &self.relation_matrix_evaluator.groups,
            PreparedRelationGroups::ReducedEvaluation(_)
        );
        let evaluate_relation_weight = || {
            // `cfg_join!` may execute this closure on a Rayon worker which does
            // not inherit the caller's `stage2_verifier` span. Carry the
            // authenticated relation mode on this worker-local owner so phase
            // diagnostics cannot silently lose coefficient-packing folds.
            let _span =
                tracing::info_span!("stage2_relation_weight", reduced = relation_is_reduced)
                    .entered();
            match self.setup_claim {
                Some(claim) => self
                    .relation_matrix_evaluator
                    .eval_flat_at_point_with_deferred_setup::<F>(challenges, self.alpha, claim),
                None => self
                    .relation_matrix_evaluator
                    .eval_flat_at_point::<F>(challenges, self.setup, self.alpha),
            }
        };
        let (relation_weight, coefficient_packing_weight) = match &self.opening_semantics.0 {
            OpeningFamily::EvaluationTrace(_) => (evaluate_relation_weight()?, E::zero()),
            OpeningFamily::SubringCoefficientPacking(packing) => {
                let (relation_weight, coefficient_packing_weight) =
                    cfg_join!(evaluate_relation_weight, || packing
                        .weight_at_point(challenges));
                (relation_weight?, coefficient_packing_weight?)
            }
        };
        let compression_oracle = {
            let _span = tracing::info_span!(
                "stage2_compression_oracle",
                reduced = matches!(
                    self.compression,
                    Stage2CompressionOracle::ReducedEvaluation { .. }
                )
            )
            .entered();
            evaluate_compression_oracle(
                &self.compression,
                self.setup,
                &self.stage1_point,
                challenges,
                w_eval,
            )?
        };
        let relation_oracle =
            w_eval * (relation_weight + coefficient_packing_weight) + compression_oracle;
        let trace_oracle = match &self.opening_semantics.0 {
            OpeningFamily::EvaluationTrace(trace) => {
                let _span = tracing::info_span!("stage2_trace_oracle").entered();
                trace.row_weight * w_eval * trace.trace.evaluate_at_point(challenges)?
            }
            OpeningFamily::SubringCoefficientPacking(_) => E::zero(),
        };
        let physical_l2_oracle = if self.physical_l2_families.is_empty() {
            E::zero()
        } else {
            let weight_eval = eval_boolean_pair_tensor_families::<_, false, false>(
                challenges,
                &self.stage1_point,
                &self.physical_l2_families,
            )?;
            w_eval * weight_eval
        };

        // A zero batching challenge removes the virtual term. Avoid the
        // unnecessary EqPolynomial evaluation in that degenerate case.
        if self.batching_coeff.is_zero() {
            return Ok(relation_oracle + trace_oracle + physical_l2_oracle);
        }
        let virtual_oracle = {
            let _span = tracing::info_span!("stage2_virtual_oracle").entered();
            let eq_val = EqPolynomial::mle(&self.stage1_point, challenges)?;
            eq_val * w_eval * (w_eval + E::one())
        };
        Ok(self.batching_coeff * virtual_oracle
            + relation_oracle
            + trace_oracle
            + physical_l2_oracle)
    }
}

/// Stage 2 sumcheck rounds and the prover's witness evaluation, before the
/// output claim is checked.
pub(crate) struct Stage2RoundReplay<E: Field> {
    pub(crate) output_claim: E,
    pub(crate) challenges: Vec<E>,
    pub(crate) witness_eval: E,
}

/// Checked Stage 2 output: the fold's next opening point and claim.
pub(crate) struct Stage2Output<E: Field> {
    pub(crate) point: Vec<E>,
    pub(crate) witness_eval: E,
}

/// Replay the Stage 2 rounds from the batched Stage 1, relation, opening, and
/// physical L2 input claims.
pub(crate) fn replay_stage2<F, E>(
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    level: u32,
    stage1: &Stage1Replay<'_, E>,
    relation_claim: E,
    opening_semantics: &Stage2OpeningSemantics<'_, E>,
    shape: akita_sumcheck::SumcheckShape,
) -> Result<Stage2RoundReplay<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let input_claim = stage1.batching_coeff * stage1.range_image_evaluation
        + relation_claim
        + opening_semantics.opening_claim()
        + stage1.physical_l2_claim;
    let mut channel = akita_types::GrindingSumcheckVerifier::<F, E>::new(
        grinding,
        akita_types::SumcheckProtocol::Stage2,
        level,
        0,
    );
    let replay =
        akita_sumcheck::verify_sumcheck_rounds::<F, E, _>(&mut channel, 0, input_claim, shape)?;
    let witness_eval = akita_types::stage2_w_eval::<F, E, _>(grinding, level, E::zero())?;
    Ok(Stage2RoundReplay {
        output_claim: replay.output_claim,
        challenges: replay.challenges,
        witness_eval,
    })
}

/// Check the Stage 2 output claim once the Stage 3 setup claim is known.
pub(crate) fn validate_stage2_replay<F, E>(
    setup: &AkitaVerifierSetup<F>,
    stage1: Stage1Replay<'_, E>,
    rs: &RingSwitchVerifyOutput<E>,
    setup_claim: Option<E>,
    opening_semantics: Stage2OpeningSemantics<'_, E>,
    replay: Stage2RoundReplay<E>,
) -> Result<Stage2Output<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    let witness_eval = replay.witness_eval;
    let stage2_verifier = AkitaStage2Verifier::<F, E>::new(Stage2VerifierInput {
        batching_coeff: stage1.batching_coeff,
        witness_eval,
        stage1_point: stage1.stage1_point,
        relation_matrix_evaluator: &rs.relation_matrix_evaluator,
        compression: stage1.compression,
        setup: setup.expanded(),
        alpha: rs.alpha,
        setup_claim,
        col_bits: rs.relation_address_geometry.relation_lane_variable_count(),
        ring_bits: rs
            .relation_address_geometry
            .relation_coefficient_variable_count(),
        opening_semantics,
        physical_l2_claim: stage1.physical_l2_claim,
        physical_l2_families: stage1.physical_l2_families,
    })?;

    let expected = stage2_verifier.expected_output_claim(&replay.challenges)?;
    if replay.output_claim != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(Stage2Output {
        point: replay.challenges,
        witness_eval,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_compression_oracle<F, E>(
    compression: &Stage2CompressionOracle<'_, E>,
    setup: &AkitaExpandedSetup<F>,
    stage1_point: &[E],
    point: &[E],
    witness_evaluation: E,
) -> Result<E, AkitaError>
where
    F: Field,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    match compression {
        Stage2CompressionOracle::QuotientLift {
            weights,
            support,
            binary_batching,
        } => {
            let relation_weight = weights.evaluate_at_point(point)?;
            let binary_weight =
                support.evaluate_restricted_equality_at_point(stage1_point, point)?;
            Ok(witness_evaluation * relation_weight
                + *binary_batching
                    * binary_weight
                    * witness_evaluation
                    * (witness_evaluation + E::one()))
        }
        Stage2CompressionOracle::ReducedEvaluation {
            weights,
            support,
            binary_batching,
        } => {
            let relation_weight = weights.evaluate_at_point(setup, point)?;
            let binary_weight =
                support.evaluate_restricted_equality_at_point(stage1_point, point)?;
            Ok(witness_evaluation * relation_weight
                + *binary_batching
                    * binary_weight
                    * witness_evaluation
                    * (witness_evaluation + E::one()))
        }
        Stage2CompressionOracle::Raw => Ok(E::zero()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coefficient_packing_relation::prepare_coefficient_packing_verifier_batch_semantics;
    use crate::coefficient_packing_relation::tests::materialize_stage2;
    use crate::relation::{FlatRelationContext, RelationMatrixEvaluator};
    use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
    use akita_types::{
        prepare_coefficient_packing_batch_semantics, relation_rhs_coeff_len, AkitaSetupDescriptor,
        BasisMode, CoefficientPackingBatchSemanticInputs, CommitmentPayloadMode, DigitRangePlan,
        FlatMatrix, OpenCommitMatrixParams, OpeningClaimsLayout, OpeningMethod,
        PreparedSubringCoefficientPackingPoint, RelationAddressGeometry, RelationRangeImagePlan,
        RelationWitnessGeometry, RingRelationGroupOpening, RingRelationInstance, RingVec,
        SisModulusProfileId, SubringCoefficientPackingGeometry, WitnessLayout,
    };
    use jolt_field::Zero;
    use jolt_field::{Ext2, Prime64Offset59};
    use std::sync::Arc;

    type F = Prime64Offset59;
    type E = Ext2<F>;

    #[test]
    fn packing_batch_drives_stage2_claim_and_compact_weight_once() {
        let s = 64;
        let d_a = 256;
        let d_d = 128;
        let challenge_config = SparseChallengeConfig::production_for_ring_dim(s).unwrap();
        let mut params = akita_types::CommittedGroupParams::params_only(
            SisModulusProfileId::Q64Offset59,
            d_a,
            2,
            2,
            2,
            2,
            challenge_config,
        )
        .with_decomp(4, 6, 2, 2, 2)
        .unwrap();
        params.payload_mode = CommitmentPayloadMode::Raw;
        params.own_group_mut().opening.opening_method = OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: s,
        };
        let opening = params.open().matrix;
        params.open_matrix = OpenCommitMatrixParams::new_unchecked(
            opening.security_policy(),
            opening.sis_table_key().table_digest,
            opening.sis_modulus_profile(),
            opening.output_rank(),
            opening.input_width(),
            opening.coeff_linf_bound(),
            d_d,
        );
        let opening_batch = OpeningClaimsLayout::new(11, 2).unwrap();
        let relation_geometry =
            RelationWitnessGeometry::for_level(&params, &opening_batch, 2).unwrap();
        let witness_layout = WitnessLayout::new(
            &params,
            &opening_batch,
            &relation_geometry,
            1,
            akita_types::RelationQuotientPlan::for_field_bits(&params, F::MODULUS_BITS)
                .expect("relation quotient plan"),
        )
        .unwrap();
        let relation_address_geometry = RelationAddressGeometry::for_relation(
            &relation_geometry,
            d_d,
            witness_layout.live_coeff_len(),
        )
        .unwrap();
        let relation_plan = RelationRangeImagePlan::new(
            relation_geometry.clone(),
            relation_address_geometry,
            DigitRangePlan::new(4).unwrap(),
            witness_layout.clone(),
            &opening_batch,
        )
        .unwrap();
        let geometry = SubringCoefficientPackingGeometry::try_new(2, d_a, s).unwrap();
        let prepared_point = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            6,
            4,
            11,
            &(0..11)
                .map(|index| E::from_u64(2 + index as u64))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let challenges = Challenges::from_sparse(
            (0..2 * prepared_point.num_live_blocks())
                .map(|challenge| SparseChallenge {
                    positions: (0..challenge_config.weight())
                        .map(|term| ((term + challenge) % s) as u32)
                        .collect(),
                    coeffs: (0..challenge_config.count_pm1)
                        .map(|term| if term.is_multiple_of(2) { 1 } else { -1 })
                        .chain((0..challenge_config.count_pm2).map(|_| 2))
                        .collect(),
                })
                .collect(),
            prepared_point.num_live_blocks(),
            2,
        )
        .unwrap();
        let relation = RingRelationInstance::new(
            vec![RingRelationGroupOpening::coefficient_packing(
                akita_types::CoefficientPackingChallenges::new(geometry, challenges).unwrap(),
            )],
            2,
            opening_batch.clone(),
            vec![F::from_u64(3), F::from_u64(5)],
            RingVec::from_coeffs_with_ring_dim(
                [F::from_u64(3), F::from_u64(5)]
                    .into_iter()
                    .flat_map(|coefficient| {
                        let mut ring = vec![F::zero(); d_a];
                        ring[0] = coefficient;
                        ring
                    })
                    .collect(),
                d_a,
            )
            .unwrap(),
            RingVec::from_coeffs(vec![
                F::zero();
                relation_rhs_coeff_len(relation_geometry.rhs_layout())
                    .unwrap()
            ]),
            params.role_dims(),
        )
        .unwrap();
        let alpha = E::from_u64(17);
        let claim_coefficients = vec![E::from_u64(7), E::from_u64(11)];
        let tau1 = (0..relation_plan.relation_row_index_num_vars().unwrap())
            .map(|index| E::from_u64(13 + index as u64))
            .collect::<Vec<_>>();
        let batch = prepare_coefficient_packing_verifier_batch_semantics(
            CoefficientPackingBatchSemanticInputs {
                level_params: &params,
                opening_batch: &opening_batch,
                relation_plan: &relation_plan,
                relation: &relation,
                prepared_points: &[(0, &prepared_point)],
                alpha,
                tau1: &tau1,
                claim_coefficients: &claim_coefficients,
            },
        )
        .unwrap();
        let expanded_oracle =
            prepare_coefficient_packing_batch_semantics(CoefficientPackingBatchSemanticInputs {
                level_params: &params,
                opening_batch: &opening_batch,
                relation_plan: &relation_plan,
                relation: &relation,
                prepared_points: &[(0, &prepared_point)],
                alpha,
                tau1: &tau1,
                claim_coefficients: &claim_coefficients,
            })
            .unwrap();
        let evaluator = RelationMatrixEvaluator {
            relation_address_geometry,
            groups: crate::relation::PreparedRelationGroups::QuotientLift(Vec::new()),
            log_basis: params.open().digits.log_basis,
            eq_tau1: Arc::from(Vec::<E>::new()),
            flat_context: FlatRelationContext {
                level_params: params.clone(),
                opening_batch: opening_batch.clone(),
                witness_layout: Arc::new(witness_layout),
                extension_degree: <E as ExtField<F>>::DEGREE,
            },
        };
        let setup: AkitaExpandedSetup<F> =
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                AkitaSetupDescriptor {
                    max_num_vars: 0,
                    max_num_batched_polys: 0,
                    num_field_elements: 0,
                    setup_seed: [0u8; 32].into(),
                },
                FlatMatrix::from_flat_data(Vec::new()),
            );
        let domain = relation_address_geometry.digit_witness_domain();
        let scalar_opening = E::from_u64(19);
        let verifier = AkitaStage2Verifier::<F, E>::new(Stage2VerifierInput {
            batching_coeff: E::zero(),
            witness_eval: E::from_u64(23),
            stage1_point: vec![E::zero(); domain.num_vars()],
            relation_matrix_evaluator: &evaluator,
            compression: Stage2CompressionOracle::Raw,
            setup: &setup,
            alpha,
            setup_claim: None,
            col_bits: relation_address_geometry.relation_lane_variable_count(),
            ring_bits: relation_address_geometry.relation_coefficient_variable_count(),
            opening_semantics: Stage2OpeningSemantics::packing(&batch, &[(0, scalar_opening)])
                .unwrap(),
            physical_l2_claim: E::zero(),
            physical_l2_families: Vec::new(),
        })
        .unwrap();
        assert_eq!(
            verifier.opening_semantics.opening_claim(),
            batch.groups()[0].scalar_claim_weight() * scalar_opening
        );
        let point = (0..domain.num_vars())
            .map(|index| E::from_u64(29 + index as u64))
            .collect::<Vec<_>>();
        let OpeningFamily::SubringCoefficientPacking(packing) = &verifier.opening_semantics.0
        else {
            panic!("expected packing semantics");
        };
        assert_eq!(
            packing.weight_at_point(&point).unwrap(),
            batch.groups()[0]
                .compact_factors()
                .evaluate_relation_at_point(&point)
                .unwrap()
                + akita_algebra::poly::multilinear_eval(
                    &materialize_stage2(
                        &expanded_oracle.groups()[0],
                        expanded_oracle.groups()[0]
                            .physical_field_len()
                            .next_power_of_two(),
                    ),
                    &point,
                )
                .unwrap()
        );

        assert!(Stage2OpeningSemantics::packing(
            &batch,
            &[(0, scalar_opening), (0, scalar_opening)]
        )
        .is_err());
    }
}

#[cfg(test)]
#[path = "stage2/compressed_reduced_tests.rs"]
mod compressed_reduced_tests;
