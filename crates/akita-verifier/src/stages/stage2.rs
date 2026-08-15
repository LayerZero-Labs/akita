//! Verifier for the Akita stage-2 fused sumcheck.

use crate::protocol::evaluation_trace::PreparedEvaluationTrace;
use crate::protocol::ring_switch::RelationMatrixEvaluator;
use akita_algebra::{
    eq_poly::EqPolynomial,
    offset_eq::{eval_boolean_pair_tensor_families, EqPairTensorFamily},
};
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
    MulBaseUnreduced,
};
use akita_sumcheck::SumcheckInstanceVerifier;
use akita_types::{
    AkitaExpandedSetup, CoefficientPackingBatchSemantics, CoefficientPackingGroupSemantics,
    CompressionRelationWeights, FpExtEncoding, NegativeBinarySupport, RingRelationInstance,
};

/// Verifier for the stage-2 fused virtual-claim and relation sumcheck.
pub(crate) struct AkitaStage2Verifier<'a, F: FieldCore, E: FieldCore> {
    batching_coeff: E,
    range_image_evaluation: E,
    witness_eval: E,
    stage1_point: Vec<E>,
    relation_matrix_evaluator: &'a RelationMatrixEvaluator<E>,
    compression_relation_weights: Option<&'a CompressionRelationWeights<E>>,
    negative_binary_support: Option<&'a NegativeBinarySupport>,
    binary_batching: Option<E>,
    setup_claim: Option<E>,
    setup: &'a AkitaExpandedSetup<F>,
    alpha: E,
    num_rounds: usize,
    relation_claim: E,
    evaluation_trace: Option<PreparedEvaluationTrace<E>>,
    evaluation_trace_row_weight: E,
    evaluation_trace_opening_claim: E,
    coefficient_packing_groups: &'a [CoefficientPackingGroupSemantics<E>],
    coefficient_packing_opening_claim: E,
    physical_l2_claim: E,
    physical_l2_families: Vec<EqPairTensorFamily<E>>,
    _marker: std::marker::PhantomData<F>,
}

impl<'a, F, E> AkitaStage2Verifier<'a, F, E>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: ExtField<F> + FpExtEncoding<F> + FromPrimitiveInt + MulBaseUnreduced<F>,
{
    /// Construct a verifier from the shared stage-2 context and the witness
    /// oracle selected by the current proof level.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "AkitaStage2Verifier::new")]
    pub(crate) fn new(
        batching_coeff: E,
        range_image_evaluation: E,
        witness_eval: E,
        stage1_point: Vec<E>,
        relation_matrix_evaluator: &'a RelationMatrixEvaluator<E>,
        compression_relation_weights: Option<&'a CompressionRelationWeights<E>>,
        negative_binary_support: Option<&'a NegativeBinarySupport>,
        binary_batching: Option<E>,
        setup: &'a AkitaExpandedSetup<F>,
        alpha: E,
        setup_claim: Option<E>,
        relation_claim: E,
        col_bits: usize,
        ring_bits: usize,
        evaluation_trace: Option<PreparedEvaluationTrace<E>>,
        evaluation_trace_row_weight: E,
        evaluation_trace_opening_claim: E,
        relation: &RingRelationInstance<F>,
        relation_row_point: &[E],
        claim_coefficients: &[E],
        coefficient_packing_batch: Option<&'a CoefficientPackingBatchSemantics<F, E>>,
        coefficient_packing_scalar_openings: &[(usize, E)],
        physical_l2_claim: E,
        physical_l2_families: Vec<EqPairTensorFamily<E>>,
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
        if physical_l2_families.is_empty() && !physical_l2_claim.is_zero() {
            return Err(AkitaError::InvalidProof);
        }
        let context = relation_matrix_evaluator
            .flat_context
            .as_ref()
            .ok_or(AkitaError::InvalidProof)?;
        let coefficient_packing_groups = if let Some(batch) = coefficient_packing_batch {
            batch.validate_context(
                &context.level_params,
                &context.opening_batch,
                relation,
                alpha,
                relation_row_point,
                claim_coefficients,
            )?;
            if batch.relation_plan().witness_layout() != context.witness_layout.as_ref()
                || batch.relation_plan().relation_address_geometry()
                    != relation_matrix_evaluator.relation_address_geometry
            {
                return Err(AkitaError::InvalidSetup(
                    "coefficient-packing verifier plan disagrees with ring switch".into(),
                ));
            }
            batch.groups()
        } else {
            &[]
        };
        let expected_packing_groups = relation
            .group_openings()
            .iter()
            .filter(|opening| opening.coefficient_packing_geometry().is_some())
            .count();
        if evaluation_trace.is_some() == (expected_packing_groups != 0) {
            return Err(AkitaError::InvalidSetup(
                "Stage 2 opening semantics disagree with the relation methods".into(),
            ));
        }
        if coefficient_packing_groups.len() != expected_packing_groups {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing verifier batch is incomplete".into(),
            ));
        }
        let mut coefficient_packing_opening_claim = E::zero();
        if coefficient_packing_scalar_openings.len() != coefficient_packing_groups.len() {
            return Err(AkitaError::InvalidProof);
        }
        for semantics in coefficient_packing_groups {
            let group_index = semantics.group_index();
            let authenticated_opening = coefficient_packing_scalar_openings
                .iter()
                .find_map(|&(group, opening)| (group == group_index).then_some(opening))
                .ok_or(AkitaError::InvalidProof)?;
            coefficient_packing_opening_claim +=
                semantics.stage2_terms().scalar_claim_weight() * authenticated_opening;
        }
        Ok(Self {
            batching_coeff,
            range_image_evaluation,
            witness_eval,
            stage1_point,
            relation_matrix_evaluator,
            compression_relation_weights,
            negative_binary_support,
            binary_batching,
            setup_claim,
            setup,
            alpha,
            num_rounds,
            relation_claim,
            evaluation_trace,
            evaluation_trace_row_weight,
            evaluation_trace_opening_claim,
            coefficient_packing_groups,
            coefficient_packing_opening_claim,
            physical_l2_claim,
            physical_l2_families,
            _marker: std::marker::PhantomData,
        })
    }

    fn coefficient_packing_weight_at_point(&self, point: &[E]) -> Result<E, AkitaError> {
        self.coefficient_packing_groups
            .iter()
            .try_fold(E::zero(), |sum, semantics| {
                Ok(sum
                    + semantics.relation_events().evaluate_at_point(point)?
                    + semantics.stage2_terms().evaluate_at_point(point)?)
            })
    }
}

impl<'a, F, E> SumcheckInstanceVerifier<E> for AkitaStage2Verifier<'a, F, E>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: ExtField<F> + FpExtEncoding<F> + FromPrimitiveInt + MulBaseUnreduced<F>,
{
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> E {
        self.batching_coeff * self.range_image_evaluation
            + self.relation_claim
            + self.evaluation_trace_opening_claim
            + self.coefficient_packing_opening_claim
            + self.physical_l2_claim
    }

    #[tracing::instrument(skip_all, name = "stage2_expected_output_claim")]
    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        let w_eval = {
            let _span = tracing::info_span!("stage2_witness_eval").entered();
            self.witness_eval
        };

        let relation_weight = {
            let _span = tracing::info_span!("stage2_relation_weight").entered();
            self.relation_matrix_evaluator.eval_flat_at_point::<F>(
                challenges,
                self.setup,
                self.alpha,
                self.setup_claim,
            )?
        };
        let compression_oracle = match (
            self.compression_relation_weights,
            self.negative_binary_support,
            self.binary_batching,
        ) {
            (Some(weights), Some(support), Some(binary_batching)) => {
                let compression_relation_weight = weights.evaluate_at_point(challenges)?;
                let binary_weight = support
                    .evaluate_restricted_equality_at_point(&self.stage1_point, challenges)?;
                w_eval * compression_relation_weight
                    + binary_batching * binary_weight * w_eval * (w_eval + E::one())
            }
            (None, None, None) => E::zero(),
            _ => return Err(AkitaError::InvalidProof),
        };
        let coefficient_packing_weight = self.coefficient_packing_weight_at_point(challenges)?;
        let relation_oracle =
            w_eval * (relation_weight + coefficient_packing_weight) + compression_oracle;
        let trace_oracle = if let Some(evaluation_trace) = &self.evaluation_trace {
            let _span = tracing::info_span!("stage2_trace_oracle").entered();
            let trace_weight = evaluation_trace.evaluate_at_point(challenges)?;
            self.evaluation_trace_row_weight * w_eval * trace_weight
        } else {
            E::zero()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ring_switch::{FlatRelationContext, RelationMatrixEvaluator};
    use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
    use akita_field::{Ext2, Prime64Offset59};
    use akita_types::{
        prepare_coefficient_packing_batch_semantics, r_decomp_levels, relation_rhs_coeff_len,
        AkitaSetupDescriptor, BasisMode, CoefficientPackingBatchSemanticInputs,
        CommitmentPayloadMode, DigitRangePlan, FlatMatrix, OpenCommitMatrixParams,
        OpeningClaimsLayout, OpeningMethod, PreparedSubringCoefficientPackingPoint,
        RelationAddressGeometry, RelationRangeImagePlan, RelationWitnessGeometry,
        RingRelationGroupOpening, RingVec, SisModulusProfileId, SubringCoefficientPackingGeometry,
        WitnessLayout,
    };
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
        params.opening_method = OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: s,
        };
        let opening = params.open_commit_matrix;
        params.open_commit_matrix = OpenCommitMatrixParams::new_unchecked(
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
            r_decomp_levels::<F>(params.log_basis_open),
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
            vec![
                RingRelationGroupOpening::subring_coefficient_packing(geometry, challenges)
                    .unwrap(),
            ],
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
            RingVec::from_coeffs(Vec::new()),
            params.role_dims(),
        )
        .unwrap();
        let alpha = E::from_u64(17);
        let claim_coefficients = vec![E::from_u64(7), E::from_u64(11)];
        let tau1 = (0..relation_plan.relation_row_index_num_vars().unwrap())
            .map(|index| E::from_u64(13 + index as u64))
            .collect::<Vec<_>>();
        let batch =
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
            groups: Vec::new(),
            log_basis: params.log_basis_open,
            eq_tau1: Arc::from(Vec::<E>::new()),
            flat_context: Some(FlatRelationContext {
                level_params: params.clone(),
                opening_batch: opening_batch.clone(),
                witness_layout: Arc::new(witness_layout),
                extension_degree: <E as ExtField<F>>::EXT_DEGREE,
            }),
            setup_plan_cache: Default::default(),
        };
        let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
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
        let verifier = AkitaStage2Verifier::new(
            E::zero(),
            E::zero(),
            E::from_u64(23),
            vec![E::zero(); domain.num_vars()],
            &evaluator,
            None,
            None,
            None,
            &setup,
            alpha,
            None,
            E::zero(),
            relation_address_geometry.relation_lane_variable_count(),
            relation_address_geometry.relation_coefficient_variable_count(),
            None,
            E::zero(),
            E::zero(),
            &relation,
            &tau1,
            &claim_coefficients,
            Some(&batch),
            &[(0, scalar_opening)],
            E::zero(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            verifier.input_claim(),
            batch.groups()[0].stage2_terms().scalar_claim_weight() * scalar_opening
        );
        let point = (0..domain.num_vars())
            .map(|index| E::from_u64(29 + index as u64))
            .collect::<Vec<_>>();
        assert_eq!(
            verifier
                .coefficient_packing_weight_at_point(&point)
                .unwrap(),
            batch.groups()[0]
                .relation_events()
                .evaluate_at_point(&point)
                .unwrap()
                + batch.groups()[0]
                    .stage2_terms()
                    .evaluate_at_point(&point)
                    .unwrap()
        );

        let mut wrong_tau1 = tau1.clone();
        wrong_tau1[0] += E::one();
        assert!(AkitaStage2Verifier::new(
            E::zero(),
            E::zero(),
            E::one(),
            vec![E::zero(); domain.num_vars()],
            &evaluator,
            None,
            None,
            None,
            &setup,
            alpha,
            None,
            E::zero(),
            relation_address_geometry.relation_lane_variable_count(),
            relation_address_geometry.relation_coefficient_variable_count(),
            None,
            E::zero(),
            E::zero(),
            &relation,
            &wrong_tau1,
            &claim_coefficients,
            Some(&batch),
            &[(0, scalar_opening)],
            E::zero(),
            Vec::new(),
        )
        .is_err());
        assert!(AkitaStage2Verifier::new(
            E::zero(),
            E::zero(),
            E::one(),
            vec![E::zero(); domain.num_vars()],
            &evaluator,
            None,
            None,
            None,
            &setup,
            alpha,
            None,
            E::zero(),
            relation_address_geometry.relation_lane_variable_count(),
            relation_address_geometry.relation_coefficient_variable_count(),
            None,
            E::zero(),
            E::zero(),
            &relation,
            &tau1,
            &claim_coefficients,
            Some(&batch),
            &[(0, scalar_opening), (0, scalar_opening)],
            E::zero(),
            Vec::new(),
        )
        .is_err());
    }
}
