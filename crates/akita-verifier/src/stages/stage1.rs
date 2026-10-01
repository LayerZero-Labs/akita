//! Stage-1 verifier instances for Akita range-check proofs.
//!
//! This module owns verifier-side replay for both the compact single-stage
//! `b <= 8` path and the staged range-check tree used for larger bases. The
//! prover-side compact witness scans and two-round-prefix kernels stay in the
//! prover/root path.

use crate::stages::ring_switch::{PreparedStage2Compression, RingSwitchVerifyOutput};
use crate::stages::stage2::Stage2CompressionOracle;
use akita_algebra::offset_eq::EqPairTensorFamily;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::GrindingReplay;
use akita_types::{
    batch_l2_virtual_evaluations, CommittedGroupParams, FpExtEncoding, InnerCommitSecurityRoute,
    PhysicalResponsePlan, RelationRangeImagePlan,
};
use akita_types::{DigitRangeEqualityPoint, DigitRangePlan};
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};
pub(crate) struct Stage1VerifyOutput<E: Field> {
    pub(crate) point: Vec<E>,
    pub(crate) range_image_evaluation: E,
    pub(crate) physical_l2_virtual_evaluations: Option<Vec<E>>,
}

pub(crate) struct RangeLeafVerifierInput<E: Field> {
    pub(crate) equality_point: Vec<E>,
    pub(crate) input_claim: E,
    pub(crate) polynomial_coefficients: Vec<E>,
}

/// Stage-1 range-check verifier, including the root/leaf tree choreography.
pub struct Stage1Verifier<E: Field> {
    equality_point: DigitRangeEqualityPoint<E>,
    plan: DigitRangePlan,
}

impl<E: Field> Stage1Verifier<E> {
    /// Construct the stage-1 verifier from a checked range topology.
    pub fn new(equality_point: DigitRangeEqualityPoint<E>, plan: DigitRangePlan) -> Self {
        Self {
            equality_point,
            plan,
        }
    }
}

impl<E: Field + Ring + AkitaSerialize> Stage1Verifier<E> {
    fn verify_product_prefix<F>(
        &self,
        grinding: &mut akita_types::VerifierGrinding<'_, '_>,
        level: u32,
    ) -> Result<RangeLeafVerifierInput<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let product_stage_arities = self.plan.product_stage_arities();
        let rounds = self.equality_point.coordinates().len();
        let leaf_coeffs = self.plan.leaf_coeffs::<E>();
        let mut current_equality_point = self.equality_point.coordinates().to_vec();
        let mut current_claim = E::zero();
        let mut current_weights = vec![E::one()];
        for (stage_index, &arity) in product_stage_arities.iter().enumerate() {
            let expected = self
                .plan
                .stage_shape(rounds, stage_index)
                .ok_or(AkitaError::InvalidProof)?;
            let stage = u32::try_from(stage_index).map_err(|_| AkitaError::InvalidProof)?;
            let mut channel = akita_types::GrindingSumcheckVerifier::<F, E>::new(
                grinding,
                akita_types::SumcheckProtocol::Stage1,
                level,
                stage,
            );
            let replay = akita_sumcheck::verify_eq_factored_sumcheck_rounds::<F, E, _>(
                &current_equality_point,
                current_claim,
                akita_sumcheck::SumcheckShape::new(rounds, arity)?,
                &mut channel,
                0,
            )?;
            let mut child_claims = akita_transcript::extension_slots::<E>(expected.child_claims)?;
            akita_types::stage1_child_claims::<F, E, _>(grinding, level, stage, &mut child_claims)?;
            let expected_output = current_weights
                .iter()
                .zip(child_claims.chunks_exact(arity))
                .fold(E::zero(), |acc, (&weight, claims)| {
                    acc + weight
                        * claims
                            .iter()
                            .copied()
                            .fold(E::one(), |product, claim| product * claim)
                });
            if replay.output_claim != expected_output {
                return Err(AkitaError::InvalidProof);
            }
            let gamma = grinding.grinded_ext_challenge::<F, E>(
                akita_types::GrindingSite::Stage1InterstageBatch { level, stage },
            )?;
            current_weights = self
                .plan
                .interstage_batch_weights(gamma, child_claims.len());
            current_claim = self.plan.batch_claims(&current_weights, &child_claims)?;
            current_equality_point = replay.challenges;
        }
        Ok(RangeLeafVerifierInput {
            equality_point: current_equality_point,
            input_claim: current_claim,
            polynomial_coefficients: self
                .plan
                .batch_leaf_polynomials(&current_weights, &leaf_coeffs)?,
        })
    }

    /// Replay the non-L2 stage-1 proof directly from Spongefish.
    pub(crate) fn verify<F>(
        &self,
        grinding: &mut akita_types::VerifierGrinding<'_, '_>,
        physical_l2: Option<(
            &akita_types::PhysicalResponsePlan,
            akita_types::SisModulusProfileId,
            u128,
        )>,
        level: u32,
    ) -> Result<Stage1VerifyOutput<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + akita_types::FpExtEncoding<F>,
    {
        let leaf = self.verify_product_prefix::<F>(grinding, level)?;
        let stage = u32::try_from(self.plan.product_stage_arities().len())
            .map_err(|_| AkitaError::InvalidProof)?;
        if let Some((plan, profile, cap)) = physical_l2 {
            let replay = super::physical_l2_norm::verify_physical_l2_norm::<F, E>(
                plan,
                super::physical_l2_norm::PhysicalL2RangeClaim {
                    equality_point: &leaf.equality_point,
                    input_claim: leaf.input_claim,
                    leaf_coefficients: &leaf.polynomial_coefficients,
                    range_stage: stage,
                },
                profile,
                cap,
                grinding,
                level,
            )?;
            return Ok(Stage1VerifyOutput {
                point: replay.point,
                range_image_evaluation: replay.range_image_evaluation,
                physical_l2_virtual_evaluations: Some(replay.virtual_evaluations),
            });
        }
        let degree_bound = leaf.polynomial_coefficients.len().saturating_sub(1);
        let mut channel = akita_types::GrindingSumcheckVerifier::<F, E>::new(
            grinding,
            akita_types::SumcheckProtocol::Stage1,
            level,
            stage,
        );
        let replay = akita_sumcheck::verify_eq_factored_sumcheck_rounds::<F, E, _>(
            &leaf.equality_point,
            leaf.input_claim,
            akita_sumcheck::SumcheckShape::new(leaf.equality_point.len(), degree_bound)?,
            &mut channel,
            0,
        )?;
        let range_image_evaluation =
            akita_types::stage1_range_image::<F, E, _>(grinding, level, stage, E::zero())?;
        let expected_output = self
            .plan
            .evaluate_leaf_polynomial(&leaf.polynomial_coefficients, range_image_evaluation);
        if replay.output_claim != expected_output {
            return Err(AkitaError::InvalidProof);
        }
        Ok(Stage1VerifyOutput {
            point: replay.challenges,
            range_image_evaluation,
            physical_l2_virtual_evaluations: None,
        })
    }
}

pub(crate) struct Stage1Replay<'a, E: Field> {
    pub(crate) batching_coeff: E,
    pub(crate) compression: Stage2CompressionOracle<'a, E>,
    pub(crate) range_image_evaluation: E,
    pub(crate) stage1_point: Vec<E>,
    pub(crate) physical_l2_claim: E,
    pub(crate) physical_l2_families: Vec<EqPairTensorFamily<E>>,
}

pub(crate) fn verify_stage1<'a, F, E>(
    rs: &'a RingSwitchVerifyOutput<E>,
    lp: &CommittedGroupParams,
    relation_plan: &RelationRangeImagePlan,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    level: u32,
    layout: &akita_types::NonterminalLevelLayout,
) -> Result<Stage1Replay<'a, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: ExtField<F> + FpExtEncoding<F> + Ring + AkitaSerialize,
{
    let num_rounds = rs.relation_address_geometry.relation_point_variable_count();
    if rs.tau0.len() != num_rounds {
        return Err(AkitaError::InvalidSize {
            expected: num_rounds,
            actual: rs.tau0.len(),
        });
    }
    let digit_range_equality_col_bits = rs
        .tau0
        .len()
        .checked_sub(rs.digit_range_equality_low_variable_count)
        .ok_or(AkitaError::InvalidProof)?;
    let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
        &rs.tau0,
        digit_range_equality_col_bits,
        rs.digit_range_equality_low_variable_count,
    )?;
    let stage1_verifier = Stage1Verifier::new(equality_point, DigitRangePlan::new(rs.b)?);
    let (stage1_stages, stage1_norm) = DigitRangePlan::new(rs.b)?
        .proof_shapes_for_route(num_rounds, lp.inner().matrix.security_route())?;
    if !stage1_stages.iter().copied().eq(layout.stage1_stages())
        || stage1_norm != layout.stage1_norm()
    {
        return Err(AkitaError::InvalidSetup(
            "native Stage 1 replay disagrees with the level grammar".into(),
        ));
    }
    let physical_plan = PhysicalResponsePlan::new(lp, relation_plan)?;
    let physical = physical_plan
        .as_ref()
        .map(|plan| {
            let InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap, ..
            } = lp.inner().matrix.security_route()
            else {
                return Err(AkitaError::InvalidSetup(
                    "physical response plan exists for a non-L2 route".into(),
                ));
            };
            Ok((
                plan,
                lp.inner().matrix.sis_modulus_profile(),
                response_l2_sq_cap,
            ))
        })
        .transpose()?;
    let replay = stage1_verifier.verify::<F>(grinding, physical, level)?;
    let (physical_l2_claim, physical_l2_families) = match (
        replay.physical_l2_virtual_evaluations,
        physical_plan.as_ref(),
    ) {
        (Some(evaluations), Some(plan)) => {
            let eta = grinding.grinded_ext_challenge::<F, E>(
                akita_types::GrindingSite::L2VirtualBatch { level },
            )?;
            let (claim, batching) = batch_l2_virtual_evaluations(eta, &evaluations);
            (claim, plan.virtualization_families(&batching)?)
        }
        (None, None) => (E::zero(), Vec::new()),
        _ => return Err(AkitaError::InvalidProof),
    };
    let compression = match &rs.compression {
        PreparedStage2Compression::Raw => Stage2CompressionOracle::Raw,
        PreparedStage2Compression::QuotientLift { weights, support } => {
            Stage2CompressionOracle::QuotientLift {
                weights,
                support,
                binary_batching: grinding.grinded_ext_challenge::<F, E>(
                    akita_types::GrindingSite::CompressionBinary { level },
                )?,
            }
        }
        PreparedStage2Compression::ReducedEvaluation { weights, support } => {
            Stage2CompressionOracle::ReducedEvaluation {
                weights,
                support,
                binary_batching: grinding.grinded_ext_challenge::<F, E>(
                    akita_types::GrindingSite::CompressionBinary { level },
                )?,
            }
        }
    };
    let batching_coeff =
        grinding.grinded_ext_challenge::<F, E>(akita_types::GrindingSite::Stage2Batch { level })?;
    Ok(Stage1Replay {
        batching_coeff,
        compression,
        range_image_evaluation: replay.range_image_evaluation,
        stage1_point: replay.point,
        physical_l2_claim,
        physical_l2_families,
    })
}
