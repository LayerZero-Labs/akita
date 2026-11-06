//! Stage-1 verifier instances for Akita range-check proofs.
//!
//! This module owns verifier-side replay for both the compact single-stage
//! `b <= 8` path and the staged range-check tree used for larger bases. The
//! prover-side compact witness scans and two-round-prefix kernels stay in the
//! prover/root path.

use crate::claims::{ClaimReplay, SumcheckFormat, SumcheckInstance};
use crate::stages::ring_switch::{PreparedStage2Compression, RingSwitchVerifyOutput};
use crate::stages::stage2::Stage2CompressionOracle;
use akita_algebra::offset_eq::EqPairTensorFamily;
use akita_error::AkitaError;
use akita_params::InnerCommitSecurityRoute;
use akita_params::{CommittedGroupParams, DigitRangePlan};
use akita_serialization::AkitaSerialize;
use akita_transcript::ChallengeCodec;
use akita_types::DigitRangeEqualityPoint;
use akita_types::{
    batch_l2_virtual_evaluations, FpExtEncoding, PhysicalResponsePlan, RelationRangeImagePlan,
};
use jolt_algebra::{CanonicalEncoding, ExtField, Field, Ring};
pub(crate) struct Stage1VerifyOutput<E: Field, V = E> {
    pub(crate) point: Vec<E>,
    pub(crate) range_image_evaluation: V,
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
    /// The product stages, which only the clear verifier replays.
    fn verify_product_prefix<F, G>(
        &self,
        grinding: &mut G,
        level: u32,
    ) -> Result<RangeLeafVerifierInput<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + ChallengeCodec,
        G: ClaimReplay<F, E>,
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
            let (challenges, output_claim) = grinding.single_sumcheck(
                akita_params::SumcheckProtocol::Stage1,
                level,
                stage,
                SumcheckInstance {
                    invocation: 0,
                    rounds,
                    degree: arity,
                    format: SumcheckFormat::EqFactored {
                        equality_point: std::mem::take(&mut current_equality_point),
                    },
                },
                G::clear_path(current_claim)?,
            )?;
            let output_claim = G::clear_values(&[output_claim])?
                .pop()
                .ok_or(AkitaError::InvalidProof)?;
            let mut child_claims = akita_transcript::extension_slots::<E, _>(
                grinding.state_mut(),
                expected.child_claims,
            )?;
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
            if output_claim != expected_output {
                return Err(AkitaError::InvalidProof);
            }
            let gamma = grinding.grinded_ext_challenge::<E>(
                akita_params::GrindingSite::Stage1InterstageBatch { level, stage },
            )?;
            current_weights = self
                .plan
                .interstage_batch_weights(gamma, child_claims.len());
            current_claim = self.plan.batch_claims(&current_weights, &child_claims)?;
            current_equality_point = challenges;
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
    pub(crate) fn verify<F, G>(
        &self,
        grinding: &mut G,
        physical_l2: Option<(
            &akita_types::PhysicalResponsePlan,
            akita_params::SisModulusProfileId,
            u128,
        )>,
        level: u32,
    ) -> Result<Stage1VerifyOutput<E, G::Claim>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + akita_types::FpExtEncoding<F> + ChallengeCodec,
        G: ClaimReplay<F, E>,
    {
        let leaf = self.verify_product_prefix::<F, _>(grinding, level)?;
        let stage = u32::try_from(self.plan.product_stage_arities().len())
            .map_err(|_| AkitaError::InvalidProof)?;
        if let Some((plan, profile, cap)) = physical_l2 {
            let replay = super::physical_l2_norm::verify_physical_l2_norm::<F, E, _>(
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
                range_image_evaluation: G::clear_path(replay.range_image_evaluation)?,
                physical_l2_virtual_evaluations: Some(replay.virtual_evaluations),
            });
        }
        let input = if self.plan.product_stage_arities().is_empty() {
            G::constant(leaf.input_claim)
        } else {
            G::clear_path(leaf.input_claim)?
        };
        let degree_bound = leaf.polynomial_coefficients.len().saturating_sub(1);
        let (point, output_claim) = grinding.single_sumcheck(
            akita_params::SumcheckProtocol::Stage1,
            level,
            stage,
            SumcheckInstance {
                invocation: 0,
                rounds: leaf.equality_point.len(),
                degree: degree_bound,
                format: SumcheckFormat::EqFactored {
                    equality_point: leaf.equality_point,
                },
            },
            input,
        )?;
        let range_image_evaluation = grinding.range_image(level, stage)?;
        let [range_image, output_claim] = <[E; 2]>::try_from(G::clear_values(&[
            range_image_evaluation.clone(),
            output_claim,
        ])?)
        .map_err(|_| AkitaError::InvalidProof)?;
        let expected_output = self
            .plan
            .evaluate_leaf_polynomial(&leaf.polynomial_coefficients, range_image);
        if output_claim != expected_output {
            return Err(AkitaError::InvalidProof);
        }
        Ok(Stage1VerifyOutput {
            point,
            range_image_evaluation,
            physical_l2_virtual_evaluations: None,
        })
    }
}

pub(crate) struct Stage1Replay<'a, E: Field, V = E> {
    pub(crate) batching_coeff: E,
    pub(crate) compression: Stage2CompressionOracle<'a, E>,
    pub(crate) range_image_evaluation: V,
    pub(crate) stage1_point: Vec<E>,
    pub(crate) physical_l2_claim: E,
    pub(crate) physical_l2_families: Vec<EqPairTensorFamily<E>>,
}

pub(crate) fn verify_stage1<'a, F, E, G>(
    rs: &'a RingSwitchVerifyOutput<E>,
    lp: &CommittedGroupParams,
    relation_plan: &RelationRangeImagePlan,
    grinding: &mut G,
    level: u32,
    layout: &akita_params::NonterminalLevelLayout,
) -> Result<Stage1Replay<'a, E, G::Claim>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: ExtField<F> + FpExtEncoding<F> + Ring + AkitaSerialize + ChallengeCodec,
    G: ClaimReplay<F, E>,
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
    let replay = stage1_verifier.verify::<F, _>(grinding, physical, level)?;
    let (physical_l2_claim, physical_l2_families) = match (
        replay.physical_l2_virtual_evaluations,
        physical_plan.as_ref(),
    ) {
        (Some(evaluations), Some(plan)) => {
            let eta = grinding
                .grinded_ext_challenge::<E>(akita_params::GrindingSite::L2VirtualBatch { level })?;
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
                binary_batching: grinding.grinded_ext_challenge::<E>(
                    akita_params::GrindingSite::CompressionBinary { level },
                )?,
            }
        }
        PreparedStage2Compression::ReducedEvaluation { weights, support } => {
            Stage2CompressionOracle::ReducedEvaluation {
                weights,
                support,
                binary_batching: grinding.grinded_ext_challenge::<E>(
                    akita_params::GrindingSite::CompressionBinary { level },
                )?,
            }
        }
    };
    let batching_coeff =
        grinding.grinded_ext_challenge::<E>(akita_params::GrindingSite::Stage2Batch { level })?;
    Ok(Stage1Replay {
        batching_coeff,
        compression,
        range_image_evaluation: replay.range_image_evaluation,
        stage1_point: replay.point,
        physical_l2_claim,
        physical_l2_families,
    })
}
