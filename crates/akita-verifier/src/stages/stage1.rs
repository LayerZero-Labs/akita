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
use akita_types::RangeLeaf;
use akita_types::VerifierGrinding;
use akita_types::{
    batch_l2_virtual_evaluations, FpExtEncoding, PhysicalResponsePlan, RelationRangeImagePlan,
};
use jolt_algebra::{CanonicalEncoding, ExtField, Field, Ring};
/// A replayed Stage 1: its point, the range image there, and the L2 route's
/// virtual evaluations.
pub struct Stage1VerifyOutput<E: Field, V = E> {
    pub point: Vec<E>,
    pub range_image_evaluation: V,
    pub physical_l2_virtual_evaluations: Option<Vec<E>>,
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
    /// The product stages, which only the clear verifier replays: a masked
    /// level's basis has none (`specs/whiteout-akita.md` §4.4).
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

    /// Replay a stage-1 proof in the clear, on the physical L2 route when
    /// `physical_l2` is given and the L∞ route otherwise.
    pub fn verify_clear<F>(
        &self,
        grinding: &mut VerifierGrinding<'_, '_>,
        physical_l2: Option<(
            &akita_types::PhysicalResponsePlan,
            akita_params::SisModulusProfileId,
            u128,
        )>,
        level: u32,
    ) -> Result<Stage1VerifyOutput<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + akita_types::FpExtEncoding<F> + ChallengeCodec,
    {
        self.verify::<F, _>(grinding, physical_l2, level)
    }

    /// Replay the stage-1 proof directly from Spongefish.
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
        grinding.check_range_leaf(
            RangeLeaf::new(self.plan, &leaf.polynomial_coefficients)?,
            range_image_evaluation.clone(),
            output_claim,
        )?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use akita_params::{
        ChallengeFieldOrder, GrindingPlan, GrindingRun, GrindingSite, SumcheckProtocol,
    };
    use akita_transcript::{
        exchange_extension_group, new_prover_channel, new_verifier_channel, public_extensions,
    };
    use akita_types::{grinding_sumcheck_body_site, GrindingReplay, ProverGrinding};
    use jolt_algebra::{FpExt4, Prime32Offset99, Zero};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    /// A one-round basis-8 leaf proof whose round polynomial is zero, so the
    /// leaf's output claim is zero, followed by `range_image`.
    fn verify_leaf(range_image: E) -> Result<Stage1VerifyOutput<E>, AkitaError> {
        let level = 1;
        let order = ChallengeFieldOrder::from_full_capacity(128).unwrap();
        let round_site = GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::Stage1,
            level,
            stage: 0,
            round: 0,
        };
        let plan = GrindingPlan::new(
            vec![GrindingRun::proof_of_work(round_site, 1, order).unwrap()],
            order,
        )
        .unwrap();
        let mut transcript = new_prover_channel(b"leaf");
        let mut prover = ProverGrinding::new(&mut transcript, &plan);
        let claim_site = akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_SUMCHECK,
            invocation: SumcheckProtocol::Stage1.tag(),
            level,
            detail: akita_sumcheck::SumcheckRole::Claim as u32,
            ..akita_transcript::ProtocolSiteId::default()
        };
        public_extensions::<F, E, _>(prover.state_mut(), claim_site, &[E::zero()]).unwrap();
        exchange_extension_group::<F, E, _>(
            prover.state_mut(),
            grinding_sumcheck_body_site(SumcheckProtocol::Stage1, level, 0, 0, 0),
            &mut [E::zero(); 4],
        )
        .unwrap();
        prover.grinded_ext_challenge::<E>(round_site).unwrap();
        akita_types::stage1_range_image::<F, E, _>(&mut prover, level, 0, range_image).unwrap();
        prover.finish().unwrap();
        let proof = transcript.narg_string().to_vec();

        let mut transcript = new_verifier_channel(b"leaf", &proof);
        let mut verifier = VerifierGrinding::new(&mut transcript, &plan);
        let equality_point =
            DigitRangeEqualityPoint::from_column_then_ring_challenges(&[E::from_u64(5)], 1, 0)
                .unwrap();
        Stage1Verifier::new(equality_point, DigitRangePlan::new(8).unwrap()).verify_clear::<F>(
            &mut verifier,
            None,
            level,
        )
    }

    #[test]
    fn leaf_check_accepts_exactly_the_range_roots() {
        // The basis-8 leaf is Π_{k<4} (X − k(k+1)), so it vanishes exactly at
        // the range images 0, 2, 6 and 12.
        for root in [0, 2, 6, 12] {
            let output = verify_leaf(E::from_u64(root)).unwrap();
            assert_eq!(output.range_image_evaluation, E::from_u64(root));
        }
        for non_root in [1, 3, 7, 13] {
            assert!(matches!(
                verify_leaf(E::from_u64(non_root)),
                Err(AkitaError::InvalidProof)
            ));
        }
    }
}
