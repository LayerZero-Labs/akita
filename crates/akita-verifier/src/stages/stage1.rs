//! Stage-1 verifier instances for Akita range-check proofs.
//!
//! This module owns verifier-side replay for both the compact single-stage
//! `b <= 8` path and the staged range-check tree used for larger bases. The
//! prover-side compact witness scans and two-round-prefix kernels stay in the
//! prover/root path.

use akita_challenges::NativeVerifierFoldDraw;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::{
    draw_group_fold_challenges, CommittedGroupParams, DigitRangeEqualityPoint, DigitRangePlan,
    GroupFoldChallenges, OpeningClaimsLayout,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};
pub(crate) struct NativeStage1VerifyOutput<E: Field> {
    pub(crate) point: Vec<E>,
    pub(crate) range_image_evaluation: E,
    pub(crate) physical_l2_virtual_evaluations: Option<Vec<E>>,
}

pub(crate) struct RangeLeafVerifierInput<E: Field> {
    pub(crate) equality_point: Vec<E>,
    pub(crate) input_claim: E,
    pub(crate) polynomial_coefficients: Vec<E>,
}

/// Native Spongefish replay of all sparse fold roots for one recursive level.
pub(crate) fn derive_multi_group_stage1_challenges_native<F, E>(
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    level: u32,
    opening_batch: &OpeningClaimsLayout,
    lp: &CommittedGroupParams,
) -> Result<Vec<GroupFoldChallenges>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    E: ExtField<F>,
{
    let mut group_challenges = Vec::with_capacity(opening_batch.num_groups());
    for group_index in 0..opening_batch.num_groups() {
        let group_lp = lp.group_params_geometry(opening_batch, group_index)?;
        let k_g = opening_batch.group_layout(group_index)?.num_polynomials();
        let group = u32::try_from(group_index).map_err(|_| AkitaError::InvalidProof)?;
        let drawn = {
            let mut live = NativeVerifierFoldDraw::new(grinding.state_mut(), level, group);
            draw_group_fold_challenges::<F, E, _>(&mut live, &group_lp, group_index, k_g)?
        };
        let coordinate_count = group_lp
            .num_live_blocks()
            .checked_mul(k_g)
            .ok_or(AkitaError::InvalidProof)?;
        grinding.record_fold_challenges(level, group, coordinate_count)?;
        group_challenges.push(drawn);
    }
    Ok(group_challenges)
}

/// Stage-1 range-check verifier, including the root/leaf tree choreography.
pub struct AkitaStage1Verifier<E: Field> {
    equality_point: DigitRangeEqualityPoint<E>,
    plan: DigitRangePlan,
}

impl<E: Field> AkitaStage1Verifier<E> {
    /// Construct the stage-1 verifier from a checked range topology.
    pub fn new(equality_point: DigitRangeEqualityPoint<E>, plan: DigitRangePlan) -> Self {
        Self {
            equality_point,
            plan,
        }
    }
}

impl<E: Field + Ring + AkitaSerialize> AkitaStage1Verifier<E> {
    fn verify_product_prefix_native<F>(
        &self,
        grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
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
            let mut channel = akita_types::NativeGrindingSumcheckVerifier::<F, E>::new(
                grinding,
                akita_types::SumcheckProtocol::Stage1,
                level,
                stage,
            );
            let replay = akita_sumcheck::verify_eq_factored_sumcheck_rounds_native::<F, E, _>(
                &current_equality_point,
                current_claim,
                akita_sumcheck::NativeSumcheckShape::new(rounds, arity)?,
                &mut channel,
                0,
            )?;
            let child_claims = akita_types::native_stage1_verifier_child_claims::<F, E>(
                grinding,
                level,
                stage,
                expected.child_claims,
            )?;
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
    pub(crate) fn verify_native<F>(
        &self,
        grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
        physical_l2: Option<(
            &akita_types::PhysicalResponsePlan,
            akita_types::SisModulusProfileId,
            u128,
        )>,
        level: u32,
    ) -> Result<NativeStage1VerifyOutput<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + akita_types::FpExtEncoding<F>,
    {
        let leaf = self.verify_product_prefix_native::<F>(grinding, level)?;
        let stage = u32::try_from(self.plan.product_stage_arities().len())
            .map_err(|_| AkitaError::InvalidProof)?;
        if let Some((plan, profile, cap)) = physical_l2 {
            let replay = super::physical_l2_norm::verify_physical_l2_norm_native::<F, E>(
                plan,
                super::physical_l2_norm::NativePhysicalL2RangeClaim {
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
            return Ok(NativeStage1VerifyOutput {
                point: replay.point,
                range_image_evaluation: replay.range_image_evaluation,
                physical_l2_virtual_evaluations: Some(replay.virtual_evaluations),
            });
        }
        let degree_bound = leaf.polynomial_coefficients.len().saturating_sub(1);
        let mut channel = akita_types::NativeGrindingSumcheckVerifier::<F, E>::new(
            grinding,
            akita_types::SumcheckProtocol::Stage1,
            level,
            stage,
        );
        let replay = akita_sumcheck::verify_eq_factored_sumcheck_rounds_native::<F, E, _>(
            &leaf.equality_point,
            leaf.input_claim,
            akita_sumcheck::NativeSumcheckShape::new(leaf.equality_point.len(), degree_bound)?,
            &mut channel,
            0,
        )?;
        let range_image_evaluation =
            akita_types::native_stage1_verifier_range_image::<F, E>(grinding, level, stage)?;
        let expected_output = self
            .plan
            .evaluate_leaf_polynomial(&leaf.polynomial_coefficients, range_image_evaluation);
        if replay.output_claim != expected_output {
            return Err(AkitaError::InvalidProof);
        }
        Ok(NativeStage1VerifyOutput {
            point: replay.challenges,
            range_image_evaluation,
            physical_l2_virtual_evaluations: None,
        })
    }
}
