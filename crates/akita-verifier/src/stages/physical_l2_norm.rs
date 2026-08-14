//! Verifier replay for the schedule-selected physical response norm.

use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{SumcheckInstanceVerifier, SumcheckInstanceVerifierExt};
use akita_transcript::labels::{
    ABSORB_L2_NORM_INTEGER, ABSORB_L2_NORM_SUBCLAIM, ABSORB_L2_VIRTUAL_EVALUATION,
    CHALLENGE_L2_NORM_BATCH, CHALLENGE_L2_NORM_MERGE, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    reconstruct_l2_sq_from_gram, FpExtEncoding, PhysicalL2NormProof, PhysicalL2NormProofShape,
    PhysicalResponsePlan, SisModulusProfileId,
};

pub(crate) struct PhysicalL2VerifierReplay<E: FieldCore> {
    pub(crate) point: Vec<E>,
    pub(crate) virtual_evaluations: Vec<E>,
}

pub(crate) struct PhysicalL2RangeClaim<'a, E> {
    pub(crate) equality_point: &'a [E],
    pub(crate) input_claim: E,
    pub(crate) leaf_coefficients: &'a [E],
    pub(crate) image_evaluation: E,
}

struct PhysicalL2NormVerifier<'a, E: FieldCore> {
    plan: &'a PhysicalResponsePlan,
    proof: &'a PhysicalL2NormProof<E>,
    range_equality_point: &'a [E],
    range_leaf_coefficients: &'a [E],
    range_image_evaluation: E,
    subclaim_weights: Vec<E>,
    input_claim: E,
    norm_merge: E,
}

impl<E: FieldCore + FromPrimitiveInt> SumcheckInstanceVerifier<E>
    for PhysicalL2NormVerifier<'_, E>
{
    fn num_rounds(&self) -> usize {
        self.plan.domain().num_vars()
    }

    fn degree_bound(&self) -> usize {
        self.range_leaf_coefficients.len()
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn expected_output_claim(&self, point: &[E]) -> Result<E, AkitaError> {
        let range_equality = EqPolynomial::mle(self.range_equality_point, point)?;
        let range_leaf = self
            .range_leaf_coefficients
            .iter()
            .rev()
            .fold(E::zero(), |acc, &coefficient| {
                acc * self.range_image_evaluation + coefficient
            });
        let norm = match self.plan.shape() {
            PhysicalL2NormProofShape::Direct { .. } => {
                let value = self
                    .proof
                    .virtual_evaluations
                    .first()
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?;
                value * value
            }
            shape @ PhysicalL2NormProofShape::LimbGram { .. } => {
                let layout = shape.limb_gram_layout()?.ok_or(AkitaError::InvalidProof)?;
                let mut pair_selectors = vec![E::zero(); layout.pair_count()];
                let mut block_start_sum = E::zero();
                for (block_index, block_range) in layout.block_ranges().enumerate() {
                    let block_end_sum = EqPolynomial::prefix_sum(point, block_range.end)?;
                    let block_weight = block_end_sum - block_start_sum;
                    for ((left, right), selector) in
                        layout.limb_pairs().zip(pair_selectors.iter_mut())
                    {
                        let index =
                            layout
                                .subclaim_index(block_index, left, right)
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup("L2 selector index overflow".into())
                                })?;
                        let weight = self
                            .subclaim_weights
                            .get(index)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)?;
                        *selector += weight * block_weight;
                    }
                    block_start_sum = block_end_sum;
                }
                let mut sum = E::zero();
                for ((left, right), selector) in layout.limb_pairs().zip(pair_selectors) {
                    let left = *self
                        .proof
                        .virtual_evaluations
                        .get(left)
                        .ok_or(AkitaError::InvalidProof)?;
                    let right = *self
                        .proof
                        .virtual_evaluations
                        .get(right)
                        .ok_or(AkitaError::InvalidProof)?;
                    sum += selector * left * right;
                }
                sum
            }
        };
        Ok(range_equality * range_leaf + self.norm_merge * norm)
    }
}

fn centered_lift<F, E>(value: E, profile: SisModulusProfileId) -> Result<i128, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let coordinates = value.ext_coords();
    let Some((&first, tail)) = coordinates.split_first() else {
        return Err(AkitaError::InvalidProof);
    };
    if tail.iter().any(|coordinate| !coordinate.is_zero()) {
        return Err(AkitaError::InvalidProof);
    }
    let modulus = profile.modulus();
    if modulus > i128::MAX as u128 {
        return Err(AkitaError::InvalidSetup(
            "centered limb lifting is only defined for small fields".into(),
        ));
    }
    let canonical = first.to_canonical_u128();
    if canonical <= modulus / 2 {
        i128::try_from(canonical).map_err(|_| AkitaError::InvalidProof)
    } else {
        let magnitude = modulus - canonical;
        i128::try_from(magnitude)
            .map(|value| -value)
            .map_err(|_| AkitaError::InvalidProof)
    }
}

fn validate_integer_claim<F, E>(
    plan: &PhysicalResponsePlan,
    proof: &PhysicalL2NormProof<E>,
    profile: SisModulusProfileId,
    cap: u128,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let modulus = profile.modulus();
    let modulus_minus_one = modulus
        .checked_sub(1)
        .ok_or_else(|| AkitaError::InvalidSetup("L2 modulus profile has an empty field".into()))?;
    if F::from_canonical_u128_checked(modulus_minus_one).is_none()
        || F::from_canonical_u128_checked(modulus).is_some()
    {
        return Err(AkitaError::InvalidSetup(
            "L2 modulus profile disagrees with the proof base field".into(),
        ));
    }
    if proof.response_l2_sq > cap {
        return Err(AkitaError::InvalidProof);
    }
    plan.shape()
        .validate_integer_soundness(profile, plan.fold_basis(), plan.fold_digit_count())?;
    match plan.shape() {
        PhysicalL2NormProofShape::Direct { .. } => {
            if !proof.subclaims.is_empty()
                || proof.virtual_evaluations.len() != 1
                || proof.response_l2_sq >= modulus
            {
                return Err(AkitaError::InvalidProof);
            }
        }
        shape @ PhysicalL2NormProofShape::LimbGram { block_len, .. } => {
            let layout = shape.limb_gram_layout()?.ok_or(AkitaError::InvalidProof)?;
            if proof.subclaims.len() != layout.subclaim_count()
                || proof.virtual_evaluations.len() != layout.limb_count()
            {
                return Err(AkitaError::InvalidProof);
            }
            let digit_abs = (plan.fold_basis() / 2) as u128;
            let claim_abs_bound = (block_len as u128)
                .checked_mul(
                    digit_abs
                        .checked_mul(digit_abs)
                        .ok_or_else(|| AkitaError::InvalidSetup("L2 limb bound overflow".into()))?,
                )
                .ok_or_else(|| AkitaError::InvalidSetup("L2 limb bound overflow".into()))?;
            let integers = proof
                .subclaims
                .iter()
                .copied()
                .map(|claim| centered_lift::<F, E>(claim, profile))
                .collect::<Result<Vec<_>, _>>()?;
            if integers
                .iter()
                .any(|value| value.unsigned_abs() > claim_abs_bound)
                || reconstruct_l2_sq_from_gram(plan.shape(), plan.fold_basis(), &integers)?
                    != proof.response_l2_sq
            {
                return Err(AkitaError::InvalidProof);
            }
        }
    }
    Ok(())
}

pub(crate) fn verify_physical_l2_norm<F, E, T>(
    plan: &PhysicalResponsePlan,
    proof: &PhysicalL2NormProof<E>,
    range: PhysicalL2RangeClaim<'_, E>,
    profile: SisModulusProfileId,
    cap: u128,
    transcript: &mut T,
) -> Result<PhysicalL2VerifierReplay<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FpExtEncoding<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    if range.equality_point.len() != plan.domain().num_vars() || range.leaf_coefficients.len() < 3 {
        return Err(AkitaError::InvalidSetup(
            "fused Stage-1 leaf has inconsistent range geometry".into(),
        ));
    }
    validate_integer_claim::<F, E>(plan, proof, profile, cap)?;
    transcript.append_serde(ABSORB_L2_NORM_INTEGER, &proof.response_l2_sq);
    for claim in &proof.subclaims {
        transcript.append_serde(ABSORB_L2_NORM_SUBCLAIM, claim);
    }
    let mut subclaim_weights = Vec::new();
    let norm_input_claim = match plan.shape() {
        PhysicalL2NormProofShape::Direct { .. } => E::from_u128(proof.response_l2_sq),
        PhysicalL2NormProofShape::LimbGram { .. } => {
            let gamma = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_L2_NORM_BATCH);
            let mut power = E::one();
            for _ in 0..proof.subclaims.len() {
                subclaim_weights.push(power);
                power *= gamma;
            }
            proof
                .subclaims
                .iter()
                .zip(&subclaim_weights)
                .fold(E::zero(), |sum, (&claim, &weight)| sum + claim * weight)
        }
    };
    let norm_merge = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_L2_NORM_MERGE);
    let verifier = PhysicalL2NormVerifier {
        plan,
        proof,
        range_equality_point: range.equality_point,
        range_leaf_coefficients: range.leaf_coefficients,
        range_image_evaluation: range.image_evaluation,
        subclaim_weights,
        input_claim: range.input_claim + norm_merge * norm_input_claim,
        norm_merge,
    };
    let point = verifier.verify::<F, T, _>(&proof.sumcheck, transcript, |tr| {
        sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
    })?;
    for evaluation in &proof.virtual_evaluations {
        transcript.append_serde(ABSORB_L2_VIRTUAL_EVALUATION, evaluation);
    }
    Ok(PhysicalL2VerifierReplay {
        point,
        virtual_evaluations: proof.virtual_evaluations.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::stage1::AkitaStage1Verifier;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::{FpExt4, Prime32Offset99};
    use akita_serialization::{AkitaDeserialize, AkitaSerialize};
    use akita_transcript::AkitaTranscript;
    use akita_types::proof::PhysicalL2NormProofWireShape;
    use akita_types::sis::{
        role_a_collision_l2_sq_for_response_bound, sis_l2_table_key_for_collision_sq,
        DEFAULT_SIS_SECURITY_POLICY,
    };
    use akita_types::{
        AkitaStage1Proof, CommitmentRingDims, CommittedGroupParams, DigitRangeEqualityPoint,
        DigitRangePlan, InnerCommitMatrixParams, OpeningClaimsLayout, RelationAddressGeometry,
        RelationRangeImagePlan, SisL2TableDigest, WitnessLayout,
    };

    type SmallExt = FpExt4<Prime32Offset99>;

    fn limb_gram_fixture() -> (PhysicalResponsePlan, Vec<i8>, u128) {
        const RING_DIMENSION: usize = 64;
        const POSITIONS_PER_BLOCK: usize = 4;
        const LIVE_RING_ELEMENTS: usize = 8;
        const FOLD_DIGITS: usize = 2;
        const RESPONSE_L2_SQ_CAP: u128 = 1 << 20;

        let profile = SisModulusProfileId::Q32Offset99;
        let shape = PhysicalL2NormProofShape::LimbGram {
            physical_response_len: 512,
            block_len: 128,
            limb_count: FOLD_DIGITS,
        };
        let collision_l2_sq = role_a_collision_l2_sq_for_response_bound(1, RESPONSE_L2_SQ_CAP)
            .expect("collision square");
        let table_key = sis_l2_table_key_for_collision_sq(
            DEFAULT_SIS_SECURITY_POLICY,
            SisL2TableDigest::CURRENT,
            profile,
            RING_DIMENSION as u32,
            collision_l2_sq,
        )
        .expect("small-field L2 table row");
        let inner = InnerCommitMatrixParams::try_new_l2_with_min_rank(
            table_key,
            POSITIONS_PER_BLOCK * FOLD_DIGITS,
            RESPONSE_L2_SQ_CAP,
            shape,
        )
        .expect("L2 A matrix");
        let mut params = CommittedGroupParams::params_only(
            profile,
            RING_DIMENSION,
            2,
            inner.output_rank(),
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        );
        params.inner_commit_matrix = inner;
        params = params
            .with_decomp(POSITIONS_PER_BLOCK, LIVE_RING_ELEMENTS, FOLD_DIGITS, 1, 1)
            .expect("complete scalar params");
        params.num_digits_fold = FOLD_DIGITS;

        let opening = OpeningClaimsLayout::new(9, 1).expect("opening layout");
        let relation_witness_geometry =
            akita_types::RelationWitnessGeometry::for_evaluation_trace_execution(&params, &opening)
                .expect("evaluation-trace relation geometry");
        let witness_layout =
            WitnessLayout::new(&params, &opening, &relation_witness_geometry, 1, 1)
                .expect("witness layout");
        let relation_geometry = RelationAddressGeometry::new(
            CommitmentRingDims::uniform(RING_DIMENSION),
            RING_DIMENSION,
            witness_layout.live_coeff_len(),
        )
        .expect("relation geometry");
        let relation_plan = RelationRangeImagePlan::new(
            relation_witness_geometry,
            relation_geometry,
            DigitRangePlan::new(4).expect("digit range"),
            witness_layout,
            &opening,
        )
        .expect("relation plan");
        let physical = PhysicalResponsePlan::new(&params, &relation_plan)
            .expect("physical response plan")
            .expect("L2 route");
        assert_eq!(physical.shape(), shape);
        assert!(
            physical
                .shape()
                .limb_gram_layout()
                .expect("checked limb layout")
                .expect("limb layout")
                .block_count()
                > 1
        );

        let witness = (0..physical.domain().live_len())
            .map(|index| [-1, 0, 1, 0][index % 4])
            .collect();
        (physical, witness, RESPONSE_L2_SQ_CAP)
    }

    fn prove_limb_gram(
        plan: &PhysicalResponsePlan,
        witness: Vec<i8>,
    ) -> (
        AkitaStage1Proof<SmallExt>,
        Vec<SmallExt>,
        DigitRangeEqualityPoint<SmallExt>,
    ) {
        let range_plan = DigitRangePlan::new(16).expect("range plan");
        let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
            &vec![SmallExt::zero(); plan.domain().num_vars()],
            plan.domain().num_vars(),
            0,
        )
        .expect("range equality point");
        let prover = akita_prover::DigitRangeProver::new(
            std::sync::Arc::from(witness),
            range_plan,
            plan.domain(),
            equality_point.clone(),
        )
        .expect("L2 digit-range prover");
        let mut transcript =
            AkitaTranscript::<Prime32Offset99>::new(b"akita/physical-l2-small-field-integration");
        let (proof, point) = prover
            .prove::<Prime32Offset99, _>(&mut transcript, Some(plan))
            .expect("prove multi-block limb Gram norm");
        (proof, point, equality_point)
    }

    fn verify_limb_gram(
        plan: &PhysicalResponsePlan,
        proof: &AkitaStage1Proof<SmallExt>,
        equality_point: DigitRangeEqualityPoint<SmallExt>,
        cap: u128,
    ) -> Result<PhysicalL2VerifierReplay<SmallExt>, AkitaError> {
        let mut transcript =
            AkitaTranscript::<Prime32Offset99>::new(b"akita/physical-l2-small-field-integration");
        let stage1 =
            AkitaStage1Verifier::new(equality_point, DigitRangePlan::new(16).expect("range plan"));
        let leaf =
            stage1.verify_product_prefix::<Prime32Offset99, _>(&proof.stages, &mut transcript)?;
        let norm_proof = proof.norm_proof.as_ref().ok_or(AkitaError::InvalidProof)?;
        verify_physical_l2_norm::<Prime32Offset99, SmallExt, _>(
            plan,
            norm_proof,
            PhysicalL2RangeClaim {
                equality_point: &leaf.equality_point,
                input_claim: leaf.input_claim,
                leaf_coefficients: &leaf.polynomial_coefficients,
                image_evaluation: proof.range_image_evaluation,
            },
            SisModulusProfileId::Q32Offset99,
            cap,
            &mut transcript,
        )
    }

    #[test]
    fn centered_lift_accepts_both_boundary_representatives() {
        type F = Prime32Offset99;
        let profile = SisModulusProfileId::Q32Offset99;
        let modulus = profile.modulus();
        let half = modulus / 2;
        let positive = F::from_canonical_u128_checked(half).expect("positive boundary");
        let negative = F::from_canonical_u128_checked(half + 1).expect("negative boundary");

        assert_eq!(
            centered_lift::<F, F>(positive, profile).unwrap(),
            half as i128
        );
        assert_eq!(
            centered_lift::<F, F>(negative, profile).unwrap(),
            -(half as i128)
        );
    }
}
