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
            PhysicalL2NormProofShape::LimbGram {
                physical_response_len,
                block_len,
                limb_count,
            } => {
                let equality = EqPolynomial::evals_prefix(point, physical_response_len)?;
                let pair_count = limb_count
                    .checked_mul(limb_count.checked_add(1).ok_or_else(|| {
                        AkitaError::InvalidSetup("L2 limb-pair count overflow".into())
                    })?)
                    .and_then(|value| value.checked_div(2))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("L2 limb-pair count overflow".into())
                    })?;
                let mut pair_selectors = vec![E::zero(); pair_count];
                for (physical_index, equality_weight) in equality.into_iter().enumerate() {
                    let block = physical_index / block_len;
                    for (pair, selector) in pair_selectors.iter_mut().enumerate() {
                        let index = block
                            .checked_mul(pair_count)
                            .and_then(|base| base.checked_add(pair))
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("L2 selector index overflow".into())
                            })?;
                        let weight = self
                            .subclaim_weights
                            .get(index)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)?;
                        *selector += weight * equality_weight;
                    }
                }
                let mut sum = E::zero();
                let mut pair = 0usize;
                for left in 0..limb_count {
                    for right in left..limb_count {
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
                        let selector = pair_selectors
                            .get(pair)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)?;
                        sum += selector * left * right;
                        pair += 1;
                    }
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
    let coordinates = value.to_ext_coords();
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
        PhysicalL2NormProofShape::LimbGram {
            block_len,
            limb_count,
            ..
        } => {
            if proof.subclaims.len()
                != plan
                    .shape()
                    .subclaim_count()
                    .ok_or_else(|| AkitaError::InvalidSetup("L2 subclaim count overflow".into()))?
                || proof.virtual_evaluations.len() != limb_count
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
    use akita_field::Prime32Offset99;

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
