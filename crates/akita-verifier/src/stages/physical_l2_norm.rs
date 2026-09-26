//! Verifier replay for the schedule-selected physical response norm.

use akita_algebra::eq_poly::EqPolynomial;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_sumcheck::SumcheckInstanceVerifier;
use akita_types::{
    reconstruct_l2_sq_from_gram, FpExtEncoding, PhysicalL2NormProofShape, PhysicalResponsePlan,
    SisModulusProfileId,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

pub(crate) struct PhysicalL2VerifierReplay<E: Field> {
    pub(crate) point: Vec<E>,
    pub(crate) virtual_evaluations: Vec<E>,
    pub(crate) range_image_evaluation: E,
}

pub(crate) struct NativePhysicalL2RangeClaim<'a, E> {
    pub(crate) equality_point: &'a [E],
    pub(crate) input_claim: E,
    pub(crate) leaf_coefficients: &'a [E],
    pub(crate) range_stage: u32,
}

struct PhysicalL2NormVerifier<'a, E: Field> {
    plan: &'a PhysicalResponsePlan,
    virtual_evaluations: &'a [E],
    range_equality_point: &'a [E],
    range_leaf_coefficients: &'a [E],
    range_image_evaluation: E,
    subclaim_weights: Vec<E>,
    input_claim: E,
    norm_merge: E,
}

impl<E: Field + Ring> SumcheckInstanceVerifier<E> for PhysicalL2NormVerifier<'_, E> {
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
                        .virtual_evaluations
                        .get(left)
                        .ok_or(AkitaError::InvalidProof)?;
                    let right = *self
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
    F: Field + CanonicalEncoding,
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
    let canonical = first.to_u128_checked().ok_or(AkitaError::InvalidProof)?;
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
    response_l2_sq: u128,
    subclaims: &[E],
    virtual_evaluations: &[E],
    profile: SisModulusProfileId,
    cap: u128,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let modulus = profile.modulus();
    let modulus_minus_one = modulus
        .checked_sub(1)
        .ok_or_else(|| AkitaError::InvalidSetup("L2 modulus profile has an empty field".into()))?;
    if F::from_u128_checked(modulus_minus_one).is_none() || F::from_u128_checked(modulus).is_some()
    {
        return Err(AkitaError::InvalidSetup(
            "L2 modulus profile disagrees with the proof base field".into(),
        ));
    }
    if response_l2_sq > cap {
        return Err(AkitaError::InvalidProof);
    }
    plan.shape()
        .validate_integer_soundness(profile, plan.fold_basis(), plan.fold_digit_count())?;
    match plan.shape() {
        PhysicalL2NormProofShape::Direct { .. } => {
            if !subclaims.is_empty() || virtual_evaluations.len() != 1 || response_l2_sq >= modulus
            {
                return Err(AkitaError::InvalidProof);
            }
        }
        shape @ PhysicalL2NormProofShape::LimbGram { block_len, .. } => {
            let layout = shape.limb_gram_layout()?.ok_or(AkitaError::InvalidProof)?;
            if subclaims.len() != layout.subclaim_count()
                || virtual_evaluations.len() != layout.limb_count()
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
            let integers = subclaims
                .iter()
                .copied()
                .map(|claim| centered_lift::<F, E>(claim, profile))
                .collect::<Result<Vec<_>, _>>()?;
            if integers
                .iter()
                .any(|value| value.unsigned_abs() > claim_abs_bound)
                || reconstruct_l2_sq_from_gram(plan.shape(), plan.fold_basis(), &integers)?
                    != response_l2_sq
            {
                return Err(AkitaError::InvalidProof);
            }
        }
    }
    Ok(())
}

/// Replay a physical-L2 proof directly from the native Spongefish stream.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_physical_l2_norm_native<F, E>(
    plan: &PhysicalResponsePlan,
    range: NativePhysicalL2RangeClaim<'_, E>,
    profile: SisModulusProfileId,
    cap: u128,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    level: u32,
) -> Result<PhysicalL2VerifierReplay<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F> + Ring + AkitaSerialize,
{
    if range.equality_point.len() != plan.domain().num_vars() || range.leaf_coefficients.len() < 3 {
        return Err(AkitaError::InvalidSetup(
            "fused Stage-1 leaf has inconsistent range geometry".into(),
        ));
    }
    let (subclaim_count, virtual_count) = match plan.shape() {
        PhysicalL2NormProofShape::Direct { .. } => (0, 1),
        shape @ PhysicalL2NormProofShape::LimbGram { .. } => {
            let layout = shape.limb_gram_layout()?.ok_or(AkitaError::InvalidProof)?;
            (layout.subclaim_count(), layout.limb_count())
        }
    };
    let prefix = akita_types::native_l2_verifier_prefix::<F, E>(grinding, level, subclaim_count)?;
    let mut subclaim_weights = Vec::new();
    let norm_input_claim = match plan.shape() {
        PhysicalL2NormProofShape::Direct { .. } => E::from_u128(prefix.response_l2_sq),
        PhysicalL2NormProofShape::LimbGram { .. } => {
            let gamma = grinding.grinded_ext_challenge::<F, E>(
                akita_types::GrindingSite::L2SubclaimBatch { level },
            )?;
            let mut power = E::one();
            for _ in 0..prefix.subclaims.len() {
                subclaim_weights.push(power);
                power *= gamma;
            }
            prefix
                .subclaims
                .iter()
                .zip(&subclaim_weights)
                .fold(E::zero(), |sum, (&claim, &weight)| sum + claim * weight)
        }
    };
    let norm_merge =
        grinding.grinded_ext_challenge::<F, E>(akita_types::GrindingSite::L2NormMerge { level })?;
    let input_claim = range.input_claim + norm_merge * norm_input_claim;
    let mut channel = akita_types::NativeGrindingSumcheckVerifier::<F, E>::new(
        grinding,
        akita_types::SumcheckProtocol::PhysicalL2,
        level,
        0,
    );
    let replay = akita_sumcheck::verify_sumcheck_rounds_native::<F, E, _>(
        &mut channel,
        0,
        input_claim,
        akita_sumcheck::NativeSumcheckShape::new(
            plan.domain().num_vars(),
            range.leaf_coefficients.len(),
        )?,
    )?;
    let virtual_evaluations = akita_types::native_l2_verifier_virtual_evaluations::<F, E>(
        grinding,
        level,
        virtual_count,
    )?;
    let range_image_evaluation = akita_types::native_stage1_verifier_range_image::<F, E>(
        grinding,
        level,
        range.range_stage,
    )?;
    validate_integer_claim::<F, E>(
        plan,
        prefix.response_l2_sq,
        &prefix.subclaims,
        &virtual_evaluations,
        profile,
        cap,
    )?;
    let verifier = PhysicalL2NormVerifier {
        plan,
        virtual_evaluations: &virtual_evaluations,
        range_equality_point: range.equality_point,
        range_leaf_coefficients: range.leaf_coefficients,
        range_image_evaluation,
        subclaim_weights,
        input_claim,
        norm_merge,
    };
    if replay.output_claim != verifier.expected_output_claim(&replay.challenges)? {
        return Err(AkitaError::InvalidProof);
    }
    Ok(PhysicalL2VerifierReplay {
        point: replay.challenges,
        virtual_evaluations,
        range_image_evaluation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Prime32Offset99;

    #[test]
    fn centered_lift_accepts_both_boundary_representatives() {
        type F = Prime32Offset99;
        let profile = SisModulusProfileId::Q32Offset99;
        let modulus = profile.modulus();
        let half = modulus / 2;
        let positive = F::from_u128_checked(half).expect("positive boundary");
        let negative = F::from_u128_checked(half + 1).expect("negative boundary");

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
