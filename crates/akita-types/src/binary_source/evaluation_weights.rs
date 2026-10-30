//! Direct public coefficient weights for the two binary terminal identities.

use akita_algebra::ring::{shifted_remainder_evaluations, PlusTrinomial};
use akita_challenges::{Challenges, FoldChallengeFamily};
use akita_error::{checked, AkitaError};
use akita_params::{RelationPolynomial, RelationRowGeometry};
use jolt_field::Field;

use super::{
    binary_scalar_lift, BinaryEvaluationBounds, BinarySourceDescriptor, BinaryTerminalClaim,
    BINARY_PARITY_CARRY_DIGITS, BINARY_VALUE_CARRY_DIGITS,
};
use crate::native_root_relation::NativeRootWeights;
use crate::root_repacking::RootCoefficientRepacking;

/// Addresses of the response low digit, authenticated U view and both carries.
#[derive(Clone, Copy, Debug)]
pub struct BinaryEvaluationSegments {
    pub response: usize,
    pub partials: usize,
    pub parity_carry: usize,
    pub value_carry: usize,
}

impl BinaryTerminalClaim {
    /// Compile parity and value identities into the fused Stage-2 public oracle.
    ///
    /// The response reads its low balanced digit only, undoing signed packing.
    /// U and both carries read the same outgoing table as the ordinary suffix.
    /// Two independently sampled batching weights combine the identities after
    /// U, K and K_val have been fixed by their commitments.
    #[allow(clippy::too_many_arguments)]
    pub fn relation_weights<E: Field>(
        &self,
        descriptor: &BinarySourceDescriptor,
        repacking: &RootCoefficientRepacking,
        segments: BinaryEvaluationSegments,
        challenges: &Challenges,
        alpha: E,
        batches: [E; 2],
        proof_modulus: u128,
    ) -> Result<NativeRootWeights<E>, AkitaError> {
        let (row_weights, column_weights) = self.equality_weights(descriptor)?;
        let columns = descriptor.index_map().columns();
        let scalar_rows = row_weights.len();
        BinaryEvaluationBounds::new(scalar_rows, columns, proof_modulus)?;
        let ring_rows = checked::exact_div(scalar_rows, 4).ok_or(AkitaError::InvalidProof)?;
        let owner = descriptor
            .polynomials()
            .iter()
            .position(|id| id == self.polynomial())
            .ok_or(AkitaError::InvalidProof)?;
        let polynomials = descriptor.polynomials().len();
        let expected_response =
            checked::product([polynomials, ring_rows]).ok_or(AkitaError::InvalidProof)?;
        let expected_partials =
            checked::product([polynomials, columns]).ok_or(AkitaError::InvalidProof)?;
        let response_shape = repacking
            .segments()
            .get(segments.response)
            .ok_or(AkitaError::InvalidProof)?;
        if response_shape.elements != expected_response
            || response_shape.digits != 4
            || response_shape.geometry.polynomial() != RelationPolynomial::minus_trinomial(648)?
            || response_shape.geometry.coordinate_plane_count() != 1
        {
            return Err(AkitaError::InvalidProof);
        }
        for (segment, elements, digits) in [
            (segments.partials, expected_partials, 1),
            (
                segments.parity_carry,
                polynomials,
                BINARY_PARITY_CARRY_DIGITS,
            ),
            (segments.value_carry, polynomials, BINARY_VALUE_CARRY_DIGITS),
        ] {
            let shape = repacking
                .segments()
                .get(segment)
                .ok_or(AkitaError::InvalidProof)?;
            if shape.elements != elements
                || shape.digits != digits
                || shape.geometry.polynomial() != RelationPolynomial::plus_trinomial(162)?
                || shape.geometry.coordinate_plane_count() != 1
            {
                return Err(AkitaError::InvalidProof);
            }
        }
        let scalar_challenges = binary_scalar_challenges(descriptor, challenges)?;
        let mut terms = Vec::new();
        for (row, weight) in row_weights.into_iter().enumerate() {
            let multiplier = binary_scalar_lift(weight).map(E::from_i128);
            let mut kernel = [E::zero(); 162];
            shifted_remainder_evaluations::<E, PlusTrinomial>(&multiplier, alpha, &mut kernel)
                .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            let element =
                checked::mul_add(owner, ring_rows, row / 4).ok_or(AkitaError::InvalidProof)?;
            for (coefficient, weight) in kernel.into_iter().enumerate() {
                let packed =
                    checked::mul_add(4, coefficient, row % 4).ok_or(AkitaError::InvalidProof)?;
                let address =
                    repacking.coefficient_address(segments.response, element, 0, packed)?;
                let sign = if coefficient % 2 == 0 {
                    E::one()
                } else {
                    -E::one()
                };
                terms.push((address, batches[0] * sign * weight));
            }
        }
        for (column, weight) in column_weights.into_iter().enumerate() {
            let challenge_index =
                checked::mul_add(owner, columns, column).ok_or(AkitaError::InvalidProof)?;
            let challenge = scalar_challenges
                .get(challenge_index)
                .ok_or(AkitaError::InvalidProof)?;
            let multiplier = binary_scalar_lift(weight).map(E::from_i128);
            let mut value_kernel = [E::zero(); 162];
            let mut parity_kernel = [E::zero(); 162];
            shifted_remainder_evaluations::<E, PlusTrinomial>(
                &multiplier,
                alpha,
                &mut value_kernel,
            )
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            shifted_remainder_evaluations::<E, PlusTrinomial>(
                &challenge.map(E::from_i128),
                alpha,
                &mut parity_kernel,
            )
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            for (coefficient, (value, parity)) in
                value_kernel.into_iter().zip(parity_kernel).enumerate()
            {
                terms.extend(repacking.coefficient_weights(
                    segments.partials,
                    challenge_index,
                    coefficient,
                    batches[1] * value - batches[0] * parity,
                )?);
            }
        }
        let mut power = E::one();
        for coefficient in 0..162 {
            for (segment, batch) in [
                (segments.parity_carry, batches[0]),
                (segments.value_carry, batches[1]),
            ] {
                terms.extend(repacking.coefficient_weights(
                    segment,
                    owner,
                    coefficient,
                    -E::from_u64(2) * batch * power,
                )?);
            }
            power *= alpha;
        }
        let value = akita_algebra::binary::BinaryField162::from_bytes(self.value())
            .ok_or(AkitaError::InvalidProof)?;
        let target = binary_scalar_lift(value)
            .into_iter()
            .rev()
            .fold(E::zero(), |acc, coefficient| {
                acc * alpha + E::from_i128(coefficient)
            });
        NativeRootWeights::new(repacking.domain_len(), terms, batches[1] * target)
    }
}

/// Validate and lift the replayed bounded-weight binary fold challenges.
pub fn binary_scalar_challenges(
    descriptor: &BinarySourceDescriptor,
    challenges: &Challenges,
) -> Result<Vec<[i128; 162]>, AkitaError> {
    if challenges.num_claims() != descriptor.polynomials().len()
        || challenges.num_live_blocks_per_claim() != descriptor.index_map().columns()
    {
        return Err(AkitaError::InvalidProof);
    }
    challenges
        .as_slice()
        .iter()
        .map(|challenge| {
            challenge.validate_dyn(162)?;
            if challenge.positions.len() > FoldChallengeFamily::BinaryRoot.weight()
                || challenge
                    .coeffs
                    .iter()
                    .any(|&coefficient| coefficient != -1 && coefficient != 1)
            {
                return Err(AkitaError::InvalidProof);
            }
            let mut scalar = [0; 162];
            for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                *scalar
                    .get_mut(position as usize)
                    .ok_or(AkitaError::InvalidProof)? = i128::from(coefficient);
            }
            Ok(scalar)
        })
        .collect()
}

/// Standard scalar-ring geometry for the bounded U and carry segments.
pub fn binary_scalar_geometry() -> Result<RelationRowGeometry, AkitaError> {
    RelationRowGeometry::for_polynomial(RelationPolynomial::plus_trinomial(162)?, 1, 256)
}
