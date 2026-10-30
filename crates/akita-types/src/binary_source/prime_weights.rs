//! Public coefficient weights authenticating committed prime evaluation planes.

use akita_algebra::ring::{shifted_remainder_evaluations, PlusTrinomial};
use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use akita_params::{binary_setup::BINARY_SETUP_P, RelationPolynomial};
use jolt_field::Field;

use super::{
    binary_scalar_challenges, BinarySourceDescriptor, PrimePlaneManifest, PRIME_PLANE_DIGITS,
};
use crate::native_root_relation::NativeRootWeights;
use crate::root_repacking::RootCoefficientRepacking;

/// Compile each plane's ring identity and scalar reading into Stage 2.
///
/// Two independent batches per plane are sampled after the outgoing commitment.
/// The first checks `sum e_r V_r - sum c_b EP_b = 0` at `alpha`, with the
/// complete four-digit response and the signed packing undone. The second
/// checks `sum col_b <omega, EP_b> = value`. Plane elements are ordered by
/// unique manifest plane, then column; their coefficients have 43 digits.
#[allow(clippy::too_many_arguments)]
pub fn prime_plane_relation_weights<E: Field>(
    descriptor: &BinarySourceDescriptor,
    manifest: &PrimePlaneManifest,
    repacking: &RootCoefficientRepacking,
    response_segment: usize,
    plane_segment: usize,
    challenges: &Challenges,
    alpha: E,
    batches: &[E],
) -> Result<NativeRootWeights<E>, AkitaError> {
    if E::from_u128(BINARY_SETUP_P) != E::zero()
        || batches.len()
            != checked::product([manifest.planes().len(), 2]).ok_or(AkitaError::InvalidProof)?
    {
        return Err(AkitaError::InvalidProof);
    }
    let columns = descriptor.index_map().columns();
    let scalar_rows = checked::exact_div(descriptor.index_map().padded_carriers(), columns)
        .ok_or(AkitaError::InvalidProof)?;
    let ring_rows = checked::exact_div(scalar_rows, 4).ok_or(AkitaError::InvalidProof)?;
    for (segment, elements, digits, polynomial) in [
        (
            response_segment,
            checked::product([descriptor.polynomials().len(), ring_rows]),
            4,
            RelationPolynomial::minus_trinomial(648)?,
        ),
        (
            plane_segment,
            checked::product([manifest.planes().len(), columns]),
            PRIME_PLANE_DIGITS,
            RelationPolynomial::plus_trinomial(162)?,
        ),
    ] {
        let shape = repacking
            .segments()
            .get(segment)
            .ok_or(AkitaError::InvalidProof)?;
        if Some(shape.elements) != elements
            || shape.digits != digits
            || shape.geometry.polynomial() != polynomial
            || shape.geometry.coordinate_plane_count() != 1
        {
            return Err(AkitaError::InvalidProof);
        }
    }
    let scalar_challenges = binary_scalar_challenges(descriptor, challenges)?;
    let term_count = checked::sum([
        checked::product([scalar_rows, 162, 4]).ok_or(AkitaError::InvalidProof)?,
        checked::product([columns, 162, PRIME_PLANE_DIGITS]).ok_or(AkitaError::InvalidProof)?,
    ])
    .and_then(|count| checked::product([manifest.planes().len(), count]))
    .ok_or(AkitaError::InvalidProof)?;
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(term_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    let mut target = E::zero();
    for (plane_index, claim) in manifest.planes().iter().enumerate() {
        let (rows, column_weights, omega) = claim.weights::<E>(descriptor)?;
        let batch_index = checked::product([plane_index, 2]).ok_or(AkitaError::InvalidProof)?;
        let relation_batch = *batches.get(batch_index).ok_or(AkitaError::InvalidProof)?;
        let reading_batch = *batches
            .get(checked::sum([batch_index, 1]).ok_or(AkitaError::InvalidProof)?)
            .ok_or(AkitaError::InvalidProof)?;
        for (row, row_weight) in rows.into_iter().enumerate() {
            if row_weight == E::zero() {
                continue;
            }
            let element = checked::mul_add(claim.polynomial, ring_rows, row / 4)
                .ok_or(AkitaError::InvalidProof)?;
            let mut power = relation_batch * row_weight;
            for coefficient in 0..162 {
                let packed =
                    checked::mul_add(coefficient, 4, row % 4).ok_or(AkitaError::InvalidProof)?;
                let signed = if coefficient % 2 == 0 { power } else { -power };
                terms.extend(repacking.coefficient_weights(
                    response_segment,
                    element,
                    packed,
                    signed,
                )?);
                power *= alpha;
            }
        }
        for (column, column_weight) in column_weights.into_iter().enumerate() {
            let challenge_index = checked::mul_add(claim.polynomial, columns, column)
                .ok_or(AkitaError::InvalidProof)?;
            let multiplier = scalar_challenges
                .get(challenge_index)
                .ok_or(AkitaError::InvalidProof)?
                .map(E::from_i128);
            let mut kernel = [E::zero(); 162];
            shifted_remainder_evaluations::<E, PlusTrinomial>(&multiplier, alpha, &mut kernel)
                .map_err(|error| AkitaError::Internal(error.to_string()))?;
            let element =
                checked::mul_add(plane_index, columns, column).ok_or(AkitaError::InvalidProof)?;
            for (coefficient, relation) in kernel.into_iter().enumerate() {
                let weight = reading_batch
                    * column_weight
                    * *omega.get(coefficient).ok_or(AkitaError::InvalidProof)?
                    - relation_batch * relation;
                terms.extend(repacking.coefficient_weights(
                    plane_segment,
                    element,
                    coefficient,
                    weight,
                )?);
            }
        }
        target += reading_batch * E::from_u128(claim.value);
    }
    NativeRootWeights::new(repacking.domain_len(), terms, target)
}
