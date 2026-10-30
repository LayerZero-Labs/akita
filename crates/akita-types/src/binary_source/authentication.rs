//! Public digit commitments and their direct root authentication weights.

use akita_algebra::ring::{CyclotomicRing, ResidueKernelPoint};
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::{
    SetupStreamLabel, SetupStreamModulus, SetupStreamRing, SetupStreamRole, TypedSetupStream,
    BINARY_SETUP_P,
};
use jolt_field::Field;

use super::BinaryRootWitnessPlan;
use crate::native_root_relation::NativeRootWeights;

/// Ordinary ring degree used by the explicit experimental image and U setup.
pub const BINARY_DIGIT_COMMITMENT_DEGREE: usize = 64;

fn validate_label<F: Field>(label: SetupStreamLabel) -> Result<usize, AkitaError> {
    if label.level != 0
        || label.row != 0
        || label.column != 0
        || label.rank != 2
        || label.degree != BINARY_DIGIT_COMMITMENT_DEGREE as u32
        || label.ring != SetupStreamRing::Negacyclic
        || label.modulus != SetupStreamModulus::Prime128Offset275
        || !matches!(
            label.role,
            SetupStreamRole::ImageCommitment | SetupStreamRole::WitnessCommitment
        )
        || F::from_u128(BINARY_SETUP_P) != F::zero()
        || F::from_u128(BINARY_SETUP_P - 1) != -F::one()
    {
        return Err(AkitaError::InvalidSetup(
            "invalid binary digit commitment setup".into(),
        ));
    }
    usize::try_from(label.rank).map_err(|_| AkitaError::InvalidProof)
}

fn setup_entry<F: Field>(
    seed: &[u8; 32],
    label: SetupStreamLabel,
    row: u32,
    column: usize,
) -> Result<CyclotomicRing<F, BINARY_DIGIT_COMMITMENT_DEGREE>, AkitaError> {
    let mut stream = TypedSetupStream::new(
        seed,
        SetupStreamLabel {
            row,
            column: u64::try_from(column).map_err(|_| AkitaError::InvalidProof)?,
            ..label
        },
    )?;
    let mut coefficients = [F::zero(); BINARY_DIGIT_COMMITMENT_DEGREE];
    for coefficient in &mut coefficients {
        *coefficient = F::from_u128(stream.sample_p()?);
    }
    Ok(CyclotomicRing::from_coefficients(coefficients))
}

/// Commit compact balanced digits using the typed P setup in an ordinary ring.
///
/// Digits follow segment/element/digit/natural-coefficient order, omitting every
/// virtual tail in the root coefficient map. Consecutive chunks of 64 form the
/// setup columns; the final chunk is zero-filled. The returned payload is row
/// major. Image digits use `ImageCommitment`; U uses `WitnessCommitment`.
/// This experimental geometry carries no SIS security claim.
pub fn binary_public_digit_commitment<F: Field>(
    seed: &[u8; 32],
    label: SetupStreamLabel,
    digits: &[i8],
) -> Result<Vec<F>, AkitaError> {
    let rows = validate_label::<F>(label)?;
    if digits.is_empty() || digits.iter().any(|digit| !(-4..=3).contains(digit)) {
        return Err(AkitaError::InvalidInput(
            "binary commitment needs nonempty balanced digits".into(),
        ));
    }
    let count =
        checked::product([rows, BINARY_DIGIT_COMMITMENT_DEGREE]).ok_or(AkitaError::InvalidProof)?;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidProof)?;
    for row in 0..label.rank {
        let mut image = CyclotomicRing::zero();
        for (column, chunk) in digits.chunks(BINARY_DIGIT_COMMITMENT_DEGREE).enumerate() {
            let mut coefficients = [F::zero(); BINARY_DIGIT_COMMITMENT_DEGREE];
            for (output, &digit) in coefficients.iter_mut().zip(chunk) {
                *output = F::from_i64(i64::from(digit));
            }
            setup_entry(seed, label, row, column)?
                .mul_accumulate_into(&CyclotomicRing::from_coefficients(coefficients), &mut image);
        }
        payload.extend_from_slice(image.coefficients());
    }
    Ok(payload)
}

/// Authenticate the image digits or U view against the outgoing witness.
///
/// The ordinary negacyclic product is reduced before evaluation at `alpha`.
/// Each compact digit is read directly from its authenticated repacking address;
/// the image digits are not recomposed into T before applying B. No table or
/// quotient supplied in the clear enters these weights.
#[allow(clippy::too_many_arguments)]
pub fn binary_digit_authentication_weights<E: Field>(
    plan: &BinaryRootWitnessPlan,
    segment: usize,
    seed: &[u8; 32],
    label: SetupStreamLabel,
    payload: &[E],
    alpha: E,
    row_batches: &[E],
) -> Result<NativeRootWeights<E>, AkitaError> {
    let rows = validate_label::<E>(label)?;
    if !matches!(
        (segment, label.role),
        (1, SetupStreamRole::ImageCommitment) | (3, SetupStreamRole::WitnessCommitment)
    ) {
        return Err(AkitaError::InvalidSetup(
            "binary digit authentication role differs from its segment".into(),
        ));
    }
    let expected =
        checked::product([rows, BINARY_DIGIT_COMMITMENT_DEGREE]).ok_or(AkitaError::InvalidProof)?;
    if payload.len() != expected || row_batches.len() != rows {
        return Err(AkitaError::InvalidProof);
    }
    let shape = plan
        .repacking()
        .segments()
        .get(segment)
        .ok_or(AkitaError::InvalidProof)?;
    let width = checked::product([
        shape.geometry.polynomial_modulus_dimension(),
        shape.geometry.coordinate_plane_count(),
    ])
    .ok_or(AkitaError::InvalidProof)?;
    let count =
        checked::product([shape.elements, shape.digits, width]).ok_or(AkitaError::InvalidProof)?;
    let columns =
        checked::div_ceil(count, BINARY_DIGIT_COMMITMENT_DEGREE).ok_or(AkitaError::InvalidProof)?;
    let term_count = checked::product([rows, count]).ok_or(AkitaError::InvalidProof)?;
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(term_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    let mut claim = E::zero();
    let kernel = ResidueKernelPoint::new(alpha, BINARY_DIGIT_COMMITMENT_DEGREE)?;
    for (row, (target, &batch)) in payload
        .chunks_exact(BINARY_DIGIT_COMMITMENT_DEGREE)
        .zip(row_batches)
        .enumerate()
    {
        claim += batch
            * target
                .iter()
                .rev()
                .fold(E::zero(), |sum, &coefficient| sum * alpha + coefficient);
        let row = u32::try_from(row).map_err(|_| AkitaError::InvalidProof)?;
        for column in 0..columns {
            let matrix = setup_entry(seed, label, row, column)?;
            let mut weights = [E::zero(); BINARY_DIGIT_COMMITMENT_DEGREE];
            kernel.field_kernel_into(matrix.coefficients(), &mut weights)?;
            for (coefficient, &weight) in weights.iter().enumerate() {
                let compact = checked::mul_add(column, BINARY_DIGIT_COMMITMENT_DEGREE, coefficient)
                    .ok_or(AkitaError::InvalidProof)?;
                if compact >= count {
                    break;
                }
                let lane = compact / width;
                let address = plan.repacking().coefficient_address(
                    segment,
                    lane / shape.digits,
                    lane % shape.digits,
                    compact % width,
                )?;
                terms.push((address, batch * weight));
            }
        }
    }
    NativeRootWeights::new(plan.repacking().domain_len(), terms, claim)
}

#[cfg(test)]
#[path = "authentication_tests.rs"]
mod tests;
