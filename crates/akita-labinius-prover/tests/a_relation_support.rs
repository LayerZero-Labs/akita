//! Independent schoolbook matrix-row remainders and integer carries for tests.

#![cfg(feature = "labinius")]
#![allow(dead_code)]

use akita_algebra::{embed_scalar, SmoothFftField, TrinomialModulus};
use akita_challenges::BinaryChallenge;
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{
    commitment::BinaryClearCommitment, endpoint::pack_response, profile::BinaryClearSetup,
    source::challenge_scalar,
};

/// Form unreduced products and reduce their row residuals by descending division.
/// This reference computes no trace map or transform.
pub(crate) fn a_relation_remainders<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<F, D, M>,
    commitment: &BinaryClearCommitment<F, D, M>,
    fold_challenges: &[BinaryChallenge],
    response: &[[i64; 162]],
) -> Result<Vec<Vec<F>>, AkitaError> {
    let image_count =
        checked::product([setup.columns(), setup.n_a()]).ok_or(AkitaError::InvalidProof)?;
    if commitment.images.len() != image_count
        || fold_challenges.len() != setup.columns()
        || response.len() != setup.scalar_rows()
        || response
            .iter()
            .flatten()
            .any(|&value| value < setup.lower() || value > setup.upper())
    {
        return Err(AkitaError::InvalidProof);
    }
    let packed = pack_response(setup, response)?;
    let mut embedded = Vec::new();
    embedded
        .try_reserve_exact(setup.columns())
        .map_err(|_| AkitaError::InvalidProof)?;
    for challenge in fold_challenges {
        challenge
            .validate(setup.profile())
            .map_err(|_| AkitaError::InvalidProof)?;
        embedded.push(
            embed_scalar::<F, 162, D, M>(&challenge_scalar(challenge)?)
                .map_err(|_| AkitaError::InvalidProof)?,
        );
    }
    let product_len = checked::product([2, D])
        .and_then(|length| length.checked_sub(1))
        .ok_or(AkitaError::InvalidProof)?;
    let mut remainders = Vec::new();
    remainders
        .try_reserve_exact(setup.n_a())
        .map_err(|_| AkitaError::InvalidProof)?;
    for row in 0..setup.n_a() {
        let mut residual = Vec::new();
        residual
            .try_reserve_exact(product_len)
            .map_err(|_| AkitaError::InvalidProof)?;
        residual.resize(product_len, F::zero());
        let start = checked::product([row, setup.m()]).ok_or(AkitaError::InvalidProof)?;
        let range = checked::range(start, setup.m()).ok_or(AkitaError::InvalidProof)?;
        let matrix_row = setup.matrix().get(range).ok_or(AkitaError::InvalidProof)?;
        for (matrix, value) in matrix_row.iter().zip(&packed) {
            let product = matrix
                .schoolbook_product_coefficients(value)
                .map_err(|_| AkitaError::InvalidProof)?;
            for (accumulator, coefficient) in residual.iter_mut().zip(product) {
                *accumulator += coefficient;
            }
        }
        for (column, challenge) in embedded.iter().enumerate() {
            let image_index =
                checked::mul_add(column, setup.n_a(), row).ok_or(AkitaError::InvalidProof)?;
            let image = commitment
                .images
                .get(image_index)
                .ok_or(AkitaError::InvalidProof)?;
            let product = challenge
                .schoolbook_product_coefficients(image)
                .map_err(|_| AkitaError::InvalidProof)?;
            for (accumulator, coefficient) in residual.iter_mut().zip(product) {
                *accumulator -= coefficient;
            }
        }
        for degree in (D..product_len).rev() {
            let coefficient = residual[degree];
            residual[degree - D] -= coefficient;
            residual[degree - D / 2] -= F::from_i64(i64::from(M::MIDDLE_COEFFICIENT)) * coefficient;
        }
        residual.truncate(D);
        remainders.push(residual);
    }
    Ok(remainders)
}

/// Independently center the field remainder and divide its coefficients over Z.
pub(crate) fn a_relation_carry<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<F, D, M>,
    commitment: &BinaryClearCommitment<F, D, M>,
    fold_challenges: &[BinaryChallenge],
    response: &[[i64; 162]],
    range: akita_params::sis::labinius::LabiniusSignedDigitRange,
) -> Result<Vec<i128>, AkitaError> {
    let q0 = i128::from(setup.commitment_modulus().small_modulus().ok_or_else(|| {
        AkitaError::InvalidSetup("A carry requires a small commitment modulus".into())
    })?);
    let (lower, upper) = range.interval();
    a_relation_remainders(setup, commitment, fold_challenges, response)?
        .into_iter()
        .flatten()
        .map(|coefficient| {
            let lifted = akita_labinius_verifier::commitment::centered_coefficient(coefficient)?;
            if lifted % q0 != 0 || lifted / q0 < lower || lifted / q0 > upper {
                return Err(AkitaError::InvalidProof);
            }
            Ok(lifted / q0)
        })
        .collect()
}
