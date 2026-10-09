//! Own-ring matrix-row remainders and their centered integer carries.

#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

use akita_algebra::{embed_scalar, SmoothFftField, TrinomialModulus, TrinomialRing};
use akita_challenges::BinaryChallenge;
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{
    commitment::centered_coefficient, endpoint::pack_response, profile::BinaryClearSetup,
    source::challenge_scalar, BinaryClearCommitment,
};
use akita_params::sis::labinius::LabiniusSignedDigitRange;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::PreparedCommitMatrix;

type ColumnCarryWork<'a, F, const D: usize, M> = (
    &'a mut [TrinomialRing<F, D, M>],
    (&'a TrinomialRing<F, D, M>, &'a [TrinomialRing<F, D, M>]),
);

/// Compute the tag-1 matrix-row remainders, center them, and divide over Z by q0.
///
/// The matrix cache must belong to `setup`; the commitment and response must
/// have the setup's geometry, and response coefficients must be in its admitted
/// interval. Every centered coefficient must be divisible by q0 and its integer
/// carry must lie in `range`. A shared-prime setup returns `InvalidSetup`.
/// Parallel column products use independent workspaces and canonical ordering.
pub fn a_relation_carry<F, const D: usize, M>(
    prepared: &PreparedCommitMatrix<F, D, M>,
    setup: &BinaryClearSetup<F, D, M>,
    commitment: &BinaryClearCommitment<F, D, M>,
    fold_challenges: &[BinaryChallenge],
    response: &[[i64; 162]],
    range: LabiniusSignedDigitRange,
) -> Result<Vec<i128>, AkitaError>
where
    F: SmoothFftField,
    M: TrinomialModulus + Send + Sync,
{
    let image_count =
        checked::product([setup.columns(), setup.n_a()]).ok_or(AkitaError::InvalidProof)?;
    if commitment.images.len() != image_count
        || fold_challenges.len() != setup.columns()
        || response.len() != setup.scalar_rows()
        || response
            .iter()
            .flatten()
            .any(|&v| v < setup.lower() || v > setup.upper())
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
    prepared.check_setup(setup)?;
    let q0 = i128::from(setup.commitment_modulus().small_modulus().ok_or_else(|| {
        AkitaError::InvalidSetup("A carry requires a small commitment modulus".into())
    })?);
    let domain = prepared.domain();
    let mut lhs = Vec::new();
    lhs.try_reserve_exact(setup.n_a())
        .map_err(|_| AkitaError::InvalidProof)?;
    lhs.resize(setup.n_a(), domain.zero_ntt());
    let mut workspace = domain.workspace();
    let mut transformed = domain.zero_ntt();
    for (element, value) in packed.iter().enumerate() {
        domain.forward_into_with_workspace(value, &mut transformed, &mut workspace);
        for (accumulator, row) in lhs
            .iter_mut()
            .zip(prepared.matrix_ntt().chunks_exact(setup.m()))
        {
            accumulator.add_assign_pointwise_mul(
                row.get(element).ok_or(AkitaError::InvalidProof)?,
                &transformed,
            );
        }
    }
    let mut rhs = Vec::new();
    rhs.try_reserve_exact(image_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    rhs.resize(
        image_count,
        TrinomialRing::zero().map_err(|_| AkitaError::InvalidProof)?,
    );
    let column_work =
        |workspace: &mut _, (output, (challenge, images)): ColumnCarryWork<'_, F, D, M>| {
            let challenge_ntt = domain.forward_with_workspace(challenge, workspace);
            for (result, image) in output.iter_mut().zip(images) {
                let image_ntt = domain.forward_with_workspace(image, workspace);
                *result = domain
                    .inverse_with_workspace(&challenge_ntt.pointwise_mul(&image_ntt), workspace);
            }
        };
    #[cfg(feature = "parallel")]
    rhs.par_chunks_mut(setup.n_a())
        .zip(
            embedded
                .par_iter()
                .zip(commitment.images.par_chunks(setup.n_a())),
        )
        .for_each_init(|| domain.workspace(), column_work);
    #[cfg(not(feature = "parallel"))]
    for item in rhs
        .chunks_mut(setup.n_a())
        .zip(embedded.iter().zip(commitment.images.chunks(setup.n_a())))
    {
        column_work(&mut workspace, item);
    }
    let mut carry = Vec::new();
    carry
        .try_reserve_exact(checked::product([setup.n_a(), D]).ok_or(AkitaError::InvalidProof)?)
        .map_err(|_| AkitaError::InvalidProof)?;
    let (lower, upper) = range.interval();
    for (row, accumulator) in lhs.iter().enumerate() {
        let mut remainder = domain
            .inverse_with_workspace(accumulator, &mut workspace)
            .into_coefficients();
        for column in rhs.chunks_exact(setup.n_a()) {
            for (destination, &coefficient) in remainder.iter_mut().zip(
                column
                    .get(row)
                    .ok_or(AkitaError::InvalidProof)?
                    .coefficients(),
            ) {
                *destination -= coefficient;
            }
        }
        // Admission ensures 6*H_A < P, giving the honest remainder its unique
        // centered lift. This divisibility test is over the integers.
        for coefficient in remainder {
            let lifted = centered_coefficient(coefficient)?;
            if lifted % q0 != 0 || lifted / q0 < lower || lifted / q0 > upper {
                return Err(AkitaError::InvalidProof);
            }
            carry.push(lifted / q0);
        }
    }
    Ok(carry)
}
