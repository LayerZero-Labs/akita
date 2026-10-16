use akita_algebra::TrinomialModulus;
use akita_challenges::BinaryChallenge;
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{
    commitment::{folded_image_remainder, matrix_row_remainder},
    endpoint::pack_response,
    profile::BinaryClearSetup,
    BinaryClearCommitment,
};
use akita_params::sis::labinius::LabiniusSignedRange;

use super::matrix_remainders_limb;

/// Integer carries of the commitment rows, ordered by row then coefficient:
/// `KA_i = (rem_Phi(sum_j A_ij p_j) - rem_Phi(sum_col iota(c_col) T_(col,i))) / q`.
///
/// Both remainders are exact over the integers, with the matrix and the
/// images read as canonical residues. The commitment and response must have
/// the setup's geometry and response coefficients must be in its admitted
/// interval. A coefficient that is not divisible by `q`, or whose carry leaves
/// `range`, returns `InvalidProof`.
pub fn a_relation_carry<const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<D, M>,
    commitment: &BinaryClearCommitment,
    fold_challenges: &[BinaryChallenge],
    response: &[[i64; 162]],
    range: LabiniusSignedRange,
) -> Result<Vec<i128>, AkitaError> {
    let image_len =
        checked::product([setup.columns(), setup.n_a(), D]).ok_or(AkitaError::InvalidProof)?;
    if commitment.images.len() != image_len
        || commitment
            .images
            .iter()
            .any(|&value| value >= setup.modulus())
        || fold_challenges.len() != setup.columns()
        || response.len() != setup.scalar_rows()
        || response
            .iter()
            .flatten()
            .any(|&v| v < setup.lower() || v > setup.upper())
    {
        return Err(AkitaError::InvalidProof);
    }
    for challenge in fold_challenges {
        challenge
            .validate(setup.profile())
            .map_err(|_| AkitaError::InvalidProof)?;
    }
    let packed = pack_response(setup, response)?;
    let limb_rows = matrix_remainders_limb(setup, &packed)?;
    let q = i128::from(setup.modulus());
    let (lower, upper) = range.interval();
    let mut carry = Vec::new();
    carry
        .try_reserve_exact(checked::product([setup.n_a(), D]).ok_or(AkitaError::InvalidProof)?)
        .map_err(|_| AkitaError::InvalidProof)?;
    for row in 0..setup.n_a() {
        let reference_row;
        let response_row = if let Some(rows) = &limb_rows {
            let start = checked::product([row, D]).ok_or(AkitaError::InvalidProof)?;
            rows.get(checked::range(start, D).ok_or(AkitaError::InvalidProof)?)
                .ok_or(AkitaError::InvalidProof)?
        } else {
            reference_row = matrix_row_remainder(setup, row, &packed)?;
            &reference_row
        };
        let image_row = folded_image_remainder(setup, commitment, fold_challenges, row)?;
        for (&left, right) in response_row.iter().zip(image_row) {
            let residual = left.checked_sub(right).ok_or(AkitaError::InvalidProof)?;
            let quotient = residual / q;
            if residual % q != 0 || quotient < lower || quotient > upper {
                return Err(AkitaError::InvalidProof);
            }
            carry.push(quotient);
        }
    }
    Ok(carry)
}
