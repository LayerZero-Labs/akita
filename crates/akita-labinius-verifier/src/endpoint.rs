//! Left expansion, characteristic-zero folding, and direct endpoint checks.

use akita_algebra::binary::{
    field_switch::{embed_source, SwitchField},
    BinaryField162,
};
use akita_algebra::fft::SmoothFftField;
use akita_algebra::ring::{
    embed_scalar, pack_scalar_components, TrinomialModulus, TrinomialNttDomain, TrinomialRing,
};
use akita_challenges::BinaryChallenge;
use akita_error::{checked, AkitaError};

use crate::commitment::{apply_matrix, BinaryClearCommitment};
use crate::frontend::BinaryEvaluationClaim;
use crate::profile::BinaryClearSetup;
use crate::source::{challenge_binary, challenge_scalar, equality_weights, scalar_from_signed};

/// Evaluate each source column at the low-order scalar-row coordinates.
pub fn left_expansion<H: SwitchField>(
    source: &[H::Source],
    point: &[BinaryField162],
    scalar_rows: usize,
    columns: usize,
) -> Result<Vec<BinaryField162>, AkitaError> {
    let length = checked::product([scalar_rows, columns])
        .ok_or_else(|| AkitaError::InvalidInput("left expansion size overflow".into()))?;
    if !scalar_rows.is_power_of_two() || !columns.is_power_of_two() || source.len() != length {
        return Err(AkitaError::InvalidInput(
            "left expansion geometry mismatch".into(),
        ));
    }
    let row_vars = scalar_rows.trailing_zeros() as usize;
    if point.len() != length.trailing_zeros() as usize {
        return Err(AkitaError::InvalidInput(
            "left expansion point dimension mismatch".into(),
        ));
    }
    let row_point = point.get(..row_vars).ok_or(AkitaError::InvalidProof)?;
    let weights = equality_weights(row_point)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(columns)
        .map_err(|_| AkitaError::InvalidInput("left expansion allocation failed".into()))?;
    for words in source.chunks_exact(scalar_rows) {
        let mut value = BinaryField162::ZERO;
        for (&weight, &word) in weights.iter().zip(words) {
            value += weight * embed_source::<H>(word);
        }
        result.push(value);
    }
    Ok(result)
}

/// Fold binary source columns in the integer scalar trinomial ring.
///
/// Accumulation and reduction are checked signed integer operations, prior to
/// either the prime embedding or the modulo-two projection.
pub fn fold_integer<H: SwitchField>(
    source: &[H::Source],
    scalar_rows: usize,
    columns: usize,
    challenges: &[BinaryChallenge],
) -> Result<Vec<[i64; 162]>, AkitaError> {
    let length = checked::product([scalar_rows, columns])
        .ok_or_else(|| AkitaError::InvalidInput("integer fold size overflow".into()))?;
    if !scalar_rows.is_power_of_two()
        || !columns.is_power_of_two()
        || source.len() != length
        || challenges.len() != columns
    {
        return Err(AkitaError::InvalidInput(
            "integer fold geometry mismatch".into(),
        ));
    }
    for challenge in challenges {
        challenge_binary(challenge)?;
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(scalar_rows)
        .map_err(|_| AkitaError::InvalidInput("integer response allocation failed".into()))?;
    for row in 0..scalar_rows {
        let mut product = [0i128; 323];
        for (column, challenge) in challenges.iter().enumerate() {
            let index =
                checked::mul_add(column, scalar_rows, row).ok_or(AkitaError::InvalidProof)?;
            let word = *source.get(index).ok_or(AkitaError::InvalidProof)?;
            let bits = embed_source::<H>(word).to_bytes();
            for term in challenge.terms() {
                for bit in 0..162 {
                    let byte = bits.get(bit / 8).ok_or(AkitaError::InvalidProof)?;
                    if byte & (1u8 << (bit % 8)) == 0 {
                        continue;
                    }
                    let position = usize::from(term.position)
                        .checked_add(bit)
                        .ok_or(AkitaError::InvalidProof)?;
                    let accumulator = product.get_mut(position).ok_or(AkitaError::InvalidProof)?;
                    *accumulator = accumulator
                        .checked_add(i128::from(term.coefficient))
                        .ok_or_else(|| {
                            AkitaError::InvalidInput("integer fold accumulator overflow".into())
                        })?;
                }
            }
        }
        for degree in (162..323).rev() {
            let leading = *product.get(degree).ok_or(AkitaError::InvalidProof)?;
            *product.get_mut(degree).ok_or(AkitaError::InvalidProof)? = 0;
            let reduced = degree - 162;
            for position in [reduced, reduced + 81] {
                let accumulator = product.get_mut(position).ok_or(AkitaError::InvalidProof)?;
                *accumulator = accumulator.checked_sub(leading).ok_or_else(|| {
                    AkitaError::InvalidInput("integer fold reduction overflow".into())
                })?;
            }
        }
        let mut response = [0i64; 162];
        for (destination, &coefficient) in response.iter_mut().zip(product.iter()) {
            *destination = i64::try_from(coefficient)
                .map_err(|_| AkitaError::InvalidInput("integer response exceeds i64".into()))?;
        }
        result.push(response);
    }
    Ok(result)
}

/// Project signed integer coefficients by Euclidean parity.
pub fn response_parity(response: &[i64; 162]) -> Result<BinaryField162, AkitaError> {
    let mut bytes = [0u8; 21];
    for (index, &coefficient) in response.iter().enumerate() {
        if coefficient.rem_euclid(2) == 1 {
            let byte = bytes.get_mut(index / 8).ok_or(AkitaError::InvalidProof)?;
            *byte |= 1u8 << (index % 8);
        }
    }
    BinaryField162::from_bytes(&bytes).ok_or(AkitaError::InvalidProof)
}

/// Embed and pack the exact decoded response, using contiguous scalar components.
pub fn pack_response<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<F, D, M>,
    response: &[[i64; 162]],
) -> Result<Vec<TrinomialRing<F, D, M>>, AkitaError> {
    if response.len() != setup.scalar_rows() {
        return Err(AkitaError::InvalidProof);
    }
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(setup.m())
        .map_err(|_| AkitaError::InvalidInput("packed response allocation failed".into()))?;
    for rows in response.chunks_exact(setup.k()) {
        let mut components = Vec::with_capacity(setup.k());
        for row in rows {
            components.push(scalar_from_signed::<F>(row)?);
        }
        packed.push(pack_scalar_components(&components).map_err(|_| AkitaError::InvalidProof)?);
    }
    Ok(packed)
}

/// Check that the column expansion evaluates to the terminal binary claim.
pub fn verify_left_expansion<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<F, D, M>,
    claim: &BinaryEvaluationClaim,
    u: &[BinaryField162],
) -> Result<(), AkitaError> {
    if claim.point.len() != setup.num_vars() || u.len() != setup.columns() {
        return Err(AkitaError::InvalidProof);
    }
    let col_point = claim
        .point
        .get(setup.row_vars()..)
        .ok_or(AkitaError::InvalidProof)?;
    let col_weights = equality_weights(col_point)?;
    if BinaryField162::dot_product(&col_weights, u) != Some(claim.value) {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

/// Check the left expansion and prime, binary, and accepted-range endpoints.
///
/// All checks use the same integer response; zero switch weights do not bypass
/// any source-opening check.
pub fn verify_endpoints<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<F, D, M>,
    commitment: &BinaryClearCommitment<F, D, M>,
    claim: &BinaryEvaluationClaim,
    u: &[BinaryField162],
    challenges: &[BinaryChallenge],
    response: &[[i64; 162]],
) -> Result<(), AkitaError> {
    verify_left_expansion(setup, claim, u)?;
    let image_count =
        checked::product([setup.n_a(), setup.columns()]).ok_or(AkitaError::InvalidProof)?;
    if commitment.images.len() != image_count
        || challenges.len() != setup.columns()
        || response.len() != setup.scalar_rows()
    {
        return Err(AkitaError::InvalidProof);
    }
    if response
        .iter()
        .flatten()
        .any(|&coefficient| coefficient < setup.lower() || coefficient > setup.upper())
    {
        return Err(AkitaError::InvalidProof);
    }
    for challenge in challenges {
        challenge
            .validate(setup.profile())
            .map_err(|_| AkitaError::InvalidProof)?;
    }
    let row_point = claim
        .point
        .get(..setup.row_vars())
        .ok_or(AkitaError::InvalidProof)?;
    let lhs = apply_matrix(setup, &pack_response(setup, response)?)?;
    let domain = TrinomialNttDomain::<F, D, M>::new()
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
    let mut workspace = domain.workspace();
    let mut embedded = Vec::new();
    embedded
        .try_reserve_exact(challenges.len())
        .map_err(|_| AkitaError::InvalidInput("embedded challenges allocation failed".into()))?;
    for challenge in challenges {
        embedded.push(
            embed_scalar(&challenge_scalar::<F>(challenge)?)
                .map_err(|_| AkitaError::InvalidProof)?,
        );
    }
    for (row, expected) in lhs.iter().enumerate() {
        let mut rhs = TrinomialRing::zero().map_err(|_| AkitaError::InvalidProof)?;
        for (column, challenge) in embedded.iter().enumerate() {
            let index =
                checked::mul_add(column, setup.n_a(), row).ok_or(AkitaError::InvalidProof)?;
            let image = commitment
                .images
                .get(index)
                .ok_or(AkitaError::InvalidProof)?;
            rhs += domain.multiply_with_workspace(challenge, image, &mut workspace);
        }
        if expected != &rhs {
            return Err(AkitaError::InvalidProof);
        }
    }
    let row_weights = equality_weights(row_point)?;
    let mut binary_lhs = BinaryField162::ZERO;
    for (&weight, response) in row_weights.iter().zip(response) {
        binary_lhs += weight * response_parity(response)?;
    }
    let mut binary_rhs = BinaryField162::ZERO;
    for (&value, challenge) in u.iter().zip(challenges) {
        binary_rhs += value * challenge_binary(challenge)?;
    }
    if binary_lhs != binary_rhs {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}
