//! Vectorizable characteristic-zero response folding, independent of the root prover.

#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

use akita_algebra::binary::field_switch::{embed_source, SwitchField};
use akita_challenges::{BinaryChallenge, BinaryChallengeProfile};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{endpoint, source::challenge_binary};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Fold source columns with the reference endpoint's validation and integer semantics.
///
/// Each embedded word is expanded once, then shifted signed additions operate
/// on contiguous lanes, allowing LLVM to vectorize without per-bit branches.
/// A checked sum of term counts bounds every partial convolution coefficient by
/// B. We use i16 only for B <= i16::MAX, then widen before reduction: degrees
/// 243..322 have magnitude <= B, degrees 162..242 <= 2B, and final coefficients
/// <= 4B. Thus neither accumulation, i64 reduction, nor conversion can overflow
/// on this path, and the reference's i128 overflow errors are unreachable here.
/// Larger bounds retain the reference's checked arithmetic and errors.
///
/// With `parallel`, independent rows write disjoint output slots. Source bytes
/// are prepared on the calling thread because the sealed `SwitchField` contract
/// does not require its associated source type to implement `Sync`. Scratch
/// reservation failure falls back to the reference, preserving its error API.
pub fn fold_integer<H: SwitchField>(
    source: &[H::Source],
    scalar_rows: usize,
    columns: usize,
    challenges: &[BinaryChallenge],
    profile: &BinaryChallengeProfile,
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
        challenge_binary(challenge, profile)?;
    }
    let bound = checked::sum(challenges.iter().map(BinaryChallenge::weight));
    if bound.is_none_or(|bound| bound > i16::MAX as usize) {
        return endpoint::fold_integer::<H>(source, scalar_rows, columns, challenges, profile);
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(scalar_rows)
        .map_err(|_| AkitaError::InvalidInput("integer response allocation failed".into()))?;
    result.resize(scalar_rows, [0i64; 162]);

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && scalar_rows > 1 {
        let mut words = Vec::new();
        if words.try_reserve_exact(length).is_err() {
            drop(result);
            return endpoint::fold_integer::<H>(source, scalar_rows, columns, challenges, profile);
        }
        words.extend(
            source
                .iter()
                .map(|&word| embed_source::<H>(word).to_bytes()),
        );
        result
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(row, response)| {
                fold_row(challenges, response, |column| {
                    let index = checked::mul_add(column, scalar_rows, row)
                        .ok_or(AkitaError::InvalidProof)?;
                    words.get(index).copied().ok_or(AkitaError::InvalidProof)
                })
            })?;
        return Ok(result);
    }

    for (row, response) in result.iter_mut().enumerate() {
        fold_row(challenges, response, |column| {
            let index =
                checked::mul_add(column, scalar_rows, row).ok_or(AkitaError::InvalidProof)?;
            let word = *source.get(index).ok_or(AkitaError::InvalidProof)?;
            Ok(embed_source::<H>(word).to_bytes())
        })?;
    }
    Ok(result)
}

fn fold_row(
    challenges: &[BinaryChallenge],
    response: &mut [i64; 162],
    mut word_at: impl FnMut(usize) -> Result<[u8; 21], AkitaError>,
) -> Result<(), AkitaError> {
    let mut product = [0i16; 323];
    for (column, challenge) in challenges.iter().enumerate() {
        let bits = word_at(column)?;
        let mut lanes = [0i16; 162];
        for (chunk, byte) in lanes.chunks_mut(8).zip(bits) {
            for (bit, lane) in chunk.iter_mut().enumerate() {
                *lane = i16::from((byte >> bit) & 1);
            }
        }
        for term in challenge.terms() {
            let range = checked::range(usize::from(term.position), lanes.len())
                .ok_or(AkitaError::InvalidProof)?;
            let destination = product.get_mut(range).ok_or(AkitaError::InvalidProof)?;
            if term.coefficient == 1 {
                for (accumulator, &lane) in destination.iter_mut().zip(&lanes) {
                    *accumulator += lane;
                }
            } else {
                for (accumulator, &lane) in destination.iter_mut().zip(&lanes) {
                    *accumulator -= lane;
                }
            }
        }
    }
    let mut product = product.map(i64::from);
    for degree in (162..323).rev() {
        let leading = *product.get(degree).ok_or(AkitaError::InvalidProof)?;
        *product.get_mut(degree).ok_or(AkitaError::InvalidProof)? = 0;
        for position in [degree - 162, degree - 81] {
            *product.get_mut(position).ok_or(AkitaError::InvalidProof)? -= leading;
        }
    }
    for (destination, &coefficient) in response.iter_mut().zip(&product) {
        *destination = coefficient;
    }
    Ok(())
}
