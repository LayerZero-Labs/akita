//! Compact private carriers and the existing packed arithmetic kernels.

use super::*;
use crate::kernels::binary::{
    fold::fold_integer,
    limb_commit::{commit_binary_clear_limb_prepared, PreparedLimbCommitMatrix},
};
use akita_algebra::{binary::BinaryField128, ring::trinomial::module_map::pack_signed_components};
use akita_challenges::{
    BinaryChallenge, BinaryChallengeProfile, BinaryScalarRing, Challenges,
    FoldChallengeDistribution,
};
use akita_types::binary::{BinarySourceDescriptor, BINARY_RESPONSE_INTERVAL};

pub(super) fn pack_logical_words(
    descriptor: &BinarySourceDescriptor,
    words: &[Vec<u128>],
) -> Result<Vec<u128>, AkitaError> {
    let index = descriptor.index_map();
    let owners = descriptor.polynomials().len();
    if words.len() != owners || words.iter().any(|words| words.len() != index.logical_len()) {
        return Err(AkitaError::InvalidInput(
            "binary logical word table differs from its descriptor".into(),
        ));
    }
    let mut sources = super::import::allocate_kernel_carriers(descriptor)?;
    let words_per_carrier = 128 / index.word_width();
    for (owner, polynomial) in words.iter().enumerate() {
        let owner_start =
            checked::product([owner, index.padded_carriers()]).ok_or(AkitaError::InvalidProof)?;
        for (word, &value) in polynomial.iter().enumerate() {
            if index.word_width() < 128 && value >> index.word_width() != 0 {
                return Err(AkitaError::InvalidInput(
                    "binary logical word exceeds its width".into(),
                ));
            }
            let address = checked::sum([
                owner_start,
                index.kernel_carrier_index(word / words_per_carrier)?,
            ])
            .ok_or(AkitaError::InvalidProof)?;
            let shift = checked::product([word % words_per_carrier, index.word_width()])
                .ok_or(AkitaError::InvalidProof)?;
            *sources.get_mut(address).ok_or(AkitaError::InvalidProof)? |= value << shift;
        }
    }
    Ok(sources)
}

pub(super) fn image_hint(
    descriptor: &BinarySourceDescriptor,
    rank: usize,
    matrix: &[Vec<[i128; SOURCE_RING_DEGREE]>],
    sources: &[u128],
) -> Result<Vec<[Q24Image; SOURCE_RING_DEGREE]>, AkitaError> {
    let index = descriptor.index_map();
    let rows = checked::product([4, index.columns()])
        .and_then(|block| checked::exact_div(index.padded_carriers(), block))
        .ok_or(AkitaError::InvalidProof)?;
    let matrix_count =
        checked::product([rank, rows, SOURCE_RING_DEGREE]).ok_or(AkitaError::InvalidProof)?;
    let mut residues = Vec::new();
    residues
        .try_reserve_exact(matrix_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    for &coefficient in matrix.iter().flatten().flatten() {
        residues.push(
            u32::try_from(coefficient.rem_euclid(i128::from(BINARY_SOURCE_Q)))
                .map_err(|_| AkitaError::InvalidProof)?,
        );
    }
    let prepared = PreparedLimbCommitMatrix::prepare(
        BINARY_SOURCE_Q,
        SOURCE_RING_DEGREE,
        rank,
        rows,
        &residues,
    )?;
    drop(residues);
    let columns = checked::product([descriptor.polynomials().len(), index.columns()])
        .ok_or(AkitaError::InvalidProof)?;
    let residues =
        commit_binary_clear_limb_prepared::<BinaryField128>(&prepared, columns, sources)?;
    residues
        .chunks_exact(SOURCE_RING_DEGREE)
        .map(|element| {
            let mut encoded = [Q24Image::from_residue(0)?; SOURCE_RING_DEGREE];
            for (destination, &residue) in encoded.iter_mut().zip(element) {
                *destination = Q24Image::from_residue(residue)?;
            }
            Ok(encoded)
        })
        .collect()
}

/// Fold the packed carriers and construct the root relation's residual rows.
#[cfg(test)]
pub(super) fn probe_packed(
    descriptor: &BinarySourceDescriptor,
    matrix: &[Vec<[i128; SOURCE_RING_DEGREE]>],
    sources: &[u128],
    images: &[[Q24Image; SOURCE_RING_DEGREE]],
    challenges: &Challenges,
    family: FoldChallengeDistribution,
) -> Result<Option<crate::binary_root::BinaryRootRelation>, AkitaError> {
    let Some(response) = fold_packed(descriptor, sources, challenges, family)? else {
        return Ok(None);
    };
    let _span = tracing::info_span!("binary_root_residual_rows").entered();
    crate::binary_root::BinaryRootRelation::from_response(
        descriptor, matrix, response, images, challenges, family,
    )
    .map(Some)
}

/// Fold the packed carriers into the bounded response, or `None` when a
/// clipped coefficient rejects this nonce.
pub(super) fn fold_packed(
    descriptor: &BinarySourceDescriptor,
    sources: &[u128],
    challenges: &Challenges,
    family: FoldChallengeDistribution,
) -> Result<Option<Vec<[i128; SOURCE_RING_DEGREE]>>, AkitaError> {
    let index = descriptor.index_map();
    let polynomials = descriptor.polynomials().len();
    let scalar_rows = checked::exact_div(index.padded_carriers(), index.columns())
        .ok_or(AkitaError::InvalidProof)?;
    let ring_rows = checked::exact_div(scalar_rows, 4).ok_or(AkitaError::InvalidProof)?;
    let count =
        checked::product([polynomials, index.padded_carriers()]).ok_or(AkitaError::InvalidProof)?;
    if family != FoldChallengeDistribution::BinaryRoot
        || sources.len() != count
        || challenges.num_claims() != polynomials
        || challenges.num_live_blocks_per_claim() != index.columns()
    {
        return Err(AkitaError::InvalidInput(
            "packed binary fold geometry mismatch".into(),
        ));
    }
    let profile =
        BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, family.weight())?;
    let binary = challenges
        .as_slice()
        .iter()
        .map(|challenge| BinaryChallenge::try_from_sparse(challenge, &profile))
        .collect::<Result<Vec<_>, _>>()?;
    let _span = tracing::info_span!("binary_root_fold_integer").entered();
    let mut response = Vec::new();
    response
        .try_reserve_exact(
            checked::product([polynomials, ring_rows]).ok_or(AkitaError::InvalidProof)?,
        )
        .map_err(|_| AkitaError::InvalidProof)?;
    for (source, challenges) in sources
        .chunks_exact(index.padded_carriers())
        .zip(binary.chunks_exact(index.columns()))
    {
        let scalar = fold_integer::<BinaryField128>(
            source,
            scalar_rows,
            index.columns(),
            challenges,
            &profile,
        )?;
        for components in scalar.chunks_exact(4) {
            let mut lifted = [[0i128; 162]; 4];
            for (destination, source) in lifted.iter_mut().zip(components) {
                for (out, &value) in destination.iter_mut().zip(source) {
                    *out = i128::from(value);
                }
            }
            let packed = pack_signed_components(&lifted)?;
            if packed.iter().any(|&coefficient| {
                coefficient < i128::from(BINARY_RESPONSE_INTERVAL.0)
                    || coefficient > i128::from(BINARY_RESPONSE_INTERVAL.1)
            }) {
                return Ok(None);
            }
            response.push(packed);
        }
    }
    Ok(Some(response))
}

#[cfg(test)]
mod tests;
