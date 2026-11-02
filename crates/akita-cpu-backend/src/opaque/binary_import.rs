//! Explicit compact import; the signed index map remains the sole address authority.

use super::{MAX_BINARY_SOURCE_MATERIAL_BYTES, SOURCE_RING_DEGREE};
use akita_error::{checked, AkitaError};
use akita_types::binary_source::BinarySourceDescriptor;
use std::borrow::Cow;
use std::mem::size_of;
use std::sync::Arc;

pub(super) enum BinarySourceInput {
    Logical(Vec<Vec<u128>>),
    Packed {
        logical_len: usize,
        carriers: Vec<Vec<u128>>,
    },
}

#[derive(Clone)]
pub(super) enum BinarySourceCoefficients {
    Expanded(Arc<[[i128; SOURCE_RING_DEGREE]]>),
    Packed(Arc<[Vec<u128>]>),
}

impl BinarySourceInput {
    pub(super) fn prepare(
        self,
        descriptor: &BinarySourceDescriptor,
    ) -> Result<BinarySourceCoefficients, AkitaError> {
        match self {
            Self::Logical(words) => Ok(BinarySourceCoefficients::Expanded(
                pack_logical_words(descriptor, &words)?.into(),
            )),
            Self::Packed {
                logical_len,
                carriers,
            } => {
                let index = descriptor.index_map();
                let words_per_carrier = 128 / index.word_width();
                let count = checked::div_ceil(logical_len, words_per_carrier)
                    .ok_or(AkitaError::InvalidProof)?;
                if logical_len != index.logical_len()
                    || carriers.len() != descriptor.polynomials().len()
                    || carriers.iter().any(|polynomial| polynomial.len() != count)
                {
                    return Err(AkitaError::InvalidInput(
                        "packed binary carriers differ from their descriptor".into(),
                    ));
                }
                let tail = logical_len % words_per_carrier;
                if tail != 0 {
                    let used_bits = checked::product([tail, index.word_width()])
                        .ok_or(AkitaError::InvalidProof)?;
                    for polynomial in &carriers {
                        let last = polynomial.last().ok_or(AkitaError::InvalidProof)?;
                        if last >> used_bits != 0 {
                            return Err(AkitaError::InvalidInput(
                                "packed binary carrier has nonzero unused high bits".into(),
                            ));
                        }
                    }
                }
                // Preserve the existing expanded-kernel allocation ceiling.
                // Compact import is not permission to admit larger schedules.
                source_ring_count(descriptor)?;
                Ok(BinarySourceCoefficients::Packed(carriers.into()))
            }
        }
    }
}

impl BinarySourceCoefficients {
    pub(super) fn expanded<'a>(
        &'a self,
        descriptor: &BinarySourceDescriptor,
    ) -> Result<Cow<'a, [[i128; SOURCE_RING_DEGREE]]>, AkitaError> {
        match self {
            Self::Expanded(sources) => Ok(Cow::Borrowed(sources)),
            Self::Packed(polynomials) => {
                let count = source_ring_count(descriptor)?;
                let index = descriptor.index_map();
                let rows = checked::product([4, index.columns()])
                    .and_then(|block| checked::exact_div(index.padded_carriers(), block))
                    .ok_or(AkitaError::InvalidProof)?;
                let mut sources = Vec::new();
                sources
                    .try_reserve_exact(count)
                    .map_err(|_| AkitaError::InvalidProof)?;
                sources.resize(count, [0; SOURCE_RING_DEGREE]);
                for (owner, carriers) in polynomials.iter().enumerate() {
                    for (carrier, &value) in carriers.iter().enumerate() {
                        for coordinate in 0..128 {
                            let position = index.scalar_to_ring(carrier, coordinate)?;
                            let element = checked::mul_add(owner, index.columns(), position.column)
                                .and_then(|column| {
                                    checked::mul_add(column, rows, position.ring_row)
                                })
                                .ok_or(AkitaError::InvalidProof)?;
                            *sources
                                .get_mut(element)
                                .and_then(|ring| ring.get_mut(position.coefficient))
                                .ok_or(AkitaError::InvalidProof)? =
                                i128::from(position.sign) * ((value >> coordinate) & 1) as i128;
                        }
                    }
                }
                Ok(Cow::Owned(sources))
            }
        }
    }

    pub(super) fn canonical_carriers(
        &self,
        descriptor: &BinarySourceDescriptor,
        owner: usize,
    ) -> Result<Vec<u128>, AkitaError> {
        use akita_algebra::ring::trinomial::module_map::{
            scalar_to_binary, unpack_signed_components,
        };
        let index = descriptor.index_map();
        if owner >= descriptor.polynomials().len() {
            return Err(AkitaError::InvalidProof);
        }
        let mut carriers = Vec::new();
        carriers
            .try_reserve_exact(index.padded_carriers())
            .map_err(|_| AkitaError::InvalidProof)?;
        match self {
            Self::Packed(polynomials) => {
                carriers.extend_from_slice(polynomials.get(owner).ok_or(AkitaError::InvalidProof)?);
                carriers.resize(index.padded_carriers(), 0);
            }
            Self::Expanded(sources) => {
                let rows = checked::product([4, index.columns()])
                    .and_then(|block| checked::exact_div(index.padded_carriers(), block))
                    .ok_or(AkitaError::InvalidProof)?;
                for carrier in 0..index.padded_carriers() {
                    let position = index.scalar_to_ring(carrier, 0)?;
                    let element = checked::mul_add(owner, index.columns(), position.column)
                        .and_then(|column| checked::mul_add(column, rows, position.ring_row))
                        .ok_or(AkitaError::InvalidProof)?;
                    let components = unpack_signed_components(
                        sources.get(element).ok_or(AkitaError::InvalidProof)?,
                    )?;
                    let [low, high, padding] = scalar_to_binary(
                        components
                            .get(position.coefficient)
                            .ok_or(AkitaError::InvalidProof)?,
                    )?
                    .to_words();
                    if padding != 0 {
                        return Err(AkitaError::InvalidInput(
                            "nonzero source carrier padding".into(),
                        ));
                    }
                    carriers.push(u128::from(low) | (u128::from(high) << 64));
                }
            }
        }
        Ok(carriers)
    }
}

fn source_ring_count(descriptor: &BinarySourceDescriptor) -> Result<usize, AkitaError> {
    let count = checked::exact_div(descriptor.index_map().padded_carriers(), 4)
        .and_then(|rings| checked::product([descriptor.polynomials().len(), rings]))
        .ok_or(AkitaError::InvalidProof)?;
    let bytes = checked::product([count, size_of::<[i128; SOURCE_RING_DEGREE]>()])
        .ok_or(AkitaError::InvalidProof)?;
    if bytes > MAX_BINARY_SOURCE_MATERIAL_BYTES {
        return Err(AkitaError::InvalidInput(
            "binary source allocation exceeds its budget".into(),
        ));
    }
    Ok(count)
}

fn pack_logical_words(
    descriptor: &akita_types::binary_source::BinarySourceDescriptor,
    words: &[Vec<u128>],
) -> Result<Vec<[i128; SOURCE_RING_DEGREE]>, AkitaError> {
    let index = descriptor.index_map();
    let owners = descriptor.polynomials().len();
    if words.len() != owners || words.iter().any(|words| words.len() != index.logical_len()) {
        return Err(AkitaError::InvalidInput(
            "binary logical word table differs from its descriptor".into(),
        ));
    }
    let rows = checked::product([4, index.columns()])
        .and_then(|block| checked::exact_div(index.padded_carriers(), block))
        .ok_or(AkitaError::InvalidProof)?;
    let count = source_ring_count(descriptor)?;
    let mut sources = Vec::new();
    sources
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidProof)?;
    sources.resize(count, [0; SOURCE_RING_DEGREE]);
    for (owner, polynomial) in words.iter().enumerate() {
        for (word, &value) in polynomial.iter().enumerate() {
            if index.word_width() < 128 && value >> index.word_width() != 0 {
                return Err(AkitaError::InvalidInput(
                    "binary logical word exceeds its width".into(),
                ));
            }
            for bit in 0..index.word_width() {
                let position = index.logical_to_ring(word, bit)?;
                let element = checked::mul_add(owner, index.columns(), position.column)
                    .and_then(|column| checked::mul_add(column, rows, position.ring_row))
                    .ok_or(AkitaError::InvalidProof)?;
                let coefficient = sources
                    .get_mut(element)
                    .and_then(|element| element.get_mut(position.coefficient))
                    .ok_or(AkitaError::InvalidProof)?;
                *coefficient = i128::from(position.sign) * ((value >> bit) & 1) as i128;
            }
        }
    }
    Ok(sources)
}

#[cfg(test)]
#[path = "binary_import_tests.rs"]
mod tests;
