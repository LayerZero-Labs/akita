//! Explicit compact import; the signed index map remains the sole address authority.

use super::packed::pack_logical_words;
use super::MAX_BINARY_SOURCE_MATERIAL_BYTES;
use akita_error::AkitaError;
use akita_types::binary_source::BinarySourceDescriptor;
use jolt_algebra::checked;
use std::mem::size_of;

pub(super) enum BinarySourceInput {
    Logical(Vec<Vec<u128>>),
    Packed {
        logical_len: usize,
        carriers: Vec<Vec<u128>>,
    },
}

impl BinarySourceInput {
    /// Lay the source out as owner-major kernel-order carriers.
    pub(super) fn prepare(
        self,
        descriptor: &BinarySourceDescriptor,
    ) -> Result<Vec<u128>, AkitaError> {
        match self {
            Self::Logical(words) => pack_logical_words(descriptor, &words),
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
                let mut sources = allocate_kernel_carriers(descriptor)?;
                for (owner, polynomial) in carriers.iter().enumerate() {
                    let owner_start = checked::product([owner, index.padded_carriers()])
                        .ok_or(AkitaError::InvalidProof)?;
                    for (carrier, &value) in polynomial.iter().enumerate() {
                        let address =
                            checked::sum([owner_start, index.kernel_carrier_index(carrier)?])
                                .ok_or(AkitaError::InvalidProof)?;
                        *sources.get_mut(address).ok_or(AkitaError::InvalidProof)? = value;
                    }
                }
                Ok(sources)
            }
        }
    }
}

/// Zeroed owner-major kernel carriers, within the private material budget.
pub(super) fn allocate_kernel_carriers(
    descriptor: &BinarySourceDescriptor,
) -> Result<Vec<u128>, AkitaError> {
    let count = checked::product([
        descriptor.polynomials().len(),
        descriptor.index_map().padded_carriers(),
    ])
    .ok_or(AkitaError::InvalidProof)?;
    let bytes = checked::product([count, size_of::<u128>()]).ok_or(AkitaError::InvalidProof)?;
    if bytes > MAX_BINARY_SOURCE_MATERIAL_BYTES {
        return Err(AkitaError::InvalidInput(
            "binary source allocation exceeds its budget".into(),
        ));
    }
    let mut sources = Vec::new();
    sources
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidProof)?;
    sources.resize(count, 0);
    Ok(sources)
}

#[cfg(test)]
#[path = "binary_import_tests.rs"]
mod tests;
