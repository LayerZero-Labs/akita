//! Deterministic expansion of balanced-ternary JL matrices.

mod aes_ctr;

use crate::sampler::shake256_root;
use aes_ctr::Aes128CtrExpander;
use akita_algebra::jl::{TernaryProjectionMatrix, TernaryProjectionShape};
use akita_error::AkitaError;

/// Version of the canonical balanced-ternary matrix expansion.
pub const BALANCED_TERNARY_EXPANSION_VERSION: u32 = 1;

const BALANCED_TERNARY_DOMAIN: &[u8] = b"akita/jl/balanced-ternary/aes128-ctr";

/// Expand a 32-byte Fiat--Shamir seed into a canonical balanced-ternary matrix.
///
/// Every entry is computationally indistinguishable from an independent draw
/// that is `0` with probability `1/2` and `-1` or `+1` with probability `1/4`
/// each. The expansion derives a matrix-specific AES-128 key and base block
/// from the domain tag, expansion version, shape, and seed. Disjoint AES-CTR
/// streams fill the complete nonzero and raw-positive planes. This follows the
/// hardware-accelerated expansion pattern used by Greyhound while retaining
/// the balanced-ternary law required by the certified bounds.
///
/// The exact byte encoding is
/// `SHAKE256(domain || version_le32 || rows_le64 || cols_le64 || seed)[0..32]`
/// for the root. The first 16 root bytes are the AES-128 key and the final 16
/// are a base block, parsed as two little-endian words `(base_counter,
/// base_nonce)`. AES input block `counter` of stream `s` is
/// `(base_counter + counter)_le64 || (base_nonce XOR s)_le64`; stream `0` fills
/// the row-major nonzero plane and stream `1` fills the row-major raw-positive
/// plane. AES output bytes are concatenated in increasing counter order. Bits
/// and columns are little-endian within each byte.
///
/// # Errors
///
/// Returns an error if the shape cannot be encoded, allocation fails, or the
/// resulting packed matrix is not canonical.
pub fn expand_balanced_ternary_matrix(
    seed: &[u8; 32],
    shape: TernaryProjectionShape,
) -> Result<TernaryProjectionMatrix, AkitaError> {
    let rows = u64::try_from(shape.rows())
        .map_err(|_| AkitaError::InvalidInput("ternary row count exceeds u64".into()))?;
    let cols = u64::try_from(shape.cols())
        .map_err(|_| AkitaError::InvalidInput("ternary column count exceeds u64".into()))?;
    let version_bytes = BALANCED_TERNARY_EXPANSION_VERSION.to_le_bytes();
    let rows_bytes = rows.to_le_bytes();
    let cols_bytes = cols.to_le_bytes();
    let root = shake256_root(&[
        BALANCED_TERNARY_DOMAIN,
        &version_bytes,
        &rows_bytes,
        &cols_bytes,
        seed,
    ])
    .map_err(|message| AkitaError::InvalidInput(message.into()))?;
    let key = std::array::from_fn(|index| root[index]);
    let base_block = std::array::from_fn(|index| root[index + 16]);
    let expander = Aes128CtrExpander::new(&key, base_block);
    let mut nonzero = try_zeroed_bytes(shape.plane_len())?;
    let mut positive = try_zeroed_bytes(shape.plane_len())?;
    expander.fill_stream(0, &mut nonzero);
    expander.fill_stream(1, &mut positive);
    for (nonzero_row, positive_row) in nonzero
        .chunks_exact_mut(shape.row_bytes())
        .zip(positive.chunks_exact_mut(shape.row_bytes()))
    {
        for (positive_byte, &nonzero_byte) in positive_row.iter_mut().zip(&*nonzero_row) {
            *positive_byte &= nonzero_byte;
        }
        let final_byte = shape.row_bytes() - 1;
        nonzero_row[final_byte] &= shape.final_byte_live_mask();
        positive_row[final_byte] &= shape.final_byte_live_mask();
    }
    TernaryProjectionMatrix::from_bitplanes(shape, nonzero, positive)
}

fn try_zeroed_bytes(len: usize) -> Result<Vec<u8>, AkitaError> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|_| {
        AkitaError::InvalidInput(format!(
            "balanced-ternary expansion allocation failed for {len} bytes"
        ))
    })?;
    bytes.resize(len, 0);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(matrix: &TernaryProjectionMatrix) -> Vec<i8> {
        let shape = matrix.shape();
        (0..shape.rows())
            .flat_map(|row| (0..shape.cols()).map(move |col| matrix.entry(row, col).unwrap()))
            .collect()
    }

    #[test]
    fn expansion_is_deterministic_and_shape_bound() {
        let seed = [0x5au8; 32];
        let shape = TernaryProjectionShape::new(7, 19).unwrap();
        let first = expand_balanced_ternary_matrix(&seed, shape).unwrap();
        let second = expand_balanced_ternary_matrix(&seed, shape).unwrap();
        assert_eq!(first, second);

        let different_seed = expand_balanced_ternary_matrix(&[0x5bu8; 32], shape).unwrap();
        assert_ne!(first, different_seed);
        let different_shape =
            expand_balanced_ternary_matrix(&seed, TernaryProjectionShape::new(7, 20).unwrap())
                .unwrap();
        assert_ne!(entries(&first), entries(&different_shape));
    }

    #[test]
    fn expansion_has_the_balanced_ternary_law() {
        let shape = TernaryProjectionShape::new(256, 1024).unwrap();
        let matrix = expand_balanced_ternary_matrix(&[0xa5u8; 32], shape).unwrap();
        let mut counts = [0usize; 3];
        for value in entries(&matrix) {
            counts[(value + 1) as usize] += 1;
        }
        let total = shape.rows() * shape.cols();
        assert!((counts[0] as isize - (total / 4) as isize).unsigned_abs() < total / 100);
        assert!((counts[1] as isize - (total / 2) as isize).unsigned_abs() < total / 100);
        assert!((counts[2] as isize - (total / 4) as isize).unsigned_abs() < total / 100);
    }

    #[test]
    fn expansion_known_answer() {
        let matrix = expand_balanced_ternary_matrix(
            &[0x42u8; 32],
            TernaryProjectionShape::new(2, 16).unwrap(),
        )
        .unwrap();
        assert_eq!(
            entries(&matrix),
            vec![
                0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, -1, 1, 0, 0, 0, 1, 1, 1, -1, 0, -1, 0, 1, 0,
                0, 0, 0, -1, 0, 0,
            ]
        );
    }
}
