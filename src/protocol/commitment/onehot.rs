//! One-hot commitment path for regular one-hot ring elements.
//!
//! Exploits the sparsity of one-hot witnesses (coefficients in {0,1}) to
//! eliminate all inner ring multiplications. The inner Ajtai `t = A * s`
//! reduces to summing selected columns of `A` with negacyclic rotations.

use std::collections::BTreeMap;

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::{CanonicalField, FieldCore};

/// Describes a nonzero ring element within one block of the commitment layout.
#[derive(Debug, Clone)]
pub struct SparseBlockEntry {
    /// Position within the block (0..2^M).
    pub pos_in_block: usize,
    /// Coefficient indices that are 1 within this ring element.
    pub nonzero_coeffs: Vec<usize>,
}

/// Map a regular one-hot witness to sparse ring block entries.
///
/// - `onehot_k`: chunk size K. The witness has T chunks of K field elements,
///   each chunk containing exactly one 1.
/// - `indices`: length-T slice where `indices[c]` is the hot position in
///   chunk `c` (must be in `[0, K)`).
/// - `r`, `m`: commitment config parameters (2^R blocks of 2^M ring elements).
/// - `D`: ring degree (const generic on caller side, passed as runtime here).
///
/// Returns one `Vec<SparseBlockEntry>` per block (outer len = 2^R).
///
/// # Errors
///
/// Returns an error if K and D are not "nicely matched" (one must divide
/// the other), if any index is out of range, or if the dimensions don't
/// fill the commitment layout.
pub fn map_onehot_to_sparse_blocks(
    onehot_k: usize,
    indices: &[Option<usize>],
    r: usize,
    m: usize,
    d: usize,
) -> Result<Vec<Vec<SparseBlockEntry>>, HachiError> {
    if onehot_k == 0 || d == 0 {
        return Err(HachiError::InvalidInput(
            "onehot_k and D must be nonzero".into(),
        ));
    }
    if !(onehot_k % d == 0 || d % onehot_k == 0) {
        return Err(HachiError::InvalidInput(format!(
            "K={onehot_k} and D={d} must be nicely matched (one divides the other)"
        )));
    }

    let num_chunks = indices.len();
    let total_field_elems = num_chunks
        .checked_mul(onehot_k)
        .ok_or_else(|| HachiError::InvalidInput("T*K overflow".into()))?;
    if total_field_elems % d != 0 {
        return Err(HachiError::InvalidInput(format!(
            "T*K={total_field_elems} is not divisible by D={d}"
        )));
    }
    let total_ring_elems = total_field_elems / d;
    let num_blocks = 1usize << r;
    let block_len = 1usize << m;
    if total_ring_elems != num_blocks * block_len {
        return Err(HachiError::InvalidSize {
            expected: num_blocks * block_len,
            actual: total_ring_elems,
        });
    }

    // Accumulate nonzero coefficients per ring element index.
    let mut ring_elem_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (c, opt) in indices.iter().enumerate() {
        let Some(&idx) = opt.as_ref() else { continue };
        if idx >= onehot_k {
            return Err(HachiError::InvalidInput(format!(
                "index {idx} out of range for chunk size K={onehot_k} at position {c}"
            )));
        }
        let field_pos = c * onehot_k + idx;
        let ring_elem_idx = field_pos / d;
        let coeff_idx = field_pos % d;
        ring_elem_map
            .entry(ring_elem_idx)
            .or_default()
            .push(coeff_idx);
    }

    // Sequential block layout matching commit_coeffs: block i = ring elements
    // [i*block_len, (i+1)*block_len).
    let mut blocks: Vec<Vec<SparseBlockEntry>> = vec![Vec::new(); num_blocks];
    for (ring_elem_idx, nonzero_coeffs) in ring_elem_map {
        let block_idx = ring_elem_idx / block_len;
        let pos_in_block = ring_elem_idx % block_len;
        blocks[block_idx].push(SparseBlockEntry {
            pos_in_block,
            nonzero_coeffs,
        });
    }

    Ok(blocks)
}

/// Sparse inner Ajtai: compute `t = A * s` for a one-hot block.
///
/// Instead of materializing the full decomposed vector `s` and doing a dense
/// matvec, we accumulate only the nonzero contributions:
///
/// ```text
/// t[a] = sum_{entry} A[a][entry.pos * delta].mul_by_monomial_sum(entry.nonzero_coeffs)
/// ```
#[allow(non_snake_case)]
pub(crate) fn inner_ajtai_onehot_t_only<F: FieldCore + CanonicalField, const D: usize>(
    A: &[Vec<CyclotomicRing<F, D>>],
    sparse_entries: &[SparseBlockEntry],
    _block_len: usize,
    _delta: usize,
) -> Vec<CyclotomicRing<F, D>> {
    let n_a = A.len();

    let mut t = vec![CyclotomicRing::<F, D>::zero(); n_a];
    for entry in sparse_entries {
        let col = entry.pos_in_block * _delta;
        for a in 0..n_a {
            t[a] += A[a][col].mul_by_monomial_sum(&entry.nonzero_coeffs);
        }
    }

    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_onehot_k_gt_d() {
        // K=16, D=4, T=2 chunks => 32 field elements => 8 ring elements
        // R=1 (2 blocks), M=2 (4 per block) => 8 ring elements total
        let k = 16;
        let d = 4;
        let indices = vec![Some(3), Some(10)];
        let blocks = map_onehot_to_sparse_blocks(k, &indices, 1, 2, d).unwrap();

        assert_eq!(blocks.len(), 2);
        let total_entries: usize = blocks.iter().map(|b| b.len()).sum();
        assert_eq!(total_entries, 2, "T=2 nonzero ring elements");

        for block in &blocks {
            for entry in block {
                assert_eq!(entry.nonzero_coeffs.len(), 1, "K>D => single monomial");
            }
        }
    }

    #[test]
    fn map_onehot_k_eq_d() {
        // K=4, D=4, T=4 chunks => 16 field elements => 4 ring elements
        // R=1 (2 blocks), M=1 (2 per block)
        let k = 4;
        let d = 4;
        let indices = vec![Some(0), Some(2), Some(3), Some(1)];
        let blocks = map_onehot_to_sparse_blocks(k, &indices, 1, 1, d).unwrap();

        assert_eq!(blocks.len(), 2);
        let total_entries: usize = blocks.iter().map(|b| b.len()).sum();
        assert_eq!(total_entries, 4, "K=D => every ring element is nonzero");

        for block in &blocks {
            for entry in block {
                assert_eq!(entry.nonzero_coeffs.len(), 1, "K=D => single monomial");
            }
        }
    }

    #[test]
    fn map_onehot_k_lt_d() {
        // K=4, D=8, T=8 chunks => 32 field elements => 4 ring elements
        // R=1 (2 blocks), M=1 (2 per block)
        let k = 4;
        let d = 8;
        let indices = vec![
            Some(0),
            Some(2),
            Some(3),
            Some(1),
            Some(0),
            Some(0),
            Some(3),
            Some(3),
        ];
        let blocks = map_onehot_to_sparse_blocks(k, &indices, 1, 1, d).unwrap();

        assert_eq!(blocks.len(), 2);
        let total_entries: usize = blocks.iter().map(|b| b.len()).sum();
        assert_eq!(total_entries, 4, "D>K => all ring elements nonzero");

        for block in &blocks {
            for entry in block {
                assert_eq!(
                    entry.nonzero_coeffs.len(),
                    2,
                    "D=2K => 2 nonzero coeffs per ring element"
                );
            }
        }
    }

    #[test]
    fn map_onehot_rejects_non_divisible() {
        let result = map_onehot_to_sparse_blocks(3, &[Some(0), Some(1)], 0, 1, 4);
        assert!(result.is_err());
    }
}
