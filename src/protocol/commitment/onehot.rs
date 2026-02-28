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
pub(crate) struct SparseBlockEntry {
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
pub(crate) fn map_onehot_to_sparse_blocks(
    onehot_k: usize,
    indices: &[usize],
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
    for (c, &idx) in indices.iter().enumerate() {
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

    // Distribute into blocks using the same interleaved layout as commit_coeffs:
    //   block_idx = ring_elem_idx % num_blocks
    //   pos_in_block = ring_elem_idx / num_blocks
    // (equivalently: flat_idx = (pos_in_block << R) + block_idx)
    let mut blocks: Vec<Vec<SparseBlockEntry>> = vec![Vec::new(); num_blocks];
    for (ring_elem_idx, nonzero_coeffs) in ring_elem_map {
        let block_idx = ring_elem_idx & (num_blocks - 1);
        let pos_in_block = ring_elem_idx >> r;
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
///
/// Also returns `s` (densely materialized) for the opening proof hint.
#[allow(non_snake_case)]
pub(crate) fn inner_ajtai_onehot<F: FieldCore + CanonicalField, const D: usize>(
    A: &[Vec<CyclotomicRing<F, D>>],
    sparse_entries: &[SparseBlockEntry],
    block_len: usize,
    delta: usize,
) -> (Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>) {
    let n_a = A.len();
    let inner_width = block_len * delta;

    // Build s: mostly zeros, with level-0 entries for nonzero ring elements.
    let mut s = vec![CyclotomicRing::<F, D>::zero(); inner_width];
    for entry in sparse_entries {
        let mut coeffs = [F::zero(); D];
        for &ci in &entry.nonzero_coeffs {
            coeffs[ci] = F::one();
        }
        s[entry.pos_in_block * delta] = CyclotomicRing::from_coefficients(coeffs);
    }

    // Compute t[a] = sum over nonzero entries of A[a][pos*delta] * f_j,
    // where f_j is the monomial sum at that position.
    let mut t = vec![CyclotomicRing::<F, D>::zero(); n_a];
    for entry in sparse_entries {
        let col = entry.pos_in_block * delta;
        for a in 0..n_a {
            t[a] += A[a][col].mul_by_monomial_sum(&entry.nonzero_coeffs);
        }
    }

    (t, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::F;

    #[test]
    fn map_onehot_k_gt_d() {
        // K=16, D=4, T=2 chunks => 32 field elements => 8 ring elements
        // R=1 (2 blocks), M=2 (4 per block) => 8 ring elements total
        let k = 16;
        let d = 4;
        let indices = vec![3, 10];
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
        let indices = vec![0, 2, 3, 1];
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
        let indices = vec![0, 2, 3, 1, 0, 0, 3, 3];
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
        let result = map_onehot_to_sparse_blocks(3, &[0, 1], 0, 1, 4);
        assert!(result.is_err());
    }

    #[test]
    fn inner_ajtai_onehot_single_monomial() {
        const D: usize = 4;
        type R = CyclotomicRing<F, D>;

        // A is 2x4 (N_A=2, inner_width = block_len * delta = 2 * 2 = 4)
        let a: Vec<Vec<R>> = vec![
            vec![
                R::from_coefficients(std::array::from_fn(|i| F::from_u64((i + 1) as u64))),
                R::from_coefficients(std::array::from_fn(|i| F::from_u64((i + 10) as u64))),
                R::from_coefficients(std::array::from_fn(|i| F::from_u64((i + 20) as u64))),
                R::from_coefficients(std::array::from_fn(|i| F::from_u64((i + 30) as u64))),
            ],
            vec![
                R::from_coefficients(std::array::from_fn(|i| F::from_u64((i + 5) as u64))),
                R::from_coefficients(std::array::from_fn(|i| F::from_u64((i + 15) as u64))),
                R::from_coefficients(std::array::from_fn(|i| F::from_u64((i + 25) as u64))),
                R::from_coefficients(std::array::from_fn(|i| F::from_u64((i + 35) as u64))),
            ],
        ];

        // One nonzero entry at pos=1, coefficient index 2 => monomial X^2
        let entries = vec![SparseBlockEntry {
            pos_in_block: 1,
            nonzero_coeffs: vec![2],
        }];

        let (t, s) = inner_ajtai_onehot(&a, &entries, 2, 2);

        // t[row] should equal A[row][1*2] * X^2 = A[row][2].negacyclic_shift(2)
        for row in 0..2 {
            let expected = a[row][2].negacyclic_shift(2);
            assert_eq!(t[row], expected);
        }

        // s should have a nonzero entry at position 1*2 = 2
        assert_eq!(s[2].coefficients()[2], F::one());
        assert!(s[0] == R::zero());
        assert!(s[1] == R::zero());
        assert!(s[3] == R::zero());
    }
}
