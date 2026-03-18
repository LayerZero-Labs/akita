//! One-hot commitment path for regular one-hot ring elements.
//!
//! Exploits the sparsity of one-hot witnesses (coefficients in {0,1}) to
//! eliminate all inner ring multiplications. The inner Ajtai `t = A * s`
//! reduces to summing selected columns of `A` with negacyclic rotations.

use crate::algebra::fields::{HasAdditivePacking, PackedAdditive};
use crate::algebra::ring::{CyclotomicRing, PackedNegacyclicRing};
use crate::error::HachiError;
use crate::protocol::commitment::utils::flat_matrix::RingMatrixView;
use crate::protocol::hachi_poly_ops::OneHotIndex;
use crate::{CanonicalField, FieldCore};

/// Describes a nonzero ring element within one block of the commitment layout.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseBlockEntry {
    /// Position within the block (0..2^M).
    pub pos_in_block: usize,
    /// Coefficient indices that are 1 within this ring element.
    pub nonzero_coeffs: Vec<usize>,
}

/// Flat sparse-block layout for packed one-hot kernels.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PackedSparseBlockLayout {
    positions: Vec<usize>,
    coeff_offsets: Vec<usize>,
    coeffs: Vec<usize>,
}

/// Borrowed sparse entry view over [`PackedSparseBlockLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedSparseEntryRef<'a> {
    /// Position within the block (0..2^M).
    pub pos_in_block: usize,
    /// Coefficient indices that are 1 within this ring element.
    pub nonzero_coeffs: &'a [usize],
}

impl PackedSparseBlockLayout {
    #[inline]
    pub(crate) fn new() -> Self {
        Self::with_capacity(0, 0)
    }

    #[inline]
    pub(crate) fn with_capacity(entries: usize, coeffs: usize) -> Self {
        Self {
            positions: Vec::with_capacity(entries),
            coeff_offsets: {
                let mut offsets = Vec::with_capacity(entries + 1);
                offsets.push(0);
                offsets
            },
            coeffs: Vec::with_capacity(coeffs),
        }
    }

    /// Build the flat layout used by packed one-hot kernels.
    #[cfg(test)]
    pub(crate) fn from_entries(entries: &[SparseBlockEntry]) -> Self {
        let mut positions = Vec::with_capacity(entries.len());
        let mut coeff_offsets = Vec::with_capacity(entries.len() + 1);
        let mut coeffs = Vec::new();
        coeff_offsets.push(0);
        for entry in entries {
            positions.push(entry.pos_in_block);
            coeffs.extend_from_slice(&entry.nonzero_coeffs);
            coeff_offsets.push(coeffs.len());
        }
        Self {
            positions,
            coeff_offsets,
            coeffs,
        }
    }

    #[inline]
    fn push_entry_from_vec(&mut self, pos_in_block: usize, coeffs: &mut Vec<usize>) {
        if self.coeff_offsets.is_empty() {
            self.coeff_offsets.push(0);
        }
        self.positions.push(pos_in_block);
        self.coeffs.append(coeffs);
        self.coeff_offsets.push(self.coeffs.len());
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    #[inline]
    fn entry(&self, idx: usize) -> PackedSparseEntryRef<'_> {
        let start = self.coeff_offsets[idx];
        let end = self.coeff_offsets[idx + 1];
        PackedSparseEntryRef {
            pos_in_block: self.positions[idx],
            nonzero_coeffs: &self.coeffs[start..end],
        }
    }

    #[inline]
    pub(crate) fn entries(&self) -> impl Iterator<Item = PackedSparseEntryRef<'_>> + '_ {
        (0..self.positions.len()).map(|idx| self.entry(idx))
    }
}

pub(crate) fn validate_onehot_sparse_shape(
    onehot_k: usize,
    num_chunks: usize,
    r: usize,
    m: usize,
    d: usize,
) -> Result<(usize, usize), HachiError> {
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

    Ok((num_blocks, block_len))
}

pub(crate) fn validate_onehot_indices<I: OneHotIndex>(
    onehot_k: usize,
    indices: &[Option<I>],
) -> Result<(), HachiError> {
    for (chunk_idx, opt) in indices.iter().enumerate() {
        let Some(&idx_raw) = opt.as_ref() else {
            continue;
        };
        let idx = idx_raw.as_usize();
        if idx >= onehot_k {
            return Err(HachiError::InvalidInput(format!(
                "index {idx} out of range for chunk size K={onehot_k} at position {chunk_idx}"
            )));
        }
    }
    Ok(())
}

#[inline]
fn onehot_block_field_range(block_idx: usize, block_len: usize, d: usize) -> (usize, usize) {
    let field_start = block_idx * block_len * d;
    let field_end = field_start + block_len * d;
    (field_start, field_end)
}

#[inline]
fn onehot_block_chunk_range(
    num_chunks: usize,
    onehot_k: usize,
    block_idx: usize,
    block_len: usize,
    d: usize,
) -> (usize, usize, usize, usize) {
    let (field_start, field_end) = onehot_block_field_range(block_idx, block_len, d);
    let start_chunk = field_start / onehot_k;
    let end_chunk = field_end.div_ceil(onehot_k).min(num_chunks);
    (field_start, field_end, start_chunk, end_chunk)
}

pub(crate) fn packed_sparse_block_for_block<I: OneHotIndex>(
    onehot_k: usize,
    indices: &[Option<I>],
    block_idx: usize,
    block_len: usize,
    d: usize,
) -> Result<PackedSparseBlockLayout, HachiError> {
    let (field_start, field_end, start_chunk, end_chunk) =
        onehot_block_chunk_range(indices.len(), onehot_k, block_idx, block_len, d);
    let candidate_chunks = end_chunk.saturating_sub(start_chunk);
    let mut block =
        PackedSparseBlockLayout::with_capacity(candidate_chunks.min(block_len), candidate_chunks);
    let mut current_pos: Option<usize> = None;
    let mut current_coeffs = Vec::new();

    let mut flush_current = |pos_in_block: usize, coeffs: &mut Vec<usize>| {
        block.push_entry_from_vec(pos_in_block, coeffs);
    };

    for (local_chunk_idx, opt) in indices[start_chunk..end_chunk].iter().enumerate() {
        let chunk_idx = start_chunk + local_chunk_idx;
        let Some(&idx_raw) = opt.as_ref() else {
            continue;
        };
        let idx = idx_raw.as_usize();
        if idx >= onehot_k {
            return Err(HachiError::InvalidInput(format!(
                "index {idx} out of range for chunk size K={onehot_k} at position {chunk_idx}"
            )));
        }

        let field_pos = chunk_idx * onehot_k + idx;
        if field_pos < field_start || field_pos >= field_end {
            continue;
        }

        let local_field_pos = field_pos - field_start;
        let pos_in_block = local_field_pos / d;
        let coeff_idx = local_field_pos % d;

        match current_pos {
            Some(existing_pos) if existing_pos == pos_in_block => current_coeffs.push(coeff_idx),
            Some(existing_pos) => {
                flush_current(existing_pos, &mut current_coeffs);
                current_pos = Some(pos_in_block);
                current_coeffs.push(coeff_idx);
            }
            None => {
                current_pos = Some(pos_in_block);
                current_coeffs.push(coeff_idx);
            }
        }
    }

    if let Some(existing_pos) = current_pos {
        flush_current(existing_pos, &mut current_coeffs);
    }

    Ok(block)
}

pub(crate) fn map_onehot_to_packed_sparse_blocks<I: OneHotIndex>(
    onehot_k: usize,
    indices: &[Option<I>],
    r: usize,
    m: usize,
    d: usize,
) -> Result<Vec<PackedSparseBlockLayout>, HachiError> {
    let (num_blocks, block_len) = validate_onehot_sparse_shape(onehot_k, indices.len(), r, m, d)?;
    validate_onehot_indices(onehot_k, indices)?;
    let mut blocks = vec![PackedSparseBlockLayout::new(); num_blocks];

    let mut current_ring_elem_idx: Option<usize> = None;
    let mut current_coeffs = Vec::new();

    let mut flush_current = |ring_elem_idx: usize, coeffs: &mut Vec<usize>| {
        let block_idx = ring_elem_idx / block_len;
        let pos_in_block = ring_elem_idx % block_len;
        blocks[block_idx].push_entry_from_vec(pos_in_block, coeffs);
    };

    for (chunk_idx, opt) in indices.iter().enumerate() {
        let Some(&idx_raw) = opt.as_ref() else {
            continue;
        };
        let idx = idx_raw.as_usize();

        let field_pos = chunk_idx * onehot_k + idx;
        let ring_elem_idx = field_pos / d;
        let coeff_idx = field_pos % d;

        match current_ring_elem_idx {
            Some(current_idx) if current_idx == ring_elem_idx => current_coeffs.push(coeff_idx),
            Some(current_idx) => {
                flush_current(current_idx, &mut current_coeffs);
                current_ring_elem_idx = Some(ring_elem_idx);
                current_coeffs.push(coeff_idx);
            }
            None => {
                current_ring_elem_idx = Some(ring_elem_idx);
                current_coeffs.push(coeff_idx);
            }
        }
    }

    if let Some(current_idx) = current_ring_elem_idx {
        flush_current(current_idx, &mut current_coeffs);
    }

    Ok(blocks)
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
pub fn map_onehot_to_sparse_blocks<I: OneHotIndex>(
    onehot_k: usize,
    indices: &[Option<I>],
    r: usize,
    m: usize,
    d: usize,
) -> Result<Vec<Vec<SparseBlockEntry>>, HachiError> {
    let packed_blocks = map_onehot_to_packed_sparse_blocks(onehot_k, indices, r, m, d)?;
    Ok(packed_blocks
        .into_iter()
        .map(|block| {
            block
                .entries()
                .map(|entry| SparseBlockEntry {
                    pos_in_block: entry.pos_in_block,
                    nonzero_coeffs: entry.nonzero_coeffs.to_vec(),
                })
                .collect()
        })
        .collect())
}

/// Sparse inner Ajtai: compute `t = A * s` for a one-hot block.
///
/// Instead of materializing the full decomposed vector `s` and doing a dense
/// matvec, we accumulate only the nonzero contributions using fused
/// shift-accumulate (no intermediate temporaries):
///
/// ```text
/// t[a] += A[a][entry.pos * num_digits] * (X^{k_1} + X^{k_2} + ...)
/// ```
#[cfg(test)]
#[allow(non_snake_case)]
pub(crate) fn inner_ajtai_onehot_t_only<F: FieldCore + CanonicalField, const D: usize>(
    A: &[Vec<CyclotomicRing<F, D>>],
    sparse_entries: &[SparseBlockEntry],
    _block_len: usize,
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>> {
    let n_a = A.len();

    let mut t = vec![CyclotomicRing::<F, D>::zero(); n_a];
    for entry in sparse_entries {
        let col = entry.pos_in_block * num_digits;
        for a in 0..n_a {
            A[a][col].mul_by_monomial_sum_into(&mut t[a], &entry.nonzero_coeffs);
        }
    }

    t
}

/// Wide-accumulator variant of [`inner_ajtai_onehot_t_only`].
///
/// Accumulates into `WideCyclotomicRing<W, D>` (carry-free i32 additions),
/// flushing into reduced ring totals when the signed-add budget is exhausted.
/// This avoids per-addition modular reduction while keeping overflow explicit.
fn flush_onehot_wide_chunk<F, P, const D: usize>(
    reduced: &mut [CyclotomicRing<F, D>],
    wide_chunk: &mut [PackedNegacyclicRing<P, D>],
) where
    F: FieldCore + CanonicalField + HasAdditivePacking,
    P: PackedAdditive<Scalar = F::AdditiveWide>,
{
    for (dst, wide) in reduced.iter_mut().zip(wide_chunk.iter_mut()) {
        *dst += wide.reduce();
        *wide = PackedNegacyclicRing::zero();
    }
}

fn inner_ajtai_onehot_wide_with_budget<F, P, const D: usize>(
    a: &RingMatrixView<'_, F, D>,
    sparse_entries: &PackedSparseBlockLayout,
    _block_len: usize,
    num_digits: usize,
    budget: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasAdditivePacking,
    P: PackedAdditive<Scalar = F::AdditiveWide>,
{
    assert!(budget > 0, "budget must be positive");
    let n_a = a.num_rows();
    let mut reduced = vec![CyclotomicRing::<F, D>::zero(); n_a];
    let mut wide_chunk = vec![PackedNegacyclicRing::<P, D>::zero(); n_a];
    let mut remaining_budget = budget;

    for entry in sparse_entries.entries() {
        let col = entry.pos_in_block * num_digits;
        let mut consumed = 0usize;
        while consumed < entry.nonzero_coeffs.len() {
            if remaining_budget == 0 {
                flush_onehot_wide_chunk::<F, P, D>(&mut reduced, &mut wide_chunk);
                remaining_budget = budget;
            }
            let take = remaining_budget.min(entry.nonzero_coeffs.len() - consumed);
            let coeff_chunk = &entry.nonzero_coeffs[consumed..consumed + take];
            for (a_idx, t_w) in wide_chunk.iter_mut().enumerate() {
                let a_wide = PackedNegacyclicRing::<P, D>::from_ring(&a.row(a_idx)[col]);
                a_wide.mul_by_monomial_sum_into(t_w, coeff_chunk);
            }
            consumed += take;
            remaining_budget -= take;
        }
    }

    if remaining_budget != budget {
        flush_onehot_wide_chunk::<F, P, D>(&mut reduced, &mut wide_chunk);
    }

    reduced
}

#[allow(non_snake_case)]
pub(crate) fn inner_ajtai_onehot_wide<F, const D: usize>(
    A: &RingMatrixView<'_, F, D>,
    sparse_entries: &PackedSparseBlockLayout,
    _block_len: usize,
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasAdditivePacking,
{
    inner_ajtai_onehot_wide_with_budget::<F, F::AdditivePacking, D>(
        A,
        sparse_entries,
        _block_len,
        num_digits,
        F::ADDITIVE_WIDE_HEADROOM,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::{Fp64, HasAdditiveWide, NoAdditivePacking, Prime128M8M4M1M0};
    use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn map_onehot_k_gt_d() {
        // K=16, D=4, T=2 chunks => 32 field elements => 8 ring elements
        // R=1 (2 blocks), M=2 (4 per block) => 8 ring elements total
        let k = 16;
        let d = 4;
        let indices: Vec<Option<u32>> = vec![Some(3), Some(10)];
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
        let indices: Vec<Option<u32>> = vec![Some(0), Some(2), Some(3), Some(1)];
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
        let indices: Vec<Option<u32>> = vec![
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
        let result = map_onehot_to_sparse_blocks(3, &[Some(0usize), Some(1)], 0, 1, 4);
        assert!(result.is_err());
    }

    #[test]
    fn wide_matches_reference() {
        type F = Fp64<4294967197>;
        const D: usize = 64;

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let n_a = 3;
        let block_len = 4;
        let num_digits = 5;
        let a_matrix: Vec<Vec<CyclotomicRing<F, D>>> = (0..n_a)
            .map(|_| {
                (0..block_len * num_digits)
                    .map(|_| CyclotomicRing::random(&mut rng))
                    .collect()
            })
            .collect();

        let entries = vec![
            SparseBlockEntry {
                pos_in_block: 0,
                nonzero_coeffs: vec![1, 7, 15],
            },
            SparseBlockEntry {
                pos_in_block: 2,
                nonzero_coeffs: vec![0, 63],
            },
        ];

        let a_flat = FlatMatrix::from_ring_matrix(&a_matrix);
        let a_view = a_flat.view::<D>();
        let ref_result = inner_ajtai_onehot_t_only(&a_matrix, &entries, block_len, num_digits);
        let packed_entries = PackedSparseBlockLayout::from_entries(&entries);
        let wide_result = inner_ajtai_onehot_wide(&a_view, &packed_entries, block_len, num_digits);

        assert_eq!(ref_result.len(), wide_result.len());
        for (r, w) in ref_result.iter().zip(wide_result.iter()) {
            assert_eq!(r, w, "wide result must match reference");
        }
    }

    #[test]
    fn wide_matches_reference_fp128() {
        type F = Prime128M8M4M1M0;
        const D: usize = 64;

        let mut rng = StdRng::seed_from_u64(0xcafe_1234);
        let n_a = 2;
        let block_len = 2;
        let num_digits = 3;
        let a_matrix: Vec<Vec<CyclotomicRing<F, D>>> = (0..n_a)
            .map(|_| {
                (0..block_len * num_digits)
                    .map(|_| CyclotomicRing::random(&mut rng))
                    .collect()
            })
            .collect();

        let entries = vec![
            SparseBlockEntry {
                pos_in_block: 0,
                nonzero_coeffs: vec![0, 5, 32, 63],
            },
            SparseBlockEntry {
                pos_in_block: 1,
                nonzero_coeffs: vec![10],
            },
        ];

        let a_flat = FlatMatrix::from_ring_matrix(&a_matrix);
        let a_view = a_flat.view::<D>();
        let ref_result = inner_ajtai_onehot_t_only(&a_matrix, &entries, block_len, num_digits);
        let packed_entries = PackedSparseBlockLayout::from_entries(&entries);
        let wide_result = inner_ajtai_onehot_wide(&a_view, &packed_entries, block_len, num_digits);

        assert_eq!(ref_result.len(), wide_result.len());
        for (r, w) in ref_result.iter().zip(wide_result.iter()) {
            assert_eq!(r, w, "wide result must match reference (Fp128)");
        }
    }

    #[test]
    fn wide_backend_parity_fp128() {
        type F = Prime128M8M4M1M0;
        const D: usize = 64;

        let mut rng = StdRng::seed_from_u64(0xabcd_1234);
        let n_a = 2;
        let block_len = 4;
        let num_digits = 3;
        let a_matrix: Vec<Vec<CyclotomicRing<F, D>>> = (0..n_a)
            .map(|_| {
                (0..block_len * num_digits)
                    .map(|_| CyclotomicRing::random(&mut rng))
                    .collect()
            })
            .collect();
        let entries = vec![
            SparseBlockEntry {
                pos_in_block: 0,
                nonzero_coeffs: vec![1, 9, 17],
            },
            SparseBlockEntry {
                pos_in_block: 2,
                nonzero_coeffs: vec![0, 15, 31, 63],
            },
        ];

        let a_flat = FlatMatrix::from_ring_matrix(&a_matrix);
        let a_view = a_flat.view::<D>();
        let packed_entries = PackedSparseBlockLayout::from_entries(&entries);
        let packed = inner_ajtai_onehot_wide(&a_view, &packed_entries, block_len, num_digits);
        let scalar = inner_ajtai_onehot_wide_with_budget::<
            F,
            NoAdditivePacking<<F as HasAdditiveWide>::AdditiveWide>,
            D,
        >(
            &a_view,
            &packed_entries,
            block_len,
            num_digits,
            <F as HasAdditiveWide>::ADDITIVE_WIDE_HEADROOM,
        );
        assert_eq!(packed, scalar);
    }

    #[test]
    fn wide_matches_reference_fp128_forced_flush() {
        type F = Prime128M8M4M1M0;
        const D: usize = 64;

        let mut rng = StdRng::seed_from_u64(0xface_2468);
        let n_a = 2;
        let block_len = 2;
        let num_digits = 3;
        let a_matrix: Vec<Vec<CyclotomicRing<F, D>>> = (0..n_a)
            .map(|_| {
                (0..block_len * num_digits)
                    .map(|_| CyclotomicRing::random(&mut rng))
                    .collect()
            })
            .collect();

        let entries = vec![
            SparseBlockEntry {
                pos_in_block: 0,
                nonzero_coeffs: (0..11).collect(),
            },
            SparseBlockEntry {
                pos_in_block: 1,
                nonzero_coeffs: vec![3, 9, 17, 31, 45, 63],
            },
        ];

        let a_flat = FlatMatrix::from_ring_matrix(&a_matrix);
        let a_view = a_flat.view::<D>();
        let ref_result = inner_ajtai_onehot_t_only(&a_matrix, &entries, block_len, num_digits);
        let packed_entries = PackedSparseBlockLayout::from_entries(&entries);
        let wide_result = inner_ajtai_onehot_wide_with_budget::<
            F,
            <F as HasAdditivePacking>::AdditivePacking,
            D,
        >(&a_view, &packed_entries, block_len, num_digits, 4);

        assert_eq!(ref_result.len(), wide_result.len());
        for (r, w) in ref_result.iter().zip(wide_result.iter()) {
            assert_eq!(r, w, "forced-flush result must match reference (Fp128)");
        }
    }
}
