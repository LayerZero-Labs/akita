//! One-hot commitment path for regular one-hot ring elements.
//!
//! Exploits the sparsity of one-hot witnesses (coefficients in {0,1}) to
//! eliminate all inner ring multiplications. The inner Ajtai `t = A * s`
//! reduces to summing selected columns of `A` with negacyclic rotations.

use std::collections::BTreeMap;

use crate::algebra::fields::wide::{HasWide, ReduceTo};
use crate::algebra::ring::{CyclotomicRing, WideCyclotomicRing};
use crate::error::HachiError;
use crate::protocol::commitment::utils::flat_matrix::RingMatrixView;
use crate::protocol::hachi_poly_ops::OneHotIndex;
use crate::{AdditiveGroup, CanonicalField, FieldCore};

/// Describes a nonzero ring element within one block of the commitment layout.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseBlockEntry {
    /// Position within the block (0..2^M).
    pub pos_in_block: usize,
    /// Coefficient indices that are 1 within this ring element.
    pub nonzero_coeffs: Vec<usize>,
}

/// Compact regular one-hot entry used when each nonzero ring element carries a
/// single hot coefficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegularOneHotEntry {
    pos_in_block: u32,
    coeff_idx: u16,
}

/// Flat per-block storage: all non-zero entries laid out in one contiguous
/// buffer, keyed by block index via a tiny offsets array.
///
/// Compared to the previous `Vec<Vec<Entry>>`:
///   - Single heap allocation for entries instead of one per block.
///   - Single tiny allocation for block offsets (`(num_blocks + 1) * 4 B`).
///   - Block `i` entries: `&entries[offsets[i] as usize..offsets[i + 1] as usize]`.
///
/// Entries are sorted by `(block_idx, pos_in_block)` so the per-block slice
/// is ascending in `pos_in_block`, matching the invariant the accumulators
/// rely on (they do `partition_point` on `pos_in_block`).
#[derive(Debug, Clone)]
pub struct FlatBlocks<E> {
    entries: Vec<E>,
    /// `len == num_blocks + 1`, `offsets[0] == 0`, `offsets[num_blocks] == entries.len()`.
    offsets: Vec<u32>,
}

impl<E> FlatBlocks<E> {
    /// Build a flat block layout from a pre-bucketed `Vec<Vec<E>>`.
    ///
    /// Used by tests and by callers that already have bucketed storage;
    /// the production mapping functions build the flat form directly without
    /// going through this path.
    #[cfg(test)]
    pub(crate) fn from_buckets(buckets: Vec<Vec<E>>) -> Self {
        let num_blocks = buckets.len();
        let mut offsets = Vec::with_capacity(num_blocks + 1);
        let total: usize = buckets.iter().map(Vec::len).sum();
        let mut entries = Vec::with_capacity(total);
        offsets.push(0);
        for mut bucket in buckets {
            entries.append(&mut bucket);
            // `entries.len()` is bounded by `total` which was `sum(Vec::len)`;
            // both fit in `u32` on all supported layouts.
            offsets.push(u32::try_from(entries.len()).expect("flat block offset overflows u32"));
        }
        Self { entries, offsets }
    }

    /// Number of blocks.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Total number of stored non-zero entries across all blocks.
    #[inline]
    pub fn total_entries(&self) -> usize {
        self.entries.len()
    }

    /// Slice of entries for block `i`.
    #[inline]
    pub fn block(&self, i: usize) -> &[E] {
        let lo = self.offsets[i] as usize;
        let hi = self.offsets[i + 1] as usize;
        // SAFETY-equivalent: invariant `offsets` is monotonic non-decreasing
        // and bounded by `entries.len()`, enforced in `from_buckets`.
        &self.entries[lo..hi]
    }

    /// Iterator over per-block slices in ascending block order.
    pub fn iter_blocks(&self) -> FlatBlocksIter<'_, E> {
        FlatBlocksIter {
            entries: &self.entries,
            offsets: &self.offsets,
            cursor: 0,
        }
    }
}

/// Iterator yielding per-block entry slices from a [`FlatBlocks`].
pub struct FlatBlocksIter<'a, E> {
    entries: &'a [E],
    offsets: &'a [u32],
    cursor: usize,
}

impl<'a, E> Iterator for FlatBlocksIter<'a, E> {
    type Item = &'a [E];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor + 1 >= self.offsets.len() {
            return None;
        }
        let lo = self.offsets[self.cursor] as usize;
        let hi = self.offsets[self.cursor + 1] as usize;
        self.cursor += 1;
        Some(&self.entries[lo..hi])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.offsets.len() - 1 - self.cursor;
        (remaining, Some(remaining))
    }
}

impl<'a, E> ExactSizeIterator for FlatBlocksIter<'a, E> {}

/// Shape-agnostic view into per-block entries.
///
/// All hot kernels iterate `for block_idx in 0..num_blocks { view.block(block_idx) }`
/// so they can consume either the owned flat layout or a borrowed slice view
/// (used by the batched decompose-fold path, which concatenates multiple
/// polynomials' block slices).
pub(crate) trait BlockView<E> {
    fn num_blocks(&self) -> usize;
    fn block(&self, i: usize) -> &[E];
}

impl<E> BlockView<E> for FlatBlocks<E> {
    #[inline]
    fn num_blocks(&self) -> usize {
        FlatBlocks::num_blocks(self)
    }
    #[inline]
    fn block(&self, i: usize) -> &[E] {
        FlatBlocks::block(self, i)
    }
}

/// Slice-of-slices view used by the batched path which concatenates block
/// slices from several polynomials.
impl<E> BlockView<E> for [&[E]] {
    #[inline]
    fn num_blocks(&self) -> usize {
        self.len()
    }
    #[inline]
    fn block(&self, i: usize) -> &[E] {
        self[i]
    }
}

impl<E> BlockView<E> for Vec<&[E]> {
    #[inline]
    fn num_blocks(&self) -> usize {
        self.len()
    }
    #[inline]
    fn block(&self, i: usize) -> &[E] {
        self[i]
    }
}

/// Flat regular one-hot blocks.
pub type FlatRegularBlocks = FlatBlocks<RegularOneHotEntry>;
/// Flat general one-hot blocks.
pub type FlatSparseBlocks = FlatBlocks<SparseBlockEntry>;

impl RegularOneHotEntry {
    #[inline]
    /// Construct a compact regular one-hot entry.
    ///
    /// # Errors
    ///
    /// Returns an error if either the block position or coefficient index does
    /// not fit in the compact storage format.
    pub fn new(pos_in_block: usize, coeff_idx: usize) -> Result<Self, HachiError> {
        Ok(Self {
            pos_in_block: u32::try_from(pos_in_block).map_err(|_| {
                HachiError::InvalidInput(format!(
                    "regular one-hot block position {pos_in_block} does not fit in u32"
                ))
            })?,
            coeff_idx: u16::try_from(coeff_idx).map_err(|_| {
                HachiError::InvalidInput(format!(
                    "regular one-hot coefficient index {coeff_idx} does not fit in u16"
                ))
            })?,
        })
    }

    #[inline]
    /// Position within the block (0..2^M).
    pub fn pos_in_block(self) -> usize {
        self.pos_in_block as usize
    }

    #[inline]
    /// Hot coefficient index inside the ring element.
    pub fn coeff_idx(self) -> usize {
        self.coeff_idx as usize
    }
}

/// Map a regular one-hot witness to sparse ring block entries, stored in the
/// flat layout used by the hot accumulator and column-sweep kernels.
///
/// - `onehot_k`: chunk size K. The witness has T chunks of K field elements,
///   each chunk containing exactly one 1.
/// - `indices`: length-T slice where `indices[c]` is the hot position in
///   chunk `c` (must be in `[0, K)`).
/// - `block_len`: number of ring elements per block (must be a power of two
///   that divides the total ring-element count).
/// - `D`: ring degree (const generic on caller side, passed as runtime here).
///
/// Returns a [`FlatSparseBlocks`] with `num_blocks = total_ring_elems /
/// block_len` blocks and all non-zero entries in one contiguous buffer.
///
/// # Errors
///
/// Returns an error if K and D are not "nicely matched" (one must divide
/// the other), if any index is out of range, or if `block_len` does not tile
/// the ring-element count.
pub fn map_onehot_to_sparse_blocks<I: OneHotIndex>(
    onehot_k: usize,
    indices: &[Option<I>],
    block_len: usize,
    d: usize,
) -> Result<FlatSparseBlocks, HachiError> {
    if onehot_k == 0 || d == 0 {
        return Err(HachiError::InvalidInput(
            "onehot_k and D must be nonzero".into(),
        ));
    }
    if !(onehot_k.is_multiple_of(d) || d.is_multiple_of(onehot_k)) {
        return Err(HachiError::InvalidInput(format!(
            "K={onehot_k} and D={d} must be nicely matched (one divides the other)"
        )));
    }
    if block_len == 0 || !block_len.is_power_of_two() {
        return Err(HachiError::InvalidInput(format!(
            "block_len={block_len} must be a nonzero power of two"
        )));
    }

    let num_chunks = indices.len();
    let total_field_elems = num_chunks
        .checked_mul(onehot_k)
        .ok_or_else(|| HachiError::InvalidInput("T*K overflow".into()))?;
    if !total_field_elems.is_multiple_of(d) {
        return Err(HachiError::InvalidInput(format!(
            "T*K={total_field_elems} is not divisible by D={d}"
        )));
    }
    let total_ring_elems = total_field_elems / d;
    if !total_ring_elems.is_multiple_of(block_len) {
        return Err(HachiError::InvalidSize {
            expected: total_ring_elems,
            actual: block_len,
        });
    }
    let num_blocks = total_ring_elems / block_len;

    let mut ring_elem_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (c, opt) in indices.iter().enumerate() {
        let Some(&idx_raw) = opt.as_ref() else {
            continue;
        };
        let idx = idx_raw.as_usize();
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
    // [i*block_len, (i+1)*block_len). `BTreeMap` iterates in ascending
    // `ring_elem_idx`, so per-block slices end up sorted by `pos_in_block`.
    let total_entries = ring_elem_map.len();
    let mut entries: Vec<SparseBlockEntry> = Vec::with_capacity(total_entries);
    let mut offsets: Vec<u32> = Vec::with_capacity(num_blocks + 1);
    offsets.push(0);
    let mut current_block = 0usize;
    for (ring_elem_idx, nonzero_coeffs) in ring_elem_map {
        let block_idx = ring_elem_idx / block_len;
        let pos_in_block = ring_elem_idx % block_len;
        while current_block < block_idx {
            offsets.push(u32::try_from(entries.len()).map_err(|_| {
                HachiError::InvalidInput("flat block offset overflows u32".to_string())
            })?);
            current_block += 1;
        }
        entries.push(SparseBlockEntry {
            pos_in_block,
            nonzero_coeffs,
        });
    }
    while current_block < num_blocks {
        offsets.push(u32::try_from(entries.len()).map_err(|_| {
            HachiError::InvalidInput("flat block offset overflows u32".to_string())
        })?);
        current_block += 1;
    }
    debug_assert_eq!(offsets.len(), num_blocks + 1);
    debug_assert_eq!(offsets[num_blocks] as usize, entries.len());

    Ok(FlatBlocks { entries, offsets })
}

/// Map a one-hot witness to compact regular block entries when each nonzero
/// ring element contains a single hot coefficient.
///
/// This applies to the common `K >= D` case, where each chunk spans one or
/// more ring elements but still contributes exactly one nonzero coefficient in
/// exactly one ring element.
///
/// `block_len` is the number of ring elements per block and must be a power of
/// two that divides the total ring-element count. The output is a
/// [`FlatRegularBlocks`] with `num_blocks = total_ring_elems / block_len`
/// blocks backed by a single contiguous entry buffer.
///
/// # Errors
///
/// Returns an error if the layout is incompatible with the compact regular
/// representation, any hot index is out of range, or `block_len` does not tile
/// the ring-element count.
pub fn map_onehot_to_regular_blocks<I: OneHotIndex>(
    onehot_k: usize,
    indices: &[Option<I>],
    block_len: usize,
    d: usize,
) -> Result<FlatRegularBlocks, HachiError> {
    if onehot_k == 0 || d == 0 {
        return Err(HachiError::InvalidInput(
            "onehot_k and D must be nonzero".into(),
        ));
    }
    if onehot_k < d || !onehot_k.is_multiple_of(d) {
        return Err(HachiError::InvalidInput(format!(
            "regular one-hot layout requires K >= D with K divisible by D, got K={onehot_k}, D={d}"
        )));
    }
    if block_len == 0 || !block_len.is_power_of_two() {
        return Err(HachiError::InvalidInput(format!(
            "block_len={block_len} must be a nonzero power of two"
        )));
    }

    let num_chunks = indices.len();
    let total_field_elems = num_chunks
        .checked_mul(onehot_k)
        .ok_or_else(|| HachiError::InvalidInput("T*K overflow".into()))?;
    if !total_field_elems.is_multiple_of(d) {
        return Err(HachiError::InvalidInput(format!(
            "T*K={total_field_elems} is not divisible by D={d}"
        )));
    }
    let total_ring_elems = total_field_elems / d;
    if !total_ring_elems.is_multiple_of(block_len) {
        return Err(HachiError::InvalidSize {
            expected: total_ring_elems,
            actual: block_len,
        });
    }
    let num_blocks = total_ring_elems / block_len;

    // In the regular layout each non-None chunk produces exactly one entry
    // at `ring_elem_idx = (c*K + idx) / D`. Because K is a multiple of D and
    // indices are processed in chunk order, the resulting stream of
    // `ring_elem_idx` values is monotonically non-decreasing, so we can
    // stream entries straight into a single flat buffer and emit block
    // boundaries as we cross them. No BTreeMap needed.
    let total_entries = indices.iter().filter(|opt| opt.is_some()).count();
    let mut entries: Vec<RegularOneHotEntry> = Vec::with_capacity(total_entries);
    let mut offsets: Vec<u32> = Vec::with_capacity(num_blocks + 1);
    offsets.push(0);
    let mut current_block = 0usize;

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

        let field_pos = chunk_idx
            .checked_mul(onehot_k)
            .and_then(|base| base.checked_add(idx))
            .ok_or_else(|| HachiError::InvalidInput("field position overflow".into()))?;
        let ring_elem_idx = field_pos / d;
        let coeff_idx = field_pos % d;
        let block_idx = ring_elem_idx / block_len;
        let pos_in_block = ring_elem_idx % block_len;
        debug_assert!(
            block_idx >= current_block,
            "regular onehot: entries must be non-decreasing in block index"
        );
        while current_block < block_idx {
            offsets.push(u32::try_from(entries.len()).map_err(|_| {
                HachiError::InvalidInput("flat block offset overflows u32".to_string())
            })?);
            current_block += 1;
        }
        entries.push(RegularOneHotEntry::new(pos_in_block, coeff_idx)?);
    }
    while current_block < num_blocks {
        offsets.push(u32::try_from(entries.len()).map_err(|_| {
            HachiError::InvalidInput("flat block offset overflows u32".to_string())
        })?);
        current_block += 1;
    }
    debug_assert_eq!(offsets.len(), num_blocks + 1);
    debug_assert_eq!(offsets[num_blocks] as usize, entries.len());

    Ok(FlatBlocks { entries, offsets })
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
/// then reduces once at the end. This avoids per-addition modular reduction.
#[allow(non_snake_case)]
pub(crate) fn inner_ajtai_onehot_wide<F, const D: usize>(
    A: &RingMatrixView<'_, F, D>,
    sparse_entries: &[SparseBlockEntry],
    _block_len: usize,
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let n_a = A.num_rows();
    let mut t_wide = vec![WideCyclotomicRing::<F::Wide, D>::zero(); n_a];

    for entry in sparse_entries {
        let col = entry.pos_in_block * num_digits;
        for (a_idx, t_w) in t_wide.iter_mut().enumerate() {
            let a_wide = WideCyclotomicRing::from_ring(&A.row(a_idx)[col]);
            a_wide.mul_by_monomial_sum_into(t_w, &entry.nonzero_coeffs);
        }
    }

    t_wide.into_iter().map(|w| w.reduce()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::{Fp64, Prime128Offset275};
    use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn map_onehot_k_gt_d() {
        // K=16, D=4, T=2 chunks => 32 field elements => 8 ring elements
        // block_len=4 => 2 blocks of 4 ring elements each.
        let k = 16;
        let d = 4;
        let indices: Vec<Option<u32>> = vec![Some(3), Some(10)];
        let blocks = map_onehot_to_sparse_blocks(k, &indices, 4, d).unwrap();

        assert_eq!(blocks.num_blocks(), 2);
        assert_eq!(blocks.total_entries(), 2, "T=2 nonzero ring elements");

        for block in blocks.iter_blocks() {
            for entry in block {
                assert_eq!(entry.nonzero_coeffs.len(), 1, "K>D => single monomial");
            }
        }
    }

    #[test]
    fn map_onehot_k_eq_d() {
        // K=4, D=4, T=4 chunks => 16 field elements => 4 ring elements
        // block_len=2 => 2 blocks of 2 ring elements each.
        let k = 4;
        let d = 4;
        let indices: Vec<Option<u32>> = vec![Some(0), Some(2), Some(3), Some(1)];
        let blocks = map_onehot_to_sparse_blocks(k, &indices, 2, d).unwrap();

        assert_eq!(blocks.num_blocks(), 2);
        assert_eq!(
            blocks.total_entries(),
            4,
            "K=D => every ring element is nonzero"
        );

        for block in blocks.iter_blocks() {
            for entry in block {
                assert_eq!(entry.nonzero_coeffs.len(), 1, "K=D => single monomial");
            }
        }
    }

    #[test]
    fn map_onehot_k_lt_d() {
        // K=4, D=8, T=8 chunks => 32 field elements => 4 ring elements
        // block_len=2 => 2 blocks of 2 ring elements each.
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
        let blocks = map_onehot_to_sparse_blocks(k, &indices, 2, d).unwrap();

        assert_eq!(blocks.num_blocks(), 2);
        assert_eq!(
            blocks.total_entries(),
            4,
            "D>K => all ring elements nonzero"
        );

        for block in blocks.iter_blocks() {
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
        let result = map_onehot_to_sparse_blocks(3, &[Some(0usize), Some(1)], 2, 4);
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

        let a_flat_elems: Vec<CyclotomicRing<F, D>> = a_matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let a_flat = FlatMatrix::from_ring_slice(&a_flat_elems);
        let a_view = a_flat.ring_view::<D>(n_a, block_len * num_digits);
        let ref_result = inner_ajtai_onehot_t_only(&a_matrix, &entries, block_len, num_digits);
        let wide_result = inner_ajtai_onehot_wide(&a_view, &entries, block_len, num_digits);

        assert_eq!(ref_result.len(), wide_result.len());
        for (r, w) in ref_result.iter().zip(wide_result.iter()) {
            assert_eq!(r, w, "wide result must match reference");
        }
    }

    #[test]
    fn wide_matches_reference_fp128() {
        type F = Prime128Offset275;
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

        let a_flat_elems: Vec<CyclotomicRing<F, D>> = a_matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let a_flat = FlatMatrix::from_ring_slice(&a_flat_elems);
        let a_view = a_flat.ring_view::<D>(n_a, block_len * num_digits);
        let ref_result = inner_ajtai_onehot_t_only(&a_matrix, &entries, block_len, num_digits);
        let wide_result = inner_ajtai_onehot_wide(&a_view, &entries, block_len, num_digits);

        assert_eq!(ref_result.len(), wide_result.len());
        for (r, w) in ref_result.iter().zip(wide_result.iter()) {
            assert_eq!(r, w, "wide result must match reference (Fp128)");
        }
    }
}
