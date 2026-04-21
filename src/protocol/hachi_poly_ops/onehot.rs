//! One-hot polynomial: sparse witness exploiting monomial structure.
//!
//! [`OneHotPoly`] implements [`HachiPolyOps`](super::HachiPolyOps) for
//! polynomials with at most one nonzero field element per chunk of size
//! `onehot_k`. All four operations exploit sparsity, avoiding inner ring
//! multiplications during commit and decomposing only nonzero monomials.
//!
//! Module contents:
//!   - [`OneHotIndex`] trait for position-index types, plus impls for
//!     `u8`/`u16`/`u32`/`usize`.
//!   - Sparse entry types ([`SparseBlockEntry`], [`RegularOneHotEntry`])
//!     and the contiguous flat layout ([`FlatBlocks`]) they are stored in.
//!     Kernels consume blocks via a plain `&[&[E]]` slice-of-slices so the
//!     single-polynomial and batched paths share a signature.
//!   - The mapping helpers [`map_onehot_to_sparse_blocks`] and
//!     [`map_onehot_to_regular_blocks`] that compile a witness into flat
//!     per-block storage.
//!   - [`OneHotPoly`] itself with its [`HachiPolyOps`](super::HachiPolyOps)
//!     impl.
//!   - The inner-Ajtai kernels ([`inner_ajtai_onehot_wide`] and the
//!     column-sweep variants) that turn those blocks into commitments.

use crate::algebra::fields::wide::{HasWide, ReduceTo};
use crate::algebra::ring::cyclotomic::WideCyclotomicRing;
use crate::algebra::ring::sparse_challenge::SparseChallenge;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::flat_matrix::{FlatMatrix, RingMatrixView};
use crate::protocol::commitment::utils::linear::decompose_rows_i8_into;
use crate::protocol::hachi_poly_ops::helpers::{
    build_decompose_fold_witness, regular_onehot_accumulate, sparse_onehot_accumulate,
};
use crate::protocol::hachi_poly_ops::{CommitInnerWitness, DecomposeFoldWitness, HachiPolyOps};
use crate::protocol::proof::{DirectWitnessProof, FlatDigitBlocks, FlatRingVec};
use crate::{AdditiveGroup, CanonicalField, FieldCore};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::OnceLock;

/// Types usable as one-hot position indices.
///
/// Implemented for `u8`, `u16`, `u32`, and `usize`.
pub trait OneHotIndex: Copy + Send + Sync + std::fmt::Debug + 'static {
    /// Convert to `usize` for indexing.
    fn as_usize(self) -> usize;
}

impl OneHotIndex for u8 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for u16 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for u32 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for usize {
    #[inline]
    fn as_usize(self) -> usize {
        self
    }
}

// =============================================================================
// Sparse entry types and flat per-block storage.
//
// These were the public surface of the previous `commitment/onehot` module;
// they are kept here so every one-hot building block (entry types, flat
// storage, mapping helpers, inner-Ajtai kernels, the polynomial itself) lives
// in one place.
// =============================================================================

/// Describes a nonzero ring element within one block of the commitment layout.
///
/// Storage mirrors [`RegularOneHotEntry`]: the block position fits in `u32`
/// and every coefficient index is `< D`, so they fit in `u16`. The fields
/// are private; all reads and construction go through the accessors below.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SparseBlockEntry {
    pos_in_block: u32,
    nonzero_coeffs: Vec<u16>,
}

impl SparseBlockEntry {
    /// Construct a compact sparse block entry.
    ///
    /// # Errors
    ///
    /// Returns an error if the block position does not fit in `u32` or any
    /// coefficient index does not fit in `u16`.
    pub(crate) fn new(pos_in_block: usize, nonzero_coeffs: Vec<usize>) -> Result<Self, HachiError> {
        let pos_in_block = u32::try_from(pos_in_block).map_err(|_| {
            HachiError::InvalidInput(format!(
                "sparse one-hot block position {pos_in_block} does not fit in u32"
            ))
        })?;
        let mut packed_coeffs: Vec<u16> = Vec::with_capacity(nonzero_coeffs.len());
        for ci in nonzero_coeffs {
            let ci_u16 = u16::try_from(ci).map_err(|_| {
                HachiError::InvalidInput(format!(
                    "sparse one-hot coefficient index {ci} does not fit in u16"
                ))
            })?;
            packed_coeffs.push(ci_u16);
        }
        Ok(Self {
            pos_in_block,
            nonzero_coeffs: packed_coeffs,
        })
    }

    #[inline]
    /// Position within the block (0..2^M).
    pub(crate) fn pos_in_block(&self) -> usize {
        self.pos_in_block as usize
    }

    #[inline]
    /// Hot coefficient indices within the ring element, packed as `u16`.
    pub(crate) fn nonzero_coeffs(&self) -> &[u16] {
        &self.nonzero_coeffs
    }
}

/// Compact regular one-hot entry used when each nonzero ring element carries a
/// single hot coefficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RegularOneHotEntry {
    pos_in_block: u32,
    coeff_idx: u16,
}

impl RegularOneHotEntry {
    #[inline]
    /// Construct a compact regular one-hot entry.
    ///
    /// # Errors
    ///
    /// Returns an error if either the block position or coefficient index does
    /// not fit in the compact storage format.
    pub(crate) fn new(pos_in_block: usize, coeff_idx: usize) -> Result<Self, HachiError> {
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
    pub(crate) fn pos_in_block(self) -> usize {
        self.pos_in_block as usize
    }

    #[inline]
    /// Hot coefficient index inside the ring element.
    pub(crate) fn coeff_idx(self) -> usize {
        self.coeff_idx as usize
    }
}

/// Flat per-block storage: all non-zero entries laid out in one contiguous
/// buffer, keyed by block index via a tiny offsets array.
///
/// Compared to the previous `Vec<Vec<Entry>>` layout:
///   - Single heap allocation for entries instead of one per block.
///   - Single tiny allocation for block offsets (`(num_blocks + 1) * 4 B`).
///   - Block `i` entries: `&entries[offsets[i] as usize..offsets[i + 1] as usize]`.
///
/// Entries are sorted by `(block_idx, pos_in_block)` so the per-block slice
/// is ascending in `pos_in_block`, matching the invariant the accumulators
/// rely on (they do `partition_point` on `pos_in_block`).
#[derive(Debug, Clone)]
pub(crate) struct FlatBlocks<E> {
    entries: Vec<E>,
    /// `len == num_blocks + 1`, `offsets[0] == 0`, `offsets[num_blocks] == entries.len()`.
    offsets: Vec<u32>,
}

impl<E> FlatBlocks<E> {
    /// Number of blocks.
    #[inline]
    pub(crate) fn num_blocks(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Total number of stored non-zero entries across all blocks.
    ///
    /// Only used from tests; kept crate-visible so test modules in sibling
    /// files can reach it without breaking the `FlatBlocks` encapsulation.
    #[cfg(test)]
    #[inline]
    pub(crate) fn total_entries(&self) -> usize {
        self.entries.len()
    }

    /// Slice of entries for block `i`.
    #[inline]
    pub(crate) fn block(&self, i: usize) -> &[E] {
        let lo = self.offsets[i] as usize;
        let hi = self.offsets[i + 1] as usize;
        // SAFETY-equivalent: `offsets` is monotonic non-decreasing and
        // bounded by `entries.len()`, enforced by the constructors that
        // produce `FlatBlocks` values (`map_onehot_to_regular_blocks`,
        // `map_onehot_to_sparse_blocks`, and the test-only
        // `test_helpers::from_buckets`).
        &self.entries[lo..hi]
    }

    /// Iterator over per-block slices in ascending block order.
    pub(crate) fn iter_blocks(&self) -> FlatBlocksIter<'_, E> {
        FlatBlocksIter {
            entries: &self.entries,
            offsets: &self.offsets,
            cursor: 0,
        }
    }
}

/// Iterator yielding per-block entry slices from a [`FlatBlocks`].
pub(crate) struct FlatBlocksIter<'a, E> {
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

/// Flat regular one-hot blocks.
pub(crate) type FlatRegularBlocks = FlatBlocks<RegularOneHotEntry>;
/// Flat general one-hot blocks.
pub(crate) type FlatSparseBlocks = FlatBlocks<SparseBlockEntry>;

/// Kind of K/D compatibility required by the caller.
///
/// The regular (single-hot-coefficient-per-ring-element) mapper needs the
/// stricter `K >= D && D | K` check, the general sparse mapper accepts
/// either side dividing the other.
#[derive(Copy, Clone)]
enum KdCompat {
    NicelyMatched,
    RegularLayout,
}

/// Shared preconditions for the two `map_onehot_to_*_blocks` mappers.
///
/// Returns the `num_blocks` that the flat layout will have, or an error if
/// any of the layout parameters are inconsistent.
fn validate_onehot_layout(
    onehot_k: usize,
    num_chunks: usize,
    block_len: usize,
    d: usize,
    kd_compat: KdCompat,
) -> Result<usize, HachiError> {
    if onehot_k == 0 || d == 0 {
        return Err(HachiError::InvalidInput(
            "onehot_k and D must be nonzero".into(),
        ));
    }
    match kd_compat {
        KdCompat::NicelyMatched => {
            if !(onehot_k.is_multiple_of(d) || d.is_multiple_of(onehot_k)) {
                return Err(HachiError::InvalidInput(format!(
                    "K={onehot_k} and D={d} must be nicely matched (one divides the other)"
                )));
            }
        }
        KdCompat::RegularLayout => {
            if onehot_k < d || !onehot_k.is_multiple_of(d) {
                return Err(HachiError::InvalidInput(format!(
                    "regular one-hot layout requires K >= D with K divisible by D, got K={onehot_k}, D={d}"
                )));
            }
        }
    }
    if block_len == 0 || !block_len.is_power_of_two() {
        return Err(HachiError::InvalidInput(format!(
            "block_len={block_len} must be a nonzero power of two"
        )));
    }
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
    Ok(total_ring_elems / block_len)
}

/// Push an offset row or return a descriptive error if the row count ever
/// exceeds `u32::MAX`.
#[inline]
fn push_offset(offsets: &mut Vec<u32>, len: usize) -> Result<(), HachiError> {
    let off = u32::try_from(len)
        .map_err(|_| HachiError::InvalidInput("flat block offset overflows u32".to_string()))?;
    offsets.push(off);
    Ok(())
}

/// Map a regular one-hot witness to sparse ring block entries, stored in the
/// flat layout used by the hot accumulator and column-sweep kernels.
///
/// - `onehot_k`: chunk size K. The witness has T chunks of K field elements,
///   each chunk containing exactly one 1.
/// - `indices`: length-T slice where `indices[c]` is the hot position in
///   chunk `c`, or [`INDICES_NONE`] for an all-zero chunk. Real positions
///   must be in `[0, K)`.
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
pub(crate) fn map_onehot_to_sparse_blocks(
    onehot_k: usize,
    indices: &[u32],
    block_len: usize,
    d: usize,
) -> Result<FlatSparseBlocks, HachiError> {
    let num_blocks = validate_onehot_layout(
        onehot_k,
        indices.len(),
        block_len,
        d,
        KdCompat::NicelyMatched,
    )?;

    let mut ring_elem_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (c, &packed) in indices.iter().enumerate() {
        if packed == INDICES_NONE {
            continue;
        }
        let idx = packed as usize;
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

    // Sequential block layout: block i = ring elements [i*block_len,
    // (i+1)*block_len). `BTreeMap` iterates in ascending `ring_elem_idx`,
    // so per-block slices end up sorted by `pos_in_block`.
    let total_entries = ring_elem_map.len();
    let mut entries: Vec<SparseBlockEntry> = Vec::with_capacity(total_entries);
    let mut offsets: Vec<u32> = Vec::with_capacity(num_blocks + 1);
    offsets.push(0);
    let mut current_block = 0usize;
    for (ring_elem_idx, nonzero_coeffs) in ring_elem_map {
        let block_idx = ring_elem_idx / block_len;
        let pos_in_block = ring_elem_idx % block_len;
        while current_block < block_idx {
            push_offset(&mut offsets, entries.len())?;
            current_block += 1;
        }
        entries.push(SparseBlockEntry::new(pos_in_block, nonzero_coeffs)?);
    }
    while current_block < num_blocks {
        push_offset(&mut offsets, entries.len())?;
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
pub(crate) fn map_onehot_to_regular_blocks(
    onehot_k: usize,
    indices: &[u32],
    block_len: usize,
    d: usize,
) -> Result<FlatRegularBlocks, HachiError> {
    let num_blocks = validate_onehot_layout(
        onehot_k,
        indices.len(),
        block_len,
        d,
        KdCompat::RegularLayout,
    )?;

    // In the regular layout each non-None chunk produces exactly one entry
    // at `ring_elem_idx = (c*K + idx) / D`. Because K is a multiple of D and
    // indices are processed in chunk order, the resulting stream of
    // `ring_elem_idx` values is monotonically non-decreasing, so we can
    // stream entries straight into a single flat buffer and emit block
    // boundaries as we cross them. No BTreeMap needed.
    let total_entries = indices.iter().filter(|&&p| p != INDICES_NONE).count();
    let mut entries: Vec<RegularOneHotEntry> = Vec::with_capacity(total_entries);
    let mut offsets: Vec<u32> = Vec::with_capacity(num_blocks + 1);
    offsets.push(0);
    let mut current_block = 0usize;

    for (chunk_idx, &packed) in indices.iter().enumerate() {
        if packed == INDICES_NONE {
            continue;
        }
        let idx = packed as usize;
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
            push_offset(&mut offsets, entries.len())?;
            current_block += 1;
        }
        entries.push(RegularOneHotEntry::new(pos_in_block, coeff_idx)?);
    }
    while current_block < num_blocks {
        push_offset(&mut offsets, entries.len())?;
        current_block += 1;
    }
    debug_assert_eq!(offsets.len(), num_blocks + 1);
    debug_assert_eq!(offsets[num_blocks] as usize, entries.len());

    Ok(FlatBlocks { entries, offsets })
}

/// Wide-accumulator sparse inner Ajtai: compute `t = A * s` for a one-hot block.
///
/// Instead of materializing the full decomposed vector `s` and doing a dense
/// matvec, we accumulate only the nonzero contributions using fused
/// shift-accumulate into `WideCyclotomicRing<W, D>` (carry-free i32
/// additions), then reduce once at the end:
///
/// ```text
/// t[a] += A[a][entry.pos * num_digits] * (X^{k_1} + X^{k_2} + ...)
/// ```
///
/// Using the wide accumulator avoids per-addition modular reduction versus
/// a direct field-ring accumulator.
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
        let col = entry.pos_in_block() * num_digits;
        for (a_idx, t_w) in t_wide.iter_mut().enumerate() {
            let a_wide = WideCyclotomicRing::from_ring(&A.row(a_idx)[col]);
            for &ci in entry.nonzero_coeffs() {
                a_wide.shift_accumulate_into(t_w, ci as usize);
            }
        }
    }

    t_wide.into_iter().map(|w| w.reduce()).collect()
}

// =============================================================================
// OneHotPoly: caller-facing polynomial backed by the flat per-block storage
// above. The HachiPolyOps impl follows further down.
// =============================================================================

/// Sentinel stored in [`OneHotPoly::indices`] to mark an all-zero (None)
/// chunk. Valid hot positions are bounded by `onehot_k`, which we cap to
/// `u32::MAX - 1` at construction, so this value can never collide with a
/// real hot index.
pub(crate) const INDICES_NONE: u32 = u32::MAX;

#[derive(Debug, Clone)]
pub(crate) enum OneHotBlocks {
    Regular(FlatRegularBlocks),
    General(FlatSparseBlocks),
}

impl OneHotBlocks {
    #[inline]
    fn num_blocks(&self) -> usize {
        match self {
            OneHotBlocks::Regular(blocks) => blocks.num_blocks(),
            OneHotBlocks::General(blocks) => blocks.num_blocks(),
        }
    }
}

/// One-hot polynomial: sparse witness with at most one nonzero field element
/// per chunk of size `onehot_k`.
///
/// Exploits sparsity in all four operations, avoiding inner ring
/// multiplications during commit and decomposing only nonzero monomials.
///
/// The polynomial is stored layout-agnostically as the flat list of hot
/// indices supplied at construction. Each op takes `block_len` at call time
/// and the per-block bucketing is materialized lazily on the first call and
/// cached for subsequent calls (as a `(block_len, OneHotBlocks)` pair inside
/// a `OnceLock`). That mirrors how [`DensePoly`] accepts `block_len` per op,
/// and keeps `OneHotPoly` free of the commit-layout parameters it used to
/// bake in at construction.
///
/// Generic over `I`: the index type accepted at construction time. Use `u8`
/// when `onehot_k <= 256` to reduce temporary index storage.
#[derive(Debug)]
pub struct OneHotPoly<F: FieldCore, const D: usize, I: OneHotIndex = usize> {
    pub(crate) num_vars: usize,
    pub(crate) onehot_k: usize,
    /// Per-chunk hot-position indices, with [`INDICES_NONE`] encoding a
    /// `None` (all-zero) chunk. Stored as packed `u32` (4 B / chunk) rather
    /// than `Option<usize>` (16 B / chunk) because at nv=32 with K=256 the
    /// index vector has 2^24 entries; the packed form saves ~192 MB of
    /// heap per polynomial.
    pub(crate) indices: Vec<u32>,
    pub(crate) total_ring_elems: usize,
    pub(crate) block_cache: OnceLock<(usize, OneHotBlocks)>,
    pub(crate) _marker: PhantomData<(F, I)>,
}

impl<F: FieldCore, const D: usize, I: OneHotIndex> Clone for OneHotPoly<F, D, I> {
    fn clone(&self) -> Self {
        let block_cache = OnceLock::new();
        if let Some((block_len, blocks)) = self.block_cache.get() {
            let _ = block_cache.set((*block_len, blocks.clone()));
        }
        Self {
            num_vars: self.num_vars,
            onehot_k: self.onehot_k,
            indices: self.indices.clone(),
            total_ring_elems: self.total_ring_elems,
            block_cache,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldCore, const D: usize, I: OneHotIndex> OneHotPoly<F, D, I> {
    /// Build a one-hot polynomial from chunk size and hot-position indices.
    ///
    /// `indices[c]` is the hot position in chunk `c` (`None` for all-zero chunks).
    ///
    /// The commit-layout split (how blocks are tiled within the polynomial)
    /// is no longer baked in at construction. Each op receives `block_len`
    /// from the caller and the per-block representation is materialized on
    /// demand.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent, any index is out of
    /// range, or `onehot_k` and `D` are not nicely matched.
    pub fn new(onehot_k: usize, indices: Vec<Option<I>>) -> Result<Self, HachiError> {
        if onehot_k == 0 {
            return Err(HachiError::InvalidInput(
                "onehot_k must be nonzero".to_string(),
            ));
        }
        // Hot positions must fit alongside the [`INDICES_NONE`] sentinel.
        if onehot_k >= INDICES_NONE as usize {
            return Err(HachiError::InvalidInput(format!(
                "onehot_k={onehot_k} exceeds the maximum storable chunk size ({})",
                INDICES_NONE - 1
            )));
        }
        if !(onehot_k.is_multiple_of(D) || D.is_multiple_of(onehot_k)) {
            return Err(HachiError::InvalidInput(format!(
                "onehot_k={onehot_k} and D={D} must be nicely matched (one divides the other)"
            )));
        }
        let total_field_elems = indices.len().checked_mul(onehot_k).ok_or_else(|| {
            HachiError::InvalidInput("onehot total field element count overflow".to_string())
        })?;
        if !total_field_elems.is_power_of_two() {
            return Err(HachiError::InvalidInput(format!(
                "onehot total field elements {total_field_elems} is not a power of two"
            )));
        }
        if !total_field_elems.is_multiple_of(D) {
            return Err(HachiError::InvalidInput(format!(
                "total field elements {total_field_elems} is not divisible by D={D}"
            )));
        }
        let total_ring_elems = total_field_elems / D;
        let mut packed: Vec<u32> = Vec::with_capacity(indices.len());
        for (chunk_idx, opt) in indices.iter().enumerate() {
            match opt {
                Some(raw) => {
                    let idx = raw.as_usize();
                    if idx >= onehot_k {
                        return Err(HachiError::InvalidInput(format!(
                            "index {idx} out of range for chunk size K={onehot_k} at position {chunk_idx}"
                        )));
                    }
                    packed.push(idx as u32);
                }
                None => packed.push(INDICES_NONE),
            }
        }
        Ok(Self {
            num_vars: total_field_elems.trailing_zeros() as usize,
            onehot_k,
            indices: packed,
            total_ring_elems,
            block_cache: OnceLock::new(),
            _marker: PhantomData,
        })
    }

    /// Whether the regular (single-hot-coeff per ring element) layout applies.
    fn use_regular_blocks(&self) -> bool {
        self.onehot_k >= D && self.onehot_k.is_multiple_of(D)
    }

    /// Return cached per-block storage, building it on first call for
    /// `block_len`.
    ///
    /// Subsequent calls must pass the same `block_len`; differing `block_len`
    /// is rejected rather than silently rebuilt because it indicates a
    /// layout mismatch between ops on the same polynomial.
    fn blocks_for(&self, block_len: usize) -> Result<&OneHotBlocks, HachiError> {
        // Fast path: cache already built for this `block_len`.
        if let Some((cached_len, blocks)) = self.block_cache.get() {
            if *cached_len == block_len {
                return Ok(blocks);
            }
            return Err(HachiError::InvalidInput(format!(
                "OneHotPoly was first used with block_len={cached_len} but is now being \
                 used with block_len={block_len}; all ops on the same \
                 polynomial must share a single layout"
            )));
        }
        // Slow path: build blocks and install them. Validate `block_len`
        // *before* building so the error path is cheap.
        if block_len == 0 || !block_len.is_power_of_two() {
            return Err(HachiError::InvalidInput(format!(
                "block_len={block_len} must be a nonzero power of two"
            )));
        }
        if !self.total_ring_elems.is_multiple_of(block_len) {
            return Err(HachiError::InvalidSize {
                expected: self.total_ring_elems,
                actual: block_len,
            });
        }
        let (cached_len, blocks) = {
            let _span = tracing::debug_span!("OneHotPoly::build_blocks", block_len).entered();
            self.block_cache.get_or_init(|| {
                let blocks = self
                    .build_blocks_inner(block_len)
                    .expect("block_len validated above");
                (block_len, blocks)
            })
        };
        if *cached_len != block_len {
            // A concurrent caller installed a different `block_len` before
            // our closure ran. Report the mismatch instead of silently
            // accepting the mismatched cache.
            return Err(HachiError::InvalidInput(format!(
                "OneHotPoly was first used with block_len={cached_len} but is now being \
                 used with block_len={block_len}; all ops on the same \
                 polynomial must share a single layout"
            )));
        }
        Ok(blocks)
    }

    fn build_blocks_inner(&self, block_len: usize) -> Result<OneHotBlocks, HachiError> {
        if self.use_regular_blocks() {
            Ok(OneHotBlocks::Regular(map_onehot_to_regular_blocks(
                self.onehot_k,
                &self.indices,
                block_len,
                D,
            )?))
        } else {
            Ok(OneHotBlocks::General(map_onehot_to_sparse_blocks(
                self.onehot_k,
                &self.indices,
                block_len,
                D,
            )?))
        }
    }

    fn decompose_fold_regular_onehot(
        &self,
        regular_blocks: &FlatRegularBlocks,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> DecomposeFoldWitness<F, D>
    where
        F: CanonicalField,
    {
        let num_blocks = challenges.len().min(regular_blocks.num_blocks());
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let block_views: Vec<&[RegularOneHotEntry]> = regular_blocks.iter_blocks().collect();

        let coeff_accum_digit0: Vec<[i32; D]> = {
            let _span = tracing::info_span!("onehot_regular_accumulate").entered();
            regular_onehot_accumulate::<D>(&block_views, challenges, num_blocks, block_len)
        };

        let coeff_accum = if num_digits == 1 {
            coeff_accum_digit0
        } else {
            let _span = tracing::info_span!("onehot_regular_expand").entered();
            let mut expanded = Vec::with_capacity(block_len * num_digits);
            for coeffs in coeff_accum_digit0 {
                expanded.push(coeffs);
                for _ in 1..num_digits {
                    expanded.push([0i32; D]);
                }
            }
            expanded
        };

        let _span = tracing::info_span!("onehot_regular_convert").entered();
        build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
    }

    fn decompose_fold_sparse_onehot(
        &self,
        sparse_blocks: &FlatSparseBlocks,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> DecomposeFoldWitness<F, D>
    where
        F: CanonicalField,
    {
        let inner_width = block_len * num_digits;
        let num_blocks = challenges.len().min(sparse_blocks.num_blocks());
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let block_views: Vec<&[SparseBlockEntry]> = sparse_blocks.iter_blocks().collect();

        let coeff_accum = {
            let _span = tracing::info_span!("onehot_sparse_accumulate").entered();
            sparse_onehot_accumulate::<D>(
                &block_views,
                challenges,
                num_blocks,
                inner_width,
                num_digits,
            )
        };

        let _span = tracing::info_span!("onehot_sparse_convert").entered();
        build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
    }

    fn decompose_fold_batched_regular_onehot(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F, D>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let mut flat_blocks: Vec<&[RegularOneHotEntry]> = Vec::with_capacity(total_blocks);
        for poly in polys {
            // `blocks_for` was already called by the public batched entry
            // point; this just reads the cached layout.
            let (_, cached) = poly.block_cache.get()?;
            let OneHotBlocks::Regular(blocks) = cached else {
                return None;
            };
            flat_blocks.extend(blocks.iter_blocks());
        }
        if flat_blocks.is_empty() {
            return None;
        }
        let active_blocks = flat_blocks.len().min(total_blocks);
        let modulus = (-F::one()).to_canonical_u128() + 1;

        let coeff_accum_digit0 = {
            let _span = tracing::info_span!("onehot_regular_accumulate_batched").entered();
            regular_onehot_accumulate::<D>(&flat_blocks, challenges, active_blocks, block_len)
        };

        let coeff_accum = if num_digits == 1 {
            coeff_accum_digit0
        } else {
            let _span = tracing::info_span!("onehot_regular_expand_batched").entered();
            let mut expanded = Vec::with_capacity(block_len * num_digits);
            for coeffs in coeff_accum_digit0 {
                expanded.push(coeffs);
                for _ in 1..num_digits {
                    expanded.push([0i32; D]);
                }
            }
            expanded
        };

        let _span = tracing::info_span!("onehot_regular_convert_batched").entered();
        Some(build_decompose_fold_witness::<F, D>(coeff_accum, modulus))
    }

    fn decompose_fold_batched_sparse_onehot(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F, D>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let mut flat_blocks: Vec<&[SparseBlockEntry]> = Vec::with_capacity(total_blocks);
        for poly in polys {
            let (_, cached) = poly.block_cache.get()?;
            let OneHotBlocks::General(blocks) = cached else {
                return None;
            };
            flat_blocks.extend(blocks.iter_blocks());
        }
        if flat_blocks.is_empty() {
            return None;
        }
        let active_blocks = flat_blocks.len().min(total_blocks);
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let inner_width = block_len * num_digits;

        let coeff_accum = {
            let _span = tracing::info_span!("onehot_sparse_accumulate_batched").entered();
            sparse_onehot_accumulate::<D>(
                &flat_blocks,
                challenges,
                active_blocks,
                inner_width,
                num_digits,
            )
        };

        let _span = tracing::info_span!("onehot_sparse_convert_batched").entered();
        Some(build_decompose_fold_witness::<F, D>(coeff_accum, modulus))
    }
}

impl<F, const D: usize, I: OneHotIndex> HachiPolyOps<F, D> for OneHotPoly<F, D, I>
where
    F: FieldCore + CanonicalField + HasWide,
{
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        self.total_ring_elems
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate_ring(&self, scalars: &[F]) -> CyclotomicRing<F, D> {
        // `evaluate_ring` is layout-free: it only needs the absolute ring
        // index per hot entry, not a per-block split. Iterate the raw
        // indices directly so we do not need to touch the block cache.
        let onehot_k = self.onehot_k;
        cfg_fold_reduce!(
            0..self.indices.len(),
            || CyclotomicRing::<F, D>::zero(),
            |mut acc: CyclotomicRing<F, D>, chunk_idx: usize| {
                let packed = self.indices[chunk_idx];
                if packed != INDICES_NONE {
                    let field_pos = chunk_idx * onehot_k + packed as usize;
                    let ring_idx = field_pos / D;
                    let coeff_idx = field_pos % D;
                    if ring_idx < scalars.len() {
                        acc.coeffs[coeff_idx] += scalars[ring_idx];
                    }
                }
                acc
            },
            |a, b| a + b
        )
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(block_len)
            .expect("OneHotPoly::fold_blocks: invalid block_len for this polynomial");
        let num_blocks = blocks.num_blocks();
        match blocks {
            OneHotBlocks::Regular(flat) => cfg_into_iter!(0..num_blocks)
                .map(|i| fold_regular_onehot_block(flat.block(i), scalars, block_len))
                .collect(),
            OneHotBlocks::General(flat) => cfg_into_iter!(0..num_blocks)
                .map(|i| fold_sparse_onehot_block(flat.block(i), scalars, block_len))
                .collect(),
        }
    }

    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let blocks = self
            .blocks_for(block_len)
            .expect("OneHotPoly::evaluate_and_fold: invalid block_len for this polynomial");
        let num_blocks = blocks.num_blocks();
        let folded: Vec<CyclotomicRing<F, D>> = match blocks {
            OneHotBlocks::Regular(flat) => cfg_into_iter!(0..num_blocks)
                .map(|i| fold_regular_onehot_block(flat.block(i), fold_scalars, block_len))
                .collect(),
            OneHotBlocks::General(flat) => cfg_into_iter!(0..num_blocks)
                .map(|i| fold_sparse_onehot_block(flat.block(i), fold_scalars, block_len))
                .collect(),
        };
        let eval = folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + f_i.scale(s_i)
            });
        (eval, folded)
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold")]
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        let blocks = self
            .blocks_for(block_len)
            .expect("OneHotPoly::decompose_fold: invalid block_len for this polynomial");
        match blocks {
            OneHotBlocks::Regular(blocks) => {
                self.decompose_fold_regular_onehot(blocks, challenges, block_len, num_digits)
            }
            OneHotBlocks::General(blocks) => {
                self.decompose_fold_sparse_onehot(blocks, challenges, block_len, num_digits)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold_batched")]
    fn decompose_fold_batched(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F, D>> {
        // Materialize per-poly block caches up front so every poly agrees on
        // `block_len` before we touch the batched kernels.
        for poly in polys {
            poly.blocks_for(block_len).expect(
                "OneHotPoly::decompose_fold_batched: invalid block_len for one of the polynomials",
            );
        }
        let first = polys.first()?;
        let (_, first_blocks) = first
            .block_cache
            .get()
            .expect("block cache was just built above");
        match first_blocks {
            OneHotBlocks::Regular(_) => Self::decompose_fold_batched_regular_onehot(
                polys, challenges, block_len, num_digits,
            ),
            OneHotBlocks::General(_) => {
                Self::decompose_fold_batched_sparse_onehot(polys, challenges, block_len, num_digits)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::commit_inner")]
    fn commit_inner(
        &self,
        a_matrix: &FlatMatrix<F>,
        _ntt_a: &NttSlotCache<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<FlatDigitBlocks<D>, HachiError> {
        let blocks = self.blocks_for(block_len)?;
        let a_view = a_matrix.ring_view::<D>(n_a, matrix_stride);
        let num_blocks = blocks.num_blocks();
        let active_a_cols = num_cols_a(block_len, num_digits_commit)?;
        if active_a_cols > a_view.num_cols() {
            return Err(HachiError::InvalidSetup(format!(
                "active A width {active_a_cols} exceeds setup envelope {}",
                a_view.num_cols()
            )));
        }
        let zero_block_len = n_a.checked_mul(num_digits_open).unwrap();

        let t_all = match blocks {
            OneHotBlocks::Regular(blocks) => {
                let views: Vec<&[RegularOneHotEntry]> = blocks.iter_blocks().collect();
                onehot_column_sweep_ajtai_regular::<F, D>(
                    &a_view,
                    &views,
                    n_a,
                    active_a_cols,
                    num_digits_commit,
                )
            }
            OneHotBlocks::General(blocks) => {
                let views: Vec<&[SparseBlockEntry]> = blocks.iter_blocks().collect();
                onehot_column_sweep_ajtai::<F, D>(
                    &a_view,
                    &views,
                    n_a,
                    active_a_cols,
                    num_digits_commit,
                )
            }
        };

        let mut t_hat = FlatDigitBlocks::zeroed(vec![zero_block_len; num_blocks])?;
        let dst_blocks = t_hat.split_blocks_mut();
        #[cfg(feature = "parallel")]
        cfg_into_iter!(dst_blocks)
            .zip(cfg_iter!(t_all))
            .for_each(|(dst, t_i)| {
                if !t_i.iter().all(|r| *r == CyclotomicRing::zero()) {
                    decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis);
                }
            });
        #[cfg(not(feature = "parallel"))]
        dst_blocks
            .into_iter()
            .zip(t_all.iter())
            .for_each(|(dst, t_i)| {
                if !t_i.iter().all(|r| *r == CyclotomicRing::zero()) {
                    decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis);
                }
            });

        Ok(t_hat)
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::commit_inner_witness")]
    fn commit_inner_witness(
        &self,
        a_matrix: &FlatMatrix<F>,
        _ntt_a: &NttSlotCache<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<CommitInnerWitness<F, D>, HachiError> {
        let blocks = self.blocks_for(block_len)?;
        let a_view = a_matrix.ring_view::<D>(n_a, matrix_stride);
        let active_a_cols = num_cols_a(block_len, num_digits_commit)?;
        if active_a_cols > a_view.num_cols() {
            return Err(HachiError::InvalidSetup(format!(
                "active A width {active_a_cols} exceeds setup envelope {}",
                a_view.num_cols()
            )));
        }
        let zero_block_len = n_a.checked_mul(num_digits_open).unwrap();

        let t = match blocks {
            OneHotBlocks::Regular(blocks) => {
                let views: Vec<&[RegularOneHotEntry]> = blocks.iter_blocks().collect();
                onehot_column_sweep_ajtai_regular::<F, D>(
                    &a_view,
                    &views,
                    n_a,
                    active_a_cols,
                    num_digits_commit,
                )
            }
            OneHotBlocks::General(blocks) => {
                let views: Vec<&[SparseBlockEntry]> = blocks.iter_blocks().collect();
                onehot_column_sweep_ajtai::<F, D>(
                    &a_view,
                    &views,
                    n_a,
                    active_a_cols,
                    num_digits_commit,
                )
            }
        };

        let mut t_hat = FlatDigitBlocks::zeroed(vec![zero_block_len; t.len()])?;
        let dst_blocks = t_hat.split_blocks_mut();
        #[cfg(feature = "parallel")]
        cfg_into_iter!(dst_blocks)
            .zip(cfg_iter!(t))
            .for_each(|(dst, t_i)| {
                if !t_i.iter().all(|r| *r == CyclotomicRing::zero()) {
                    decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis);
                }
            });
        #[cfg(not(feature = "parallel"))]
        dst_blocks.into_iter().zip(t.iter()).for_each(|(dst, t_i)| {
            if !t_i.iter().all(|r| *r == CyclotomicRing::zero()) {
                decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis);
            }
        });

        Ok(CommitInnerWitness { t, t_hat })
    }

    fn direct_root_witness(&self) -> Result<DirectWitnessProof<F>, HachiError> {
        let total_evals = 1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            HachiError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })?;
        let mut evals = vec![F::zero(); total_evals];
        for (chunk_idx, &packed) in self.indices.iter().enumerate() {
            if packed == INDICES_NONE {
                continue;
            }
            let field_pos = chunk_idx
                .checked_mul(self.onehot_k)
                .and_then(|base| base.checked_add(packed as usize))
                .ok_or_else(|| {
                    HachiError::InvalidInput("onehot direct witness index overflow".to_string())
                })?;
            if field_pos >= evals.len() {
                return Err(HachiError::InvalidInput(format!(
                    "onehot direct witness index {field_pos} out of range for {} evals",
                    evals.len()
                )));
            }
            evals[field_pos] = F::one();
        }
        Ok(DirectWitnessProof::FieldElements(FlatRingVec::from_coeffs(
            evals,
        )))
    }
}

fn num_cols_a(block_len: usize, num_digits_commit: usize) -> Result<usize, HachiError> {
    block_len
        .checked_mul(num_digits_commit)
        .ok_or_else(|| HachiError::InvalidSetup("active A width overflow".to_string()))
}

fn fold_regular_onehot_block<F: FieldCore, const D: usize>(
    entries: &[RegularOneHotEntry],
    scalars: &[F],
    block_len: usize,
) -> CyclotomicRing<F, D> {
    let mut coeffs_acc = [F::zero(); D];
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            coeffs_acc[entry.coeff_idx()] += scalars[pos];
        }
    }
    CyclotomicRing::from_coefficients(coeffs_acc)
}

fn fold_sparse_onehot_block<F: FieldCore, const D: usize>(
    entries: &[SparseBlockEntry],
    scalars: &[F],
    block_len: usize,
) -> CyclotomicRing<F, D> {
    let mut coeffs_acc = [F::zero(); D];
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            let s = scalars[pos];
            for &ci in entry.nonzero_coeffs() {
                coeffs_acc[ci as usize] += s;
            }
        }
    }
    CyclotomicRing::from_coefficients(coeffs_acc)
}

fn inner_ajtai_regular_onehot_wide<F, const D: usize>(
    a_view: &crate::protocol::commitment::utils::flat_matrix::RingMatrixView<'_, F, D>,
    regular_entries: &[RegularOneHotEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: crate::AdditiveGroup + From<F> + crate::algebra::fields::wide::ReduceTo<F>,
{
    let n_a = a_view.num_rows();
    let mut t_wide = vec![WideCyclotomicRing::<F::Wide, D>::zero(); n_a];

    for entry in regular_entries {
        let col = entry.pos_in_block() * num_digits;
        let coeff_idx = entry.coeff_idx();
        for (a_idx, t_w) in t_wide.iter_mut().enumerate() {
            let a_wide = WideCyclotomicRing::from_ring(&a_view.row(a_idx)[col]);
            a_wide.shift_accumulate_into(t_w, coeff_idx);
        }
    }

    t_wide.into_iter().map(|w| w.reduce()).collect()
}
fn inner_ajtai_regular_onehot_chunked<F, const D: usize>(
    a_view: &crate::protocol::commitment::utils::flat_matrix::RingMatrixView<'_, F, D>,
    regular_entries: &[RegularOneHotEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: crate::AdditiveGroup + From<F> + crate::algebra::fields::wide::ReduceTo<F>,
{
    let n_a = a_view.num_rows();
    let mut t = vec![CyclotomicRing::<F, D>::zero(); n_a];

    for chunk in regular_entries.chunks(MAX_WIDE_SHIFT_ACCUMULATIONS) {
        let partial = inner_ajtai_regular_onehot_wide(a_view, chunk, num_digits);
        for (dst, src) in t.iter_mut().zip(partial.iter()) {
            *dst += *src;
        }
    }

    t
}

/// L2 cache budget (in bytes) for the tile of wide accumulators in the
/// column-sweep commit.  Each tile's `accums` allocation is capped to this
/// size so the scatter loop stays L2-resident.
///
/// 2 MB is a conservative middle ground: fits in Apple M-series L2
/// (~4 MB/core) and exceeds most x86 per-core L2 (~256 KB–1 MB) only
/// modestly, relying on the shared L3 backstop.
///
// TODO: benchmark column-sweep on x86 vs ARM with budget values
// (512 KB, 1 MB, 2 MB, 4 MB) at production configs to determine
// whether a smaller or arch-specific budget helps on x86.
const L2_TILE_BUDGET: usize = 1 << 21;

/// Wide accumulators use 16-bit chunks in `i32` limbs, so they can safely
/// absorb at most 32,768 unit-scale additions before overflow.
const MAX_WIDE_SHIFT_ACCUMULATIONS: usize = 1 << 15;

/// Minimum blocks-per-thread required before enabling the column-sweep kernel.
const SWEEP_THRESHOLD: usize = 32;

/// One tile-local hot entry: `(a-column, local-block-index, coefficient-index)`.
///
/// All entries from one L2 tile are bucketed into this flat vector so the
/// outer loop can load each A-column exactly once, then scatter the column's
/// contribution into every block whose entry lands in that column.
type ColEntry = (usize, u32, u16);

/// Inner two-level-tiled column-sweep, shared between the regular and sparse
/// wrappers.
///
/// Threads partition blocks evenly (outer, for parallelism); within each
/// thread, blocks are processed in L2-sized tiles (inner, for cache
/// locality). For each tile, `push_entries` writes one `(col, local_b,
/// coeff_idx)` tuple per hot contribution; sort-by-col then drives a single
/// sweep per A row.
#[inline]
fn column_sweep_core<E, F, const D: usize>(
    a_view: &crate::protocol::commitment::utils::flat_matrix::RingMatrixView<'_, F, D>,
    blocks: &[&[E]],
    n_a: usize,
    num_digits_commit: usize,
    push_entries: impl Fn(&[E], u32, usize, &mut Vec<ColEntry>) + Send + Sync + Copy,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    E: Sync,
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: crate::AdditiveGroup + From<F> + crate::algebra::fields::wide::ReduceTo<F>,
{
    let num_blocks = blocks.len();
    let accum_bytes = n_a * D * std::mem::size_of::<F::Wide>();
    let block_tile = if accum_bytes > 0 {
        (L2_TILE_BUDGET / accum_bytes).max(1)
    } else {
        num_blocks
    };

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(num_blocks).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let blocks_per_thread = num_blocks.div_ceil(num_threads);

    let thread_results: Vec<Vec<Vec<CyclotomicRing<F, D>>>> = cfg_into_iter!(0..num_threads)
        .map(|tid| {
            let block_start = tid * blocks_per_thread;
            let block_end = (block_start + blocks_per_thread).min(num_blocks);
            if block_start >= block_end {
                return Vec::new();
            }
            let my_count = block_end - block_start;

            let mut result: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(my_count);
            result.resize_with(my_count, Vec::new);

            // Reuse across tiles so earlier capacity carries over, but only
            // allocate buckets for columns that are actually touched.
            let mut col_entries: Vec<ColEntry> = Vec::new();

            for tile_start in (0..my_count).step_by(block_tile) {
                let tile_end = (tile_start + block_tile).min(my_count);
                let tile_len = tile_end - tile_start;

                col_entries.clear();
                for local_b in 0..tile_len {
                    let block_entries = blocks[block_start + tile_start + local_b];
                    push_entries(
                        block_entries,
                        local_b as u32,
                        num_digits_commit,
                        &mut col_entries,
                    );
                }
                col_entries.sort_unstable_by_key(|&(col, _, _)| col);

                let mut accums: Vec<Vec<WideCyclotomicRing<F::Wide, D>>> = (0..tile_len)
                    .map(|_| vec![WideCyclotomicRing::zero(); n_a])
                    .collect();

                for a_idx in 0..n_a {
                    let a_row = a_view.row(a_idx);
                    let mut idx = 0usize;
                    while idx < col_entries.len() {
                        let col = col_entries[idx].0;
                        let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
                        while idx < col_entries.len() && col_entries[idx].0 == col {
                            let (_, lb, ci) = col_entries[idx];
                            a_wide.shift_accumulate_into(
                                &mut accums[lb as usize][a_idx],
                                ci as usize,
                            );
                            idx += 1;
                        }
                    }
                }

                for (local_b, row_accums) in accums.into_iter().enumerate() {
                    result[tile_start + local_b] =
                        row_accums.into_iter().map(|w| w.reduce()).collect();
                }
            }

            result
        })
        .collect();

    let mut out: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(num_blocks);
    for thread_blocks in thread_results {
        out.extend(thread_blocks);
    }
    out
}

/// Column-sweep Ajtai commitment for regular one-hot blocks.
///
/// Uses [`column_sweep_core`] for the tiled sweep plus a safety fallback when
/// any block has more than `MAX_WIDE_SHIFT_ACCUMULATIONS` hot entries (the
/// wide accumulator would overflow) and a small-block fast path when
/// `blocks_per_thread` is already L2-friendly.
fn onehot_column_sweep_ajtai_regular<F, const D: usize>(
    a_view: &crate::protocol::commitment::utils::flat_matrix::RingMatrixView<'_, F, D>,
    regular_blocks: &[&[RegularOneHotEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: crate::AdditiveGroup + From<F> + crate::algebra::fields::wide::ReduceTo<F>,
{
    let num_blocks = regular_blocks.len();
    debug_assert!(
        active_a_cols <= a_view.num_cols(),
        "active A width exceeds setup envelope"
    );
    if regular_blocks
        .iter()
        .any(|entries| entries.len() > MAX_WIDE_SHIFT_ACCUMULATIONS)
    {
        return cfg_into_iter!(0..num_blocks)
            .map(|i| {
                inner_ajtai_regular_onehot_chunked(a_view, regular_blocks[i], num_digits_commit)
            })
            .collect();
    }

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(num_blocks).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;
    let blocks_per_thread = num_blocks.div_ceil(num_threads);

    if blocks_per_thread <= SWEEP_THRESHOLD {
        return cfg_into_iter!(0..num_blocks)
            .map(|i| inner_ajtai_regular_onehot_wide(a_view, regular_blocks[i], num_digits_commit))
            .collect();
    }

    column_sweep_core::<RegularOneHotEntry, F, D>(
        a_view,
        regular_blocks,
        n_a,
        num_digits_commit,
        |block_entries, local_b, num_digits, sink| {
            for entry in block_entries {
                let col = entry.pos_in_block() * num_digits;
                sink.push((col, local_b, entry.coeff_idx() as u16));
            }
        },
    )
}

/// Column-sweep Ajtai commitment for one-hot sparse blocks.
///
/// Same two-level tiling as [`onehot_column_sweep_ajtai_regular`]; each hot
/// ring element may contribute multiple coefficients, so `push_entries`
/// fans out the `nonzero_coeffs` list into individual `ColEntry` tuples.
fn onehot_column_sweep_ajtai<F, const D: usize>(
    a_view: &crate::protocol::commitment::utils::flat_matrix::RingMatrixView<'_, F, D>,
    sparse_blocks: &[&[SparseBlockEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: crate::AdditiveGroup + From<F> + crate::algebra::fields::wide::ReduceTo<F>,
{
    let num_blocks = sparse_blocks.len();
    debug_assert!(
        active_a_cols <= a_view.num_cols(),
        "active A width exceeds setup envelope"
    );

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads().min(num_blocks).max(1);
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;
    let blocks_per_thread = num_blocks.div_ceil(num_threads);

    if blocks_per_thread <= SWEEP_THRESHOLD {
        return cfg_into_iter!(0..num_blocks)
            .map(|i| inner_ajtai_onehot_wide(a_view, sparse_blocks[i], 0, num_digits_commit))
            .collect();
    }

    column_sweep_core::<SparseBlockEntry, F, D>(
        a_view,
        sparse_blocks,
        n_a,
        num_digits_commit,
        |block_entries, local_b, num_digits, sink| {
            for entry in block_entries {
                let col = entry.pos_in_block() * num_digits;
                for &ci in entry.nonzero_coeffs() {
                    sink.push((col, local_b, ci));
                }
            }
        },
    )
}

/// Test-only helpers for this module that need access to private invariants
/// (`FlatBlocks`' monotonic `offsets` / contiguous `entries`, and the
/// non-wide reference path for `inner_ajtai_onehot_wide`).
///
/// Gated on `#[cfg(test)]` so the production binary never sees them.
#[cfg(test)]
pub(crate) mod test_helpers {
    use super::{CyclotomicRing, FlatBlocks, SparseBlockEntry};
    use crate::{CanonicalField, FieldCore};

    /// Build a flat block layout from a pre-bucketed `Vec<Vec<E>>`.
    ///
    /// The production paths (`map_onehot_to_regular_blocks`,
    /// `map_onehot_to_sparse_blocks`) stream entries directly into the flat
    /// form without ever materialising per-block `Vec`s. This constructor
    /// exists only so tests that hand-assemble block-bucketed storage can
    /// still feed it into kernels that consume `FlatBlocks`.
    pub(crate) fn from_buckets<E>(buckets: Vec<Vec<E>>) -> FlatBlocks<E> {
        let num_blocks = buckets.len();
        let mut offsets = Vec::with_capacity(num_blocks + 1);
        let total: usize = buckets.iter().map(Vec::len).sum();
        let mut entries = Vec::with_capacity(total);
        offsets.push(0);
        for mut bucket in buckets {
            entries.append(&mut bucket);
            // `entries.len()` is bounded by `total = sum(Vec::len)` which
            // was accepted as `usize`; it is always safe to downcast to
            // `u32` on all supported layouts used by tests.
            offsets.push(u32::try_from(entries.len()).expect("flat block offset overflows u32"));
        }
        FlatBlocks { entries, offsets }
    }

    /// Reference (non-wide) sparse inner Ajtai used to cross-check
    /// [`super::inner_ajtai_onehot_wide`].
    ///
    /// Production code always uses the wide accumulator; this simpler
    /// variant only exists so tests can assert the two paths agree.
    #[allow(non_snake_case)]
    pub(crate) fn inner_ajtai_onehot_t_only<F: FieldCore + CanonicalField, const D: usize>(
        A: &[Vec<CyclotomicRing<F, D>>],
        sparse_entries: &[SparseBlockEntry],
        num_digits: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let n_a = A.len();
        let mut t = vec![CyclotomicRing::<F, D>::zero(); n_a];
        for entry in sparse_entries {
            let col = entry.pos_in_block() * num_digits;
            for a in 0..n_a {
                for &ci in entry.nonzero_coeffs() {
                    A[a][col].shift_accumulate_into(&mut t[a], ci as usize);
                }
            }
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::inner_ajtai_onehot_t_only;
    use super::*;
    use crate::algebra::fields::{Fp64, Pow2Offset24Field, Prime128Offset275};
    use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
    use crate::FromSmallInt;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn aggregate_witnesses<F: FieldCore, const D: usize>(
        witnesses: &[DecomposeFoldWitness<F, D>],
    ) -> DecomposeFoldWitness<F, D> {
        let mut acc = witnesses[0].clone();
        for witness in &witnesses[1..] {
            for (dst, src) in acc.z_pre.iter_mut().zip(witness.z_pre.iter()) {
                *dst += *src;
            }
            for (dst, src) in acc
                .centered_coeffs
                .iter_mut()
                .zip(witness.centered_coeffs.iter())
            {
                for k in 0..D {
                    dst[k] += src[k];
                }
            }
        }
        acc.centered_inf_norm = acc
            .centered_coeffs
            .iter()
            .flat_map(|coeffs| coeffs.iter())
            .map(|coeff| coeff.unsigned_abs())
            .max()
            .unwrap_or(0);
        acc
    }

    // -------------------------------------------------------------------------
    // Tests for the flat-storage mapping helpers and the sparse inner-Ajtai
    // reference implementation. Originally in `commitment/onehot.rs`.
    // -------------------------------------------------------------------------

    #[test]
    fn map_onehot_k_gt_d() {
        // K=16, D=4, T=2 chunks => 32 field elements => 8 ring elements
        // block_len=4 => 2 blocks of 4 ring elements each.
        let k = 16;
        let d = 4;
        let indices: Vec<u32> = vec![3, 10];
        let blocks = map_onehot_to_sparse_blocks(k, &indices, 4, d).unwrap();

        assert_eq!(blocks.num_blocks(), 2);
        assert_eq!(blocks.total_entries(), 2, "T=2 nonzero ring elements");

        for block in blocks.iter_blocks() {
            for entry in block {
                assert_eq!(entry.nonzero_coeffs().len(), 1, "K>D => single monomial");
            }
        }
    }

    #[test]
    fn map_onehot_k_eq_d() {
        // K=4, D=4, T=4 chunks => 16 field elements => 4 ring elements
        // block_len=2 => 2 blocks of 2 ring elements each.
        let k = 4;
        let d = 4;
        let indices: Vec<u32> = vec![0, 2, 3, 1];
        let blocks = map_onehot_to_sparse_blocks(k, &indices, 2, d).unwrap();

        assert_eq!(blocks.num_blocks(), 2);
        assert_eq!(
            blocks.total_entries(),
            4,
            "K=D => every ring element is nonzero"
        );

        for block in blocks.iter_blocks() {
            for entry in block {
                assert_eq!(entry.nonzero_coeffs().len(), 1, "K=D => single monomial");
            }
        }
    }

    #[test]
    fn map_onehot_k_lt_d() {
        // K=4, D=8, T=8 chunks => 32 field elements => 4 ring elements
        // block_len=2 => 2 blocks of 2 ring elements each.
        let k = 4;
        let d = 8;
        let indices: Vec<u32> = vec![0, 2, 3, 1, 0, 0, 3, 3];
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
                    entry.nonzero_coeffs().len(),
                    2,
                    "D=2K => 2 nonzero coeffs per ring element"
                );
            }
        }
    }

    #[test]
    fn map_onehot_rejects_non_divisible() {
        let result = map_onehot_to_sparse_blocks(3, &[0u32, 1], 2, 4);
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
            SparseBlockEntry::new(0, vec![1, 7, 15]).unwrap(),
            SparseBlockEntry::new(2, vec![0, 63]).unwrap(),
        ];

        let a_flat_elems: Vec<CyclotomicRing<F, D>> = a_matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let a_flat = FlatMatrix::from_ring_slice(&a_flat_elems);
        let a_view = a_flat.ring_view::<D>(n_a, block_len * num_digits);
        let ref_result = inner_ajtai_onehot_t_only(&a_matrix, &entries, num_digits);
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
            SparseBlockEntry::new(0, vec![0, 5, 32, 63]).unwrap(),
            SparseBlockEntry::new(1, vec![10]).unwrap(),
        ];

        let a_flat_elems: Vec<CyclotomicRing<F, D>> = a_matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let a_flat = FlatMatrix::from_ring_slice(&a_flat_elems);
        let a_view = a_flat.ring_view::<D>(n_a, block_len * num_digits);
        let ref_result = inner_ajtai_onehot_t_only(&a_matrix, &entries, num_digits);
        let wide_result = inner_ajtai_onehot_wide(&a_view, &entries, block_len, num_digits);

        assert_eq!(ref_result.len(), wide_result.len());
        for (r, w) in ref_result.iter().zip(wide_result.iter()) {
            assert_eq!(r, w, "wide result must match reference (Fp128)");
        }
    }

    // -------------------------------------------------------------------------
    // Tests that exercise the column-sweep kernels and the OneHotPoly-level
    // behaviour defined above.
    // -------------------------------------------------------------------------

    #[test]
    fn regular_onehot_large_block_uses_safe_accumulator_path() {
        type F = Pow2Offset24Field;
        const D: usize = 64;

        let block_len = MAX_WIDE_SHIFT_ACCUMULATIONS + 1;
        let max_coeff = F::from_canonical_u128_reduced((1u128 << 24) - 4);
        let dense_ring = CyclotomicRing::from_coefficients([max_coeff; D]);
        let a_matrix = [vec![dense_ring; block_len]];
        let bucket: Vec<RegularOneHotEntry> = (0..block_len)
            .map(|pos| RegularOneHotEntry::new(pos, pos % D).unwrap())
            .collect();
        let regular_blocks = super::test_helpers::from_buckets(vec![bucket.clone()]);

        let a_flat = FlatMatrix::from_ring_slice(&a_matrix[0]);
        let a_view = a_flat.ring_view::<D>(1, block_len);

        let regular_views: Vec<&[RegularOneHotEntry]> = regular_blocks.iter_blocks().collect();
        let got =
            onehot_column_sweep_ajtai_regular::<F, D>(&a_view, &regular_views, 1, block_len, 1);
        let expected = inner_ajtai_regular_onehot_chunked::<F, D>(&a_view, &bucket, 1);

        assert_eq!(got.len(), 1);
        assert_eq!(got[0], expected);
    }

    #[test]
    fn batched_regular_onehot_decompose_fold_matches_individual_aggregation() {
        type F = Pow2Offset24Field;
        const D: usize = 64;

        let block_len = 64;
        let mut indices0 = vec![None; 128];
        indices0[0] = Some(1usize);
        indices0[17] = Some(5usize);
        indices0[64] = Some(9usize);
        indices0[91] = Some(33usize);
        let mut indices1 = vec![None; 128];
        indices1[3] = Some(7usize);
        indices1[29] = Some(11usize);
        indices1[64] = Some(19usize);
        indices1[100] = Some(21usize);
        let polys = [
            OneHotPoly::<F, D>::new(block_len, indices0).unwrap(),
            OneHotPoly::<F, D>::new(block_len, indices1).unwrap(),
        ];
        let challenges = vec![
            SparseChallenge {
                positions: vec![0, 5],
                coeffs: vec![1, -1],
            },
            SparseChallenge {
                positions: vec![2, 7],
                coeffs: vec![1, 1],
            },
            SparseChallenge {
                positions: vec![4, 11],
                coeffs: vec![-1, 2],
            },
            SparseChallenge {
                positions: vec![8, 13],
                coeffs: vec![1, -2],
            },
        ];

        let expected = aggregate_witnesses(
            &polys
                .iter()
                .zip(challenges.chunks(2))
                .map(|(poly, poly_challenges)| {
                    poly.decompose_fold(poly_challenges, block_len, 1, 0)
                })
                .collect::<Vec<_>>(),
        );
        let poly_refs: Vec<&OneHotPoly<F, D>> = polys.iter().collect();
        let got = <OneHotPoly<F, D> as HachiPolyOps<F, D>>::decompose_fold_batched(
            &poly_refs,
            &challenges,
            block_len,
            1,
            0,
        )
        .expect("onehot batched path should apply");

        assert_eq!(got, expected);
    }

    #[test]
    fn regular_onehot_evaluate_and_fold_matches_factorized_eval() {
        type F = Pow2Offset24Field;
        const D: usize = 64;

        let poly =
            OneHotPoly::<F, D>::new(64, vec![Some(1usize), None, Some(9usize), Some(17usize)])
                .unwrap();
        let block_len = 2usize;
        let fold_scalars = vec![F::from_u64(3), F::from_u64(5)];
        let eval_outer_scalars = vec![F::from_u64(7), F::from_u64(11)];

        let (eval, folded) = poly.evaluate_and_fold(&eval_outer_scalars, &fold_scalars, block_len);
        let expected_folded = poly.fold_blocks(&fold_scalars, block_len);
        assert_eq!(folded, expected_folded);

        let full_scalars: Vec<F> = eval_outer_scalars
            .iter()
            .flat_map(|outer| fold_scalars.iter().map(move |inner| *outer * *inner))
            .collect();
        let expected_eval = poly.evaluate_ring(&full_scalars);
        assert_eq!(eval, expected_eval);
    }

    #[test]
    fn sparse_onehot_evaluate_and_fold_matches_factorized_eval() {
        type F = Pow2Offset24Field;
        const D: usize = 64;

        let poly = OneHotPoly::<F, D>::new(
            32,
            vec![
                Some(1usize),
                None,
                Some(7usize),
                Some(12usize),
                None,
                Some(3usize),
                None,
                Some(15usize),
            ],
        )
        .unwrap();
        let block_len = 2usize;
        let fold_scalars = vec![F::from_u64(2), F::from_u64(4)];
        let eval_outer_scalars = vec![F::from_u64(3), F::from_u64(5)];

        let (eval, folded) = poly.evaluate_and_fold(&eval_outer_scalars, &fold_scalars, block_len);
        let expected_folded = poly.fold_blocks(&fold_scalars, block_len);
        assert_eq!(folded, expected_folded);

        let full_scalars: Vec<F> = eval_outer_scalars
            .iter()
            .flat_map(|outer| fold_scalars.iter().map(move |inner| *outer * *inner))
            .collect();
        let expected_eval = poly.evaluate_ring(&full_scalars);
        assert_eq!(eval, expected_eval);
    }
}
