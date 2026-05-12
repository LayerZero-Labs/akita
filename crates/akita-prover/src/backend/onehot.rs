//! One-hot polynomial: sparse witness with at most one nonzero field
//! element per chunk of size `onehot_k`.
//!
//! [`OneHotPoly`] is a backend for [`AkitaPolyOps`](akita_prover::AkitaPolyOps)
//! that implements the four prover operations (ring evaluation, per-block
//! fold, decompose+fold, and inner-Ajtai commit) by iterating only over
//! the nonzero monomial positions.
//!
//! # Module layout
//!
//! The file is organised as three layers — entry types,
//! flat block storage, and the polynomial + its [`AkitaPolyOps`] impl.
//!
//!   - [`OneHotIndex`]: a tiny trait implemented for `u8`/`u16`/`u32`/
//!     `usize` so callers can hand [`OneHotPoly::new`] a `Vec<Option<I>>`
//!     at the narrowest width that fits their hot positions.
//!   - Per-block entry types: [`SingleChunkEntry`] (packed `u32 + u16`,
//!     used when each ring element covers at most one hot element —
//!     i.e. `K >= D && D | K`) and [`MultiChunkEntry`] (`u32 +
//!     Vec<u16>`, used when a ring element can cover zero to many
//!     hot elements — i.e. `K < D` with `K | D`). Coefficient indices fit
//!     in `u16` because the supported ring degrees are small; the
//!     bound is enforced in [`OneHotPoly::build_blocks_inner`].
//!   - [`FlatBlocks<E>`]: a container storing the
//!     variable-length per-block entry lists in one contiguous `Vec<E>`
//!     plus a `Vec<u32>` offsets array.
//!   - [`OneHotBlocks`]: a two-variant enum that wraps the built
//!     `FlatBlocks<E>` so [`OneHotPoly`]'s ops can dispatch to the right
//!     kernel based on the actual layout in use.
//!   - [`OneHotPoly<F, D, I>`]: the caller-facing polynomial.

use akita_algebra::ring::cyclotomic::WideCyclotomicRing;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::fields::wide::{HasWide, ReduceTo};
use akita_field::parallel::*;
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore};
use akita_types::{DirectWitnessProof, FlatDigitBlocks, FlatRingVec};
use akita_types::{FlatMatrix, RingMatrixView};
use std::marker::PhantomData;
use std::sync::OnceLock;

use crate::backend::poly_helpers::{build_decompose_fold_witness, fill_rotated_challenge};
use crate::kernels::crt_ntt::NttSlotCache;
use crate::kernels::linear::decompose_rows_i8_into;
use crate::{AkitaPolyOps, CommitInnerWitness, DecomposeFoldWitness};

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

/// Compact record for a single nonzero ring element in the
/// single-chunk layout.
///
/// In the single-chunk layout each ring element overlaps at most one
/// one-hot chunk, so the ring has exactly one hot coefficient (value 1)
/// and `D - 1` zero coefficients. We store nothing about the zero
/// rings and nothing about the zero coefficients of the nonzero ring;
/// the entry just pins down *which* ring element we are talking about
/// (`pos_in_block`, inside the flat per-block layout) and *which* of
/// its `D` coefficients is the hot one (`coeff_idx`).
///
/// This layout applies when `K >= D && D | K`: one one-hot chunk spans
/// `K/D` consecutive ring elements, so every ring element falls
/// entirely inside one chunk and hence contains at most one hot
/// coefficient.
///
/// # Example
///
/// Take `K = 64`, `D = 32`, and look at the first chunk. Its flat
/// field-position range is `[0, 64)`; it contributes to ring elements
/// `0` (coefficients at positions `[0, 32)`) and `1` (positions
/// `[32, 64)`). Say the hot position inside this chunk is 60, so
/// field position 60 is 1 and all other positions in `[0, 64)` are 0.
/// Then:
///
/// - `ring_idx = 60 / 32 = 1` (ring element 0 has no hot coefficient
///   and is skipped entirely; ring element 1 carries the hot one);
/// - `coeff_idx = 60 % 32 = 28`.
///
/// If that ring lives in the first block of the flat layout,
/// `pos_in_block = 1` (the second ring element of block 0). The stored
/// entry is `SingleChunkEntry { pos_in_block: 1, coeff_idx: 28 }`, and
/// no entry is emitted for ring 0.
///
/// # Invariants
///
/// Fields are private and accessed via `pos_in_block()` / `coeff_idx()`.
/// The caller-owned invariants `pos_in_block < block_len <= u32::MAX`
/// and `coeff_idx < D <= 65536` are pre-validated in
/// [`FlatBlocks::<SingleChunkEntry>::from_indices`]; the
/// constructor just stores the already-narrowed fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SingleChunkEntry {
    pos_in_block: u32,
    coeff_idx: u16,
}

impl SingleChunkEntry {
    /// Construct a single-chunk entry from already-validated native-width fields.
    #[inline]
    pub(crate) fn new(pos_in_block: u32, coeff_idx: u16) -> Self {
        Self {
            pos_in_block,
            coeff_idx,
        }
    }

    /// Position within the block (0..block_len).
    #[inline]
    pub(crate) fn pos_in_block(self) -> usize {
        self.pos_in_block as usize
    }

    /// Index of the single hot coefficient inside the ring element (0..D).
    #[inline]
    pub(crate) fn coeff_idx(self) -> usize {
        self.coeff_idx as usize
    }
}

/// Compact record for a single nonzero ring element in the
/// multi-chunk layout.
///
/// In the multi-chunk layout one ring element spans exactly `D/K`
/// whole consecutive one-hot chunks, so the ring can carry anywhere
/// from zero to `D/K` hot coefficients. We only emit an entry for
/// rings that have at least one, and within that entry we store
/// exactly which coefficients are hot (`nonzero_coeffs`) and where
/// the ring lives in the flat per-block layout (`pos_in_block`).
/// Everything else about the ring (its zero coefficients, its
/// neighbouring zero rings) is left implicit.
///
/// This layout applies when `K < D` with `K | D`: each ring element
/// contains exactly `D/K` whole consecutive chunks, each contributing
/// at most one hot coefficient to that ring.
///
/// # Worked example
///
/// Take `K = 8`, `D = 32`, so each ring element covers `D/K = 4`
/// consecutive chunks. Look at ring element 0, whose flat
/// field-position range is `[0, 32)` — chunks 0, 1, 2, 3 live inside
/// it:
///
/// - chunk 0 (field positions `[0, 8)`): hot at chunk-local index 3,
///   i.e. field position 3 → contributes `coeff_idx = 3`;
/// - chunk 1 (positions `[8, 16)`): all zero, contributes nothing;
/// - chunk 2 (positions `[16, 24)`): hot at chunk-local index 5, i.e.
///   field position 21 → contributes `coeff_idx = 21`;
/// - chunk 3 (positions `[24, 32)`): all zero, contributes nothing.
///
/// `coeff_idx` for a ring is just `field_pos % D` — the chunk boundary
/// doesn't enter the computation once we've landed inside the ring. If
/// this ring sits at position 0 in its block, the stored entry is
/// `MultiChunkEntry { pos_in_block: 0, nonzero_coeffs: [3, 21] }`. No
/// entry is emitted for rings whose four covering chunks are all zero.
///
/// # Why this representation
///
/// As with [`SingleChunkEntry`], we pay nothing for the zero rings and
/// nothing for the zero coefficients of the nonzero rings, so memory
/// stays proportional to the number of distinct nonzero rings and the
/// kernels skip the zeros on the hot path.
///
/// # Invariants
///
/// Fields are private and accessed via `pos_in_block()` /
/// `nonzero_coeffs()`. The caller-owned invariants
/// `pos_in_block < block_len <= u32::MAX` and every
/// `coeff < D <= 65536` are pre-validated in
/// [`FlatBlocks::<MultiChunkEntry>::from_indices`]; the
/// constructor just stores the already-narrowed fields.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MultiChunkEntry {
    pos_in_block: u32,
    nonzero_coeffs: Vec<u16>,
}

impl MultiChunkEntry {
    /// Construct a multi-chunk entry from already-validated native-width
    /// fields.
    #[inline]
    pub(crate) fn new(pos_in_block: u32, nonzero_coeffs: Vec<u16>) -> Self {
        Self {
            pos_in_block,
            nonzero_coeffs,
        }
    }

    /// Position within the block (0..block_len).
    #[inline]
    pub(crate) fn pos_in_block(&self) -> usize {
        self.pos_in_block as usize
    }

    /// Hot coefficient indices inside the ring element, each `< D`.
    #[inline]
    pub(crate) fn nonzero_coeffs(&self) -> &[u16] {
        &self.nonzero_coeffs
    }
}

/// Flat vector storing only the non-zero rings.
///
/// `offsets` says which entries belong to each block: block `i` occupies
/// `entries[offsets[i] as usize..offsets[i + 1] as usize]`.
///
/// Within one block, each entry records the position of a non-zero ring
/// (`pos_in_block`) together with the hot coefficient data for that ring
/// (`coeff_idx` for [`SingleChunkEntry`], `nonzero_coeffs` for
/// [`MultiChunkEntry`]).
///
/// Entries are sorted by `(block_idx, pos_in_block)`, so each per-block slice
/// is ascending in `pos_in_block`, matching the invariant the accumulators
/// rely on (they do `partition_point` on `pos_in_block`).
#[derive(Debug, Clone)]
pub(crate) struct FlatBlocks<E> {
    entries: Vec<E>,
    /// `len == num_blocks + 1`, `offsets[0] == 0`, `offsets[num_blocks] == entries.len()`.
    offsets: Vec<u32>,
}

impl<E> FlatBlocks<E> {
    #[inline]
    fn with_capacity(num_blocks: usize, entry_capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(num_blocks + 1);
        offsets.push(0);
        Self {
            entries: Vec::with_capacity(entry_capacity),
            offsets,
        }
    }

    /// Number of blocks.
    #[inline]
    pub(crate) fn num_blocks(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Slice of entries for block `i`.
    pub(crate) fn block(&self, i: usize) -> &[E] {
        let num_blocks = self.num_blocks();
        assert!(
            i < num_blocks,
            "FlatBlocks::block: block index {i} out of range for {num_blocks} blocks"
        );
        let lo = self.offsets[i] as usize;
        let hi = self.offsets[i + 1] as usize;
        assert!(
            lo <= hi,
            "FlatBlocks::block: malformed offsets for block {i}: lo={lo} > hi={hi}"
        );
        &self.entries[lo..hi]
    }

    #[inline]
    fn advance_to_block(&mut self, current_block: &mut usize, block_idx: usize, num_blocks: usize) {
        debug_assert!(
            block_idx <= num_blocks,
            "FlatBlocks: block index {block_idx} out of range for {num_blocks} blocks"
        );
        while *current_block < block_idx {
            self.offsets.push(self.entries.len() as u32);
            *current_block += 1;
        }
    }

    #[inline]
    fn push_entry(
        &mut self,
        current_block: &mut usize,
        block_idx: usize,
        num_blocks: usize,
        entry: E,
    ) {
        debug_assert!(
            block_idx < num_blocks,
            "FlatBlocks: block index {block_idx} out of range for {num_blocks} blocks"
        );
        self.advance_to_block(current_block, block_idx, num_blocks);
        self.entries.push(entry);
    }

    fn finish_build(mut self, current_block: usize, num_blocks: usize) -> Self {
        let mut current_block = current_block;
        self.advance_to_block(&mut current_block, num_blocks, num_blocks);
        debug_assert_eq!(self.offsets.len(), num_blocks + 1);
        debug_assert_eq!(self.offsets[num_blocks] as usize, self.entries.len());
        self
    }
}

impl FlatBlocks<MultiChunkEntry> {
    /// Build a multi-chunk-layout one-hot `FlatBlocks` from an index witness.
    ///
    /// This applies exactly to the `K < D && K | D` case, where each
    /// ring element contains `D/K` whole consecutive chunks. Grouping
    /// the witness by those chunk ranges lets us materialize each
    /// nonzero ring in one pass.
    ///
    /// # Errors
    ///
    /// Returns an error only if the internal offsets vector (bounded by
    /// `num_blocks + 1`) overflows `u32::MAX`.
    pub(crate) fn from_indices<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        block_len: usize,
        d: usize,
        num_blocks: usize,
    ) -> Result<Self, AkitaError> {
        assert!(
            onehot_k < d && d.is_multiple_of(onehot_k),
            "FlatBlocks::<MultiChunkEntry>::from_indices: K={onehot_k} and D={d} must satisfy K < D with K | D"
        );
        assert!(
            u32::try_from(block_len).is_ok(),
            "FlatBlocks::<MultiChunkEntry>::from_indices: block_len={block_len} must fit in u32"
        );
        assert!(
            d <= usize::from(u16::MAX) + 1,
            "FlatBlocks::<MultiChunkEntry>::from_indices: D={d} must be <= 65536 so coeff_idx fits in u16"
        );

        let chunks_per_ring = d / onehot_k;
        assert!(
            indices.len().is_multiple_of(chunks_per_ring),
            "FlatBlocks::<MultiChunkEntry>::from_indices: index witness length {} must be divisible by D/K={chunks_per_ring}",
            indices.len()
        );
        let total_entries = indices.iter().filter(|opt| opt.is_some()).count();
        let mut blocks = FlatBlocks::<MultiChunkEntry>::with_capacity(num_blocks, total_entries);
        let mut current_block = 0usize;

        for (ring_elem_idx, ring_chunks) in indices.chunks(chunks_per_ring).enumerate() {
            let mut nonzero_coeffs = Vec::with_capacity(ring_chunks.len());

            for (chunk_offset, opt) in ring_chunks.iter().copied().enumerate() {
                let Some(raw) = opt else {
                    continue;
                };
                let idx = raw.as_usize();
                assert!(
                    idx < onehot_k,
                    "FlatBlocks::<MultiChunkEntry>::from_indices: index {idx} out of range for K={onehot_k} in ring {ring_elem_idx}, chunk offset {chunk_offset}"
                );
                let coeff_idx = chunk_offset
                    .checked_mul(onehot_k)
                    .and_then(|base| base.checked_add(idx))
                    .ok_or_else(|| AkitaError::InvalidInput("coefficient index overflow".into()))?;
                debug_assert!(
                    coeff_idx < d,
                    "multi-chunk onehot: coefficient indices inside one ring must stay < D"
                );
                nonzero_coeffs.push(coeff_idx as u16);
            }

            if nonzero_coeffs.is_empty() {
                continue;
            }

            let block_idx = ring_elem_idx / block_len;
            let pos_in_block = (ring_elem_idx % block_len) as u32;
            assert!(
                block_idx >= current_block,
                "multi-chunk onehot: entries must be non-decreasing in block index"
            );
            blocks.push_entry(
                &mut current_block,
                block_idx,
                num_blocks,
                MultiChunkEntry::new(pos_in_block, nonzero_coeffs),
            );
        }

        Ok(blocks.finish_build(current_block, num_blocks))
    }
}

impl FlatBlocks<SingleChunkEntry> {
    /// Build a single-chunk-layout one-hot `FlatBlocks` from an index witness.
    ///
    /// This applies to the common `K >= D && D | K` case, where each
    /// chunk spans one or more ring elements but still contributes
    /// exactly one nonzero coefficient in exactly one ring element.
    ///
    /// Like [`FlatBlocks::<MultiChunkEntry>::from_indices`],
    /// this constructor assumes its caller has already validated the
    /// structural preconditions: `K >= D && D | K`, `block_len` is a
    /// power of two that tiles the ring-element count, `block_len <=
    /// u32::MAX` and `D <= 65536`, and every `Some(idx)` entry in
    /// `indices` is in `[0, onehot_k)`. In production the sole caller is
    /// [`OneHotPoly::build_blocks_inner`].
    ///
    /// # Errors
    ///
    /// Returns an error only if the internal offsets vector (bounded by
    /// `num_blocks + 1`) overflows `u32::MAX`.
    pub(crate) fn from_indices<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        block_len: usize,
        d: usize,
        num_blocks: usize,
    ) -> Result<Self, AkitaError> {
        debug_assert!(
            onehot_k >= d && onehot_k.is_multiple_of(d),
            "FlatBlocks::<SingleChunkEntry>::from_indices: K={onehot_k} and D={d} must satisfy K >= D with D | K"
        );
        debug_assert!(
            u32::try_from(block_len).is_ok(),
            "FlatBlocks::<SingleChunkEntry>::from_indices: block_len={block_len} must fit in u32"
        );
        debug_assert!(
            d <= usize::from(u16::MAX) + 1,
            "FlatBlocks::<SingleChunkEntry>::from_indices: D={d} must be <= 65536 so coeff_idx fits in u16"
        );

        let total_entries = indices.iter().filter(|opt| opt.is_some()).count();
        let mut blocks = FlatBlocks::<SingleChunkEntry>::with_capacity(num_blocks, total_entries);
        let mut current_block = 0usize;

        for (chunk_idx, opt) in indices.iter().copied().enumerate() {
            let Some(raw) = opt else {
                continue;
            };
            let idx = raw.as_usize();
            debug_assert!(
                idx < onehot_k,
                "FlatBlocks::<SingleChunkEntry>::from_indices: index {idx} out of range for K={onehot_k} at position {chunk_idx}"
            );

            let field_pos = chunk_idx
                .checked_mul(onehot_k)
                .and_then(|base| base.checked_add(idx))
                .ok_or_else(|| AkitaError::InvalidInput("field position overflow".into()))?;
            let ring_elem_idx = field_pos / d;
            let coeff_idx = (field_pos % d) as u16;
            let block_idx = ring_elem_idx / block_len;
            let pos_in_block = (ring_elem_idx % block_len) as u32;
            debug_assert!(
                block_idx >= current_block,
                "single-chunk onehot: entries must be non-decreasing in block index"
            );
            blocks.push_entry(
                &mut current_block,
                block_idx,
                num_blocks,
                SingleChunkEntry::new(pos_in_block, coeff_idx),
            );
        }

        Ok(blocks.finish_build(current_block, num_blocks))
    }
}

/// Wide-accumulator multi-chunk inner Ajtai: compute `t = A * s` for a
/// one-hot block.
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
pub(crate) fn inner_ajtai_wide_multi_chunk<F, const D: usize>(
    A: &RingMatrixView<'_, F, D>,
    multi_chunk_entries: &[MultiChunkEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let n_a = A.num_rows();
    let mut t_wide = vec![WideCyclotomicRing::<F::Wide, D>::zero(); n_a];

    for entry in multi_chunk_entries {
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

#[derive(Debug, Clone)]
pub(crate) enum OneHotBlocks {
    SingleChunk(FlatBlocks<SingleChunkEntry>),
    MultiChunk(FlatBlocks<MultiChunkEntry>),
}

impl OneHotBlocks {
    #[inline]
    fn num_blocks(&self) -> usize {
        match self {
            OneHotBlocks::SingleChunk(blocks) => blocks.num_blocks(),
            OneHotBlocks::MultiChunk(blocks) => blocks.num_blocks(),
        }
    }
}

/// One-hot polynomial: sparse witness with at most one nonzero field element
/// per chunk of size `onehot_k`.
///
/// The polynomial is stored layout-agnostically as the flat list of hot
/// indices supplied at construction. Each op takes `block_len` at call time
/// and the per-block bucketing is materialized lazily on the first call and
/// cached for subsequent calls (as a `(block_len, OneHotBlocks)` pair inside
/// a `OnceLock`). That mirrors how [`DensePoly`](crate::DensePoly) accepts `block_len` per op,
/// and keeps `OneHotPoly` free of the commit-layout parameters it used to
/// bake in at construction.
///
/// Generic over `I`: the index type accepted and stored per chunk. Use `u8`
/// when `onehot_k <= 256` to reduce index storage footprint.
#[derive(Debug, Clone)]
pub struct OneHotPoly<F: FieldCore, const D: usize, I: OneHotIndex = usize> {
    pub(crate) num_vars: usize,
    pub(crate) onehot_k: usize,
    /// Per-chunk hot-position indices. `None` denotes an all-zero chunk.
    pub(crate) indices: Vec<Option<I>>,
    pub(crate) total_ring_elems: usize,
    pub(crate) block_cache: OnceLock<(usize, OneHotBlocks)>,
    pub(crate) _marker: PhantomData<(F, I)>,
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
    pub fn new(onehot_k: usize, indices: Vec<Option<I>>) -> Result<Self, AkitaError> {
        if onehot_k == 0 {
            return Err(AkitaError::InvalidInput(
                "onehot_k must be nonzero".to_string(),
            ));
        }
        if !(onehot_k.is_multiple_of(D) || D.is_multiple_of(onehot_k)) {
            return Err(AkitaError::InvalidInput(format!(
                "onehot_k={onehot_k} and D={D} must be nicely matched (one divides the other)"
            )));
        }
        let total_field_elems = indices.len().checked_mul(onehot_k).ok_or_else(|| {
            AkitaError::InvalidInput("onehot total field element count overflow".to_string())
        })?;
        if !total_field_elems.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "onehot total field elements {total_field_elems} is not a power of two"
            )));
        }
        if !total_field_elems.is_multiple_of(D) {
            return Err(AkitaError::InvalidInput(format!(
                "total field elements {total_field_elems} is not divisible by D={D}"
            )));
        }
        let total_ring_elems = total_field_elems / D;
        for (chunk_idx, opt) in indices.iter().copied().enumerate() {
            if let Some(raw) = opt {
                let idx = raw.as_usize();
                if idx >= onehot_k {
                    return Err(AkitaError::InvalidInput(format!(
                        "index {idx} out of range for chunk size K={onehot_k} at position {chunk_idx}"
                    )));
                }
            }
        }
        Ok(Self {
            num_vars: total_field_elems.trailing_zeros() as usize,
            onehot_k,
            indices,
            total_ring_elems,
            block_cache: OnceLock::new(),
            _marker: PhantomData,
        })
    }

    /// Number of field-evaluation slots in each compact one-hot chunk.
    #[inline]
    pub fn onehot_k(&self) -> usize {
        self.onehot_k
    }

    /// Per-chunk hot-position indices. `None` denotes an all-zero chunk.
    #[inline]
    pub fn indices(&self) -> &[Option<I>] {
        &self.indices
    }

    /// Return cached per-block storage, building it on first call for
    /// `block_len`.
    ///
    /// Subsequent calls must pass the same `block_len`; differing `block_len`
    /// is rejected rather than silently rebuilt because it indicates a
    /// layout mismatch between ops on the same polynomial.
    fn blocks_for(&self, block_len: usize) -> Result<&OneHotBlocks, AkitaError> {
        // Fast path: cache already built for this `block_len`.
        if let Some((cached_len, blocks)) = self.block_cache.get() {
            if *cached_len == block_len {
                return Ok(blocks);
            }
            return Err(AkitaError::InvalidInput(format!(
                "OneHotPoly was first used with block_len={cached_len} but is now being \
                 used with block_len={block_len}; all ops on the same \
                 polynomial must share a single layout"
            )));
        }
        // Slow path: build blocks and install them. Validate `block_len`
        // *before* building so the error path is cheap.
        if block_len == 0 || !block_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "block_len={block_len} must be a nonzero power of two"
            )));
        }
        if !self.total_ring_elems.is_multiple_of(block_len) {
            return Err(AkitaError::InvalidSize {
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
            return Err(AkitaError::InvalidInput(format!(
                "OneHotPoly was first used with block_len={cached_len} but is now being \
                 used with block_len={block_len}; all ops on the same \
                 polynomial must share a single layout"
            )));
        }
        Ok(blocks)
    }

    fn build_blocks_inner(&self, block_len: usize) -> Result<OneHotBlocks, AkitaError> {
        // `blocks_for` has already validated that `block_len` is a nonzero
        // power of two and that `total_ring_elems % block_len == 0`, and
        // `OneHotPoly::new` has validated that K, D, and every per-chunk
        // index are in range. Here we only need to compute `num_blocks`
        // for the flat-layout offsets array and check that `block_len`
        // and `D` fit in the packed entry field widths.
        if u32::try_from(block_len).is_err() {
            return Err(AkitaError::InvalidInput(format!(
                "block_len={block_len} exceeds u32::MAX and cannot be packed into an entry"
            )));
        }
        // Coefficient indices inside a ring element are `< D` and get
        // packed as `u16` in the entry types below (see
        // `SingleChunkEntry::coeff_idx` and `MultiChunkEntry::nonzero_coeffs`).
        // Reject out-of-range `D` here rather than silently truncating below.
        if D > usize::from(u16::MAX) + 1 {
            return Err(AkitaError::InvalidInput(format!(
                "D={D} exceeds 65536 and cannot be packed into SingleChunkEntry::coeff_idx / MultiChunkEntry::nonzero_coeffs (both `u16`)"
            )));
        }
        let num_blocks = self.total_ring_elems / block_len;

        // The single-chunk (one-hot-chunk-per-ring-element) layout
        // applies when K >= D && D | K; otherwise fall back to the
        // multi-chunk layout.
        if self.onehot_k >= D && self.onehot_k.is_multiple_of(D) {
            Ok(OneHotBlocks::SingleChunk(
                FlatBlocks::<SingleChunkEntry>::from_indices(
                    self.onehot_k,
                    &self.indices,
                    block_len,
                    D,
                    num_blocks,
                )?,
            ))
        } else {
            Ok(OneHotBlocks::MultiChunk(
                FlatBlocks::<MultiChunkEntry>::from_indices(
                    self.onehot_k,
                    &self.indices,
                    block_len,
                    D,
                    num_blocks,
                )?,
            ))
        }
    }

    fn decompose_fold_single_chunk_onehot(
        &self,
        single_chunk_blocks: &FlatBlocks<SingleChunkEntry>,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> DecomposeFoldWitness<F, D>
    where
        F: CanonicalField,
    {
        let num_blocks = challenges.len().min(single_chunk_blocks.num_blocks());
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let block_views: Vec<&[SingleChunkEntry]> = (0..single_chunk_blocks.num_blocks())
            .map(|i| single_chunk_blocks.block(i))
            .collect();

        let coeff_accum_digit0: Vec<[i32; D]> = {
            let _span = tracing::info_span!("onehot_single_chunk_accumulate").entered();
            single_chunk_onehot_accumulate::<D>(&block_views, challenges, num_blocks, block_len)
        };

        let coeff_accum = if num_digits == 1 {
            coeff_accum_digit0
        } else {
            let _span = tracing::info_span!("onehot_single_chunk_expand").entered();
            let mut expanded = Vec::with_capacity(block_len * num_digits);
            for coeffs in coeff_accum_digit0 {
                expanded.push(coeffs);
                for _ in 1..num_digits {
                    expanded.push([0i32; D]);
                }
            }
            expanded
        };

        let _span = tracing::info_span!("onehot_single_chunk_convert").entered();
        build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
    }

    fn decompose_fold_multi_chunk_onehot(
        &self,
        multi_chunk_blocks: &FlatBlocks<MultiChunkEntry>,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> DecomposeFoldWitness<F, D>
    where
        F: CanonicalField,
    {
        let inner_width = block_len * num_digits;
        let num_blocks = challenges.len().min(multi_chunk_blocks.num_blocks());
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let block_views: Vec<&[MultiChunkEntry]> = (0..multi_chunk_blocks.num_blocks())
            .map(|i| multi_chunk_blocks.block(i))
            .collect();

        let coeff_accum = {
            let _span = tracing::info_span!("onehot_multi_chunk_accumulate").entered();
            multi_chunk_onehot_accumulate::<D>(
                &block_views,
                challenges,
                num_blocks,
                inner_width,
                num_digits,
            )
        };

        let _span = tracing::info_span!("onehot_multi_chunk_convert").entered();
        build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
    }

    fn decompose_fold_batched_single_chunk_onehot(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F, D>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let mut flat_blocks: Vec<&[SingleChunkEntry]> = Vec::with_capacity(total_blocks);
        for poly in polys {
            // `blocks_for` was already called by the public batched entry
            // point; this just reads the cached layout.
            let (_, cached) = poly.block_cache.get()?;
            let OneHotBlocks::SingleChunk(blocks) = cached else {
                return None;
            };
            for i in 0..blocks.num_blocks() {
                flat_blocks.push(blocks.block(i));
            }
        }
        if flat_blocks.is_empty() {
            return None;
        }
        let active_blocks = flat_blocks.len().min(total_blocks);
        let modulus = (-F::one()).to_canonical_u128() + 1;

        let coeff_accum_digit0 = {
            let _span = tracing::info_span!("onehot_single_chunk_accumulate_batched").entered();
            single_chunk_onehot_accumulate::<D>(&flat_blocks, challenges, active_blocks, block_len)
        };

        let coeff_accum = if num_digits == 1 {
            coeff_accum_digit0
        } else {
            let _span = tracing::info_span!("onehot_single_chunk_expand_batched").entered();
            let mut expanded = Vec::with_capacity(block_len * num_digits);
            for coeffs in coeff_accum_digit0 {
                expanded.push(coeffs);
                for _ in 1..num_digits {
                    expanded.push([0i32; D]);
                }
            }
            expanded
        };

        let _span = tracing::info_span!("onehot_single_chunk_convert_batched").entered();
        Some(build_decompose_fold_witness::<F, D>(coeff_accum, modulus))
    }

    fn decompose_fold_batched_multi_chunk_onehot(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F, D>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let mut flat_blocks: Vec<&[MultiChunkEntry]> = Vec::with_capacity(total_blocks);
        for poly in polys {
            let (_, cached) = poly.block_cache.get()?;
            let OneHotBlocks::MultiChunk(blocks) = cached else {
                return None;
            };
            for i in 0..blocks.num_blocks() {
                flat_blocks.push(blocks.block(i));
            }
        }
        if flat_blocks.is_empty() {
            return None;
        }
        let active_blocks = flat_blocks.len().min(total_blocks);
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let inner_width = block_len * num_digits;

        let coeff_accum = {
            let _span = tracing::info_span!("onehot_multi_chunk_accumulate_batched").entered();
            multi_chunk_onehot_accumulate::<D>(
                &flat_blocks,
                challenges,
                active_blocks,
                inner_width,
                num_digits,
            )
        };

        let _span = tracing::info_span!("onehot_multi_chunk_convert_batched").entered();
        Some(build_decompose_fold_witness::<F, D>(coeff_accum, modulus))
    }
}

impl<F, const D: usize, I: OneHotIndex> AkitaPolyOps<F, D> for OneHotPoly<F, D, I>
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

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(block_len)
            .expect("OneHotPoly::fold_blocks: invalid block_len for this polynomial");
        let num_blocks = blocks.num_blocks();
        match blocks {
            OneHotBlocks::SingleChunk(flat) => cfg_into_iter!(0..num_blocks)
                .map(|i| fold_single_chunk_onehot_block(flat.block(i), scalars, block_len))
                .collect(),
            OneHotBlocks::MultiChunk(flat) => cfg_into_iter!(0..num_blocks)
                .map(|i| fold_multi_chunk_onehot_block(flat.block(i), scalars, block_len))
                .collect(),
        }
    }

    fn fold_blocks_ring(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(block_len)
            .expect("OneHotPoly::fold_blocks_ring: invalid block_len for this polynomial");
        let num_blocks = blocks.num_blocks();
        match blocks {
            OneHotBlocks::SingleChunk(flat) => cfg_into_iter!(0..num_blocks)
                .map(|i| fold_single_chunk_onehot_block_ring(flat.block(i), scalars, block_len))
                .collect(),
            OneHotBlocks::MultiChunk(flat) => cfg_into_iter!(0..num_blocks)
                .map(|i| fold_multi_chunk_onehot_block_ring(flat.block(i), scalars, block_len))
                .collect(),
        }
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
            OneHotBlocks::SingleChunk(blocks) => {
                self.decompose_fold_single_chunk_onehot(blocks, challenges, block_len, num_digits)
            }
            OneHotBlocks::MultiChunk(blocks) => {
                self.decompose_fold_multi_chunk_onehot(blocks, challenges, block_len, num_digits)
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
            OneHotBlocks::SingleChunk(_) => Self::decompose_fold_batched_single_chunk_onehot(
                polys, challenges, block_len, num_digits,
            ),
            OneHotBlocks::MultiChunk(_) => Self::decompose_fold_batched_multi_chunk_onehot(
                polys, challenges, block_len, num_digits,
            ),
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
    ) -> Result<FlatDigitBlocks<D>, AkitaError> {
        let blocks = self.blocks_for(block_len)?;
        let a_view = a_matrix.ring_view::<D>(n_a, matrix_stride);
        let num_blocks = blocks.num_blocks();
        let active_a_cols = num_cols_a(block_len, num_digits_commit)?;
        if active_a_cols > a_view.num_cols() {
            return Err(AkitaError::InvalidSetup(format!(
                "active A width {active_a_cols} exceeds setup envelope {}",
                a_view.num_cols()
            )));
        }
        let zero_block_len = n_a.checked_mul(num_digits_open).unwrap();

        let t_all = match blocks {
            OneHotBlocks::SingleChunk(blocks) => {
                let views: Vec<&[SingleChunkEntry]> =
                    (0..blocks.num_blocks()).map(|i| blocks.block(i)).collect();
                column_sweep_ajtai_single_chunk::<F, D>(
                    &a_view,
                    &views,
                    n_a,
                    active_a_cols,
                    num_digits_commit,
                )
            }
            OneHotBlocks::MultiChunk(blocks) => {
                let views: Vec<&[MultiChunkEntry]> =
                    (0..blocks.num_blocks()).map(|i| blocks.block(i)).collect();
                column_sweep_ajtai_multi_chunk::<F, D>(
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
    ) -> Result<CommitInnerWitness<F, D>, AkitaError> {
        let blocks = self.blocks_for(block_len)?;
        let a_view = a_matrix.ring_view::<D>(n_a, matrix_stride);
        let active_a_cols = num_cols_a(block_len, num_digits_commit)?;
        if active_a_cols > a_view.num_cols() {
            return Err(AkitaError::InvalidSetup(format!(
                "active A width {active_a_cols} exceeds setup envelope {}",
                a_view.num_cols()
            )));
        }
        let zero_block_len = n_a.checked_mul(num_digits_open).unwrap();

        let t = match blocks {
            OneHotBlocks::SingleChunk(blocks) => {
                let views: Vec<&[SingleChunkEntry]> =
                    (0..blocks.num_blocks()).map(|i| blocks.block(i)).collect();
                column_sweep_ajtai_single_chunk::<F, D>(
                    &a_view,
                    &views,
                    n_a,
                    active_a_cols,
                    num_digits_commit,
                )
            }
            OneHotBlocks::MultiChunk(blocks) => {
                let views: Vec<&[MultiChunkEntry]> =
                    (0..blocks.num_blocks()).map(|i| blocks.block(i)).collect();
                column_sweep_ajtai_multi_chunk::<F, D>(
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

        Ok(CommitInnerWitness {
            recomposed_inner_rows: t,
            decomposed_inner_rows: t_hat,
        })
    }

    fn direct_root_witness(&self) -> Result<DirectWitnessProof<F>, AkitaError> {
        let total_evals = 1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })?;
        let mut evals = vec![F::zero(); total_evals];
        for (chunk_idx, opt) in self.indices.iter().copied().enumerate() {
            let Some(raw) = opt else {
                continue;
            };
            let field_pos = chunk_idx
                .checked_mul(self.onehot_k)
                .and_then(|base| base.checked_add(raw.as_usize()))
                .ok_or_else(|| {
                    AkitaError::InvalidInput("onehot direct witness index overflow".to_string())
                })?;
            if field_pos >= evals.len() {
                return Err(AkitaError::InvalidInput(format!(
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

fn num_cols_a(block_len: usize, num_digits_commit: usize) -> Result<usize, AkitaError> {
    block_len
        .checked_mul(num_digits_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))
}

fn fold_single_chunk_onehot_block<F: FieldCore, const D: usize>(
    entries: &[SingleChunkEntry],
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

fn fold_multi_chunk_onehot_block<F: FieldCore, const D: usize>(
    entries: &[MultiChunkEntry],
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

fn fold_single_chunk_onehot_block_ring<F: FieldCore, const D: usize>(
    entries: &[SingleChunkEntry],
    scalars: &[CyclotomicRing<F, D>],
    block_len: usize,
) -> CyclotomicRing<F, D> {
    let mut acc = CyclotomicRing::<F, D>::zero();
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            scalars[pos].shift_accumulate_into(&mut acc, entry.coeff_idx());
        }
    }
    acc
}

fn fold_multi_chunk_onehot_block_ring<F: FieldCore, const D: usize>(
    entries: &[MultiChunkEntry],
    scalars: &[CyclotomicRing<F, D>],
    block_len: usize,
) -> CyclotomicRing<F, D> {
    let mut acc = CyclotomicRing::<F, D>::zero();
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            for &coeff_idx in entry.nonzero_coeffs() {
                scalars[pos].shift_accumulate_into(&mut acc, coeff_idx as usize);
            }
        }
    }
    acc
}

fn inner_ajtai_wide_single_chunk<F, const D: usize>(
    a_view: &akita_types::RingMatrixView<'_, F, D>,
    single_chunk_entries: &[SingleChunkEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + akita_field::fields::wide::ReduceTo<F>,
{
    let n_a = a_view.num_rows();
    let mut t_wide = vec![WideCyclotomicRing::<F::Wide, D>::zero(); n_a];

    for entry in single_chunk_entries {
        let col = entry.pos_in_block() * num_digits;
        let coeff_idx = entry.coeff_idx();
        for (a_idx, t_w) in t_wide.iter_mut().enumerate() {
            let a_wide = WideCyclotomicRing::from_ring(&a_view.row(a_idx)[col]);
            a_wide.shift_accumulate_into(t_w, coeff_idx);
        }
    }

    t_wide.into_iter().map(|w| w.reduce()).collect()
}
fn inner_ajtai_wide_single_chunk_tiled<F, const D: usize>(
    a_view: &akita_types::RingMatrixView<'_, F, D>,
    single_chunk_entries: &[SingleChunkEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + akita_field::fields::wide::ReduceTo<F>,
{
    let n_a = a_view.num_rows();
    let mut t = vec![CyclotomicRing::<F, D>::zero(); n_a];

    for tile in single_chunk_entries.chunks(MAX_WIDE_SHIFT_ACCUMULATIONS) {
        let partial = inner_ajtai_wide_single_chunk(a_view, tile, num_digits);
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
    a_view: &akita_types::RingMatrixView<'_, F, D>,
    blocks: &[&[E]],
    n_a: usize,
    num_digits_commit: usize,
    push_entries: impl Fn(&[E], u32, usize, &mut Vec<ColEntry>) + Send + Sync + Copy,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    E: Sync,
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + akita_field::fields::wide::ReduceTo<F>,
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

/// Column-sweep Ajtai commitment for single-chunk one-hot blocks.
///
/// Uses [`column_sweep_core`] for the tiled sweep plus a safety fallback when
/// any block has more than `MAX_WIDE_SHIFT_ACCUMULATIONS` hot entries (the
/// wide accumulator would overflow) and a small-block fast path when
/// `blocks_per_thread` is already L2-friendly.
fn column_sweep_ajtai_single_chunk<F, const D: usize>(
    a_view: &akita_types::RingMatrixView<'_, F, D>,
    single_chunk_blocks: &[&[SingleChunkEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + akita_field::fields::wide::ReduceTo<F>,
{
    let num_blocks = single_chunk_blocks.len();
    debug_assert!(
        active_a_cols <= a_view.num_cols(),
        "active A width exceeds setup envelope"
    );
    if single_chunk_blocks
        .iter()
        .any(|entries| entries.len() > MAX_WIDE_SHIFT_ACCUMULATIONS)
    {
        return cfg_into_iter!(0..num_blocks)
            .map(|i| {
                inner_ajtai_wide_single_chunk_tiled(
                    a_view,
                    single_chunk_blocks[i],
                    num_digits_commit,
                )
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
            .map(|i| {
                inner_ajtai_wide_single_chunk(a_view, single_chunk_blocks[i], num_digits_commit)
            })
            .collect();
    }

    column_sweep_core::<SingleChunkEntry, F, D>(
        a_view,
        single_chunk_blocks,
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

/// Column-sweep Ajtai commitment for multi-chunk one-hot blocks.
///
/// Same two-level tiling as [`column_sweep_ajtai_single_chunk`]; each hot
/// ring element may contribute multiple coefficients, so `push_entries`
/// fans out the `nonzero_coeffs` list into individual `ColEntry` tuples.
fn column_sweep_ajtai_multi_chunk<F, const D: usize>(
    a_view: &akita_types::RingMatrixView<'_, F, D>,
    multi_chunk_blocks: &[&[MultiChunkEntry]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + akita_field::fields::wide::ReduceTo<F>,
{
    let num_blocks = multi_chunk_blocks.len();
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
            .map(|i| inner_ajtai_wide_multi_chunk(a_view, multi_chunk_blocks[i], num_digits_commit))
            .collect();
    }

    column_sweep_core::<MultiChunkEntry, F, D>(
        a_view,
        multi_chunk_blocks,
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

/// Position-parallel accumulation for multi-chunk one-hot witnesses.
///
/// `multi_chunk_blocks` is a slice-of-slices view over per-block entries.
/// Both single-polynomial callers (which collect once via
/// `FlatBlocks::block`) and batched callers (which concatenate slices
/// across polynomials) feed through the same signature.
pub(super) fn multi_chunk_onehot_accumulate<const D: usize>(
    multi_chunk_blocks: &[&[MultiChunkEntry]],
    challenges: &[SparseChallenge],
    num_blocks: usize,
    inner_width: usize,
    num_digits: usize,
) -> Vec<[i32; D]> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(inner_width.max(1));
    let pos_chunk = inner_width.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= inner_width {
                return Vec::new();
            }
            let pos_end = (pos_start + pos_chunk).min(inner_width);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i32; D]; len];
            let mut rotated = vec![[0i16; D]; D];

            for (block_idx, challenge) in challenges.iter().enumerate().take(num_blocks) {
                let entries = multi_chunk_blocks[block_idx];
                let lo = entries.partition_point(|e| e.pos_in_block() * num_digits < pos_start);
                let hi = entries.partition_point(|e| e.pos_in_block() * num_digits < pos_end);
                if lo >= hi {
                    continue;
                }

                fill_rotated_challenge::<D>(&mut rotated, challenge);

                for entry in &entries[lo..hi] {
                    let local_pos = entry.pos_in_block() * num_digits - pos_start;
                    for &ci in entry.nonzero_coeffs() {
                        let rot = &rotated[ci as usize];
                        let dst = &mut acc[local_pos];
                        for k in 0..D {
                            dst[k] += rot[k] as i32;
                        }
                    }
                }
            }

            acc
        })
        .collect();

    chunks.into_iter().flatten().collect()
}

/// Position-partitioned accumulation for single-chunk one-hot witnesses,
/// where each nonzero ring element carries exactly one hot coefficient.
///
/// See [`multi_chunk_onehot_accumulate`] for the block-view convention.
pub(super) fn single_chunk_onehot_accumulate<const D: usize>(
    single_chunk_blocks: &[&[SingleChunkEntry]],
    challenges: &[SparseChallenge],
    num_blocks: usize,
    block_len: usize,
) -> Vec<[i32; D]> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len).max(1);
    let pos_chunk = block_len.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            let pos_end = (pos_start + pos_chunk).min(block_len);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i32; D]; len];
            let mut rotated = vec![[0i16; D]; D];

            for (block_idx, challenge) in challenges.iter().enumerate().take(num_blocks) {
                let entries = single_chunk_blocks[block_idx];
                let lo = entries.partition_point(|entry| entry.pos_in_block() < pos_start);
                let hi = entries.partition_point(|entry| entry.pos_in_block() < pos_end);
                if lo >= hi {
                    continue;
                }

                fill_rotated_challenge::<D>(&mut rotated, challenge);
                for entry in &entries[lo..hi] {
                    let dst = &mut acc[entry.pos_in_block() - pos_start];
                    let rot = &rotated[entry.coeff_idx()];
                    for k in 0..D {
                        dst[k] += rot[k] as i32;
                    }
                }
            }

            acc
        })
        .collect();

    chunks.into_iter().flatten().collect()
}

/// Test-only helpers for this module that need access to private invariants
/// (`FlatBlocks`' monotonic `offsets` / contiguous `entries`, and the
/// non-wide reference path for `inner_ajtai_wide_multi_chunk`).
///
/// Gated on `#[cfg(test)]` so the production binary never sees them.
#[cfg(test)]
pub(crate) mod test_helpers {
    use super::{CyclotomicRing, FlatBlocks, MultiChunkEntry, OneHotIndex, OneHotPoly};
    use akita_field::parallel::*;
    use akita_field::{CanonicalField, FieldCore};

    /// Reference ring-space evaluation for [`OneHotPoly`].
    ///
    /// Computes the global weighted sum `y = Σᵢ scalars[i] · self[i]`.
    /// `scalars` has length >= `num_ring_elems`; excess entries are ignored.
    ///
    /// Only used by tests to cross-check fused prover paths
    /// (e.g. `evaluate_and_fold`) against a straight-line implementation,
    /// so it lives in `test_helpers` rather than on the production trait.
    pub(crate) fn evaluate_ring_onehot<F, const D: usize, I>(
        poly: &OneHotPoly<F, D, I>,
        scalars: &[F],
    ) -> CyclotomicRing<F, D>
    where
        F: FieldCore + CanonicalField,
        I: OneHotIndex,
    {
        let onehot_k = poly.onehot_k;
        cfg_fold_reduce!(
            0..poly.indices.len(),
            || CyclotomicRing::<F, D>::zero(),
            |mut acc: CyclotomicRing<F, D>, chunk_idx: usize| {
                if let Some(raw) = poly.indices[chunk_idx] {
                    let field_pos = chunk_idx * onehot_k + raw.as_usize();
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

    /// Build a flat block layout from a pre-bucketed `Vec<Vec<E>>`.
    ///
    /// The production paths (`FlatBlocks::<SingleChunkEntry>::from_indices`,
    /// `FlatBlocks::<MultiChunkEntry>::from_indices`) stream entries directly
    /// into the flat form without ever materialising per-block `Vec`s.
    /// This constructor exists only so tests that hand-assemble
    /// block-bucketed storage can still feed it into kernels that
    /// consume `FlatBlocks`.
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

    /// Reference (non-wide) multi-chunk inner Ajtai used to cross-check
    /// [`super::inner_ajtai_wide_multi_chunk`].
    ///
    /// Production code always uses the wide accumulator; this simpler
    /// variant only exists so tests can assert the two paths agree.
    #[allow(non_snake_case)]
    pub(crate) fn inner_ajtai_multi_chunk_t_only<F: FieldCore + CanonicalField, const D: usize>(
        A: &[Vec<CyclotomicRing<F, D>>],
        multi_chunk_entries: &[MultiChunkEntry],
        num_digits: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let n_a = A.len();
        let mut t = vec![CyclotomicRing::<F, D>::zero(); n_a];
        for entry in multi_chunk_entries {
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
    use super::test_helpers::inner_ajtai_multi_chunk_t_only;
    use super::*;
    use crate::DensePoly;
    use akita_field::fields::{Fp64, Prime128Offset275, Prime24Offset3};
    use akita_field::RandomSampling;
    use akita_types::FlatMatrix;
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

    fn materialize_onehot_as_dense<F, const D: usize, I>(
        poly: &OneHotPoly<F, D, I>,
    ) -> DensePoly<F, D>
    where
        F: FieldCore + CanonicalField,
        I: OneHotIndex,
    {
        let mut coeffs = vec![CyclotomicRing::<F, D>::zero(); poly.total_ring_elems];
        for (chunk_idx, hot_idx) in poly.indices.iter().copied().enumerate() {
            let Some(raw) = hot_idx else {
                continue;
            };
            let field_pos = chunk_idx * poly.onehot_k + raw.as_usize();
            let ring_idx = field_pos / D;
            let coeff_idx = field_pos % D;
            coeffs[ring_idx].coeffs[coeff_idx] += F::one();
        }
        DensePoly::<F, D>::from_ring_coeffs(coeffs)
    }

    fn test_ring_scalar<F, const D: usize>(seed: u64) -> CyclotomicRing<F, D>
    where
        F: CanonicalField,
    {
        CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
            F::from_canonical_u128_reduced(u128::from(seed + idx as u64 + 1))
        }))
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
        let indices: Vec<Option<usize>> = vec![Some(3), Some(10)];
        let num_blocks = 2;
        let blocks =
            FlatBlocks::<SingleChunkEntry>::from_indices(k, &indices, 4, d, num_blocks).unwrap();

        assert_eq!(blocks.num_blocks(), 2);
        let total_entries: usize = (0..blocks.num_blocks())
            .map(|i| blocks.block(i).len())
            .sum();
        assert_eq!(total_entries, 2, "T=2 nonzero ring elements");

        let block0 = blocks.block(0);
        assert_eq!(block0.len(), 1);
        assert_eq!(block0[0].pos_in_block(), 0);
        assert_eq!(block0[0].coeff_idx(), 3);

        let block1 = blocks.block(1);
        assert_eq!(block1.len(), 1);
        assert_eq!(block1[0].pos_in_block(), 2);
        assert_eq!(block1[0].coeff_idx(), 2);
    }

    #[test]
    fn map_onehot_k_eq_d() {
        // K=4, D=4, T=4 chunks => 16 field elements => 4 ring elements
        // block_len=2 => 2 blocks of 2 ring elements each.
        let k = 4;
        let d = 4;
        let indices: Vec<Option<usize>> = vec![Some(0), Some(2), Some(3), Some(1)];
        let num_blocks = 2;
        let blocks =
            FlatBlocks::<SingleChunkEntry>::from_indices(k, &indices, 2, d, num_blocks).unwrap();

        assert_eq!(blocks.num_blocks(), 2);
        let total_entries: usize = (0..blocks.num_blocks())
            .map(|i| blocks.block(i).len())
            .sum();
        assert_eq!(total_entries, 4, "K=D => every ring element is nonzero");

        let block0 = blocks.block(0);
        assert_eq!(block0.len(), 2);
        assert_eq!(block0[0].pos_in_block(), 0);
        assert_eq!(block0[0].coeff_idx(), 0);
        assert_eq!(block0[1].pos_in_block(), 1);
        assert_eq!(block0[1].coeff_idx(), 2);

        let block1 = blocks.block(1);
        assert_eq!(block1.len(), 2);
        assert_eq!(block1[0].pos_in_block(), 0);
        assert_eq!(block1[0].coeff_idx(), 3);
        assert_eq!(block1[1].pos_in_block(), 1);
        assert_eq!(block1[1].coeff_idx(), 1);
    }

    #[test]
    fn map_onehot_k_lt_d() {
        // K=4, D=8, T=8 chunks => 32 field elements => 4 ring elements
        // block_len=2 => 2 blocks of 2 ring elements each.
        let k = 4;
        let d = 8;
        let indices: Vec<Option<usize>> = vec![
            Some(0),
            Some(2),
            Some(3),
            Some(1),
            Some(0),
            Some(0),
            Some(3),
            Some(3),
        ];
        let num_blocks = 2;
        let blocks =
            FlatBlocks::<MultiChunkEntry>::from_indices(k, &indices, 2, d, num_blocks).unwrap();

        assert_eq!(blocks.num_blocks(), 2);
        let total_entries: usize = (0..blocks.num_blocks())
            .map(|i| blocks.block(i).len())
            .sum();
        assert_eq!(total_entries, 4, "D>K => all ring elements nonzero");

        let block0 = blocks.block(0);
        assert_eq!(block0.len(), 2);
        assert_eq!(block0[0].pos_in_block(), 0);
        assert_eq!(block0[0].nonzero_coeffs(), &[0, 6]);
        assert_eq!(block0[1].pos_in_block(), 1);
        assert_eq!(block0[1].nonzero_coeffs(), &[3, 5]);

        let block1 = blocks.block(1);
        assert_eq!(block1.len(), 2);
        assert_eq!(block1[0].pos_in_block(), 0);
        assert_eq!(block1[0].nonzero_coeffs(), &[0, 4]);
        assert_eq!(block1[1].pos_in_block(), 1);
        assert_eq!(block1[1].nonzero_coeffs(), &[3, 7]);
    }

    #[test]
    #[should_panic(expected = "FlatBlocks::block: block index 1 out of range for 1 blocks")]
    fn flat_blocks_block_panics_on_out_of_range_index() {
        let blocks = super::test_helpers::from_buckets(vec![vec![1u16]]);
        let _ = blocks.block(1);
    }

    #[test]
    fn onehot_poly_rejects_non_divisible_k_d() {
        // K=3 and D=4: neither divides the other. `OneHotPoly::new` must
        // refuse to construct. The nicely-matched K/D invariant is what
        // lets `FlatBlocks::from_{single,multi}_chunk_onehot` skip their
        // own K/D check; this test pins the upstream guard that enforces
        // it.
        type F = Prime24Offset3;
        const D: usize = 4;
        let result = OneHotPoly::<F, D>::new(3, vec![Some(0usize), Some(1)]);
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
            MultiChunkEntry::new(0, vec![1u16, 7, 15]),
            MultiChunkEntry::new(2, vec![0u16, 63]),
        ];

        let a_flat_elems: Vec<CyclotomicRing<F, D>> = a_matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let a_flat = FlatMatrix::from_ring_slice(&a_flat_elems);
        let a_view = a_flat.ring_view::<D>(n_a, block_len * num_digits);
        let ref_result = inner_ajtai_multi_chunk_t_only(&a_matrix, &entries, num_digits);
        let wide_result = inner_ajtai_wide_multi_chunk(&a_view, &entries, num_digits);

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
            MultiChunkEntry::new(0, vec![0u16, 5, 32, 63]),
            MultiChunkEntry::new(1, vec![10u16]),
        ];

        let a_flat_elems: Vec<CyclotomicRing<F, D>> = a_matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let a_flat = FlatMatrix::from_ring_slice(&a_flat_elems);
        let a_view = a_flat.ring_view::<D>(n_a, block_len * num_digits);
        let ref_result = inner_ajtai_multi_chunk_t_only(&a_matrix, &entries, num_digits);
        let wide_result = inner_ajtai_wide_multi_chunk(&a_view, &entries, num_digits);

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
    fn single_chunk_onehot_large_block_uses_safe_accumulator_path() {
        type F = Prime24Offset3;
        const D: usize = 64;

        let block_len = MAX_WIDE_SHIFT_ACCUMULATIONS + 1;
        let max_coeff = F::from_canonical_u128_reduced((1u128 << 24) - 4);
        let dense_ring = CyclotomicRing::from_coefficients([max_coeff; D]);
        let a_matrix = [vec![dense_ring; block_len]];
        let bucket: Vec<SingleChunkEntry> = (0..block_len)
            .map(|pos| SingleChunkEntry::new(pos as u32, (pos % D) as u16))
            .collect();
        let single_chunk_blocks = super::test_helpers::from_buckets(vec![bucket.clone()]);

        let a_flat = FlatMatrix::from_ring_slice(&a_matrix[0]);
        let a_view = a_flat.ring_view::<D>(1, block_len);

        let single_chunk_views: Vec<&[SingleChunkEntry]> = (0..single_chunk_blocks.num_blocks())
            .map(|i| single_chunk_blocks.block(i))
            .collect();
        let got =
            column_sweep_ajtai_single_chunk::<F, D>(&a_view, &single_chunk_views, 1, block_len, 1);
        let expected = inner_ajtai_wide_single_chunk_tiled::<F, D>(&a_view, &bucket, 1);

        assert_eq!(got.len(), 1);
        assert_eq!(got[0], expected);
    }

    #[test]
    fn batched_single_chunk_onehot_decompose_fold_matches_individual_aggregation() {
        type F = Prime24Offset3;
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
        let got = <OneHotPoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_batched(
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
    fn single_chunk_onehot_evaluate_and_fold_matches_factorized_eval() {
        type F = Prime24Offset3;
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
        let expected_eval = super::test_helpers::evaluate_ring_onehot(&poly, &full_scalars);
        assert_eq!(eval, expected_eval);
    }

    #[test]
    fn single_chunk_onehot_ring_fold_matches_dense_materialization() {
        type F = Prime24Offset3;
        const D: usize = 8;

        let poly =
            OneHotPoly::<F, D>::new(16, vec![Some(1usize), None, Some(13usize), Some(7usize)])
                .unwrap();
        let dense = materialize_onehot_as_dense(&poly);
        let block_len = 4usize;
        let fold_scalars = vec![
            test_ring_scalar::<F, D>(10),
            test_ring_scalar::<F, D>(40),
            test_ring_scalar::<F, D>(90),
            test_ring_scalar::<F, D>(120),
        ];

        assert_eq!(
            poly.fold_blocks_ring(&fold_scalars, block_len),
            dense.fold_blocks_ring(&fold_scalars, block_len)
        );
    }

    #[test]
    fn multi_chunk_onehot_evaluate_and_fold_matches_factorized_eval() {
        type F = Prime24Offset3;
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
        let expected_eval = super::test_helpers::evaluate_ring_onehot(&poly, &full_scalars);
        assert_eq!(eval, expected_eval);
    }

    #[test]
    fn multi_chunk_onehot_ring_fold_matches_dense_materialization() {
        type F = Prime24Offset3;
        const D: usize = 16;

        let poly = OneHotPoly::<F, D>::new(
            4,
            vec![
                Some(0usize),
                Some(3usize),
                None,
                Some(2usize),
                Some(1usize),
                None,
                Some(3usize),
                Some(0usize),
                None,
                Some(2usize),
                Some(1usize),
                None,
                Some(3usize),
                None,
                Some(0usize),
                Some(2usize),
            ],
        )
        .unwrap();
        let dense = materialize_onehot_as_dense(&poly);
        let block_len = 2usize;
        let fold_scalars = vec![test_ring_scalar::<F, D>(7), test_ring_scalar::<F, D>(80)];

        assert_eq!(
            poly.fold_blocks_ring(&fold_scalars, block_len),
            dense.fold_blocks_ring(&fold_scalars, block_len)
        );
    }
}
