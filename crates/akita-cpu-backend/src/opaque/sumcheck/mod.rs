//! Akita-specific sumcheck stage implementations.
//!
//! Generic sumcheck proof types, traits, and drivers live in `akita-sumcheck`.
//! Consumer-owned digit-range, relation/range-image, and stage implementations
//! live here; recursive witness storage remains in `opaque::recursive`.

pub(crate) mod digit_range;
mod physical_l2_norm;
pub(crate) mod relation_range_image;
mod stage3;
pub(crate) mod two_round_prefix;

// --- Shared helpers ------------------------------------------------------

use jolt_field::Field;

/// Pairs per parallel work item when a sum-check round has a single outer row.
///
/// Stage 1 x rounds and Stage 2 lane rounds run after every coefficient
/// variable is bound, so their tables are one row and must split along pairs.
/// The tile holds whole split-eq inner blocks: a power-of-two block no larger
/// than the tile divides it, and a block of all live pairs gives one tile.
#[inline]
pub(crate) fn single_row_tile_pairs(block_size: usize) -> usize {
    const SINGLE_ROW_TILE_PAIRS: usize = 1 << 10;
    block_size.max(SINGLE_ROW_TILE_PAIRS)
}

/// Fold adjacent evaluations in a live-prefix row at a challenge `r`, treating
/// indices past the materialized prefix as implicit zero-padding.
#[inline]
pub(crate) fn fold_prefix_pair_with_zero_padding<E: Field>(row: &[E], left: usize, r: E) -> E {
    let v0 = row.get(left).copied().unwrap_or_else(E::zero);
    let v1 = row.get(left + 1).copied().unwrap_or_else(E::zero);
    v0 + r * (v1 - v0)
}
