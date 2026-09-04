//! Akita-specific sumcheck stage implementations.
//!
//! Generic sumcheck proof types, traits, and drivers live in `akita-sumcheck`.
//! This module keeps the digit-range, relation/range-image, and setup instances
//! beside the prover-internal prefix optimizations they depend on.

pub mod akita_stage3;
pub mod digit_range;
mod physical_l2_norm;
pub mod relation_range_image;
pub mod two_round_prefix;

pub use akita_stage3::AkitaStage3Prover;
pub use digit_range::{DigitRangeProver, LowBasisRangeCheckProver};
pub(crate) use relation_range_image::AdditionalRelationTerms;
pub use relation_range_image::RelationRangeImageProver;
pub(crate) use relation_range_image::{DenseRelationWeights, RelationWeightOracle};

// --- Shared helpers ------------------------------------------------------

use jolt_field::Field;

/// Checked two-dimensional partition for parallel reductions over outer rows
/// and fixed-size inner tiles.
pub(crate) struct ReductionTiles {
    inner_len: usize,
    tile_len: usize,
    tiles_per_outer: usize,
    work_items: usize,
}

pub(crate) struct ReductionTile {
    pub(crate) outer: usize,
    pub(crate) inner: std::ops::Range<usize>,
}

impl ReductionTiles {
    pub(crate) fn new(outer_len: usize, inner_len: usize, preferred_tile_len: usize) -> Self {
        assert!(outer_len != 0, "reduction partition needs an outer item");
        assert!(inner_len != 0, "reduction partition needs an inner item");
        assert!(
            preferred_tile_len != 0,
            "reduction partition needs a nonzero tile size"
        );
        let tile_len = preferred_tile_len.min(inner_len);
        let tiles_per_outer = inner_len.div_ceil(tile_len);
        let work_items = outer_len
            .checked_mul(tiles_per_outer)
            .expect("reduction work-item count overflow");
        Self {
            inner_len,
            tile_len,
            tiles_per_outer,
            work_items,
        }
    }

    pub(crate) fn work_items(&self) -> std::ops::Range<usize> {
        0..self.work_items
    }

    pub(crate) fn decode(&self, work_item: usize) -> ReductionTile {
        debug_assert!(work_item < self.work_items);
        let outer = work_item / self.tiles_per_outer;
        let start = (work_item % self.tiles_per_outer) * self.tile_len;
        ReductionTile {
            outer,
            inner: start..(start + self.tile_len).min(self.inner_len),
        }
    }
}

/// Fold adjacent evaluations in a live-prefix row at a challenge `r`, treating
/// indices past the materialized prefix as implicit zero-padding.
#[inline]
pub(crate) fn fold_prefix_pair_with_zero_padding<E: Field>(row: &[E], left: usize, r: E) -> E {
    let v0 = row.get(left).copied().unwrap_or_else(E::zero);
    let v1 = row.get(left + 1).copied().unwrap_or_else(E::zero);
    v0 + r * (v1 - v0)
}

#[cfg(test)]
mod reduction_tile_tests {
    use super::*;

    #[test]
    fn reduction_tiles_cover_each_inner_item_once_per_outer_item() {
        let tiles = ReductionTiles::new(2, 5, 2);
        let decoded = tiles
            .work_items()
            .map(|work_item| tiles.decode(work_item))
            .map(|tile| (tile.outer, tile.inner))
            .collect::<Vec<_>>();

        assert_eq!(
            decoded,
            vec![
                (0, 0..2),
                (0, 2..4),
                (0, 4..5),
                (1, 0..2),
                (1, 2..4),
                (1, 4..5),
            ]
        );
    }
}
