//! Akita-specific sumcheck stage implementations.
//!
//! Generic sumcheck proof types, traits, and drivers live in `akita-sumcheck`.
//! This module keeps the digit-range, relation/range-image, and setup instances
//! beside the prover-internal prefix optimizations they depend on.

pub mod akita_stage3;
pub mod digit_range;
pub mod relation_range_image;
pub mod two_round_prefix;

pub use akita_stage3::AkitaStage3Prover;
pub use digit_range::DigitRangeProver;
pub use relation_range_image::RelationRangeImageProver;

// --- Shared helpers ------------------------------------------------------

use akita_field::FieldCore;

/// Fold a pair of adjacent evaluations in a full-width row at a challenge `r`,
/// with implicit zero-padding when the index falls past the end.
#[inline]
pub(crate) fn fold_full_prefix_pair<E: FieldCore>(row: &[E], left: usize, r: E) -> E {
    let v0 = row.get(left).copied().unwrap_or_else(E::zero);
    let v1 = row.get(left + 1).copied().unwrap_or_else(E::zero);
    v0 + r * (v1 - v0)
}
