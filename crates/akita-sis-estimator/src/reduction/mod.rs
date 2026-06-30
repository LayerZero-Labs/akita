//! Reduction cost models.

pub mod adps16;
pub mod delta;

pub use adps16::{adps16_log2_cost, adps16_short_vectors, log2_to_cost_value, ShortVectors};
pub use delta::delta;
