//! Physical polynomial sources, import layouts, and source-specific operations.

pub(crate) mod dense;
pub(crate) mod flat_blocks;
pub(crate) mod onehot;
pub(crate) mod packed_digits;
pub(crate) mod poly;
#[doc(hidden)]
#[allow(missing_docs)]
pub(crate) mod poly_helpers;
pub(crate) mod sparse_ring;

pub use dense::DensePoly;
pub(crate) use onehot::commit_onehot_sources;
pub use onehot::{OneHotIndex, OneHotPoly, OneHotSource};
