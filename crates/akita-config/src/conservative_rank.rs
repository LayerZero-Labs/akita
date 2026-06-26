//! Conservative-rank one-hot preset aliases for standalone commitment groups.
//!
//! The conservative B-row widening is selected by
//! [`CommitmentConfig::get_params_for_group_commit`]. These aliases provide the
//! phase-1 public config surface while reusing the existing proof-optimized
//! one-hot policy for all non-grouped operations.

/// fp128 conservative-rank one-hot presets.
pub mod fp128 {
    pub use crate::proof_optimized::fp128::{D128OneHot, D32OneHot, D64OneHot};
}

/// fp64 conservative-rank one-hot presets.
pub mod fp64 {
    pub use crate::proof_optimized::fp64::{D128OneHot, D256OneHot, D32OneHot, D64OneHot};
}

/// fp32 conservative-rank one-hot presets.
pub mod fp32 {
    pub use crate::proof_optimized::fp32::{D128OneHot, D256OneHot, D64OneHot};
}
