//! Public prover API entry points and setup artifacts.

pub mod commitment;
mod prepared_group;
pub mod setup;
pub mod setup_prefix;

pub use commitment::{
    commit, for_each_outer_slice_input, resolve_polynomial_group_layout, CommitOutput, GroupContext,
};
pub use prepared_group::{ErasedPreparedProverGroup, PreparedGroupProveOps, PreparedProverGroup};
pub use setup::AkitaProverSetup;
pub use setup_prefix::commit_setup_prefix;
