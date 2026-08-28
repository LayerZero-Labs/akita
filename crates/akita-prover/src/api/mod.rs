//! Public prover API entry points and setup artifacts.

pub mod commitment;
mod prepared_group;
pub mod setup;
pub mod setup_prefix;

pub(crate) use commitment::CommitmentWithHint;
pub use commitment::{commit, resolve_polynomial_group_layout, CommitOutput, GroupContext};
pub use prepared_group::{PreparedGroupProveOps, PreparedProverGroup};
pub use setup::AkitaProverSetup;
pub use setup_prefix::commit_setup_prefix;
