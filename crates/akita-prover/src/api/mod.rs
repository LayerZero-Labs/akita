//! Public prover API entry points and setup artifacts.

pub mod commitment;
mod prepared_group;
pub mod setup;
pub mod setup_prefix;

pub use commitment::{
    commit, commit_with_params, prepare_commit_inputs, CommitOutput, CommitmentWithHint,
    GroupPosition,
};
pub use prepared_group::{PreparedGroupProveOps, PreparedProverGroup};
pub use setup::AkitaProverSetup;
pub use setup_prefix::commit_setup_prefix;
