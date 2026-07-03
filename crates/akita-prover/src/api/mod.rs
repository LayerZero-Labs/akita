//! Public prover API entry points and setup artifacts.

pub mod commitment;
mod scheme;
pub mod setup;
pub mod setup_prefix;

pub use commitment::{
    batched_commit, batched_commit_with_params, commit, commit_final_group, commit_group,
    commit_with_params, prepare_batched_commit_inputs, prepare_commit_inputs, CommitmentWithHint,
    CommittedGroupHandle, CommittedGroupScheduleMeta, CommittedGroupWithHint,
};
pub use scheme::CommitmentProver;
pub use setup::AkitaProverSetup;
pub use setup_prefix::{commit_setup_prefix, SetupPrefixCommitShape};
