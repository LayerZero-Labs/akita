//! Public prover API entry points and setup artifacts.

pub mod commitment;
mod scheme;
pub mod setup;

pub use commitment::{
    batched_commit_with_params, batched_commit_with_policy, commit_with_params, commit_with_policy,
    prepare_batched_commit_inputs, prepare_commit_inputs,
    verify_root_direct_commitments_with_params, PreparedBatchedCommitInputs, PreparedCommitInputs,
};
pub use scheme::CommitmentProver;
pub use setup::AkitaProverSetup;
