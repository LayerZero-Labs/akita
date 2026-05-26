//! Public prover API entry points and setup artifacts.

pub mod commitment;
mod scheme;
pub mod setup;

pub use commitment::{
    batched_commit, batched_commit_with_params, commit, commit_tiered_with_params,
    commit_with_params, prepare_batched_commit_inputs, prepare_commit_inputs,
};
pub use scheme::CommitmentProver;
pub use setup::AkitaProverSetup;
