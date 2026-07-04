//! Public prover API entry points and setup artifacts.

pub mod commitment;
mod scheme;
pub mod setup;
pub mod setup_prefix;

pub use commitment::{
    batched_commit, batched_commit_with_params, commit_final_group, prepare_batched_commit_inputs,
    CommitmentWithHint,
};
pub use scheme::CommitmentProver;
pub use setup::AkitaProverSetup;
pub use setup_prefix::commit_setup_prefix;
