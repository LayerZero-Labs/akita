//! Public prover API entry points and setup artifacts.

pub mod commitment;
mod scheme;
pub mod setup;
pub mod setup_prefix;

pub use commitment::{
    batched_commit_with_params, batched_commit_with_policy, commit_with_params, commit_with_policy,
    prepare_batched_commit_inputs, prepare_commit_inputs,
};
pub use scheme::CommitmentProver;
pub use setup::AkitaProverSetup;
pub use setup_prefix::{
    commit_setup_prefix, populate_setup_prefix_slots, select_prover_setup_prefix_slot,
};
