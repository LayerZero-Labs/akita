//! Public prover API entry points and setup artifacts.

pub mod commitment;
mod scheme;
pub mod setup;

pub use commitment::{commit_with_params, commit_with_policy, prepare_commit_inputs};
pub use scheme::CommitmentProver;
pub use setup::AkitaProverSetup;
