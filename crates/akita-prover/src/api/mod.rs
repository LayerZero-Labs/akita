//! Public prover API entry points and setup artifacts.

pub mod commitment;
mod scheme;
pub mod setup;
pub mod tiered_setup;

pub use commitment::{
    batched_commit_with_params, batched_commit_with_policy, commit_with_params, commit_with_policy,
    prepare_batched_commit_inputs, prepare_commit_inputs,
    verify_root_direct_commitments_with_params, PreparedBatchedCommitInputs, PreparedCommitInputs,
};
pub use scheme::CommitmentProver;
pub use setup::AkitaProverSetup;
pub use tiered_setup::{
    derive_tiered_setup_commitments, derive_tiered_setup_full_commitments,
    derive_tiered_setup_handle_bundle, TieredSetupHandleBundle,
};
