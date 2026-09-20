//! CPU setup construction and portable setup artifacts.

mod prover;
pub(crate) mod setup_prefix;
pub use prover::AkitaProverSetup;
pub(crate) use setup_prefix::commit_setup_prefix;
