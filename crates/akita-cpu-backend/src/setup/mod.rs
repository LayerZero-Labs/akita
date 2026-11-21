//! CPU setup construction and portable setup artifacts.

mod pinned;
mod prover;
pub(crate) mod setup_prefix;
pub use pinned::{load_prover_setup, CpuPrefixDeriver, PinnedProverSetup, MAX_HINT_FILE_BYTES};
pub use prover::AkitaProverSetup;
