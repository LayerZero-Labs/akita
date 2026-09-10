//! Public prover API entry points and setup artifacts.

mod prepared_group;
pub mod setup;
pub mod setup_prefix;

pub use prepared_group::{ErasedPreparedProverGroup, PreparedGroupProveOps, PreparedProverGroup};
pub use setup::AkitaProverSetup;
pub use setup_prefix::commit_setup_prefix;
