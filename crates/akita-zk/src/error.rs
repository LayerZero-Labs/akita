//! Error aliases for experimental ZK protocols.

use akita_field::AkitaError;

/// Result type used by `akita-zk`.
pub type ZkResult<T> = Result<T, AkitaError>;
