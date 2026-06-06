//! Compatibility adapters for external ecosystems.
//!
//! This is the single seam where `akita-field` interoperates with foreign field
//! trait hierarchies. The only module in the crate that names `jolt_field`.

/// Jolt interop: implementations of Jolt's slim field hierarchy for Akita field
/// types (feature-gated behind `jolt-compat`).
#[cfg(feature = "jolt-compat")]
pub(crate) mod jolt;
