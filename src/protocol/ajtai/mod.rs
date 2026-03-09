//! The Ajtai commitment scheme.
//!
//! This module provides the core traits and implementations for the Ajtai commitment scheme,
//! a lattice-based commitment scheme used in the Labrador protocol.

/// Trait definition for the Ajtai commitment scheme.
pub mod ajtai_commit;
/// Coefficient-based (non-NTT) Ajtai commitment implementation.
pub mod coeff;
/// NTT-based Ajtai commitment implementation.
pub mod ntt_backend;
