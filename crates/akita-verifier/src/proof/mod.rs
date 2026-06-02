//! Verifier claim preparation and direct-opening checks.

pub(crate) mod claims;
pub(crate) mod direct;

pub use direct::cleartext_witness_opening_matches;
