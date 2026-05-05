//! Verifier claim preparation and direct-opening checks.

pub mod claims;
pub mod direct;

pub use claims::{prepare_verifier_claims, PreparedVerifierClaims};
pub use direct::{
    direct_witness_field_elements, direct_witness_opening_matches, verify_root_direct_openings,
};
