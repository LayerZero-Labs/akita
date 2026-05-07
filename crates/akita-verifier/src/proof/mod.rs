//! Verifier direct-opening checks.

pub mod direct;

pub use direct::{
    direct_witness_field_elements, direct_witness_opening_matches, verify_root_direct_openings,
};
