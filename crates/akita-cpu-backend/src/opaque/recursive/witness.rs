//! Recursive witness helpers for later Akita prove levels.
//!
//! Recursive levels carry a flat digit witness that is re-chunked under the
//! current ring dimension on demand.

#![allow(missing_docs, clippy::missing_errors_doc, clippy::missing_panics_doc)]

include!("witness/handles_and_relation.rs");
include!("witness/opening_and_flat.rs");

mod coefficient_packing;

#[cfg(test)]
mod tests;

pub(crate) use coefficient_packing::suffix_witness_coefficient_packing_partials;
