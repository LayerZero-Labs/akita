//! Storage-independent recursive-prover primitives.
//!
//! CPU sessions, packed buffers, caches, and execution policy deliberately do
//! not belong in this module.

pub(crate) mod formulas;
pub(crate) mod geometry;
pub(crate) mod lookup_constants;
