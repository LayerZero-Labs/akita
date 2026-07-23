//! Characterization tests for the setup-contribution evaluation path.
//!
//! There is no production code here: the optimized path lives on
//! [`crate::protocol::ring_switch::RelationMatrixEvaluator`]. This module only
//! cross-checks it, so it compiles solely under `cfg(test)` (the parent gates
//! the whole submodule). Split by concern to keep each file small:
//!
//! - [`fixtures`] — the shape catalog and the fixture builder + assertions.
//! - [`oracle`] — the naive direct-evaluation reference implementation.
//! - [`tests`] — the `#[test]` cases wiring fixtures against the oracle.

mod fixtures;
mod oracle;
mod tests;
