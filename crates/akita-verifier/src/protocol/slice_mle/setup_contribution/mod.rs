//! Characterization tests for the setup-contribution evaluation path.
//!
//! There is no production code here: the optimized path lives on
//! [`crate::protocol::ring_switch::RelationMatrixEvaluator`]. This module only
//! holds the unit tests (and their naive direct-evaluation oracle) that
//! cross-check it, so it compiles solely under `cfg(test)`.

#[cfg(test)]
mod tests;
