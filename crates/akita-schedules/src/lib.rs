//! Feature-gated generated schedule tables for the Akita polynomial commitment scheme.
//!
//! This crate holds static table data and `*_table()` constructors only. FoldSchedule
//! resolution, identity validation, and DP fallback live in `akita-planner`; preset
//! wiring lives in `akita-config`.

pub mod generated;

pub use generated::*;
