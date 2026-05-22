//! Offline schedule planner for the Akita polynomial commitment scheme.
//!
//! `<Cfg>`-generic API. The single production entry point is
//! [`find_optimal_schedule`], which runs an exhaustive DP over
//! `(level, w_len, log_basis)` to minimize proof size for the supplied
//! schedule lookup key. The boolean `allow_table_fast_path` controls
//! whether `Cfg::schedule_table()` is consulted before DP — production
//! callers pass `true`; the `gen_schedule_tables` binary passes `false` to
//! regenerate table entries from DP.
//!
//! This crate sits *above* `akita-config` in the dependency graph and is
//! deliberately excluded from the verifier dep tree: production verifier
//! replay never reaches DP code, because preset `Cfg::schedule_plan` impls
//! always materialize from the generated schedule tables that ship with the
//! presets.
//!
//! Cross-crate test fixtures that need a runtime DP fallback (multipoint
//! incidences, presets with `table = None`, setup-matrix sizing iteration)
//! enable the `test-utils` feature and use `test_utils::PlannerCfg` — a
//! `Cfg` wrapper that routes schedule-table misses through
//! [`find_optimal_schedule`]. It is gated off by default so production
//! builds never link it.
//!
//! SIS derivation, `(m, r)` split, and table materialization live in the
//! sibling crate [`akita_derive`].

pub mod schedule_params;
#[cfg(feature = "test-utils")]
pub mod test_utils;

pub use schedule_params::find_optimal_schedule;
