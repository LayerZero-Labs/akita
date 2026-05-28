//! Offline schedule planner for the Akita polynomial commitment scheme.
//!
//! `<Cfg>`-generic API. The single production entry point is
//! [`find_schedule`], which runs an exhaustive DP over
//! `(level, w_len, log_basis)` to minimize proof size for the supplied
//! schedule lookup key. The `use_lookup` flag controls whether
//! `Cfg::schedule_table()` is consulted before DP — production callers
//! pass `true`; the `gen_schedule_tables` binary passes `false` so that
//! the output is a pure function of `Cfg` (idempotent against a freshly
//! emitted table).
//!
//! Generator metadata (family list, per-family num_vars range, table
//! lookup hook, DP regen entry point) is exposed through the
//! [`generated_families`] module; both the binary and the cross-crate
//! drift-guard test consume the same `ALL_GENERATED_FAMILIES` list so
//! the two cannot drift apart.
//!
//! This crate sits *above* `akita-config` in the dependency graph and is
//! deliberately excluded from the verifier dep tree: production verifier
//! replay never reaches DP code, because preset `Cfg::schedule_plan` impls
//! always materialize from the generated schedule tables that ship with the
//! presets.
//!
//! Cross-crate test fixtures that need a runtime DP fallback enable the
//! `test-utils` feature and use `test_utils::PlannerCfg` — a `Cfg` wrapper
//! that routes schedule-table misses through [`find_schedule`].
//!
//! SIS derivation, `(m, r)` split, and table materialization live in the
//! sibling crate [`akita_derive`].

mod ajtai_params;
pub mod generated_families;
pub mod schedule_params;
#[cfg(feature = "test-utils")]
pub mod test_utils;

pub use schedule_params::find_schedule;
