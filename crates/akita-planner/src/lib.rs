//! Proof-size parameter planner for the Akita polynomial commitment scheme.
//!
//! Free-function API: no public trait. Callers (today, `akita-config`) build
//! a value-typed [`SearchOptions`] or [`PlanPolicy`] from their
//! `CommitmentConfig` shape and call into the planner directly.
//!
//! Two production entry points:
//! - [`find_optimal_schedule`] — exhaustive DP over `(level, w_len,
//!   log_basis)`, security-aware against the generated SIS-floor tables.
//!   Consulted on table miss from `CommitmentConfig::get_params_for_prove`.
//! - [`schedule_plan_from_table`] — materialize a generated
//!   `GeneratedScheduleTable` entry into a runtime [`akita_types::Schedule`]
//!   under a config-supplied [`PlanPolicy`].
//!
//! SIS-floor lookups defer to `akita_types::generated::sis_floor`. The
//! planner crate carries no separate SIS-floor table.

pub mod derivation;
pub mod materialize;
pub mod schedule_params;

pub use derivation::{
    derived_root_commitment_layout_from_params, sis_derived_recursive_params_for_layout,
    sis_derived_root_params_for_layout, sis_secure_level_params, SisCollisionBounds, SisRoleWidths,
};
pub use materialize::{schedule_plan_from_table, schedule_plan_from_table_entry, PlanPolicy};
pub use schedule_params::{
    find_optimal_schedule, find_optimal_schedule_from_scratch, SearchOptions,
};
