//! Verifier-reachable parameter derivation for the Akita PCS.
//!
//! Two responsibilities, neither of which requires the `CommitmentConfig`
//! trait:
//!
//! - [`derivation`] — SIS-secure level parameter derivation (`(m, r)` split,
//!   security-floor lookups, root + recursive layouts). Used by config
//!   presets to implement the trait hooks they need.
//! - [`materialize`] — turn a [`akita_types::generated::GeneratedScheduleTable`]
//!   entry into a runtime [`akita_types::AkitaSchedulePlan`]. Used by
//!   `Cfg::schedule_plan` impls on table hit.
//!
//! Both layers are verifier-reachable transitively through `Cfg` hooks, so
//! every public function in this crate returns `Result<_, AkitaError>` on
//! malformed inputs and never panics on the verifier replay path.
//!
//! The offline DP search and the table-emitter binaries live in
//! [`akita_planner`], a separate crate that sits *above* `akita-config` in
//! the dependency graph.

pub mod derivation;
pub mod materialize;

pub use derivation::{
    derived_root_commitment_layout_from_params, root_direct_commit_layout,
    root_level_layout_with_log_basis, root_level_params_for_layout_with_log_basis,
    sis_derived_recursive_params_for_layout, sis_derived_root_params_for_layout,
    sis_secure_level_params, SisCollisionBounds, SisRoleWidths,
};
pub use materialize::{schedule_plan_from_table, schedule_plan_from_table_entry, PlanPolicy};
