use super::*;

use akita_schedules::planner_support::sis_key_at_dimension;

mod recursive;
mod setup_prefix;

pub(crate) use akita_schedules::planner_support::{
    scalar_root_fold_level_params_candidate, RingDimensionCandidate,
};
pub(crate) use recursive::{
    derive_candidate_level_params, derive_candidate_level_params_all_splits,
};
pub(super) use setup_prefix::derive_setup_prefix_group;
pub(crate) use setup_prefix::planned_next_witness_len;
pub use setup_prefix::suffix_opening_layout;

#[cfg(test)]
use recursive::{
    recursive_candidate_order_key, recursive_split_lower_bound, seed_recursive_split_candidates,
    RecursiveSplitLowerBoundInput,
};

#[cfg(test)]
mod tests;
