use super::*;

mod role_pricing;

use role_pricing::{projected_collision_role_price, sis_key_at_dimension};

mod recursive;
mod root;
mod setup_prefix;

pub(crate) use recursive::{
    derive_candidate_level_params_all_splits, derive_candidate_level_params_frontier,
};
pub(crate) use root::scalar_root_fold_level_params_candidate;
pub(super) use setup_prefix::derive_setup_prefix_group;
pub(crate) use setup_prefix::planned_next_witness_len;

#[cfg(test)]
use recursive::{
    recursive_candidate_order_key, recursive_split_lower_bound, seed_recursive_split_candidates,
    RecursiveSplitLowerBoundInput,
};

#[cfg(test)]
mod tests;
