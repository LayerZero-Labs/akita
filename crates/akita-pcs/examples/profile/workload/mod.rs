mod batched_onehot;
mod cache;
mod claims;
mod multi_group;
mod profile_data;
mod proof_size;
mod single_group;

pub(crate) use batched_onehot::run_batched_onehot;
use cache::assert_profile_ntt_cache_did_not_grow;
use claims::{prover_claims, verifier_claims};
pub(crate) use multi_group::{profile_setup_contribution_mode, run_recursive_multi_group_onehot};
pub(crate) use profile_data::onehot_k_for_num_vars;
use profile_data::{
    degree_one_claim_point_to_base, dense_lagrange_opening_from_evals, make_profile_onehot_poly,
    onehot_lagrange_opening, opening_from_poly, random_claim_point,
};
use proof_size::{
    assert_observed_proof_size, planned_payload_bytes, report_proof_size_against_planner,
};
pub(crate) use single_group::{run_dense_for, run_onehot};
