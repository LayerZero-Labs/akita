//! Single source of truth for SIS / Ajtai sizing primitives.
//!
//! Every SIS/Ajtai quantity in the codebase — security-floor tables,
//! secure-rank lookup, weak-binding collision norms, gadget-decomposition digit
//! counts, and per-role committed widths — lives here. No SIS/Ajtai formula may
//! be re-implemented outside this module; callers (planner DP, runtime table
//! expansion, root-layout derivation, the prover's fold-abort check) wire the
//! leaf primitives together explicitly:
//!
//! ```ignore
//! let width_s = decomposition_digits::decomposed_s_block_ring_count(
//!     num_positions_per_block, decomposition_digits::num_digits_inner(decomp, is_root))?;
//! let norm_s = norm_bound::rounded_up_role_a_inf_norm(
//!     policy, table_digest, family, d, log_basis_response, &stage1, shape,
//!     exact_fold_digit_depth, ring_subfield_norm_bound)?;
//! let n_a = ajtai_key::min_secure_rank(
//!     SisTableKey { policy, family, ring_dimension: d as u32, coeff_linf_bound: norm_s },
//!     width_s as u64)?;
//! let inner_commit_matrix = InnerCommitMatrixParams::try_new(bits, family, n_a, width_s, norm_s, d)?;
//! ```
//!
//! Layout/search orchestration stays in `akita-planner`; it composes these
//! primitives but contains no SIS formula of its own.

pub mod ajtai_key;
pub mod compression;
pub mod decomposition_digits;
pub mod fold_linf_cap;
mod generated_sis_table;
pub mod honest_fold_policy;
pub mod norm_bound;

pub use ajtai_key::{
    ceil_coeff_linf_bucket, ceil_supported_linf_bound, min_secure_rank,
    sis_table_key_for_linf_bound, InnerCommitMatrixParams, OpenCommitMatrixParams,
    OuterCommitMatrixParams, ScalarCutoff, SisMatrixRole, SisModulusProfileId, SisRoleCell,
    SisSecurityPolicyId, SisTableDigest, SisTableKey, A_ROLE_RING_DIMS, BD_ROLE_RING_DIMS,
    COEFF_LINF_BUCKETS, DEFAULT_SIS_SECURITY_POLICY, GADGET_COEFF_LINF_ANCHORS, SIS_MATRIX_ROLES,
    SUPPORTED_SIS_SECURITY_POLICIES,
};
pub use decomposition_digits::{
    balanced_digit_abs_max, compute_num_digits_field_width, decomposed_s_block_ring_count,
    decomposed_t_ring_count, decomposed_w_ring_count, fold_witness_representable_linf_bounds,
    num_digits_for_bound, num_digits_inner, num_digits_inner_for_bound, num_digits_open,
    num_digits_setup_prefix_commit, projected_role_ring_count,
};
pub use honest_fold_policy::{
    BalancedSignedDigitFoldPolicy, DigitSnapCalibration, HonestFoldPolicy, HonestFoldPolicySpec,
    HonestFoldSizingQuery, UnitOneHotFoldPolicy,
};
#[cfg(test)]
pub(crate) use norm_bound::{fold_witness_beta_inf, fold_witness_digit_plan};
pub use norm_bound::{
    fold_witness_unsnapped_linf_cap, max_response_linf_for_role_a_collision,
    rademacher_proxy_variance, role_a_collision_inf_norm_for_response_bound,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, weak_binding_inf_norm,
    FoldChallengeNorms, FoldWitnessLinfCapConfig, FoldWitnessNorms,
    FOLD_LINF_FP32_SNAP_MIN_TSTAR_RETAIN_DEN, FOLD_LINF_FP32_SNAP_MIN_TSTAR_RETAIN_NUM,
    FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN, FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM,
    FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_DEN, FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_NUM,
    MAX_FOLD_GRIND_ATTEMPTS,
};
