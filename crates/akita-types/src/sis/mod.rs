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
//!     fold_position_count, decomposition_digits::num_digits_s_commit(decomp, is_root))?;
//! let norm_s = norm_bound::rounded_up_role_a_inf_norm(
//!     bits, family, d, decomp, &stage1, shape, is_root, k, nu, fold_bits, num_claims, width_s as u64)?;
//! let n_a = ajtai_key::min_secure_rank(
//!     SisTableKey { min_security_bits: bits, family, ring_dimension: d as u32, coeff_linf_bound: norm_s },
//!     width_s as u64)?;
//! let a_key   = AjtaiKeyParams::try_new(bits, family, n_a, width_s, norm_s, d)?;
//! ```
//!
//! Layout/search orchestration (`optimal_m_r_split`, the `*_layout_from_params`
//! builders) stays in `crate::layout`; it composes these primitives but
//! contains no SIS formula of its own.

pub mod ajtai_key;
pub mod decomposition_digits;
pub mod fold_linf_cap;
pub mod fold_witness_grind;
mod generated_sis_table;
pub mod norm_bound;

pub use ajtai_key::{
    ceil_coeff_linf_bucket, ceil_supported_linf_bound, min_secure_rank,
    sis_table_key_for_linf_bound, AjtaiKeyParams, SisModulusFamily, SisTableKey,
    COEFF_LINF_BUCKETS, DEFAULT_SIS_SECURITY_BITS, SUPPORTED_SIS_SECURITY_BITS,
};
pub use decomposition_digits::{
    balanced_digit_abs_max, compute_num_digits_full_field, decomposed_s_block_ring_count,
    decomposed_t_ring_count, decomposed_w_ring_count, fold_witness_representable_linf_bounds,
    num_digits_for_bound, num_digits_open, num_digits_s_commit,
};
pub use fold_witness_grind::{FoldWitnessGrindContract, FOLD_GRIND_PROBE_ORDER_ABSORB};
pub use norm_bound::{
    fold_witness_digit_plan, fold_witness_linf_cap_policy, rademacher_proxy_variance,
    rademacher_proxy_variance_flat_challenges, rademacher_proxy_variance_tensor_challenges,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, weak_binding_inf_norm,
    FoldChallengeNorms, FoldWitnessLinfCapConfig, FoldWitnessLinfCapPolicy, FoldWitnessNorms,
    FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN, FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM,
    FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_DEN, FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_NUM,
    MAX_FOLD_GRIND_ATTEMPTS,
};
