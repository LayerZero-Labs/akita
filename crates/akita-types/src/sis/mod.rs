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
//!     block_len, decomposition_digits::num_digits_s_commit(decomp, is_root))?;
//! let (norm_s, n_a) = norm_bound::committed_fold_a_role_rank(
//!     family, d, decomp, &stage1, shape, is_root, k, nu, r_vars, num_claims, width_s as u64)?;
//! let a_key   = AjtaiKeyParams::try_new(family, n_a, width_s, norm_s, d)?;
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
    compute_num_digits_full_field, decomposed_s_block_ring_count, decomposed_t_ring_count,
    decomposed_w_ring_count, fold_witness_verifier_linf_bound, num_digits_fold,
    num_digits_for_bound, num_digits_open, num_digits_s_commit,
};
pub use fold_witness_grind::{FoldWitnessGrindContract, FOLD_GRIND_PROBE_ORDER_ABSORB};
pub use norm_bound::{
    committed_fold_a_role_rank, committed_fold_collision_linf_bound, fold_challenge_norms,
    fold_level_witness_scoring_cost, fold_witness_beta, fold_witness_honest_prover_linf_cap,
    fold_witness_linf_cap_policy, fold_witness_linf_ln_term, fold_witness_linf_tail_bound_sq,
    isqrt_ceil, l2_sq_from_linf, ring_product_infinity_norm_bound, rounded_up_collision_linf_t,
    rounded_up_collision_linf_tiered_commitment, rounded_up_collision_linf_w, FoldChallengeNorms,
    FoldWitnessLinfCapConfig, FoldWitnessLinfCapPolicy, FoldWitnessNorms,
    FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN, FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM,
    MAX_FOLD_GRIND_ATTEMPTS,
};
