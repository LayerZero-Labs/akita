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
//! let norm_s  = norm_bound::rounded_up_collision_norm_s(family, d, decomp, &stage1, shape, is_root, k, nu)?;
//! let width_s = decomposition_digits::decomposed_s_block_ring_count(
//!     block_len, decomposition_digits::num_digits_s_commit(decomp, is_root))?;
//! let n_a     = ajtai_key::min_secure_rank(family, d as u32, norm_s, width_s as u64)?;
//! let a_key   = AjtaiKeyParams::try_new(family, n_a, width_s, norm_s, d)?;
//! ```
//!
//! Layout/search orchestration (`optimal_m_r_split`, the `*_layout_from_params`
//! builders) stays in `crate::layout`; it composes these primitives but
//! contains no SIS formula of its own.

pub mod ajtai_key;
pub mod decomposition_digits;
mod generated_sis_table;
pub mod norm_bound;

pub use ajtai_key::{ceil_supported_collision, min_secure_rank, AjtaiKeyParams, SisModulusFamily};
pub use decomposition_digits::{
    compute_num_digits, compute_num_digits_full_field, decomp_depths,
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    num_digits_fold, num_digits_for_bound, num_digits_open, num_digits_s_commit,
};
pub use norm_bound::{
    fold_witness_beta, ring_product_infinity_norm_bound, rounded_up_collision_norm_s,
    rounded_up_collision_norm_t, rounded_up_collision_norm_w, FoldChallengeNorms, FoldWitnessNorms,
};
