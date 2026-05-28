//! Prover-side protocol orchestration helpers.

pub mod dispatch;
pub mod flow;
#[cfg(feature = "zk")]
pub(crate) mod masking;
pub mod prg;
pub mod quadratic_equation;
pub mod ring_switch;
pub mod sumcheck;
pub(crate) mod validation;

pub use flow::{
    build_final_proof_steps, build_folded_batched_proof_with_suffix,
    build_terminal_root_batched_proof, prepare_batched_prove_inputs, prove_batched_with_policy,
    prove_fold_level_from_quadratic, prove_folded_batched_with_policy,
    prove_recursive_fold_with_params, prove_recursive_level_with_policy,
    prove_recursive_suffix_with_policy, prove_root_direct, prove_root_fold_from_quadratic,
    prove_root_fold_with_params, prove_terminal_fold_level_from_quadratic,
    prove_terminal_recursive_fold_with_params, prove_terminal_recursive_level_with_policy,
    prove_terminal_root_fold_from_quadratic, prove_terminal_root_fold_with_params,
    PreparedBatchedProveInputs, ProveLevelOutput, RecursiveProverState, RecursiveSuffixOutcome,
    RootLevelRawOutput, SuffixLevelOutput, SuffixLevelRequest,
};
pub use quadratic_equation::QuadraticEquation;
pub use ring_switch::{commit_next_w_with_policy, RingSwitchOutput};
