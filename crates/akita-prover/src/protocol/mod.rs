//! Prover-side protocol orchestration helpers.

pub mod dispatch;
pub mod extension_opening_reduction;
pub mod flow;
#[cfg(feature = "zk")]
pub(crate) mod masking;
pub mod prg;
pub mod ring_relation;
pub mod ring_relation_witness;
pub mod ring_switch;
pub mod sumcheck;
#[cfg(feature = "zk")]
pub(crate) mod zk_hiding_commit;

pub use akita_types::{RingRelationInstance, RootLevelRawOutput};
pub use flow::{
    build_folded_batched_proof_with_suffix, build_terminal_root_batched_proof,
    prepare_batched_prove_inputs, prove_batched, prove_folded_batched, prove_root_direct,
    prove_root_fold_from_ring_relation, prove_root_fold_with_params, prove_suffix,
    prove_terminal_root_fold_from_ring_relation, prove_terminal_root_fold_with_params,
    PreparedBatchedProveInputs, ProveLevelOutput, RecursiveProverState, RecursiveSuffixOutcome,
    RootLevelProverOutput,
};
pub use ring_relation::{
    compute_relation_quotient, generate_y, RecursiveQuadraticSource, RingRelationProver,
};
pub use ring_relation_witness::RingRelationWitness;
pub use ring_switch::{commit_next_w, RingSwitchOutput};
