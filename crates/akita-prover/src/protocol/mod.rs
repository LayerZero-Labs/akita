//! Prover-side protocol orchestration helpers.

pub mod core;
pub mod extension_opening_reduction;
pub mod fold_grind;
pub mod prg;
pub mod ring_relation;
pub mod ring_relation_witness;
pub mod ring_switch;
pub mod sumcheck;

pub use akita_types::RingRelationInstance;
pub use core::{
    batched_prove, prove, prove_root, prove_suffix, ProveLevelOutput, RecursiveSuffixOutcome,
    SuffixProverState,
};
pub use fold_grind::ProverTranscriptGrind;
pub use ring_relation::RingRelationProver;
pub use ring_relation_witness::RingRelationWitness;
pub use ring_switch::{commit_w, RingSwitchOutput};
