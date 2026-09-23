//! Prover-side protocol orchestration helpers.

mod fold_grind;
mod prove;
mod ring_relation;
mod ring_switch;

pub use akita_types::RingRelationInstance;
pub use prove::{batched_prove, ProveLevelOutput, RecursiveSuffixOutcome, SuffixProverState};
pub use ring_relation::{
    validate_chunked_witness_cfg, validate_prepared_relation_groups, RingRelationProver,
};
pub use ring_switch::RingSwitchOutput;
