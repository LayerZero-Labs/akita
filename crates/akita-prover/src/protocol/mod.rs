//! Prover-side protocol orchestration helpers.

mod fold_grind;
mod prove;
mod ring_relation;
mod ring_switch;

pub use akita_types::RingRelationInstance;

pub use prove::execution;

pub use prove::{
    batched_prove, batched_prove_embedded, catch_prover_panic, prove_digit_range, DigitRangeOutput,
    ProveLevelOutput, SuffixProverState,
};
#[cfg(feature = "zk")]
pub use prove::{prove_masked_prefix, ResidualOpening};
pub use ring_relation::{
    validate_chunked_witness_cfg, validate_prepared_relation_groups, RingRelationProver,
};
pub use ring_switch::RingSwitchOutput;
