//! Prover-side protocol orchestration helpers.

mod fold_grind;
mod prove;
mod ring_relation;
mod ring_switch;

pub use akita_types::RingRelationInstance;

/// A prover channel a fold runs over: the clear transcript, or a masked
/// channel over committed pads at the levels up to a zero-knowledge cut
/// (`specs/whiteout-akita.md` §4.5). Its errors reach the sumcheck driver.
pub(crate) trait ProverFoldChannel<E: jolt_algebra::Field>:
    akita_types::ProverGrindingChannel<Error: Into<jolt_sumcheck::SumcheckError<E>>>
{
}

impl<E, C> ProverFoldChannel<E> for C
where
    E: jolt_algebra::Field,
    C: akita_types::ProverGrindingChannel<Error: Into<jolt_sumcheck::SumcheckError<E>>>,
{
}

pub use prove::execution;

pub use prove::{
    batched_prove, catch_prover_panic, prove_digit_range, DigitRangeOutput, ProveLevelOutput,
    SuffixProverState,
};
#[cfg(all(feature = "zk", feature = "test-support"))]
pub use prove::{prove_masked_prefix, ResidualOpening};
pub use ring_relation::{
    validate_chunked_witness_cfg, validate_prepared_relation_groups, RingRelationProver,
};
pub use ring_switch::RingSwitchOutput;
