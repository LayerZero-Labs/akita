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

/// Whether an installed fault-injection fault disables the prover self-checks
/// that would otherwise refuse to emit the faulty proof. Always `false`
/// without the `fault-injection` feature.
#[inline]
pub(crate) fn fault_skips_self_checks() -> bool {
    #[cfg(feature = "fault-injection")]
    {
        crate::fault_injection::skips_self_checks()
    }
    #[cfg(not(feature = "fault-injection"))]
    {
        false
    }
}
