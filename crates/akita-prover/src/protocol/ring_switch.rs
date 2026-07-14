//! Prover-owned helpers for the Akita ring-switch handoff.
use crate::api::commitment::{
    validate_commit_inner_shape, validate_commit_level_params, validate_commit_outer_input_nonempty,
};
use crate::protocol::ring_relation::compute_multi_group_relation_quotient;
use crate::{
    tensor_pack_recursive_witness, CommitmentComputeBackend, RecursiveCommitmentHintCache,
    RecursiveWitnessFlat,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::ring::scalar_powers;
use akita_algebra::CyclotomicRing;
use akita_config::CommitmentConfig;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField, RandomSampling,
};
use akita_transcript::labels::{CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::DigitBlocks;
use akita_types::RingRelationInstance;
use akita_types::{
    r_decomp_levels, AkitaCommitmentHint, AkitaExpandedSetup, FpExtEncoding, LevelParams,
    RelationMatrixRowLayout, RingVec,
};

mod coeffs;
mod commit;
mod evals;
mod finalize;
#[cfg(test)]
mod tests;

pub(crate) use coeffs::PreparedRingSwitchGroup;
pub use coeffs::RingSwitchTerminalArtifacts;
pub use coeffs::{ring_switch_build_w, RingSwitchBuildOutput};
pub use commit::{commit_w, NextWitnessCommitment};
pub use evals::{build_w_evals_compact, compute_relation_weight_evals};
pub use finalize::ring_switch_finalize;

/// D-agnostic output of the ring switch protocol, containing everything
/// needed for sumchecks and level chaining.
pub struct RingSwitchOutput<E: FieldCore> {
    /// Compact evaluation table of w, stored as x-outer/y-inner slices.
    pub w_evals_compact: Vec<i8>,
    /// Boolean x capacity, including only the zero suffix after the physical source.
    pub opening_x_cols: usize,
    /// Tau1-weighted relation table over the full Boolean coefficient domain.
    pub relation_weight_evals: Vec<E>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
    /// Challenge tau0 for F_0 sumcheck.
    pub tau0: Vec<E>,
    /// Challenge tau1 for F_alpha sumcheck.
    pub tau1: Vec<E>,
    /// Basis size b = 2^LOG_BASIS.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: E,
}
