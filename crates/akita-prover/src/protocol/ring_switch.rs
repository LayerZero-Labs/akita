//! Prover-owned helpers for the Akita ring-switch handoff.
use crate::api::commitment::{
    validate_commit_inner_shape, validate_commit_level_params, validate_commit_outer_input_nonempty,
};
use crate::protocol::RingRelationWitness;
use crate::{
    tensor_pack_recursive_witness, CommitmentComputeBackend, RecursiveCommitmentHintCache,
    RecursiveWitnessFlat,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::CyclotomicRing;
use akita_challenges::Challenges;
use akita_config::CommitmentConfig;
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField, Invertible,
    LiftBase, MulBase, RandomSampling,
};
use akita_transcript::labels::{CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::dispatch_ring_dim_result;
use akita_types::DigitBlocks;
use akita_types::RingRelationInstance;
use akita_types::{
    gadget_row_scalars, r_decomp_levels, ring_relation_segment_lengths, AkitaCommitmentHint,
    AkitaExpandedSetup, FpExtEncoding, LevelParams, MRowLayout, RingMultiplierOpeningPoint,
    RingOpeningPoint, RingRelationOpeningCounts, RingVec,
};

mod coeffs;
mod commit;
mod evals;
mod finalize;
#[cfg(test)]
mod tests;

pub use coeffs::RingSwitchTerminalArtifacts;
pub use coeffs::{ring_switch_build_w, RingSwitchBuildOutput};
pub use commit::{commit_w, NextWitnessCommitment};
pub use evals::{
    build_relation_weight_evals, build_w_evals_compact, compute_relation_column_weights,
    RelationWeightTraceBuild,
};
pub use finalize::{ring_switch_finalize, RelationWeightFinalizeInputs};

/// D-agnostic output of the ring switch protocol, containing everything
/// needed for sumchecks and level chaining.
pub struct RingSwitchOutput<E: FieldCore> {
    /// Compact evaluation table of w, stored as x-outer/y-inner slices.
    pub w_evals_compact: Vec<i8>,
    /// Physical x width before zero-extension to the next power of two.
    pub live_x_cols: usize,
    /// Materialized relation-weight polynomial evaluations for stage-2.
    pub relation_weight_evals: Vec<E>,
    /// Stage-2 relation claim `V_alpha` (includes EvaluationTrace row).
    pub relation_weight_claim: E,
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
