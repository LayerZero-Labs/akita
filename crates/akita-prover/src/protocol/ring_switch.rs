//! Prover-owned helpers for the Akita ring-switch handoff.
use crate::api::commitment::{validate_commit_inner_shape, validate_commit_level_params};
use crate::protocol::ring_relation::compute_multi_group_relation_quotient;
use crate::{tensor_pack_recursive_witness, RecursiveWitnessFlat};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_transcript::labels::{CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1};
use akita_transcript::sample_ext_challenge;
use akita_types::{
    r_decomp_levels, AkitaCommitmentHint, AkitaExpandedSetup, CommittedGroupParams,
    CompressionRelationWeights, FpExtEncoding, NegativeBinarySupport, RingVec,
};
use akita_types::{
    CoefficientPackingBatchSemantics, OpeningFamily, RelationRangeImagePlan, RingRelationInstance,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

mod coeffs;
mod commit;
mod evals;
mod finalize;
mod relation_weights;
#[cfg(test)]
mod tests;

pub use coeffs::ring_switch_build_w;
pub(crate) use coeffs::PreparedRingSwitchGroup;
pub use commit::{commit_terminal_w, commit_w, NextWitnessState, NextWitnessStateOutput};
pub(crate) use evals::build_w_evals_compact;
pub(crate) use finalize::ring_switch_finalize;
pub use relation_weights::{
    build_relation_weight_events, RelationSetupSource, RelationWeightContribution,
    RelationWeightEvent, RelationWeightEventInputs, RelationWeightEvents,
    RelationWeightFactorization,
};

/// D-agnostic output of the ring switch protocol, containing everything
/// needed for sumchecks and level chaining.
pub struct RingSwitchOutput<E: Field> {
    /// Compact evaluation table of w, stored as x-outer/y-inner slices.
    pub(crate) w_evals_compact: crate::backend::packed_digits::PackedSignedDigits,
    /// Canonical flat relation-witness domain and coefficient/lane split.
    pub(crate) relation_address_geometry: akita_types::RelationAddressGeometry,
    /// Mode-typed ordinary and compression ring-relation weights.
    pub(crate) relation_weights: crate::protocol::sumcheck::RelationWeightOracle<E>,
    /// Atomic payload-mode state for Stage-2 compression and binary terms.
    pub(crate) compression: RingSwitchCompression<E>,
    /// Low-variable count used by the protocol's Stage-1 tau0 equality point.
    pub digit_range_equality_low_variable_count: usize,
    /// Challenge tau0 for F_0 sumcheck.
    pub tau0: Vec<E>,
    /// Challenge tau1 for F_alpha sumcheck.
    pub tau1: Vec<E>,
    /// Basis size b = 2^LOG_BASIS.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: E,
}

pub(crate) enum RingSwitchCompression<E: Field> {
    Raw,
    QuotientLift {
        weights: CompressionRelationWeights<E>,
        support: NegativeBinarySupport,
    },
    ReducedEvaluation {
        support: NegativeBinarySupport,
    },
}

/// Transcript-complete ring-switch state and the exact relation authority
/// compiled from its freshly sampled challenges.
pub(crate) struct RingSwitchFinalization<E: Field> {
    pub(crate) output: RingSwitchOutput<E>,
    pub(crate) relation_plan: RelationRangeImagePlan,
    pub(crate) opening_semantics: OpeningFamily<(), CoefficientPackingBatchSemantics<E>>,
}
