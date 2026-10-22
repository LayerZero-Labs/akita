//! Prover core state shared by root orchestration during crate extraction.

pub(crate) use crate::backend::PreparedGroupOpening;
use crate::backend::{
    CommitmentHandleMetadata, OpeningSource, ProverBackend, RecursiveWitnessHandle,
};
use crate::protocol::ring_switch::{ring_switch_finalize, NextWitnessState, RingSwitchOutput};
use crate::protocol::RingRelationProver;
use crate::SetupPrefixProverRegistry;
use crate::{ProverOpeningData, RingRelationInstance};
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_params::dispatch_for_field;
use akita_params::{
    BasisMode, CommittedGroupParams, FoldParams, FoldSchedule, OpeningClaimsLayout,
    PolynomialGroupLayout, SetupContributionMode, TerminalFoldParams,
};
use akita_serialization::AkitaSerialize;
use akita_types::FpExtEncoding;
use akita_types::{
    embed_ring_subfield_scalar, ensure_trace_stage2_supported,
    proof::relation::relation_row_weight, relation_claim_from_compressed_rhs_extension,
    ring_subfield_packed_extension_opening_point, tensor_equality_factor_eval_at_point,
    tensor_opening_split, tensor_reduction_claim_from_rows, tensor_row_partials_from_columns,
    Commitment,
};
use jolt_algebra::CyclotomicRing;
use jolt_algebra::{CanonicalDecode, Fold, Unreduced};
use jolt_algebra::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, PseudoMersenne, Ring};
use jolt_transcript::Sponge;

pub(crate) struct ExtensionOpeningReduction<E: Field> {
    pub(crate) final_claims: Vec<E>,
    /// One transparent factor evaluation per opening group. The application
    /// batches the proof's terminal claims only after the complete opening
    /// payload is fixed.
    pub(crate) final_factors: Vec<E>,
}

pub mod execution;
mod fold;
mod fold_kernels;
#[cfg(all(feature = "zk", feature = "test-support"))]
mod masked;
mod opening_reduction;
mod registry;
mod root;
mod suffix;

pub use execution::{batched_prove, catch_prover_panic};
pub(in crate::protocol::prove) use fold::{prepare_fold, prove_fold, PreparedFold};
pub(in crate::protocol) use fold_kernels::*;
#[cfg(all(feature = "zk", feature = "test-support"))]
pub use masked::{prove_masked_prefix, ResidualOpening};
pub(crate) use opening_reduction::*;
pub(crate) use suffix::FoldPublicState;
pub use suffix::SuffixProverState;

/// Output from a single prove level, used to extend proof wire data and state.
pub struct ProveLevelOutput<F: Field, E: Field, MaterialHandle, WitnessHandle> {
    /// Suffix prover state for the next level.
    pub next_state: SuffixProverState<F, E, MaterialHandle, WitnessHandle>,
    /// The Stage-1 leaf check's inputs, on the ℓ∞ route.
    #[cfg_attr(
        not(all(feature = "zk", feature = "test-support")),
        expect(dead_code, reason = "read only by the zero-knowledge prefix prover")
    )]
    pub(crate) range_leaf: Option<RangeLeafClaim<E>>,
}

/// The decoded inputs of one level's Stage-1 leaf check
/// (`akita_types::check_range_leaf`): the batched leaf, the range image, and
/// Stage 1's running output claim. A zero-knowledge opening lowers the check
/// on them into its auxiliary values.
#[cfg_attr(
    not(all(feature = "zk", feature = "test-support")),
    expect(dead_code, reason = "read only by the zero-knowledge prefix prover")
)]
pub(crate) struct RangeLeafClaim<E> {
    pub(crate) leaf: akita_types::RangeLeaf<E>,
    pub(crate) range_image: E,
    pub(crate) claim: E,
}

pub(in crate::protocol::prove) struct Stage3ProveOutput<E: Field> {
    pub(in crate::protocol::prove) setup_prefix_eval: E,
    pub(in crate::protocol::prove) setup_prefix_point: Vec<E>,
}
