//! Prover core state shared by root orchestration during crate extraction.

pub(crate) use crate::backend::PreparedGroupOpening;
use crate::backend::{
    CommitmentHandleMetadata, OpeningSource, ProverBackend, RecursiveWitnessHandle,
};
use crate::protocol::ring_switch::{ring_switch_finalize, NextWitnessState, RingSwitchOutput};
use crate::protocol::RingRelationProver;
use crate::SetupPrefixProverRegistry;
use crate::{ProverOpeningData, RingRelationInstance};
use akita_algebra::CyclotomicRing;
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::dispatch_for_field;
use akita_types::FpExtEncoding;
use akita_types::{
    embed_ring_subfield_scalar, ensure_trace_stage2_supported,
    proof::relation::relation_row_weight, relation_claim_from_compressed_rhs_extension,
    ring_subfield_packed_extension_opening_point, tensor_equality_factor_eval_at_point,
    tensor_opening_split, tensor_reduction_claim_from_rows, tensor_row_partials_from_columns,
    BasisMode, Commitment, CommittedGroupParams, FoldParams, FoldSchedule, OpeningClaimsLayout,
    PolynomialGroupLayout, SetupContributionMode, TerminalFoldParams,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, PseudoMersenne, Ring};
use jolt_field::{Fold, Unreduced};

pub(crate) struct ExtensionOpeningReduction<E: Field> {
    pub(crate) final_claims: Vec<E>,
    /// One transparent factor evaluation per opening group. The application
    /// batches the proof's terminal claims only after the complete opening
    /// payload is fixed.
    pub(crate) final_factors: Vec<E>,
}

mod fold;
mod fold_kernels;
mod opening_reduction;
mod root;
mod suffix;

pub(in crate::protocol::prove) use fold::{prepare_fold, prove_fold, PreparedFold};
pub(in crate::protocol) use fold_kernels::*;
pub(crate) use opening_reduction::*;
pub use root::batched_prove;
pub use suffix::SuffixProverState;

/// Output from a single prove level, used to extend proof wire data and state.
pub struct ProveLevelOutput<F: Field, E: Field, MaterialHandle, WitnessHandle> {
    /// Suffix prover state for the next level.
    pub next_state: SuffixProverState<F, E, MaterialHandle, WitnessHandle>,
}

/// Outcome of the recursive fold suffix after the root level.
pub struct RecursiveSuffixOutcome {
    /// Total fold-level count reached, including the root level and the
    /// terminal level.
    pub num_levels: usize,
}

pub(in crate::protocol::prove) struct Stage3ProveOutput<E: Field> {
    pub(in crate::protocol::prove) setup_prefix_eval: E,
    pub(in crate::protocol::prove) setup_prefix_point: Vec<E>,
}
