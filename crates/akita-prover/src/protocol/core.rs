//! Prover core state shared by root orchestration during crate extraction.

use crate::protocol::extension_opening_reduction::{
    ExtensionOpeningReductionGroup, ExtensionOpeningReductionProver, ExtensionOpeningReductionTerm,
};
use crate::protocol::ring_switch::{
    ring_switch_build_w, ring_switch_finalize, NextWitnessState, NextWitnessStateOutput,
    RingSwitchOutput,
};
use crate::protocol::sumcheck::relation_range_image::build_evaluation_trace_weights;
use crate::protocol::sumcheck::AkitaStage3Prover;
use crate::protocol::sumcheck::{AdditionalRelationTerms, RelationRangeImageProver};
use crate::protocol::RingRelationProver;
use crate::{
    PreparedGroupProveOps, PreparedProverGroup, ProverOpeningData, RingRelationInstance,
    RingRelationWitness,
};
use akita_algebra::CyclotomicRing;
use akita_config::{bind_transcript_instance_descriptor, CommitmentConfig};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{SumcheckInstanceProverExt, SumcheckProof};
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EOR_FINAL_CLAIM, ABSORB_EVALUATION_CLAIMS,
    ABSORB_NEXT_LEVEL_WITNESS_BINDING, ABSORB_RANGE_IMAGE_EVALUATION, ABSORB_STAGE2_NEXT_W_EVAL,
    ABSORB_TERMINAL_E_HAT, ABSORB_TERMINAL_W_REMAINDER, CHALLENGE_COMPRESSION_BINARY,
    CHALLENGE_SUMCHECK_BATCH,
};
use akita_transcript::{
    append_ext_field, sample_ext_challenge, Transcript, TranscriptChallengePreview,
};
use akita_types::dispatch_for_field;
use akita_types::FpExtEncoding;
use akita_types::{
    append_claim_values_to_transcript, basis_weights,
    derive_tensor_extension_opening_claim_from_partials, embed_ring_subfield_scalar,
    embed_ring_subfield_vector, ensure_trace_stage2_supported, prepare_opening_point,
    proof::relation::relation_row_weight, recover_ring_subfield_inner_product, reduction_table_len,
    relation_claim_from_compressed_rhs_extension, ring_subfield_packed_extension_opening_point,
    root_input_witness_len, tensor_equality_factor_eval_at_point, tensor_equality_factor_evals,
    tensor_opening_split, tensor_reduction_claim_from_rows, tensor_row_partials_from_columns,
    AkitaBatchedProof, AkitaExpandedSetup, AkitaStage1Proof, AkitaStage2Proof, BasisMode,
    Commitment, CommittedGroupParams, EvaluationTraceInputs, ExtensionOpeningReductionProof,
    FoldLevelProof, FoldParams, FoldSchedule, OpeningClaimsLayout, PolynomialGroupLayout,
    PreparedOpeningPoint, RelationWitnessGeometry, RingMultiplierOpeningPoint, RingVec,
    SetupContributionMode, SetupPrefixProverRegistry, SetupSumcheckProof, TerminalFoldParams,
    TerminalLevelProof,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, PseudoMersenne, Ring};
use jolt_field::{Fold, Unreduced};

use std::sync::Arc;

pub(in crate::protocol::core) struct ExtensionOpeningReduction<E: Field> {
    pub(in crate::protocol::core) proof: ExtensionOpeningReductionProof<E>,
    /// One transparent factor evaluation per opening group. The application
    /// batches the proof's terminal claims only after the complete opening
    /// payload is fixed.
    pub(in crate::protocol::core) final_factors: Vec<E>,
}

mod extension_opening_reduction;
mod fold;
mod fold_kernels;
mod prove;
mod root_fold;
mod root_group;
mod suffix;
#[cfg(test)]
mod tests;

pub(in crate::protocol::core) use extension_opening_reduction::*;
pub(in crate::protocol::core) use fold::{
    prepare_extension_claim_fold, prepare_single_field_fold, prove_fold, ExtensionOpeningSource,
    PreparedFold,
};
pub(in crate::protocol::core) use fold_kernels::*;
pub use prove::{batched_prove, prove};
use root_fold::prove_root;
#[allow(unused_imports)]
pub(crate) use root_group::{
    PreparedCoefficientPackingGroup, PreparedEvaluationTraceGroup, PreparedGroupOpening,
    RootProverGroupMeta, RootProverGroupOpening, RootProverGroupTensor,
};
pub use suffix::{prove_suffix, SuffixProverState};

/// Output from a single prove level, used to extend proof wire data and state.
pub struct ProveLevelOutput<F: Field, E: Field> {
    /// Fold proof produced at this level.
    pub level_proof: FoldLevelProof<F, E>,
    /// Suffix prover state for the next level.
    pub next_state: SuffixProverState<F, E>,
}

/// Outcome of the recursive fold suffix after the root level.
pub struct RecursiveSuffixOutcome<F: Field, E: Field> {
    /// Non-terminal recursive folds following the root.
    pub recursive_folds: Vec<FoldLevelProof<F, E>>,
    /// Required terminal fold.
    pub terminal: TerminalLevelProof<F, E>,
    /// Total fold-level count reached, including the root level and the
    /// terminal level.
    pub num_levels: usize,
}

pub(in crate::protocol::core) type RelationRangeImageProveResult<E> =
    (SumcheckProof<E>, Vec<E>, RelationRangeImageProver<E>);

pub(in crate::protocol::core) struct Stage3ProveOutput<E: Field> {
    pub(in crate::protocol::core) proof: SetupSumcheckProof<E>,
    pub(in crate::protocol::core) setup_prefix_point: Vec<E>,
}
