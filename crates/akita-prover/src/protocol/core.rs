//! Prover core state shared by root orchestration during crate extraction.

use crate::protocol::extension_opening_reduction::{
    ExtensionOpeningReductionProver, ExtensionOpeningReductionTerm,
    SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS,
};
use crate::protocol::ring_switch::{
    ring_switch_build_w, ring_switch_finalize, NextWitnessCommitment, RingSwitchOutput,
};
use crate::protocol::sumcheck::AkitaStage3Prover;
use crate::protocol::sumcheck::{AkitaStage1Prover, AkitaStage2Prover};
use crate::protocol::RingRelationProver;
use crate::{
    ProverOpeningData, ProverTranscriptGrind, RecursiveCommitmentHintCache, RingRelationInstance,
    RingRelationWitness,
};
use akita_algebra::CyclotomicRing;
use akita_config::{bind_transcript_instance_descriptor, CommitmentConfig};
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps, HasWide};
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, Invertible, LiftBase, MulBaseUnreduced, PseudoMersenneField, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{SumcheckInstanceProverExt, SumcheckProof};
use akita_transcript::labels::ABSORB_STAGE3_NEXT_W_EVAL;
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_NEXT_LEVEL_WITNESS_BINDING,
    ABSORB_STAGE2_NEXT_W_EVAL, ABSORB_SUMCHECK_S_CLAIM, ABSORB_TERMINAL_W_REMAINDER,
    CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::FpExtEncoding;
use akita_types::{
    append_claim_values_to_transcript, basis_weights, build_trace_table_scaled,
    check_extension_opening_reduction_output, derive_tensor_extension_opening_claim_from_partials,
    embed_ring_subfield_scalar, embed_ring_subfield_vector, ensure_trace_stage2_supported,
    evaluation_trace_row_weight, prepare_opening_point, recover_ring_subfield_inner_product,
    relation_claim_from_layout_extension, relation_rhs_layout_for, reorder_stage1_coords,
    ring_subfield_packed_extension_opening_point, root_current_w_len,
    root_tensor_projection_enabled, sample_public_row_coefficients, schedule_is_root_direct,
    schedule_num_fold_levels, schedule_root_fold_step, tensor_equality_factor_eval_at_point,
    tensor_equality_factor_evals, tensor_opening_split, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, trace_public_weights_recursive,
    trace_public_weights_root_terms, trace_weight_layout_from_segment, AkitaBatchedProof,
    AkitaBatchedRootProof, AkitaCommitmentHint, AkitaExpandedSetup, AkitaIntermediateStage2Proof,
    AkitaLevelProof, AkitaStage1Proof, AkitaStage2Proof, BasisMode, BlockOrder,
    CleartextWitnessProof, Commitment, ExecutionSchedule, ExtensionOpeningReductionProof,
    LevelParams, OpeningClaims, OpeningClaimsLayout, PreparedOpeningPoint, RelationMatrixRowLayout,
    RingMultiplierOpeningPoint, RingVec, RingView, Schedule, SetupContributionMode,
    SetupPrefixProverRegistry, SetupSumcheckProof, Step, TerminalLevelProof, TraceTable,
};
use std::sync::Arc;

pub(in crate::protocol::core) struct ExtensionOpeningReduction<E: FieldCore> {
    pub(in crate::protocol::core) proof: ExtensionOpeningReductionProof<E>,
    /// EOR final sumcheck claim and transparent-factor evaluation. Retained so
    /// the prepare step can fail-fast cross-check the folded opening against
    /// the reduction output; the verifier enforces the same relation.
    pub(in crate::protocol::core) final_claim: E,
    pub(in crate::protocol::core) final_factor: E,
}

mod extension_opening_reduction;
mod fold;
mod fold_kernels;
mod prove;
mod root_fold;
mod suffix;
#[cfg(test)]
mod tests;

pub(in crate::protocol::core) use extension_opening_reduction::*;
pub(in crate::protocol::core) use fold::{prepare_fold_inner, prove_fold, PreparedFold};
pub(in crate::protocol::core) use fold_kernels::*;
pub use prove::{batched_prove, prove, prove_root_direct};
pub use root_fold::{prove_root, prove_terminal_root_fold_with_params};
pub use suffix::{prove_suffix, SuffixProverState};

/// Output from a single prove level, used to extend proof wire data and state.
pub struct ProveLevelOutput<F: FieldCore, E: FieldCore> {
    /// Fold proof produced at this level.
    pub level_proof: AkitaLevelProof<F, E>,
    /// Suffix prover state for the next level.
    pub next_state: SuffixProverState<F, E>,
}

/// Outcome of the recursive fold suffix after the root level.
pub struct RecursiveSuffixOutcome<F: FieldCore, E: FieldCore> {
    /// Recursive suffix proof steps: intermediate folds followed by terminal.
    pub steps: Vec<AkitaLevelProof<F, E>>,
    /// Total fold-level count reached, including the root level and the
    /// terminal level.
    pub num_levels: usize,
}

pub(in crate::protocol::core) type Stage2ProveResult<E> =
    (SumcheckProof<E>, Vec<E>, AkitaStage2Prover<E>);

pub(in crate::protocol::core) struct Stage3ProveOutput<E: FieldCore> {
    pub(in crate::protocol::core) proof: SetupSumcheckProof<E>,
    pub(in crate::protocol::core) next_w_point: Vec<E>,
    pub(in crate::protocol::core) next_w_eval: E,
}
