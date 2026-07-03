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
use akita_types::dispatch_ring_dim_result;
use akita_types::FpExtEncoding;
use akita_types::{
    append_claim_values_to_transcript, basis_weights, build_trace_table_scaled,
    check_extension_opening_reduction_output, derive_tensor_extension_opening_claim_from_partials,
    embed_ring_subfield_scalar, embed_ring_subfield_vector, ensure_trace_stage2_supported,
    folded_root_supports_opening_shape, prepare_opening_point, recover_ring_subfield_inner_product,
    relation_claim_from_rows_extension, reorder_stage1_coords,
    ring_subfield_packed_extension_opening_point, root_current_w_len, root_direct_schedule,
    root_tensor_projection_enabled, sample_public_row_coefficients, schedule_is_root_direct,
    schedule_num_fold_levels, schedule_root_fold_step, stage2_trace_coeff,
    tensor_equality_factor_eval_at_point, tensor_equality_factor_evals, tensor_opening_split,
    tensor_reduction_claim_from_rows, tensor_row_partials_from_columns,
    trace_public_weights_recursive, trace_public_weights_root_terms,
    trace_weight_layout_from_segment, AkitaBatchedProof, AkitaBatchedRootProof,
    AkitaCommitmentHint, AkitaExpandedSetup, AkitaIntermediateStage2Proof, AkitaLevelProof,
    AkitaStage1Proof, AkitaStage2Proof, BasisMode, BlockOrder, CleartextWitnessProof,
    ExecutionSchedule, ExtensionOpeningReductionProof, FlatRingVec, LevelParams, MRowLayout,
    OpeningClaims, OpeningClaimsLayout, PreparedOpeningPoint, RingCommitment,
    RingMultiplierOpeningPoint, Schedule, SetupContributionMode, SetupPrefixProverRegistry,
    SetupSumcheckProof, Step, TerminalLevelProof, TraceTable,
};
use std::sync::Arc;

pub(in crate::protocol::core) struct ExtensionOpeningReduction<L: FieldCore> {
    pub(in crate::protocol::core) proof: ExtensionOpeningReductionProof<L>,
    /// EOR final sumcheck claim and transparent-factor evaluation. Retained so
    /// the prepare step can fail-fast cross-check the folded opening against
    /// the reduction output; the verifier enforces the same relation.
    pub(in crate::protocol::core) final_claim: L,
    pub(in crate::protocol::core) final_factor: L,
}

mod extension_opening_reduction;
mod fold;
mod prove;
mod root_fold;
mod suffix;
#[cfg(test)]
mod tests;

pub(in crate::protocol::core) use extension_opening_reduction::*;
pub(in crate::protocol::core) use fold::{prepare_fold_inner, prove_fold, PreparedFold};
pub use prove::{batched_prove, prove, prove_root_direct};
pub use root_fold::{prove_root, prove_terminal_root_fold_with_params};
pub use suffix::{prove_suffix, SuffixProverState};

/// Output from a single prove level, used to extend proof wire data and state.
pub struct ProveLevelOutput<F: FieldCore, L: FieldCore> {
    /// Fold proof produced at this level.
    pub level_proof: AkitaLevelProof<F, L>,
    /// Suffix prover state for the next level.
    pub next_state: SuffixProverState<F, L>,
}

/// Outcome of the recursive fold suffix after the root level.
pub struct RecursiveSuffixOutcome<F: FieldCore, L: FieldCore> {
    /// Recursive suffix proof steps: intermediate folds followed by terminal.
    pub steps: Vec<AkitaLevelProof<F, L>>,
    /// Total fold-level count reached, including the root level and the
    /// terminal level.
    pub num_levels: usize,
}

pub(in crate::protocol::core) type Stage2ProveResult<L> =
    (SumcheckProof<L>, Vec<L>, AkitaStage2Prover<L>);

pub(in crate::protocol::core) struct Stage3ProveOutput<L: FieldCore> {
    pub(in crate::protocol::core) proof: SetupSumcheckProof<L>,
    pub(in crate::protocol::core) next_w_point: Vec<L>,
    pub(in crate::protocol::core) next_w_eval: L,
}

fn scalar_opening_from_folded_ring<F, E, const D: usize>(
    folded_ring: &CyclotomicRing<F, D>,
    prepared_point: &PreparedOpeningPoint<F, E, D>,
    inner_opening_point: &[E],
    basis: BasisMode,
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F>,
{
    if <E as ExtField<F>>::EXT_DEGREE == 1 {
        return (*folded_ring * prepared_point.packed_inner_point.sigma_m1())
            .coefficients()
            .first()
            .copied()
            .map(E::lift_base)
            .ok_or_else(|| AkitaError::InvalidInput("empty folded opening ring".to_string()));
    }
    if !D.is_multiple_of(<E as ExtField<F>>::EXT_DEGREE)
        || !(D / <E as ExtField<F>>::EXT_DEGREE).is_power_of_two()
    {
        return Err(AkitaError::InvalidInput(
            "extension-field degree must divide the ring dimension into power-of-two slots"
                .to_string(),
        ));
    }
    let packed_slots = D / <E as ExtField<F>>::EXT_DEGREE;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if inner_opening_point.len() > packed_inner_bits
        && inner_opening_point[packed_inner_bits..]
            .iter()
            .any(|coord| !coord.is_zero())
    {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: inner_opening_point.len(),
        });
    }
    let mut point =
        inner_opening_point[..inner_opening_point.len().min(packed_inner_bits)].to_vec();
    point.resize(packed_inner_bits, E::zero());
    let weights = basis_weights(&point, basis)?;
    let packed_inner_point = embed_ring_subfield_vector::<F, E, D>(
        &weights,
        AkitaError::InvalidInput(
            "root opening point does not encode in the ring-subfield basis".to_string(),
        ),
    )?;
    recover_ring_subfield_inner_product::<F, E, D>(folded_ring, &packed_inner_point)
}

fn row_coefficient_rings<F, L, const D: usize>(
    coefficients: &[L],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    L: FpExtEncoding<F>,
{
    coefficients
        .iter()
        .copied()
        .map(|coefficient| {
            embed_ring_subfield_scalar::<F, L, D>(
                coefficient,
                AkitaError::InvalidInput(
                    "public-row coefficient does not encode in the ring-subfield basis".to_string(),
                ),
            )
        })
        .collect()
}
