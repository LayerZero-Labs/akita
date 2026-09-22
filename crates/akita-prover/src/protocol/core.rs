//! Prover core state shared by root orchestration during crate extraction.

use crate::protocol::extension_opening_reduction::{
    ExtensionOpeningReductionGroup, ExtensionOpeningReductionProver, ExtensionOpeningReductionTerm,
};
use crate::protocol::ring_switch::{
    ring_switch_build_w, ring_switch_finalize_native, NextWitnessState, NextWitnessStateOutput,
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
use akita_config::{transcript_instance_descriptor, CommitmentConfig};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::dispatch_for_field;
use akita_types::FpExtEncoding;
use akita_types::{
    basis_weights, derive_tensor_extension_opening_claim_from_partials, embed_ring_subfield_scalar,
    embed_ring_subfield_vector, ensure_trace_stage2_supported, prepare_opening_point,
    proof::relation::relation_row_weight, recover_ring_subfield_inner_product, reduction_table_len,
    relation_claim_from_compressed_rhs_extension, ring_subfield_packed_extension_opening_point,
    tensor_equality_factor_eval_at_point, tensor_equality_factor_evals, tensor_opening_split,
    tensor_reduction_claim_from_rows, tensor_row_partials_from_columns, AkitaExpandedSetup,
    BasisMode, Commitment, CommittedGroupParams, EvaluationTraceInputs, FoldParams, FoldSchedule,
    OpeningClaimsLayout, PolynomialGroupLayout, PreparedOpeningPoint, RingMultiplierOpeningPoint,
    RingVec, SetupContributionMode, SetupPrefixProverRegistry, TerminalFoldParams,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, PseudoMersenne, Ring};
use jolt_field::{Fold, Unreduced};

use std::sync::Arc;

#[derive(Clone, Copy)]
pub(in crate::protocol) struct ExtensionOpeningReductionBinding<'a, E: Field> {
    pub(in crate::protocol) final_claims: &'a [E],
    pub(in crate::protocol) final_factors: &'a [E],
}

impl<'a, E: Field> From<&'a NativeExtensionOpeningReduction<E>>
    for ExtensionOpeningReductionBinding<'a, E>
{
    fn from(reduction: &'a NativeExtensionOpeningReduction<E>) -> Self {
        Self {
            final_claims: &reduction.final_claims,
            final_factors: &reduction.final_factors,
        }
    }
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

/// Stateless coordinator for the proving stages that share one stack selector.
///
/// The transcript, setup, schedule, and all protocol progress stay explicit in
/// method arguments and return values.
struct ProverExecutor<'stack, Stacks: ?Sized> {
    stacks: &'stack Stacks,
}

pub(in crate::protocol::core) use extension_opening_reduction::*;
pub(in crate::protocol::core) use fold::{
    prepare_extension_claim_fold_native, prepare_single_field_fold_native, prove_fold_native,
    ExtensionOpeningSource, PreparedFold,
};
pub(in crate::protocol) use fold_kernels::*;
pub use prove::batched_prove;
#[allow(unused_imports)]
pub(crate) use root_group::{
    PreparedCoefficientPackingGroup, PreparedEvaluationTraceGroup, PreparedGroupOpening,
    RootProverGroupMeta, RootProverGroupOpening, RootProverGroupTensor,
};
pub use suffix::SuffixProverState;

pub struct NativeProveLevelOutput<F: Field, E: Field, S> {
    pub next_state: SuffixProverState<F, E, S>,
}

pub struct NativeRecursiveSuffixOutcome {
    pub num_levels: usize,
}
