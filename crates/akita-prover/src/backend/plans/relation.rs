//! Public mathematical requests for backend-owned Stage 2 compilation.
use akita_error::AkitaError;
use akita_types::{EvaluationTraceGroupParameters, EvaluationTraceInputs, FpExtEncoding};
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

/// The public relation and challenge history defining its setup contribution.
pub struct RelationWeightRequest<'a, F: Field, E: Field> {
    pub relation: &'a akita_types::RingRelationInstance<F>,
    pub parameters: &'a akita_types::CommittedGroupParams,
    pub alpha: E,
    pub tau1: &'a [E],
    pub claim_coefficients: &'a [E],
    pub opening_source_len: usize,
    pub opening_ring_dimension: usize,
    pub relation_plan: &'a akita_types::RelationRangeImagePlan,
    pub groups: &'a [crate::backend::PreparedRelationGroupPublic<F, E>],
}

/// Scheduled physical-L2 virtualization inputs, before arithmetic compilation.
pub struct PhysicalL2WeightRequest<'a, E: Field> {
    pub plan: &'a akita_types::PhysicalResponsePlan,
    pub point: &'a [E],
    pub batching: &'a [E],
}

/// Public evaluation-trace parameters; contains no witness or CPU tables.
pub struct EvaluationTraceDescription<'a, E: Field> {
    pub group_parameters: Vec<EvaluationTraceGroupParameters<E>>,
    pub digit_witness_domain: akita_types::FlatBooleanDomain,
    pub relation_coefficient_block_len: usize,
    pub witness_layout: &'a akita_types::WitnessLayout,
    pub level_params: &'a akita_types::CommittedGroupParams,
    pub opening_batch: &'a akita_types::OpeningClaimsLayout,
    pub claim_coefficients: &'a [E],
}

impl<'a, E: Field> EvaluationTraceDescription<'a, E> {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new<F>(
        digit_witness_domain: akita_types::FlatBooleanDomain,
        relation_coefficient_block_len: usize,
        witness_layout: &'a akita_types::WitnessLayout,
        level_params: &'a akita_types::CommittedGroupParams,
        opening_batch: &'a akita_types::OpeningClaimsLayout,
        prepared_points: &[akita_types::PreparedOpeningPoint<F, E>],
        claim_coefficients: &'a [E],
        basis: akita_types::BasisMode,
    ) -> Result<Self, AkitaError>
    where
        F: Field + CanonicalEncoding + Ring,
        E: FpExtEncoding<F> + ExtField<F> + Ring,
    {
        let group_parameters =
            akita_types::prepare_evaluation_trace_group_parameters(&EvaluationTraceInputs {
                digit_witness_domain,
                relation_coefficient_block_len,
                witness_layout,
                level_params,
                opening_batch,
                prepared_points,
                claim_coefficients,
                basis,
            })?;
        Ok(Self {
            group_parameters,
            digit_witness_domain,
            relation_coefficient_block_len,
            witness_layout,
            level_params,
            opening_batch,
            claim_coefficients,
        })
    }
}

/// Semantic opening contributions compiled by the selected Stage 2 backend.
pub enum Stage2OpeningDescription<'a, E: Field> {
    EvaluationTrace {
        trace: EvaluationTraceDescription<'a, E>,
        output_scale: E,
    },
    /// One ring-switch authority for both relation weights and linear terms.
    CoefficientPacking(akita_types::CoefficientPackingBatchSemantics<E>),
}

/// Validated public inputs for constructing a backend-owned Stage 2 session.
pub struct ValidatedRelationSessionPlan<'a, F: Field, E: Field> {
    pub(crate) batching_coefficient: E,
    pub(crate) stage1_point: &'a [E],
    pub(crate) range_image_evaluation: E,
    pub(crate) basis: usize,
    pub(crate) relation: RelationWeightRequest<'a, F, E>,
    pub(crate) live_lane_count: usize,
    pub(crate) lane_bits: usize,
    pub(crate) coefficient_bits: usize,
    pub(crate) relation_claim: E,
    pub(crate) physical_l2_claim: E,
    pub(crate) linear_terms: Stage2OpeningDescription<'a, E>,
    pub(crate) linear_opening_claim: E,
    pub(crate) domain_len: usize,
    pub(crate) witness_len: usize,
    pub(crate) physical_l2: Option<PhysicalL2WeightRequest<'a, E>>,
    pub(crate) binary_batching: E,
}

impl<'a, F: Field, E: Field> ValidatedRelationSessionPlan<'a, F, E> {
    pub fn batching_coefficient(&self) -> E {
        self.batching_coefficient
    }
    pub fn stage1_point(&self) -> &[E] {
        self.stage1_point
    }
    pub fn range_image_evaluation(&self) -> E {
        self.range_image_evaluation
    }
    pub fn basis(&self) -> usize {
        self.basis
    }
    pub fn relation(&self) -> &RelationWeightRequest<'a, F, E> {
        &self.relation
    }
    pub fn live_lane_count(&self) -> usize {
        self.live_lane_count
    }
    pub fn lane_bits(&self) -> usize {
        self.lane_bits
    }
    pub fn coefficient_bits(&self) -> usize {
        self.coefficient_bits
    }
    pub fn relation_claim(&self) -> E {
        self.relation_claim
    }
    pub fn physical_l2_claim(&self) -> E {
        self.physical_l2_claim
    }
    pub fn linear_terms(&self) -> &Stage2OpeningDescription<'a, E> {
        &self.linear_terms
    }
    pub fn into_linear_terms(self) -> Stage2OpeningDescription<'a, E> {
        self.linear_terms
    }
    pub fn linear_opening_claim(&self) -> E {
        self.linear_opening_claim
    }
    pub fn domain_len(&self) -> usize {
        self.domain_len
    }
    pub fn witness_len(&self) -> usize {
        self.witness_len
    }
    pub fn physical_l2(&self) -> Option<&PhysicalL2WeightRequest<'a, E>> {
        self.physical_l2.as_ref()
    }
    pub fn binary_batching(&self) -> E {
        self.binary_batching
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        batching_coefficient: E,
        stage1_point: &'a [E],
        range_image_evaluation: E,
        basis: usize,
        relation: RelationWeightRequest<'a, F, E>,
        live_lane_count: usize,
        lane_bits: usize,
        coefficient_bits: usize,
        relation_claim: E,
        physical_l2_claim: E,
        linear_terms: Stage2OpeningDescription<'a, E>,
        linear_opening_claim: E,
        domain_len: usize,
        witness_len: usize,
        physical_l2: Option<PhysicalL2WeightRequest<'a, E>>,
        binary_batching: E,
    ) -> Self {
        Self {
            batching_coefficient,
            stage1_point,
            range_image_evaluation,
            basis,
            relation,
            live_lane_count,
            lane_bits,
            coefficient_bits,
            relation_claim,
            physical_l2_claim,
            linear_terms,
            linear_opening_claim,
            domain_len,
            witness_len,
            physical_l2,
            binary_batching,
        }
    }
}
