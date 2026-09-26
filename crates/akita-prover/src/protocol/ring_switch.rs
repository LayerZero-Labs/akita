//! Prover-owned helpers for the Akita ring-switch handoff.
use akita_error::AkitaError;
use akita_types::{
    CoefficientPackingBatchSemantics, OpeningFamily, RelationRangeImagePlan, RingRelationInstance,
};
use akita_types::{CommittedGroupParams, FpExtEncoding};
use jolt_field::{CanonicalEncoding, Field, MulBaseUnreduced, Ring};

pub(crate) enum NextWitnessState<F: Field> {
    OuterPayload(akita_types::RingVec<F>),
    TerminalInnerState,
}

/// D-agnostic output of the ring switch protocol, containing everything
/// needed for sumchecks and level chaining.
pub struct RingSwitchOutput<E: Field, RelationHandle> {
    /// Consumer-owned witness state for Stage 1 and Stage 2.
    pub(crate) relation_handle: RelationHandle,
    /// Public logical length bound to the opaque witness handle.
    pub(crate) witness_len: usize,
    /// Canonical flat relation-witness domain and coefficient/lane split.
    pub(crate) relation_address_geometry: akita_types::RelationAddressGeometry,
    /// Whether the validated payload requires compression binary constraints.
    pub(crate) compressed: bool,
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

/// Transcript-complete ring-switch state and the exact relation authority
/// compiled from its freshly sampled challenges.
pub(crate) struct RingSwitchFinalization<E: Field, RelationHandle> {
    pub(crate) output: RingSwitchOutput<E, RelationHandle>,
    pub(crate) relation_plan: RelationRangeImagePlan,
    pub(crate) opening_semantics: OpeningFamily<(), CoefficientPackingBatchSemantics<E>>,
}

/// Sample the relation challenges and prepare its opaque witness state.
#[tracing::instrument(skip_all, name = "ring_switch_finalize")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ring_switch_finalize<F, E, B>(
    ctx: &crate::backend::OperationCtx<'_, F, B>,
    instance: &RingRelationInstance<F>,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: u32,
    witness_handle: &B::WitnessHandle,
    lp: &CommittedGroupParams,
    opening_source_len: usize,
    opening_ring_dim: usize,
    gamma: Option<&[E]>,
    opening_claim_coefficients: &[E],
    prepared_relation_groups: &[crate::backend::PreparedRelationGroupPublic<F, E>],
) -> Result<RingSwitchFinalization<E, B::RelationHandle>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + Ring + MulBaseUnreduced<F>,
    B: crate::backend::OpaqueRelationWitnessKernel<F, E>,
{
    use crate::backend::RecursiveWitnessHandle;

    let opening_batch = instance.opening_batch();
    crate::protocol::ring_relation::validate_prepared_relation_groups(
        prepared_relation_groups,
        lp,
        opening_batch,
        instance,
    )?;
    let opening_capacity = akita_error::checked::product([opening_source_len, opening_ring_dim])
        .ok_or_else(|| AkitaError::InvalidSetup("opening capacity overflow".into()))?;
    if opening_ring_dim == 0
        || !opening_ring_dim.is_power_of_two()
        || witness_handle.manifest().logical_len() > opening_capacity
    {
        return Err(AkitaError::InvalidInput(
            "witness exceeds scheduled opening capacity".into(),
        ));
    }
    let witness_layout = instance.segment_layout(lp, None)?;
    if witness_handle.manifest().logical_len() != witness_layout.live_coeff_len() {
        return Err(AkitaError::InvalidSize {
            expected: witness_layout.live_coeff_len(),
            actual: witness_handle.manifest().logical_len(),
        });
    }
    let geometry = lp.relation_address_geometry(
        opening_batch,
        instance.extension_degree(),
        opening_ring_dim,
        witness_layout.live_coeff_len(),
    )?;
    let coefficient_count = geometry.relation_coefficient_block_len();
    if !witness_layout
        .live_coeff_len()
        .is_multiple_of(coefficient_count)
    {
        return Err(AkitaError::InvalidSetup(
            "relation witness is not coefficient aligned".into(),
        ));
    }
    let column_bits = geometry.relation_lane_variable_count();
    let coefficient_bits = geometry.relation_coefficient_variable_count();
    if gamma.is_some_and(|values| values.len() != opening_batch.num_total_polynomials()) {
        return Err(AkitaError::InvalidInput(
            "relation batching does not match claim count".into(),
        ));
    }
    let relation_plan = RelationRangeImagePlan::new(
        akita_types::RelationWitnessGeometry::for_level(lp, opening_batch, E::DEGREE)?,
        geometry,
        akita_types::DigitRangePlan::new(1usize << lp.open().digits.log_basis)?,
        witness_layout.clone(),
        opening_batch,
    )?;
    let relation_witness_plan = crate::backend::ValidatedRelationWitnessPlan::new(
        witness_layout.live_coeff_len(),
        coefficient_count,
        1,
        geometry.live_relation_lane_count(),
    );
    let prepared = ctx
        .backend()
        .prepare_relation_witness(witness_handle, &relation_witness_plan)?;
    if prepared.metadata().column_bits() != column_bits
        || prepared.metadata().coefficient_bits() != coefficient_bits
    {
        return Err(AkitaError::InvalidSetup(
            "backend relation geometry differs from the public plan".into(),
        ));
    }
    let alpha = grinding
        .grinded_ext_challenge::<F, E>(akita_types::GrindingSite::RingSwitchAlpha { level })?;
    let tau0 = grinding.grinded_ext_challenges::<F, E>(
        akita_types::GrindingSite::Tau0Point { level },
        column_bits + coefficient_bits,
    )?;
    let tau1 = grinding.grinded_ext_challenges::<F, E>(
        akita_types::GrindingSite::Tau1Point { level },
        lp.relation_row_index_num_vars(opening_batch)?,
    )?;

    let opening_semantics = match prepared_relation_groups
        .first()
        .ok_or(AkitaError::InvalidProof)?
        .kind()
    {
        OpeningFamily::EvaluationTrace(_) => OpeningFamily::EvaluationTrace(()),
        OpeningFamily::SubringCoefficientPacking(_) => {
            let points = prepared_relation_groups
                .iter()
                .enumerate()
                .map(|(index, group)| match group.kind() {
                    OpeningFamily::SubringCoefficientPacking(point) => Ok((index, point)),
                    OpeningFamily::EvaluationTrace(_) => Err(AkitaError::InvalidProof),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let semantics = akita_types::prepare_coefficient_packing_batch_semantics(
                akita_types::CoefficientPackingBatchSemanticInputs {
                    level_params: lp,
                    opening_batch,
                    relation_plan: &relation_plan,
                    relation: instance,
                    prepared_points: &points,
                    alpha,
                    tau1: &tau1,
                    claim_coefficients: opening_claim_coefficients,
                },
            )?;
            OpeningFamily::SubringCoefficientPacking(semantics)
        }
    };
    Ok(RingSwitchFinalization {
        output: RingSwitchOutput {
            relation_handle: prepared.into_relation_handle(),
            witness_len: witness_layout.live_coeff_len(),
            relation_address_geometry: geometry,
            compressed: lp.payload_mode.is_compressed(),
            digit_range_equality_low_variable_count: coefficient_bits,
            tau0,
            tau1,
            b: 1usize << lp.open().digits.log_basis,
            alpha,
        },
        relation_plan,
        opening_semantics,
    })
}
