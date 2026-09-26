use super::*;
use akita_algebra::offset_eq::{materialize_eq_tensor_left, OffsetEqWindow};
use akita_prover::backend::ValidatedRelationSessionPlan;

pub(crate) struct CompiledStage2Weights<E: Field> {
    pub ordinary: RelationWeightDescription<E>,
    pub linear: Vec<(usize, E)>,
    pub binary_intervals: Vec<Range<usize>>,
}

pub(crate) fn compile_stage2_weights<F, E>(
    setup: &AkitaExpandedSetup<F>,
    plan: &ValidatedRelationSessionPlan<'_, F, E>,
) -> Result<CompiledStage2Weights<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F> + Ring,
{
    let request = plan.relation();
    let parameters = request.parameters;
    let instance = request.relation;
    if parameters.payload_mode.is_compressed()
        && parameters.ring_relation_mode == akita_types::RingRelationMode::QuotientLift
        && akita_error::checked::product([
            request.opening_source_len,
            request.opening_ring_dimension,
        ])
        .ok_or(AkitaError::InvalidProof)?
            != plan.domain_len()
    {
        return Err(AkitaError::InvalidInput(
            "Stage 2 opening domain differs from its relation".into(),
        ));
    }
    let layout = instance.segment_layout(parameters, Some(plan.witness_len()))?;
    if layout.live_coeff_len() != plan.witness_len()
        || request.relation_plan.digit_witness_domain().domain_len() != plan.domain_len()
    {
        return Err(AkitaError::InvalidInput(
            "Stage 2 geometry differs from its relation".into(),
        ));
    }
    let points = request
        .groups
        .iter()
        .enumerate()
        .filter_map(|(index, group)| match group.kind() {
            OpeningFamily::SubringCoefficientPacking(point) => Some((index, point)),
            OpeningFamily::EvaluationTrace(_) => None,
        })
        .collect::<Vec<_>>();
    let opening_points = if points.is_empty() {
        OpeningFamily::EvaluationTrace(())
    } else if points.len() == request.groups.len() {
        OpeningFamily::SubringCoefficientPacking(points.as_slice())
    } else {
        return Err(AkitaError::InvalidProof);
    };
    let ordinary = match parameters.ring_relation_mode {
        akita_types::RingRelationMode::QuotientLift => {
            let (weights, _) = build_relation_lane_weights(RelationLaneWeightInputs {
                setup: RelationSetupSource::Matrix(setup),
                instance,
                alpha: request.alpha,
                level_params: parameters,
                relation_row_point: request.tau1,
                claim_coefficients: request.claim_coefficients,
                opening_source_len: request.opening_source_len,
                opening_ring_dim: request.opening_ring_dimension,
                relation_plan: request.relation_plan,
                opening_points,
            })?;
            RelationWeightDescription::QuotientFactored(weights.into_factorization()?)
        }
        akita_types::RingRelationMode::ReducedEvaluation => {
            if !points.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            build_reduced_dense_relation_weights(
                setup,
                instance,
                request.alpha,
                parameters,
                request.tau1,
                request.opening_source_len,
                request.opening_ring_dimension,
                request.relation_plan,
            )?
        }
    };
    let mut linear = if parameters.payload_mode.is_compressed()
        && parameters.ring_relation_mode == akita_types::RingRelationMode::QuotientLift
    {
        let weights = akita_types::build_compression_relation_weights(
            setup,
            instance,
            request.alpha,
            parameters,
            request.tau1,
            &layout,
            request.opening_ring_dimension,
            plan.domain_len(),
        )?;
        weights.into_sparse_entries()?
    } else {
        Vec::new()
    };
    let binary_intervals = if parameters.payload_mode.is_compressed() {
        akita_types::NegativeBinarySupport::new(&layout, plan.domain_len())?
            .intervals()
            .to_vec()
    } else {
        if !plan.binary_batching().is_zero() {
            return Err(AkitaError::InvalidProof);
        }
        Vec::new()
    };
    if let Some(norm) = plan.physical_l2() {
        if akita_types::PhysicalResponsePlan::new(parameters, request.relation_plan)?.as_ref()
            != Some(norm.plan)
        {
            return Err(AkitaError::InvalidInput(
                "physical L2 request differs from its schedule".into(),
            ));
        }
        let families = norm.plan.virtualization_families(norm.batching)?;
        let equality = OffsetEqWindow::new(norm.point)?;
        linear.extend(
            materialize_eq_tensor_left(&equality, &families, plan.witness_len())?
                .into_iter()
                .enumerate()
                .filter(|(_, weight)| !weight.is_zero()),
        );
        linear.sort_unstable_by_key(|(index, _)| *index);
    }
    Ok(CompiledStage2Weights {
        ordinary,
        linear,
        binary_intervals,
    })
}
