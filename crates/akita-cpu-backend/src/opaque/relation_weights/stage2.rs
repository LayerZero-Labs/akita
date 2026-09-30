use super::lane_weights::ResponseNormFactors;
use super::*;
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{
    materialize_eq_tensor_left, EqPairTensorFamily, EqPairTensorWeights, OffsetEqWindow,
};
use akita_prover::backend::ValidatedRelationSessionPlan;

pub(crate) struct CompiledStage2Weights<E: Field> {
    pub response_norm: Option<ResponseNormFactors<E>>,
    pub ordinary: RelationWeightDescription<E>,
    pub linear: Vec<(usize, E)>,
    pub binary_intervals: Vec<Range<usize>>,
}

impl<E: Field> CompiledStage2Weights<E> {
    /// Incorporate one dense response-norm table into the existing Stage 2 state.
    ///
    /// Reduced evaluations can absorb the addend in place. Quotient-factored
    /// weights retain the sparse additional-term representation used by their
    /// sumcheck path.
    fn absorb_response_norm_weights(&mut self, weights: Vec<E>) -> Result<(), AkitaError> {
        match &mut self.ordinary {
            RelationWeightDescription::ReducedEvaluations { evaluations, .. } => {
                let destination = evaluations
                    .get_mut(..weights.len())
                    .ok_or(AkitaError::InvalidProof)?;
                for (destination, weight) in destination.iter_mut().zip(weights) {
                    *destination += weight;
                }
            }
            RelationWeightDescription::QuotientFactored(_) => {
                self.linear.extend(
                    weights
                        .into_iter()
                        .enumerate()
                        .filter(|(_, weight)| !weight.is_zero()),
                );
                self.linear.sort_unstable_by_key(|(index, _)| *index);
            }
        }
        Ok(())
    }
}

/// Factor the response-norm equality table when its Stage-2 coefficient blocks
/// align with the physical ring rows. The result represents each table entry
/// as `lane_weights[lane] * coefficient_weights[coefficient]`.
fn factor_response_norm_weights<E: Field + Ring>(
    point: &[E],
    families: &[EqPairTensorFamily<E>],
    output_len: usize,
    coefficient_count: usize,
) -> Result<Option<ResponseNormFactors<E>>, AkitaError> {
    if coefficient_count == 0
        || !coefficient_count.is_power_of_two()
        || output_len == 0
        || !output_len.is_multiple_of(coefficient_count)
    {
        return Ok(None);
    }
    let coefficient_bits = coefficient_count.trailing_zeros() as usize;
    if coefficient_bits > point.len() || families.is_empty() {
        return Ok(None);
    }
    let coefficient_weights = EqPolynomial::evals(&point[..coefficient_bits])?;
    let high_equality = OffsetEqWindow::new(&point[coefficient_bits..])?;
    let live_lane_count = output_len / coefficient_count;
    let mut lane_weights = vec![E::zero(); live_lane_count];

    for family in families {
        let [ring_axis, limb_axis, row_axis] = family.axes.as_slice() else {
            return Ok(None);
        };
        let EqPairTensorWeights::Dense(limb_weights) = &limb_axis.weights else {
            return Ok(None);
        };
        let expected_row_stride = limb_axis
            .len
            .checked_mul(ring_axis.len)
            .ok_or_else(|| AkitaError::InvalidSetup("response-norm row stride overflow".into()))?;
        if family.scalar != E::one()
            || ring_axis.len == 0
            || ring_axis.left_stride != 1
            || ring_axis.right_stride != 1
            || !matches!(ring_axis.weights, EqPairTensorWeights::Unit)
            || limb_axis.len != limb_weights.len()
            || limb_axis.left_stride != ring_axis.len
            || limb_axis.right_stride != 0
            || row_axis.len == 0
            || row_axis.left_stride != expected_row_stride
            || row_axis.right_stride != ring_axis.len
            || !matches!(row_axis.weights, EqPairTensorWeights::Unit)
            || !ring_axis.len.is_multiple_of(coefficient_count)
            || !family.left_offset.is_multiple_of(coefficient_count)
            || !family.right_offset.is_multiple_of(coefficient_count)
            || !row_axis.left_stride.is_multiple_of(coefficient_count)
        {
            return Ok(None);
        }
        let ring_chunks = ring_axis.len / coefficient_count;
        let row_lane_stride = row_axis.left_stride / coefficient_count;
        let limb_lane_stride = ring_axis.len / coefficient_count;
        let last_row_start = row_axis
            .len
            .checked_sub(1)
            .and_then(|row| row.checked_mul(row_axis.left_stride))
            .and_then(|offset| family.left_offset.checked_add(offset))
            .ok_or_else(|| AkitaError::InvalidSetup("response-norm row offset overflow".into()))?;
        let last_limb_start = limb_axis
            .len
            .checked_sub(1)
            .and_then(|limb| limb.checked_mul(limb_axis.left_stride))
            .and_then(|offset| last_row_start.checked_add(offset))
            .ok_or_else(|| AkitaError::InvalidSetup("response-norm limb offset overflow".into()))?;
        let left_end = last_limb_start
            .checked_add(ring_axis.len)
            .ok_or_else(|| AkitaError::InvalidSetup("response-norm span overflow".into()))?;
        if left_end > output_len {
            return Err(AkitaError::InvalidProof);
        }

        for row in 0..row_axis.len {
            let target_row_lane = family
                .left_offset
                .checked_div(coefficient_count)
                .and_then(|base| {
                    row.checked_mul(row_lane_stride)
                        .and_then(|offset| base.checked_add(offset))
                })
                .ok_or_else(|| AkitaError::InvalidSetup("response-norm lane overflow".into()))?;
            let physical_row = row
                .checked_mul(row_axis.right_stride)
                .and_then(|offset| family.right_offset.checked_add(offset))
                .ok_or_else(|| AkitaError::InvalidSetup("response-norm address overflow".into()))?;
            for limb in 0..limb_axis.len {
                let limb_start = limb
                    .checked_mul(limb_lane_stride)
                    .and_then(|offset| target_row_lane.checked_add(offset))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("response-norm lane overflow".into())
                    })?;
                let limb_weight = limb_weights
                    .get(limb)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?;
                for ring_chunk in 0..ring_chunks {
                    let lane = limb_start.checked_add(ring_chunk).ok_or_else(|| {
                        AkitaError::InvalidSetup("response-norm lane overflow".into())
                    })?;
                    if lane >= live_lane_count {
                        return Err(AkitaError::InvalidProof);
                    }
                    let physical_coefficient = ring_chunk
                        .checked_mul(coefficient_count)
                        .and_then(|offset| physical_row.checked_add(offset))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("response-norm address overflow".into())
                        })?;
                    debug_assert!(physical_coefficient.is_multiple_of(coefficient_count));
                    lane_weights[lane] +=
                        limb_weight * high_equality.eval(physical_coefficient / coefficient_count);
                }
            }
        }
    }
    Ok(Some(ResponseNormFactors {
        coefficient_weights,
        lane_weights,
    }))
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
            let weights = build_relation_lane_weights(RelationLaneWeightInputs {
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
                packing_semantics: match plan.linear_terms() {
                    akita_prover::backend::Stage2OpeningDescription::CoefficientPacking(batch) => {
                        Some(batch)
                    }
                    akita_prover::backend::Stage2OpeningDescription::EvaluationTrace { .. } => None,
                },
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
    let linear = if parameters.payload_mode.is_compressed()
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
    let mut compiled = CompiledStage2Weights {
        response_norm: None,
        ordinary,
        linear,
        binary_intervals,
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
        let factorized = match &mut compiled.ordinary {
            RelationWeightDescription::QuotientFactored(weights) => factor_response_norm_weights(
                norm.point,
                &families,
                plan.witness_len(),
                weights.common_alpha_factor().len(),
            )?,
            RelationWeightDescription::ReducedEvaluations { .. } => None,
        };
        if let Some(factors) = factorized {
            compiled.response_norm = Some(factors);
        } else {
            let equality = OffsetEqWindow::new(norm.point)?;
            let response_norm =
                materialize_eq_tensor_left(&equality, &families, plan.witness_len())?;
            compiled.absorb_response_norm_weights(response_norm)?;
        }
    }
    Ok(compiled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::offset_eq::EqPairTensorAxis;
    use jolt_field::{FpExt4, One, Prime32Offset99, Ring, Zero};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    fn extension(seed: u64) -> E {
        E::from_base_fn(|coordinate| F::from_u64(seed + 13 * coordinate as u64))
    }

    fn response_norm_family(
        left_offset: usize,
        right_offset: usize,
        row_count: usize,
        row_stride_left: usize,
        row_stride_right: usize,
        limb_weights: Vec<E>,
        ring_dimension: usize,
    ) -> EqPairTensorFamily<E> {
        EqPairTensorFamily::new(
            left_offset,
            right_offset,
            E::one(),
            vec![
                EqPairTensorAxis::unit(ring_dimension, 1, 1),
                EqPairTensorAxis::dense(ring_dimension, 0, limb_weights),
                EqPairTensorAxis::unit(row_count, row_stride_left, row_stride_right),
            ],
        )
        .expect("valid response-norm family")
    }

    fn assert_response_norm_factor_matches_dense(
        point: &[E],
        families: &[EqPairTensorFamily<E>],
        output_len: usize,
        coefficient_count: usize,
    ) {
        let equality = OffsetEqWindow::new(point).expect("valid equality point");
        let dense = materialize_eq_tensor_left(&equality, families, output_len)
            .expect("dense response-norm table");
        let ResponseNormFactors {
            coefficient_weights: coefficients,
            lane_weights: lanes,
        } = factor_response_norm_weights(point, families, output_len, coefficient_count)
            .expect("factorization check")
            .expect("aligned geometry factors");
        let factored = (0..output_len)
            .map(|index| lanes[index / coefficient_count] * coefficients[index % coefficient_count])
            .collect::<Vec<_>>();
        assert_eq!(factored, dense);
    }

    #[test]
    fn response_norm_direct_and_gram_factors_match_dense_for_multiple_units() {
        let ring_dimension = 8;
        let coefficient_count = 4;
        let row_stride_left = ring_dimension * 3;
        let row_stride_right = ring_dimension;
        let fold_basis = E::from_u64(5);
        let direct_first = E::from_u64(17);
        let direct_limb_weights = vec![
            direct_first,
            direct_first * fold_basis,
            direct_first * fold_basis * fold_basis,
        ];
        let gram_limb_weights = vec![E::from_u64(19), E::from_u64(23), E::from_u64(29)];
        let point = (0..7)
            .map(|index| E::from_u64(31 + 7 * index as u64))
            .collect::<Vec<_>>();

        for limb_weights in [direct_limb_weights, gram_limb_weights] {
            let families = vec![
                response_norm_family(
                    0,
                    0,
                    2,
                    row_stride_left,
                    row_stride_right,
                    limb_weights.clone(),
                    ring_dimension,
                ),
                response_norm_family(
                    2 * row_stride_left,
                    2 * row_stride_right,
                    2,
                    row_stride_left,
                    row_stride_right,
                    limb_weights,
                    ring_dimension,
                ),
            ];
            assert_response_norm_factor_matches_dense(&point, &families, 96, coefficient_count);
        }
    }

    #[test]
    fn response_norm_factorization_falls_back_for_misaligned_ring_blocks() {
        let point = (0..7)
            .map(|index| E::from_u64(37 + 3 * index as u64))
            .collect::<Vec<_>>();
        let family =
            response_norm_family(0, 0, 2, 12, 6, vec![E::from_u64(41), E::from_u64(43)], 6);
        assert!(
            factor_response_norm_weights(&point, std::slice::from_ref(&family), 24, 4)
                .expect("well-formed fallback geometry")
                .is_none()
        );
        let equality = OffsetEqWindow::new(&point).unwrap();
        assert!(materialize_eq_tensor_left(&equality, &[family], 24).is_ok());
    }

    #[test]
    fn reduced_response_norm_merge_matches_the_legacy_sparse_coefficient_table() {
        let relation = (0..8).map(|index| extension(7 + index)).collect::<Vec<_>>();
        let response_norm = (0..6)
            .map(|index| {
                if index % 3 == 0 {
                    E::zero()
                } else {
                    extension(31 + index)
                }
            })
            .collect::<Vec<_>>();
        let mut legacy = relation.clone();
        for (index, weight) in response_norm.iter().copied().enumerate() {
            if !weight.is_zero() {
                *legacy.get_mut(index).expect("bounded response-norm index") += weight;
            }
        }

        let mut compiled = CompiledStage2Weights {
            response_norm: None,
            ordinary: RelationWeightDescription::ReducedEvaluations {
                evaluations: relation,
                live_len: 6,
            },
            linear: Vec::new(),
            binary_intervals: Vec::new(),
        };
        compiled
            .absorb_response_norm_weights(response_norm)
            .expect("response-norm table fits reduced relation table");

        let RelationWeightDescription::ReducedEvaluations { evaluations, .. } = compiled.ordinary
        else {
            panic!("reduced weights remain reduced");
        };
        assert_eq!(evaluations, legacy);
        assert!(compiled.linear.is_empty());
    }

    #[test]
    fn quotient_response_norm_keeps_the_legacy_sparse_representation() {
        let factorization = RelationWeightFactorization::new(
            vec![extension(1), extension(2)],
            vec![extension(3), extension(4)],
        )
        .expect("valid factorization");
        let prior = vec![(5, extension(9)), (1, extension(10))];
        let response_norm = vec![E::zero(), extension(11), E::zero(), extension(12)];
        let mut expected = prior.clone();
        expected.extend(
            response_norm
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, weight)| !weight.is_zero()),
        );
        expected.sort_unstable_by_key(|(index, _)| *index);

        let mut compiled = CompiledStage2Weights {
            response_norm: None,
            ordinary: RelationWeightDescription::QuotientFactored(factorization),
            linear: prior,
            binary_intervals: Vec::new(),
        };
        compiled
            .absorb_response_norm_weights(response_norm)
            .expect("sparse route remains valid");
        assert_eq!(compiled.linear, expected);
    }

    #[test]
    fn reduced_response_norm_rejects_a_table_larger_than_its_relation_weights() {
        let mut compiled = CompiledStage2Weights {
            response_norm: None,
            ordinary: RelationWeightDescription::ReducedEvaluations {
                evaluations: vec![extension(1); 2],
                live_len: 2,
            },
            linear: Vec::new(),
            binary_intervals: Vec::new(),
        };
        assert!(matches!(
            compiled.absorb_response_norm_weights(vec![extension(2); 3]),
            Err(AkitaError::InvalidProof)
        ));
    }
}
