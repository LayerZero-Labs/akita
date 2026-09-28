use super::*;
use akita_algebra::offset_eq::{materialize_eq_tensor_left, OffsetEqWindow};
use akita_prover::backend::ValidatedRelationSessionPlan;

pub(crate) struct CompiledStage2Weights<E: Field> {
    pub ordinary: RelationWeightDescription<E>,
    pub linear: Vec<(usize, E)>,
    pub binary_intervals: Vec<Range<usize>>,
}

impl<E: Field> CompiledStage2Weights<E> {
    /// Incorporate one dense physical-L2 table into the existing Stage 2 state.
    ///
    /// Reduced evaluations can absorb the addend in place. Quotient-factored
    /// weights retain the sparse additional-term representation used by their
    /// sumcheck path.
    fn incorporate_physical_l2(&mut self, weights: Vec<E>) -> Result<(), AkitaError> {
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
        let equality = OffsetEqWindow::new(norm.point)?;
        let physical_l2 = materialize_eq_tensor_left(&equality, &families, plan.witness_len())?;
        compiled.incorporate_physical_l2(physical_l2)?;
    }
    Ok(compiled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{FpExt4, Prime32Offset99, Ring, Zero};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    fn extension(seed: u64) -> E {
        E::from_base_fn(|coordinate| F::from_u64(seed + 13 * coordinate as u64))
    }

    #[test]
    fn reduced_physical_l2_merge_matches_the_legacy_sparse_coefficient_table() {
        let relation = (0..8).map(|index| extension(7 + index)).collect::<Vec<_>>();
        let physical_l2 = (0..6)
            .map(|index| {
                if index % 3 == 0 {
                    E::zero()
                } else {
                    extension(31 + index)
                }
            })
            .collect::<Vec<_>>();
        let mut legacy = relation.clone();
        for (index, weight) in physical_l2.iter().copied().enumerate() {
            if !weight.is_zero() {
                *legacy.get_mut(index).expect("bounded physical-L2 index") += weight;
            }
        }

        let mut compiled = CompiledStage2Weights {
            ordinary: RelationWeightDescription::ReducedEvaluations {
                evaluations: relation,
                live_len: 6,
            },
            linear: Vec::new(),
            binary_intervals: Vec::new(),
        };
        compiled
            .incorporate_physical_l2(physical_l2)
            .expect("physical-L2 table fits reduced relation table");

        let RelationWeightDescription::ReducedEvaluations { evaluations, .. } = compiled.ordinary
        else {
            panic!("reduced weights remain reduced");
        };
        assert_eq!(evaluations, legacy);
        assert!(compiled.linear.is_empty());
    }

    #[test]
    fn quotient_physical_l2_keeps_the_legacy_sparse_representation() {
        let factorization = RelationWeightFactorization::new(
            vec![extension(1), extension(2)],
            vec![extension(3), extension(4)],
        )
        .expect("valid factorization");
        let prior = vec![(5, extension(9)), (1, extension(10))];
        let physical_l2 = vec![E::zero(), extension(11), E::zero(), extension(12)];
        let mut expected = prior.clone();
        expected.extend(
            physical_l2
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, weight)| !weight.is_zero()),
        );
        expected.sort_unstable_by_key(|(index, _)| *index);

        let mut compiled = CompiledStage2Weights {
            ordinary: RelationWeightDescription::QuotientFactored(factorization),
            linear: prior,
            binary_intervals: Vec::new(),
        };
        compiled
            .incorporate_physical_l2(physical_l2)
            .expect("sparse route remains valid");
        assert_eq!(compiled.linear, expected);
    }

    #[test]
    fn reduced_physical_l2_rejects_a_table_larger_than_its_relation_weights() {
        let mut compiled = CompiledStage2Weights {
            ordinary: RelationWeightDescription::ReducedEvaluations {
                evaluations: vec![extension(1); 2],
                live_len: 2,
            },
            linear: Vec::new(),
            binary_intervals: Vec::new(),
        };
        assert!(matches!(
            compiled.incorporate_physical_l2(vec![extension(2); 3]),
            Err(AkitaError::InvalidProof)
        ));
    }
}
