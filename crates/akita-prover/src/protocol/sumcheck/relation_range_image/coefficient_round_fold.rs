use super::*;

impl<F, E> RelationRangeImageProver<F, E>
where
    F: FieldCore,
    E: ExtField<F> + FromPrimitiveInt + HasUnreducedOps + SumcheckTableOperations<F>,
{
    #[tracing::instrument(
        skip_all,
        name = "RelationRangeImageProver::fuse_folded_coefficients_and_compute_next_round"
    )]
    pub(super) fn fuse_folded_coefficients_and_compute_next_round(
        &self,
        folded_witness: &mut EvaluationTable<F, E>,
        next_alpha_factor: &[E],
        challenge: E,
    ) -> (NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.in_coefficient_round());
        debug_assert!(self.current_coefficient_width() >= 2);
        let old_coefficient_count = self.common_alpha_factor.len();
        let next_coefficient_count = old_coefficient_count / 2;
        debug_assert_eq!(next_alpha_factor.len(), next_coefficient_count);
        debug_assert_eq!(
            folded_witness.len(),
            self.live_lane_count * old_coefficient_count
        );

        let (first_equality, second_equality) = self.split_eq.remaining_eq_tables();
        let include_norm_linear = !self.can_skip_norm_linear_coeff();
        let (norm_coefficients, mut relation_coefficients) = {
            let _span = tracing::info_span!(
                "RelationRangeImageProver::fold_and_compute_stage2_coefficient_round"
            )
            .entered();
            E::fold_and_compute_stage2_coefficient_round(
                self.kernel_plan,
                folded_witness,
                self.live_lane_count,
                old_coefficient_count,
                next_alpha_factor,
                &self.relation_lane_weights,
                first_equality,
                second_equality,
                challenge,
                include_norm_linear,
            )
        };

        let trace_coefficients = {
            let _span = tracing::info_span!(
                "RelationRangeImageProver::compute_folded_coefficient_trace_round"
            )
            .entered();
            if E::DELAYED_PRODUCT_SUM_IS_EXACT {
                self.compute_folded_coefficient_trace_round::<DelayedProductSum<E>>(folded_witness)
            } else {
                self.compute_folded_coefficient_trace_round::<DirectProductSum<E>>(folded_witness)
            }
        };
        for (relation, trace) in relation_coefficients.iter_mut().zip(trace_coefficients) {
            *relation += trace;
        }

        let norm_terms = if include_norm_linear {
            NormRoundTerms::Full(norm_coefficients)
        } else {
            NormRoundTerms::SkipLinear([norm_coefficients[0], norm_coefficients[2]])
        };
        (norm_terms, relation_coefficients)
    }

    fn compute_folded_coefficient_trace_round<A>(
        &self,
        folded_witness: &EvaluationTable<F, E>,
    ) -> [E; 3]
    where
        A: ProductSumAccumulator<E>,
    {
        let coefficient_count = self.common_alpha_factor.len() / 2;
        let coefficient_pair_count = coefficient_count / 2;
        let mut relation = [E::zero(); 3];

        for lane in 0..self.live_lane_count {
            let stored_lane = if self.lanes_in_binding_order {
                reverse_power_of_two_index(lane, self.live_lane_count)
            } else {
                lane
            };
            self.evaluation_trace
                .for_each_source_in_lane(lane, |factor, source_values| {
                    let mut source_relation: [A; 3] = std::array::from_fn(|_| A::zero());
                    for coefficient_pair in 0..coefficient_pair_count {
                        let left = 2 * coefficient_pair;
                        let stored_pair =
                            reverse_power_of_two_index(coefficient_pair, coefficient_pair_count);
                        let row_0 = stored_pair * self.live_lane_count + stored_lane;
                        let row_1 = row_0 + coefficient_pair_count * self.live_lane_count;
                        let witness_0 = folded_witness.evaluation(row_0);
                        let witness_delta = folded_witness.evaluation(row_1) - witness_0;
                        let source_0 = source_values[left];
                        let source_delta = source_values[left + 1] - source_0;
                        accumulate_relation_products(
                            &mut source_relation,
                            witness_0,
                            witness_delta,
                            source_0,
                            source_delta,
                        );
                    }
                    for (coefficient, source) in relation.iter_mut().zip(source_relation) {
                        *coefficient += factor * source.finish();
                    }
                });
        }
        relation
    }
}
