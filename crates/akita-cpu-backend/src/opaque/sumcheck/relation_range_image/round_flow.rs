use super::*;
use jolt_poly::UnivariatePoly;

impl<E: Field + Ring + Unreduced + Fold> RelationRangeImageProver<E> {
    fn finish_ingested_round(&mut self) {
        if self.state.rounds_completed < self.state.num_vars {
            if self.state.cached_round_message.is_none() {
                self.state.cached_round_message = Some(
                    self.state
                        .compute_current_round_message_from_state(&mut self.phase),
                );
            }
        } else {
            self.state.cached_round_message = None;
        }
    }
}

impl<E: Field + Ring + Unreduced + Fold> RelationRoundState<E> {
    fn compute_current_round_message_from_state(
        &mut self,
        phase: &mut Phase<E>,
    ) -> RoundMessage<E> {
        enum RoundComputation<'a, E: Field> {
            Message(RoundMessage<E>, UnivariatePoly<E>),
            Terms(NormRoundTerms<'a, E>, RoundMessage<E>),
        }

        let computation = match phase {
            Phase::CompactPrefix {
                weights, engine, ..
            } => {
                let (message, norm) = self.compact_prefix_round_message(engine, weights);
                RoundComputation::Message(message, norm)
            }
            Phase::Coefficient {
                witness,
                relation: CoefficientRelation::ReducedDense(dense),
                ..
            } => {
                let (virt_terms, rel_terms) = match witness {
                    WitnessState::CompactPrefix(compact_witness) => self
                        .compute_round_compact_reduced_dense_terms(
                            compact_witness.view(),
                            dense.evaluations(),
                        ),
                    WitnessState::FoldedSuffix(folded_witness) => self
                        .compute_folded_reduced_dense_round_terms(
                            folded_witness,
                            dense.evaluations(),
                        ),
                };
                RoundComputation::Terms(virt_terms, rel_terms)
            }
            Phase::Coefficient {
                witness,
                relation: CoefficientRelation::Factored(weights),
                relation_moments,
            } => {
                if let Some(moments) = relation_moments {
                    let virt_terms = match witness {
                        WitnessState::CompactPrefix(compact_witness) => self
                            .compute_compact_partial_lane_coefficient_round_norm_terms(
                                compact_witness.view(),
                                weights,
                            ),
                        WitnessState::FoldedSuffix(folded_witness) => self
                            .compute_folded_partial_lane_coefficient_round_norm_terms(
                                folded_witness,
                                weights,
                            ),
                    };
                    let relation_coeffs =
                        moments.relation_message(weights.common_alpha_factor(), &self.linear_terms);
                    RoundComputation::Terms(virt_terms, relation_coeffs)
                } else {
                    let (message, norm) = self.compute_quotient_round_from_state(witness, weights);
                    RoundComputation::Message(message, norm)
                }
            }
            Phase::Lane { witness, lane } => {
                let (norm, relation) = lane.round_terms(
                    witness,
                    None,
                    self.split_eq.remaining_eq_tables(),
                    self.split_eq.prepare_linear_q_recovery(),
                );
                RoundComputation::Terms(norm, relation)
            }
        };
        let (message, norm_poly) = match computation {
            RoundComputation::Message(message, norm_poly) => (message, norm_poly),
            RoundComputation::Terms(virt_terms, relation_message) => {
                self.combine_terms(virt_terms, relation_message)
            }
        };
        self.prev_norm_poly = Some(norm_poly);
        message
    }
}

impl<E: Field + Ring + Unreduced> RelationRoundState<E> {
    fn compute_quotient_round_from_state(
        &self,
        witness: &WitnessState<E>,
        weights: &RelationWeightFactorization<E>,
    ) -> (RoundMessage<E>, UnivariatePoly<E>) {
        let (virt_terms, rel_coeffs) = match witness {
            WitnessState::CompactPrefix(compact_witness) => {
                if self.use_partial_lane_coefficient_round() {
                    self.compute_compact_partial_lane_coefficient_round_terms(
                        compact_witness.view(),
                        weights,
                    )
                } else {
                    self.compute_round_compact_dense_terms(compact_witness.view(), weights)
                }
            }
            WitnessState::FoldedSuffix(folded_witness) => {
                if self.use_partial_lane_coefficient_round() {
                    self.compute_folded_partial_lane_coefficient_round_terms(
                        folded_witness,
                        weights,
                    )
                } else {
                    self.compute_folded_dense_round_terms(folded_witness, weights)
                }
            }
        };
        self.combine_terms(virt_terms, rel_coeffs)
    }

    #[inline]
    pub(super) fn build_compact_w_fold_lut(
        compact_witness: PackedSignedDigitView<'_>,
        r: E,
    ) -> CompactPairFoldLut<E> {
        let bounds = compact_witness.bounds();
        CompactPairFoldLut::from_contiguous_range(
            -i16::from(bounds.negative_abs_max()),
            i16::from(bounds.positive_max()),
            r,
        )
    }

    pub(super) fn materialize_compact_witness(
        compact_witness: PackedSignedDigitView<'_>,
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        cfg_into_iter!(0..compact_witness.len().div_ceil(2))
            .map(|j| {
                fold_lut.fold(
                    compact_witness.get(2 * j).map_or(0, i16::from),
                    compact_witness.get(2 * j + 1).map_or(0, i16::from),
                )
            })
            .collect()
    }
}

impl<E: Field + Ring + Unreduced + Fold> RelationRoundState<E> {
    fn ingest_reduced_dense_challenge(
        &mut self,
        witness: &mut WitnessState<E>,
        weights: &mut DenseRelationWeights<E>,
        r: E,
    ) {
        self.split_eq.bind(r);
        self.linear_terms.fold_coefficients(r);
        match witness {
            WitnessState::CompactPrefix(compact_witness) => {
                let compact_view = compact_witness.view();
                let fold_lut = Self::build_compact_w_fold_lut(compact_view, r);
                let folded = Self::materialize_compact_witness(compact_view, &fold_lut);
                *witness = WitnessState::FoldedSuffix(folded);
            }
            WitnessState::FoldedSuffix(folded_witness) => {
                fold_evals_in_place(folded_witness, r);
            }
        }
        weights.bind(r);
    }

    /// Bind `r` in a coefficient round of the factored relation outside the
    /// compact prefix. With partial lanes, a round followed by another
    /// coefficient round also computes and caches the next round's message.
    fn ingest_factored_coefficient_challenge(
        &mut self,
        witness: &mut WitnessState<E>,
        weights: &mut RelationWeightFactorization<E>,
        relation_moments: &mut Option<CoefficientRelationMoments<E>>,
        r: E,
    ) {
        self.split_eq.bind(r);
        self.linear_terms.fold_coefficients(r);
        if let Some(moments) = relation_moments.as_mut() {
            moments.bind(r);
        }
        let coeff_count = weights.common_alpha_factor().len();
        let partial_lanes = self.use_partial_lane_coefficient_round();
        let fuse_next_round = partial_lanes && self.rounds_completed + 1 < self.coefficient_bits();
        let mut next_alpha_factor = weights.common_alpha_factor().to_vec();
        fold_evals_in_place(&mut next_alpha_factor, r);
        let folded_witness = match &mut *witness {
            WitnessState::CompactPrefix(compact_witness) => {
                let compact_view = compact_witness.view();
                let fold_lut = Self::build_compact_w_fold_lut(compact_view, r);
                Self::materialize_compact_witness(compact_view, &fold_lut)
            }
            WitnessState::FoldedSuffix(folded_witness) if fuse_next_round => {
                if let Some(moments) = relation_moments.as_ref() {
                    let (next_folded_witness, virt_terms) = self
                        .fuse_folded_coefficients_and_compute_next_round_norm_terms(
                            folded_witness,
                            weights,
                            &next_alpha_factor,
                            r,
                        );
                    let rel_coeffs =
                        moments.relation_message(&next_alpha_factor, &self.linear_terms);
                    let (message, norm_poly) = self.combine_terms(virt_terms, rel_coeffs);
                    self.prev_norm_poly = Some(norm_poly);
                    self.cached_round_message = Some(message);
                    next_folded_witness
                } else {
                    let (next_folded_witness, virt_terms, rel_coeffs) = self
                        .fuse_folded_coefficients_and_compute_next_round(
                            folded_witness,
                            weights,
                            &next_alpha_factor,
                            r,
                        );
                    let (message, norm_poly) = self.combine_terms(virt_terms, rel_coeffs);
                    self.prev_norm_poly = Some(norm_poly);
                    self.cached_round_message = Some(message);
                    next_folded_witness
                }
            }
            WitnessState::FoldedSuffix(folded_witness) if partial_lanes => {
                Self::fold_folded_coefficients(folded_witness, self.live_lane_count, coeff_count, r)
            }
            WitnessState::FoldedSuffix(folded_witness) => {
                fold_evals_in_place(folded_witness, r);
                *weights.common_alpha_factor_mut() = next_alpha_factor;
                return;
            }
        };
        *weights.common_alpha_factor_mut() = next_alpha_factor;
        *witness = WitnessState::FoldedSuffix(folded_witness);
    }
}

impl<E: Field + Ring + Unreduced + Fold> SumcheckInstanceProver<E> for RelationRangeImageProver<E> {
    fn num_rounds(&self) -> usize {
        self.state.num_vars
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> E {
        self.state.input_claim
    }

    fn compute_round_univariate(&mut self, _round: usize, previous_claim: E) -> UnivariatePoly<E> {
        let mut message = if let Some(message) = self.state.cached_round_message.take() {
            message
        } else {
            self.state
                .compute_current_round_message_from_state(&mut self.phase)
        };
        if let Some(additional) = self.additional_round_message() {
            message.add_assign(additional);
        }
        let polynomial = message.into_polynomial(previous_claim);
        #[cfg(debug_assertions)]
        if let Some(at_zero) = self.debug_dense_round_at_zero() {
            debug_assert_eq!(
                polynomial.evaluate(E::zero()),
                at_zero,
                "dense round constant disagrees with its folded oracle"
            );
        }
        polynomial
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        let _span = tracing::info_span!("RelationRangeImageProver::fold_round").entered();
        if let Some(additional) = &mut self.state.additional_relation_terms {
            additional.bind(r);
        }
        if let Some(prev_norm_poly) = self.state.prev_norm_poly.take() {
            self.state.prev_norm_claim = prev_norm_poly.evaluate(r);
        }

        match &mut self.phase {
            Phase::CompactPrefix {
                weights, engine, ..
            } => self
                .state
                .ingest_compact_prefix_challenge(weights, engine, r),
            Phase::Coefficient {
                witness,
                relation: CoefficientRelation::Factored(weights),
                relation_moments,
            } => self.state.ingest_factored_coefficient_challenge(
                witness,
                weights,
                relation_moments,
                r,
            ),
            Phase::Coefficient {
                witness,
                relation: CoefficientRelation::ReducedDense(weights),
                ..
            } => self
                .state
                .ingest_reduced_dense_challenge(witness, weights, r),
            Phase::Lane { witness, lane } => {
                self.state.ingest_lane_product_challenge(witness, lane, r)
            }
        };
        self.state.rounds_completed += 1;
        self.state.advance_phase(&mut self.phase);
        drop(_span);
        self.finish_ingested_round();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Ext2, ExtField, Prime64Offset59};

    #[test]
    fn round_hooks_after_completion_panic() {
        use jolt_field::{One, Prime128Offset275 as E};
        use std::panic::{catch_unwind, AssertUnwindSafe};
        for compute in [true, false] {
            let mut prover = RelationRangeImageProver::<E>::new_virtual_only(
                vec![0; 8],
                &[E::one(); 3],
                E::zero(),
                8,
                2,
                1,
                2,
            )
            .unwrap();
            let mut claim = prover.input_claim();
            for round in 0..prover.num_rounds() {
                let polynomial = prover.compute_round_univariate(round, claim);
                let challenge = E::from_u64(3);
                claim = polynomial.evaluate(challenge);
                prover.ingest_challenge(round, challenge);
            }
            assert!(catch_unwind(AssertUnwindSafe(|| {
                if compute {
                    let _ = prover.compute_round_univariate(prover.num_rounds(), claim);
                } else {
                    prover.ingest_challenge(prover.num_rounds(), E::one());
                }
            }))
            .is_err());
        }
    }

    #[test]
    fn compact_fold_uses_stored_extrema_and_zero_padding() {
        type F = Prime64Offset59;
        type E = Ext2<F>;
        let r = E::from_base_fn(|i| F::from_u64(19 + 7 * i as u64));
        for digits in [
            vec![],
            vec![0],
            vec![3, 1, 2],
            vec![-4, -1, -2],
            vec![-128, 127, 0],
        ] {
            let witness = PackedSignedDigits::from_i8_digits_auto(digits.clone());
            let lut = RelationRoundState::<E>::build_compact_w_fold_lut(witness.view(), r);
            let mut values = digits;
            values.push(0);
            for &left in &values {
                for &right in &values {
                    let l = E::from_i64(i64::from(left));
                    let expected = l + r * (E::from_i64(i64::from(right)) - l);
                    assert_eq!(lut.fold(i16::from(left), i16::from(right)), expected);
                }
            }
        }
    }
}

#[cfg(test)]
mod round_message_tests {
    use super::*;
    use jolt_field::{One, Prime128Offset275};

    type E = Prime128Offset275;

    #[test]
    fn round_message_reconstructs_every_coefficient_from_the_sumcheck_claim() {
        let cases = [
            vec![E::from_u64(9)],
            vec![E::from_u64(7), E::from_u64(11)],
            vec![E::from_u64(5), E::from_u64(13), E::from_u64(17)],
            vec![
                E::from_u64(3),
                E::from_u64(19),
                E::from_u64(23),
                E::from_u64(29),
            ],
        ];
        for coefficients in cases {
            let expected = UnivariatePoly::new(coefficients);
            let claim = expected.evaluate(E::zero()) + expected.evaluate(E::one());
            let actual = RoundMessage::from_polynomial(&expected).into_polynomial(claim);
            assert_eq!(actual, expected);
        }
    }
}
