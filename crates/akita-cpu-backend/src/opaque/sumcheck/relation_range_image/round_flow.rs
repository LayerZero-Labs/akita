use super::*;
use jolt_poly::UnivariatePoly;

impl<E: Field + Ring + Unreduced + Fold> RelationRangeImageProver<E> {
    fn finish_ingested_round(&mut self) {
        if self.rounds_completed < self.num_vars {
            if self.cached_round_poly.is_none() {
                self.cached_round_poly = Some(self.compute_current_round_poly_from_state());
            }
        } else {
            self.cached_round_poly = None;
        }
    }

    pub(super) fn compute_current_round_poly_from_state(&mut self) -> UnivariatePoly<E> {
        enum RoundComputation<E: Field> {
            Polynomials(UnivariatePoly<E>, UnivariatePoly<E>),
            Terms(NormRoundTerms<E>, [E; 3]),
        }

        let mut phase = self.phase.take().expect("prover phase is installed");
        let computation = match &mut phase {
            Phase::CompactPrefix {
                weights, engine, ..
            } => {
                let (poly, norm) = self.compact_prefix_round_polys(engine, weights);
                RoundComputation::Polynomials(poly, norm)
            }
            Phase::Coefficient {
                witness,
                relation: CoefficientRelation::ReducedDense(dense),
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
            } => {
                let (poly, norm) = self.compute_quotient_round_from_state(witness, weights);
                RoundComputation::Polynomials(poly, norm)
            }
            Phase::Lane { witness, lane } => {
                let (norm, relation) = lane
                    .round_terms(
                        witness,
                        None,
                        Some(self.split_eq.remaining_eq_tables()),
                        self.can_skip_norm_linear_coeff(),
                    )
                    .expect("lane round equality tables produce terms");
                RoundComputation::Terms(norm, relation)
            }
        };
        self.phase = Some(phase);
        let (poly, norm_poly) = match computation {
            RoundComputation::Polynomials(poly, norm) => (poly, norm),
            RoundComputation::Terms(virt_terms, relation) => {
                let (norm, relation) = self.polys_from_terms(virt_terms, relation);
                (self.combine_polys(&norm, &relation), norm)
            }
        };
        self.prev_norm_poly = Some(norm_poly);
        poly
    }
}

impl<E: Field + Ring + Unreduced> RelationRangeImageProver<E> {
    fn compute_quotient_round_from_state(
        &self,
        witness: &WitnessState<E>,
        weights: &RelationWeightFactorization<E>,
    ) -> (UnivariatePoly<E>, UnivariatePoly<E>) {
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
        let (norm_poly, relation_poly) = self.polys_from_terms(virt_terms, rel_coeffs);
        (self.combine_polys(&norm_poly, &relation_poly), norm_poly)
    }

    #[inline]
    pub(super) fn build_compact_w_fold_lut(
        compact_witness: PackedSignedDigitView<'_>,
        r: E,
    ) -> CompactPairFoldLut<E> {
        let min_w = compact_witness
            .iter()
            .map(i32::from)
            .min()
            .unwrap_or(0)
            .min(0);
        let max_w = compact_witness
            .iter()
            .map(i32::from)
            .max()
            .unwrap_or(0)
            .max(0);
        CompactPairFoldLut::from_contiguous_range(min_w as i16, max_w as i16, r)
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

impl<E: Field + Ring + Unreduced + Fold> RelationRangeImageProver<E> {
    fn ingest_reduced_dense_challenge(
        &mut self,
        witness: WitnessState<E>,
        mut weights: DenseRelationWeights<E>,
        r: E,
    ) -> Phase<E> {
        self.split_eq.bind(r);
        self.linear_terms.fold_coefficients(r);
        let witness = match witness {
            WitnessState::CompactPrefix(compact_witness) => {
                let compact_view = compact_witness.view();
                let fold_lut = Self::build_compact_w_fold_lut(compact_view, r);
                Self::materialize_compact_witness(compact_view, &fold_lut)
            }
            WitnessState::FoldedSuffix(mut folded_witness) => {
                fold_evals_in_place(&mut folded_witness, r);
                folded_witness
            }
        };
        weights.bind(r);
        Phase::Coefficient {
            witness: WitnessState::FoldedSuffix(witness),
            relation: CoefficientRelation::ReducedDense(weights),
        }
    }

    /// Bind `r` in a coefficient round of the factored relation outside the
    /// compact prefix. With partial lanes, a round followed by another
    /// coefficient round also computes and caches the next round's message.
    fn ingest_factored_coefficient_challenge(
        &mut self,
        witness: WitnessState<E>,
        mut weights: RelationWeightFactorization<E>,
        r: E,
    ) -> Phase<E> {
        self.split_eq.bind(r);
        self.linear_terms.fold_coefficients(r);
        let coeff_count = weights.common_alpha_factor().len();
        let partial_lanes = self.use_partial_lane_coefficient_round();
        let fuse_next_round = partial_lanes && self.rounds_completed + 1 < self.coefficient_bits();
        let mut next_alpha_factor = weights.common_alpha_factor().to_vec();
        fold_evals_in_place(&mut next_alpha_factor, r);
        let folded_witness = match witness {
            WitnessState::CompactPrefix(compact_witness) => {
                let compact_view = compact_witness.view();
                let fold_lut = Self::build_compact_w_fold_lut(compact_view, r);
                Self::materialize_compact_witness(compact_view, &fold_lut)
            }
            WitnessState::FoldedSuffix(folded_witness) if fuse_next_round => {
                let (next_folded_witness, virt_terms, rel_coeffs) = self
                    .fuse_folded_coefficients_and_compute_next_round(
                        &folded_witness,
                        &weights,
                        &next_alpha_factor,
                        r,
                    );
                self.cached_round_poly = Some(self.combine_terms(virt_terms, rel_coeffs));
                next_folded_witness
            }
            WitnessState::FoldedSuffix(folded_witness) if partial_lanes => {
                Self::fold_folded_coefficients(
                    &folded_witness,
                    self.live_lane_count,
                    coeff_count,
                    r,
                )
            }
            WitnessState::FoldedSuffix(mut folded_witness) => {
                fold_evals_in_place(&mut folded_witness, r);
                folded_witness
            }
        };
        *weights.common_alpha_factor_mut() = next_alpha_factor;
        Phase::Coefficient {
            witness: WitnessState::FoldedSuffix(folded_witness),
            relation: CoefficientRelation::Factored(weights),
        }
    }
}

impl<E: Field + Ring + Unreduced + Fold> SumcheckInstanceProver<E> for RelationRangeImageProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UnivariatePoly<E> {
        let mut polynomial = if let Some(poly) = self.cached_round_poly.take() {
            poly
        } else {
            self.compute_current_round_poly_from_state()
        };
        if let Some(additional) = self.additional_round_polynomial() {
            let mut coefficients = polynomial.into_coefficients();
            coefficients.resize(
                coefficients.len().max(additional.coefficients().len()),
                E::zero(),
            );
            for (coefficient, addition) in coefficients.iter_mut().zip(additional.coefficients()) {
                *coefficient += *addition;
            }
            polynomial = UnivariatePoly::new(coefficients);
        }
        polynomial
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        let _span = tracing::info_span!("RelationRangeImageProver::fold_round").entered();
        if let Some(additional) = &mut self.additional_relation_terms {
            additional.bind(r);
        }
        if let Some(prev_norm_poly) = self.prev_norm_poly.take() {
            self.prev_norm_claim = prev_norm_poly.evaluate(r);
        }

        let phase = self.phase.take().expect("prover phase is installed");
        let phase = match phase {
            Phase::CompactPrefix {
                witness,
                weights,
                engine,
            } => self.ingest_compact_prefix_challenge(witness, weights, engine, r),
            Phase::Coefficient {
                witness,
                relation: CoefficientRelation::Factored(weights),
            } => self.ingest_factored_coefficient_challenge(witness, weights, r),
            Phase::Coefficient {
                witness,
                relation: CoefficientRelation::ReducedDense(weights),
            } => self.ingest_reduced_dense_challenge(witness, weights, r),
            Phase::Lane { witness, lane } => self.ingest_lane_product_challenge(witness, lane, r),
        };
        self.rounds_completed += 1;
        let phase = self.advance_phase(phase);
        self.phase = Some(phase);
        drop(_span);
        self.finish_ingested_round();
    }
}
