use super::*;

impl<E: Field + Ring + Unreduced> RelationRangeImageProver<E> {
    fn finish_ingested_round(&mut self, fold_started: Instant) {
        self.rounds_completed += 1;
        if self.rounds_completed < self.num_vars {
            if self.cached_round_poly.is_none() {
                self.cached_round_poly = Some(self.compute_current_round_poly_from_state());
            }
        } else {
            self.cached_round_poly = None;
        }
        self.fold_time_total += fold_started.elapsed().as_secs_f64();
        if self.rounds_completed == self.num_vars {
            tracing::debug!(
                rounds = self.num_vars,
                scan_s = self.scan_time_total,
                fold_s = self.fold_time_total,
                "stage2 sumcheck rounds complete"
            );
        }
    }

    pub(super) fn compute_current_round_poly_from_state(&mut self) -> UniPoly<E> {
        let t_scan = Instant::now();
        let (poly, norm_poly) = match &self.relation_state {
            RelationRoundState::ReducedDense { weights: dense } => match &self.witness_state {
                WitnessState::CompactPrefix(compact_witness) => {
                    let (virt_terms, rel_terms) = self.compute_round_compact_reduced_dense_terms(
                        compact_witness.view(),
                        dense.evaluations(),
                    );
                    let (norm_poly, relation_poly) = self.polys_from_terms(virt_terms, rel_terms);
                    (self.combine_polys(&norm_poly, &relation_poly), norm_poly)
                }
                WitnessState::FoldedSuffix(folded_witness) => {
                    let (virt_terms, rel_terms) = self.compute_folded_reduced_dense_round_terms(
                        folded_witness,
                        dense.evaluations(),
                    );
                    let (norm_poly, relation_poly) = self.polys_from_terms(virt_terms, rel_terms);
                    (self.combine_polys(&norm_poly, &relation_poly), norm_poly)
                }
            },
            RelationRoundState::QuotientFactored { weights, .. } => {
                self.compute_quotient_round_from_state(weights)
            }
        };
        self.prev_norm_poly = Some(norm_poly);
        self.scan_time_total += t_scan.elapsed().as_secs_f64();
        poly
    }

    fn compute_quotient_round_from_state(
        &self,
        weights: &RelationWeightFactorization<E>,
    ) -> (UniPoly<E>, UniPoly<E>) {
        if self.using_deferred_compact_prefix() {
            if let Some(prefix) = self.deferred_compact_prefix() {
                let (virt_poly, rel_poly) = match prefix.phase {
                    DeferredCompactPrefixPhase::Round0 => {
                        prefix.skip_state.reconstruct_round0_polys()
                    }
                    DeferredCompactPrefixPhase::Round1 { first_challenge } => {
                        prefix.skip_state.reconstruct_round1_polys(first_challenge)
                    }
                };
                let combined = self.combine_polys(&virt_poly, &rel_poly);
                return (combined, virt_poly);
            }
        }

        let use_partial_lane_coefficient_round = self.use_partial_lane_coefficient_round();
        let use_partial_lane_round = self.use_partial_lane_round();
        match &self.witness_state {
            WitnessState::CompactPrefix(compact_witness) => {
                if use_partial_lane_coefficient_round {
                    let (virt_q_coeffs, rel_coeffs) = self
                        .compute_compact_partial_lane_coefficient_round_terms(
                            compact_witness.view(),
                            weights,
                        );
                    let (norm_poly, relation_poly) =
                        self.polys_from_terms(virt_q_coeffs, rel_coeffs);
                    (self.combine_polys(&norm_poly, &relation_poly), norm_poly)
                } else if use_partial_lane_round {
                    let (virt_terms, rel_coeffs) = self
                        .compute_compact_partial_lane_round_terms(compact_witness.view(), weights);
                    let (norm_poly, relation_poly) = self.polys_from_terms(virt_terms, rel_coeffs);
                    (self.combine_polys(&norm_poly, &relation_poly), norm_poly)
                } else {
                    let (virt_q_coeffs, rel_coeffs) =
                        self.compute_round_compact_dense_terms(compact_witness.view(), weights);
                    let (norm_poly, relation_poly) =
                        self.polys_from_terms(virt_q_coeffs, rel_coeffs);
                    (self.combine_polys(&norm_poly, &relation_poly), norm_poly)
                }
            }
            WitnessState::FoldedSuffix(folded_witness) => {
                if use_partial_lane_coefficient_round {
                    let (virt_q_coeffs, rel_coeffs) = self
                        .compute_folded_partial_lane_coefficient_round_terms(
                            folded_witness,
                            weights,
                        );
                    let (norm_poly, relation_poly) =
                        self.polys_from_terms(virt_q_coeffs, rel_coeffs);
                    (self.combine_polys(&norm_poly, &relation_poly), norm_poly)
                } else if use_partial_lane_round {
                    let (virt_q_coeffs, rel_coeffs) =
                        self.compute_folded_partial_lane_round_terms(folded_witness, weights);
                    let (norm_poly, relation_poly) =
                        self.polys_from_terms(virt_q_coeffs, rel_coeffs);
                    (self.combine_polys(&norm_poly, &relation_poly), norm_poly)
                } else {
                    let (virt_q_coeffs, rel_coeffs) =
                        self.compute_folded_dense_round_terms(folded_witness, weights);
                    let (norm_poly, relation_poly) =
                        self.polys_from_terms(virt_q_coeffs, rel_coeffs);
                    (self.combine_polys(&norm_poly, &relation_poly), norm_poly)
                }
            }
        }
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
    fn ingest_reduced_dense_challenge(&mut self, r: E) {
        self.split_eq.bind(r);
        let folding_lane_round = !self.in_coefficient_round();
        self.witness_state = match mem::replace(
            &mut self.witness_state,
            WitnessState::FoldedSuffix(Vec::new()),
        ) {
            WitnessState::CompactPrefix(compact_witness) => {
                let compact_view = compact_witness.view();
                let fold_lut = Self::build_compact_w_fold_lut(compact_view, r);
                self.fold_linear_terms_for_current_round(r);
                WitnessState::FoldedSuffix(Self::materialize_compact_witness(
                    compact_view,
                    &fold_lut,
                ))
            }
            WitnessState::FoldedSuffix(mut folded_witness) => {
                fold_evals_in_place(&mut folded_witness, r);
                self.fold_linear_terms_for_current_round(r);
                WitnessState::FoldedSuffix(folded_witness)
            }
        };
        if let RelationRoundState::ReducedDense { weights } = &mut self.relation_state {
            weights.bind(r);
        }
        if folding_lane_round {
            self.live_lane_count = self.live_lane_count.div_ceil(2);
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

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let mut polynomial = if let Some(poly) = self.cached_round_poly.take() {
            poly
        } else {
            self.compute_current_round_poly_from_state()
        };
        if let Some(additional) = self.additional_round_polynomial() {
            if polynomial.coeffs.len() < additional.coeffs.len() {
                polynomial.coeffs.resize(additional.coeffs.len(), E::zero());
            }
            for (coefficient, addition) in polynomial.coeffs.iter_mut().zip(additional.coeffs) {
                *coefficient += addition;
            }
        }
        polynomial
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        let t_fold = Instant::now();
        let _span = tracing::info_span!("RelationRangeImageProver::fold_round").entered();
        if let Some(additional) = &mut self.additional_relation_terms {
            additional.bind(r);
        }
        if let Some(prev_norm_poly) = self.prev_norm_poly.take() {
            self.prev_norm_claim = prev_norm_poly.evaluate(&r);
        }

        if matches!(self.relation_state, RelationRoundState::ReducedDense { .. }) {
            self.ingest_reduced_dense_challenge(r);
            drop(_span);
            self.finish_ingested_round(t_fold);
            return;
        }

        if self.using_deferred_compact_prefix() {
            let rounds_completed = self.rounds_completed;
            self.split_eq.bind(r);
            if rounds_completed == 0 {
                if let RelationRoundState::QuotientFactored {
                    prefix: QuotientPrefixState::Deferred(prefix),
                    ..
                } = &mut self.relation_state
                {
                    prefix.phase = DeferredCompactPrefixPhase::Round1 { first_challenge: r };
                }
            } else if let Some((r0, coeff_count, alpha_round2)) = match &self.relation_state {
                RelationRoundState::QuotientFactored {
                    weights,
                    prefix: QuotientPrefixState::Deferred(prefix),
                } => match prefix.phase {
                    DeferredCompactPrefixPhase::Round0 => None,
                    DeferredCompactPrefixPhase::Round1 { first_challenge } => Some((
                        first_challenge,
                        weights.common_alpha_factor().len(),
                        Self::fold_alpha_two_rounds(
                            weights.common_alpha_factor(),
                            first_challenge,
                            r,
                        ),
                    )),
                },
                _ => None,
            } {
                self.linear_terms.fold_two_coefficients(r0, r);
                // This is the two-round coefficient handoff, so the ordinary one-round
                // linear-term transition below is deliberately bypassed.
                let mut round2_terms = None;
                self.witness_state = match mem::replace(
                    &mut self.witness_state,
                    WitnessState::FoldedSuffix(Vec::new()),
                ) {
                    WitnessState::CompactPrefix(compact_witness) => {
                        if self.coefficient_bits() > 2 {
                            let (folded_witness, virt_terms, rel_coeffs) =
                                if let RelationRoundState::QuotientFactored { weights, .. } =
                                    &self.relation_state
                                {
                                    self.materialize_two_round_compact_prefix_and_compute_next_round(
                                    compact_witness.view(),
                                    weights,
                                    &alpha_round2,
                                    &self.linear_terms,
                                    r0,
                                    r,
                                    )
                                } else {
                                    return;
                                };
                            round2_terms = Some((virt_terms, rel_coeffs));
                            WitnessState::FoldedSuffix(folded_witness)
                        } else {
                            WitnessState::FoldedSuffix(Self::materialize_two_round_compact_prefix(
                                compact_witness.view(),
                                self.live_lane_count,
                                coeff_count,
                                r0,
                                r,
                            ))
                        }
                    }
                    WitnessState::FoldedSuffix(folded_witness) => {
                        WitnessState::FoldedSuffix(folded_witness)
                    }
                };
                if let RelationRoundState::QuotientFactored { weights, .. } =
                    &mut self.relation_state
                {
                    *weights.components_mut().0 = alpha_round2;
                }
                self.finish_deferred_compact_prefix();
                if let Some((virt_terms, rel_coeffs)) = round2_terms {
                    self.cached_round_poly = Some(self.combine_terms(virt_terms, rel_coeffs));
                }
            }
            drop(_span);
            self.finish_ingested_round(t_fold);
            return;
        }

        let Some(coeff_count) = (match &self.relation_state {
            RelationRoundState::QuotientFactored { weights, .. } => {
                Some(weights.common_alpha_factor().len())
            }
            RelationRoundState::ReducedDense { .. } => None,
        }) else {
            return;
        };
        self.split_eq.bind(r);
        let folding_lane_round = !self.in_coefficient_round();
        let use_partial_lane_round = self.use_partial_lane_round();
        let use_partial_lane_coefficient_round = self.use_partial_lane_coefficient_round();
        let in_coefficient_round = self.in_coefficient_round();
        let fuse_next_coefficient_round = use_partial_lane_coefficient_round
            && self.rounds_completed + 1 < self.coefficient_bits();
        let fuse_next_folded_partial_lane =
            use_partial_lane_round && self.next_uses_partial_lane_round();
        let live_lane_count = self.live_lane_count;
        let mut fused_coefficient_round = false;
        let mut fused_folded_partial_lane = false;

        self.witness_state = match mem::replace(
            &mut self.witness_state,
            WitnessState::FoldedSuffix(Vec::new()),
        ) {
            WitnessState::CompactPrefix(compact_witness) => {
                let compact_view = compact_witness.view();
                let fold_lut = Self::build_compact_w_fold_lut(compact_view, r);
                let folded_witness = if folding_lane_round && use_partial_lane_round {
                    Self::fold_compact_partial_lanes(
                        compact_view,
                        live_lane_count,
                        coeff_count,
                        &fold_lut,
                    )
                } else {
                    Self::materialize_compact_witness(compact_view, &fold_lut)
                };
                self.fold_linear_terms_for_current_round(r);
                WitnessState::FoldedSuffix(folded_witness)
            }
            WitnessState::FoldedSuffix(folded_witness) => {
                if folding_lane_round && use_partial_lane_round {
                    if fuse_next_folded_partial_lane {
                        // Fold linear terms before the fused kernel so relation terms use the same
                        // post-fold table as `compute_folded_partial_lane_round_terms`.
                        self.fold_linear_terms_for_current_round(r);
                        let (
                            next_folded_witness,
                            next_relation_lane_weights,
                            virt_terms,
                            rel_coeffs,
                        ) = if let RelationRoundState::QuotientFactored { weights, .. } =
                            &self.relation_state
                        {
                            self.fuse_folded_partial_lane_and_compute_next_round(
                                &folded_witness,
                                weights,
                                r,
                            )
                        } else {
                            return;
                        };
                        if let RelationRoundState::QuotientFactored { weights, .. } =
                            &mut self.relation_state
                        {
                            *weights.components_mut().1 = next_relation_lane_weights;
                        }
                        self.cached_round_poly = Some(self.combine_terms(virt_terms, rel_coeffs));
                        fused_folded_partial_lane = true;
                        WitnessState::FoldedSuffix(next_folded_witness)
                    } else {
                        let next_folded_witness = Self::fold_folded_partial_lanes(
                            &folded_witness,
                            live_lane_count,
                            coeff_count,
                            r,
                        );
                        self.fold_linear_terms_for_current_round(r);
                        WitnessState::FoldedSuffix(next_folded_witness)
                    }
                } else if in_coefficient_round && use_partial_lane_coefficient_round {
                    self.fold_linear_terms_for_current_round(r);
                    if fuse_next_coefficient_round {
                        let mut next_alpha_factor = match &self.relation_state {
                            RelationRoundState::QuotientFactored { weights, .. } => {
                                weights.common_alpha_factor().to_vec()
                            }
                            RelationRoundState::ReducedDense { .. } => return,
                        };
                        fold_evals_in_place(&mut next_alpha_factor, r);
                        let (next_folded_witness, virt_terms, rel_coeffs) =
                            if let RelationRoundState::QuotientFactored { weights, .. } =
                                &self.relation_state
                            {
                                self.fuse_folded_coefficients_and_compute_next_round(
                                    &folded_witness,
                                    weights,
                                    &next_alpha_factor,
                                    r,
                                )
                            } else {
                                return;
                            };
                        if let RelationRoundState::QuotientFactored { weights, .. } =
                            &mut self.relation_state
                        {
                            *weights.components_mut().0 = next_alpha_factor;
                        }
                        self.cached_round_poly = Some(self.combine_terms(virt_terms, rel_coeffs));
                        fused_coefficient_round = true;
                        WitnessState::FoldedSuffix(next_folded_witness)
                    } else {
                        WitnessState::FoldedSuffix(Self::fold_folded_coefficients(
                            &folded_witness,
                            live_lane_count,
                            coeff_count,
                            r,
                        ))
                    }
                } else {
                    let mut folded_witness = folded_witness;
                    fold_evals_in_place(&mut folded_witness, r);
                    self.fold_linear_terms_for_current_round(r);
                    WitnessState::FoldedSuffix(folded_witness)
                }
            }
        };

        if folding_lane_round {
            if use_partial_lane_round {
                if !fused_folded_partial_lane {
                    if let RelationRoundState::QuotientFactored { weights, .. } =
                        &mut self.relation_state
                    {
                        let next_relation_lane_weights =
                            Self::fold_relation_lane_weights(weights.relation_lane_weights(), r);
                        *weights.components_mut().1 = next_relation_lane_weights;
                    }
                }
            } else {
                if let RelationRoundState::QuotientFactored { weights, .. } =
                    &mut self.relation_state
                {
                    fold_evals_in_place(weights.components_mut().1, r);
                }
            }
            self.live_lane_count = self.live_lane_count.div_ceil(2);
        } else if !fused_coefficient_round {
            if let RelationRoundState::QuotientFactored { weights, .. } = &mut self.relation_state {
                fold_evals_in_place(weights.components_mut().0, r);
            }
        }

        drop(_span);
        self.finish_ingested_round(t_fold);
    }
}
