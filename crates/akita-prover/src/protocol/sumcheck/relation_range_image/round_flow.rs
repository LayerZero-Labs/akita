use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> RelationRangeImageProver<E> {
    pub(super) fn compute_current_round_poly_from_state(&mut self) -> UniPoly<E> {
        let t_scan = Instant::now();
        let use_deferred_compact_prefix = self.using_deferred_compact_prefix();
        let use_partial_lane_coefficient_round =
            !use_deferred_compact_prefix && self.use_partial_lane_coefficient_round();
        let use_partial_lane_round = !use_deferred_compact_prefix && self.use_partial_lane_round();
        let rounds_completed = self.rounds_completed;
        let poly = if use_deferred_compact_prefix {
            let (virt_poly, rel_poly) = {
                let prefix = self.ensure_deferred_compact_prefix();
                if rounds_completed == 0 {
                    let (virt_poly, rel_poly) = prefix.skip_state.reconstruct_round0_polys();
                    (virt_poly, rel_poly)
                } else {
                    let r0 = prefix
                        .first_challenge
                        .expect("round 1 prefix polynomial requested before ingesting round 0");
                    let (virt_poly, rel_poly) = prefix.skip_state.reconstruct_round1_polys(r0);
                    (virt_poly, rel_poly)
                }
            };
            let combined = self.combine_polys(&virt_poly, &rel_poly);
            self.prev_norm_poly = Some(virt_poly);
            combined
        } else {
            match &self.witness_state {
                WitnessState::CompactPrefix(compact_witness) => {
                    if use_partial_lane_coefficient_round {
                        let (virt_q_coeffs, rel_coeffs) = self
                            .compute_compact_partial_lane_coefficient_round_terms(compact_witness);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    } else if use_partial_lane_round {
                        let (virt_terms, rel_coeffs) =
                            self.compute_compact_partial_lane_round_terms(compact_witness);
                        self.combine_terms(virt_terms, rel_coeffs)
                    } else {
                        let (virt_q_coeffs, rel_coeffs) =
                            self.compute_round_compact_dense_terms(compact_witness);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    }
                }
                WitnessState::FoldedSuffix(folded_witness) => {
                    if use_partial_lane_coefficient_round {
                        let (virt_q_coeffs, rel_coeffs) = self
                            .compute_folded_partial_lane_coefficient_round_terms(folded_witness);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    } else if use_partial_lane_round {
                        let (virt_q_coeffs, rel_coeffs) =
                            self.compute_folded_partial_lane_round_terms(folded_witness);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    } else {
                        let (virt_q_coeffs, rel_coeffs) =
                            self.compute_folded_dense_round_terms(folded_witness);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    }
                }
            }
        };
        self.scan_time_total += t_scan.elapsed().as_secs_f64();
        poly
    }

    #[inline]
    pub(super) fn build_compact_w_fold_lut(compact_witness: &[i8], r: E) -> CompactPairFoldLut<E> {
        let min_w = compact_witness
            .iter()
            .copied()
            .map(i32::from)
            .min()
            .unwrap_or(0)
            .min(0);
        let max_w = compact_witness
            .iter()
            .copied()
            .map(i32::from)
            .max()
            .unwrap_or(0)
            .max(0);
        CompactPairFoldLut::from_contiguous_range(min_w as i16, max_w as i16, r)
    }

    pub(super) fn materialize_compact_witness(
        compact_witness: &[i8],
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        cfg_into_iter!(0..compact_witness.len() / 2)
            .map(|j| {
                fold_lut.fold(
                    i16::from(compact_witness[2 * j]),
                    i16::from(compact_witness[2 * j + 1]),
                )
            })
            .collect()
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps + HasOptimizedFold> SumcheckInstanceProver<E>
    for RelationRangeImageProver<E>
{
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
        if let Some(poly) = self.cached_round_poly.take() {
            poly
        } else {
            self.compute_current_round_poly_from_state()
        }
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        let t_fold = Instant::now();
        let _span = tracing::info_span!("RelationRangeImageProver::fold_round").entered();
        if let Some(prev_norm_poly) = self.prev_norm_poly.take() {
            self.prev_norm_claim = prev_norm_poly.evaluate(&r);
        }

        if self.using_deferred_compact_prefix() {
            let rounds_completed = self.rounds_completed;
            self.split_eq.bind(r);
            if rounds_completed == 0 {
                self.ensure_deferred_compact_prefix().first_challenge = Some(r);
            } else {
                let r0 = {
                    let prefix = self.ensure_deferred_compact_prefix();
                    prefix
                        .first_challenge
                        .expect("round 1 ingest requires the round 0 challenge")
                };
                let coeff_count = self.common_alpha_factor.len();
                let alpha_round2 = Self::fold_alpha_two_rounds(&self.common_alpha_factor, r0, r);
                self.evaluation_trace.fold_two_coefficients(r0, r);
                // This is the two-round coefficient handoff, so the ordinary one-round
                // trace transition below is deliberately bypassed.
                let mut round2_terms = None;
                self.witness_state = match mem::replace(
                    &mut self.witness_state,
                    WitnessState::FoldedSuffix(Vec::new()),
                ) {
                    WitnessState::CompactPrefix(compact_witness) => {
                        if self.coefficient_bits() > 2 {
                            let (folded_witness, virt_terms, rel_coeffs) = self
                                .materialize_two_round_compact_prefix_and_compute_next_round(
                                    &compact_witness,
                                    &alpha_round2,
                                    &self.evaluation_trace,
                                    r0,
                                    r,
                                );
                            round2_terms = Some((virt_terms, rel_coeffs));
                            WitnessState::FoldedSuffix(folded_witness)
                        } else {
                            WitnessState::FoldedSuffix(Self::materialize_two_round_compact_prefix(
                                &compact_witness,
                                self.live_lane_count,
                                coeff_count,
                                r0,
                                r,
                            ))
                        }
                    }
                    WitnessState::FoldedSuffix(_) => {
                        unreachable!("two-round prefix should hold compact witness")
                    }
                };
                self.common_alpha_factor = alpha_round2;
                self.deferred_compact_prefix = None;
                self.compact_prefix_stage1_point = None;
                if let Some((virt_terms, rel_coeffs)) = round2_terms {
                    self.cached_round_poly = Some(self.combine_terms(virt_terms, rel_coeffs));
                }
            }
            self.rounds_completed += 1;
            if self.rounds_completed < self.num_vars {
                if self.cached_round_poly.is_none() {
                    self.cached_round_poly = Some(self.compute_current_round_poly_from_state());
                }
            } else {
                self.cached_round_poly = None;
            }
            drop(_span);
            self.fold_time_total += t_fold.elapsed().as_secs_f64();
            if self.rounds_completed == self.num_vars {
                tracing::debug!(
                    rounds = self.num_vars,
                    scan_s = self.scan_time_total,
                    fold_s = self.fold_time_total,
                    "stage2 sumcheck rounds complete"
                );
            }
            return;
        }

        self.split_eq.bind(r);
        let folding_lane_round = !self.in_coefficient_round();
        let use_partial_lane_round = self.use_partial_lane_round();
        let use_partial_lane_coefficient_round = self.use_partial_lane_coefficient_round();
        let in_coefficient_round = self.in_coefficient_round();
        let fuse_next_coefficient_round = use_partial_lane_coefficient_round
            && self.rounds_completed + 1 < self.coefficient_bits();
        let fuse_next_folded_partial_lane =
            use_partial_lane_round && self.next_uses_partial_lane_round();
        let coeff_count = self.common_alpha_factor.len();
        let live_lane_count = self.live_lane_count;
        let mut fused_coefficient_round = false;
        let mut fused_folded_partial_lane = false;

        self.witness_state = match mem::replace(
            &mut self.witness_state,
            WitnessState::FoldedSuffix(Vec::new()),
        ) {
            WitnessState::CompactPrefix(compact_witness) => {
                let fold_lut = Self::build_compact_w_fold_lut(&compact_witness, r);
                let folded_witness = if folding_lane_round && use_partial_lane_round {
                    Self::fold_compact_partial_lanes(
                        &compact_witness,
                        live_lane_count,
                        coeff_count,
                        &fold_lut,
                    )
                } else {
                    Self::materialize_compact_witness(&compact_witness, &fold_lut)
                };
                self.fold_evaluation_trace_for_current_round(r);
                WitnessState::FoldedSuffix(folded_witness)
            }
            WitnessState::FoldedSuffix(folded_witness) => {
                if folding_lane_round && use_partial_lane_round {
                    if fuse_next_folded_partial_lane {
                        // Fold trace before the fused kernel so relation terms use the same
                        // post-fold table as `compute_folded_partial_lane_round_terms`.
                        self.fold_evaluation_trace_for_current_round(r);
                        let (
                            next_folded_witness,
                            next_relation_lane_weights,
                            virt_terms,
                            rel_coeffs,
                        ) = self
                            .fuse_folded_partial_lane_and_compute_next_round(&folded_witness, r);
                        self.relation_lane_weights = next_relation_lane_weights;
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
                        self.fold_evaluation_trace_for_current_round(r);
                        WitnessState::FoldedSuffix(next_folded_witness)
                    }
                } else if in_coefficient_round && use_partial_lane_coefficient_round {
                    self.fold_evaluation_trace_for_current_round(r);
                    if fuse_next_coefficient_round {
                        let mut next_alpha_factor = self.common_alpha_factor.clone();
                        fold_evals_in_place(&mut next_alpha_factor, r);
                        let (next_folded_witness, virt_terms, rel_coeffs) = self
                            .fuse_folded_coefficients_and_compute_next_round(
                                &folded_witness,
                                &next_alpha_factor,
                                r,
                            );
                        self.common_alpha_factor = next_alpha_factor;
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
                    self.fold_evaluation_trace_for_current_round(r);
                    WitnessState::FoldedSuffix(folded_witness)
                }
            }
        };

        if folding_lane_round {
            if use_partial_lane_round {
                if !fused_folded_partial_lane {
                    self.relation_lane_weights =
                        Self::fold_relation_lane_weights(&self.relation_lane_weights, r);
                }
            } else {
                fold_evals_in_place(&mut self.relation_lane_weights, r);
            }
            self.live_lane_count = self.live_lane_count.div_ceil(2);
        } else if !fused_coefficient_round {
            fold_evals_in_place(&mut self.common_alpha_factor, r);
        }

        self.rounds_completed += 1;
        if self.rounds_completed < self.num_vars {
            if self.cached_round_poly.is_none() {
                self.cached_round_poly = Some(self.compute_current_round_poly_from_state());
            }
        } else {
            self.cached_round_poly = None;
        }
        drop(_span);
        self.fold_time_total += t_fold.elapsed().as_secs_f64();

        if self.rounds_completed == self.num_vars {
            tracing::debug!(
                rounds = self.num_vars,
                scan_s = self.scan_time_total,
                fold_s = self.fold_time_total,
                "stage2 sumcheck rounds complete"
            );
        }
    }
}
