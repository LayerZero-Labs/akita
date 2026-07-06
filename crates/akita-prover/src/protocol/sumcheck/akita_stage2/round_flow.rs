use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    pub(super) fn compute_current_round_poly_from_state(&mut self) -> UniPoly<E> {
        let t_scan = Instant::now();
        let use_round_batching = self.using_initial_round_batch();
        let rounds_completed = self.rounds_completed;
        let poly = if use_round_batching {
            let (virt_poly, rel_poly) = {
                let prefix = self.ensure_initial_round_batch();
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
            let (virt_q_coeffs, rel_coeffs) = match &self.witness_table {
                WitnessTable::Compact(w_compact) => {
                    self.scan_round(WitnessPolynomial::CompactDigits(w_compact))
                }
                WitnessTable::Full(w_full) => {
                    self.scan_round(WitnessPolynomial::FieldEvals(w_full))
                }
            };
            self.combine_terms(virt_q_coeffs, rel_coeffs)
        };
        self.scan_time_total += t_scan.elapsed().as_secs_f64();
        poly
    }

    #[inline]
    pub(super) fn build_compact_w_fold_lut(w_compact: &[i8], r: E) -> CompactPairFoldLut<E> {
        let min_w = w_compact
            .iter()
            .copied()
            .map(i32::from)
            .min()
            .unwrap_or(0)
            .min(0);
        let max_w = w_compact
            .iter()
            .copied()
            .map(i32::from)
            .max()
            .unwrap_or(0)
            .max(0);
        CompactPairFoldLut::from_contiguous_range(min_w as i16, max_w as i16, r)
    }

    pub(super) fn fold_compact_to_full(
        w_compact: &[i8],
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        Self::fold_witness_flat_compact(w_compact, fold_lut)
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps + HasOptimizedFold> SumcheckInstanceProver<E>
    for AkitaStage2Prover<E>
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
        let _span = tracing::info_span!("AkitaStage2Prover::fold_round").entered();
        if let Some(prev_norm_poly) = self.prev_norm_poly.take() {
            self.prev_norm_claim = prev_norm_poly.evaluate(&r);
        }

        if self.using_initial_round_batch() {
            let rounds_completed = self.rounds_completed;
            self.split_eq.bind(r);
            if rounds_completed == 0 {
                self.ensure_initial_round_batch().first_challenge = Some(r);
            } else {
                let r0 = {
                    let prefix = self.ensure_initial_round_batch();
                    prefix
                        .first_challenge
                        .expect("round 1 ingest requires the round 0 challenge")
                };
                let coeff_len = self.relation_weight_coeff_len();
                let relation_round2 = Self::fold_relation_weight_initial_batch(
                    self.relation_weight.evals(),
                    self.live_segments,
                    coeff_len,
                    r0,
                    r,
                );
                let mut round2_terms = None;
                self.witness_table =
                    match mem::replace(&mut self.witness_table, WitnessTable::Full(Vec::new())) {
                        WitnessTable::Compact(w_compact) => {
                            if self.coeff_bits() > 2 {
                                let (w_full, virt_terms, rel_coeffs) = self.run_fused_fold_scan(
                                    FusedFoldScan::InitialRound2 {
                                        w_compact: &w_compact,
                                        relation_round2: &relation_round2,
                                        r0,
                                        r1: r,
                                    },
                                );
                                round2_terms = Some((virt_terms, rel_coeffs));
                                WitnessTable::Full(w_full)
                            } else {
                                WitnessTable::Full(Self::fold_witness_initial_batch(
                                    &w_compact,
                                    self.live_segments,
                                    coeff_len,
                                    r0,
                                    r,
                                ))
                            }
                        }
                        WitnessTable::Full(_) => {
                            unreachable!("initial round batch should hold compact witness")
                        }
                    };
                let next_coeff_len = coeff_len >> 2;
                self.relation_weight = RelationWeightPolynomial::from_live_evals(
                    relation_round2,
                    self.live_segments * next_coeff_len,
                )
                .expect("relation weight round-2 fold preserves shape");
                self.relation_coeff_len = next_coeff_len;
                self.initial_round_batch = None;
                self.prefix_r_stage1 = None;
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
        let folding_segment_round = !self.in_coefficient_round();
        let fold_kind = self.fold_round_kind(folding_segment_round);
        let fuse_next_segment_axis_round =
            fold_kind == FoldRoundKind::EmbeddedSegmentAxis
                && self.next_use_segment_prefix_round_after_current();
        let coeff_len = self.relation_weight_coeff_len();
        let live_segments = self.live_segments;

        self.witness_table =
            match mem::replace(&mut self.witness_table, WitnessTable::Full(Vec::new())) {
                WitnessTable::Compact(w_compact) => {
                    let fold_lut = Self::build_compact_w_fold_lut(&w_compact, r);
                    let witness_kind =
                        Self::witness_fold_kind(fold_kind, true);
                    let w_full = Self::fold_witness_polynomial(
                        WitnessFoldInput::Compact {
                            digits: &w_compact,
                            fold_lut: &fold_lut,
                        },
                        witness_kind,
                        live_segments,
                        coeff_len,
                    );
                    self.fold_relation_weight(r, fold_kind);
                    WitnessTable::Full(w_full)
                }
                WitnessTable::Full(w_full) => {
                    if fold_kind == FoldRoundKind::EmbeddedSegmentAxis {
                        if fuse_next_segment_axis_round {
                            self.fold_relation_weight(r, fold_kind);
                            let (next_w_full, virt_terms, rel_coeffs) = self.run_fused_fold_scan(
                                FusedFoldScan::SegmentAxis {
                                    w_full: &w_full,
                                    challenge: r,
                                },
                            );
                            self.cached_round_poly =
                                Some(self.combine_terms(virt_terms, rel_coeffs));
                            WitnessTable::Full(next_w_full)
                        } else {
                            let next_w_full = Self::fold_witness_polynomial(
                                WitnessFoldInput::Full {
                                    evals: &w_full,
                                    challenge: r,
                                    use_local_view_flat_fold: false,
                                },
                                fold_kind,
                                live_segments,
                                coeff_len,
                            );
                            self.fold_relation_weight(r, fold_kind);
                            WitnessTable::Full(next_w_full)
                        }
                    } else {
                        let next_w_full = Self::fold_witness_full_owned(
                            w_full,
                            fold_kind,
                            live_segments,
                            coeff_len,
                            r,
                            self.geometry.local_view().is_some(),
                        );
                        self.fold_relation_weight(r, fold_kind);
                        WitnessTable::Full(next_w_full)
                    }
                }
            };

        if folding_segment_round {
            self.live_segments = self.live_segments.div_ceil(2);
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
