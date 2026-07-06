use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    pub(super) fn compute_current_round_poly_from_state(&mut self) -> UniPoly<E> {
        let t_scan = Instant::now();
        let use_round_batching = self.using_initial_round_batch();
        let use_prefix_y_round = !use_round_batching && self.use_prefix_y_round();
        let use_prefix_x_round = !use_round_batching && self.use_prefix_x_round();
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
            match &self.w_table {
                WTable::Compact(w_compact) => {
                    if use_prefix_y_round {
                        let (virt_q_coeffs, rel_coeffs) =
                            self.compute_round_compact_prefix_y_terms(w_compact);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    } else if use_prefix_x_round {
                        let (virt_poly, rel_poly) =
                            self.compute_round_compact_prefix_x_polys(w_compact);
                        let combined = self.combine_polys(&virt_poly, &rel_poly);
                        self.prev_norm_poly = Some(virt_poly);
                        combined
                    } else {
                        let (virt_q_coeffs, rel_coeffs) =
                            self.compute_round_compact_dense_terms(w_compact);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    }
                }
                WTable::Full(w_full) => {
                    if use_prefix_y_round {
                        let (virt_q_coeffs, rel_coeffs) =
                            self.compute_round_full_prefix_y_terms(w_full);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    } else if use_prefix_x_round {
                        let (virt_q_coeffs, rel_coeffs) =
                            self.compute_round_full_prefix_x_terms(w_full);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    } else {
                        let (virt_q_coeffs, rel_coeffs) =
                            self.compute_round_full_dense_terms(w_full);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    }
                }
            }
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
        cfg_into_iter!(0..w_compact.len() / 2)
            .map(|j| fold_lut.fold(i16::from(w_compact[2 * j]), i16::from(w_compact[2 * j + 1])))
            .collect()
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
                let y_len = self.relation_weight_y_len();
                let relation_round2 = Self::fold_relation_weight_through_initial_batch(
                    self.relation_weight.evals(),
                    self.relation_weight.live_x_cols(),
                    y_len,
                    r0,
                    r,
                );
                let mut round2_terms = None;
                self.w_table = match mem::replace(&mut self.w_table, WTable::Full(Vec::new())) {
                    WTable::Compact(w_compact) => {
                        if self.ring_bits() > 2 {
                            let (w_full, virt_terms, rel_coeffs) = self
                                .fuse_compact_to_round2_and_compute_round(
                                    &w_compact,
                                    &relation_round2,
                                    r0,
                                    r,
                                );
                            round2_terms = Some((virt_terms, rel_coeffs));
                            WTable::Full(w_full)
                        } else {
                            WTable::Full(Self::fold_compact_through_initial_batch(
                                &w_compact,
                                self.live_x_cols,
                                y_len,
                                r0,
                                r,
                            ))
                        }
                    }
                    WTable::Full(_) => {
                        unreachable!("initial round batch should hold compact witness")
                    }
                };
                let next_y_len = y_len >> 2;
                self.relation_weight = RelationWeightPolynomial::from_evals(
                    relation_round2,
                    next_y_len,
                    self.relation_weight.live_x_cols(),
                    self.relation_weight.live_x_cols() * next_y_len,
                )
                .expect("relation weight round-2 fold preserves shape");
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
        let folding_x_round = !self.in_y_round();
        let use_prefix_x_round = self.use_prefix_x_round();
        let use_prefix_y_round = self.use_prefix_y_round();
        let in_y_round = self.in_y_round();
        let fuse_next_full_prefix_x =
            use_prefix_x_round && self.next_use_prefix_x_round_after_current();
        let y_len = self.relation_weight_y_len();
        let live_x_cols = self.live_x_cols;
        let mut fused_full_prefix_x = false;

        self.w_table = match mem::replace(&mut self.w_table, WTable::Full(Vec::new())) {
            WTable::Compact(w_compact) => {
                let fold_lut = Self::build_compact_w_fold_lut(&w_compact, r);
                let w_full = if folding_x_round && use_prefix_x_round {
                    Self::fold_compact_prefix_x(&w_compact, live_x_cols, y_len, &fold_lut)
                } else {
                    Self::fold_compact_to_full(&w_compact, &fold_lut)
                };
                self.fold_relation_weight_for_round(r, folding_x_round, use_prefix_x_round, use_prefix_y_round, in_y_round);
                WTable::Full(w_full)
            }
            WTable::Full(w_full) => {
                if folding_x_round && use_prefix_x_round {
                    if fuse_next_full_prefix_x {
                        self.fold_relation_weight_for_round(
                            r,
                            folding_x_round,
                            use_prefix_x_round,
                            use_prefix_y_round,
                            in_y_round,
                        );
                        let (next_w_full, virt_terms, rel_coeffs) =
                            self.fuse_full_prefix_x_and_compute_round(&w_full, r);
                        self.cached_round_poly = Some(self.combine_terms(virt_terms, rel_coeffs));
                        fused_full_prefix_x = true;
                        WTable::Full(next_w_full)
                    } else {
                        let next_w_full = Self::fold_full_prefix_x(&w_full, live_x_cols, y_len, r);
                        self.fold_relation_weight_for_round(r, folding_x_round, use_prefix_x_round, use_prefix_y_round, in_y_round);
                        WTable::Full(next_w_full)
                    }
                } else if in_y_round && use_prefix_y_round {
                    self.fold_relation_weight_for_round(r, folding_x_round, use_prefix_x_round, use_prefix_y_round, in_y_round);
                    WTable::Full(Self::fold_full_prefix_y(&w_full, live_x_cols, y_len, r))
                } else {
                    let mut w_full = w_full;
                    fold_evals_in_place(&mut w_full, r);
                    self.fold_relation_weight_for_round(r, folding_x_round, use_prefix_x_round, use_prefix_y_round, in_y_round);
                    WTable::Full(w_full)
                }
            }
        };

        if folding_x_round {
            self.live_x_cols = self.live_x_cols.div_ceil(2);
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
