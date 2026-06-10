use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    pub(super) fn compute_current_round_poly_from_state(&mut self) -> UniPoly<E> {
        let t_scan = Instant::now();
        let use_two_round_prefix = self.using_two_round_prefix();
        let use_prefix_y_round = !use_two_round_prefix && self.use_prefix_y_round();
        let use_prefix_x_round = !use_two_round_prefix && self.use_prefix_x_round();
        let rounds_completed = self.rounds_completed;
        let poly = if use_two_round_prefix {
            let (virt_poly, rel_poly) = {
                let prefix = self.ensure_two_round_prefix();
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

    pub(super) fn fold_compact_trace_to_full(trace: &[E], r: E) -> Vec<E> {
        cfg_into_iter!(0..trace.len() / 2)
            .map(|j| trace[2 * j] + r * (trace[2 * j + 1] - trace[2 * j]))
            .collect()
    }

    pub(super) fn fold_trace_pair_in_place(trace: &mut Vec<E>, r: E) {
        let half = trace.len() / 2;
        for i in 0..half {
            let a = trace[2 * i];
            let b = trace[2 * i + 1];
            trace[i] = a + r * (b - a);
        }
        trace.truncate(half);
    }

    #[allow(clippy::too_many_arguments)]
    fn fold_trace_for_w_update(
        trace: &mut Vec<E>,
        live_x_cols: usize,
        y_len: usize,
        r: E,
        folding_x_round: bool,
        use_prefix_x_round: bool,
        in_y_round: bool,
        use_prefix_y_round: bool,
    ) {
        if folding_x_round && use_prefix_x_round {
            *trace = Self::fold_full_prefix_x(trace, live_x_cols, y_len, r);
        } else if in_y_round && use_prefix_y_round {
            *trace = Self::fold_full_prefix_y(trace, live_x_cols, y_len, r);
        } else {
            Self::fold_trace_pair_in_place(trace, r);
        }
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

        if self.using_two_round_prefix() {
            let rounds_completed = self.rounds_completed;
            self.split_eq.bind(r);
            if rounds_completed == 0 {
                self.ensure_two_round_prefix().first_challenge = Some(r);
            } else {
                let r0 = {
                    let prefix = self.ensure_two_round_prefix();
                    prefix
                        .first_challenge
                        .expect("round 1 ingest requires the round 0 challenge")
                };
                let y_len = self.alpha_compact.len();
                let alpha_round2 = Self::fold_alpha_to_round2(&self.alpha_compact, r0, r);
                let trace_round2 = self.trace_compact.as_ref().map(|trace| {
                    Self::fold_trace_compact_to_round2(trace, self.live_x_cols, y_len, r0, r)
                });
                let mut round2_terms = None;
                self.w_table = match mem::replace(&mut self.w_table, WTable::Full(Vec::new())) {
                    WTable::Compact(w_compact) => {
                        if self.ring_bits() > 2 {
                            let (w_full, virt_terms, rel_coeffs) = self
                                .fuse_compact_to_round2_and_compute_round(
                                    &w_compact,
                                    &alpha_round2,
                                    trace_round2.as_deref(),
                                    r0,
                                    r,
                                );
                            round2_terms = Some((virt_terms, rel_coeffs));
                            WTable::Full(w_full)
                        } else {
                            WTable::Full(Self::fold_compact_to_round2(
                                &w_compact,
                                self.live_x_cols,
                                y_len,
                                r0,
                                r,
                            ))
                        }
                    }
                    WTable::Full(_) => unreachable!("two-round prefix should hold compact witness"),
                };
                self.trace_compact = trace_round2;
                self.alpha_compact = alpha_round2;
                self.two_round_prefix = None;
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
        let y_len = self.alpha_compact.len();
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
                if let Some(trace) = self.trace_compact.as_mut() {
                    if folding_x_round && use_prefix_x_round {
                        *trace = Self::fold_full_prefix_x(trace, live_x_cols, y_len, r);
                    } else {
                        *trace = Self::fold_compact_trace_to_full(trace, r);
                    }
                }
                WTable::Full(w_full)
            }
            WTable::Full(w_full) => {
                if folding_x_round && use_prefix_x_round {
                    if fuse_next_full_prefix_x {
                        let (next_w_full, next_m_compact, virt_terms, rel_coeffs) =
                            self.fuse_full_prefix_x_and_compute_round(&w_full, r);
                        self.m_compact = next_m_compact;
                        self.cached_round_poly = Some(self.combine_terms(virt_terms, rel_coeffs));
                        fused_full_prefix_x = true;
                        if let Some(trace) = self.trace_compact.as_mut() {
                            Self::fold_trace_for_w_update(
                                trace,
                                live_x_cols,
                                y_len,
                                r,
                                folding_x_round,
                                use_prefix_x_round,
                                in_y_round,
                                use_prefix_y_round,
                            );
                        }
                        WTable::Full(next_w_full)
                    } else {
                        let next_w_full = Self::fold_full_prefix_x(&w_full, live_x_cols, y_len, r);
                        if let Some(trace) = self.trace_compact.as_mut() {
                            Self::fold_trace_for_w_update(
                                trace,
                                live_x_cols,
                                y_len,
                                r,
                                folding_x_round,
                                use_prefix_x_round,
                                in_y_round,
                                use_prefix_y_round,
                            );
                        }
                        WTable::Full(next_w_full)
                    }
                } else if in_y_round && use_prefix_y_round {
                    if let Some(trace) = self.trace_compact.as_mut() {
                        Self::fold_trace_for_w_update(
                            trace,
                            live_x_cols,
                            y_len,
                            r,
                            folding_x_round,
                            use_prefix_x_round,
                            in_y_round,
                            use_prefix_y_round,
                        );
                    }
                    WTable::Full(Self::fold_full_prefix_y(&w_full, live_x_cols, y_len, r))
                } else {
                    let mut w_full = w_full;
                    fold_evals_in_place(&mut w_full, r);
                    if let Some(trace) = self.trace_compact.as_mut() {
                        Self::fold_trace_pair_in_place(trace, r);
                    }
                    WTable::Full(w_full)
                }
            }
        };

        if folding_x_round {
            if use_prefix_x_round {
                if !fused_full_prefix_x {
                    self.m_compact = Self::fold_m_prefix(&self.m_compact, r);
                }
            } else {
                fold_evals_in_place(&mut self.m_compact, r);
            }
            self.live_x_cols = self.live_x_cols.div_ceil(2);
        } else {
            fold_evals_in_place(&mut self.alpha_compact, r);
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
