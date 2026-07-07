use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    pub(super) fn compute_current_round_poly_from_state(&mut self) -> UniPoly<E> {
        let t_scan = Instant::now();
        let use_round_batching = self.using_initial_round_batch();
        let rounds_completed = self.rounds_completed;
        let poly = if use_round_batching {
            let (virt_poly, rel_poly) = {
                let batch = self.ensure_initial_round_batch();
                if rounds_completed == 0 {
                    let (virt_poly, rel_poly) = batch.skip_state.reconstruct_round0_polys();
                    (virt_poly, rel_poly)
                } else {
                    let r0 = batch.first_challenge.expect(
                        "round 1 initial-batch polynomial requested before ingesting round 0",
                    );
                    let (virt_poly, rel_poly) = batch.skip_state.reconstruct_round1_polys(r0);
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
                    let batch = self.ensure_initial_round_batch();
                    batch
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
                                let (w_full, virt_terms, rel_coeffs) =
                                    self.run_fused_fold_scan(FusedFoldScan::InitialRound2 {
                                        w_compact: &w_compact,
                                        relation_round2: &relation_round2,
                                        r0,
                                        r1: r,
                                    });
                                round2_terms = Some((virt_terms, rel_coeffs));
                                WitnessTable::Full(w_full)
                            } else {
                                WitnessTable::Full(Self::fold_witness_through_two_challenges(
                                    &w_compact, r0, r,
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
                self.initial_batch_stage1_point = None;
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

        self.witness_table =
            match mem::replace(&mut self.witness_table, WitnessTable::Full(Vec::new())) {
                WitnessTable::Compact(w_compact) => {
                    let fold_lut = Self::build_compact_w_fold_lut(&w_compact, r);
                    self.fold_relation_weight_flat(r);
                    WitnessTable::Full(Self::fold_witness_flat_compact(&w_compact, &fold_lut))
                }
                WitnessTable::Full(w_full) => {
                    self.fold_relation_weight_flat(r);
                    WitnessTable::Full(Self::fold_witness_field_flat(w_full, r))
                }
            };

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
