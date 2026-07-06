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
                let relation_round2 = Self::fold_relation_weight_through_two_challenges(
                    self.relation_weight.evals(),
                    r0,
                    r,
                );
                self.witness_table =
                    match mem::replace(&mut self.witness_table, WitnessTable::Full(Vec::new())) {
                        WitnessTable::Compact(w_compact) => WitnessTable::Full(
                            Self::fold_witness_through_two_challenges(&w_compact, r0, r),
                        ),
                        WitnessTable::Full(_) => {
                            unreachable!("initial round batch should hold compact witness")
                        }
                    };
                self.relation_weight = RelationWeightPolynomial::from_live_evals(
                    relation_round2.clone(),
                    relation_round2.len(),
                )
                .expect("relation weight round-2 fold preserves shape");
                self.relation_coeff_len = self.relation_weight.evals().len();
                self.live_segments = 1;
                self.initial_round_batch = None;
                self.prefix_r_stage1 = None;
            }
            self.rounds_completed += 1;
            if self.rounds_completed < self.num_vars {
                self.cached_round_poly = Some(self.compute_current_round_poly_from_state());
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
                    WitnessTable::Full(Self::fold_witness_compact_to_field(
                        &w_compact,
                        &fold_lut,
                    ))
                }
                WitnessTable::Full(w_full) => {
                    self.fold_relation_weight_flat(r);
                    WitnessTable::Full(Self::fold_witness_field_flat(w_full, r))
                }
            };

        self.rounds_completed += 1;
        if self.rounds_completed < self.num_vars {
            self.cached_round_poly = Some(self.compute_current_round_poly_from_state());
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
