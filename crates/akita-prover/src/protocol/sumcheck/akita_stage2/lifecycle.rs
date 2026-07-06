use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    /// Create a fused stage-2 virtual-claim + relation sumcheck prover.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "AkitaStage2Prover::new")]
    pub(crate) fn new(
        batching_coeff: E,
        w_evals_compact: Vec<i8>,
        stage1_point: &[E],
        s_claim: E,
        b: usize,
        relation_weight_evals: Vec<E>,
        relation_weight_claim: E,
        live_segments: usize,
        segment_bits: usize,
        coeff_bits: usize,
    ) -> Result<Self, AkitaError> {
        let num_vars = segment_bits.checked_add(coeff_bits).ok_or_else(|| {
            AkitaError::InvalidInput("stage-2 challenge width overflow".to_string())
        })?;
        if live_segments == 0 {
            return Err(AkitaError::InvalidInput(
                "live_segments must be at least 1".to_string(),
            ));
        }
        let col_bits_u32 = u32::try_from(segment_bits)
            .map_err(|_| AkitaError::InvalidInput("stage-2 column width overflow".to_string()))?;
        let ring_bits_u32 = u32::try_from(coeff_bits)
            .map_err(|_| AkitaError::InvalidInput("stage-2 ring width overflow".to_string()))?;
        let segment_capacity = 1usize
            .checked_shl(col_bits_u32)
            .ok_or_else(|| AkitaError::InvalidInput("stage-2 column width overflow".to_string()))?;
        if live_segments > segment_capacity {
            return Err(AkitaError::InvalidSize {
                expected: segment_capacity,
                actual: live_segments,
            });
        }
        let coeff_len = 1usize
            .checked_shl(ring_bits_u32)
            .ok_or_else(|| AkitaError::InvalidInput("stage-2 ring width overflow".to_string()))?;
        let witness_len = live_segments
            .checked_mul(coeff_len)
            .ok_or_else(|| AkitaError::InvalidInput("stage-2 witness size overflow".to_string()))?;
        if w_evals_compact.len() != witness_len {
            return Err(AkitaError::InvalidSize {
                expected: witness_len,
                actual: w_evals_compact.len(),
            });
        }
        if stage1_point.len() != num_vars {
            return Err(AkitaError::InvalidSize {
                expected: num_vars,
                actual: stage1_point.len(),
            });
        }
        let relation_weight =
            RelationWeightPolynomial::from_live_evals(relation_weight_evals, witness_len)?;

        let input_claim = batching_coeff * s_claim + relation_weight_claim;

        Ok(Self {
            witness_table: WitnessTable::Compact(w_evals_compact),
            b,
            batching_coeff,
            s_claim,
            input_claim,
            split_eq: GruenSplitEq::with_initial_scalar(stage1_point, batching_coeff)?,
            relation_weight,
            live_segments,
            relation_coeff_len: coeff_len,
            segment_bits,
            num_vars,
            prev_norm_claim: batching_coeff * s_claim,
            prev_norm_poly: None,
            prefix_r_stage1: can_use_stage2_initial_round_batch(coeff_bits, b)
                .then(|| stage1_point.to_vec()),
            initial_round_batch: None,
            cached_round_poly: None,
            scan_time_total: 0.0,
            fold_time_total: 0.0,
            rounds_completed: 0,
        })
    }

    /// Return the fully folded witness evaluation after the final round.
    ///
    /// # Panics
    ///
    /// Panics if called before the witness table has been fully folded to a
    /// single field element.
    pub fn final_w_eval(&self) -> E {
        match &self.witness_table {
            WitnessTable::Full(w_full) => {
                assert_eq!(w_full.len(), 1, "witness_table not fully folded");
                w_full[0]
            }
            WitnessTable::Compact(_) => panic!("witness_table remained compact after final fold"),
        }
    }

    #[inline]
    pub(super) fn coeff_bits(&self) -> usize {
        self.num_vars - self.segment_bits
    }

    #[inline]
    pub(super) fn coefficient_rounds_completed(&self) -> usize {
        self.rounds_completed.min(self.coeff_bits())
    }

    #[inline]
    pub(super) fn segment_rounds_completed(&self) -> usize {
        self.rounds_completed.saturating_sub(self.coeff_bits())
    }

    #[inline]
    pub(super) fn in_coefficient_round(&self) -> bool {
        self.rounds_completed < self.coeff_bits()
    }

    #[inline]
    pub(super) fn current_coefficient_width(&self) -> usize {
        self.coeff_bits()
            .saturating_sub(self.coefficient_rounds_completed())
    }

    #[inline]
    pub(super) fn current_segment_width(&self) -> usize {
        self.segment_bits
            .saturating_sub(self.segment_rounds_completed())
    }

    #[inline]
    pub(super) fn current_segment_capacity(&self) -> usize {
        1usize << self.current_segment_width()
    }

    #[inline]
    pub(super) fn use_coefficient_prefix_round(&self) -> bool {
        self.in_coefficient_round() && self.live_segments < self.current_segment_capacity()
    }

    #[inline]
    pub(super) fn use_segment_prefix_round(&self) -> bool {
        self.rounds_completed >= self.coeff_bits()
            && self.segment_rounds_completed() < self.segment_bits
            && self.live_segments < self.current_segment_capacity()
    }

    #[inline]
    pub(super) fn next_use_segment_prefix_round_after_current(&self) -> bool {
        self.rounds_completed >= self.coeff_bits()
            && self.segment_rounds_completed() + 1 < self.segment_bits
            && self.live_segments.div_ceil(2) < (self.current_segment_capacity() / 2)
    }

    #[inline]
    pub(crate) fn can_use_stage2_initial_round_batch(&self) -> bool {
        self.prefix_r_stage1.is_some()
    }

    #[inline]
    pub(super) fn using_initial_round_batch(&self) -> bool {
        self.rounds_completed < 2 && self.can_use_stage2_initial_round_batch()
    }

    #[inline]
    pub(super) fn can_skip_norm_linear_coeff(&self) -> bool {
        self.split_eq.can_recover_linear_q_term_from_claim()
    }

    #[inline]
    pub(super) fn norm_poly_from_terms(&self, virt_terms: NormRoundTerms<E>) -> UniPoly<E> {
        match virt_terms {
            NormRoundTerms::Full(virt_q_coeffs) => {
                self.split_eq.gruen_mul(&coeffs_to_poly(virt_q_coeffs))
            }
            NormRoundTerms::SkipLinear([q_constant, q_quadratic]) => self
                .split_eq
                .try_gruen_poly_deg_3(q_constant, q_quadratic, self.prev_norm_claim)
                .expect("split-eq norm claim recovery should succeed"),
        }
    }

    #[inline]
    pub(super) fn polys_from_terms(
        &self,
        virt_terms: NormRoundTerms<E>,
        rel_coeffs: [E; 3],
    ) -> (UniPoly<E>, UniPoly<E>) {
        let virt_poly = self.norm_poly_from_terms(virt_terms);
        let rel_poly = coeffs_to_poly(rel_coeffs);
        (virt_poly, rel_poly)
    }

    #[inline]
    pub(super) fn combine_polys(
        &self,
        virt_poly: &UniPoly<E>,
        relation_poly: &UniPoly<E>,
    ) -> UniPoly<E> {
        let max_len = virt_poly.coeffs.len().max(relation_poly.coeffs.len());
        let mut combined = vec![E::zero(); max_len];
        for (i, c) in virt_poly.coeffs.iter().enumerate() {
            combined[i] += *c;
        }
        for (i, c) in relation_poly.coeffs.iter().enumerate() {
            combined[i] += *c;
        }
        UniPoly::from_coeffs(combined)
    }

    #[inline]
    pub(super) fn combine_terms(
        &mut self,
        virt_terms: NormRoundTerms<E>,
        rel_coeffs: [E; 3],
    ) -> UniPoly<E> {
        let (virt_poly, relation_poly) = self.polys_from_terms(virt_terms, rel_coeffs);
        let combined = self.combine_polys(&virt_poly, &relation_poly);
        self.prev_norm_poly = Some(virt_poly);
        combined
    }

    pub(super) fn ensure_initial_round_batch(&mut self) -> &mut Stage2InitialRoundBatch<E> {
        if self.initial_round_batch.is_none() {
            let stage1_point = self
                .prefix_r_stage1
                .clone()
                .expect("initial round batch requested without cached stage-1 challenges");
            let coeff_bits = self.num_vars - self.segment_bits;
            let w_compact = match &self.witness_table {
                WitnessTable::Compact(w_compact) => w_compact,
                WitnessTable::Full(_) => {
                    panic!("initial round batch can only build from compact witness")
                }
            };
            let proof = build_stage2_initial_round_batch_grid(
                w_compact,
                self.relation_weight.evals(),
                &stage1_point,
                self.b,
                self.live_segments,
                self.segment_bits,
                coeff_bits,
            )
            .expect("initial round batch should be available");
            let relation_weight_claim = self.input_claim - self.batching_coeff * self.s_claim;
            let skip_state = Stage2RoundBatchState::new(
                &proof,
                &stage1_point,
                self.s_claim,
                relation_weight_claim,
                self.batching_coeff,
            )
            .expect("valid round-batch state");
            self.initial_round_batch = Some(Stage2InitialRoundBatch {
                skip_state,
                first_challenge: None,
            });
        }
        self.initial_round_batch
            .as_mut()
            .expect("initial round batch should be initialized")
    }
}
