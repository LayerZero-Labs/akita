use super::*;

fn stage2_geometry(
    lane_bits: usize,
    coefficient_bits: usize,
) -> Result<(usize, usize), AkitaError> {
    let lane_bits_u32 = u32::try_from(lane_bits)
        .map_err(|_| AkitaError::InvalidInput("stage-2 lane width overflow".to_string()))?;
    let coefficient_bits_u32 = u32::try_from(coefficient_bits)
        .map_err(|_| AkitaError::InvalidInput("stage-2 coefficient width overflow".to_string()))?;
    let lane_capacity = 1usize
        .checked_shl(lane_bits_u32)
        .ok_or_else(|| AkitaError::InvalidInput("stage-2 lane width overflow".to_string()))?;
    let coeff_count = 1usize.checked_shl(coefficient_bits_u32).ok_or_else(|| {
        AkitaError::InvalidInput("stage-2 coefficient width overflow".to_string())
    })?;
    Ok((lane_capacity, coeff_count))
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> RelationRangeImageProver<E> {
    /// Create a stage-2 instance containing only the virtual range-image term.
    ///
    /// This is the standalone companion to
    /// [`DigitRangeProver`](crate::protocol::sumcheck::DigitRangeProver):
    /// stage 1 proves that the compact balanced-digit table is pointwise in
    /// range, while this sumcheck links its carried range-image claim
    /// `S(r) = range_image_evaluation` to an opening of the same digit table
    /// through `S = w(w + 1)`. No relation or evaluation-trace term is
    /// included.
    pub fn new_virtual_only(
        w_evals_compact: impl Into<std::sync::Arc<[i8]>>,
        stage1_point: &[E],
        range_image_evaluation: E,
        b: usize,
        live_lane_count: usize,
        lane_bits: usize,
        coefficient_bits: usize,
    ) -> Result<Self, AkitaError> {
        let (lane_capacity, coeff_count) = stage2_geometry(lane_bits, coefficient_bits)?;
        Self::new(
            E::one(),
            w_evals_compact,
            stage1_point,
            range_image_evaluation,
            b,
            vec![E::zero(); coeff_count],
            vec![E::zero(); lane_capacity],
            live_lane_count,
            lane_bits,
            coefficient_bits,
            E::zero(),
            PreparedProverLinearTerms::zero(live_lane_count, coeff_count),
            E::zero(),
            None,
        )
    }

    /// Create a fused stage-2 virtual-claim + relation sumcheck prover.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "RelationRangeImageProver::new")]
    pub(crate) fn new(
        batching_coeff: E,
        w_evals_compact: impl Into<std::sync::Arc<[i8]>>,
        stage1_point: &[E],
        range_image_evaluation: E,
        b: usize,
        common_alpha_factor: Vec<E>,
        relation_lane_weights: Vec<E>,
        live_lane_count: usize,
        lane_bits: usize,
        coefficient_bits: usize,
        relation_claim: E,
        linear_terms: PreparedProverLinearTerms<E>,
        linear_opening_claim: E,
        additional_relation_terms: Option<AdditionalRelationTerms<E>>,
    ) -> Result<Self, AkitaError> {
        let w_evals_compact = w_evals_compact.into();
        let num_vars = lane_bits.checked_add(coefficient_bits).ok_or_else(|| {
            AkitaError::InvalidInput("stage-2 challenge width overflow".to_string())
        })?;
        if live_lane_count == 0 {
            return Err(AkitaError::InvalidInput(
                "live_lane_count must be at least 1".to_string(),
            ));
        }
        let (lane_capacity, coeff_count) = stage2_geometry(lane_bits, coefficient_bits)?;
        if live_lane_count > lane_capacity {
            return Err(AkitaError::InvalidSize {
                expected: lane_capacity,
                actual: live_lane_count,
            });
        }
        let witness_len = live_lane_count
            .checked_mul(coeff_count)
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
        if common_alpha_factor.len() != coeff_count {
            return Err(AkitaError::InvalidSize {
                expected: coeff_count,
                actual: common_alpha_factor.len(),
            });
        }
        if relation_lane_weights.len() != lane_capacity {
            return Err(AkitaError::InvalidSize {
                expected: lane_capacity,
                actual: relation_lane_weights.len(),
            });
        }
        linear_terms.validate_len(witness_len)?;

        // Self-consistency check: the materialized ordinary relation weights
        // plus the structured linear weights must reproduce the combined
        // relation claim. Packing keeps its Z contribution in the structured
        // representation, so checking the ordinary table in isolation would
        // reject a valid factored relation. This is a full-domain
        // `O(lane_capacity * coeff_count)` pass, so it is gated to
        // debug/test builds and never runs in release proving.
        #[cfg(debug_assertions)]
        {
            let (ordinary_relation_sum, structured_relation_sum) = w_evals_compact
                .chunks_exact(coeff_count)
                .zip(&relation_lane_weights)
                .enumerate()
                .fold(
                    (E::zero(), E::zero()),
                    |(ordinary, structured), (lane, (lane_values, &lane_weight))| {
                        lane_values
                            .iter()
                            .zip(&common_alpha_factor)
                            .enumerate()
                            .fold(
                                (ordinary, structured),
                                |(ordinary, structured), (coefficient, (&w, &alpha))| {
                                    let witness = E::from_i64(i64::from(w));
                                    (
                                        ordinary + witness * lane_weight * alpha,
                                        structured
                                            + witness
                                                * linear_terms.get(lane, coefficient, coeff_count),
                                    )
                                },
                            )
                    },
                );
            if ordinary_relation_sum + structured_relation_sum
                != relation_claim + linear_opening_claim
            {
                return Err(AkitaError::InvalidInput(
                    "materialized relation weights do not match the combined relation claim".into(),
                ));
            }
        }

        let relation_linear_claim = relation_claim + linear_opening_claim;
        let additional_claim = additional_relation_terms
            .as_ref()
            .map_or_else(E::zero, AdditionalRelationTerms::input_claim);
        let input_claim =
            batching_coeff * range_image_evaluation + relation_linear_claim + additional_claim;
        let use_two_round_prefix = can_use_stage2_two_round_prefix(coefficient_bits, b);

        Ok(Self {
            witness_state: WitnessState::CompactPrefix(w_evals_compact),
            b,
            batching_coeff,
            range_image_evaluation,
            input_claim,
            split_eq: GruenSplitEq::with_initial_scalar(stage1_point, batching_coeff)?,
            common_alpha_factor,
            relation_lane_weights,
            additional_relation_terms,
            linear_terms,
            live_lane_count,
            lane_bits,
            num_vars,
            relation_linear_claim,
            prev_norm_claim: batching_coeff * range_image_evaluation,
            prev_norm_poly: None,
            compact_prefix_stage1_point: use_two_round_prefix.then(|| stage1_point.to_vec()),
            deferred_compact_prefix: None,
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
    /// Panics if called before the folded suffix contains one field element.
    pub fn final_w_eval(&self) -> E {
        match &self.witness_state {
            WitnessState::FoldedSuffix(folded_witness) => {
                assert_eq!(folded_witness.len(), 1, "witness suffix not fully folded");
                folded_witness[0]
            }
            WitnessState::CompactPrefix(_) => {
                panic!("witness remained in compact-prefix state after final fold")
            }
        }
    }

    pub(crate) fn expected_final_claim(&self) -> Result<E, AkitaError> {
        if self.common_alpha_factor.len() != 1 || self.relation_lane_weights.len() != 1 {
            return Err(AkitaError::InvalidProof);
        }
        let witness = self.final_w_eval();
        let virtual_claim = self.split_eq.current_scalar() * witness * (witness + E::one());
        let ordinary_relation =
            witness * self.common_alpha_factor[0] * self.relation_lane_weights[0];
        let linear_claim = witness * self.linear_terms.final_value()?;
        let additional = self
            .additional_relation_terms
            .as_ref()
            .map_or(Ok(E::zero()), |terms| terms.final_claim(witness))?;
        Ok(virtual_claim + ordinary_relation + linear_claim + additional)
    }

    pub(super) fn additional_round_polynomial(&self) -> Option<UniPoly<E>> {
        let additional = self.additional_relation_terms.as_ref()?;
        Some(match &self.witness_state {
            WitnessState::CompactPrefix(compact_witness) => {
                let first_challenge = if self.rounds_completed == 0 {
                    None
                } else {
                    Some(
                        self.deferred_compact_prefix
                            .as_ref()
                            .and_then(|prefix| prefix.first_challenge)
                            .expect("compact round 1 requires the first prefix challenge"),
                    )
                };
                additional.round_polynomial_compact(compact_witness, first_challenge)
            }
            WitnessState::FoldedSuffix(folded_witness) => {
                additional.round_polynomial_folded(folded_witness)
            }
        })
    }

    #[inline]
    pub(super) fn coefficient_bits(&self) -> usize {
        self.num_vars - self.lane_bits
    }

    #[inline]
    pub(super) fn coefficient_rounds_completed(&self) -> usize {
        self.rounds_completed.min(self.coefficient_bits())
    }

    #[inline]
    pub(super) fn lane_rounds_completed(&self) -> usize {
        self.rounds_completed
            .saturating_sub(self.coefficient_bits())
    }

    #[inline]
    pub(super) fn in_coefficient_round(&self) -> bool {
        self.rounds_completed < self.coefficient_bits()
    }

    #[inline]
    pub(super) fn current_coefficient_width(&self) -> usize {
        self.coefficient_bits()
            .saturating_sub(self.coefficient_rounds_completed())
    }

    #[inline]
    pub(super) fn current_lane_width(&self) -> usize {
        self.lane_bits.saturating_sub(self.lane_rounds_completed())
    }

    #[inline]
    pub(super) fn current_lane_capacity(&self) -> usize {
        1usize << self.current_lane_width()
    }

    #[inline]
    pub(super) fn use_partial_lane_coefficient_round(&self) -> bool {
        self.in_coefficient_round() && self.live_lane_count < self.current_lane_capacity()
    }

    #[inline]
    pub(super) fn use_partial_lane_round(&self) -> bool {
        self.rounds_completed >= self.coefficient_bits()
            && self.lane_rounds_completed() < self.lane_bits
            && self.live_lane_count < self.current_lane_capacity()
    }

    #[inline]
    pub(super) fn next_uses_partial_lane_round(&self) -> bool {
        self.rounds_completed >= self.coefficient_bits()
            && self.lane_rounds_completed() + 1 < self.lane_bits
            && self.live_lane_count.div_ceil(2) < (self.current_lane_capacity() / 2)
    }

    #[inline]
    pub(crate) fn can_use_deferred_compact_prefix(&self) -> bool {
        self.compact_prefix_stage1_point.is_some()
    }

    #[inline]
    pub(super) fn using_deferred_compact_prefix(&self) -> bool {
        self.rounds_completed < 2 && self.can_use_deferred_compact_prefix()
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

    pub(super) fn ensure_deferred_compact_prefix(&mut self) -> &mut TwoRoundCompactPrefix<E> {
        if self.deferred_compact_prefix.is_none() {
            let stage1_point = self
                .compact_prefix_stage1_point
                .clone()
                .expect("two-round prefix requested without cached stage-1 challenges");
            let coefficient_bits = self.num_vars - self.lane_bits;
            let compact_witness = match &self.witness_state {
                WitnessState::CompactPrefix(compact_witness) => compact_witness,
                WitnessState::FoldedSuffix(_) => {
                    panic!("two-round prefix can only build from compact witness")
                }
            };
            let proof = build_stage2_bivariate_skip_proof_from_m_compact(
                compact_witness,
                &self.common_alpha_factor,
                &self.relation_lane_weights,
                &self.linear_terms,
                &stage1_point,
                self.b,
                self.live_lane_count,
                self.lane_bits,
                coefficient_bits,
            )
            .expect("two-round prefix should be available");
            let skip_state = Stage2BivariateSkipState::new(
                &proof,
                &stage1_point,
                self.range_image_evaluation,
                self.relation_linear_claim,
                self.batching_coeff,
            )
            .expect("valid bivariate-skip state");
            self.deferred_compact_prefix = Some(TwoRoundCompactPrefix {
                skip_state,
                first_challenge: None,
            });
        }
        self.deferred_compact_prefix
            .as_mut()
            .expect("two-round prefix should be initialized")
    }
}
