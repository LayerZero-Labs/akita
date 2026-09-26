use super::*;
use jolt_poly::UnivariatePoly;

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

impl<E: Field + Ring + Unreduced> RelationRangeImageProver<E> {
    #[cfg(test)]
    pub(crate) fn new_virtual_only(
        witness: Vec<i8>,
        stage1_point: &[E],
        range_image_evaluation: E,
        basis: usize,
        live_lanes: usize,
        lane_bits: usize,
        coefficient_bits: usize,
    ) -> Result<Self, AkitaError>
    where
        E: 'static,
    {
        let (lane_capacity, coeff_count) = stage2_geometry(lane_bits, coefficient_bits)?;
        Self::new(
            E::one(),
            PackedSignedDigits::from_i8_digits_auto(witness),
            stage1_point,
            range_image_evaluation,
            basis,
            RelationWeightOracle::QuotientFactored(RelationWeightFactorization::new(
                vec![E::zero(); coeff_count],
                vec![E::zero(); lane_capacity],
            )?),
            live_lanes,
            lane_bits,
            coefficient_bits,
            E::zero(),
            PreparedProverLinearTerms::zero(live_lanes, coeff_count),
            E::zero(),
            None,
        )
    }

    /// Create a fused stage-2 virtual-claim + relation sumcheck prover.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "RelationRangeImageProver::new")]
    pub(crate) fn new(
        batching_coeff: E,
        w_evals_compact: PackedSignedDigits,
        stage1_point: &[E],
        range_image_evaluation: E,
        b: usize,
        relation_weights: RelationWeightOracle<E>,
        live_lane_count: usize,
        lane_bits: usize,
        coefficient_bits: usize,
        relation_claim: E,
        linear_terms: PreparedProverLinearTerms<E>,
        linear_opening_claim: E,
        additional_relation_terms: Option<AdditionalRelationTerms<E>>,
    ) -> Result<Self, AkitaError>
    where
        E: 'static,
    {
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
        match &relation_weights {
            RelationWeightOracle::QuotientFactored(factorization) => {
                if factorization.common_alpha_factor().len() != coeff_count {
                    return Err(AkitaError::InvalidSize {
                        expected: coeff_count,
                        actual: factorization.common_alpha_factor().len(),
                    });
                }
                if factorization.relation_lane_weights().len() != lane_capacity {
                    return Err(AkitaError::InvalidSize {
                        expected: lane_capacity,
                        actual: factorization.relation_lane_weights().len(),
                    });
                }
            }
            RelationWeightOracle::ReducedDense(dense) => {
                let domain_len = lane_capacity.checked_mul(coeff_count).ok_or_else(|| {
                    AkitaError::InvalidInput("stage-2 relation domain overflow".into())
                })?;
                if dense.evaluations().len() != domain_len {
                    return Err(AkitaError::InvalidSize {
                        expected: domain_len,
                        actual: dense.evaluations().len(),
                    });
                }
                if dense.live_len() != witness_len {
                    return Err(AkitaError::InvalidSize {
                        expected: witness_len,
                        actual: dense.live_len(),
                    });
                }
            }
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
            let (ordinary_relation_sum, structured_relation_sum) =
                (0..witness_len).fold((E::zero(), E::zero()), |(ordinary, structured), index| {
                    let lane = index / coeff_count;
                    let coefficient = index % coeff_count;
                    let w = w_evals_compact
                        .get(index)
                        .expect("debug relation witness index is in bounds");
                    let witness = E::from_i64(i64::from(w));
                    let relation_weight = match &relation_weights {
                        RelationWeightOracle::QuotientFactored(factorization) => {
                            factorization.common_alpha_factor()[coefficient]
                                * factorization.relation_lane_weights()[lane]
                        }
                        RelationWeightOracle::ReducedDense(dense) => dense.evaluations()[index],
                    };
                    (
                        ordinary + witness * relation_weight,
                        structured + witness * linear_terms.get(lane, coefficient, coeff_count),
                    )
                });
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
        let split_eq = GruenSplitEq::with_initial_scalar(stage1_point, batching_coeff)?;
        let relation_state = match relation_weights {
            RelationWeightOracle::QuotientFactored(weights) => {
                let prefix = CompactQuotientPrefix::new(
                    &w_evals_compact,
                    weights.relation_lane_weights(),
                    &linear_terms,
                    &split_eq,
                    stage1_point,
                    batching_coeff,
                    b,
                    live_lane_count,
                    coefficient_bits,
                )
                .map_or(QuotientPrefixState::Disabled, |prefix| {
                    QuotientPrefixState::Compact(Box::new(prefix))
                });
                RelationRoundState::QuotientFactored { weights, prefix }
            }
            RelationWeightOracle::ReducedDense(weights) => {
                RelationRoundState::ReducedDense { weights }
            }
        };

        Ok(Self {
            witness_state: WitnessState::CompactPrefix(w_evals_compact),
            input_claim,
            split_eq,
            relation_state,
            additional_relation_terms,
            linear_terms,
            live_lane_count,
            lane_bits,
            num_vars,
            prev_norm_claim: batching_coeff * range_image_evaluation,
            prev_norm_poly: None,
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
    pub(crate) fn final_w_eval(&self) -> E {
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
        let witness = self.final_w_eval();
        let virtual_claim = self.split_eq.current_scalar() * witness * (witness + E::one());
        let relation_weight = match &self.relation_state {
            RelationRoundState::QuotientFactored { weights, .. } => {
                match (
                    weights.common_alpha_factor(),
                    weights.relation_lane_weights(),
                ) {
                    ([alpha], [lane]) => *alpha * *lane + self.linear_terms.final_value()?,
                    _ => return Err(AkitaError::InvalidProof),
                }
            }
            RelationRoundState::ReducedDense { weights } => {
                weights.terminal_weight()? + self.linear_terms.final_value()?
            }
            RelationRoundState::LaneProduct(lane) => lane.final_weight()?,
        };
        let additional = self
            .additional_relation_terms
            .as_ref()
            .map_or(Ok(E::zero()), |terms| terms.final_claim(witness))?;
        Ok(virtual_claim + witness * relation_weight + additional)
    }

    pub(super) fn additional_round_polynomial(&self) -> Option<UnivariatePoly<E>> {
        let additional = self.additional_relation_terms.as_ref()?;
        Some(match &self.witness_state {
            WitnessState::CompactPrefix(compact_witness) => {
                let bound = self
                    .compact_quotient_prefix()
                    .map_or(&[][..], CompactQuotientPrefix::challenges);
                additional.round_polynomial_compact(compact_witness.view(), bound)
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
    pub(super) fn in_coefficient_round(&self) -> bool {
        self.rounds_completed < self.coefficient_bits()
    }

    #[inline]
    pub(super) fn current_coefficient_width(&self) -> usize {
        self.coefficient_bits()
            .saturating_sub(self.coefficient_rounds_completed())
    }

    #[inline]
    pub(super) fn use_partial_lane_coefficient_round(&self) -> bool {
        self.in_coefficient_round() && self.live_lane_count < (1usize << self.lane_bits)
    }

    #[inline]
    pub(super) fn can_skip_norm_linear_coeff(&self) -> bool {
        self.split_eq.can_recover_linear_q_term_from_claim()
    }

    #[inline]
    pub(super) fn norm_poly_from_terms(&self, virt_terms: NormRoundTerms<E>) -> UnivariatePoly<E> {
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
    ) -> (UnivariatePoly<E>, UnivariatePoly<E>) {
        let virt_poly = self.norm_poly_from_terms(virt_terms);
        let rel_poly = coeffs_to_poly(rel_coeffs);
        (virt_poly, rel_poly)
    }

    #[inline]
    pub(super) fn combine_polys(
        &self,
        virt_poly: &UnivariatePoly<E>,
        relation_poly: &UnivariatePoly<E>,
    ) -> UnivariatePoly<E> {
        let max_len = virt_poly
            .coefficients()
            .len()
            .max(relation_poly.coefficients().len());
        let mut combined = vec![E::zero(); max_len];
        for (i, c) in virt_poly.coefficients().iter().enumerate() {
            combined[i] += *c;
        }
        for (i, c) in relation_poly.coefficients().iter().enumerate() {
            combined[i] += *c;
        }
        UnivariatePoly::new(combined)
    }

    #[inline]
    pub(super) fn combine_terms(
        &mut self,
        virt_terms: NormRoundTerms<E>,
        rel_coeffs: [E; 3],
    ) -> UnivariatePoly<E> {
        let (virt_poly, relation_poly) = self.polys_from_terms(virt_terms, rel_coeffs);
        let combined = self.combine_polys(&virt_poly, &relation_poly);
        self.prev_norm_poly = Some(virt_poly);
        combined
    }

    #[cfg(test)]
    pub(super) fn disable_compact_quotient_prefix(&mut self) {
        if let RelationRoundState::QuotientFactored { prefix, .. } = &mut self.relation_state {
            *prefix = QuotientPrefixState::Disabled;
        }
    }
}
